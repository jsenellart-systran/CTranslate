#pragma once

#include <memory>
#include "onmt/nn/Linear.h"
#include "onmt/th/Obj.h"
#include "onmt/StorageLoader.h"
#include "onmt/Utils.h"
#include "onmt/simd/MatrixMult.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename ModelT>
    class qLinear: public Linear<MatFwd, MatIn, ModelT>
    {
    public:
      qLinear(th::Table* data)
        : Linear<MatFwd, MatIn, ModelT>(data), _quant_input_buffer(nullptr)
        , _cache_emblin(0)
      {
        // Quantize the weight - ncols=width is supposed to be multiple of SIMD_VSIZE
        if (this->_wcols % SIMD_VSIZE)
          throw std::runtime_error("Weight matrix width should be multiple of 8/16 for qLinear");
        _malloc_align(_quant_weight_buffer, _quant_weight, this->_wrows * this->_wcols / SIMD_VSIZE);
        simd::Quantize(this->_weight.data(), _quant_weight, this->_wrows, this->_wcols);
      }

      virtual ~qLinear()
      {
        free(_quant_weight_buffer);
        free(_quant_input_buffer);
      }

      virtual void set_cache_emblin(size_t cache_emblin) override {
        /* we will be able to cache cache_emblin output */
        _cache_emblin = cache_emblin;
        _cached.resize(cache_emblin);
      }

      /* aligned allocation method - in c++17 we have aligned_alloc that we can use */
      void _malloc_align(void *&buffer, SIMD_TYPE *&data, size_t size)
      {
        buffer = nullptr;
        _realloc_align(buffer, data, size);
      }

      void _realloc_align(void *&buffer, SIMD_TYPE *&data, size_t size)
      {
        size_t buf_size = (size + 1) * sizeof(SIMD_TYPE);
        buffer = realloc(buffer, buf_size);
        if (!buffer)
          throw std::runtime_error("Cannot allocate memory");
        void* ptr = (void*)buffer;
        align(sizeof(SIMD_TYPE), size * sizeof(SIMD_TYPE), ptr, buf_size);
        data = reinterpret_cast<SIMD_TYPE*>(ptr);
      }

      virtual void forward_impl(const MatFwd& input) override
      {
        if (this->_rwrows)
          this->_output.resize(input.rows(), this->_rwrows);
        else
          this->_output.resize(input.rows(), this->_wrows);

        bool has_cache_emblin = input(0,0) >= 95;
        std::vector<int> cached_rows;
        int count_have_cache = 0;

        if (has_cache_emblin) {
          cached_rows.resize(input.rows());
          _cached.resize(_cache_emblin);
          int n_cached = 0;
          for(int i=0; i<input.rows(); i++) {
            bool idx_is_cached = false;
            cached_rows[i] = -1;
            ModelT v = input(i, 0);
            if (i==0) 
              v -= 100;
            if (v >= 95) {
              n_cached++;
              int vocab_id = (((int)(v+6)-100)) >> 3;
              cached_rows[i] = vocab_id;
              if (_cached[vocab_id]) {
                count_have_cache++;
                idx_is_cached = true;
              }
              v -= 100 + (vocab_id<<3);
            }
            /* dangerous but we know it is ok in that context */
            (*(MatFwd *)&input)(i,0) = v;
            if (!idx_is_cached && (i-count_have_cache) != i)
              (*(MatFwd *)&input).row(i-count_have_cache) = (*(MatFwd *)&input).row(i);
          }
        }

        if (input.rows()>count_have_cache) {
          /* quantize the input */
          _realloc_align(_quant_input_buffer, _quant_input, (input.rows()-count_have_cache) * input.cols() / SIMD_VSIZE);
          simd::Quantize(input.data(), _quant_input, (input.rows()-count_have_cache), input.cols());

          simd::MatrixMult(_quant_input, _quant_weight, this->_output.data(),
                           (input.rows()-count_have_cache), (this->_rwrows?this->_rwrows:this->_wrows), this->_wcols,
                           _subdict);

          /* add bias */
          if (this->_bias.rows() > 0)
          {
            if (this->_rwrows)
              for (int i = 0; i < input.rows() - count_have_cache; ++i)
                this->_output.row(i).noalias() += this->_rbias.transpose();
            else
              for (int i = 0; i < input.rows() - count_have_cache; ++i)
                this->_output.row(i).noalias() += this->_bias.transpose();
          }
        }

        /* if there was some caching we need to correct output */
        if (has_cache_emblin) {
          for(int output_idx=input.rows()-1; output_idx>=0; output_idx--) {
            int compute_idx = output_idx - count_have_cache;
            int vocab_id = cached_rows[output_idx];
            if (vocab_id >= 0 && _cached[vocab_id]) {
              this->_output.row(output_idx) = _cache_output.row(vocab_id);
              count_have_cache--;
            } else {
              if (vocab_id >= 0) {
                /* the cache did not exist yet, let us create it */
                if (_cache_output.rows()==0) {
                  _cache_output.resize(_cache_emblin, this->_output.cols());
                }
                _cache_output.row(vocab_id) = this->_output.row(compute_idx);
              }
              if (output_idx != compute_idx)
                this->_output.row(output_idx) = this->_output.row(compute_idx);
            }
          }
          /* mark as cached */
          for(int i=0; i<input.rows(); i++) {
            int vocab_id = cached_rows[i];
            if (vocab_id >= 0 && !_cached[vocab_id])
              _cached[vocab_id] = 1;
          }
        }

      }

      /* reduce a linear weigth matrix to a given vocabulary */
      virtual void apply_subdictionary(const std::vector<size_t>& v) override
      {
        this->_rwrows = v.size();
        _subdict = v;
        this->_rbias.resize(v.size(), 1);
        /* adjust bias */
        for (size_t i = 0; i < v.size(); i++) {
          this->_rbias.row(i) = this->_bias.row(v[i]);
        }
      }

    protected:
      void* _quant_weight_buffer;
      void* _quant_input_buffer;
      SIMD_TYPE* _quant_weight;
      SIMD_TYPE* _quant_input;
      std::vector<size_t> _subdict;
      std::vector<bool> _cached;
      size_t _cache_emblin;
      MatFwd _cache_output;
    };

  }
}
