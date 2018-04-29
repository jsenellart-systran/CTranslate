#pragma once

#include "onmt/nn/Module.h"
#include "onmt/th/Obj.h"
#include "onmt/StorageLoader.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatEmb, typename ModelT>
    class LookupTable: public Module<MatFwd>
    {
    public:
      LookupTable(th::Table* data)
        : Module<MatFwd>("nn.LookupTable")
        , _weight(StorageLoader<MatEmb, ModelT>::get_matrix(data, "weight"))
        , _cache_emblin(0)
      {
      }

      virtual void set_cache_emblin(size_t cache_emblin) {
        /* we will be able to cache cache_emblin output */
        _cache_emblin = cache_emblin;
        _cached.resize(cache_emblin);
      }

      void forward_impl(const MatFwd& input) override
      {
        this->_output.resize(input.rows(), _weight.cols());
        bool has_cache_rows = false;

        for (int i = 0; i < input.rows(); ++i) {
          int vocab_id = input(i, 0);
          if (vocab_id < (int)_cache_emblin) {
            has_cache_rows = true;
            if (_cached[vocab_id])
              this->_output(i,0) = (vocab_id<<3) + 100;
            else
            {
              this->_output.row(i).noalias() = _weight.row(vocab_id);
              this->_output(i,0) += 100 + (vocab_id<<3);
              _cached[vocab_id] = 1;
            }
          }
          else
            this->_output.row(i).noalias() = _weight.row(vocab_id);
        }

        /* mark first cell to trigger cache */
        if (has_cache_rows) {
          this->_output(0,0) += 100;
        }
      }

    private:
      MatEmb _weight;
      size_t _cache_emblin;
      std::vector<unsigned char> _cached;
    };

  }
}
