#pragma once

#include "onmt/nn/Module.h"
#include "onmt/simd/Operators.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd>
    class qSigmoid: public Module<MatFwd>
    {
    public:
      qSigmoid()
        : Module<MatFwd>("nn.Sigmoid")
      {
        _qrange = (int) 10 * simd::quant_mult;
        _cache_sigmoid = new float[2*_qrange + 1];
        for(int i=0; i < 2 * _qrange + 1; i++)
          _cache_sigmoid[i] = 1/(1+exp(-(i-_qrange)/simd::quant_mult));
      }

      void forward_impl(const MatFwd& input) override
      {
        this->_output.resize(input.rows(), input.cols());
        simd::QuantizeLookupTable(input.data(), this->_output.data(), _cache_sigmoid,
                                  _qrange, input.rows(), input.cols());
      }
    private:
      int _qrange;
      float *_cache_sigmoid;
    };

  }
}
