#ifndef NNET_DENSE_STREAM_H_
#define NNET_DENSE_STREAM_H_

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_types.h"
#include <assert.h>
#include <math.h>

namespace nnet {

template <class data_T, class res_T, typename CONFIG_T>
void dense_wrapper(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
                   typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
                   typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {
    #pragma HLS INLINE recursive
    if (CONFIG_T::strategy == nnet::latency) {
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
        dense_latency<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    } else {
        dense_resource<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void dense(hls::stream<data_T> &data_stream, hls::stream<res_T> &res_stream,
           typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
           typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {
    typename data_T::value_type data[CONFIG_T::n_in];
    #pragma HLS ARRAY_PARTITION variable=data complete

    typename res_T::value_type res[CONFIG_T::n_out];
    #pragma HLS ARRAY_PARTITION variable=res complete

DataPrepare:
    for (int i_in = 0; i_in < CONFIG_T::n_in / data_T::size; i_in++) {
        if (CONFIG_T::n_in / data_T::size > 1) {
            #pragma HLS PIPELINE
        }
        data_T data_pack = data_stream.read();
    DataPack:
        for (int i_pack = 0; i_pack < data_T::size; i_pack++) {
            #pragma HLS UNROLL
            data[i_in * data_T::size + i_pack] = data_pack[i_pack];
        }
    }

    dense_wrapper<typename data_T::value_type, typename res_T::value_type, CONFIG_T>(data, res, weights, biases);

ResWrite:
    for (unsigned i_out = 0; i_out < CONFIG_T::n_out / res_T::size; i_out++) {
        if (CONFIG_T::n_out / res_T::size > 1) {
            #pragma HLS PIPELINE
        }
        res_T res_pack;
        PRAGMA_DATA_PACK(res_pack)
    ResPack:
        for (int i_pack = 0; i_pack < res_T::size; i_pack++) {
            #pragma HLS UNROLL
            res_pack[i_pack] = res[i_out * res_T::size + i_pack];
        }
        res_stream.write(res_pack);
    }
}
  
template<class data_T, class res_T, typename CONFIG_T>
void dense_array(
    hls::stream<data_T> data_stream[CONFIG_T::n_in],
    hls::stream<res_T>  res_stream[CONFIG_T::n_out],
    typename CONFIG_T::weight_t weights[CONFIG_T::n_in*CONFIG_T::n_out],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_out])
{
    data_T data[CONFIG_T::n_in];
    #pragma HLS ARRAY_PARTITION variable=data complete

    res_T res[CONFIG_T::n_out];
    #pragma HLS ARRAY_PARTITION variable=res complete

    Data: for (int i = 0; i< CONFIG_T::n_in; i++) {
        #pragma HLS UNROLL
        data_T data_pack = data_stream[i].read();
        data[i] = data_pack;
    }

    dense_wrapper<data_T, res_T, CONFIG_T>(data, res, weights, biases);

    Res: for (int i = 0; i < CONFIG_T::n_out; i++) {
        #pragma HLS UNROLL
        res_T res_pack = res[i];
        res_stream[i].write(res_pack);
    }
}  

} // namespace nnet

#endif
