#ifndef NNET_MERGE_ARRAY_STREAM_H_
#define NNET_MERGE_ARRAY_STREAM_H_

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_mult.h"
#include <math.h>

namespace nnet {

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void add(hls::stream<input1_T> data1[CONFIG_T::n_elem], hls::stream<input2_T> data2[CONFIG_T::n_elem], hls::stream<res_T> res[CONFIG_T::n_elem]) {
    #pragma HLS PIPELINE

    for (int ii = 0; ii < CONFIG_T::n_elem; ii++) {
        res[ii].write(res_T(data1[ii].read() + data2[ii].read()));
    }
}

} // namespace nnet

#endif
