
#ifndef NNET_ARRAY_STREAM_H
#define NNET_ARRAY_STREAM_H

#include "hls_stream.h"
#include "nnet_common.h"

namespace nnet {

template<class data_T, class res_T, int N, int chan>
void clone_stream(hls::stream<data_T> data[chan], hls::stream<res_T> res1[chan], hls::stream<res_T> res2[chan]) {
    CloneLoop: for (int i = 0; i < N / chan; i++) {
        #pragma HLS PIPELINE
        for (int j = 0; j < chan; j++) {
            #pragma HLS UNROLL
            data_T in_data = data[j].read();
            res1[j].write(in_data);
            res2[j].write(in_data);
        }
    }
}

} // namespace nnet

#endif
