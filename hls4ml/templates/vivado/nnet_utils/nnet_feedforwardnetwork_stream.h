#ifndef NNET_FFN_H_
#define NNET_FFN_H_

#include "nnet_common.h"
#include "nnet_mult.h"
#include "nnet_helpers.h"
#include "hls_stream.h"
#include <math.h>
#include <iostream>

namespace nnet {

struct ffn_config {
    static const unsigned seq_len = 180;
    static const unsigned embed_dim = 182;
    static const unsigned hidden_dim = 128;
    static const unsigned in_ram_style = nnet::block;
    static const unsigned out_ram_style = nnet::block;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
};    

template<class data_T, class res_T, typename CONFIG_T>
void FeedForwardNetwork(
    hls::stream<data_T>    &data,
    hls::stream<res_T>     &res,
    typename CONFIG_T::in_proj_weight_t     in_proj_weight[CONFIG_T::embed_dim/CONFIG_T::tiling_factor[1]][CONFIG_T::hidden_dim/CONFIG_T::tiling_factor[2]][CONFIG_T::tiling_factor[1]][CONFIG_T::tiling_factor[2]],
    typename CONFIG_T::in_proj_bias_t       in_proj_bias[CONFIG_T::hidden_dim/CONFIG_T::tiling_factor[2]][CONFIG_T::tiling_factor[2]],
    typename CONFIG_T::out_proj_weight_t    out_proj_weight[CONFIG_T::hidden_dim/CONFIG_T::tiling_factor[2]][CONFIG_T::embed_dim/CONFIG_T::tiling_factor[1]][CONFIG_T::tiling_factor[2]][CONFIG_T::tiling_factor[1]],
    typename CONFIG_T::out_proj_bias_t      out_proj_bias[CONFIG_T::embed_dim/CONFIG_T::tiling_factor[1]][CONFIG_T::tiling_factor[1]])
{
    typename data_T::value_type  input_buffer[CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]][CONFIG_T::embed_dim/CONFIG_T::tiling_factor[1]][CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[1]];  
    typename res_T::value_type   output_buffer[CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]][CONFIG_T::embed_dim/CONFIG_T::tiling_factor[1]][CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[1]];
    #pragma HLS ARRAY_PARTITION variable=input_buffer   complete dim=3
    #pragma HLS ARRAY_PARTITION variable=input_buffer   complete dim=4
    #pragma HLS ARRAY_PARTITION variable=output_buffer  complete dim=3
    #pragma HLS ARRAY_PARTITION variable=output_buffer  complete dim=4
    #pragma HLS ARRAY_PARTITION variable=in_proj_weight       complete dim=3
    #pragma HLS ARRAY_PARTITION variable=in_proj_weight       complete dim=4
    #pragma HLS ARRAY_PARTITION variable=out_proj_weight       complete dim=3
    #pragma HLS ARRAY_PARTITION variable=out_proj_weight       complete dim=4
    #pragma HLS ARRAY_PARTITION variable=in_proj_bias        complete dim=2
    #pragma HLS ARRAY_PARTITION variable=out_proj_bias        complete dim=2

    typename CONFIG_T::accum_t dense1_out[CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[2]];
    typename CONFIG_T::accum_t dense2_out[CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[2]];
    typename CONFIG_T::accum_t dense_out[CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[2]];
    typename CONFIG_T::accum_t buffer[CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[1]];
    #pragma HLS ARRAY_PARTITION variable=dense1_out     complete dim=0
    #pragma HLS ARRAY_PARTITION variable=dense2_out     complete dim=0
    #pragma HLS ARRAY_PARTITION variable=dense_out      complete dim=0
    //#pragma HLS DATAFLOW
    data_T data_pack;
    store_input:
    for (int i=0; i <CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]; i=i+1){
        for (int j=0; j < CONFIG_T::embed_dim/CONFIG_T::tiling_factor[1]; j=j+1){
            #pragma HLS PIPELINE II=1
            for (int ii=0; ii < CONFIG_T::tiling_factor[0]; ++ii){
                #pragma HLS UNROLL
                for (int jj=0; jj < CONFIG_T::tiling_factor[1]; ++jj){
                    #pragma HLS UNROLL
                    if (i == 0 && j == 0) {
                        data_pack = data.read();
                    }
                    input_buffer[i][j][ii][jj] = data_pack[ii*CONFIG_T::tiling_factor[1]+jj];
                    output_buffer[i][j][ii][jj] = out_proj_bias[j][jj];
                }
            }
        }
    }
    
    typename data_T::value_type tmp_input_buffer;
    typename res_T::value_type tmp_output_buffer;
    int i = 0;
    int k = 0;
    int m = 0;
    int n = 0;
    int total_cycle = ((CONFIG_T::seq_len*CONFIG_T::hidden_dim)/(CONFIG_T::tiling_factor[0]*CONFIG_T::tiling_factor[2])) + 1;
    static bool dense1_init = false;
    //#pragma HLS DEPENDENCE variable=output_buffer intra false
    //std::cout << "dense1_out[0][0] = "<< std::endl;
    pipeline_product1n2: // 1st inner product with ijk indexing and 2nd outter product with mnp indexing
    for (int c=0; c < total_cycle; ++c){
        for (int p=0; p < CONFIG_T::embed_dim/CONFIG_T::tiling_factor[1]; p=p+1){
            #pragma HLS PIPELINE II=1
            if (p==0) {
                for (int ii=0; ii < CONFIG_T::tiling_factor[0]; ++ii){
                    #pragma HLS UNROLL
                    for (int kk=0; kk < CONFIG_T::tiling_factor[2]; ++kk){
                        #pragma HLS UNROLL
                        if (dense1_init) {
                            dense2_out[ii][kk] = in_proj_bias[kk][k];
                            if (dense1_out[ii][kk] < 0) {
                                dense1_out[ii][kk] = 0;
                            }
                        } else {
                            dense1_out[ii][kk] = in_proj_bias[kk][k];
                            if (dense2_out[ii][kk] < 0) {
                                dense2_out[ii][kk] = 0;
                            }
                        }           
                    }
                }
            }
            inner_product:
            for (int ii=0; ii < CONFIG_T::tiling_factor[0]; ++ii){
                #pragma HLS UNROLL
                for (int pp=0; pp < CONFIG_T::tiling_factor[1]; ++pp){
                    #pragma HLS UNROLL
                	tmp_output_buffer = output_buffer[m][p][ii][pp];
                    tmp_input_buffer = input_buffer[i][p][ii][pp];
                    for (int kk=0; kk < CONFIG_T::tiling_factor[2]; ++kk){
                        #pragma HLS UNROLL
                        typename CONFIG_T::accum_t temp = tmp_input_buffer * in_proj_weight[p][k][pp][kk];
                        if (dense1_init) {
                            if ((i < CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]) && (k < CONFIG_T::hidden_dim/CONFIG_T::tiling_factor[2])) {
                                dense2_out[ii][kk] += temp;
                            }
                            dense_out[ii][kk] = dense1_out[ii][kk];
                        } else {
                            if ((i < CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]) && (k < CONFIG_T::hidden_dim/CONFIG_T::tiling_factor[2])) {
                                dense1_out[ii][kk] += temp;
                            }
                            dense_out[ii][kk] = dense2_out[ii][kk];
                        }
                        if (c>0){
                        	tmp_output_buffer = tmp_output_buffer + dense_out[ii][kk] * out_proj_weight[n][p][kk][pp];
                        }
                    }
                    output_buffer[m][p][ii][pp] = tmp_output_buffer;
                }
            }
            if (p==(CONFIG_T::embed_dim/CONFIG_T::tiling_factor[1]-1)) { // last cycle of pipeline
                if (dense1_init){
                    dense1_init = false;
                } else {
                    dense1_init = true;
                }
                if (c < total_cycle-1) { //cycle 0~total_cycle-1
                    k = k + 1;
                    if (k == CONFIG_T::hidden_dim/CONFIG_T::tiling_factor[2]){
                        k = 0;
                        i = i + 1;
                    }
                }
                if (c > 0) { //cycle 1~total_cycle
                    n = n + 1;
                    if (n == CONFIG_T::hidden_dim/CONFIG_T::tiling_factor[2]){
                        n = 0;
                        m = m + 1;
                    }
                }
            }
        }
    }

    res_T res_pack;
    write_output:
    for (int i=0; i <CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]; i=i+1){
        for (int j=0; j < CONFIG_T::embed_dim/CONFIG_T::tiling_factor[1]; j=j+1){
            #pragma HLS PIPELINE II=1
            for (int ii=0; ii < CONFIG_T::tiling_factor[0]; ++ii){
                #pragma HLS UNROLL
                for (int jj=0; jj < CONFIG_T::tiling_factor[1]; ++jj){
                    #pragma HLS UNROLL
                    res_pack[ii*CONFIG_T::tiling_factor[1]+jj] = output_buffer[i][j][ii][jj];
                    if (jj == CONFIG_T::tiling_factor[1]-1 && ii == CONFIG_T::tiling_factor[0]-1) {
                        res.write(res_pack);
                    }
                }
            }
        }
    }
    
}



}


#endif
