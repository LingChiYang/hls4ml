#ifndef NNET_FFN_H_
#define NNET_FFN_H_

//#include "nnet_common.h"
//#include "nnet_mult.h"
//#include "nnet_helpers.h"
#include "hls_stream.h"
#include <math.h>
#include <iostream>

namespace nnet {

template<class data_T, class res_T, typename CONFIG_T>
void FFN(
    hls::stream<data_T>    data[CONFIG_T::n_in],
    hls::stream<res_T>     res[CONFIG_T::n_out],
    typename CONFIG_T::weight_t  weights1[CONFIG_T::n_in/CONFIG_T::block_y][CONFIG_T::hidden_dim/CONFIG_T::block_k][CONFIG_T::block_y][CONFIG_T::block_k],
    typename CONFIG_T::bias_t    biases1[CONFIG_T::hidden_dim/CONFIG_T::block_k][CONFIG_T::block_k],
    typename CONFIG_T::weight_t  weights2[CONFIG_T::hidden_dim/CONFIG_T::block_k][CONFIG_T::n_out/CONFIG_T::block_y][CONFIG_T::block_k][CONFIG_T::block_y],
    typename CONFIG_T::bias_t    biases2[CONFIG_T::n_out/CONFIG_T::block_y][CONFIG_T::block_y])
{
    data_T  input_buffer[CONFIG_T::seq_len/CONFIG_T::block_x][CONFIG_T::n_in/CONFIG_T::block_y][CONFIG_T::block_x][CONFIG_T::block_y];  
    res_T   output_buffer[CONFIG_T::seq_len/CONFIG_T::block_x][CONFIG_T::n_in/CONFIG_T::block_y][CONFIG_T::block_x][CONFIG_T::block_y];
    #pragma HLS ARRAY_PARTITION variable=input_buffer   complete dim=3
    #pragma HLS ARRAY_PARTITION variable=input_buffer   complete dim=4
    #pragma HLS ARRAY_PARTITION variable=output_buffer  complete dim=3
    #pragma HLS ARRAY_PARTITION variable=output_buffer  complete dim=4
    #pragma HLS ARRAY_PARTITION variable=weights1       complete dim=3
    #pragma HLS ARRAY_PARTITION variable=weights1       complete dim=4
    #pragma HLS ARRAY_PARTITION variable=weights2       complete dim=3
    #pragma HLS ARRAY_PARTITION variable=weights2       complete dim=4
    #pragma HLS ARRAY_PARTITION variable=biases1        complete dim=2
    #pragma HLS ARRAY_PARTITION variable=biases2        complete dim=2

    typename CONFIG_T::accum_t dense1_out[CONFIG_T::block_x][CONFIG_T::block_k];
    typename CONFIG_T::accum_t dense2_out[CONFIG_T::block_x][CONFIG_T::block_k];
    typename CONFIG_T::accum_t dense_out[CONFIG_T::block_x][CONFIG_T::block_k];
    typename CONFIG_T::accum_t buffer[CONFIG_T::block_x][CONFIG_T::block_y];
    #pragma HLS ARRAY_PARTITION variable=dense1_out     complete dim=0
    #pragma HLS ARRAY_PARTITION variable=dense2_out     complete dim=0
    #pragma HLS ARRAY_PARTITION variable=dense_out      complete dim=0
    //#pragma HLS DATAFLOW
    store_input:
    for (int i=0; i <CONFIG_T::seq_len/CONFIG_T::block_x; i=i+1){
        for (int j=0; j < CONFIG_T::n_in/CONFIG_T::block_y; j=j+1){
            for (int ii=0; ii < CONFIG_T::block_x; ++ii){
                //#pragma HLS UNROLL
                for (int jj=0; jj < CONFIG_T::block_y; ++jj){
                    #pragma HLS UNROLL
                    input_buffer[i][j][ii][jj] = data[j*CONFIG_T::block_y+jj].read();
                    output_buffer[i][j][ii][jj] = biases2[j][jj];
                }
            }
        }
    }
    
    data_T tmp_input_buffer;
    res_T tmp_output_buffer;
    int i = 0;
    int k = 0;
    int m = 0;
    int n = 0;
    int total_cycle = ((CONFIG_T::seq_len*CONFIG_T::hidden_dim)/(CONFIG_T::block_x*CONFIG_T::block_k)) + 1;
    static bool dense1_init = false;
    //#pragma HLS DEPENDENCE variable=output_buffer intra false
    //std::cout << "dense1_out[0][0] = "<< std::endl;
    pipeline_product1n2: // 1st inner product with ijk indexing and 2nd outter product with mnp indexing
    for (int c=0; c < total_cycle; ++c){
        for (int p=0; p < CONFIG_T::n_in/CONFIG_T::block_y; p=p+1){
            #pragma HLS PIPELINE II=1
            if (p==0) {
                for (int ii=0; ii < CONFIG_T::block_x; ++ii){
                    #pragma HLS UNROLL
                    for (int kk=0; kk < CONFIG_T::block_k; ++kk){
                        #pragma HLS UNROLL
                        if (dense1_init) {
                            dense2_out[ii][kk] = biases1[kk][k];
                            if (dense1_out[ii][kk] < 0) {
                                dense1_out[ii][kk] = 0;
                            }
                        } else {
                            dense1_out[ii][kk] = biases1[kk][k];
                            if (dense2_out[ii][kk] < 0) {
                                dense2_out[ii][kk] = 0;
                            }
                        }           
                    }
                }
            }
            inner_product:
            for (int ii=0; ii < CONFIG_T::block_x; ++ii){
                #pragma HLS UNROLL
                for (int pp=0; pp < CONFIG_T::block_y; ++pp){
                    #pragma HLS UNROLL
                	tmp_output_buffer = output_buffer[m][p][ii][pp];
                    tmp_input_buffer = input_buffer[i][p][ii][pp];
                    for (int kk=0; kk < CONFIG_T::block_k; ++kk){
                        #pragma HLS UNROLL
                        typename CONFIG_T::accum_t temp = tmp_input_buffer * weights1[p][k][pp][kk];
                        if (dense1_init) {
                            if ((i < CONFIG_T::seq_len/CONFIG_T::block_x) && (k < CONFIG_T::hidden_dim/CONFIG_T::block_k)) {
                                dense2_out[ii][kk] += temp;
                            }
                            dense_out[ii][kk] = dense1_out[ii][kk];
                        } else {
                            if ((i < CONFIG_T::seq_len/CONFIG_T::block_x) && (k < CONFIG_T::hidden_dim/CONFIG_T::block_k)) {
                                dense1_out[ii][kk] += temp;
                            }
                            dense_out[ii][kk] = dense2_out[ii][kk];
                        }
                        if (c>0){
                        	tmp_output_buffer = tmp_output_buffer + dense_out[ii][kk] * weights2[n][p][kk][pp];
                        }
                    }
                    output_buffer[m][p][ii][pp] = tmp_output_buffer;
                }
            }
            if (p==(CONFIG_T::n_in/CONFIG_T::block_y-1)) { // last cycle of pipeline
                if (dense1_init){
                    dense1_init = false;
                } else {
                    dense1_init = true;
                }
                if (c < total_cycle-1) { //cycle 0~total_cycle-1
                    k = k + 1;
                    if (k == CONFIG_T::hidden_dim/CONFIG_T::block_k){
                        k = 0;
                        i = i + 1;
                    }
                }
                if (c > 0) { //cycle 1~total_cycle
                    n = n + 1;
                    if (n == CONFIG_T::hidden_dim/CONFIG_T::block_k){
                        n = 0;
                        m = m + 1;
                    }
                }
            }
        }
    }
    write_output:
    for (int i=0; i <CONFIG_T::seq_len/CONFIG_T::block_x; i=i+1){
        for (int j=0; j < CONFIG_T::n_in/CONFIG_T::block_y; j=j+1){
            for (int ii=0; ii < CONFIG_T::block_x; ++ii){
                //#pragma HLS UNROLL
                for (int jj=0; jj < CONFIG_T::block_y; ++jj){
                    #pragma HLS UNROLL
                    res[j*CONFIG_T::block_y+jj].write(output_buffer[i][j][ii][jj]);
                }
            }
        }
    }
    
}

template<class data_T, class res_T, typename CONFIG_T>
void FFN(
    hls::stream<data_T>    &data,
    hls::stream<res_T>     &res,
    typename CONFIG_T::weight_t  weights1[CONFIG_T::n_in/CONFIG_T::block_y][CONFIG_T::hidden_dim/CONFIG_T::block_k][CONFIG_T::block_y][CONFIG_T::block_k],
    typename CONFIG_T::bias_t    biases1[CONFIG_T::hidden_dim/CONFIG_T::block_k][CONFIG_T::block_k],
    typename CONFIG_T::weight_t  weights2[CONFIG_T::hidden_dim/CONFIG_T::block_k][CONFIG_T::n_out/CONFIG_T::block_y][CONFIG_T::block_k][CONFIG_T::block_y],
    typename CONFIG_T::bias_t    biases2[CONFIG_T::n_out/CONFIG_T::block_y][CONFIG_T::block_y])
{
    data_T  input_buffer[CONFIG_T::seq_len/CONFIG_T::block_x][CONFIG_T::n_in/CONFIG_T::block_y][CONFIG_T::block_x][CONFIG_T::block_y];  
    res_T   output_buffer[CONFIG_T::seq_len/CONFIG_T::block_x][CONFIG_T::n_in/CONFIG_T::block_y][CONFIG_T::block_x][CONFIG_T::block_y];
    #pragma HLS ARRAY_PARTITION variable=input_buffer   complete dim=3
    #pragma HLS ARRAY_PARTITION variable=input_buffer   complete dim=4
    #pragma HLS ARRAY_PARTITION variable=output_buffer  complete dim=3
    #pragma HLS ARRAY_PARTITION variable=output_buffer  complete dim=4
    #pragma HLS ARRAY_PARTITION variable=weights1       complete dim=3
    #pragma HLS ARRAY_PARTITION variable=weights1       complete dim=4
    #pragma HLS ARRAY_PARTITION variable=weights2       complete dim=3
    #pragma HLS ARRAY_PARTITION variable=weights2       complete dim=4
    #pragma HLS ARRAY_PARTITION variable=biases1        complete dim=2
    #pragma HLS ARRAY_PARTITION variable=biases2        complete dim=2

    typename CONFIG_T::accum_t dense1_out[CONFIG_T::block_x][CONFIG_T::block_k];
    typename CONFIG_T::accum_t dense2_out[CONFIG_T::block_x][CONFIG_T::block_k];
    typename CONFIG_T::accum_t dense_out[CONFIG_T::block_x][CONFIG_T::block_k];
    typename CONFIG_T::accum_t buffer[CONFIG_T::block_x][CONFIG_T::block_y];
    #pragma HLS ARRAY_PARTITION variable=dense1_out     complete dim=0
    #pragma HLS ARRAY_PARTITION variable=dense2_out     complete dim=0
    #pragma HLS ARRAY_PARTITION variable=dense_out      complete dim=0
    //#pragma HLS DATAFLOW
    store_input:
    for (int i=0; i <CONFIG_T::seq_len/CONFIG_T::block_x; i=i+1){
        for (int j=0; j < CONFIG_T::n_in/CONFIG_T::block_y; j=j+1){
            for (int ii=0; ii < CONFIG_T::block_x; ++ii){
                //#pragma HLS UNROLL
                for (int jj=0; jj < CONFIG_T::block_y; ++jj){
                    #pragma HLS UNROLL
                    input_buffer[i][j][ii][jj] = data.read();
                    output_buffer[i][j][ii][jj] = biases2[j][jj];
                }
            }
        }
    }
    
    data_T tmp_input_buffer;
    res_T tmp_output_buffer;
    int i = 0;
    int k = 0;
    int m = 0;
    int n = 0;
    int total_cycle = ((CONFIG_T::seq_len*CONFIG_T::hidden_dim)/(CONFIG_T::block_x*CONFIG_T::block_k)) + 1;
    static bool dense1_init = false;
    //#pragma HLS DEPENDENCE variable=output_buffer intra false
    //std::cout << "dense1_out[0][0] = "<< std::endl;
    pipeline_product1n2: // 1st inner product with ijk indexing and 2nd outter product with mnp indexing
    for (int c=0; c < total_cycle; ++c){
        for (int p=0; p < CONFIG_T::n_in/CONFIG_T::block_y; p=p+1){
            #pragma HLS PIPELINE II=1
            if (p==0) {
                for (int ii=0; ii < CONFIG_T::block_x; ++ii){
                    #pragma HLS UNROLL
                    for (int kk=0; kk < CONFIG_T::block_k; ++kk){
                        #pragma HLS UNROLL
                        if (dense1_init) {
                            dense2_out[ii][kk] = biases1[kk][k];
                            if (dense1_out[ii][kk] < 0) {
                                dense1_out[ii][kk] = 0;
                            }
                        } else {
                            dense1_out[ii][kk] = biases1[kk][k];
                            if (dense2_out[ii][kk] < 0) {
                                dense2_out[ii][kk] = 0;
                            }
                        }           
                    }
                }
            }
            inner_product:
            for (int ii=0; ii < CONFIG_T::block_x; ++ii){
                #pragma HLS UNROLL
                for (int pp=0; pp < CONFIG_T::block_y; ++pp){
                    #pragma HLS UNROLL
                	tmp_output_buffer = output_buffer[m][p][ii][pp];
                    tmp_input_buffer = input_buffer[i][p][ii][pp];
                    for (int kk=0; kk < CONFIG_T::block_k; ++kk){
                        #pragma HLS UNROLL
                        typename CONFIG_T::accum_t temp = tmp_input_buffer * weights1[p][k][pp][kk];
                        if (dense1_init) {
                            if ((i < CONFIG_T::seq_len/CONFIG_T::block_x) && (k < CONFIG_T::hidden_dim/CONFIG_T::block_k)) {
                                dense2_out[ii][kk] += temp;
                            }
                            dense_out[ii][kk] = dense1_out[ii][kk];
                        } else {
                            if ((i < CONFIG_T::seq_len/CONFIG_T::block_x) && (k < CONFIG_T::hidden_dim/CONFIG_T::block_k)) {
                                dense1_out[ii][kk] += temp;
                            }
                            dense_out[ii][kk] = dense2_out[ii][kk];
                        }
                        if (c>0){
                        	tmp_output_buffer = tmp_output_buffer + dense_out[ii][kk] * weights2[n][p][kk][pp];
                        }
                    }
                    output_buffer[m][p][ii][pp] = tmp_output_buffer;
                }
            }
            if (p==(CONFIG_T::n_in/CONFIG_T::block_y-1)) { // last cycle of pipeline
                if (dense1_init){
                    dense1_init = false;
                } else {
                    dense1_init = true;
                }
                if (c < total_cycle-1) { //cycle 0~total_cycle-1
                    k = k + 1;
                    if (k == CONFIG_T::hidden_dim/CONFIG_T::block_k){
                        k = 0;
                        i = i + 1;
                    }
                }
                if (c > 0) { //cycle 1~total_cycle
                    n = n + 1;
                    if (n == CONFIG_T::hidden_dim/CONFIG_T::block_k){
                        n = 0;
                        m = m + 1;
                    }
                }
            }
        }
    }
    write_output:
    for (int i=0; i <CONFIG_T::seq_len/CONFIG_T::block_x; i=i+1){
        for (int j=0; j < CONFIG_T::n_in/CONFIG_T::block_y; j=j+1){
            for (int ii=0; ii < CONFIG_T::block_x; ++ii){
                //#pragma HLS UNROLL
                for (int jj=0; jj < CONFIG_T::block_y; ++jj){
                    #pragma HLS UNROLL
                    res.write(output_buffer[i][j][ii][jj]);
                }
            }
        }
    }
    
}


}


#endif
