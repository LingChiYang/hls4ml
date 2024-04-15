//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

#ifndef NNET_LAYERNORM_SINGLE_STREAM_H_
#define NNET_LAYERNORM_SINGLE_STREAM_H_

#include "nnet_common.h"
#include "nnet_dense.h"
#include "hls_stream.h"
#include <math.h>
#include <iostream>

#include "hls_math.h"

namespace nnet {

struct layernorm_config {
    static const unsigned seq_len = 180;
    static const unsigned embed_dim = 182;
    static const unsigned table_size = 1024;
    static constexpr double table_range = 1;
    static const unsigned log_table_range = 10;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
    typedef model_default_t table_t;
};

template<typename CONFIG_T, int N_TABLE, int dim>
void init_n_invert_sqr_table(typename CONFIG_T::table_t table_out[N_TABLE])
{
    //float inv_range = CONFIG_T::table_range;
    // Inversion function:
    //   result = 1/sqrt(x)
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range 0 to +0.01)
        float in_val = ii/float(N_TABLE >> CONFIG_T::log_table_range);
        // Next, compute lookup table function
        if (in_val > 0.0) table_out[ii] = 1.0/sqrt(in_val);
        else table_out[ii] = 0.0;
    }
    //print all table value
    //for (int i = 0; i < N_TABLE; i++){
    //    std::cout << "table_out[" << i << "]: " << table_out[i] << std::endl;
    //}
}

template<class data_T, class res_T, typename CONFIG_T>
void LayerNormalize(
    hls::stream<data_T>    &data,
    hls::stream<res_T>     &res,
    typename CONFIG_T::scale_t  scale[CONFIG_T::embed_dim/CONFIG_T::tiling_factor[1]][CONFIG_T::tiling_factor[1]],
    typename CONFIG_T::bias_t   bias[CONFIG_T::embed_dim/CONFIG_T::tiling_factor[1]][CONFIG_T::tiling_factor[1]]
)
{
    typename data_T::value_type in_val[CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]][CONFIG_T::embed_dim/CONFIG_T::tiling_factor[1]][CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[1]];
    typename res_T::value_type outval[CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]][CONFIG_T::embed_dim/CONFIG_T::tiling_factor[1]][CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[1]];
    #pragma HLS ARRAY_PARTITION variable=scale complete dim=2
    #pragma HLS ARRAY_PARTITION variable=bias complete dim=2
    #pragma HLS ARRAY_PARTITION variable=in_val complete dim=3
    #pragma HLS ARRAY_PARTITION variable=in_val complete dim=4
    #pragma HLS ARRAY_PARTITION variable=outval complete dim=4
    #pragma HLS ARRAY_PARTITION variable=outval complete dim=3

    #ifdef __HLS_SYN__
        bool initialized = false;
        typename CONFIG_T::table_t invert_sqr_table[CONFIG_T::table_size];
    #else
        static bool initialized = false;
        static typename CONFIG_T::table_t invert_sqr_table[CONFIG_T::table_size];
    #endif
    if (!initialized) {
        init_n_invert_sqr_table<CONFIG_T, CONFIG_T::table_size, CONFIG_T::embed_dim>(invert_sqr_table);
        initialized = true;
    }

    const unsigned int tf_T = CONFIG_T::tiling_factor[0];
    const unsigned int tf_N = CONFIG_T::tiling_factor[1];
    const unsigned int T = CONFIG_T::seq_len/tf_T;
    const unsigned int N = CONFIG_T::embed_dim/tf_N;

    data_T data_pack;
    store_input: 
    for (int j=0; j < T; ++j){
        for (int i=0; i < N; ++i){
            #pragma HLS PIPELINE
            for (int jj=0; jj < tf_T; ++jj){
                #pragma HLS UNROLL
                for (int ii=0; ii < tf_N; ++ii){
                    #pragma HLS UNROLL
                    if (jj == 0 && ii == 0) {
                        data_pack = data.read();
                    }
                    in_val[j][i][jj][ii] = data_pack[jj*tf_N+ii];
                }
            }
        }
    }
    std::ofstream file2("layernorm_in_matrix.txt", std::ios::app);
    if (file2.is_open()) {
        for (int i = 0; i < T; ++i) {
            for (int ii = 0; ii < tf_T; ++ii) {
                for (int j = 0; j < N; ++j) {
                    for (int jj = 0; jj < tf_N; ++jj) {
                        if (j == N-1) {
                            file2 << in_val[i][j][ii][jj];
                        } else {
                            file2 << in_val[i][j][ii][jj] << " ";
                        }
                    }
                }
                file2 << "\n";
            }
        }
        file2.close();
    } else {
        std::cout << "Unable to open file";
    }
    const typename CONFIG_T::mean_t embed_dim_inv = 1.0/CONFIG_T::embed_dim;
    typename CONFIG_T::mean_t xsqrsum_1[CONFIG_T::tiling_factor[0]];
    typename CONFIG_T::mean_t xsum_1[CONFIG_T::tiling_factor[0]];
    typename CONFIG_T::mean_t prev_xsum_1[CONFIG_T::tiling_factor[0]];
    typename CONFIG_T::mean_t xsqrsum_2[CONFIG_T::tiling_factor[0]];
    typename CONFIG_T::mean_t xsum_2[CONFIG_T::tiling_factor[0]];
    typename CONFIG_T::mean_t prev_xsum_2[CONFIG_T::tiling_factor[0]];
    typename CONFIG_T::mean_t xsum[CONFIG_T::tiling_factor[0]];
    typename CONFIG_T::mean_t xsqrsum[CONFIG_T::tiling_factor[0]];
    typename CONFIG_T::mean_t xmean[CONFIG_T::tiling_factor[0]];
    bool write_buffer1[tf_T];
    typename CONFIG_T::table_t deno_inver[tf_T];
    for (int jj=0; jj < tf_T; ++jj){
        write_buffer1[jj] = true;
    }
    #pragma HLS ARRAY_PARTITION variable=xsqrsum_1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=xsum_1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=xsqrsum_2 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=xsum_2 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=xsqrsum complete dim=1
    #pragma HLS ARRAY_PARTITION variable=xsum complete dim=1
    #pragma HLS ARRAY_PARTITION variable=deno_inver complete dim=1
    #pragma HLS ARRAY_PARTITION variable=write_buffer1 complete dim=1
    layerNorm:  for (int j=0; j <= T; ++j){
                    for (int i=0; i < N; ++i){
                        #pragma HLS PIPELINE
                        for (int jj=0; jj < tf_T; ++jj){
                            #pragma HLS UNROLL
                            if (i == 0){
                                if (write_buffer1[jj] == true){
                                    xsqrsum_1[jj] = 0;
                                    xsum_1[jj] = 0;
                                } else {
                                    xsqrsum_2[jj] = 0;
                                    xsum_2[jj] = 0;
                                }
                            }
                            for (int ii=0; ii < tf_N; ++ii){
                                #pragma HLS UNROLL
                                if (j < T){
                                    typename CONFIG_T::mean_t tmp = in_val[j][i][jj][ii];
                                    typename CONFIG_T::mean_t tmp2 = tmp*tmp*embed_dim_inv;
                                    if (write_buffer1[jj] == true){
                                        xsum_1[jj] = xsum_1[jj] + tmp;
                                        xsqrsum_1[jj] = xsqrsum_1[jj] + tmp2;//(tmp - xsum_1[jj])*(tmp - prev_xsum_1[jj]);
                                    } else {
                                        xsum_2[jj] = xsum_2[jj] + tmp;
                                        xsqrsum_2[jj] = xsqrsum_2[jj] + tmp2;//(tmp - xsum_2[jj])*(tmp - prev_xsum_2[jj]);
                                    }
                                }
                                if (j > 0){
                                    if (write_buffer1[jj] == false){
                                        xsum[jj] = xsum_1[jj];
                                    } else {
                                        xsum[jj] = xsum_2[jj];
                                    }
                                    xmean[jj] = xsum[jj]*embed_dim_inv;
                                    outval[j-1][i][jj][ii] = (in_val[j-1][i][jj][ii] - xmean[jj])*deno_inver[jj]*scale[i][ii] + bias[i][ii];
                                }
                            }
                            if (i == (N-1)){
                                write_buffer1[jj] = !write_buffer1[jj];
                                if (write_buffer1[jj] == false){
                                    xsqrsum[jj] = xsqrsum_1[jj];
                                    xsum[jj] = xsum_1[jj];
                                } else {
                                    xsqrsum[jj] = xsqrsum_2[jj];
                                    xsum[jj] = xsum_2[jj];
                                }
                                xmean[jj] = xsum[jj]*embed_dim_inv;
                                typename CONFIG_T::mean_t tmp3 = xsqrsum[jj]-xmean[jj]*xmean[jj];
                                //typename CONFIG_T::mean_t tmp3 = CONFIG_T::embed_dim*xsqrsum[jj]-xsum[jj]*xsum[jj];
                                int index = tmp3*(CONFIG_T::table_size) >> CONFIG_T::log_table_range;
                                if (index < 0)   index = 0;
                                if (index > CONFIG_T::table_size-1) index = CONFIG_T::table_size-1;
                                deno_inver[jj] = (typename CONFIG_T::table_t) invert_sqr_table[index];
                            }
                        }
                    }
                }
    std::ofstream file("layernorm_out_matrix.txt", std::ios::app);
    if (file.is_open()) {
        for (int i = 0; i < T; ++i) {
            for (int ii = 0; ii < tf_T; ++ii) {
                for (int j = 0; j < N; ++j) {
                    for (int jj = 0; jj < tf_N; ++jj) {
                        if (j == N-1) {
                            file << outval[i][j][ii][jj];
                        } else {
                            file << outval[i][j][ii][jj] << " ";
                        }
                    }
                }
                file << "\n";
            }
        }
        file.close();
    } else {
        std::cout << "Unable to open file";
    }
    store_output:   for (int j=0; j < T; ++j){
                        for (int i=0; i < N; ++i){
                            #pragma HLS PIPELINE
                            for (int jj=0; jj < tf_T; ++jj){
                                #pragma HLS UNROLL
                                for (int ii=0; ii < tf_N; ++ii){
                                    #pragma HLS UNROLL
                                    res_T res_pack;
                                    res_pack[jj*tf_N+ii] = outval[j][i][jj][ii];
                                    if (jj == tf_T-1 && ii == tf_N-1){
                                        res.write(res_pack);
                                    }
                                }
                            }
                        }
                    }
    
}


}

#endif
