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
    static const unsigned feature_dim = 182;
    static const unsigned table_size = 1024;
    static constexpr double table_range = 1;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
    typedef model_default_t table_t;
};

template<typename CONFIG_T, int N_TABLE, int dim>
void init_n_invert_sqr_table(typename CONFIG_T::table_t table_out[N_TABLE])
{
    float inv_range = CONFIG_T::table_range;
    // Inversion function:
    //   result = 1/sqrt(x)
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range 0 to +0.01)
        float in_val = inv_range*ii/float(N_TABLE);
        // Next, compute lookup table function
        if (in_val > 0.0) table_out[ii] = float(dim)/sqrt(in_val);
        else table_out[ii] = 0.0;
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void LayerNormalize(
    hls::stream<data_T>    &data,
    hls::stream<res_T>     &res,
    typename CONFIG_T::scale_t  scale[CONFIG_T::feature_dim/CONFIG_T::tiling_factor[1]][CONFIG_T::tiling_factor[1]],
    typename CONFIG_T::bias_t   bias[CONFIG_T::feature_dim/CONFIG_T::tiling_factor[1]][CONFIG_T::tiling_factor[1]]
)
{
    data_T in_val[CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]][CONFIG_T::feature_dim/CONFIG_T::tiling_factor[1]][CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[1]];
    data_T outval[CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]][CONFIG_T::feature_dim/CONFIG_T::tiling_factor[1]][CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[1]];
    #pragma HLS ARRAY_PARTITION variable=scale complete dim=2
    #pragma HLS ARRAY_PARTITION variable=bias complete dim=2
    #pragma HLS ARRAY_PARTITION variable=in_val complete dim=3
    #pragma HLS ARRAY_PARTITION variable=in_val complete dim=4
    #pragma HLS ARRAY_PARTITION variable=outval complete dim=4
    #pragma HLS ARRAY_PARTITION variable=outval complete dim=3

    int inv_range_inv = (int) 1/ CONFIG_T::table_range;
    typename CONFIG_T::table_t deno_inver = 0;
    #ifdef __HLS_SYN__
        bool initialized = false;
        typename CONFIG_T::table_t invert_sqr_table[CONFIG_T::table_size];
    #else
        static bool initialized = false;
        static typename CONFIG_T::table_t invert_sqr_table[CONFIG_T::table_size];
    #endif
    if (!initialized) {
        init_n_invert_sqr_table<CONFIG_T, CONFIG_T::table_size, CONFIG_T::feature_dim>(invert_sqr_table);
        initialized = true;
    }

    const unsigned int tf_T = CONFIG_T::tiling_factor[0];
    const unsigned int tf_N = CONFIG_T::tiling_factor[1];
    const unsigned int T = CONFIG_T::seq_len/tf_T;
    const unsigned int N = CONFIG_T::feature_dim/tf_N;

    data_T data_pack;
    store_input: 
    for (int j=0; j < T; ++j){
        for (int i=0; i < N; ++i){
            for (int jj=0; jj < tf_T; ++jj){
                for (int ii=0; ii < tf_N; ++ii){
                    #pragma HLS PIPELINE
                    if (jj == 0 && ii == 0) {
                        data_pack = data.read();
                    }
                    in_val[j][i][jj][ii] = data_pack[jj*tf_N+ii];
                }
            }
        }
    }
    typename CONFIG_T::mean_t xsqrsum_1[CONFIG_T::block_x];
    typename CONFIG_T::mean_t xsum_1[CONFIG_T::block_x];
    typename CONFIG_T::mean_t prev_xsum_1[CONFIG_T::block_x];
    typename CONFIG_T::mean_t xsqrsum_2[CONFIG_T::block_x];
    typename CONFIG_T::mean_t xsum_2[CONFIG_T::block_x];
    typename CONFIG_T::mean_t prev_xsum_2[CONFIG_T::block_x];
    typename CONFIG_T::mean_t xsum[CONFIG_T::block_x];
    typename CONFIG_T::mean_t xsqrsum[CONFIG_T::block_x];
    #pragma HLS ARRAY_PARTITION variable=xsqrsum_1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=xsum_1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=prev_xsum_1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=xsqrsum_2 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=xsum_2 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=prev_xsum_2 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=xsqrsum complete dim=1
    #pragma HLS ARRAY_PARTITION variable=xsum complete dim=1
    bool mean_init = false;
    layerNorm:  for (int j=0; j <= T; ++j){
                    for (int i=0; i < N; ++i){
                        #pragma HLS PIPELINE
                        for (int jj=0; jj < tf_T; ++jj){
                            #pragma HLS UNROLL
                            if (i == 0){
                                if (mean_init == false){
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
                                    typename CONFIG_T::mean_t tmp2 = tmp*tmp;
                                    if (mean_init == false){
                                        xsum_1[jj] = xsum_1[jj] + tmp;
                                        xsqrsum_1[jj] = xsqrsum_1[jj] + tmp2;//(tmp - xsum_1[jj])*(tmp - prev_xsum_1[jj]);
                                        //prev_xsum_1[jj] = xsum_1[jj];
                                    } else {
                                        xsum_2[jj] = xsum_2[jj] + tmp;
                                        xsqrsum_2[jj] = xsqrsum_2[jj] + tmp2;//(tmp - xsum_2[jj])*(tmp - prev_xsum_2[jj]);
                                        //prev_xsum_2[jj] = xsum_2[jj];
                                    }
                                }
                                if (j > 0){
                                    if (mean_init == false){
                                        xsum[jj] = xsum_1[jj];
                                    } else {
                                        xsum[jj] = xsum_2[jj];
                                    }
                                    outval[j-1][i][jj][ii] = (in_val[j-1][i][jj][ii]*CONFIG_T::feature_dim - xsum[jj])*deno_inver*scale[i][ii] + bias[i][ii];
                                    //outval[j-1][i][jj][ii] = (in_val[j-1][i][jj][ii]*dim - prev_xsum[jj])*deno_inver*scale[i][ii] + bias[i][ii];
                                }
                            }
                            if (i == (N-1)){
                                if (mean_init == false){
                                    xsqrsum[jj] = xsqrsum_1[jj];
                                    xsum[jj] = xsum_1[jj];
                                    mean_init = true;
                                } else {
                                    xsqrsum[jj] = xsqrsum_2[jj];
                                    xsum[jj] = xsum_2[jj];
                                    mean_init = false;
                                }
                                typename CONFIG_T::mean_t tmp3 = CONFIG_T::feature_dim*xsqrsum[jj]-xsum[jj]*xsum[jj];
                                int index = tmp3*(CONFIG_T::table_size)*inv_range_inv;
                                if (CONFIG_T::table_range > 1) index = tmp3*(CONFIG_T::table_size)/ (int)CONFIG_T::table_range;
                                if (index < 0)   index = 0;
                                if (index > CONFIG_T::table_size-1) index = CONFIG_T::table_size-1;
                                deno_inver = (typename CONFIG_T::table_t) invert_sqr_table[index];
                            }
                        }
                    }
                }

    store_output:   for (int j=0; j < T; ++j){
                        for (int i=0; i < N; ++i){
                            for (int jj=0; jj < tf_T; ++jj){
                                for (int ii=0; ii < tf_N; ++ii){
                                    #pragma HLS PIPELINE
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

template<class data_T, class res_T, typename CONFIG_T>
void layernormalize(
    hls::stream<data_T>    &data,
    hls::stream<res_T>     &res,
    typename CONFIG_T::scale_t  scale[CONFIG_T::n_in/CONFIG_T::seq_len/CONFIG_T::block_y][CONFIG_T::block_y],
    typename CONFIG_T::bias_t   bias[CONFIG_T::n_in/CONFIG_T::seq_len/CONFIG_T::block_y][CONFIG_T::block_y]
)
{
    static const unsigned dim = CONFIG_T::n_in/CONFIG_T::seq_len;
    data_T in_val[CONFIG_T::seq_len/CONFIG_T::block_x][dim/CONFIG_T::block_y][CONFIG_T::block_x][CONFIG_T::block_y];
    data_T outval[CONFIG_T::seq_len/CONFIG_T::block_x][dim/CONFIG_T::block_y][CONFIG_T::block_x][CONFIG_T::block_y];
    #pragma HLS ARRAY_PARTITION variable=scale complete dim=2
    #pragma HLS ARRAY_PARTITION variable=bias complete dim=2
    #pragma HLS ARRAY_PARTITION variable=in_val complete dim=3
    #pragma HLS ARRAY_PARTITION variable=in_val complete dim=4
    #pragma HLS ARRAY_PARTITION variable=outval complete dim=4
    #pragma HLS ARRAY_PARTITION variable=outval complete dim=3

    int inv_range_inv = (int) 1/ CONFIG_T::table_range;
    typename CONFIG_T::table_t deno_inver = 0;
    #ifdef __HLS_SYN__
        bool initialized = false;
        typename CONFIG_T::table_t invert_sqr_table[CONFIG_T::table_size];
    #else
        static bool initialized = false;
        static typename CONFIG_T::table_t invert_sqr_table[CONFIG_T::table_size];
    #endif
    if (!initialized) {
        init_n_invert_sqr_table<CONFIG_T, CONFIG_T::table_size, dim>(invert_sqr_table);
        initialized = true;
    }
    
    store_input: for (int j=0; j <CONFIG_T::seq_len/CONFIG_T::block_x; ++j){
        for (int i=0; i < dim/CONFIG_T::block_y; ++i){
            for (int jj=0; jj < CONFIG_T::block_x; ++jj){
                for (int ii=0; ii < CONFIG_T::block_y; ++ii){
                    #pragma HLS PIPELINE
                    in_val[j][i][jj][ii] = data.read();
                }
            }
        }
    }
    typename CONFIG_T::mean_t xsqrsum_1[CONFIG_T::block_x];
    typename CONFIG_T::mean_t xsum_1[CONFIG_T::block_x];
    typename CONFIG_T::mean_t prev_xsum_1[CONFIG_T::block_x];
    typename CONFIG_T::mean_t xsqrsum_2[CONFIG_T::block_x];
    typename CONFIG_T::mean_t xsum_2[CONFIG_T::block_x];
    typename CONFIG_T::mean_t prev_xsum_2[CONFIG_T::block_x];
    typename CONFIG_T::mean_t xsum[CONFIG_T::block_x];
    typename CONFIG_T::mean_t xsqrsum[CONFIG_T::block_x];
    #pragma HLS ARRAY_PARTITION variable=xsqrsum_1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=xsum_1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=prev_xsum_1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=xsqrsum_2 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=xsum_2 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=prev_xsum_2 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=xsqrsum complete dim=1
    #pragma HLS ARRAY_PARTITION variable=xsum complete dim=1
    bool mean_init = false;
    layerNorm:  for (int j=0; j <= CONFIG_T::seq_len/CONFIG_T::block_x; ++j){
                    for (int i=0; i < dim/CONFIG_T::block_y; ++i){
                        #pragma HLS PIPELINE
                        for (int jj=0; jj < CONFIG_T::block_x; ++jj){
                            #pragma HLS UNROLL
                            if (i == 0){
                                if (mean_init == false){
                                    xsqrsum_1[jj] = 0;
                                    xsum_1[jj] = 0;
                                } else {
                                    xsqrsum_2[jj] = 0;
                                    xsum_2[jj] = 0;
                                }
                            }
                            for (int ii=0; ii < CONFIG_T::block_y; ++ii){
                                #pragma HLS UNROLL
                                if (j < CONFIG_T::seq_len/CONFIG_T::block_x){
                                    typename CONFIG_T::mean_t tmp = in_val[j][i][jj][ii];
                                    typename CONFIG_T::mean_t tmp2 = tmp*tmp;
                                    if (mean_init == false){
                                        xsum_1[jj] = xsum_1[jj] + tmp;
                                        xsqrsum_1[jj] = xsqrsum_1[jj] + tmp2;//(tmp - xsum_1[jj])*(tmp - prev_xsum_1[jj]);
                                        //prev_xsum_1[jj] = xsum_1[jj];
                                    } else {
                                        xsum_2[jj] = xsum_2[jj] + tmp;
                                        xsqrsum_2[jj] = xsqrsum_2[jj] + tmp2;//(tmp - xsum_2[jj])*(tmp - prev_xsum_2[jj]);
                                        //prev_xsum_2[jj] = xsum_2[jj];
                                    }
                                }
                                if (j > 0){
                                    if (mean_init == false){
                                        xsum[jj] = xsum_1[jj];
                                    } else {
                                        xsum[jj] = xsum_2[jj];
                                    }
                                    outval[j-1][i][jj][ii] = (in_val[j-1][i][jj][ii]*dim - xsum[jj])*deno_inver*scale[i][ii] + bias[i][ii];
                                    //outval[j-1][i][jj][ii] = (in_val[j-1][i][jj][ii]*dim - prev_xsum[jj])*deno_inver*scale[i][ii] + bias[i][ii];
                                }
                            }
                            if (i == (dim/CONFIG_T::block_y-1)){
                                if (mean_init == false){
                                    //print xsqrsum_1[jj];
                                    //std::cout << "xsqrsum_1: " << xsqrsum_1[jj] << std::endl;
                                    xsqrsum[jj] = xsqrsum_1[jj];
                                    xsum[jj] = xsum_1[jj];
                                    mean_init = true;
                                } else {
                                    //std::cout << "xsqrsum_2: " << xsqrsum_2[jj] << std::endl;
                                    xsqrsum[jj] = xsqrsum_2[jj];
                                    xsum[jj] = xsum_2[jj];
                                    mean_init = false;
                                }
                                typename CONFIG_T::mean_t tmp3 = dim*xsqrsum[jj]-xsum[jj]*xsum[jj];
                                int index = tmp3*(CONFIG_T::table_size)*inv_range_inv;
                                if (CONFIG_T::table_range > 1) index = tmp3*(CONFIG_T::table_size)/ (int)CONFIG_T::table_range;
                                if (index < 0)   index = 0;
                                if (index > CONFIG_T::table_size-1) index = CONFIG_T::table_size-1;
                                deno_inver = (typename CONFIG_T::table_t) invert_sqr_table[index];
                            }
                        }
                    }
                }

    store_output:   for (int j=0; j <CONFIG_T::seq_len/CONFIG_T::block_x; ++j){
                        for (int i=0; i < dim/CONFIG_T::block_y; ++i){
                            for (int jj=0; jj < CONFIG_T::block_x; ++jj){
                                for (int ii=0; ii < CONFIG_T::block_y; ++ii){
                                    #pragma HLS PIPELINE
                                    res.write(outval[j][i][jj][ii]);
                                }
                            }
                        }
                    }
    
}

}

#endif
