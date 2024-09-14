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
#include <iomanip>
#include "hls_math.h"

namespace nnet {

struct layernorm_config {
    static const unsigned seq_len = 180;
    static const unsigned embed_dim = 182;
    static const unsigned table_size = 1024;
    static constexpr double table_range = 1;
    static const unsigned log_table_range = 10;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
};

template<typename CONFIG_T, int N_TABLE, int dim>
void init_n_invert_sqr_table(typename CONFIG_T::var_table_t table_out[N_TABLE])
{
    //float inv_range = CONFIG_T::table_range;
    // Inversion function:
    //   result = 1/sqrt(x)
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range 0 to +0.01)
        double in_val;
        if (N_TABLE > CONFIG_T::table_range) {
            in_val = ii/double(N_TABLE / CONFIG_T::table_range);
        } else {
            in_val = ii*double(CONFIG_T::table_range / N_TABLE);
        }
        //float in_val = ii/float(N_TABLE / CONFIG_T::table_range);
        // Next, compute lookup table function
        if (in_val > 0.0) table_out[ii] = 1.0/sqrt(in_val);
        else table_out[ii] = 1024; // a large number
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void LayerNormalize(
    hls::stream<data_T>    &data,
    hls::stream<res_T>     &res,
    typename CONFIG_T::scale_t  scale[CONFIG_T::embed_dim],
    typename CONFIG_T::bias_t   bias[CONFIG_T::embed_dim]
)
{
    typename data_T::value_type in_val[CONFIG_T::seq_len*CONFIG_T::embed_dim];
    typename res_T::value_type outval[CONFIG_T::seq_len*CONFIG_T::embed_dim];
    //#pragma HLS ARRAY_PARTITION variable=scale complete dim=2
    //#pragma HLS ARRAY_PARTITION variable=bias complete dim=2
    //#pragma HLS ARRAY_PARTITION variable=in_val complete dim=3
    //#pragma HLS ARRAY_PARTITION variable=in_val complete dim=4
    //#pragma HLS ARRAY_PARTITION variable=outval complete dim=4
    //#pragma HLS ARRAY_PARTITION variable=outval complete dim=3
    #pragma DATAFLOW
    #pragma HLS BIND_STORAGE variable=in_val type=ram_s2p impl=bram
    #pragma HLS BIND_STORAGE variable=outval type=ram_s2p impl=bram

    #ifdef __HLS_SYN__
        bool initialized = false;
        typename CONFIG_T::var_table_t invert_sqr_table[CONFIG_T::table_size];
    #else
        static bool initialized = false;
        static typename CONFIG_T::var_table_t invert_sqr_table[CONFIG_T::table_size];
    #endif
    if (!initialized) {
        init_n_invert_sqr_table<CONFIG_T, CONFIG_T::table_size, CONFIG_T::embed_dim>(invert_sqr_table);
        initialized = true;
    }

    const unsigned int tf_T = CONFIG_T::tiling_factor[0];
    const unsigned int tf_N = CONFIG_T::tiling_factor[1];
    const unsigned int T = CONFIG_T::seq_len/tf_T;
    const unsigned int N = CONFIG_T::embed_dim/tf_N;
    #pragma HLS ARRAY_RESHAPE variable=in_val cyclic factor=tf_N*tf_T dim=1
    #pragma HLS ARRAY_RESHAPE variable=outval cyclic factor=tf_N*tf_T dim=1
    #pragma HLS ARRAY_RESHAPE variable=scale cyclic factor=tf_N dim=1
    #pragma HLS ARRAY_RESHAPE variable=bias cyclic factor=tf_N dim=1
    constexpr float dim_inv = 1.0/CONFIG_T::embed_dim;
    constexpr int int_bits_embed_dim = ceil(log2(dim_inv));
    const ap_ufixed<18,int_bits_embed_dim,AP_RND_CONV> embed_dim_inv = dim_inv;
    //std::cout << "int_bits_embed_dim = " << int_bits_embed_dim << std::endl;
    //std::cout << "embed_dim_inv = " << embed_dim_inv << std::endl;
    typename CONFIG_T::accum_t xsqrsum_1[CONFIG_T::tiling_factor[0]];
    typename CONFIG_T::accum_t xsum_1[CONFIG_T::tiling_factor[0]];
    typename CONFIG_T::accum_t prev_xsum_1[CONFIG_T::tiling_factor[0]];
    typename CONFIG_T::accum_t xsqrsum_2[CONFIG_T::tiling_factor[0]];
    typename CONFIG_T::accum_t xsum_2[CONFIG_T::tiling_factor[0]];
    typename CONFIG_T::accum_t prev_xsum_2[CONFIG_T::tiling_factor[0]];
    typename CONFIG_T::accum_t xsum[CONFIG_T::tiling_factor[0]];
    typename CONFIG_T::accum_t xsqrsum[CONFIG_T::tiling_factor[0]];
    typename CONFIG_T::mean_t xmean[CONFIG_T::tiling_factor[0]];
    typename data_T::value_type row_buffer[CONFIG_T::embed_dim*CONFIG_T::tiling_factor[0]];
    //#pragma HLS STREAM variable=row_buffer depth=2 type=pipo
    bool write_buffer1[tf_T];
    typename CONFIG_T::var_table_t deno_inver[tf_T];
    for (int jj=0; jj < tf_T; ++jj){
        write_buffer1[jj] = true;
    }
    int index;
    #pragma HLS ARRAY_PARTITION variable=xsqrsum_1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=xsum_1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=xsqrsum_2 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=xsum_2 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=xsqrsum complete dim=1
    #pragma HLS ARRAY_PARTITION variable=xsum complete dim=1
    #pragma HLS ARRAY_PARTITION variable=deno_inver complete dim=1
    #pragma HLS ARRAY_PARTITION variable=write_buffer1 complete dim=1
    //#pragma HLS ARRAY_PARTITION variable=row_buffer complete dim=2
    //#pragma HLS ARRAY_PARTITION variable=row_buffer complete dim=3
    #pragma HLS ARRAY_RESHAPE variable=row_buffer cyclic factor=tf_N*tf_T dim=1
    typename data_T::value_type tmp;
    data_T data_pack;
    res_T res_pack;
    typename CONFIG_T::accum_t xsum_debug[CONFIG_T::seq_len];
    typename CONFIG_T::accum_t xvar_debug[CONFIG_T::seq_len];
    typename res_T::value_type res_debug[T][N][tf_T][tf_N];
    typename data_T::value_type data_debug[T][N][tf_T][tf_N];

    typename CONFIG_T::accum_t norm_tmp;

    typename CONFIG_T::accum_t var_in_debug[CONFIG_T::seq_len][CONFIG_T::tiling_factor[0]];
    typename CONFIG_T::var_table_t var_out_debug[CONFIG_T::seq_len][CONFIG_T::tiling_factor[0]];
    //typename CONFIG_T::sum_sqr_t sqr_debug[CONFIG_T::seq_len][CONFIG_T::embed_dim/CONFIG_T::tiling_factor[1]][CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[1]];
    layerNorm:  for (int j=0; j <= T; ++j){
                    for (int i=0; i < N; ++i){
                        #pragma HLS PIPELINE
                        if (j < T) {
                            data_pack = data.read();
                        }
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
                                    tmp = data_pack[jj*tf_N+ii];
                                    data_debug[j][i][jj][ii] = tmp;
                                    typename CONFIG_T::accum_t tmp2 = tmp*tmp*embed_dim_inv;
                                    //std::cout << tmp2 << " ";
                                    //sqr_debug[j][i][jj][ii] = tmp2;
                                    if (write_buffer1[jj] == true){
                                        xsum_1[jj] = xsum_1[jj] + tmp;
                                        //if (j==132){
                                        //    std::cout << "xsum_1[" << jj << "] = " << xsum_1[jj] << std::endl;
                                        //}
                                        xsqrsum_1[jj] = xsqrsum_1[jj] + tmp2;//(tmp - xsum_1[jj])*(tmp - prev_xsum_1[jj]);
                                    } else {
                                        //if (j==132){
                                        //    std::cout << "xsum_2[" << jj << "] = " << xsum_2[jj] << " " << tmp << std::endl;
                                        //}
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
                                    //std::cout << "xsum[" << jj << "] = " << xsum[jj] << std::endl;
                                    xmean[jj] = xsum[jj]*embed_dim_inv;
                                    //if (j < 6) {
                                    //    std::cout << "row_buf = " << (row_buffer[i*tf_T*tf_N + ii*tf_T + jj] - xmean[jj])*deno_inver[jj] << std::endl;
                                    //} 
                                    norm_tmp = (row_buffer[i*tf_T*tf_N + ii*tf_T + jj] - xmean[jj])*deno_inver[jj];
                                    //std::cout << "norm_tmp = " << norm_tmp << std::endl;
                                    res_pack[jj*tf_N+ii] = norm_tmp*scale[i*tf_N + ii] + bias[i*tf_N + ii];
                                    //save 133th row
                                    //set the precision to 10
                                    //std::cout << std::fixed << std::setprecision(10);
                                    //if (j == 133){
                                    //    std::cout << "norm[" << j << "] = "
                                    //    << row_buffer[i*tf_T*tf_N + ii*tf_T + jj] << " "
                                    //    << xsum[jj] << " "
                                    //    << xmean[jj] << " "
                                    //    << deno_inver[jj] << " "
                                    //    << row_buffer[i*tf_T*tf_N + ii*tf_T + jj] - xmean[jj] << " "
                                    //    << norm_tmp << " " 
                                    //    << norm_tmp*scale[i*tf_N + ii] + bias[i*tf_N + ii]
                                    //    << " " << res_pack[jj*tf_N+ii] 
                                    //    << " " << scale[i*tf_N + ii] << " " << bias[i*tf_N + ii] << std::endl;
                                    //}
                                }
                                if (j < T){
                                    row_buffer[i*tf_T*tf_N + ii*tf_T + jj] = tmp;
                                }
                            }
                            if (i == (N-1)){
                                write_buffer1[jj] = !write_buffer1[jj];
                                if (write_buffer1[jj] == false){
                                    //std::cout << "xsqrsum_1[" << jj << "] = " << xsqrsum_1[jj] << " ";
                                    xsqrsum[jj] = xsqrsum_1[jj];
                                    xsum[jj] = xsum_1[jj];
                                } else {
                                    //std::cout << "xsqrsum_2[" << jj << "] = " << xsqrsum_2[jj] << " ";
                                    xsqrsum[jj] = xsqrsum_2[jj];
                                    xsum[jj] = xsum_2[jj];
                                }
                                xmean[jj] = xsum[jj]*embed_dim_inv;
                                xsum_debug[j*tf_T + jj] = xmean[jj];
                                //std::cout << "sqrsum = " << xsqrsum[jj] << " ";
                                typename CONFIG_T::accum_t tmp3 = xsqrsum[jj]-xmean[jj]*xmean[jj];
                                //std::cout << "tmp3 = " << tmp3 << " ";
                                xvar_debug[j*tf_T + jj] = tmp3;
                                //typename CONFIG_T::mean_t tmp3 = CONFIG_T::embed_dim*xsqrsum[jj]-xsum[jj]*xsum[jj];
                                var_in_debug[j][jj] = tmp3;
                                //if (j < 6) {
                                //    std::cout << "tmp3 = " << tmp3 << std::endl;
                                //}
                                if (CONFIG_T::table_range > CONFIG_T::table_size) {
                                    index = (tmp3*CONFIG_T::table_size)/CONFIG_T::table_range;
                                } else {
                                    index = tmp3*(CONFIG_T::table_size / CONFIG_T::table_range);
                                }
                                if (index < 0)   index = 0;
                                if (index > CONFIG_T::table_size-1) index = CONFIG_T::table_size-1;
                                deno_inver[jj] = (typename CONFIG_T::var_table_t) invert_sqr_table[index];
                                //if (j < 6) {
                                //    std::cout << "deno_inver[" << jj << "] = " << deno_inver[jj] << std::endl;
                                //}
                                var_out_debug[j][jj] = deno_inver[jj];
                            }
                        }
                        if (j > 0){
                            for (int jj=0; jj < tf_T; ++jj){
                                for (int ii=0; ii < tf_N; ++ii){
                                    res_debug[j-1][i][jj][ii] = res_pack[jj*tf_N+ii];
                                }
                            }
                        }
                        if (j > 0){
                           res.write(res_pack);
                        }
                    }
                }
    
    //std::cout << "lnres_debug = " << std::endl;
    //for (int i = 0; i < T; i++) {
    //    for (int ii = 0; ii < tf_T; ii++) {
    //        for (int j = 0; j < N; j++) {
    //            for (int jj = 0; jj < tf_N; jj++) {
    //                std::cout << res_debug[i][j][ii][jj] << " ";
    //            }
    //        }
    //        std::cout << std::endl;
    //    }
    //}
    /*
    std::cout << "var_out_debug = " << std::endl;
    for (int j=0; j < T; ++j){
        for (int jj=0; jj < tf_T; ++jj){
            std::cout << var_out_debug[j][jj] << " ";
        }
    }
    std::cout << std::endl;
    */
    //std::cout << "var_out_debug = " << std::endl;
    //for (int j=0; j < T; ++j){
    //    for (int jj=0; jj < tf_T; ++jj){
    //        std::cout << var_out_debug[j][jj] << " ";
    //    }
    //}
    //std::cout << std::endl;
    //std::cout << "xsum_debug = " << std::endl;
    //for (int i = 0; i < CONFIG_T::seq_len; i++) {
    //    std::cout << xsum_debug[i] << " ";
    //}
    //std::cout << std::endl;
    //save var_out_debug to file
    


    /*
    std::ofstream var_out_debug_file;
    var_out_debug_file.open("var_out_debug.txt", std::ios_base::app);
    var_out_debug_file << std::fixed << std::setprecision(15);
    for (int j=0; j < T; ++j){
        for (int jj=0; jj < tf_T; ++jj){
            var_out_debug_file << var_out_debug[j][jj] << std::endl;
        }
    }
    //var_out_debug_file << std::endl;
    var_out_debug_file.close();
    //save var_debug
    std::ofstream xvar_debug_file;
    xvar_debug_file.open("xvar_debug.txt", std::ios_base::app);
    xvar_debug_file << std::fixed << std::setprecision(15);
    for (int i = 0; i < CONFIG_T::seq_len; i++) {
        xvar_debug_file << xvar_debug[i] << std::endl;
    }
    //xvar_debug_file << std::endl;
    xvar_debug_file.close();
    
    //save sum_debug
    
    std::ofstream xsum_debug_file;
    xsum_debug_file.open("xsum_debug.txt", std::ios_base::app);
    xsum_debug_file << std::fixed << std::setprecision(15);
    for (int i = 0; i < CONFIG_T::seq_len; i++) {
        if (i == CONFIG_T::seq_len-1){
            xsum_debug_file << xsum_debug[i];
        } else {
            xsum_debug_file << xsum_debug[i] << " ";
        }
    }
    xsum_debug_file << std::endl;
    xsum_debug_file.close();
    
    
    std::ofstream ln_res_debug_file;
    ln_res_debug_file.open("ln_res_debug.txt", std::ios_base::app);
    ln_res_debug_file << std::fixed << std::setprecision(15);
    for (int i = 0; i < T; i++) {
        for (int ii = 0; ii < tf_T; ii++) {
            for (int j = 0; j < N; j++) {
                for (int jj = 0; jj < tf_N; jj++) {
                    if (j == N-1 && jj == tf_N-1){
                        ln_res_debug_file << res_debug[i][j][ii][jj];
                    } else {
                        ln_res_debug_file << res_debug[i][j][ii][jj] << " ";
                    }
                }
            }
            ln_res_debug_file << std::endl;
        }
    }
    ln_res_debug_file << std::endl;
    ln_res_debug_file.close();
    
    std::ofstream ln_in_debug_file;
    ln_in_debug_file.open("ln_in_debug.txt", std::ios_base::app);
    ln_in_debug_file << std::fixed << std::setprecision(15);
    for (int i = 0; i < T; i++) {
        for (int ii = 0; ii < tf_T; ii++) {
            for (int j = 0; j < N; j++) {
                for (int jj = 0; jj < tf_N; jj++) {
                    if (j == N-1 && jj == tf_N-1){
                        ln_in_debug_file << data_debug[i][j][ii][jj];
                    } else {
                        ln_in_debug_file << data_debug[i][j][ii][jj] << " ";
                    }
                }
            }
            ln_in_debug_file << std::endl;
        }
    }
    ln_in_debug_file << std::endl;
    ln_in_debug_file.close();
    
    */

    //std::cout << "lnres_debug = " << std::endl;
    //for (int i = 0; i < T; i++) {
    //    for (int ii = 0; ii < tf_T; ii++) {
    //        for (int j = 0; j < N; j++) {
    //            for (int jj = 0; jj < tf_N; jj++) {
    //                std::cout << res_debug[i][j][ii][jj] << " ";
    //            }
    //        }
    //        std::cout << std::endl;
    //    }
    //}
    //save var_in_debug and var_out_debug to file
    /*
    std::ofstream var_in_debug_file;
    var_in_debug_file.open("var_in_debug.txt", std::ios_base::app);
    for (int j=0; j < T; ++j){
        for (int jj=0; jj < tf_T; ++jj){
            if (j == T-1){
                var_in_debug_file << var_in_debug[j][jj];
            } else {
                var_in_debug_file << var_in_debug[j][jj] << " ";
            }
        }
    }
    var_in_debug_file << std::endl;
    var_in_debug_file.close();
    std::ofstream var_out_debug_file;
    var_out_debug_file.open("var_out_debug.txt", std::ios_base::app);
    for (int j=0; j < T; ++j){
        for (int jj=0; jj < tf_T; ++jj){
            if (j == T-1){
                var_out_debug_file << var_out_debug[j][jj];
            } else {
                var_out_debug_file << var_out_debug[j][jj] << " ";
            }
        }
    }
    var_out_debug_file << std::endl;
    var_out_debug_file.close();

    //save sqr_debug to file
    std::ofstream sqr_debug_file;
    sqr_debug_file.open("sqr_debug.txt", std::ios_base::app);
    for (int j=0; j < T; ++j){
        for (int i=0; i < N; ++i){
            for (int jj=0; jj < tf_T; ++jj){
                for (int ii=0; ii < tf_N; ++ii){
                    if (j == T-1 && i == N-1){
                        sqr_debug_file << sqr_debug[j][i][jj][ii];
                    } else {
                        sqr_debug_file << sqr_debug[j][i][jj][ii] << " ";
                    }
                }
            }
        }
    }
    sqr_debug_file << std::endl;
    */
    /*
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
    */
    
}


}

#endif
