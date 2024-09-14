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
/*
template<class data_T, typename CONFIG_T>
void init_buffer(
    typename CONFIG_T::accum_t hidden_buffer1[CONFIG_T::tiling_factor[0]*CONFIG_T::tiling_factor[1]],
    typename CONFIG_T::accum_t hidden_buffer2[CONFIG_T::tiling_factor[0]*CONFIG_T::tiling_factor[1]],
    bool write_buffer1[CONFIG_T::tiling_factor[0]*CONFIG_T::tiling_factor[1]])
{
    for (int i=0; i < CONFIG_T::tiling_factor[0]*CONFIG_T::tiling_factor[1]; i=i+1){
        write_buffer1[i] = True;
        hidden_buffer1[i] = 0;
        hidden_buffer2[i] = 0;
    }
}

template<class data_T, typename CONFIG_T>
void store_pipo(
    typename CONFIG_T::accum_t hidden_buffer[CONFIG_T::tiling_factor[0]*CONFIG_T::tiling_factor[1]],
    typename CONFIG_T::accum_t hidden_buffer1[CONFIG_T::tiling_factor[0]*CONFIG_T::tiling_factor[1]],
    typename CONFIG_T::accum_t hidden_buffer2[CONFIG_T::tiling_factor[0]*CONFIG_T::tiling_factor[1]],
    bool write_buffer1[CONFIG_T::tiling_factor[0]*CONFIG_T::tiling_factor[1]])
{
    for (int i=0; i < CONFIG_T::tiling_factor[0]*CONFIG_T::tiling_factor[1]; i=i+1){
        if (write_buffer1[i]){
            hidden_buffer1[i] = hidden_buffer[i];
        } else {
            hidden_buffer2[i] = hidden_buffer[i];
        }
    }
}

template<class data_T, typename CONFIG_T>
void relu(
    typename CONFIG_T::accum_t hidden_buffer1[CONFIG_T::tiling_factor[0]*CONFIG_T::tiling_factor[1]],
    typename CONFIG_T::accum_t hidden_buffer2[CONFIG_T::tiling_factor[0]*CONFIG_T::tiling_factor[1]],
    typename CONFIG_T::accum_t relu_buffer[CONFIG_T::tiling_factor[0]*CONFIG_T::tiling_factor[1]],
    bool write_buffer1[CONFIG_T::tiling_factor[0]*CONFIG_T::tiling_factor[1]])
{
    for (int i=0; i < CONFIG_T::tiling_factor[0]*CONFIG_T::tiling_factor[1]; i=i+1){
        relu_buffer[i] = hidden_buffer1[i] > static_cast<typename CONFIG_T::accum_t>(0) ? hidden_buffer1[i] : static_cast<typename CONFIG_T::accum_t>(0);
        if (write_buffer1[i]){
            hidden_buffer1[i] = relu_buffer[i];
        } else {
            hidden_buffer2[i] = relu_buffer[i];
        }
    }
}
template<class data_T, class res_T, typename CONFIG_T>
void FFN_product(
    hls::stream<data_T>    &data,
    hls::stream<res_T>     &res,
    typename CONFIG_T::in_proj_weight_t     in_proj_weight[CONFIG_T::embed_dim*CONFIG_T::hidden_dim],
    typename CONFIG_T::in_proj_bias_t       in_proj_bias[CONFIG_T::hidden_dim],
    typename CONFIG_T::out_proj_weight_t    out_proj_weight[CONFIG_T::hidden_dim*CONFIG_T::embed_dim],
    typename CONFIG_T::out_proj_bias_t      out_proj_bias[CONFIG_T::embed_dim])
{
    typename data_T::value_type  input_buffer[CONFIG_T::embed_dim];  
    typename res_T::value_type   output_buffer[CONFIG_T::embed_dim];
    const unsigned tf_T = CONFIG_T::tiling_factor[0];
    const unsigned tf_N = CONFIG_T::tiling_factor[1];
    const unsigned tf_H = CONFIG_T::tiling_factor[2];
    const unsigned int T = CONFIG_T::seq_len/tf_T;
    const unsigned int N = CONFIG_T::embed_dim/tf_N;
    const unsigned int H = CONFIG_T::head_dim/tf_H;
    #pragma HLS ARRAY_RESHAPE variable=input_buffer cyclic factor=tf_N dim=1
    #pragma HLS ARRAY_RESHAPE variable=output_buffer cyclic factor=tf_N dim=1
    #pragma HLS ARRAY_PARTITION variable=in_proj_weight cyclic factor=tf_H*tf_N dim=1
    #pragma HLS ARRAY_PARTITION variable=in_proj_bias cyclic factor=tf_H dim=1
    #pragma HLS ARRAY_PARTITION variable=out_proj_weight cyclic factor=tf_H*tf_N dim=1
    #pragma HLS ARRAY_PARTITION variable=out_proj_bias cyclic factor=tf_N dim=1 
    
    data_T data_pack;
    res_T res_pack;
    static bool write_buffer1[tf_T*tf_H];
    typename CONFIG_T::accum_t hidden_buffer1[tf_H*tf_T];
    typename CONFIG_T::accum_t hidden_buffer2[tf_H*tf_T];
    typename CONFIG_T::accum_t hidden_buffer[tf_H*tf_T];
    typename CONFIG_T::accum_t relu_buffer[tf_H*tf_T];
    typename CONFIG_T::accum_t hidden_bias[tf_H];
    #pragma HLS ARRAY_PARTITION variable=hidden_buffer1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=hidden_buffer2 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=hidden_buffer complete dim=1
    #pragma HLS ARRAY_PARTITION variable=relu_buffer complete dim=1
    #pragma HLS ARRAY_PARTITION variable=hidden_bias complete dim=1
    for (int i=0; i < tf_T*tf_H; i=i+1){
        write_buffer1[i] = True;
    }
    int ipj_t_idx = 0;
    int ipj_h_idx = 0;
    int opj_t_idx = 0;
    int opj_h_idx = 0;
    int inp_blk_idx = 0;
    int ipjw_blk_idx = 0;
    int ipjb_blk_idx = 0;
    int out_blk_idx = 0;
    int opjw_blk_idx = 0;
    int opjb_blk_idx = 0;

    pipeline_product:
    for (int c = 0; c < T*H+1; c++){
        for (int n_idx = 0; n_idx < N; n_idx++){
            #pragma HLS PIPELINE II=1
            inp_blk_idx = n_idx*tf_N*tf_T;
            ipjw_blk_idx = (ipj_h_idx*N+n_idx)*tf_N*tf_H;
            ipjb_blk_idx = ipj_h_idx*tf_H;
            out_blk_idx = n_idx*tf_N*tf_T;
            opjw_blk_idx = (opj_h_idx*N+n_idx)*tf_N*tf_H;
            opjb_blk_idx = n_idx*tf_N;

            if (ipj_h_idx == 0 && c < T*H){
                data_pack = data.read();
            }
            for (int t = 0; t < tf_T; t++){
                #pragma HLS UNROLL
                for (int n = 0; n < tf_N; n++){
                    #pragma HLS UNROLL
                    if (ipj_h_idx == 0){
                        input_buffer[inp_blk_idx + t*tf_N + n] = data_pack[t*tf_N + n];
                    }
                }
            }
            if (ipj_h_idx == 0 && c < T*H) {
                init_buffer<data_T, CONFIG_T>(hidden_buffer1, hidden_buffer2, write_buffer1);
            }
            for (int t = 0; t < tf_T; t++){
                #pragma HLS UNROLL
                for (int h = 0; h < tf_H; h++){
                    #pragma HLS UNROLL
                    if (ipj_h_idx == 0 && c < T*H){
                        hidden_buffer[t*tf_H + h] = in_proj_bias[ipjb_blk_idx + h];
                    }
                    for (int n = 0; n < tf_N; n++){
                        #pragma HLS UNROLL
                        if (c < T*H){
                            hidden_buffer[t*tf_H + h] += input_buffer[inp_blk_idx + t*tf_N + n] * in_proj_weight[ipjw_blk_idx + h*tf_N + n];
                        }
                    }
                    
            if (n_idx == 0 && c < T*H) {
                hidden_bias = in_proj_bias + ipjb_blk_idx;
            } else {
                hidden_bias = 0;
            }
            block_matmulaccum<data_T, CONFIG_T>(hidden_buffer,
                                                input_buffer + inp_blk_idx, 
                                                in_proj_weight + ipjw_blk_idx,  
                                                hidden_bias,
                                                hidden_buffer,
                                                tf_T, tf_N, tf_H);
            store_pipo<data_T, CONFIG_T>(hidden_buffer, hidden_buffer1, hidden_buffer2, write_buffer1);
            relu<CONFIG_T>(hidden_buffer1, hidden_buffer2, relu_buffer, write_buffer1);
            if (opj_h_idx == 0 && c > 0){
                output_bias = out_proj_bias + opjb_blk_idx;
            } else {
                output_bias = 0;
            }
            if 
            block_matmulaccum<data_T, CONFIG_T>(out_proj_input,
                                                relu_buffer, 
                                                out_proj_weight + opjw_blk_idx,
                                                output_bias, 
                                                output_buffer + out_blk_idx,
                                                tf_T, tf_H, tf_N);
            for (int t = 0; t < tf_T; t++){
                #pragma HLS UNROLL
                for (int n = 0; n < tf_N; n++){
                    #pragma HLS UNROLL
                    if (ipj_h_idx == 0){
                        res_pack[n*tf_T + t] = output_buffer[out_blk_idx + n*tf_T + t];
                    }
                }
            }
            if (opj_h_idx ==  && c > 0){
                res.wrtie(res_pack);
            }
            if (n_idx == N-1){
                invert_write_buffer(write_buffer1);
                opj_h_idx = ipj_h_idx;
                opj_t_idx = ipj_t_idx;
                if (c < T*H){
                    ipj_h_idx = ipj_h_idx + 1;
                    if (ipj_h_idx == H){
                        ipj_h_idx = 0;
                        ipj_t_idx = ipj_t_idx + 1;
                    }
                }
            }
        }
    }
*/
template<class data_T, class res_T, typename CONFIG_T>
void FeedForwardNetwork(
    hls::stream<data_T>    &data,
    hls::stream<res_T>     &res,
    typename CONFIG_T::in_proj_weight_t     in_proj_weight[CONFIG_T::embed_dim*CONFIG_T::hidden_dim],
    typename CONFIG_T::in_proj_bias_t       in_proj_bias[CONFIG_T::hidden_dim],
    typename CONFIG_T::out_proj_weight_t    out_proj_weight[CONFIG_T::hidden_dim*CONFIG_T::embed_dim],
    typename CONFIG_T::out_proj_bias_t      out_proj_bias[CONFIG_T::embed_dim])
{
    typename data_T::value_type  row_buffer[CONFIG_T::embed_dim];  
    #pragma HLS ARRAY_PARTITION variable=row_buffer         cyclic factor=CONFIG_T::tiling_factor[1] dim=1
    #pragma HLS ARRAY_PARTITION variable=in_proj_weight     cyclic factor=CONFIG_T::tiling_factor[1]*CONFIG_T::tiling_factor[2] dim=1
    #pragma HLS ARRAY_PARTITION variable=out_proj_weight    cyclic factor=CONFIG_T::tiling_factor[1]*CONFIG_T::tiling_factor[2] dim=1    
    #pragma HLS ARRAY_PARTITION variable=in_proj_bias       cyclic factor=CONFIG_T::tiling_factor[2] dim=1
    #pragma HLS ARRAY_PARTITION variable=out_proj_bias      cyclic factor=CONFIG_T::tiling_factor[1] dim=1
    const unsigned tf_T = CONFIG_T::tiling_factor[0];
    const unsigned tf_N = CONFIG_T::tiling_factor[1];
    const unsigned tf_H = CONFIG_T::tiling_factor[2];
    const unsigned T = CONFIG_T::seq_len/CONFIG_T::tiling_factor[0];
    const unsigned N = CONFIG_T::embed_dim/CONFIG_T::tiling_factor[1];
    const unsigned H = CONFIG_T::hidden_dim/CONFIG_T::tiling_factor[2];
    
    typename CONFIG_T::hidden_t hidden_buffer_ping[H][CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[2]];
    #pragma HLS ARRAY_PARTITION variable=hidden_buffer_ping complete dim=2
    #pragma HLS ARRAY_PARTITION variable=hidden_buffer_ping complete dim=3
    typename CONFIG_T::hidden_t hidden_buffer_pong[H][CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[2]];
    #pragma HLS ARRAY_PARTITION variable=hidden_buffer_pong complete dim=2
    #pragma HLS ARRAY_PARTITION variable=hidden_buffer_pong complete dim=3
    typename CONFIG_T::accum_t hidden_buffer[CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[2]];
    #pragma HLS ARRAY_PARTITION variable=hidden_buffer complete dim=1
    #pragma HLS ARRAY_PARTITION variable=hidden_buffer complete dim=2
    typename CONFIG_T::accum_t tmp_hidden_buffer[CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[2]];
    #pragma HLS ARRAY_PARTITION variable=tmp_hidden_buffer complete dim=1
    #pragma HLS ARRAY_PARTITION variable=tmp_hidden_buffer complete dim=2
    typename CONFIG_T::accum_t tmp_hidden_buffer2[CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[2]];
    #pragma HLS ARRAY_PARTITION variable=tmp_hidden_buffer complete dim=1
    #pragma HLS ARRAY_PARTITION variable=tmp_hidden_buffer complete dim=2
    typename CONFIG_T::accum_t tmp_output_buffer[CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[1]];
    #pragma HLS ARRAY_PARTITION variable=tmp_output_buffer complete dim=1
    #pragma HLS ARRAY_PARTITION variable=tmp_output_buffer complete dim=2
    res_T res_pack;
    data_T data_pack;
    typename CONFIG_T::accum_t temp_i;
    typename CONFIG_T::accum_t temp_o;

    int ni = 0;
    int hi = 0;
    int no = 0;
    int ho = 0;
    int in_proj_weight_offset;
    int in_proj_bias_offset;
    int out_proj_weight_offset;
    int out_proj_bias_offset;
    int input_offset;
    bool write_hidden_ping[tf_T*tf_H];
    for (int i=0; i < tf_T*tf_H; i=i+1){
        write_hidden_ping[i] = true;
    }
    //ap_fixed<18,8,AP_RND_CONV> linear_debug[CONFIG_T::seq_len][CONFIG_T::hidden_dim];
    //typename CONFIG_T::accum_t data_debug[T][N][tf_T][tf_N];
    typename CONFIG_T::accum_t ress_debug[T*N*tf_T*tf_N];
    typename CONFIG_T::accum_t hidden_debug[T][H][tf_T][tf_H];
    //std::cout << " ffndata = " << std::endl;
    ////initialize row_buffer
    //for (int i=0; i < CONFIG_T::embed_dim; i=i+1){
    //    row_buffer[i] = 0;
    //}
    //std::ofstream data_debug_file;
    //data_debug_file.open("ffn_data_debug.txt", std::ios_base::app);
    //data_debug_file << std::fixed << std::setprecision(15);
    //std::cout << "ffn_res_debug.txt" << std::endl;
    //std::ofstream res_debug_file;
    //res_debug_file.open("ffn_res_debug.txt", std::ios_base::app);
    //res_debug_file << std::fixed << std::setprecision(15);
    //std::cout << "ffn_res_debug_after.txt" << std::endl;
    //std::ofstream hidden_debug_file;
    //hidden_debug_file.open("ffn_hidden_debug.txt", std::ios_base::app);
    //hidden_debug_file << std::fixed << std::setprecision(15);

    pipeline_product1n2: 
    for (int c=0; c < T+1; ++c){
        for (int k=0; k < N*H; k=k+1){
            #pragma HLS PIPELINE II=1
            in_proj_weight_offset = (ni*H + hi)*tf_H*tf_N;
            in_proj_bias_offset = hi*tf_H;
            out_proj_weight_offset = (ho*N + no)*tf_H*tf_N;
            out_proj_bias_offset = no*tf_N;
            input_offset = (c*N + ni)*tf_T*tf_N;
            if (hi==0 && c<T) {
                data_pack = data.read();
            }
            
            if (c<T) {
                for (int tt=0; tt < tf_T; ++tt){
                    #pragma HLS UNROLL
                    for (int hh=0; hh < tf_H; ++hh){
                        #pragma HLS UNROLL
                        for (int nn=0; nn < tf_N; ++nn){
                            #pragma HLS UNROLL
                            if (hi==0 && hh==0) {
                                //data_debug[c][ni][tt][nn] = data_pack[tt*tf_N + nn];
                                //if (ni==N-1) {
                                //    data_debug_file << data_pack[tt*tf_N + nn] << std::endl;
                                //} else {
                                //    data_debug_file << data_pack[tt*tf_N + nn] << " ";
                                //}
                                row_buffer[input_offset + tt*tf_N + nn] = data_pack[tt*tf_N + nn];
                
                            }
                            if (ni==0 && nn==0) {
                                tmp_hidden_buffer[tt][hh] = in_proj_bias[in_proj_bias_offset + hh];
                            } 
                            tmp_hidden_buffer[tt][hh] += row_buffer[input_offset + tt*tf_N + nn] * in_proj_weight[in_proj_weight_offset + nn*tf_H + hh];
                        }
                    }
                }
                if (ni == N-1){
                    for (int tt=0; tt < tf_T; ++tt){
                        #pragma HLS UNROLL
                        for (int hh=0; hh < tf_H; ++hh){
                            #pragma HLS UNROLL
                            //hidden_debug[c][hi][tt][hh] = tmp_hidden_buffer[tt][hh];
                            //if (hi==H-1){
                            //    hidden_debug_file << tmp_hidden_buffer[tt][hh] << std::endl;
                            //} else {
                            //    hidden_debug_file << tmp_hidden_buffer[tt][hh] << " ";
                            //}
                            tmp_hidden_buffer[tt][hh] = (tmp_hidden_buffer[tt][hh] > 0) ? tmp_hidden_buffer[tt][hh] : static_cast<typename CONFIG_T::accum_t>(0);
                            
                            if (write_hidden_ping[tt*tf_H + hh]){
                                hidden_buffer_ping[hi][tt][hh] = tmp_hidden_buffer[tt][hh];
                                
                            } else {
                                hidden_buffer_pong[hi][tt][hh] = tmp_hidden_buffer[tt][hh];
                                
                            }
                        }
                    }
                
                }
            }
            
            if (c>0) {
                for (int tt=0; tt < tf_T; ++tt){
                    #pragma HLS UNROLL
                    for (int nn=0; nn < tf_N; ++nn){
                        #pragma HLS UNROLL
                        for (int hh=0; hh < tf_H; ++hh){
                            #pragma HLS UNROLL
                            if (ho==0 && hh==0) {
                                tmp_output_buffer[tt][nn] = out_proj_bias[out_proj_bias_offset + nn];
                            }
                            if (write_hidden_ping[tt*tf_H + hh]){
                                tmp_hidden_buffer2[tt][hh] = hidden_buffer_pong[ho][tt][hh];
                            } else {
                                tmp_hidden_buffer2[tt][hh] = hidden_buffer_ping[ho][tt][hh];
                            }
                            tmp_output_buffer[tt][nn] += tmp_hidden_buffer2[tt][hh] * out_proj_weight[out_proj_weight_offset + hh*tf_N + nn];
                        }
                    }
                }
                if (ho == H-1){
                    for (int tt=0; tt < tf_T; ++tt){
                        #pragma HLS UNROLL
                        for (int nn=0; nn < tf_N; ++nn){
                            #pragma HLS UNROLL
                            //std::cout << static_cast<typename res_T::value_type>(tmp_output_buffer[tt][nn]) << "(" << tmp_output_buffer[tt][nn] << ") ";
                            //if (no == N-1) {
                            //    std::cout << std::endl;
                            //}
                            //ress_debug[(c-1)*N*tf_T*tf_N + no*tf_T*tf_N + tt*tf_N + nn] = tmp_output_buffer[tt][nn];
                            //if (no == N-1 && tt == 0 && nn == 0) {     
                            //    res_debug_file << static_cast<typename res_T::value_type>(tmp_output_buffer[tt][nn]);
                            //} else {
                            //    //std::cout << res_debug[i][j][ii][jj] << " ";
                            //    res_debug_file << static_cast<typename res_T::value_type>(tmp_output_buffer[tt][nn]) << " ";
                            //}
                            //std::cout << "tmp_output_buffer[tt][nn]: " << tmp_output_buffer[tt][nn] << std::endl;
                            //if (no == N-1 && tt == 0 && nn == 0) {
                            //    res_debug_file << static_cast<typename res_T::value_type>(tmp_output_buffer[tt][nn]) << std::endl;
                            //} else {
                            //    res_debug_file << static_cast<typename res_T::value_type>(tmp_output_buffer[tt][nn]) << " ";
                            //}
                            res_pack[tt*tf_N + nn] = tmp_output_buffer[tt][nn];
                        }
                    }
                    res.write(res_pack);
                }
            }
            
            if (k == N*H-1){
                for (int tt=0; tt < tf_T; ++tt){
                    #pragma HLS UNROLL
                    for (int hh=0; hh < tf_H; ++hh){
                        #pragma HLS UNROLL
                        write_hidden_ping[tt*tf_H + hh] = !write_hidden_ping[tt*tf_H + hh];
                    }
                }
            }
            if (c < T) {
                ni = ni + 1;
                if (ni == N){
                    ni = 0;
                    hi = hi + 1;
                    if (hi == H){
                        hi = 0;
                    }
                }
            }
            if (c > 0) {
                ho = ho + 1;
                if (ho == H){
                    ho = 0;
                    no = no + 1;
                    if (no == N){
                        no = 0;
                    }
                }
            }
            
        }
        //res_debug_file << std::endl;
    }
    //print data debug
    //std::cout << "data_debug = "<< std::endl;
    //for (int i=0; i < T; i=i+1){
    //    for (int ii=0; ii < tf_T; ii=ii+1){
    //        for (int j=0; j < N; j=j+1){
    //            for (int jj=0; jj < tf_N; jj=jj+1){
    //                std::cout << data_debug[i][j][ii][jj] << " ";
    //            }
    //        }
    //        std::cout << std::endl;
    //    }
    //}
    //std::cout << "ffnhidden_debug"<< std::endl;
    //for (int i=0; i < T; i=i+1){
    //    for (int ii=0; ii < tf_T; ii=ii+1){
    //        for (int j=0; j < H; j=j+1){
    //            for (int jj=0; jj < tf_H; jj=jj+1){
    //                std::cout << hidden_debug[i][j][ii][jj] << " ";
    //            }
    //        }
    //        std::cout << std::endl;
    //    }
    //}
    //std::cout << "ffnres_debug"<< std::endl;
    //for (int i=0; i < T; i=i+1){
    //    for (int ii=0; ii < tf_T; ii=ii+1){
    //        for (int j=0; j < N; j=j+1){
    //            for (int jj=0; jj < tf_N; jj=jj+1){
    //                std::cout << res_debug[i][j][ii][jj] << " ";
    //            }
    //        }
    //        std::cout << std::endl;
    //    }
    //}

    //save res_debug
    //std::ofstream res_debug_file;
    //res_debug_file.open("ffn_res_debug.txt", std::ios_base::app);
    //res_debug_file << std::fixed << std::setprecision(15);
    //for (int i=0; i < T; i=i+1){
    //    for (int ii=0; ii < tf_T; ii=ii+1){
    //        for (int j=0; j < N; j=j+1){
    //            for (int jj=0; jj < tf_N; jj=jj+1){
    //                if (j == N-1 && jj == tf_N-1) {
    //                    //std::cout << res_debug[i][j][ii][jj] << std::endl;
    //                    res_debug_file << ress_debug[i*N*tf_T*tf_N + j*tf_T*tf_N + ii*tf_N + jj];
    //                } else {
    //                    //std::cout << res_debug[i][j][ii][jj] << " ";
    //                    res_debug_file << ress_debug[i*N*tf_T*tf_N + j*tf_T*tf_N + ii*tf_N + jj] << " ";
    //                }
    //            }
    //        }
    //        res_debug_file << std::endl;
    //    }
    //}
    //res_debug_file.close();

    //save hidden_debug
    //std::ofstream hidden_debug_file;
    //hidden_debug_file.open("ffn_hidden_debug.txt", std::ios_base::app);
    //hidden_debug_file << std::fixed << std::setprecision(15);
    //for (int i=0; i < T; i=i+1){
    //    for (int ii=0; ii < tf_T; ii=ii+1){
    //        for (int j=0; j < H; j=j+1){
    //            for (int jj=0; jj < tf_H; jj=jj+1){
    //                if (j == H-1 && jj == tf_H-1) {
    //                    hidden_debug_file << hidden_debug[i][j][ii][jj];
    //                } else {
    //                    hidden_debug_file << hidden_debug[i][j][ii][jj] << " ";
    //                }
    //            }
    //        }
    //        hidden_debug_file << std::endl;
    //    }
    //}
    //hidden_debug_file.close();
    //data_debug_file.close();

    


    //print linear debug
    //std::cout << "linear_debug"<< std::endl;
    //for (int i=0; i < CONFIG_T::seq_len; i=i+1){
    //    for (int j=0; j < CONFIG_T::hidden_dim; j=j+1){
    //        std::cout << linear_debug[i][j] << " ";
    //    }
    //    std::cout << std::endl;
    //}
    ////print first row of out_proj weight
    //std::cout << "out_proj_weight[0][0] = "<< std::endl;
    //for (int i=0; i < H; i=i+1){
    //    for (int ii=0; ii < tf_H; ii=ii+1){
    //        std::cout << out_proj_weight[i][0][ii][0] << " ";
    //    }
    //}
    //std::cout << std::endl;
    //res_T res_pack;
    //write_output:
    //for (int i=0; i <CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]; i=i+1){
    //    for (int j=0; j < CONFIG_T::embed_dim/CONFIG_T::tiling_factor[1]; j=j+1){
    //        #pragma HLS PIPELINE II=1
    //        for (int ii=0; ii < CONFIG_T::tiling_factor[0]; ++ii){
    //            #pragma HLS UNROLL
    //            for (int jj=0; jj < CONFIG_T::tiling_factor[1]; ++jj){
    //                #pragma HLS UNROLL
    //                res_pack[ii*CONFIG_T::tiling_factor[1]+jj] = output_buffer[i][j][ii][jj];
    //                if (jj == CONFIG_T::tiling_factor[1]-1 && ii == CONFIG_T::tiling_factor[0]-1) {
    //                    res.write(res_pack);
    //                }
    //            }
    //        }
    //    }
    //}
    
}

template<class data_T, class res_T, typename CONFIG_T>
void FFN_in_out_product(
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
    //#pragma HLS BIND_STORAGE variable=input_buffer type=ram_s2p impl=uram
    //#pragma HLS BIND_STORAGE variable=output_buffer type=ram_s2p impl=uram

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
                    if (ii == 0 && jj == 0) {
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
    typename CONFIG_T::accum_t linear_debug[CONFIG_T::seq_len][CONFIG_T::hidden_dim];
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
                            dense2_out[ii][kk] = in_proj_bias[k][kk];
                            linear_debug[m*CONFIG_T::tiling_factor[0]+ii][n*CONFIG_T::tiling_factor[2]+kk] = dense1_out[ii][kk];
                            if (dense1_out[ii][kk] < 0) {
                                dense1_out[ii][kk] = 0;
                            }
                        } else {
                            dense1_out[ii][kk] = in_proj_bias[k][kk];
                            linear_debug[m*CONFIG_T::tiling_factor[0]+ii][n*CONFIG_T::tiling_factor[2]+kk] = dense2_out[ii][kk];
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
                                //std::cout << "c=" << c << " i=" << i << " k=" << k << " p=" << p << " " <<dense2_out[ii][kk] << std::endl;
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

    //print linear debug
    //for (int i=0; i < CONFIG_T::seq_len; i=i+1){
    //    for (int j=0; j < CONFIG_T::hidden_dim; j=j+1){
    //        std::cout << linear_debug[i][j] << " ";
    //    }
    //    std::cout << std::endl;
    //}

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

//template<class data_T, class res_T, typename CONFIG_T>
//void FeedForwardNetwork(
//    hls::stream<data_T>    &data,
//    hls::stream<res_T>     &res,
//    typename CONFIG_T::in_proj_weight_t     in_proj_weight[CONFIG_T::embed_dim/CONFIG_T::tiling_factor[1]][CONFIG_T::hidden_dim/CONFIG_T::tiling_factor[2]][CONFIG_T::tiling_factor[1]][CONFIG_T::tiling_factor[2]],
//    typename CONFIG_T::in_proj_bias_t       in_proj_bias[CONFIG_T::hidden_dim/CONFIG_T::tiling_factor[2]][CONFIG_T::tiling_factor[2]],
//    typename CONFIG_T::out_proj_weight_t    out_proj_weight[CONFIG_T::hidden_dim/CONFIG_T::tiling_factor[2]][CONFIG_T::embed_dim/CONFIG_T::tiling_factor[1]][CONFIG_T::tiling_factor[2]][CONFIG_T::tiling_factor[1]],
//    typename CONFIG_T::out_proj_bias_t      out_proj_bias[CONFIG_T::embed_dim/CONFIG_T::tiling_factor[1]][CONFIG_T::tiling_factor[1]])
//{
//    assert(CONFIG_T::seq_len % CONFIG_T::tiling_factor[0] == 0);
//    assert(CONFIG_T::embed_dim % CONFIG_T::tiling_factor[1] == 0);
//    assert(CONFIG_T::hidden_dim % CONFIG_T::tiling_factor[2] == 0);
//    const unsigned T = CONFIG_T::seq_len/CONFIG_T::tiling_factor[0];
//    const unsigned N = CONFIG_T::embed_dim/CONFIG_T::tiling_factor[1];
//    const unsigned H = CONFIG_T::hidden_dim/CONFIG_T::tiling_factor[2];
//    if (CONFIG_T::seq_len*CONFIG_T::embed_dim >= CONFIG_T::hidden_dim*CONFIG_T::tiling_factor[0]) {
//        FFN_out_in_product<data_T, res_T, CONFIG_T>(data, res, in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias);
//    } else {
//        FFN_in_out_product<data_T, res_T, CONFIG_T>(data, res, in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias);
//    }
//}

}


#endif
