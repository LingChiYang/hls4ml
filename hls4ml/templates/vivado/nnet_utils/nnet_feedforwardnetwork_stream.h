#ifndef NNET_FFN_H_
#define NNET_FFN_H_

#include "nnet_common.h"
#include "nnet_mult.h"
#include "nnet_helpers.h"
#include "hls_stream.h"
#include "hls_streamofblocks.h"
#include <math.h>
#include <cmath>
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

template <typename CONFIG_T> 
void init_cdf_table(typename CONFIG_T::cdf_table_t table_out[CONFIG_T::cdf_table_size])
{
    for (int ii = 0; ii < CONFIG_T::cdf_table_size; ii++) {
        // 将表索引转换为X值(范围 -4 到 +4)
        double in_val = 2 * float(CONFIG_T::cdf_table_range) * (ii - float(CONFIG_T::cdf_table_size) / 2.0) / float(CONFIG_T::cdf_table_size);
        // 计算GELU函数值
        double cdf = 0.5 * (1.0 + erf(in_val / sqrt(2)));
        typename CONFIG_T::cdf_table_t real_val = cdf;
        table_out[ii] = real_val;
    }
}

template<typename CONFIG_T>
typename CONFIG_T::cdf_table_t lookup_cdf(
    typename CONFIG_T::accum_t data)
{
    #ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::cdf_table_t cdf_table[CONFIG_T::cdf_table_size];
    #else
    static bool initialized = false;
    static typename CONFIG_T::cdf_table_t cdf_table[CONFIG_T::cdf_table_size];
    #endif

    if (!initialized) {
        init_cdf_table<CONFIG_T>(cdf_table);
        initialized = true;
    }

    int data_round = int(data*(CONFIG_T::cdf_table_size/(CONFIG_T::cdf_table_range*2)));
    int index = data_round + CONFIG_T::cdf_table_range*(CONFIG_T::cdf_table_size/(CONFIG_T::cdf_table_range*2));
    
    if (index < 0)   index = 0;
    if (index > CONFIG_T::cdf_table_size-1) index = CONFIG_T::cdf_table_size-1;
    return cdf_table[index];
}

template<class data_T, class res_T, typename CONFIG_T>
void FeedForwardNetwork(
    hls::stream<data_T>    &data,
    hls::stream<res_T>     &res,
    typename CONFIG_T::in_proj_weight_t     in_proj_weight[CONFIG_T::embed_dim*CONFIG_T::hidden_dim],
    typename CONFIG_T::in_proj_bias_t       in_proj_bias[CONFIG_T::hidden_dim],
    typename CONFIG_T::out_proj_weight_t    out_proj_weight[CONFIG_T::hidden_dim*CONFIG_T::embed_dim],
    typename CONFIG_T::out_proj_bias_t      out_proj_bias[CONFIG_T::embed_dim])
{
    #pragma HLS DATAFLOW
    #pragma HLS ARRAY_RESHAPE variable=in_proj_weight     cyclic factor=CONFIG_T::tiling_factor[1]*CONFIG_T::tiling_factor[2] dim=1
    #pragma HLS ARRAY_RESHAPE variable=out_proj_weight    cyclic factor=CONFIG_T::tiling_factor[1]*CONFIG_T::tiling_factor[2] dim=1    
    #pragma HLS ARRAY_RESHAPE variable=in_proj_bias       cyclic factor=CONFIG_T::tiling_factor[2] dim=1
    #pragma HLS ARRAY_RESHAPE variable=out_proj_bias      cyclic factor=CONFIG_T::tiling_factor[1] dim=1
    const unsigned tf_T = CONFIG_T::tiling_factor[0];
    const unsigned tf_N = CONFIG_T::tiling_factor[1];
    const unsigned tf_H = CONFIG_T::tiling_factor[2];
    const unsigned T = CONFIG_T::seq_len/CONFIG_T::tiling_factor[0];
    const unsigned N = CONFIG_T::embed_dim/CONFIG_T::tiling_factor[1];
    const unsigned H = CONFIG_T::hidden_dim/CONFIG_T::tiling_factor[2];
    
    using linear_in_buf = typename data_T::value_type[N*tf_N*tf_T];
    using linear_hidden_buf = typename CONFIG_T::hidden_t[H*tf_H*tf_T];
    using linear_out_buf = typename res_T::value_type[N*tf_N*tf_T];
    hls::stream_of_blocks<linear_in_buf, 2> linear_in;
    #pragma HLS ARRAY_RESHAPE variable=linear_in cyclic factor=tf_N*tf_T dim=1
    hls::stream_of_blocks<linear_hidden_buf, 2> linear_hidden;
    #pragma HLS ARRAY_RESHAPE variable=linear_hidden cyclic factor=tf_H*tf_T dim=1
    hls::stream_of_blocks<linear_out_buf, 2> linear_out;
    #pragma HLS ARRAY_RESHAPE variable=linear_out cyclic factor=tf_N*tf_T dim=1
    typename CONFIG_T::accum_t linear_tmp[tf_T*tf_H];
    #pragma HLS ARRAY_PARTITION variable=linear_tmp complete dim=1
    typename CONFIG_T::accum_t linear_tmp2[tf_T*tf_N];
    #pragma HLS ARRAY_PARTITION variable=linear_tmp2 complete dim=1

    fifo_stream_of_block_convert: 
    for(int t=0; t<T; t++){
        hls::write_lock<linear_in_buf> arrA(linear_in);
        for(int n=0; n<N; n++){
            data_T data_pack = data.read();
            for(int tt=0; tt<tf_T; tt++){
                #pragma HLS UNROLL
                for(int nn=0; nn<tf_N; nn++){
                    #pragma HLS UNROLL
                    arrA[n*tf_N*tf_T+tt*tf_N+nn] = data_pack[tt*tf_N+nn];
                }
            }
        }
    }

    linear_sb_in:
    for(int t=0; t<T; t++){
        hls::read_lock<linear_in_buf> arrA(linear_in);
        hls::write_lock<linear_hidden_buf> arrB(linear_hidden);
        for(int h=0; h<H; h++){
            for(int n=0; n<N; n++){
                #pragma HLS PIPELINE II=1
                for(int tt=0; tt<tf_T; tt++){
                    #pragma HLS UNROLL
                    for(int hh=0; hh<tf_H; hh++){
                        #pragma HLS UNROLL
                        for(int nn=0; nn<tf_N; nn++){
                            #pragma HLS UNROLL
                            if(n==0 && nn==0){
                                linear_tmp[tt*tf_H+hh] = in_proj_bias[h*tf_H+hh];
                            } 
                            linear_tmp[tt*tf_H+hh] += arrA[n*tf_N*tf_T+tt*tf_N+nn] * in_proj_weight[n*H*tf_H*tf_N + h*tf_H*tf_N + nn*tf_H + hh];
                        }
                    }
                }
                if(n==N-1){
                    for(int tt=0; tt<tf_T; tt++){
                        #pragma HLS UNROLL
                        for(int hh=0; hh<tf_H; hh++){
                            #pragma HLS UNROLL
                            if (CONFIG_T::activation_gelu){
                                arrB[h*tf_H+tt*tf_H+hh] = linear_tmp[tt*tf_H+hh] * lookup_cdf<CONFIG_T>(linear_tmp[tt*tf_H+hh]);
                            } else {
                                arrB[h*tf_H+tt*tf_H+hh] = (linear_tmp[tt*tf_H+hh] < 0) ? static_cast<typename CONFIG_T::accum_t>(0) : linear_tmp[tt*tf_H+hh];
                            }
                        }
                    }
                    // 儲存linear_tmp至linear.txt
                    std::ofstream outfile;
                    outfile.open("linear.txt", std::ios_base::app);
                    for(int tt=0; tt<tf_T; tt++){
                        for(int hh=0; hh<tf_H; hh++){
                            outfile << linear_tmp[tt*tf_H+hh] << " ";
                        }
                        outfile << "\n";
                    }
                    outfile.close();
                }
            }
        }
    }

    linear_sb_out:
    for(int t=0; t<T; t++){
        hls::read_lock<linear_hidden_buf> arrB(linear_hidden);
        hls::write_lock<linear_out_buf> arrC(linear_out);
        for(int n=0; n<N; n++){
            for(int h=0; h<H; h++){
                #pragma HLS PIPELINE II=1
                for(int tt=0; tt<tf_T; tt++){
                    #pragma HLS UNROLL
                    for(int nn=0; nn<tf_N; nn++){
                        #pragma HLS UNROLL
                        for(int hh=0; hh<tf_H; hh++){
                            #pragma HLS UNROLL
                            if(h==0 && hh==0){
                                linear_tmp2[tt*tf_N+nn] = out_proj_bias[n*tf_N+nn];
                            } 
                            linear_tmp2[tt*tf_N+nn] += arrB[h*tf_H*tf_T+tt*tf_H+hh] * out_proj_weight[h*N*tf_N*tf_H + n*tf_N*tf_H + nn*tf_H + hh];
                        }
                    }
                }
                if (h==H-1){
                    for(int tt=0; tt<tf_T; tt++){
                        #pragma HLS UNROLL
                        for(int nn=0; nn<tf_N; nn++){
                            #pragma HLS UNROLL
                            arrC[n*tf_N*tf_T+tt*tf_N+nn] = linear_tmp2[tt*tf_N+nn];
                        }
                    }
                }
            }
        }
    }
    res_T res_pack;
    stream_of_block_fifo_convert:
    for(int t=0; t<T; t++){
        hls::read_lock<linear_out_buf> arrC(linear_out);
        for(int n=0; n<N; n++){
            #pragma HLS PIPELINE II=1
            for(int tt=0; tt<tf_T; tt++){
                #pragma HLS UNROLL
                for(int nn=0; nn<tf_N; nn++){
                    #pragma HLS UNROLL
                    res_pack[tt*tf_N+nn] = arrC[n*tf_N*tf_T+tt*tf_N+nn];
                }
            }
            res.write(res_pack);
        }
    }
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
