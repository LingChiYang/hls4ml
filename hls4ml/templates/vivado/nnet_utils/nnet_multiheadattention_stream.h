#ifndef NNET_MHT_SS_H_
#define NNET_MHT_SS_H_

#include "nnet_common.h"
#include "nnet_mult.h"
#include "hls_stream.h"
#include <iostream>
#include <math.h>
#include "nnet_helpers.h"
//#include "nnet_activation.h"

namespace nnet {

struct mha_config {
    static const unsigned n_head = 1;
    static const unsigned head_dim = 100;
    static const unsigned feature_dim = 100;
    static const unsigned seq_len = 100;
    static const unsigned qkv_ram_style = nnet::block;
    static const unsigned attn_ram_style = nnet::block;
    static const unsigned out_ram_style = nnet::block;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
};


template <typename CONFIG_T> 
void init_exp_table(typename CONFIG_T::table_t table_out[CONFIG_T::table_size])
{
    for (int ii = 0; ii < CONFIG_T::table_size; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range -8 to +8)
        float in_val = 2 * float(CONFIG_T::exp_range) * (ii - float(CONFIG_T::table_size) / 2.0) / float(CONFIG_T::table_size);
        // Next, compute lookup table function
        typename CONFIG_T::table_t real_val = std::exp(in_val);
        // std::cout << "Lookup table In Value: " << in_val << " Result: " << real_val << std::endl;
        table_out[ii] = real_val;
    }
}

template <typename CONFIG_T>
void init_invert_table(typename CONFIG_T::table_t table_out[CONFIG_T::table_size])
{
    // Inversion function:
    //   result = 1/x
    for (int ii = 0; ii < CONFIG_T::table_size; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range 0 to +64)
        float in_val = float(CONFIG_T::inv_range) * ii / float(CONFIG_T::table_size);
        // Next, compute lookup table function
        if (in_val > 0.0)
            table_out[ii] = 1.0 / in_val;
        else
            table_out[ii] = 0.0;
    }
}
template <typename CONFIG_T, int N_TABLE> void init_exp_table_legacy(typename CONFIG_T::table_t table_out[N_TABLE]) {
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range -8 to +8)
        float in_val = 2 * 8.0 * (ii - float(N_TABLE) / 2.0) / float(N_TABLE);
        // Next, compute lookup table function
        typename CONFIG_T::table_t real_val = std::exp(in_val);
        // std::cout << "Lookup table In Value: " << in_val << " Result: " << real_val << std::endl;
        table_out[ii] = real_val;
    }
}

template <typename CONFIG_T, int N_TABLE> void init_invert_table_legacy(typename CONFIG_T::table_t table_out[N_TABLE]) {
    // Inversion function:
    //   result = 1/x
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range 0 to +64)
        float in_val = 64.0 * ii / float(N_TABLE);
        // Next, compute lookup table function
        if (in_val > 0.0)
            table_out[ii] = 1.0 / in_val;
        else
            table_out[ii] = 0.0;
    }
}

template<class data_T, typename CONFIG_T>
typename CONFIG_T::exp_table_t lookup_exp(
    data_T data)
{
    #ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::exp_table_t exp_table[CONFIG_T::table_size];
    #else
    static bool initialized = false;
    static typename CONFIG_T::exp_table_t exp_table[CONFIG_T::table_size];
    #endif

    if (!initialized) {
        //init_exp_table_legacy<CONFIG_T, CONFIG_T::table_size>(exp_table);
        init_exp_table<CONFIG_T>(exp_table);
        initialized = true;
    }

    int data_round = int(data*(CONFIG_T::table_size/(CONFIG_T::exp_range*2)));
    int index = data_round + CONFIG_T::exp_range*(CONFIG_T::table_size/(CONFIG_T::exp_range*2));
    //print index
    if (index < 0)   index = 0;
    if (index > CONFIG_T::table_size-1) index = CONFIG_T::table_size-1;
    return exp_table[index];
}

template<class data_T, typename CONFIG_T>
typename CONFIG_T::inv_table_t lookup_inv(
    data_T data)
{
    #ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::inv_table_t inv_table[CONFIG_T::table_size];
    #else
    static bool initialized = false;
    static typename CONFIG_T::inv_table_t inv_table[CONFIG_T::table_size];
    #endif

    if (!initialized) {
        //init_invert_table_legacy<CONFIG_T, CONFIG_T::table_size>(inv_table);
        init_invert_table<CONFIG_T>(inv_table);
        initialized = true;
    }

    int index = int(data*(CONFIG_T::table_size/CONFIG_T::inv_range));
    if (index < 0)   index = 0;
    if (index > CONFIG_T::table_size-1) index = CONFIG_T::table_size-1;
    return inv_table[index];
}


//update row_sum and row_max
template<typename CONFIG_T>
void Update_RowMaxSum(
    typename CONFIG_T::accum_t new_row_sum[CONFIG_T::n_head][CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::accum_t prev_row_sum[CONFIG_T::n_head][CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::accum_t new_row_max[CONFIG_T::n_head][CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::accum_t prev_row_max[CONFIG_T::n_head][CONFIG_T::tiling_factor[0]])
{
    #pragma HLS ARRAY_PARTITION variable=new_row_sum complete dim=0
    #pragma HLS ARRAY_PARTITION variable=prev_row_sum complete dim=0
    #pragma HLS ARRAY_PARTITION variable=new_row_max complete dim=0
    #pragma HLS ARRAY_PARTITION variable=prev_row_max complete dim=0
    for (int i = 0; i < CONFIG_T::n_head; i++) {
        #pragma HLS UNROLL
        for (int ii = 0; ii < CONFIG_T::tiling_factor[0]; ii++) {
            #pragma HLS UNROLL
            prev_row_max[i][ii] = new_row_max[i][ii];
            prev_row_sum[i][ii] = new_row_sum[i][ii];
        }
    }
}


template<class data_T, class res_T, typename CONFIG_T>
void MultiHeadAttention(
    hls::stream<data_T>    &data_qkv,
    hls::stream<res_T>     &res,
    typename CONFIG_T::in_proj_weight_t  in_proj_weight[3][CONFIG_T::n_head][CONFIG_T::embed_dim/CONFIG_T::tiling_factor[1]][CONFIG_T::head_dim/CONFIG_T::tiling_factor[2]][CONFIG_T::tiling_factor[1]][CONFIG_T::tiling_factor[2]],
    typename CONFIG_T::in_proj_bias_t    in_proj_bias[3][CONFIG_T::n_head][CONFIG_T::head_dim/CONFIG_T::tiling_factor[2]][CONFIG_T::tiling_factor[2]],
    typename CONFIG_T::out_proj_weight_t out_proj_weight[CONFIG_T::n_head][CONFIG_T::head_dim/CONFIG_T::tiling_factor[2]][CONFIG_T::embed_dim/CONFIG_T::tiling_factor[1]][CONFIG_T::tiling_factor[2]][CONFIG_T::tiling_factor[1]],  // n_head,head_size_v,dim
    typename CONFIG_T::out_proj_bias_t   out_proj_bias[CONFIG_T::embed_dim/CONFIG_T::tiling_factor[1]][CONFIG_T::tiling_factor[1]])
{
    const unsigned int tf_T = CONFIG_T::tiling_factor[0];
    const unsigned int tf_N = CONFIG_T::tiling_factor[1];
    const unsigned int tf_H = CONFIG_T::tiling_factor[2];
    const unsigned int T = CONFIG_T::seq_len/tf_T;
    const unsigned int N = CONFIG_T::embed_dim/tf_N;
    const unsigned int H = CONFIG_T::head_dim/tf_H;
    //data_T data_q_buf[T][N][tf_T][CONFIG_T::tiling_factor[1]];
    typename data_T::value_type data_qkv_buf[T][N][tf_T][CONFIG_T::tiling_factor[1]];
    typename data_T::value_type K[CONFIG_T::n_head][T][H][tf_T][tf_H];
    typename data_T::value_type V[CONFIG_T::n_head][T][H][tf_T][tf_H];
    typename data_T::value_type Q[CONFIG_T::n_head][T][H][tf_T][tf_H];
    typename data_T::value_type O[CONFIG_T::n_head][T][H][tf_T][tf_H];
    typename data_T::value_type M[T][N][tf_T][CONFIG_T::tiling_factor[1]];
    //#pragma HLS STREAM off variable=data_q_buf depth=2 
    //#pragma HLS STREAM off variable=data_vk_buf depth=2 
    //#pragma HLS STREAM off variable=K depth=2
    //#pragma HLS STREAM off variable=V depth=2
    //#pragma HLS STREAM off variable=Q depth=2
    //#pragma HLS STREAM off variable=O depth=2
    //#pragma HLS STREAM off variable=M depth=2

    #pragma HLS ARRAY_PARTITION variable=data_qkv_buf complete dim=3
    #pragma HLS ARRAY_PARTITION variable=data_qkv_buf complete dim=4
    //#pragma HLS ARRAY_PARTITION variable=data_vk_buf complete dim=3
    //#pragma HLS ARRAY_PARTITION variable=data_vk_buf complete dim=4
    #pragma HLS ARRAY_PARTITION variable=K complete dim=1
    #pragma HLS ARRAY_PARTITION variable=K complete dim=4
    #pragma HLS ARRAY_PARTITION variable=K complete dim=5
    #pragma HLS ARRAY_PARTITION variable=V complete dim=1
    #pragma HLS ARRAY_PARTITION variable=V complete dim=4
    #pragma HLS ARRAY_PARTITION variable=V complete dim=5
    #pragma HLS ARRAY_PARTITION variable=Q complete dim=1
    #pragma HLS ARRAY_PARTITION variable=Q complete dim=4
    #pragma HLS ARRAY_PARTITION variable=Q complete dim=5
    #pragma HLS ARRAY_PARTITION variable=O complete dim=1
    #pragma HLS ARRAY_PARTITION variable=O complete dim=4
    #pragma HLS ARRAY_PARTITION variable=O complete dim=5
    #pragma HLS ARRAY_PARTITION variable=M complete dim=3
    #pragma HLS ARRAY_PARTITION variable=M complete dim=4
    #pragma HLS DATAFLOW
    typename data_T::value_type QK0[CONFIG_T::n_head][tf_T][tf_T];
    typename data_T::value_type QK1[CONFIG_T::n_head][tf_T][tf_T];
    typename CONFIG_T::exp_table_t P[CONFIG_T::n_head][tf_T][tf_T];
    #pragma HLS ARRAY_PARTITION variable=QK0 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=QK1 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=P complete dim=0
    //#pragma HLS RESOURCE variable=key_weight core=XPM_MEMORY uram
    //#pragma HLS RESOURCE variable=value_weight core=XPM_MEMORY uram
    //#pragma HLS RESOURCE variable=query_weight core=XPM_MEMORY uram
    //#pragma HLS RESOURCE variable=attention_output_weight core=XPM_MEMORY uram

    #pragma HLS ARRAY_PARTITION variable=key_weight complete dim=1
    #pragma HLS ARRAY_PARTITION variable=key_weight complete dim=4
    #pragma HLS ARRAY_PARTITION variable=key_weight complete dim=5
    #pragma HLS ARRAY_PARTITION variable=value_weight complete dim=1
    #pragma HLS ARRAY_PARTITION variable=value_weight complete dim=4
    #pragma HLS ARRAY_PARTITION variable=value_weight complete dim=5
    #pragma HLS ARRAY_PARTITION variable=query_weight complete dim=1
    #pragma HLS ARRAY_PARTITION variable=query_weight complete dim=4
    #pragma HLS ARRAY_PARTITION variable=query_weight complete dim=5
    #pragma HLS ARRAY_PARTITION variable=attention_output_weight complete dim=1
    #pragma HLS ARRAY_PARTITION variable=attention_output_weight complete dim=4
    #pragma HLS ARRAY_PARTITION variable=attention_output_weight complete dim=5
    #pragma HLS ARRAY_PARTITION variable=key_bias complete dim=1
    #pragma HLS ARRAY_PARTITION variable=key_bias complete dim=3
    #pragma HLS ARRAY_PARTITION variable=value_bias complete dim=1
    #pragma HLS ARRAY_PARTITION variable=value_bias complete dim=3
    #pragma HLS ARRAY_PARTITION variable=query_bias complete dim=1
    #pragma HLS ARRAY_PARTITION variable=query_bias complete dim=3
    #pragma HLS ARRAY_PARTITION variable=attention_output_bias complete dim=2

    data_T data_pack;
    store_data: 
    for (int i = 0; i < T; i++) {
        for (int j = 0; j < N; j++) {
            for (int ii = 0; ii < tf_T; ii++) {
                for (int jj = 0; jj < CONFIG_T::tiling_factor[1]; jj++) {
                    #pragma HLS PIPELINE II = 1
                    if (jj==0 && ii==0){
                        data_pack = data_qkv.read();
                    }
                    data_qkv_buf[i][j][ii][jj] = data_pack[ii*CONFIG_T::tiling_factor[1] + jj];
                    //data_vk_buf[i][j][ii][jj] = data_vk[j*CONFIG_T::tiling_factor[1] + jj].read();
                    //data_q_buf[i][j][ii][jj].write(data_q[j*CONFIG_T::tiling_factor[1] + jj].read());
                    //data_vk_buf[i][j][ii][jj].write(data_vk[j*CONFIG_T::tiling_factor[1] + jj].read());
                }
            }
        }
    }

    typename CONFIG_T::accum_t rowsum[CONFIG_T::n_head][tf_T];
    typename CONFIG_T::accum_t new_rowsum[CONFIG_T::n_head][tf_T];
    typename CONFIG_T::accum_t new_rowmax[CONFIG_T::n_head][tf_T];
    typename CONFIG_T::accum_t prev_rowmax[CONFIG_T::n_head][tf_T];
    typename CONFIG_T::accum_t prev_rowsum[CONFIG_T::n_head][tf_T];
    typename CONFIG_T::accum_t rowmax[CONFIG_T::n_head][tf_T];
    
    #pragma HLS ARRAY_PARTITION variable=rowsum complete dim=0
    #pragma HLS ARRAY_PARTITION variable=new_rowsum complete dim=0
    #pragma HLS ARRAY_PARTITION variable=new_rowmax complete dim=0
    #pragma HLS ARRAY_PARTITION variable=prev_rowmax complete dim=0
    #pragma HLS ARRAY_PARTITION variable=prev_rowsum complete dim=0
    #pragma HLS ARRAY_PARTITION variable=rowmax complete dim=0
    const typename data_T::value_type dk = 1.0/sqrt(CONFIG_T::head_dim);
    int index;
    int data_round; 
    typename CONFIG_T::accum_t tmp_k[CONFIG_T::n_head][tf_H];
    typename CONFIG_T::accum_t tmp_v[CONFIG_T::n_head][tf_H];
    typename CONFIG_T::accum_t tmp_q[CONFIG_T::n_head][tf_H];
    #pragma HLS ARRAY_PARTITION variable=tmp_k complete dim=0
    #pragma HLS ARRAY_PARTITION variable=tmp_v complete dim=0
    #pragma HLS ARRAY_PARTITION variable=tmp_q complete dim=0
    compute_KVQ:  
    for (int i = 0; i < T; i++) {
        for (int k = 0; k < H; k++) {
            for (int j = 0; j < N; j++) {
                #pragma HLS PIPELINE II = 1
                for (int h = 0; h < CONFIG_T::n_head; h++) {//48dsp
                    #pragma HLS UNROLL
                    for (int ii = 0; ii < tf_T; ii++) {
                        #pragma HLS UNROLL
                        for (int kk = 0; kk < tf_H; kk++) {
                            #pragma HLS UNROLL
                            if (j==0){
                                tmp_k[h][kk] = in_proj_bias[1][h][k][kk];
                                tmp_v[h][kk] = in_proj_bias[2][h][k][kk];
                                tmp_q[h][kk] = in_proj_bias[0][h][k][kk];
                            }
                            for (int jj = 0; jj < CONFIG_T::tiling_factor[1]; jj++) {
                                #pragma HLS UNROLL
                                tmp_k[h][kk] += data_qkv_buf[i][j][ii][jj]*in_proj_weight[1][h][j][k][jj][kk],
                                tmp_v[h][kk] += data_qkv_buf[i][j][ii][jj]*in_proj_weight[2][h][j][k][jj][kk],
                                tmp_q[h][kk] += data_qkv_buf[i][j][ii][jj]*in_proj_weight[0][h][j][k][jj][kk];
                            }
                            K[h][i][k][ii][kk] = tmp_k[h][kk];
                            V[h][i][k][ii][kk] = tmp_v[h][kk];
                            Q[h][i][k][ii][kk] = tmp_q[h][kk];
                        }
                    }
                }
            }           
        }
    }

    //print Q
    //for (int h = 0; h < CONFIG_T::n_head; h++) {
    //    for (int i = 0; i < T; i++) {
    //        for (int ii = 0; ii < tf_T; ii++) {
    //            for (int j = 0; j < H; j++) {
    //                for (int jj = 0; jj < tf_H; jj++) {
    //                    std::cout << Q[h][i][j][ii][jj] << " ";
    //                }
    //            }
    //            std::cout << std::endl;
    //        }
    //    }
    //    std::cout << "change head" << std::endl;
    //}
    

    typename CONFIG_T::exp_table_t prev_exp_tmp[CONFIG_T::n_head][tf_T];
    typename CONFIG_T::exp_table_t exp_tmp[CONFIG_T::n_head][tf_T];
    typename CONFIG_T::inv_table_t inv_row_sum[CONFIG_T::n_head][tf_T];
    #pragma HLS ARRAY_PARTITION variable=inv_row_sum complete dim=0
    #pragma HLS ARRAY_PARTITION variable=prev_exp_tmp complete dim=0
    #pragma HLS ARRAY_PARTITION variable=exp_tmp complete dim=0
    int i = 0;
    int j = 0;
    int m = 0;
    int n = 0;
    typename CONFIG_T::accum_t tmp_o;
    int total_cycle = ((CONFIG_T::seq_len*CONFIG_T::seq_len)/(tf_T*tf_T)) + 1;
    bool write_QK0[CONFIG_T::n_head][tf_T][tf_T];
    #pragma HLS ARRAY_PARTITION variable=write_QK0 complete dim=0
    for (int i = 0; i < CONFIG_T::n_head; i++) {
        for (int ii = 0; ii < tf_T; ii++) {
            for (int jj = 0; jj < tf_T; jj++) {
                write_QK0[i][ii][jj] = true;
            }
        }
    } 
    compute_att_pipeline:   
    for (int c=0; c <= T*T; ++c){
        for (int k=0; k<H; k++) {
            #pragma HLS PIPELINE II = 1
            if (k==0) {
                for (int h = 0; h < CONFIG_T::n_head; h++) {//8dsp
                    #pragma HLS UNROLL
                    for (int ii = 0; ii < tf_T; ii++) {
                        #pragma HLS UNROLL
                        for (int ii2 = 0; ii2 < tf_T; ii2++) {
                            #pragma HLS UNROLL
                            if (write_QK0[h][ii][ii2]){
                                if (c < T*T) { 
                                    QK0[h][ii][ii2] = 0;
                                }
                                if (c > 0) { 
                                    if (ii2 == 0){
                                        rowmax[h][ii] = QK1[h][ii][ii2];
                                        rowsum[h][ii] = 0;
                                    } else {
                                        rowmax[h][ii] = rowmax[h][ii] > QK1[h][ii][ii2] ? rowmax[h][ii] : QK1[h][ii][ii2];
                                    }
                                    P[h][ii][ii2] = lookup_exp<typename CONFIG_T::accum_t, CONFIG_T>(QK1[h][ii][ii2] - rowmax[h][ii]);
                                    rowsum[h][ii] += P[h][ii][ii2]; 
                                }
                            } else {
                                if (c < T*T) { 
                                    QK1[h][ii][ii2] = 0;
                                }
                                if (c > 0) { 
                                    if (ii2 == 0){
                                        rowmax[h][ii] = QK0[h][ii][ii2];
                                        rowsum[h][ii] = 0;
                                    } else {
                                        rowmax[h][ii] = rowmax[h][ii] > QK0[h][ii][ii2] ? rowmax[h][ii] : QK0[h][ii][ii2];
                                    }
                                    P[h][ii][ii2] = lookup_exp<typename CONFIG_T::accum_t, CONFIG_T>(QK0[h][ii][ii2] - rowmax[h][ii]);
                                    rowsum[h][ii] += P[h][ii][ii2]; 
                                }
                            }   
                        }
                        if (j==0){
                            prev_rowmax[h][ii] = -128;
                            prev_rowsum[h][ii] = 0;
                            new_rowmax[h][ii] = rowmax[h][ii];
                        } else {
                            new_rowmax[h][ii] = prev_rowmax[h][ii] > rowmax[h][ii] ? prev_rowmax[h][ii] : rowmax[h][ii];
                        }
                        prev_exp_tmp[h][ii] = lookup_exp<typename CONFIG_T::accum_t, CONFIG_T>(prev_rowmax[h][ii] - new_rowmax[h][ii]);
                        exp_tmp[h][ii] = lookup_exp<typename CONFIG_T::accum_t, CONFIG_T>(rowmax[h][ii] - new_rowmax[h][ii]);
                        new_rowsum[h][ii] = prev_exp_tmp[h][ii]*prev_rowsum[h][ii] + rowsum[h][ii];
                    }
                }
            }
            for (int h = 0; h < CONFIG_T::n_head; h++) { //16dsp
                #pragma HLS UNROLL
                for (int ii = 0; ii < tf_T; ii++) {
                    #pragma HLS UNROLL
                    for (int ii2 = 0; ii2 < tf_T; ii2++) {
                        #pragma HLS UNROLL
                        for (int kk = 0; kk < tf_H; kk++) {
                            #pragma HLS UNROLL
                            typename CONFIG_T::accum_t tmp = 0;
                            tmp = Q[h][m][k][ii][kk]*K[h][n][k][ii2][kk] * dk;
                            if (write_QK0[h][ii][ii2]){
                                QK0[h][ii][ii2] += tmp;
                            } else {
                                QK1[h][ii][ii2] += tmp;
                            }
                        }
                        if (k==H-1) {
                            write_QK0[h][ii][ii2] = !write_QK0[h][ii][ii2];
                        }
                    }
                }
            }
            for (int h = 0; h < CONFIG_T::n_head; h++) {//48 dsp
                #pragma HLS UNROLL
                for (int ii = 0; ii < tf_T; ii++) {
                    #pragma HLS UNROLL
                    for (int kk = 0; kk < tf_H; kk++) {
                        #pragma HLS UNROLL
                        typename CONFIG_T::accum_t tmp = 0;
                        typename CONFIG_T::accum_t tmp2 = 0;
                        for (int ii2 = 0; ii2 < tf_T; ii2++) {
                            #pragma HLS UNROLL
                           tmp += P[h][ii][ii2]*V[h][j][k][ii2][kk];
                        }
                        //initialize O
                        if (j==0){
                            tmp_o = 0;
                        } else {
                            tmp_o = O[h][i][k][ii][kk];
                        }
                        tmp2 = prev_exp_tmp[h][ii]*tmp_o + tmp;
                        if (c > 0)  {
                            if (j==T-1){
                                inv_row_sum[h][ii] = lookup_inv<typename CONFIG_T::accum_t, CONFIG_T>(new_rowsum[h][ii]);
                                O[h][i][k][ii][kk] = tmp2*inv_row_sum[h][ii];
                            } else {
                                O[h][i][k][ii][kk] = tmp2;
                            }
                        }
                    }
                }
            }
            Update_RowMaxSum<CONFIG_T>(new_rowsum, prev_rowsum, new_rowmax, prev_rowmax);
            if (k==H-1) {
                if (c > 0) { //cycle 0~total_cycle-1
                    j++;
                    if (j == T){
                        j = 0;
                        i++;
                    }
                }
                if (c < T*T) { //cycle 1~total_cycle
                    n++;
                    if (n == T){
                        n = 0;
                        m++;
                    }
                }
            }
        }   
    }
    //print O
    for (int h = 0; h < CONFIG_T::n_head; h++) {
        for (int i = 0; i < T; i++) {
            for (int ii = 0; ii < tf_T; ii++) {
                for (int j = 0; j < H; j++) {
                    for (int jj = 0; jj < tf_H; jj++) {
                        std::cout << O[h][i][j][ii][jj] << " ";
                    }
                }
                std::cout << std::endl;
            }
        }
        std::cout << "change head" << std::endl;
    }
    typename CONFIG_T::accum_t tmp_m;                        
    compute_output: 
    for (int i = 0; i < T; i++) {
        for (int k = 0; k < H; k++) {
            for (int j = 0; j < N; j++) {
                #pragma HLS PIPELINE II = 1
                for (int ii = 0; ii < tf_T; ii++) {
                    #pragma HLS UNROLL
                    for (int jj = 0; jj < CONFIG_T::tiling_factor[1]; jj++) {
                        #pragma HLS UNROLL
                        if (k==0){
                            tmp_m = out_proj_bias[j][jj];
                        } else {
                            tmp_m = M[i][j][ii][jj];
                        }
                        for (int kk = 0; kk < tf_H; kk++) {
                            #pragma HLS UNROLL
                            for (int h = 0; h < CONFIG_T::n_head; h++) {//16dsp
                                #pragma HLS UNROLL
                                tmp_m += O[h][i][k][ii][kk] * out_proj_weight[h][k][j][kk][jj];
                            }
                        }
                        M[i][j][ii][jj] = tmp_m;
                    }
                }
            }
        }
    }

    //print M
    //for (int i = 0; i < T; i++) {
    //    for (int ii = 0; ii < tf_T; ii++) {
    //        for (int j = 0; j < N; j++) {
    //            for (int jj = 0; jj < CONFIG_T::tiling_factor[1]; jj++) {
    //                std::cout << M[i][j][ii][jj] << " ";
    //            }
    //        }
    //        std::cout << std::endl;
    //    }
    //}

    res_T res_pack;
    
    write_output:
    for (int i = 0; i < T; i++) {
        for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II = 1
            for (int ii = 0; ii < tf_T; ii++) {
                #pragma HLS UNROLL
                for (int jj = 0; jj < CONFIG_T::tiling_factor[1]; jj++) {
                    #pragma HLS UNROLL
                    res_pack[ii*CONFIG_T::tiling_factor[1] + jj] = M[i][j][ii][jj];
                    if (jj==CONFIG_T::tiling_factor[1]-1 && ii==tf_T-1){
                        res.write(res_pack);
                    }
                }
            }
        }
    }
    
}


template<typename CONFIG_T>
void Init_RowMaxSum(
    typename CONFIG_T::accum_t row_sum[CONFIG_T::n_head][CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::accum_t new_row_sum[CONFIG_T::n_head][CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::accum_t prev_row_sum[CONFIG_T::n_head][CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::accum_t row_max[CONFIG_T::n_head][CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::accum_t new_row_max[CONFIG_T::n_head][CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::accum_t prev_row_max[CONFIG_T::n_head][CONFIG_T::tiling_factor[0]])
{
    for (int i = 0; i < CONFIG_T::n_head; i++) {
        #pragma HLS UNROLL
        for (int ii = 0; ii < CONFIG_T::tiling_factor[0]; ii++) {
            #pragma HLS UNROLL
            //set prev_row_max to negative infinity
            prev_row_max[i][ii] = -128;
            prev_row_sum[i][ii] = 0;
        }
    }
}

//initialize QK
template<typename CONFIG_T>
void Init_Attention(
    typename CONFIG_T::accum_t QK[CONFIG_T::n_head][CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[0]])
{
    for (int i = 0; i < CONFIG_T::n_head; i++) {
        #pragma HLS UNROLL
        for (int ii = 0; ii < CONFIG_T::tiling_factor[0]; ii++) {
            #pragma HLS UNROLL
            for (int ll = 0; ll < CONFIG_T::tiling_factor[0]; ll++) {
                #pragma HLS UNROLL
                QK[i][ii][ll] = 0;
            }
        }   
    }
}
//compute current tile row_max and row_sum
template<typename CONFIG_T>
void Compute_CurrentTile_RowMaxSum(
    typename CONFIG_T::accum_t QK[CONFIG_T::n_head][CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::accum_t row_max[CONFIG_T::n_head][CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::accum_t row_sum[CONFIG_T::n_head][CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::exp_table_t P[CONFIG_T::n_head][CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[0]])
{
    for (int i = 0; i < CONFIG_T::n_head; i++) {
        #pragma HLS UNROLL
        for (int ii = 0; ii < CONFIG_T::tiling_factor[0]; ii++) {
            #pragma HLS UNROLL
            for (int ll = 0; ll < CONFIG_T::tiling_factor[0]; ll++) {
                #pragma HLS UNROLL
                if (ll == 0){
                    row_max[i][ii] = QK[i][ii][ll];
                } else {
                    row_max[i][ii] = row_max[i][ii] > QK[i][ii][ll] ? row_max[i][ii] : QK[i][ii][ll];
                }
            }
            row_sum[i][ii] = 0;
            for (int ll = 0; ll < CONFIG_T::tiling_factor[0]; ll++) {
                #pragma HLS UNROLL
                P[i][ii][ll] = lookup_exp<typename CONFIG_T::accum_t, CONFIG_T>(QK[i][ii][ll] - row_max[i][ii]);
                row_sum[i][ii] += P[i][ii][ll];
            }
        }
    }
}


//update row_sum and row_max
/*
template<typename CONFIG_T>
void Update_RowMaxSum(
    typename CONFIG_T::accum_t new_row_sum[CONFIG_T::n_head][CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::accum_t prev_row_sum[CONFIG_T::n_head][CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::accum_t new_row_max[CONFIG_T::n_head][CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::accum_t prev_row_max[CONFIG_T::n_head][CONFIG_T::tiling_factor[0]])
{
    #pragma HLS ARRAY_PARTITION variable=new_row_sum complete dim=0
    #pragma HLS ARRAY_PARTITION variable=prev_row_sum complete dim=0
    #pragma HLS ARRAY_PARTITION variable=new_row_max complete dim=0
    #pragma HLS ARRAY_PARTITION variable=prev_row_max complete dim=0
    for (int i = 0; i < CONFIG_T::n_head; i++) {
        #pragma HLS UNROLL
        for (int ii = 0; ii < CONFIG_T::tiling_factor[0]; ii++) {
            #pragma HLS UNROLL
            prev_row_max[i][ii] = new_row_max[i][ii];
            prev_row_sum[i][ii] = new_row_sum[i][ii];
        }
    }
}


template<class data_T, class res_T, typename CONFIG_T>
void MultiHeadAttention(
    hls::stream<data_T>    &data_qkv,
    hls::stream<res_T>     &res,
    typename CONFIG_T::in_proj_weight_t  in_proj_weight[3][CONFIG_T::n_head][CONFIG_T::embed_dim/CONFIG_T::tiling_factor[1]][CONFIG_T::head_dim/CONFIG_T::tiling_factor[2]][CONFIG_T::tiling_factor[1]][CONFIG_T::tiling_factor[2]],
    typename CONFIG_T::in_proj_bias_t    in_proj_bias[3][CONFIG_T::n_head][CONFIG_T::head_dim/CONFIG_T::tiling_factor[2]][CONFIG_T::tiling_factor[2]],
    typename CONFIG_T::out_proj_weight_t out_proj_weight[CONFIG_T::n_head][CONFIG_T::head_dim/CONFIG_T::tiling_factor[2]][CONFIG_T::embed_dim/CONFIG_T::tiling_factor[1]][CONFIG_T::tiling_factor[2]][CONFIG_T::tiling_factor[1]],  // n_head,head_size_v,dim
    typename CONFIG_T::out_proj_bias_t   out_proj_bias[CONFIG_T::embed_dim/CONFIG_T::tiling_factor[1]][CONFIG_T::tiling_factor[1]])
{
    const unsigned int tf_T = CONFIG_T::tiling_factor[0];
    const unsigned int tf_N = CONFIG_T::tiling_factor[1];
    const unsigned int tf_H = CONFIG_T::tiling_factor[2];
    const unsigned int T = CONFIG_T::seq_len/tf_T;
    const unsigned int N = CONFIG_T::embed_dim/tf_N;
    const unsigned int H = CONFIG_T::head_dim/tf_H;
    typename data_T::value_type data_qkv_buf[CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]][CONFIG_T::embed_dim/CONFIG_T::tiling_factor[1]][CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[1]];
    typename data_T::value_type K[CONFIG_T::n_head][CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]][CONFIG_T::head_dim/CONFIG_T::tiling_factor[2]][CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[2]];
    typename data_T::value_type V[CONFIG_T::n_head][CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]][CONFIG_T::head_dim/CONFIG_T::tiling_factor[2]][CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[2]];
    typename data_T::value_type Q[CONFIG_T::n_head][CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]][CONFIG_T::head_dim/CONFIG_T::tiling_factor[2]][CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[2]];
    typename data_T::value_type O[CONFIG_T::n_head][CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]][CONFIG_T::head_dim/CONFIG_T::tiling_factor[2]][CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[2]];
    typename data_T::value_type M[CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]][CONFIG_T::embed_dim/CONFIG_T::tiling_factor[1]][CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[1]];
    //#pragma HLS STREAM off variable=data_q_buf depth=2 
    //#pragma HLS STREAM off variable=data_vk_buf depth=2 
    //#pragma HLS STREAM off variable=K depth=2
    //#pragma HLS STREAM off variable=V depth=2
    //#pragma HLS STREAM off variable=Q depth=2
    //#pragma HLS STREAM off variable=O depth=2
    //#pragma HLS STREAM off variable=M depth=2

    #pragma HLS ARRAY_PARTITION variable=data_q_buf complete dim=3
    #pragma HLS ARRAY_PARTITION variable=data_q_buf complete dim=4
    #pragma HLS ARRAY_PARTITION variable=data_vk_buf complete dim=3
    #pragma HLS ARRAY_PARTITION variable=data_vk_buf complete dim=4
    #pragma HLS ARRAY_PARTITION variable=K complete dim=1
    #pragma HLS ARRAY_PARTITION variable=K complete dim=4
    #pragma HLS ARRAY_PARTITION variable=K complete dim=5
    #pragma HLS ARRAY_PARTITION variable=V complete dim=1
    #pragma HLS ARRAY_PARTITION variable=V complete dim=4
    #pragma HLS ARRAY_PARTITION variable=V complete dim=5
    #pragma HLS ARRAY_PARTITION variable=Q complete dim=1
    #pragma HLS ARRAY_PARTITION variable=Q complete dim=4
    #pragma HLS ARRAY_PARTITION variable=Q complete dim=5
    #pragma HLS ARRAY_PARTITION variable=O complete dim=1
    #pragma HLS ARRAY_PARTITION variable=O complete dim=4
    #pragma HLS ARRAY_PARTITION variable=O complete dim=5
    #pragma HLS ARRAY_PARTITION variable=M complete dim=3
    #pragma HLS ARRAY_PARTITION variable=M complete dim=4
    #pragma HLS DATAFLOW
    typename data_T::value_type QK0[CONFIG_T::n_head][CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[0]];
    typename data_T::value_type QK1[CONFIG_T::n_head][CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[0]];
    typename CONFIG_T::exp_table_t P[CONFIG_T::n_head][CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[0]];
    #pragma HLS ARRAY_PARTITION variable=QK0 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=QK1 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=P complete dim=0
    //#pragma HLS RESOURCE variable=key_weight core=XPM_MEMORY uram
    //#pragma HLS RESOURCE variable=value_weight core=XPM_MEMORY uram
    //#pragma HLS RESOURCE variable=query_weight core=XPM_MEMORY uram
    //#pragma HLS RESOURCE variable=attention_output_weight core=XPM_MEMORY uram

    #pragma HLS ARRAY_PARTITION variable=key_weight complete dim=1
    #pragma HLS ARRAY_PARTITION variable=key_weight complete dim=4
    #pragma HLS ARRAY_PARTITION variable=key_weight complete dim=5
    #pragma HLS ARRAY_PARTITION variable=value_weight complete dim=1
    #pragma HLS ARRAY_PARTITION variable=value_weight complete dim=4
    #pragma HLS ARRAY_PARTITION variable=value_weight complete dim=5
    #pragma HLS ARRAY_PARTITION variable=query_weight complete dim=1
    #pragma HLS ARRAY_PARTITION variable=query_weight complete dim=4
    #pragma HLS ARRAY_PARTITION variable=query_weight complete dim=5
    #pragma HLS ARRAY_PARTITION variable=attention_output_weight complete dim=1
    #pragma HLS ARRAY_PARTITION variable=attention_output_weight complete dim=4
    #pragma HLS ARRAY_PARTITION variable=attention_output_weight complete dim=5
    #pragma HLS ARRAY_PARTITION variable=key_bias complete dim=1
    #pragma HLS ARRAY_PARTITION variable=key_bias complete dim=3
    #pragma HLS ARRAY_PARTITION variable=value_bias complete dim=1
    #pragma HLS ARRAY_PARTITION variable=value_bias complete dim=3
    #pragma HLS ARRAY_PARTITION variable=query_bias complete dim=1
    #pragma HLS ARRAY_PARTITION variable=query_bias complete dim=3
    #pragma HLS ARRAY_PARTITION variable=attention_output_bias complete dim=2


    data_T data_pack;
    store_data: 
    for (int i = 0; i < T; i++) {
        for (int j = 0; j < N; j++) {
            for (int ii = 0; ii < tf_T; ii++) {
                for (int jj = 0; jj < CONFIG_T::tiling_factor[1]; jj++) {
                    #pragma HLS PIPELINE II = 1
                    if (jj==0 && ii==0){
                        data_pack = data_qkv.read();
                    }
                    data_qkv_buf[i][j][ii][jj] = data_pack[ii*CONFIG_T::tiling_factor[1] + jj];
                    //data_vk_buf[i][j][ii][jj] = data_vk[j*CONFIG_T::tiling_factor[1] + jj].read();
                    //data_q_buf[i][j][ii][jj].write(data_q[j*CONFIG_T::tiling_factor[1] + jj].read());
                    //data_vk_buf[i][j][ii][jj].write(data_vk[j*CONFIG_T::tiling_factor[1] + jj].read());
                }
            }
        }
    }

    typename CONFIG_T::accum_t rowsum[CONFIG_T::n_head][CONFIG_T::tiling_factor[0]];
    typename CONFIG_T::accum_t new_rowsum[CONFIG_T::n_head][CONFIG_T::tiling_factor[0]];
    typename CONFIG_T::accum_t new_rowmax[CONFIG_T::n_head][CONFIG_T::tiling_factor[0]];
    typename CONFIG_T::accum_t prev_rowmax[CONFIG_T::n_head][CONFIG_T::tiling_factor[0]];
    typename CONFIG_T::accum_t prev_rowsum[CONFIG_T::n_head][CONFIG_T::tiling_factor[0]];
    typename CONFIG_T::accum_t rowmax[CONFIG_T::n_head][CONFIG_T::tiling_factor[0]];
    
    #pragma HLS ARRAY_PARTITION variable=rowsum complete dim=0
    #pragma HLS ARRAY_PARTITION variable=new_rowsum complete dim=0
    #pragma HLS ARRAY_PARTITION variable=new_rowmax complete dim=0
    #pragma HLS ARRAY_PARTITION variable=prev_rowmax complete dim=0
    #pragma HLS ARRAY_PARTITION variable=prev_rowsum complete dim=0
    #pragma HLS ARRAY_PARTITION variable=rowmax complete dim=0
    typename data_T::value_type dk = 1.0/sqrt(CONFIG_T::head_dim);
    int index;
    int data_round; 
    typename CONFIG_T::accum_t tmp_k[CONFIG_T::n_head][CONFIG_T::tiling_factor[2]];
    typename CONFIG_T::accum_t tmp_v[CONFIG_T::n_head][CONFIG_T::tiling_factor[2]];
    typename CONFIG_T::accum_t tmp_q[CONFIG_T::n_head][CONFIG_T::tiling_factor[2]];
    #pragma HLS ARRAY_PARTITION variable=tmp_k complete dim=0
    #pragma HLS ARRAY_PARTITION variable=tmp_v complete dim=0
    #pragma HLS ARRAY_PARTITION variable=tmp_q complete dim=0
    compute_KVQ:  
    for (int i = 0; i < T; i++) {
        for (int k = 0; k < H; k++) {
            for (int j = 0; j < N; j++) {
                #pragma HLS PIPELINE II = 1
                for (int h = 0; h < CONFIG_T::n_head; h++) {//48dsp
                    #pragma HLS UNROLL
                    for (int ii = 0; ii < tf_T; ii++) {
                        #pragma HLS UNROLL
                        for (int kk = 0; kk < tf_H; kk++) {
                            #pragma HLS UNROLL
                            if (j==0){
                                tmp_k[h][kk] = in_proj_bias[1][h][k][kk];
                                tmp_v[h][kk] = in_proj_bias[2][h][k][kk];
                                tmp_q[h][kk] = in_proj_bias[0][h][k][kk];
                            }
                            for (int jj = 0; jj < CONFIG_T::tiling_factor[1]; jj++) {
                                #pragma HLS UNROLL
                                tmp_k[h][kk] += data_qkv_buf[i][j][ii][jj]*in_proj_weight[1][h][j][k][jj][kk],
                                tmp_v[h][kk] += data_qkv_buf[i][j][ii][jj]*in_proj_weight[2][h][j][k][jj][kk],
                                tmp_q[h][kk] += data_qkv_buf[i][j][ii][jj]*in_proj_weight[0][h][j][k][jj][kk];
                            }
                            K[h][i][k][ii][kk] = tmp_k[h][kk];
                            V[h][i][k][ii][kk] = tmp_v[h][kk];
                            Q[h][i][k][ii][kk] = tmp_q[h][kk];
                        }
                    }
                }
            }           
        }
    }
    
    //print Q K V
    //std::cout << "Q" << std::endl;
    //for (int h = 0; h < CONFIG_T::n_head; h++) {
    //    for (int i = 0; i < CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]; i++) {
    //        for (int ii = 0; ii < CONFIG_T::tiling_factor[0]; ii++) {
    //            for (int j = 0; j < CONFIG_T::head_dim/CONFIG_T::tiling_factor[2]; j++) {
    //                for (int jj = 0; jj < CONFIG_T::tiling_factor[2]; jj++) {
    //                    std::cout << Q[h][i][j][ii][jj] << " ";
    //                }
    //            }
    //            std::cout << std::endl;
    //        }
    //    }
    //    std::cout << "change Q head" << std::endl;
    //}
    ////std::cout << "K" << std::endl;
    //for (int h = 0; h < CONFIG_T::n_head; h++) {
    //    for (int i = 0; i < CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]; i++) {
    //        for (int ii = 0; ii < CONFIG_T::tiling_factor[0]; ii++) {
    //            for (int j = 0; j < CONFIG_T::head_dim/CONFIG_T::tiling_factor[2]; j++) {
    //                for (int jj = 0; jj < CONFIG_T::tiling_factor[2]; jj++) {
    //                    std::cout << K[h][i][j][ii][jj] << " ";
    //                }
    //            }
    //            std::cout << std::endl;
    //        }
    //    }
    //    std::cout << "change K head" << std::endl;
    //}
    //std::cout << "V" << std::endl;
    //for (int h = 0; h < CONFIG_T::n_head; h++) {
    //    for (int i = 0; i < CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]; i++) {
    //        for (int ii = 0; ii < CONFIG_T::tiling_factor[0]; ii++) {
    //            for (int j = 0; j < CONFIG_T::head_dim/CONFIG_T::tiling_factor[2]; j++) {
    //                for (int jj = 0; jj < CONFIG_T::tiling_factor[2]; jj++) {
    //                    std::cout << V[h][i][j][ii][jj] << " ";
    //                }
    //            }
    //            std::cout << std::endl;
    //        }
    //    }
    //    std::cout << "change head" << std::endl;
    //}


    typename CONFIG_T::exp_table_t prev_exp_tmp[CONFIG_T::n_head][CONFIG_T::tiling_factor[0]];
    typename CONFIG_T::exp_table_t exp_tmp[CONFIG_T::n_head][CONFIG_T::tiling_factor[0]];
    typename CONFIG_T::inv_table_t inv_row_sum[CONFIG_T::n_head][CONFIG_T::tiling_factor[0]];
    #pragma HLS ARRAY_PARTITION variable=inv_row_sum complete dim=0
    #pragma HLS ARRAY_PARTITION variable=prev_exp_tmp complete dim=0
    #pragma HLS ARRAY_PARTITION variable=exp_tmp complete dim=0
    int i = 0;
    int j = 0;
    int m = 0;
    int n = 0;
    typename CONFIG_T::accum_t QK_debug[CONFIG_T::n_head][CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]][CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[0]];
    typename CONFIG_T::accum_t tmp_o;
    int total_cycle = ((CONFIG_T::seq_len*CONFIG_T::seq_len)/(CONFIG_T::tiling_factor[0]*CONFIG_T::tiling_factor[0])) + 1;
    bool init_QK0 = false; 
    compute_att_pipeline:   
    for (int c=0; c < total_cycle; ++c){
        for (int k=0; k<CONFIG_T::head_dim/CONFIG_T::tiling_factor[2]; k++) {
            #pragma HLS PIPELINE II = 1
            if (k==0) {
                if (init_QK0){
                    if (c < total_cycle-1) { Init_Attention<CONFIG_T>(QK0); }
                    if (c > 0) { Compute_CurrentTile_RowMaxSum<CONFIG_T>(QK1, rowmax, rowsum, P); }
                } else {
                    if (c < total_cycle-1) { Init_Attention<CONFIG_T>(QK1); }
                    if (c > 0) { Compute_CurrentTile_RowMaxSum<CONFIG_T>(QK0, rowmax, rowsum, P); }
                }
                for (int h = 0; h < CONFIG_T::n_head; h++) {//8dsp
                    #pragma HLS UNROLL
                    for (int ii = 0; ii < CONFIG_T::tiling_factor[0]; ii++) {
                        #pragma HLS UNROLL
                        if (j==0){
                            prev_rowmax[h][ii] = -8;
                            prev_rowsum[h][ii] = 0;
                            new_rowmax[h][ii] = rowmax[h][ii];
                        } else {
                            new_rowmax[h][ii] = prev_rowmax[h][ii] > rowmax[h][ii] ? prev_rowmax[h][ii] : rowmax[h][ii];
                        }
                        //std::cout << "prev_rowmax rowmax new_rowmax" << prev_rowmax[h][ii] << " " << rowmax[h][ii] << " " << new_rowmax[h][ii] << std::endl;
                        prev_exp_tmp[h][ii] = lookup_exp<typename CONFIG_T::accum_t, CONFIG_T>(prev_rowmax[h][ii] - new_rowmax[h][ii]);
                        exp_tmp[h][ii] = lookup_exp<typename CONFIG_T::accum_t, CONFIG_T>(rowmax[h][ii] - new_rowmax[h][ii]);
                        //std::cout << "prev_exp_tmp exp_tmp" << prev_exp_tmp[h][ii] << " " << exp_tmp[h][ii] << std::endl;
                        new_rowsum[h][ii] = prev_exp_tmp[h][ii]*prev_rowsum[h][ii] + rowsum[h][ii];
                        //std::cout << "i j m n k h ii: " << i << " " << j << " " << m << " " << n << " " << k << " " << h << " " << ii << std::endl;
                        //std::cout << "new_rowsum prev_rowsum rowsum: " << new_rowsum[h][ii] << " " << prev_rowsum[h][ii] << " " << rowsum[h][ii] << std::endl;
                    }
                }
            }
            for (int h = 0; h < CONFIG_T::n_head; h++) { //16dsp
                #pragma HLS UNROLL
                for (int mm = 0; mm < CONFIG_T::tiling_factor[0]; mm++) {
                    #pragma HLS UNROLL
                    for (int nn = 0; nn < CONFIG_T::tiling_factor[0]; nn++) {
                        #pragma HLS UNROLL
                        for (int kk = 0; kk < CONFIG_T::tiling_factor[2]; kk++) {
                            #pragma HLS UNROLL
                            typename CONFIG_T::accum_t tmp = 0;
                            tmp = Q[h][m][k][mm][kk]*K[h][n][k][nn][kk] * dk;
                            std::cout << "h m n Q K " << h << " " << m << " " << n << " " << Q[h][m][k][mm][kk] << " " << K[h][n][k][nn][kk] << std::endl; 
                            if (init_QK0){
                                QK0[h][mm][nn] += tmp;
                                std::cout << "QK0: " << QK0[h][mm][nn] << std::endl;
                            } else {
                                QK1[h][mm][nn] += tmp;
                                std::cout << "QK1: " << QK1[h][mm][nn] << std::endl;
                            }
                        }
                        if (init_QK0){
                            QK_debug[h][m][n][mm][nn] = QK0[h][mm][nn];
                        } else {
                            QK_debug[h][m][n][mm][nn] = QK1[h][mm][nn];
                        }
                    }
                }
            }
            for (int h = 0; h < CONFIG_T::n_head; h++) {//48 dsp
                #pragma HLS UNROLL
                for (int ii = 0; ii < CONFIG_T::tiling_factor[0]; ii++) {
                    #pragma HLS UNROLL
                    for (int kk = 0; kk < CONFIG_T::tiling_factor[2]; kk++) {
                        #pragma HLS UNROLL
                        typename CONFIG_T::accum_t tmp = 0;
                        typename CONFIG_T::accum_t tmp2 = 0;
                        for (int ii2 = 0; ii2 < CONFIG_T::tiling_factor[0]; ii2++) {
                            #pragma HLS UNROLL
                           tmp += P[h][ii][ii2]*V[h][j][k][ii2][kk];
                        }
                        //initialize O
                        if (j==0){
                            tmp_o = 0;
                        } else {
                            tmp_o = O[h][i][k][ii][kk];
                        }
                        tmp2 = prev_exp_tmp[h][ii]*tmp_o + tmp;
                        if (c > 0)  {
                            if (j==(CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]-1)){
                                inv_row_sum[h][ii] = lookup_inv<typename CONFIG_T::accum_t, CONFIG_T>(new_rowsum[h][ii]);
                                O[h][i][k][ii][kk] = tmp2*inv_row_sum[h][ii];
                            } else {
                                O[h][i][k][ii][kk] = tmp2;
                            }
                        }
                    }
                }
            }
            Update_RowMaxSum<CONFIG_T>(new_rowsum, prev_rowsum, new_rowmax, prev_rowmax);
            if (k==(CONFIG_T::head_dim/CONFIG_T::tiling_factor[1]-1)) {
                if (init_QK0){
                    init_QK0 = false;
                } else {
                    init_QK0 = true;
                }
                if (c > 0) { //cycle 0~total_cycle-1
                    j++;
                    if (j == CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]){
                        j = 0;
                        i++;
                    }
                }
                if (c < total_cycle-1) { //cycle 1~total_cycle
                    n++;
                    if (n == CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]){
                        n = 0;
                        m++;
                    }
                }
            }
        }   
    }

    //print QK_debug
    for (int h = 0; h < CONFIG_T::n_head; h++) {
        for (int i = 0; i < CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]; i++) {
            for (int ii = 0; ii < CONFIG_T::tiling_factor[0]; ii++) {
                for (int j = 0; j < CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]; j++) {
                    for (int jj = 0; jj < CONFIG_T::tiling_factor[0]; jj++) {
                        std::cout << QK_debug[h][i][j][ii][jj] << " ";
                    }
                }
                std::cout << std::endl;
            }
        }
        std::cout << "change qk head" << std::endl;
    }

    //print O
    //for (int h = 0; h < CONFIG_T::n_head; h++) {
    //    for (int i = 0; i < CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]; i++) {
    //        for (int ii = 0; ii < CONFIG_T::tiling_factor[0]; ii++) {
    //            for (int j = 0; j < CONFIG_T::head_dim/CONFIG_T::tiling_factor[2]; j++) {
    //                for (int jj = 0; jj < CONFIG_T::tiling_factor[2]; jj++) {
    //                    std::cout << O[h][i][j][ii][jj] << " ";
    //                }
    //            }
    //            std::cout << std::endl;
    //        }
    //    }
    //    std::cout << "change head" << std::endl;
    //}
    typename CONFIG_T::accum_t tmp_m;                        
    compute_output: 
    for (int i = 0; i < CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]; i++) {
        for (int k = 0; k < CONFIG_T::head_dim/CONFIG_T::tiling_factor[2]; k++) {
            for (int j = 0; j < CONFIG_T::embed_dim/CONFIG_T::tiling_factor[1]; j++) {
                #pragma HLS PIPELINE II = 1
                for (int ii = 0; ii < CONFIG_T::tiling_factor[0]; ii++) {
                    #pragma HLS UNROLL
                    for (int jj = 0; jj < CONFIG_T::tiling_factor[1]; jj++) {
                        #pragma HLS UNROLL
                        if (k==0){
                            tmp_m = out_proj_bias[j][jj];
                        } else {
                            tmp_m = M[i][j][ii][jj];
                        }
                        for (int kk = 0; kk < CONFIG_T::tiling_factor[2]; kk++) {
                            #pragma HLS UNROLL
                            for (int h = 0; h < CONFIG_T::n_head; h++) {//16dsp
                                #pragma HLS UNROLL
                                tmp_m += O[h][i][k][ii][kk] * out_proj_weight[h][k][j][kk][jj];
                            }
                        }
                        M[i][j][ii][jj] = tmp_m;
                    }
                }
            }
        }
    }
    
    res_T res_pack;
    write_output:
    for (int i = 0; i < T; i++) {
        for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II = 1
            for (int ii = 0; ii < tf_T; ii++) {
                #pragma HLS UNROLL
                for (int jj = 0; jj < CONFIG_T::tiling_factor[1]; jj++) {
                    #pragma HLS UNROLL
                    res_pack[ii*CONFIG_T::tiling_factor[1] + jj] = M[i][j][ii][jj];
                    if (jj==CONFIG_T::tiling_factor[1]-1 && ii==tf_T-1){
                        res.write(res_pack);
                    }
                }
            }
        }
    }
    
}
*/
}

#endif
