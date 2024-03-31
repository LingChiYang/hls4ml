#ifndef NNET_MHT_SS_H_
#define NNET_MHT_SS_H_

#include "nnet_common.h"
#include "nnet_mult.h"
#include "nnet_activation.h"
#include "hls_stream.h"
#include <iostream>
#include <math.h>
#include "nnet_helpers.h"

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
        init_exp_table_legacy<CONFIG_T, CONFIG_T::table_size>(exp_table);
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
        init_invert_table_legacy<CONFIG_T, CONFIG_T::table_size>(inv_table);
        initialized = true;
    }

    int index = int(data*(CONFIG_T::table_size/CONFIG_T::inv_range));
    if (index < 0)   index = 0;
    if (index > CONFIG_T::table_size-1) index = CONFIG_T::table_size-1;
    return inv_table[index];
}

//initialize row_sum and row_max

template<typename CONFIG_T>
void Init_RowMaxSum(
    typename CONFIG_T::softmax_config1::accum_t row_sum[CONFIG_T::num_heads][CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::softmax_config1::accum_t new_row_sum[CONFIG_T::num_heads][CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::softmax_config1::accum_t prev_row_sum[CONFIG_T::num_heads][CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::softmax_config1::accum_t row_max[CONFIG_T::num_heads][CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::softmax_config1::accum_t new_row_max[CONFIG_T::num_heads][CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::softmax_config1::accum_t prev_row_max[CONFIG_T::num_heads][CONFIG_T::tiling_factor[0]])
{
    for (int i = 0; i < CONFIG_T::num_heads; i++) {
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
    typename CONFIG_T::softmax_config1::accum_t QK[CONFIG_T::num_heads][CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[0]])
{
    for (int i = 0; i < CONFIG_T::num_heads; i++) {
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
    typename CONFIG_T::softmax_config1::accum_t QK[CONFIG_T::num_heads][CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::softmax_config1::accum_t row_max[CONFIG_T::num_heads][CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::softmax_config1::accum_t row_sum[CONFIG_T::num_heads][CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::softmax_config1::exp_table_t P[CONFIG_T::num_heads][CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[0]])
{
    for (int i = 0; i < CONFIG_T::num_heads; i++) {
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
                P[i][ii][ll] = lookup_exp<typename CONFIG_T::softmax_config1::accum_t, typename CONFIG_T::softmax_config1>(QK[i][ii][ll] - row_max[i][ii]);
                row_sum[i][ii] += P[i][ii][ll];
            }
        }
    }
}


//update row_sum and row_max
template<typename CONFIG_T>
void Update_RowMaxSum(
    typename CONFIG_T::softmax_config1::accum_t new_row_sum[CONFIG_T::num_heads][CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::softmax_config1::accum_t prev_row_sum[CONFIG_T::num_heads][CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::softmax_config1::accum_t new_row_max[CONFIG_T::num_heads][CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::softmax_config1::accum_t prev_row_max[CONFIG_T::num_heads][CONFIG_T::tiling_factor[0]])
{
    #pragma HLS ARRAY_PARTITION variable=new_row_sum complete dim=0
    #pragma HLS ARRAY_PARTITION variable=prev_row_sum complete dim=0
    #pragma HLS ARRAY_PARTITION variable=new_row_max complete dim=0
    #pragma HLS ARRAY_PARTITION variable=prev_row_max complete dim=0
    for (int i = 0; i < CONFIG_T::num_heads; i++) {
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
    typename CONFIG_T::weight_t  in_proj_weight[3][CONFIG_T::num_heads][CONFIG_T::feature_dim/CONFIG_T::tiling_factor[1]][CONFIG_T::head_dim_key/CONFIG_T::tiling_factor[2]][CONFIG_T::tiling_factor[1]][CONFIG_T::tiling_factor[2]],
    typename CONFIG_T::bias_t    in_proj_bias[3][CONFIG_T::num_heads][CONFIG_T::head_dim_key/CONFIG_T::tiling_factor[2]][CONFIG_T::tiling_factor[2]],
    typename CONFIG_T::weight_t  attention_output_weight[CONFIG_T::num_heads][CONFIG_T::head_dim_key/CONFIG_T::tiling_factor[2]][CONFIG_T::feature_dim/CONFIG_T::tiling_factor[1]][CONFIG_T::tiling_factor[2]][CONFIG_T::tiling_factor[1]],  // num_heads,head_size_v,dim
    typename CONFIG_T::bias_t    attention_output_bias[CONFIG_T::feature_dim/CONFIG_T::tiling_factor[1]][CONFIG_T::tiling_factor[1]])
{

    //data_T data_q_buf[CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]][CONFIG_T::feature_dim/CONFIG_T::tiling_factor[1]][CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[1]];
    data_T data_qkv_buf[CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]][CONFIG_T::feature_dim/CONFIG_T::tiling_factor[1]][CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[1]];
    data_T K[CONFIG_T::num_heads][CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]][CONFIG_T::head_dim_key/CONFIG_T::tiling_factor[2]][CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[2]];
    data_T V[CONFIG_T::num_heads][CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]][CONFIG_T::head_dim_key/CONFIG_T::tiling_factor[2]][CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[2]];
    data_T Q[CONFIG_T::num_heads][CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]][CONFIG_T::head_dim_key/CONFIG_T::tiling_factor[2]][CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[2]];
    data_T O[CONFIG_T::num_heads][CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]][CONFIG_T::head_dim_key/CONFIG_T::tiling_factor[2]][CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[2]];
    data_T M[CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]][CONFIG_T::feature_dim/CONFIG_T::tiling_factor[1]][CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[1]];
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
    data_T QK0[CONFIG_T::num_heads][CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[0]];
    data_T QK1[CONFIG_T::num_heads][CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[0]];
    typename CONFIG_T::softmax_config1::exp_table_t P[CONFIG_T::num_heads][CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[0]];
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


    store_data: 
    for (int i = 0; i < CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]; i++) {
        for (int j = 0; j < CONFIG_T::feature_dim/CONFIG_T::tiling_factor[1]; j++) {
            for (int ii = 0; ii < CONFIG_T::tiling_factor[0]; ii++) {
                for (int jj = 0; jj < CONFIG_T::tiling_factor[1]; jj++) {
                    #pragma HLS PIPELINE II = 1
                    data_qkv_buf[i][j][ii][jj] = data_qkv.read()[ii][jj];
                    //data_vk_buf[i][j][ii][jj] = data_vk[j*CONFIG_T::tiling_factor[1] + jj].read();
                    //data_q_buf[i][j][ii][jj].write(data_q[j*CONFIG_T::tiling_factor[1] + jj].read());
                    //data_vk_buf[i][j][ii][jj].write(data_vk[j*CONFIG_T::tiling_factor[1] + jj].read());
                }
            }
        }
    }

    typename CONFIG_T::softmax_config1::accum_t rowsum[CONFIG_T::num_heads][CONFIG_T::tiling_factor[0]];
    typename CONFIG_T::softmax_config1::accum_t new_rowsum[CONFIG_T::num_heads][CONFIG_T::tiling_factor[0]];
    typename CONFIG_T::softmax_config1::accum_t new_rowmax[CONFIG_T::num_heads][CONFIG_T::tiling_factor[0]];
    typename CONFIG_T::softmax_config1::accum_t prev_rowmax[CONFIG_T::num_heads][CONFIG_T::tiling_factor[0]];
    typename CONFIG_T::softmax_config1::accum_t prev_rowsum[CONFIG_T::num_heads][CONFIG_T::tiling_factor[0]];
    typename CONFIG_T::softmax_config1::accum_t rowmax[CONFIG_T::num_heads][CONFIG_T::tiling_factor[0]];
    
    #pragma HLS ARRAY_PARTITION variable=rowsum complete dim=0
    #pragma HLS ARRAY_PARTITION variable=new_rowsum complete dim=0
    #pragma HLS ARRAY_PARTITION variable=new_rowmax complete dim=0
    #pragma HLS ARRAY_PARTITION variable=prev_rowmax complete dim=0
    #pragma HLS ARRAY_PARTITION variable=prev_rowsum complete dim=0
    #pragma HLS ARRAY_PARTITION variable=rowmax complete dim=0
    const data_T dk = 1.0/sqrt(CONFIG_T::head_dim_key);
    int index;
    int data_round; 
    typename CONFIG_T::accum_t tmp_k[CONFIG_T::num_heads][CONFIG_T::tiling_factor[2]];
    typename CONFIG_T::accum_t tmp_v[CONFIG_T::num_heads][CONFIG_T::tiling_factor[2]];
    typename CONFIG_T::accum_t tmp_q[CONFIG_T::num_heads][CONFIG_T::tiling_factor[2]];
    #pragma HLS ARRAY_PARTITION variable=tmp_k complete dim=0
    #pragma HLS ARRAY_PARTITION variable=tmp_v complete dim=0
    #pragma HLS ARRAY_PARTITION variable=tmp_q complete dim=0
    compute_KVQ:  
    for (int i = 0; i < CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]; i++) {
        for (int k = 0; k < CONFIG_T::head_dim_key/CONFIG_T::tiling_factor[2]; k++) {
            for (int j = 0; j < CONFIG_T::feature_dim/CONFIG_T::tiling_factor[1]; j++) {
                #pragma HLS PIPELINE II = 1
                for (int h = 0; h < CONFIG_T::num_heads; h++) {//48dsp
                    #pragma HLS UNROLL
                    for (int ii = 0; ii < CONFIG_T::tiling_factor[0]; ii++) {
                        #pragma HLS UNROLL
                        for (int kk = 0; kk < CONFIG_T::tiling_factor[2]; kk++) {
                            #pragma HLS UNROLL
                            if (j==0){
                                tmp_k[h][kk] = key_bias[h][k][kk];
                                tmp_v[h][kk] = value_bias[h][k][kk];
                                tmp_q[h][kk] = query_bias[h][k][kk];
                            }
                            for (int jj = 0; jj < CONFIG_T::tiling_factor[1]; jj++) {
                                #pragma HLS UNROLL
                                tmp_k[h][kk] += data_qkv_buf[i][j][ii][jj]*key_weight[h][j][k][jj][kk],
                                tmp_v[h][kk] += data_qkv_buf[i][j][ii][jj]*value_weight[h][j][k][jj][kk],
                                tmp_q[h][kk] += data_qkv_buf[i][j][ii][jj]*query_weight[h][j][k][jj][kk];
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
    

    typename CONFIG_T::softmax_config1::exp_table_t prev_exp_tmp[CONFIG_T::num_heads][CONFIG_T::tiling_factor[0]];
    typename CONFIG_T::softmax_config1::exp_table_t exp_tmp[CONFIG_T::num_heads][CONFIG_T::tiling_factor[0]];
    typename CONFIG_T::softmax_config1::inv_table_t inv_row_sum[CONFIG_T::num_heads][CONFIG_T::tiling_factor[0]];
    #pragma HLS ARRAY_PARTITION variable=inv_row_sum complete dim=0
    #pragma HLS ARRAY_PARTITION variable=prev_exp_tmp complete dim=0
    #pragma HLS ARRAY_PARTITION variable=exp_tmp complete dim=0
    int i = 0;
    int j = 0;
    int m = 0;
    int n = 0;
    typename CONFIG_T::accum_t tmp_o;
    int total_cycle = ((CONFIG_T::seq_len*CONFIG_T::seq_len)/(CONFIG_T::tiling_factor[0]*CONFIG_T::tiling_factor[0])) + 1;
    bool init_QK0 = false; 
    compute_att_pipeline:   
    for (int c=0; c < total_cycle; ++c){
        for (int k=0; k<CONFIG_T::head_dim_key/CONFIG_T::tiling_factor[2]; k++) {
            #pragma HLS PIPELINE II = 1
            if (k==0) {
                if (init_QK0){
                    if (c < total_cycle-1) { Init_Attention<CONFIG_T>(QK0); }
                    if (c > 0) { Compute_CurrentTile_RowMaxSum<CONFIG_T>(QK1, rowmax, rowsum, P); }
                } else {
                    if (c < total_cycle-1) { Init_Attention<CONFIG_T>(QK1); }
                    if (c > 0) { Compute_CurrentTile_RowMaxSum<CONFIG_T>(QK0, rowmax, rowsum, P); }
                }
                for (int i = 0; i < CONFIG_T::num_heads; i++) {//8dsp
                    #pragma HLS UNROLL
                    for (int ii = 0; ii < CONFIG_T::tiling_factor[0]; ii++) {
                        #pragma HLS UNROLL
                        if (j==0){
                            prev_rowmax[i][ii] = -128;
                            prev_rowsum[i][ii] = 0;
                            new_rowmax[i][ii] = rowmax[i][ii];
                        } else {
                            new_rowmax[i][ii] = prev_rowmax[i][ii] > rowmax[i][ii] ? prev_rowmax[i][ii] : rowmax[i][ii];
                        }
                        prev_exp_tmp[i][ii] = lookup_exp<typename CONFIG_T::softmax_config1::accum_t, typename CONFIG_T::softmax_config1>(prev_rowmax[i][ii] - new_rowmax[i][ii]);
                        exp_tmp[i][ii] = lookup_exp<typename CONFIG_T::softmax_config1::accum_t, typename CONFIG_T::softmax_config1>(rowmax[i][ii] - new_rowmax[i][ii]);
                        new_rowsum[i][ii] = prev_exp_tmp[i][ii]*prev_rowsum[i][ii] + rowsum[i][ii];
                    }
                }
            }
            for (int h = 0; h < CONFIG_T::num_heads; h++) { //16dsp
                #pragma HLS UNROLL
                for (int mm = 0; mm < CONFIG_T::tiling_factor[0]; mm++) {
                    #pragma HLS UNROLL
                    for (int nn = 0; nn < CONFIG_T::tiling_factor[0]; nn++) {
                        #pragma HLS UNROLL
                        for (int kk = 0; kk < CONFIG_T::tiling_factor[2]; kk++) {
                            #pragma HLS UNROLL
                            typename CONFIG_T::accum_t tmp = 0;
                            tmp = Q[h][m][k][mm][kk]*K[h][n][k][nn][kk] * dk;
                            if (init_QK0){
                                QK0[h][mm][nn] += tmp;
                            } else {
                                QK1[h][mm][nn] += tmp;
                            }
                        }
                    }
                }
            }
            for (int h = 0; h < CONFIG_T::num_heads; h++) {//48 dsp
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
                                inv_row_sum[h][ii] = lookup_inv<typename CONFIG_T::softmax_config1::accum_t, typename CONFIG_T::softmax_config1>(new_rowsum[h][ii]);
                                O[h][i][k][ii][kk] = tmp2*inv_row_sum[h][ii];
                            } else {
                                O[h][i][k][ii][kk] = tmp2;
                            }
                        }
                    }
                }
            }
            Update_RowMaxSum<CONFIG_T>(new_rowsum, prev_rowsum, new_rowmax, prev_rowmax);
            if (k==(CONFIG_T::head_dim_key/CONFIG_T::tiling_factor[1]-1)) {
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
    typename CONFIG_T::accum_t tmp_m;                        
    compute_output: 
    for (int i = 0; i < CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]; i++) {
        for (int k = 0; k < CONFIG_T::head_dim_key/CONFIG_T::tiling_factor[2]; k++) {
            for (int j = 0; j < CONFIG_T::feature_dim/CONFIG_T::tiling_factor[1]; j++) {
                #pragma HLS PIPELINE II = 1
                for (int ii = 0; ii < CONFIG_T::tiling_factor[0]; ii++) {
                    #pragma HLS UNROLL
                    for (int jj = 0; jj < CONFIG_T::tiling_factor[1]; jj++) {
                        #pragma HLS UNROLL
                        if (k==0){
                            tmp_m = attention_output_bias[j][jj];
                        } else {
                            tmp_m = M[i][j][ii][jj];
                        }
                        for (int kk = 0; kk < CONFIG_T::tiling_factor[2]; kk++) {
                            #pragma HLS UNROLL
                            for (int h = 0; h < CONFIG_T::num_heads; h++) {//16dsp
                                #pragma HLS UNROLL
                                tmp_m += O[h][i][k][ii][kk] * attention_output_weight[h][k][j][kk][jj];
                            }
                        }
                        M[i][j][ii][jj] = tmp_m;
                    }
                }
            }
        }
    }
    
    write_output:
    for (int i = 0; i < CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]; i++) {
        for (int j = 0; j < CONFIG_T::feature_dim/CONFIG_T::tiling_factor[1]; j++) {
            for (int ii = 0; ii < CONFIG_T::tiling_factor[0]; ii++) {
                for (int jj = 0; jj < CONFIG_T::tiling_factor[1]; jj++) {
                    #pragma HLS PIPELINE II = 1
                    res[j*CONFIG_T::tiling_factor[1] + jj].write(M[i][j][ii][jj]);
                }
            }
        }
    }
    
}

template<class data_T, class res_T, typename CONFIG_T>
void multiheadattention(
    hls::stream<data_T>    &data_q,
    hls::stream<data_T>    &data_vk,
    hls::stream<res_T>     &res,
    typename CONFIG_T::weight_t  attention_output_weight[CONFIG_T::num_heads][CONFIG_T::head_dim_key/CONFIG_T::tiling_factor[2]][CONFIG_T::feature_dim/CONFIG_T::tiling_factor[1]][CONFIG_T::tiling_factor[2]][CONFIG_T::tiling_factor[1]],  // num_heads,head_size_v,dim
    typename CONFIG_T::bias_t    attention_output_bias[CONFIG_T::feature_dim/CONFIG_T::tiling_factor[1]][CONFIG_T::tiling_factor[1]],
    typename CONFIG_T::weight_t  key_weight[CONFIG_T::num_heads][CONFIG_T::feature_dim/CONFIG_T::tiling_factor[1]][CONFIG_T::head_dim_key/CONFIG_T::tiling_factor[2]][CONFIG_T::tiling_factor[1]][CONFIG_T::tiling_factor[2]],  // n_head,dim,head_dim
    typename CONFIG_T::bias_t    key_bias[CONFIG_T::num_heads][CONFIG_T::head_dim_key/CONFIG_T::tiling_factor[2]][CONFIG_T::tiling_factor[2]],
    typename CONFIG_T::weight_t  query_weight[CONFIG_T::num_heads][CONFIG_T::feature_dim/CONFIG_T::tiling_factor[1]][CONFIG_T::head_dim_key/CONFIG_T::tiling_factor[2]][CONFIG_T::tiling_factor[1]][CONFIG_T::tiling_factor[2]], //same shape as key
    typename CONFIG_T::bias_t    query_bias[CONFIG_T::num_heads][CONFIG_T::head_dim_key/CONFIG_T::tiling_factor[2]][CONFIG_T::tiling_factor[2]],
    typename CONFIG_T::weight_t  value_weight[CONFIG_T::num_heads][CONFIG_T::feature_dim/CONFIG_T::tiling_factor[1]][CONFIG_T::head_dim_key/CONFIG_T::tiling_factor[2]][CONFIG_T::tiling_factor[1]][CONFIG_T::tiling_factor[2]],
    typename CONFIG_T::bias_t    value_bias[CONFIG_T::num_heads][CONFIG_T::head_dim_key/CONFIG_T::tiling_factor[2]][CONFIG_T::tiling_factor[2]])
{

    data_T data_q_buf[CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]][CONFIG_T::feature_dim/CONFIG_T::tiling_factor[1]][CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[1]];
    data_T data_vk_buf[CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]][CONFIG_T::feature_dim/CONFIG_T::tiling_factor[1]][CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[1]];
    data_T K[CONFIG_T::num_heads][CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]][CONFIG_T::head_dim_key/CONFIG_T::tiling_factor[2]][CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[2]];
    data_T V[CONFIG_T::num_heads][CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]][CONFIG_T::head_dim_key/CONFIG_T::tiling_factor[2]][CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[2]];
    data_T Q[CONFIG_T::num_heads][CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]][CONFIG_T::head_dim_key/CONFIG_T::tiling_factor[2]][CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[2]];
    data_T O[CONFIG_T::num_heads][CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]][CONFIG_T::head_dim_key/CONFIG_T::tiling_factor[2]][CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[2]];
    data_T M[CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]][CONFIG_T::feature_dim/CONFIG_T::tiling_factor[1]][CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[1]];

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
    data_T QK0[CONFIG_T::num_heads][CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[0]];
    data_T QK1[CONFIG_T::num_heads][CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[0]];
    typename CONFIG_T::softmax_config1::exp_table_t P[CONFIG_T::num_heads][CONFIG_T::tiling_factor[0]][CONFIG_T::tiling_factor[0]];
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


    store_data: 
    for (int i = 0; i < CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]; i++) {
        for (int j = 0; j < CONFIG_T::feature_dim/CONFIG_T::tiling_factor[1]; j++) {
            for (int ii = 0; ii < CONFIG_T::tiling_factor[0]; ii++) {
                for (int jj = 0; jj < CONFIG_T::tiling_factor[1]; jj++) {
                    #pragma HLS PIPELINE II = 1
                    data_q_buf[i][j][ii][jj] = data_q.read();
                    data_vk_buf[i][j][ii][jj] = data_vk.read();
                }
            }
        }
    }

    typename CONFIG_T::softmax_config1::accum_t rowsum[CONFIG_T::num_heads][CONFIG_T::tiling_factor[0]];
    typename CONFIG_T::softmax_config1::accum_t new_rowsum[CONFIG_T::num_heads][CONFIG_T::tiling_factor[0]];
    typename CONFIG_T::softmax_config1::accum_t new_rowmax[CONFIG_T::num_heads][CONFIG_T::tiling_factor[0]];
    typename CONFIG_T::softmax_config1::accum_t prev_rowmax[CONFIG_T::num_heads][CONFIG_T::tiling_factor[0]];
    typename CONFIG_T::softmax_config1::accum_t prev_rowsum[CONFIG_T::num_heads][CONFIG_T::tiling_factor[0]];
    typename CONFIG_T::softmax_config1::accum_t rowmax[CONFIG_T::num_heads][CONFIG_T::tiling_factor[0]];
    
    #pragma HLS ARRAY_PARTITION variable=rowsum complete dim=0
    #pragma HLS ARRAY_PARTITION variable=new_rowsum complete dim=0
    #pragma HLS ARRAY_PARTITION variable=new_rowmax complete dim=0
    #pragma HLS ARRAY_PARTITION variable=prev_rowmax complete dim=0
    #pragma HLS ARRAY_PARTITION variable=prev_rowsum complete dim=0
    #pragma HLS ARRAY_PARTITION variable=rowmax complete dim=0
    const data_T dk = 1.0/sqrt(CONFIG_T::head_dim_key);
    int index;
    int data_round; 
    typename CONFIG_T::accum_t tmp_k[CONFIG_T::num_heads][CONFIG_T::tiling_factor[2]];
    typename CONFIG_T::accum_t tmp_v[CONFIG_T::num_heads][CONFIG_T::tiling_factor[2]];
    typename CONFIG_T::accum_t tmp_q[CONFIG_T::num_heads][CONFIG_T::tiling_factor[2]];
    #pragma HLS ARRAY_PARTITION variable=tmp_k complete dim=0
    #pragma HLS ARRAY_PARTITION variable=tmp_v complete dim=0
    #pragma HLS ARRAY_PARTITION variable=tmp_q complete dim=0
    compute_KVQ:  
    for (int i = 0; i < CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]; i++) {
        for (int k = 0; k < CONFIG_T::head_dim_key/CONFIG_T::tiling_factor[2]; k++) {
            for (int j = 0; j < CONFIG_T::feature_dim/CONFIG_T::tiling_factor[1]; j++) {
                #pragma HLS PIPELINE II = 1
                for (int h = 0; h < CONFIG_T::num_heads; h++) {//48dsp
                    #pragma HLS UNROLL
                    for (int ii = 0; ii < CONFIG_T::tiling_factor[0]; ii++) {
                        #pragma HLS UNROLL
                        for (int kk = 0; kk < CONFIG_T::tiling_factor[2]; kk++) {
                            #pragma HLS UNROLL
                            if (j==0){
                                tmp_k[h][kk] = key_bias[h][k][kk];
                                tmp_v[h][kk] = value_bias[h][k][kk];
                                tmp_q[h][kk] = query_bias[h][k][kk];
                            }
                            for (int jj = 0; jj < CONFIG_T::tiling_factor[1]; jj++) {
                                #pragma HLS UNROLL
                                tmp_k[h][kk] += data_vk_buf[i][j][ii][jj]*key_weight[h][j][k][jj][kk],
                                tmp_v[h][kk] += data_vk_buf[i][j][ii][jj]*value_weight[h][j][k][jj][kk],
                                tmp_q[h][kk] += data_q_buf[i][j][ii][jj]*query_weight[h][j][k][jj][kk];
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
    

    typename CONFIG_T::softmax_config1::exp_table_t prev_exp_tmp[CONFIG_T::num_heads][CONFIG_T::tiling_factor[0]];
    typename CONFIG_T::softmax_config1::exp_table_t exp_tmp[CONFIG_T::num_heads][CONFIG_T::tiling_factor[0]];
    typename CONFIG_T::softmax_config1::inv_table_t inv_row_sum[CONFIG_T::num_heads][CONFIG_T::tiling_factor[0]];
    #pragma HLS ARRAY_PARTITION variable=inv_row_sum complete dim=0
    #pragma HLS ARRAY_PARTITION variable=prev_exp_tmp complete dim=0
    #pragma HLS ARRAY_PARTITION variable=exp_tmp complete dim=0
    int i = 0;
    int j = 0;
    int m = 0;
    int n = 0;
    typename CONFIG_T::accum_t tmp_o;
    int total_cycle = ((CONFIG_T::seq_len*CONFIG_T::seq_len)/(CONFIG_T::tiling_factor[0]*CONFIG_T::tiling_factor[0])) + 1;
    bool init_QK0 = false; 
    compute_att_pipeline:   
    for (int c=0; c < total_cycle; ++c){
        for (int k=0; k<CONFIG_T::head_dim_key/CONFIG_T::tiling_factor[2]; k++) {
            #pragma HLS PIPELINE II = 1
            if (k==0) {
                if (init_QK0){
                    if (c < total_cycle-1) { Init_Attention<CONFIG_T>(QK0); }
                    if (c > 0) { Compute_CurrentTile_RowMaxSum<CONFIG_T>(QK1, rowmax, rowsum, P); }
                } else {
                    if (c < total_cycle-1) { Init_Attention<CONFIG_T>(QK1); }
                    if (c > 0) { Compute_CurrentTile_RowMaxSum<CONFIG_T>(QK0, rowmax, rowsum, P); }
                }
                for (int i = 0; i < CONFIG_T::num_heads; i++) {//8dsp
                    #pragma HLS UNROLL
                    for (int ii = 0; ii < CONFIG_T::tiling_factor[0]; ii++) {
                        #pragma HLS UNROLL
                        if (j==0){
                            prev_rowmax[i][ii] = -128;
                            prev_rowsum[i][ii] = 0;
                            new_rowmax[i][ii] = rowmax[i][ii];
                        } else {
                            new_rowmax[i][ii] = prev_rowmax[i][ii] > rowmax[i][ii] ? prev_rowmax[i][ii] : rowmax[i][ii];
                        }
                        prev_exp_tmp[i][ii] = lookup_exp<typename CONFIG_T::softmax_config1::accum_t, typename CONFIG_T::softmax_config1>(prev_rowmax[i][ii] - new_rowmax[i][ii]);
                        exp_tmp[i][ii] = lookup_exp<typename CONFIG_T::softmax_config1::accum_t, typename CONFIG_T::softmax_config1>(rowmax[i][ii] - new_rowmax[i][ii]);
                        new_rowsum[i][ii] = prev_exp_tmp[i][ii]*prev_rowsum[i][ii] + rowsum[i][ii];
                    }
                }
            }
            for (int h = 0; h < CONFIG_T::num_heads; h++) { //16dsp
                #pragma HLS UNROLL
                for (int mm = 0; mm < CONFIG_T::tiling_factor[0]; mm++) {
                    #pragma HLS UNROLL
                    for (int nn = 0; nn < CONFIG_T::tiling_factor[0]; nn++) {
                        #pragma HLS UNROLL
                        for (int kk = 0; kk < CONFIG_T::tiling_factor[2]; kk++) {
                            #pragma HLS UNROLL
                            typename CONFIG_T::accum_t tmp = 0;
                            tmp = Q[h][m][k][mm][kk]*K[h][n][k][nn][kk] * dk;
                            if (init_QK0){
                                QK0[h][mm][nn] += tmp;
                            } else {
                                QK1[h][mm][nn] += tmp;
                            }
                        }
                    }
                }
            }
            for (int h = 0; h < CONFIG_T::num_heads; h++) {//48 dsp
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
                                inv_row_sum[h][ii] = lookup_inv<typename CONFIG_T::softmax_config1::accum_t, typename CONFIG_T::softmax_config1>(new_rowsum[h][ii]);
                                O[h][i][k][ii][kk] = tmp2*inv_row_sum[h][ii];
                            } else {
                                O[h][i][k][ii][kk] = tmp2;
                            }
                        }
                    }
                }
            }
            Update_RowMaxSum<CONFIG_T>(new_rowsum, prev_rowsum, new_rowmax, prev_rowmax);
            if (k==(CONFIG_T::head_dim_key/CONFIG_T::tiling_factor[1]-1)) {
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
    typename CONFIG_T::accum_t tmp_m;                        
    compute_output: 
    for (int i = 0; i < CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]; i++) {
        for (int k = 0; k < CONFIG_T::head_dim_key/CONFIG_T::tiling_factor[2]; k++) {
            for (int j = 0; j < CONFIG_T::feature_dim/CONFIG_T::tiling_factor[1]; j++) {
                #pragma HLS PIPELINE II = 1
                for (int ii = 0; ii < CONFIG_T::tiling_factor[0]; ii++) {
                    #pragma HLS UNROLL
                    for (int jj = 0; jj < CONFIG_T::tiling_factor[1]; jj++) {
                        #pragma HLS UNROLL
                        if (k==0){
                            tmp_m = attention_output_bias[j][jj];
                        } else {
                            tmp_m = M[i][j][ii][jj];
                        }
                        for (int kk = 0; kk < CONFIG_T::tiling_factor[2]; kk++) {
                            #pragma HLS UNROLL
                            for (int h = 0; h < CONFIG_T::num_heads; h++) {//16dsp
                                #pragma HLS UNROLL
                                tmp_m += O[h][i][k][ii][kk] * attention_output_weight[h][k][j][kk][jj];
                            }
                        }
                        M[i][j][ii][jj] = tmp_m;
                    }
                }
            }
        }
    }
    
    write_output:
    for (int i = 0; i < CONFIG_T::seq_len/CONFIG_T::tiling_factor[0]; i++) {
        for (int j = 0; j < CONFIG_T::feature_dim/CONFIG_T::tiling_factor[1]; j++) {
            for (int ii = 0; ii < CONFIG_T::tiling_factor[0]; ii++) {
                for (int jj = 0; jj < CONFIG_T::tiling_factor[1]; jj++) {
                    #pragma HLS PIPELINE II = 1
                    res.write(M[i][j][ii][jj]);
                }
            }
        }
    }
    
}
}

#endif
