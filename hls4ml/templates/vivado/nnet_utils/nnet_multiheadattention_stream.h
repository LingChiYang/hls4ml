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
void init_exp_table(typename CONFIG_T::exp_table_t table_out[CONFIG_T::exp_table_size])
{
    for (int ii = 0; ii < CONFIG_T::exp_table_size; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range -8 to +8)
        float in_val = 2 * float(CONFIG_T::exp_range) * (ii - float(CONFIG_T::exp_table_size) / 2.0) / float(CONFIG_T::exp_table_size);
        // Next, compute lookup table function
        typename CONFIG_T::exp_table_t real_val = std::exp(in_val);
        // std::cout << "Lookup table In Value: " << in_val << " Result: " << real_val << std::endl;
        table_out[ii] = real_val;
    }
}

template <typename CONFIG_T>
void init_invert_table(typename CONFIG_T::inv_table_t table_out[CONFIG_T::inv_table_size])
{
    // Inversion function:
    //   result = 1/x
    for (int ii = 0; ii < CONFIG_T::inv_table_size; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range 0 to +64)
        float in_val = float(CONFIG_T::inv_range) * ii / float(CONFIG_T::inv_table_size);
        // Next, compute lookup table function
        if (in_val > 0.0)
            table_out[ii] = 1.0 / in_val;
        else
            table_out[ii] = 0.0;
    }
}

template<typename CONFIG_T>
typename CONFIG_T::exp_table_t lookup_exp(
    typename CONFIG_T::accum_t data)
{
    #ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::exp_table_t exp_table[CONFIG_T::exp_table_size];
    #else
    static bool initialized = false;
    static typename CONFIG_T::exp_table_t exp_table[CONFIG_T::exp_table_size];
    #endif

    if (!initialized) {
        //init_exp_table_legacy<CONFIG_T, CONFIG_T::table_size>(exp_table);
        init_exp_table<CONFIG_T>(exp_table);
        initialized = true;
    }
    //std::cout << "fixed point data before: " << data << std::endl;
    int data_round = int(data*(CONFIG_T::exp_table_size/(CONFIG_T::exp_range*2)));
    //std::cout << "data_round: " << data_round << std::endl;
    //std::cout << "fixed point data: " << static_cast<typename CONFIG_T::accum_t>(data)*(CONFIG_T::exp_range*2)/CONFIG_T::exp_table_size << std::endl;
    int index = data_round + CONFIG_T::exp_range*(CONFIG_T::exp_table_size/(CONFIG_T::exp_range*2));
    //print index
    if (index < 0)   index = 0;
    if (index > CONFIG_T::exp_table_size-1) index = CONFIG_T::exp_table_size-1;
    return exp_table[index];
}

template<typename CONFIG_T>
typename CONFIG_T::inv_table_t lookup_inv(
    typename CONFIG_T::row_sum_t data)
{
    #ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::inv_table_t inv_table[CONFIG_T::inv_table_size];
    #else
    static bool initialized = false;
    static typename CONFIG_T::inv_table_t inv_table[CONFIG_T::inv_table_size];
    #endif

    if (!initialized) {
        //init_invert_table_legacy<CONFIG_T, CONFIG_T::table_size>(inv_table);
        init_invert_table<CONFIG_T>(inv_table);
        initialized = true;
    }
    //std::cout << "fixed point data before: " << data << std::endl;
    int index = int(data*(CONFIG_T::inv_table_size/CONFIG_T::inv_range));
    //std::cout << "index: " << index << std::endl;
    if (index < 0)   index = 0;
    if (index > CONFIG_T::inv_table_size-1) index = CONFIG_T::inv_table_size-1;
    return inv_table[index];
}

template<typename CONFIG_T>
void ComputeCurrentTileRowMax(
    typename CONFIG_T::accum_t rowmax[CONFIG_T::n_head*CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::mask_t mask[CONFIG_T::tiling_factor[0]*CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::accum_t QK0[CONFIG_T::n_head*CONFIG_T::tiling_factor[0]*CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::accum_t QK1[CONFIG_T::n_head*CONFIG_T::tiling_factor[0]*CONFIG_T::tiling_factor[0]],
    const bool write_QK0[CONFIG_T::n_head*CONFIG_T::tiling_factor[0]*CONFIG_T::tiling_factor[0]],
    const int tf_T,
    const int T,
    const int c)
{
    for (int h = 0; h < CONFIG_T::n_head; h++) {
        #pragma HLS UNROLL
        for (int ii = 0; ii < tf_T; ii++) {
            #pragma HLS UNROLL
            rowmax[h*tf_T + ii] = static_cast<typename CONFIG_T::accum_t>(-8);
            for (int ii2 = 0; ii2 < tf_T; ii2++) {
                #pragma HLS UNROLL
                if (c < T*T) { 
                    if (write_QK0[h*tf_T*tf_T + ii*tf_T + ii2]) {QK0[h*tf_T*tf_T + ii*tf_T + ii2] = 0;} else {QK1[h*tf_T*tf_T + ii*tf_T + ii2] = 0;}
                }
                if (c > 0) {
                    rowmax[h*tf_T + ii] = (mask[ii*tf_T + ii2] == 1) ? rowmax[h*tf_T + ii] : 
                                    (write_QK0[h*tf_T*tf_T + ii*tf_T + ii2]) ? (rowmax[h*tf_T + ii] > QK1[h*tf_T*tf_T + ii*tf_T + ii2]) ? rowmax[h*tf_T + ii] : QK1[h*tf_T*tf_T + ii*tf_T + ii2] :
                                    (rowmax[h*tf_T + ii] > QK0[h*tf_T*tf_T + ii*tf_T + ii2]) ? rowmax[h*tf_T + ii] : QK0[h*tf_T*tf_T + ii*tf_T + ii2];
                }
            }
        }
    }
}

template<typename CONFIG_T>
void DetermineNewRowMax(
    typename CONFIG_T::accum_t new_rowmax[CONFIG_T::n_head*CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::accum_t prev_rowmax[CONFIG_T::n_head*CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::accum_t rowmax[CONFIG_T::n_head*CONFIG_T::tiling_factor[0]],
    const int tf_T)
{
    for (int h = 0; h < CONFIG_T::n_head; h++) {
        #pragma HLS UNROLL
        for (int ii = 0; ii < tf_T; ii++) {
            #pragma HLS UNROLL
            new_rowmax[h*tf_T + ii] = prev_rowmax[h*tf_T + ii] > rowmax[h*tf_T + ii] ? prev_rowmax[h*tf_T + ii] : rowmax[h*tf_T + ii];
        }
    }
}

template<typename CONFIG_T>
void InitRowMaxSum(
    typename CONFIG_T::row_sum_t prev_rowsum[CONFIG_T::n_head*CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::accum_t prev_rowmax[CONFIG_T::n_head*CONFIG_T::tiling_factor[0]],
    const int tf_T)
{
    for (int i = 0; i < CONFIG_T::n_head; i++) {
        #pragma HLS UNROLL
        for (int ii = 0; ii < tf_T; ii++) {
            #pragma HLS UNROLL
            //set prev_row_max to negative infinity
            prev_rowmax[i*tf_T + ii] = -8;
            prev_rowsum[i*tf_T + ii] = 0;
        }
    }
}

template<typename CONFIG_T>
void DetermineExpRowMaxDiff(
    typename CONFIG_T::accum_t new_rowmax[CONFIG_T::n_head*CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::accum_t prev_rowmax[CONFIG_T::n_head*CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::exp_table_t prev_exp_tmp[CONFIG_T::n_head*CONFIG_T::tiling_factor[0]],
    const int tf_T)
{
    for (int h = 0; h < CONFIG_T::n_head; h++) {
        #pragma HLS UNROLL
        for (int ii = 0; ii < tf_T; ii++) {
            #pragma HLS UNROLL
            prev_exp_tmp[h*tf_T + ii] = lookup_exp<CONFIG_T>(prev_rowmax[h*tf_T + ii] - new_rowmax[h*tf_T + ii]);
        }
    }
}

template<typename CONFIG_T>
void DetermineAttnWeight(
    typename CONFIG_T::exp_table_t P[CONFIG_T::n_head*CONFIG_T::tiling_factor[0]*CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::accum_t QK0[CONFIG_T::n_head*CONFIG_T::tiling_factor[0]*CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::accum_t QK1[CONFIG_T::n_head*CONFIG_T::tiling_factor[0]*CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::accum_t QK[CONFIG_T::n_head*CONFIG_T::tiling_factor[0]*CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::accum_t new_rowmax[CONFIG_T::n_head*CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::mask_t mask[CONFIG_T::tiling_factor[0]*CONFIG_T::tiling_factor[0]],
    const bool write_QK0[CONFIG_T::n_head*CONFIG_T::tiling_factor[0]*CONFIG_T::tiling_factor[0]],
    const int tf_T)
{
    for (int h = 0; h < CONFIG_T::n_head; h++) {
        #pragma HLS UNROLL
        for (int ii = 0; ii < tf_T; ii++) {
            #pragma HLS UNROLL
            for (int jj = 0; jj < tf_T; jj++) {
                #pragma HLS UNROLL
                if (write_QK0[h*tf_T*tf_T + ii*tf_T + jj]) {
                    QK[h*tf_T*tf_T + ii*tf_T + jj] = QK1[h*tf_T*tf_T + ii*tf_T + jj];
                } else {
                    QK[h*tf_T*tf_T + ii*tf_T + jj] = QK0[h*tf_T*tf_T + ii*tf_T + jj];
                }
                //std::cout << "QK[" << h << "][" << ii << "][" << jj << "]: " << QK[h*tf_T*tf_T + ii*tf_T + jj] << std::endl;
                //std::cout << "exp table" << std::endl;
                QK[h*tf_T*tf_T + ii*tf_T + jj] = QK[h*tf_T*tf_T + ii*tf_T + jj] - new_rowmax[h*tf_T + ii];
                P[h*tf_T*tf_T + ii*tf_T + jj] = lookup_exp<CONFIG_T>(QK[h*tf_T*tf_T + ii*tf_T + jj]);
                P[h*tf_T*tf_T + ii*tf_T + jj] = mask[ii*tf_T + jj] ? static_cast<typename CONFIG_T::exp_table_t>(0) : P[h*tf_T*tf_T + ii*tf_T + jj];
            }
        }
    }
}

template<typename CONFIG_T>
void DetermineNewRowSum(
    typename CONFIG_T::row_sum_t new_rowsum[CONFIG_T::n_head*CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::row_sum_t prev_rowsum[CONFIG_T::n_head*CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::exp_table_t prev_exp_tmp[CONFIG_T::n_head*CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::row_sum_t rowsum[CONFIG_T::n_head*CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::exp_table_t P[CONFIG_T::n_head*CONFIG_T::tiling_factor[0]*CONFIG_T::tiling_factor[0]],
    const int tf_T)
{
    for (int h = 0; h < CONFIG_T::n_head; h++) {
        #pragma HLS UNROLL
        for (int ii = 0; ii < tf_T; ii++) {
            #pragma HLS UNROLL
            rowsum[h*tf_T + ii] = 0;
            for (int jj = 0; jj < tf_T; jj++) {
                #pragma HLS UNROLL
                rowsum[h*tf_T + ii] += P[h*tf_T*tf_T + ii*tf_T + jj];
            }
            new_rowsum[h*tf_T + ii] = prev_exp_tmp[h*tf_T + ii] * prev_rowsum[h*tf_T + ii] + rowsum[h*tf_T + ii];
            //std::cout << "new_rowsum[" << h << "][" << ii << "]: " << new_rowsum[h*tf_T + ii] << std::endl;
            //std::cout << "prev_exp_tmp[" << h << "][" << ii << "]: " << prev_exp_tmp[h*tf_T + ii] << std::endl;
            //std::cout << "prev_rowsum[" << h << "][" << ii << "]: " << prev_rowsum[h*tf_T + ii] << std::endl;
            //std::cout << "rowsum[" << h << "][" << ii << "]: " << rowsum[h*tf_T + ii] << std::endl;
        }
    }
}

template<typename CONFIG_T>
void DetermineProductQK(
    typename CONFIG_T::in_proj_out_t Q[CONFIG_T::n_head*CONFIG_T::tiling_factor[0]*CONFIG_T::tiling_factor[2]],
    typename CONFIG_T::in_proj_out_t K[CONFIG_T::n_head*CONFIG_T::tiling_factor[0]*CONFIG_T::tiling_factor[2]],
    typename CONFIG_T::accum_t QK0[CONFIG_T::n_head*CONFIG_T::tiling_factor[0]*CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::accum_t QK1[CONFIG_T::n_head*CONFIG_T::tiling_factor[0]*CONFIG_T::tiling_factor[0]],
    ap_ufixed<18,0,AP_RND_CONV> dk,
    const bool write_QK0[CONFIG_T::n_head*CONFIG_T::tiling_factor[0]*CONFIG_T::tiling_factor[0]],
    const int tf_T,
    const int tf_H)
{
    for (int h = 0; h < CONFIG_T::n_head; h++) { //16dsp
        #pragma HLS UNROLL
        for (int ii = 0; ii < tf_T; ii++) {
            #pragma HLS UNROLL
            for (int ii2 = 0; ii2 < tf_T; ii2++) {
                #pragma HLS UNROLL
                for (int kk = 0; kk < tf_H; kk++) {
                    #pragma HLS UNROLL
                    typename CONFIG_T::accum_t tmp = 0;
                    tmp = Q[h*tf_T*tf_H + ii*tf_H + kk]*K[h*tf_T*tf_H + ii2*tf_H + kk] * dk;
                    if (write_QK0[h*tf_T*tf_T + ii*tf_T + ii2]){
                        QK0[h*tf_T*tf_T + ii*tf_T + ii2] += tmp;
                    } else {
                        QK1[h*tf_T*tf_T + ii*tf_T + ii2] += tmp;
                    }
                }
            }
        }
    }
}

template<typename CONFIG_T>
void InvertWriteQK(
    bool write_QK0[CONFIG_T::n_head*CONFIG_T::tiling_factor[0]*CONFIG_T::tiling_factor[0]],
    const int tf_T)
{
    for (int h = 0; h < CONFIG_T::n_head; h++) {
        #pragma HLS UNROLL
        for (int ii = 0; ii < tf_T; ii++) {
            #pragma HLS UNROLL
            for (int ii2 = 0; ii2 < tf_T; ii2++) {
                #pragma HLS UNROLL
                write_QK0[h*tf_T*tf_T + ii*tf_T + ii2] = !write_QK0[h*tf_T*tf_T + ii*tf_T + ii2];
            }
        }
    }
}

template<typename CONFIG_T>
void DetermineProductPVO(
    typename CONFIG_T::exp_table_t P[CONFIG_T::n_head*CONFIG_T::tiling_factor[0]*CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::in_proj_out_t V[CONFIG_T::n_head*CONFIG_T::tiling_factor[0]*CONFIG_T::tiling_factor[2]],
    typename CONFIG_T::out_proj_in_t O[CONFIG_T::n_head*CONFIG_T::tiling_factor[0]*CONFIG_T::tiling_factor[2]],
    typename CONFIG_T::row_sum_t new_rowsum[CONFIG_T::n_head*CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::exp_table_t prev_exp_tmp[CONFIG_T::n_head*CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::inv_table_t inv_rowsum[CONFIG_T::n_head*CONFIG_T::tiling_factor[0]],
    const int tf_T,
    const int tf_H,
    const int T,
    const int c,
    const int v_idx)
{
    #pragma HLS dependence variable=O type=inter false
    typename CONFIG_T::accum_t O_buf[CONFIG_T::n_head*CONFIG_T::tiling_factor[0]*CONFIG_T::tiling_factor[2]];
    #pragma HLS ARRAY_PARTITION variable=O_buf complete dim=0
    for (int h = 0; h < CONFIG_T::n_head; h++) {//48 dsp
        #pragma HLS UNROLL
        for (int ii = 0; ii < tf_T; ii++) {
            #pragma HLS UNROLL
            for (int kk = 0; kk < tf_H; kk++) {
                #pragma HLS UNROLL
                typename CONFIG_T::accum_t tmp = 0;
                typename CONFIG_T::accum_t tmp2 = 0;
                typename CONFIG_T::accum_t tmp_o = 0;
                for (int ii2 = 0; ii2 < tf_T; ii2++) {
                    #pragma HLS UNROLL
                   tmp += P[h*tf_T*tf_T + ii*tf_T + ii2]*V[h*tf_T*tf_H + ii*tf_H + kk];
                }
                O_buf[h*tf_T*tf_H + ii*tf_H + kk] = (v_idx == 0) ? static_cast<typename CONFIG_T::out_proj_in_t>(0) : O[h*tf_T*tf_H + ii*tf_H + kk];
                O_buf[h*tf_T*tf_H + ii*tf_H + kk] = prev_exp_tmp[h*tf_T + ii]*O_buf[h*tf_T*tf_H + ii*tf_H + kk] + tmp;
            }
        }
    }
    if (c > 0) {
        for (int h = 0; h < CONFIG_T::n_head; h++) {
            #pragma HLS UNROLL
            for (int ii = 0; ii < tf_T; ii++) {
                #pragma HLS UNROLL
                for (int kk = 0; kk < tf_H; kk++) {
                    #pragma HLS UNROLL
                    if (c > 0)  {
                        if (v_idx == T-1){
                            inv_rowsum[h*tf_T + ii] = lookup_inv<CONFIG_T>(new_rowsum[h*tf_T + ii]);
                            O[h*tf_T*tf_H + ii*tf_H + kk] = static_cast<typename CONFIG_T::out_proj_in_t>(O_buf[h*tf_T*tf_H + ii*tf_H + kk])*inv_rowsum[h*tf_T + ii];
                        } else {
                            O[h*tf_T*tf_H + ii*tf_H + kk] = O_buf[h*tf_T*tf_H + ii*tf_H + kk];
                        }
                    }
                    //O[h*tf_T*tf_H + ii*tf_H + kk] = lookup_inv<CONFIG_T>(new_rowsum[h*tf_T + ii])*O_buf[h*tf_T*tf_H + ii*tf_H + kk];
                }
            }
        }
    }
}

template<typename CONFIG_T>
void UpdateRowMaxSum(
    typename CONFIG_T::row_sum_t new_row_sum[CONFIG_T::n_head*CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::row_sum_t prev_row_sum[CONFIG_T::n_head*CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::accum_t new_row_max[CONFIG_T::n_head*CONFIG_T::tiling_factor[0]],
    typename CONFIG_T::accum_t prev_row_max[CONFIG_T::n_head*CONFIG_T::tiling_factor[0]])
{
    for (int i = 0; i < CONFIG_T::n_head; i++) {
        #pragma HLS UNROLL
        for (int ii = 0; ii < CONFIG_T::tiling_factor[0]; ii++) {
            #pragma HLS UNROLL
            prev_row_max[i*CONFIG_T::tiling_factor[0] + ii] = new_row_max[i*CONFIG_T::tiling_factor[0] + ii];
            prev_row_sum[i*CONFIG_T::tiling_factor[0] + ii] = new_row_sum[i*CONFIG_T::tiling_factor[0] + ii];
        }
    }
}


template<class data_T, class res_T, typename CONFIG_T>
void MultiHeadAttention(
    hls::stream<data_T>    &data_qkv,
    hls::stream<res_T>     &res,
    typename CONFIG_T::in_proj_weight_t     in_proj_weight[3*CONFIG_T::n_head*CONFIG_T::embed_dim*CONFIG_T::head_dim], // embed_dim, 3, n_head, head_dim
    typename CONFIG_T::in_proj_bias_t       in_proj_bias[3*CONFIG_T::n_head*CONFIG_T::head_dim],
    typename CONFIG_T::out_proj_weight_t    out_proj_weight[CONFIG_T::n_head*CONFIG_T::head_dim*CONFIG_T::embed_dim],  // n_head, head_dim, embeb_dim
    typename CONFIG_T::out_proj_bias_t      out_proj_bias[CONFIG_T::embed_dim],
    typename CONFIG_T::mask_t               mask[CONFIG_T::seq_len*CONFIG_T::seq_len])
{
    const unsigned int tf_T = CONFIG_T::tiling_factor[0];
    const unsigned int tf_N = CONFIG_T::tiling_factor[1];
    const unsigned int tf_H = CONFIG_T::tiling_factor[2];
    const unsigned int T = CONFIG_T::seq_len/tf_T;
    const unsigned int N = CONFIG_T::embed_dim/tf_N;
    const unsigned int H = CONFIG_T::head_dim/tf_H;
    #pragma HLS ARRAY_RESHAPE variable=in_proj_weight   cyclic factor=3*CONFIG_T::n_head*tf_H*tf_N dim=1
    #pragma HLS ARRAY_RESHAPE variable=in_proj_bias     cyclic factor=3*CONFIG_T::n_head*tf_H dim=1
    #pragma HLS ARRAY_RESHAPE variable=out_proj_weight  cyclic factor=CONFIG_T::n_head*tf_H*tf_N dim=1
    #pragma HLS ARRAY_RESHAPE variable=out_proj_bias    cyclic factor=CONFIG_T::n_head*tf_N dim=1
    #pragma HLS ARRAY_PARTITION variable=mask complete dim=0
    typename CONFIG_T::in_proj_out_t K[CONFIG_T::seq_len*CONFIG_T::n_head*CONFIG_T::head_dim];
    typename CONFIG_T::in_proj_out_t V[CONFIG_T::seq_len*CONFIG_T::n_head*CONFIG_T::head_dim];
    typename CONFIG_T::in_proj_out_t Q[CONFIG_T::seq_len*CONFIG_T::n_head*CONFIG_T::head_dim];
    typename CONFIG_T::out_proj_in_t O[CONFIG_T::seq_len*CONFIG_T::n_head*CONFIG_T::head_dim];
    typename data_T::value_type row_buffer[CONFIG_T::embed_dim*tf_T];
    #pragma HLS ARRAY_RESHAPE variable=K cyclic factor=CONFIG_T::n_head*tf_H*tf_T dim=1
    #pragma HLS ARRAY_RESHAPE variable=V cyclic factor=CONFIG_T::n_head*tf_H*tf_T dim=1
    #pragma HLS ARRAY_RESHAPE variable=Q cyclic factor=CONFIG_T::n_head*tf_H*tf_T dim=1
    #pragma HLS ARRAY_RESHAPE variable=O cyclic factor=CONFIG_T::n_head*tf_H*tf_T dim=1
    #pragma HLS ARRAY_RESHAPE variable=row_buffer cyclic factor=tf_H*tf_T dim=1
    typename CONFIG_T::accum_t QK0[CONFIG_T::n_head*tf_T*tf_T];
    typename CONFIG_T::accum_t QK1[CONFIG_T::n_head*tf_T*tf_T];
    typename CONFIG_T::accum_t QK[CONFIG_T::n_head*tf_T*tf_T];
    typename CONFIG_T::exp_table_t P[CONFIG_T::n_head*tf_T*tf_T];
    typename CONFIG_T::row_sum_t rowsum[CONFIG_T::n_head*tf_T];
    typename CONFIG_T::row_sum_t new_rowsum[CONFIG_T::n_head*tf_T];
    typename CONFIG_T::row_sum_t prev_rowsum[CONFIG_T::n_head*tf_T];
    typename CONFIG_T::accum_t rowmax[CONFIG_T::n_head*tf_T];
    typename CONFIG_T::accum_t prev_rowmax[CONFIG_T::n_head*tf_T];
    typename CONFIG_T::accum_t new_rowmax[CONFIG_T::n_head*tf_T];

    #pragma HLS ARRAY_PARTITION variable=QK0 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=QK1 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=QK complete dim=0
    #pragma HLS ARRAY_PARTITION variable=P complete dim=0
    #pragma HLS ARRAY_PARTITION variable=rowsum complete dim=0
    #pragma HLS ARRAY_PARTITION variable=new_rowsum complete dim=0
    #pragma HLS ARRAY_PARTITION variable=new_rowmax complete dim=0
    #pragma HLS ARRAY_PARTITION variable=prev_rowmax complete dim=0
    #pragma HLS ARRAY_PARTITION variable=prev_rowsum complete dim=0
    #pragma HLS ARRAY_PARTITION variable=rowmax complete dim=0

    #pragma HLS DATAFLOW

    data_T data_pack;
    const ap_ufixed<18,0,AP_RND_CONV> dk = 1.0/sqrt(CONFIG_T::head_dim);
    typename CONFIG_T::accum_t tmp_k[CONFIG_T::n_head*tf_H];
    typename CONFIG_T::accum_t tmp_v[CONFIG_T::n_head*tf_H];
    typename CONFIG_T::accum_t tmp_q[CONFIG_T::n_head*tf_H];
    #pragma HLS ARRAY_PARTITION variable=tmp_k complete dim=0
    #pragma HLS ARRAY_PARTITION variable=tmp_v complete dim=0
    #pragma HLS ARRAY_PARTITION variable=tmp_q complete dim=0
    int in_proj_weight_offset = 0;
    int in_proj_bias_offset = 0;
    int in_proj_input_offset = 0;
    int in_proj_output_offset = 0;
    typename CONFIG_T::accum_t tmp_qk_debug[CONFIG_T::head_dim];
    compute_KVQ:  
    for (int i = 0; i < T; i++) {
        for (int k = 0; k < H; k++) {
            for (int j = 0; j < N; j++) {
                #pragma HLS PIPELINE II = 1
                in_proj_weight_offset = (j*H + k)*tf_H*tf_N*CONFIG_T::n_head*3; //(embed_dim/tf, head_dim/tf, 3, n_head, tf_H, tf_N)
                in_proj_bias_offset = k*tf_H*CONFIG_T::n_head*3;
                in_proj_input_offset = j*tf_T*tf_H;
                in_proj_output_offset = (i*H + k)*tf_H*tf_T*CONFIG_T::n_head;
                if (k==0){
                    data_pack = data_qkv.read();
                }
                for (int h = 0; h < CONFIG_T::n_head; h++) {//48dsp
                    #pragma HLS UNROLL
                    for (int ii = 0; ii < tf_T; ii++) {
                        #pragma HLS UNROLL
                        for (int kk = 0; kk < tf_H; kk++) {
                            #pragma HLS UNROLL
                            for (int jj = 0; jj < tf_N; jj++) {
                                #pragma HLS UNROLL
                                if (h==0 && k==0 && kk==0){
                                    row_buffer[in_proj_input_offset + ii*tf_H + kk] = data_pack[jj*tf_T + ii];
                                }
                                if (j==0 && jj==0){
                                    tmp_k[h*tf_H + kk] = in_proj_bias[in_proj_bias_offset + 1*CONFIG_T::n_head*tf_H + h*tf_H + kk];
                                    tmp_v[h*tf_H + kk] = in_proj_bias[in_proj_bias_offset + 2*CONFIG_T::n_head*tf_H + h*tf_H + kk];
                                    tmp_q[h*tf_H + kk] = in_proj_bias[in_proj_bias_offset + 0*CONFIG_T::n_head*tf_H + h*tf_H + kk];
                                }
                                tmp_k[h*tf_H + kk] += row_buffer[in_proj_input_offset + ii*tf_H + jj]*in_proj_weight[in_proj_weight_offset + 1*CONFIG_T::n_head*tf_H*tf_N + h*tf_H*tf_N + jj*tf_H + kk];
                                tmp_v[h*tf_H + kk] += row_buffer[in_proj_input_offset + ii*tf_H + jj]*in_proj_weight[in_proj_weight_offset + 2*CONFIG_T::n_head*tf_H*tf_N + h*tf_H*tf_N + jj*tf_H + kk];
                                tmp_q[h*tf_H + kk] += row_buffer[in_proj_input_offset + ii*tf_H + jj]*in_proj_weight[in_proj_weight_offset + 0*CONFIG_T::n_head*tf_H*tf_N + h*tf_H*tf_N + jj*tf_H + kk];
                                //if (h==0 && i==36 && k==25){
                                //    std::cout << "tmp_qk_debug: " << std::fixed << std::setprecision(20) << tmp_q[h*tf_H + kk] << " " << row_buffer[in_proj_input_offset]*in_proj_weight[in_proj_weight_offset] << std::endl;
                                //}
                                //if (h==0 && i==36 && k==25 && j==N-1){
                                //    std::cout << "tmp_qk_debug_final: " << std::fixed << std::setprecision(20) << static_cast<typename CONFIG_T::in_proj_out_t>(tmp_q[h*tf_H + kk]) << std::endl;
                                //
                                //}
                            }
                            //if (j==N-1){
                            //    K[in_proj_output_offset + h*tf_T*tf_H + ii*tf_H + kk] = tmp_k[h*tf_H + kk];
                            //    V[in_proj_output_offset + h*tf_T*tf_H + ii*tf_H + kk] = tmp_v[h*tf_H + kk];
                            //    Q[in_proj_output_offset + h*tf_T*tf_H + ii*tf_H + kk] = tmp_q[h*tf_H + kk];
                            //}
                        }
                    }
                }
                if (j == N - 1) {
                    for (int h = 0; h < CONFIG_T::n_head; h++) {
                        for (int ii = 0; ii < tf_T; ii++) {
                            for (int kk = 0; kk < tf_H; kk++) {
                                #pragma HLS UNROLL
                                K[in_proj_output_offset + h*tf_T*tf_H + ii*tf_H + kk] = tmp_k[h*tf_H + kk];
                                V[in_proj_output_offset + h*tf_T*tf_H + ii*tf_H + kk] = tmp_v[h*tf_H + kk];
                                Q[in_proj_output_offset + h*tf_T*tf_H + ii*tf_H + kk] = tmp_q[h*tf_H + kk];
                            }
                        }
                    }
                }
            }           
        }
    }
    //print K
    //std::cout << "K" << std::endl;
    //for (int t = 0; t < T; t++) {
    //    for (int tt = 0; tt < tf_T; tt++) {
    //        for (int i = 0; i < CONFIG_T::n_head; i++) {
    //            for (int h = 0; h < H; h++) {
    //                for (int hh = 0; hh < tf_H; hh++) {
    //                    std::cout << K[(t*H + h)*tf_H*tf_T*CONFIG_T::n_head + i*tf_T*tf_H + tt*tf_H + hh] << " ";
    //                }
    //            }
    //        }
    //        std::cout << std::endl;
    //    }
    //}


    typename CONFIG_T::exp_table_t prev_exp_tmp[CONFIG_T::n_head*tf_T];
    typename CONFIG_T::exp_table_t exp_tmp[CONFIG_T::n_head*tf_T];
    typename CONFIG_T::inv_table_t inv_rowsum[CONFIG_T::n_head*tf_T];
    bool write_QK0[CONFIG_T::n_head*tf_T*tf_T];
    #pragma HLS ARRAY_PARTITION variable=inv_rowsum complete dim=0
    #pragma HLS ARRAY_PARTITION variable=prev_exp_tmp complete dim=0
    #pragma HLS ARRAY_PARTITION variable=write_QK0 complete dim=0
    int q_idx = 0;
    int k_idx = 0;
    int v_idx = 0;
    int o_idx = 0;
    //typename data_T::value_type QK_debug[CONFIG_T::n_head][T][T][tf_T][tf_T];
    //typename CONFIG_T::accum_t row_sum_debug[CONFIG_T::n_head][T][tf_T];
    for (int i = 0; i < CONFIG_T::n_head; i++) {
        for (int ii = 0; ii < tf_T; ii++) {
            for (int jj = 0; jj < tf_T; jj++) {
                write_QK0[i*tf_T*tf_T + ii*tf_T + jj] = true;
            }
        }
    } 
    typename CONFIG_T::exp_table_t exp_weight_debug[CONFIG_T::n_head][T][T];
    typename CONFIG_T::accum_t attn_weight_debug[CONFIG_T::n_head][T][T];
    typename CONFIG_T::accum_t row_sum_debug[CONFIG_T::n_head][T][T];
    typename CONFIG_T::accum_t row_max_debug[CONFIG_T::n_head][T][T];
    compute_attn_pipeline:   
    for (int c = 0; c < T*T+1; c++){
        for (int hd_idx = 0; hd_idx < H; hd_idx++) { //hd_idx is the index of the head dimension
            #pragma HLS PIPELINE II = 1
            if (hd_idx == 0) { //compute rowmax and rowsum after finishing each attention tile
                int mask_offset = (o_idx*T + v_idx)*tf_T*tf_T; // get next mask tile
                if (v_idx == 0) { 
                    //print prev_rowsum, prev_rowmax
                    //std::cout << "prev_rowsum" << std::endl;
                    //for (int i = 0; i < CONFIG_T::n_head; i++) {
                    //    for (int ii = 0; ii < tf_T; ii++) {
                    //        std::cout << prev_rowsum[i*tf_T + ii] << " ";
                    //    }
                    //    std::cout << std::endl;
                    //}
                    //std::cout << "prev_rowmax" << std::endl;
                    //for (int i = 0; i < CONFIG_T::n_head; i++) {
                    //    for (int ii = 0; ii < tf_T; ii++) {
                    //        std::cout << prev_rowmax[i*tf_T + ii] << " ";
                    //    }
                    //    std::cout << std::endl;
                    //}
                    InitRowMaxSum<CONFIG_T>(prev_rowsum, prev_rowmax, tf_T); 
                } //Before writing O at the begining, initialize rowmax and rowsum
                //std::cout << "dk: " << 2 << std::endl;
                ComputeCurrentTileRowMax<CONFIG_T>(rowmax, mask + mask_offset, QK0, QK1, write_QK0, tf_T, T, c);
                //std::cout << "dk: " << 3 << std::endl;
                if (c > 0) {
                    DetermineNewRowMax<CONFIG_T>(new_rowmax, prev_rowmax, rowmax, tf_T);
                    DetermineExpRowMaxDiff<CONFIG_T>(new_rowmax, prev_rowmax, prev_exp_tmp, tf_T);
                    DetermineAttnWeight<CONFIG_T>(P, QK0, QK1, QK, new_rowmax, mask + mask_offset, write_QK0, tf_T);
                    for (int h=0; h < CONFIG_T::n_head; h++) {
                        for (int ii = 0; ii < tf_T; ii++) {
                            for (int jj = 0; jj < tf_T; jj++) {
                                attn_weight_debug[h][o_idx][v_idx] = QK[h*tf_T*tf_T + ii*tf_T + jj];
                                exp_weight_debug[h][o_idx][v_idx] = P[h*tf_T*tf_T + ii*tf_T + jj];
                            }
                        }
                    }
                    //attn_weight_debug[0][o_idx][v_idx] = QK[0];
                    //exp_weight_debug[0][o_idx][v_idx] = P[0];
                    DetermineNewRowSum<CONFIG_T>(new_rowsum, prev_rowsum, prev_exp_tmp, rowsum, P, tf_T);
                    for (int h=0; h < CONFIG_T::n_head; h++) {
                        row_sum_debug[h][o_idx][v_idx] = new_rowsum[h];
                        row_max_debug[h][o_idx][v_idx] = new_rowmax[h];
                    }
                }
            }
            int q_offset = (q_idx*H + hd_idx)*CONFIG_T::n_head*tf_H*tf_T;
            int k_offset = (k_idx*H + hd_idx)*CONFIG_T::n_head*tf_H*tf_T;
            int o_offset = (o_idx*H + hd_idx)*CONFIG_T::n_head*tf_H*tf_T;
            int v_offset = (v_idx*H + hd_idx)*CONFIG_T::n_head*tf_H*tf_T;
            if (c < T*T) {
                DetermineProductQK<CONFIG_T>(Q + q_offset, K + k_offset, QK0, QK1, dk, write_QK0, tf_T, tf_H);
            }
            //std::cout << "dk: " << 7 << std::endl;
            if (hd_idx == H-1) {InvertWriteQK<CONFIG_T>(write_QK0, tf_T);}
            //std::cout << "dk: " << 8 << std::endl;
            if (c > 0) {
                DetermineProductPVO<CONFIG_T>(P, V + v_offset, O + o_offset, new_rowsum, prev_exp_tmp, inv_rowsum, tf_T, tf_H, T, c, v_idx);
            }
            //std::cout << "dk: " << 9 << std::endl;
            UpdateRowMaxSum<CONFIG_T>(new_rowsum, prev_rowsum, new_rowmax, prev_rowmax);
            //std::cout << "dk: " << 10 << std::endl;
            if (hd_idx == H-1) {
                v_idx = k_idx;
                o_idx = q_idx;
                if (c < T*T) { 
                    k_idx++;
                    if (k_idx == T){
                        k_idx = 0;
                        q_idx++;
                    }
                }
            }
        }   
    }

    typename CONFIG_T::accum_t tile_buffer[tf_T*tf_N];
    typename res_T::value_type res_debug[T][N][tf_T][tf_N];
    res_T res_pack;
    int out_proj_weight_offset = 0;
    int out_proj_bias_offset = 0;
    int out_proj_input_offset = 0;
    int out_proj_output_offset = 0;
    #pragma HLS ARRAY_PARTITION variable=tile_buffer complete dim=0                       
    compute_output: 
    for (int i = 0; i < T; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < H; k++) {
                #pragma HLS PIPELINE II = 1
                out_proj_weight_offset = (k*N + j)*tf_H*tf_N*CONFIG_T::n_head; //(embed_dim/tf, head_dim/tf, n_head, tf_H, tf_N)
                out_proj_bias_offset = j*tf_N;
                out_proj_input_offset = (i*H + k)*tf_H*tf_T*CONFIG_T::n_head;
                //out_proj_output_offset = (i*H + k)*tf_H*tf_T*CONFIG_T::n_head;
                for (int ii = 0; ii < tf_T; ii++) {
                    #pragma HLS UNROLL
                    for (int jj = 0; jj < tf_N; jj++) {
                        #pragma HLS UNROLL
                        if (k==0){
                            tile_buffer[ii*tf_N + jj] = out_proj_bias[out_proj_bias_offset + jj];
                        } 
                        for (int kk = 0; kk < tf_H; kk++) {
                            #pragma HLS UNROLL
                            for (int h = 0; h < CONFIG_T::n_head; h++) {//16dsp
                                #pragma HLS UNROLL
                                tile_buffer[ii*tf_N + jj] += O[out_proj_input_offset + h*tf_T*tf_H + ii*tf_H + kk]*out_proj_weight[out_proj_weight_offset + h*tf_H*tf_N + kk*tf_N + jj];
                            }
                        }
                        res_pack[ii*tf_N + jj] = tile_buffer[ii*tf_N + jj];
                        res_debug[i][j][ii][jj] = res_pack[ii*tf_N + jj];
                    }
                }
                if (k==H-1){
                    res.write(res_pack);
                }
            }
        }
    }
    //save O
    std::ofstream O_debug_file;
    O_debug_file.open("O_debug.txt", std::ios_base::app);
    O_debug_file << std::fixed << std::setprecision(15);
    for (int i = 0; i < CONFIG_T::n_head; i++) {
        for (int t = 0; t < T; t++) {
            for (int tt = 0; tt < tf_T; tt++) {
                for (int h = 0; h < H; h++) {
                    for (int hh = 0; hh < tf_H; hh++) {
                        if (h == H-1 && hh == tf_H-1) {
                            O_debug_file << O[(t*H + h)*tf_H*tf_T*CONFIG_T::n_head + i*tf_T*tf_H + tt*tf_H + hh];
                        } else {
                            O_debug_file << O[(t*H + h)*tf_H*tf_T*CONFIG_T::n_head + i*tf_T*tf_H + tt*tf_H + hh] << " ";
                        }
                    }
                }
                O_debug_file << std::endl;
            }
        }
    }
    O_debug_file << std::endl;
    O_debug_file.close();
    //save row_sum_debug
    std::ofstream row_sum_debug_file;
    row_sum_debug_file.open("row_sum_debug.txt", std::ios_base::app);
    row_sum_debug_file << std::fixed << std::setprecision(15);
    for (int h = 0; h < CONFIG_T::n_head; h++) {
        for (int i = 0; i < T; i++) {
            for (int j = 0; j < T; j++) {
                if (j == T-1) {
                    row_sum_debug_file << row_sum_debug[h][i][j];
                } else {
                    row_sum_debug_file << row_sum_debug[h][i][j] << " ";
                }
            }
            row_sum_debug_file << std::endl;
        }
    }
    row_sum_debug_file << std::endl;
    row_sum_debug_file.close();
    //save row_max_debug
    std::ofstream row_max_debug_file;
    row_max_debug_file.open("row_max_debug.txt", std::ios_base::app);
    row_max_debug_file << std::fixed << std::setprecision(15);
    for (int h = 0; h < CONFIG_T::n_head; h++) {
        for (int i = 0; i < T; i++) {
            for (int j = 0; j < T; j++) {
                if (j == T-1) {
                    row_max_debug_file << row_max_debug[h][i][j];
                } else {
                    row_max_debug_file << row_max_debug[h][i][j] << " ";
                }
            }
            row_max_debug_file << std::endl;
        }
    }
    row_max_debug_file << std::endl;
    row_max_debug_file.close();
    //save K
    std::ofstream K_debug_file;
    K_debug_file.open("K_debug.txt", std::ios_base::app);
    K_debug_file << std::fixed << std::setprecision(15);
    for (int i = 0; i < CONFIG_T::n_head; i++) {
        for (int t = 0; t < T; t++) {
            for (int tt = 0; tt < tf_T; tt++) {
                for (int h = 0; h < H; h++) {
                    for (int hh = 0; hh < tf_H; hh++) {
                        if (h == H-1 && hh == tf_H-1) {
                            K_debug_file << K[(t*H + h)*tf_H*tf_T*CONFIG_T::n_head + i*tf_T*tf_H + tt*tf_H + hh];
                        } else {
                            K_debug_file << K[(t*H + h)*tf_H*tf_T*CONFIG_T::n_head + i*tf_T*tf_H + tt*tf_H + hh] << " ";
                        }
                    }
                }
                K_debug_file << std::endl;
            }
        }
    }
    K_debug_file << std::endl;
    K_debug_file.close();

    //save Q
    std::ofstream Q_debug_file;
    Q_debug_file.open("Q_debug.txt", std::ios_base::app);
    Q_debug_file << std::fixed << std::setprecision(15);
    for (int i = 0; i < CONFIG_T::n_head; i++) {
        for (int t = 0; t < T; t++) {
            for (int tt = 0; tt < tf_T; tt++) {
                for (int h = 0; h < H; h++) {
                    for (int hh = 0; hh < tf_H; hh++) {
                        if (h == H-1 && hh == tf_H-1) {
                            Q_debug_file << Q[(t*H + h)*tf_H*tf_T*CONFIG_T::n_head + i*tf_T*tf_H + tt*tf_H + hh];
                        } else {
                            Q_debug_file << Q[(t*H + h)*tf_H*tf_T*CONFIG_T::n_head + i*tf_T*tf_H + tt*tf_H + hh] << " ";
                        }
                    }
                }
                Q_debug_file << std::endl;
            }
        }
    }
    Q_debug_file << std::endl;
    Q_debug_file.close();
    //save V
    std::ofstream V_debug_file;
    V_debug_file.open("V_debug.txt", std::ios_base::app);
    V_debug_file << std::fixed << std::setprecision(15);
    for (int i = 0; i < CONFIG_T::n_head; i++) {
        for (int t = 0; t < T; t++) {
            for (int tt = 0; tt < tf_T; tt++) {
                for (int h = 0; h < H; h++) {
                    for (int hh = 0; hh < tf_H; hh++) {
                        if (h == H-1 && hh == tf_H-1) {
                            V_debug_file << V[(t*H + h)*tf_H*tf_T*CONFIG_T::n_head + i*tf_T*tf_H + tt*tf_H + hh];
                        } else {
                            V_debug_file << V[(t*H + h)*tf_H*tf_T*CONFIG_T::n_head + i*tf_T*tf_H + tt*tf_H + hh] << " ";
                        }
                    }
                }
                V_debug_file << std::endl;
            }
        }
    }
    V_debug_file << std::endl;
    V_debug_file.close();
    //save attn_weight_debug
    std::ofstream attn_weight_debug_file;
    attn_weight_debug_file.open("attn_weight_debug.txt", std::ios_base::app);
    attn_weight_debug_file << std::fixed << std::setprecision(15);
    for (int h = 0; h < CONFIG_T::n_head; h++) {
        for (int i = 0; i < T; i++) {
            for (int j = 0; j < T; j++) {
                if (j == T-1) {
                    attn_weight_debug_file << attn_weight_debug[h][i][j];
                } else {
                    attn_weight_debug_file << attn_weight_debug[h][i][j] << " ";
                }
            }
            attn_weight_debug_file << std::endl;
        }
    }
    attn_weight_debug_file << std::endl;
    attn_weight_debug_file.close();
    //save exp_weight_debug
    std::ofstream exp_weight_debug_file;
    exp_weight_debug_file.open("exp_weight_debug.txt", std::ios_base::app);
    exp_weight_debug_file << std::fixed << std::setprecision(15);
    for (int h = 0; h < CONFIG_T::n_head; h++) {
        for (int i = 0; i < T; i++) {
            for (int j = 0; j < T; j++) {
                if (j == T-1) {
                    exp_weight_debug_file << exp_weight_debug[h][i][j];
                } else {
                    exp_weight_debug_file << exp_weight_debug[h][i][j] << " ";
                }
            }
            exp_weight_debug_file << std::endl;
        }
    }
    exp_weight_debug_file << std::endl;
    exp_weight_debug_file.close();
    //save res_bug
    std::ofstream mha_out_debug_file;
    mha_out_debug_file.open("mha_out_debug.txt", std::ios_base::app);
    mha_out_debug_file << std::fixed << std::setprecision(15);
    for (int i = 0; i < T; i++) {
        for (int ii = 0; ii < tf_T; ii++) {
            for (int j = 0; j < N; j++) {
                for (int jj = 0; jj < tf_N; jj++) {
                    if (j == N-1 && jj == tf_N-1) {
                        mha_out_debug_file << res_debug[i][j][ii][jj];
                    } else {
                        mha_out_debug_file << res_debug[i][j][ii][jj] << " ";
                    }
                }
            }
            mha_out_debug_file << std::endl;
        }
    }
    mha_out_debug_file << std::endl;
    mha_out_debug_file.close();

    //std::cout << "res_debug" << std::endl;
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
    //std::cout << "attn_weight_debug" << std::endl;
    //for (int i = 0; i < T; i++) {
    //    for (int j = 0; j < T; j++) {
    //        std::cout << attn_weight_debug[0][i][j] << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << "exp_weight_debug" << std::endl;
    //for (int i = 0; i < T; i++) {
    //    for (int j = 0; j < T; j++) {
    //        std::cout << exp_weight_debug[0][i][j] << " ";
    //    }
    //    std::cout << std::endl;
    //}
    //write_output:
    //for (int i = 0; i < T; i++) {
    //    for (int j = 0; j < N; j++) {
    //        #pragma HLS PIPELINE II = 1
    //        for (int ii = 0; ii < tf_T; ii++) {
    //            #pragma HLS UNROLL
    //            for (int jj = 0; jj < CONFIG_T::tiling_factor[1]; jj++) {
    //                #pragma HLS UNROLL
    //                res_pack[ii*CONFIG_T::tiling_factor[1] + jj] = M[i][j][ii][jj];
    //                if (jj==CONFIG_T::tiling_factor[1]-1 && ii==tf_T-1){
    //                    res.write(res_pack);
    //                }
    //            }
    //        }
    //    }
    //}
    
}

/*
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
