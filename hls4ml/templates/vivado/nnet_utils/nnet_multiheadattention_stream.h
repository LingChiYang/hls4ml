#ifndef NNET_MHT_SS_H_
#define NNET_MHT_SS_H_

#include "nnet_common.h"
#include "nnet_mult.h"
#include "hls_stream.h"
#include <iostream>
#include <math.h>
#include "nnet_helpers.h"
#include "hls_streamofblocks.h"
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
        double in_val = 2 * double(CONFIG_T::exp_range) * (ii - double(CONFIG_T::exp_table_size) / 2.0) / double(CONFIG_T::exp_table_size);
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
        double in_val = double(CONFIG_T::inv_range) * ii / double(CONFIG_T::inv_table_size);
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
    typename CONFIG_T::out_proj_in_t O[CONFIG_T::seq_len*CONFIG_T::n_head*CONFIG_T::head_dim];
    typename data_T::value_type row_buffer[CONFIG_T::embed_dim*tf_T];
    typename CONFIG_T::in_proj_out_t K[CONFIG_T::seq_len*CONFIG_T::n_head*CONFIG_T::head_dim];
    typename CONFIG_T::in_proj_out_t V[CONFIG_T::seq_len*CONFIG_T::n_head*CONFIG_T::head_dim];
    typename CONFIG_T::in_proj_out_t Q[CONFIG_T::seq_len*CONFIG_T::n_head*CONFIG_T::head_dim];
    #pragma HLS ARRAY_RESHAPE variable=K cyclic factor=CONFIG_T::n_head*tf_H*tf_T dim=1
    #pragma HLS ARRAY_RESHAPE variable=V cyclic factor=CONFIG_T::n_head*tf_H*tf_T dim=1
    #pragma HLS ARRAY_RESHAPE variable=Q cyclic factor=CONFIG_T::n_head*tf_H*tf_T dim=1
    #pragma HLS ARRAY_RESHAPE variable=O cyclic factor=CONFIG_T::n_head*tf_H*tf_T dim=1
    #pragma HLS ARRAY_RESHAPE variable=row_buffer cyclic factor=tf_N*tf_T dim=1

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
    typename CONFIG_T::accum_t data_debug[T][N];
    compute_KVQ:  
    for (int i = 0; i < T; i++) {
        for (int k = 0; k < H; k++) {
            for (int j = 0; j < N; j++) {
                #pragma HLS PIPELINE II = 1
                in_proj_weight_offset = (j*H + k)*tf_H*tf_N*CONFIG_T::n_head*3; //(embed_dim/tf, head_dim/tf, 3, n_head, tf_H, tf_N)
                in_proj_bias_offset = k*tf_H*CONFIG_T::n_head*3;
                in_proj_input_offset = j*tf_T*tf_N;
                in_proj_output_offset = (i*H + k)*tf_H*tf_T*CONFIG_T::n_head;
                if (k==0){
                    data_pack = data_qkv.read();
                    //data_debug[i][j] = data_pack[0];
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
                                    row_buffer[in_proj_input_offset + ii*tf_N + jj] = data_pack[ii*tf_N + jj];
                                }
                                if (j==0 && jj==0){
                                    tmp_k[h*tf_H + kk] = in_proj_bias[in_proj_bias_offset + 1*CONFIG_T::n_head*tf_H + h*tf_H + kk];
                                    tmp_v[h*tf_H + kk] = in_proj_bias[in_proj_bias_offset + 2*CONFIG_T::n_head*tf_H + h*tf_H + kk];
                                    tmp_q[h*tf_H + kk] = in_proj_bias[in_proj_bias_offset + 0*CONFIG_T::n_head*tf_H + h*tf_H + kk];
                                }
                                tmp_k[h*tf_H + kk] += row_buffer[in_proj_input_offset + ii*tf_N + jj]*in_proj_weight[in_proj_weight_offset + 1*CONFIG_T::n_head*tf_H*tf_N + h*tf_H*tf_N + jj*tf_H + kk];
                                tmp_v[h*tf_H + kk] += row_buffer[in_proj_input_offset + ii*tf_N + jj]*in_proj_weight[in_proj_weight_offset + 2*CONFIG_T::n_head*tf_H*tf_N + h*tf_H*tf_N + jj*tf_H + kk];
                                tmp_q[h*tf_H + kk] += row_buffer[in_proj_input_offset + ii*tf_N + jj]*in_proj_weight[in_proj_weight_offset + 0*CONFIG_T::n_head*tf_H*tf_N + h*tf_H*tf_N + jj*tf_H + kk];
                            }
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

    // Add this code to write Q, K, and V to files
    // std::ofstream Q_file("Q.txt", std::ios::app);
    // std::ofstream K_file("K.txt", std::ios::app);
    // std::ofstream V_file("V.txt", std::ios::app);

    // for (int i = 0; i < CONFIG_T::seq_len * CONFIG_T::n_head * CONFIG_T::head_dim; i++) {
    //     Q_file << Q[i] << " ";
    //     K_file << K[i] << " ";
    //     V_file << V[i] << " ";

    //     if ((i + 1) % (CONFIG_T::n_head * CONFIG_T::head_dim) == 0) {
    //         Q_file << "\n";
    //         K_file << "\n";
    //         V_file << "\n";
    //     }
    // }

    // Q_file.close();
    // K_file.close();
    // V_file.close();


    typename CONFIG_T::exp_table_t prev_exp_tmp[CONFIG_T::n_head * tf_T];
    typename CONFIG_T::exp_table_t exp_tmp[CONFIG_T::n_head * tf_T];
    typename CONFIG_T::inv_table_t inv_rowsum[CONFIG_T::n_head * tf_T];
    typename CONFIG_T::exp_table_t P[CONFIG_T::n_head * tf_T * tf_T];
    typename CONFIG_T::row_sum_t rowsum[CONFIG_T::n_head * tf_T];
    typename CONFIG_T::row_sum_t new_rowsum[CONFIG_T::n_head * tf_T];
    typename CONFIG_T::row_sum_t prev_rowsum[CONFIG_T::n_head * tf_T];
    typename CONFIG_T::accum_t rowmax[CONFIG_T::n_head * tf_T];
    typename CONFIG_T::accum_t prev_rowmax[CONFIG_T::n_head * tf_T];
    typename CONFIG_T::accum_t new_rowmax[CONFIG_T::n_head * tf_T];
    typename CONFIG_T::accum_t QK[CONFIG_T::n_head * tf_T * tf_T];
    #pragma HLS ARRAY_PARTITION variable=inv_rowsum     complete dim=0
    #pragma HLS ARRAY_PARTITION variable=prev_exp_tmp   complete dim=0
    #pragma HLS ARRAY_PARTITION variable=P              complete dim=0
    #pragma HLS ARRAY_PARTITION variable=rowsum         complete dim=0
    #pragma HLS ARRAY_PARTITION variable=new_rowsum     complete dim=0
    #pragma HLS ARRAY_PARTITION variable=new_rowmax     complete dim=0
    #pragma HLS ARRAY_PARTITION variable=prev_rowmax    complete dim=0
    #pragma HLS ARRAY_PARTITION variable=prev_rowsum    complete dim=0
    #pragma HLS ARRAY_PARTITION variable=rowmax         complete dim=0
    #pragma HLS ARRAY_PARTITION variable=QK             complete dim=0
    int q_idx = 0;
    int k_idx = 0;
    int v_idx = 0;
    int o_idx = 0;
    //define P stream with vector
    typedef array<typename CONFIG_T::exp_table_t, CONFIG_T::n_head * tf_T * (tf_T+1)> P_exp_aggr_strm_type;
    hls::stream<P_exp_aggr_strm_type> P_exp_aggr_strm;
    #pragma HLS aggregate variable=P_exp_aggr_strm compact=bit
    #pragma HLS STREAM variable=P_exp_aggr_strm depth=1024
    P_exp_aggr_strm_type P_exp_aggr_block;
    QK_SCALE_DOT_PRODUCT:
    for (int i = 0; i < T; i++) {
        for (int j = 0; j < T; j++) {
            for (int hd = 0; hd < H; hd++) {
                #pragma HLS PIPELINE II = 1
                int q_offset = (i*H + hd)*CONFIG_T::n_head*tf_H*tf_T;
                int k_offset = (j*H + hd)*CONFIG_T::n_head*tf_H*tf_T;
                QK_PRODUCT:
                    for (int h = 0; h < CONFIG_T::n_head; h++) {
                        for (int ii = 0; ii < tf_T; ii++) {
                            for (int jj = 0; jj < tf_T; jj++) {
                                for (int kk = 0; kk < tf_H; kk++) {
                                    if (hd == 0 && kk == 0){
                                        QK[h*tf_T*tf_T + ii*tf_T + jj] = 0;
                                    }
                                    QK[h*tf_T*tf_T + ii*tf_T + jj] += Q[q_offset + h*tf_H*tf_T + ii*tf_H + kk] * K[k_offset + h*tf_H*tf_T + jj*tf_H + kk] * dk;
                                }
                            }
                        }
                    }
                if (j == 0 && hd == 0){
                    INIT_MAX:
                        for (int h = 0; h < CONFIG_T::n_head; h++) {
                            for (int ii = 0; ii < tf_T; ii++) {
                                prev_rowmax[h*tf_T + ii] = -8;
                                rowmax[h*tf_T + ii] = -8;
                            }
                        }
                }
                if (hd == H-1){
                    GET_LOACL_MAX:
                        for (int h = 0; h < CONFIG_T::n_head; h++) {
                            for (int ii = 0; ii < tf_T; ii++) {
                                rowmax[h*tf_T + ii] = -8;
                                for (int jj = 0; jj < tf_T; jj++) {
                                    rowmax[h*tf_T + ii] = (rowmax[h*tf_T + ii] > QK[h*tf_T*tf_T + ii*tf_T + jj]) ? rowmax[h*tf_T + ii] 
                                                        : (mask[i*tf_T*tf_T*T + j*tf_T*tf_T + ii*tf_T + jj]) ? static_cast<typename CONFIG_T::accum_t>(-8) : QK[h*tf_T*tf_T + ii*tf_T + jj];
                                }
                            }
                        }
                    GET_GLOBAL_MAX:
                        for (int h = 0; h < CONFIG_T::n_head; h++) {
                            for (int ii = 0; ii < tf_T; ii++) {
                                new_rowmax[h*tf_T + ii] = (prev_rowmax[h*tf_T + ii] > rowmax[h*tf_T + ii]) ? prev_rowmax[h*tf_T + ii] : rowmax[h*tf_T + ii];
                            }
                        }
                    GET_DIFF_MAX:
                        for (int h = 0; h < CONFIG_T::n_head; h++) {
                            for (int ii = 0; ii < tf_T; ii++) {
                                prev_exp_tmp[h*tf_T + ii] = lookup_exp<CONFIG_T>(prev_rowmax[h*tf_T + ii] - new_rowmax[h*tf_T + ii]);
                            }
                        }
                    UPDATE_GLOBAL_MAX:
                        for (int h = 0; h < CONFIG_T::n_head; h++) {
                            for (int ii = 0; ii < tf_T; ii++) {
                                prev_rowmax[h*tf_T + ii] = new_rowmax[h*tf_T + ii];
                            }
                        }
                    STORE_BLOCK:
                        for (int h = 0; h < CONFIG_T::n_head; h++) {
                            for (int ii = 0; ii < tf_T; ii++) {
                                for (int jj = 0; jj < tf_T; jj++) {
                                    P_exp_aggr_block[h*tf_T*tf_T + ii*tf_T + jj] = lookup_exp<CONFIG_T>(QK[h*tf_T*tf_T + ii*tf_T + jj] - new_rowmax[h*tf_T + jj]);
                                }
                                P_exp_aggr_block[CONFIG_T::n_head*tf_T*tf_T + h*tf_T + ii] = prev_exp_tmp[h*tf_T + ii];
                            }
                        }
                    P_exp_aggr_strm.write(P_exp_aggr_block);
                    // Write to files
                    // std::ofstream exp_diff_file("exp_diff.txt", std::ios::app);
                    // std::ofstream P_file("P.txt", std::ios::app);
                    // std::ofstream QK_file("QK.txt", std::ios::app);
                    // std::ofstream new_rowmax_file("new_rowmax.txt", std::ios::app);

                    // for (int h = 0; h < CONFIG_T::n_head; h++) {
                    //     for (int ii = 0; ii < tf_T; ii++) {
                    //         exp_diff_file << P_exp_aggr_block[CONFIG_T::n_head*tf_T*tf_T + h*tf_T + ii] << " ";
                    //         new_rowmax_file << new_rowmax[h*tf_T + ii] << " ";
                    //         for (int jj = 0; jj < tf_T; jj++) {
                    //             P_file << P_exp_aggr_block[h*tf_T*tf_T + ii*tf_T + jj] << " ";
                    //             QK_file << QK[h*tf_T*tf_T + ii*tf_T + jj] << " ";
                    //         }
                    //         P_file << "\n";
                    //         QK_file << "\n";
                    //     }
                    //     exp_diff_file << "\n";
                    //     new_rowmax_file << "\n";
                    // }

                    // exp_diff_file.close();
                    // P_file.close();
                    // QK_file.close();
                    // new_rowmax_file.close();
                }
            }
        }
    }

    typename CONFIG_T::out_proj_in_t O_ff[CONFIG_T::n_head * tf_T * tf_H];
    #pragma HLS ARRAY_PARTITION variable=O_ff   complete dim=0
    typename CONFIG_T::out_proj_in_t O_ram[CONFIG_T::n_head * tf_T * CONFIG_T::head_dim];
    #pragma HLS ARRAY_RESHAPE variable=O_ram    cyclic factor=CONFIG_T::n_head*tf_T*tf_H dim=1
    typename CONFIG_T::accum_t PV[CONFIG_T::n_head * tf_T * tf_H];
    #pragma HLS ARRAY_PARTITION variable=PV     complete dim=0
    typename CONFIG_T::exp_table_t P_ff[CONFIG_T::n_head * tf_T * tf_T];
    #pragma HLS ARRAY_PARTITION variable=P_ff     complete dim=0
    P_exp_aggr_strm_type P_exp_aggr_block_read;
    using O_buf = typename CONFIG_T::out_proj_in_t[CONFIG_T::n_head * tf_T * CONFIG_T::head_dim];
    hls::stream_of_blocks<O_buf, 4> O_sob;
    #pragma HLS ARRAY_RESHAPE variable=O_sob    cyclic factor=CONFIG_T::n_head*tf_T*tf_H dim=1

    PVO_PRODUCT:
    for (int i = 0; i < T; i++) {
        hls::write_lock<O_buf> O_write_block(O_sob);
        for (int j = 0; j < T; j++) {
            for (int hd = 0; hd < H; hd++) {
                #pragma HLS PIPELINE II = 1
                int o_offset = (i*H + hd)*CONFIG_T::n_head*tf_H*tf_T;
                int v_offset = (j*H + hd)*CONFIG_T::n_head*tf_H*tf_T;
                typename CONFIG_T::accum_t tmp[CONFIG_T::n_head*tf_T];
                if (hd == 0){
                    if (j == 0){
                    INIT_SUM:
                        for (int h = 0; h < CONFIG_T::n_head; h++) {
                            for (int ii = 0; ii < tf_T; ii++) {
                                prev_rowsum[h*tf_T + ii] = 0;
                            }
                        }
                    }
                    P_exp_aggr_block_read = P_exp_aggr_strm.read();

                    LOAD_BLOCK:
                        for (int h = 0; h < CONFIG_T::n_head; h++) {
                            for (int ii = 0; ii < tf_T; ii++) {
                                for (int jj = 0; jj < tf_T; jj++) {
                                    P_ff[h*tf_T*tf_T + ii*tf_T + jj] = (mask[i*tf_T*tf_T*T + j*tf_T*tf_T + ii*tf_T + jj]) ? static_cast<typename CONFIG_T::exp_table_t>(0) : P_exp_aggr_block_read[h*tf_T*tf_T + ii*tf_T + jj];
                                }
                            }
                        }
                    GET_SUM:
                        for (int h = 0; h < CONFIG_T::n_head; h++) {
                            for (int ii = 0; ii < tf_T; ii++) {
                                rowsum[h*tf_T + ii] = 0;
                                for (int jj = 0; jj < tf_T; jj++) {
                                    rowsum[h*tf_T + ii] += P_ff[h*tf_T*tf_T + ii*tf_T + jj];
                                }
                            }
                        }
                    UPDATE_SUM:
                        for (int h = 0; h < CONFIG_T::n_head; h++) {
                            for (int ii = 0; ii < tf_T; ii++) {
                                prev_rowsum[h*tf_T + ii] = P_exp_aggr_block_read[CONFIG_T::n_head*tf_T*tf_T + h*tf_T + ii]*prev_rowsum[h*tf_T + ii] + rowsum[h*tf_T + ii];
                                
                                // 儲存更新後的prev_row_sum至prev_row_sum.txt
                                // std::ofstream prev_row_sum_file("prev_row_sum.txt", std::ios::app);
                                // prev_row_sum_file << prev_rowsum[h*tf_T + ii] << " ";
                                // if (ii == tf_T - 1) prev_row_sum_file << "\n";
                                // prev_row_sum_file.close();
                            }
                        }
                }        
                PRELOAD_O_from_RAM_to_FF:
                    for (int h = 0; h < CONFIG_T::n_head; h++) {
                        for (int ii = 0; ii < tf_T; ii++) {
                            for (int jj = 0; jj < tf_H; jj++) {
                                if (j == 0)
                                    O_ff[h*tf_T*tf_H + ii*tf_H + jj] = static_cast<typename CONFIG_T::out_proj_in_t>(0);
                                else
                                    O_ff[h*tf_T*tf_H + ii*tf_H + jj] = O_ram[hd*tf_T*tf_H*CONFIG_T::n_head + h*tf_T*tf_H + ii*tf_H + jj];
                            }
                        }
                    }
                PV_PRODUCT:
                    for (int h = 0; h < CONFIG_T::n_head; h++) {
                        for (int ii = 0; ii < tf_T; ii++) {
                            for (int jj = 0; jj < tf_H; jj++) {
                                PV[h*tf_T*tf_H + ii*tf_H + jj] = 0;
                                for (int kk = 0; kk < tf_T; kk++) {
                                    #pragma HLS UNROLL
                                    PV[h*tf_T*tf_H + ii*tf_H + jj] += P_ff[h*tf_T*tf_T + ii*tf_T + kk]*V[v_offset + h*tf_T*tf_H + kk*tf_H + jj];
                                }
                            }
                        }
                    }
                UPDATE_O_FF:
                    for (int h = 0; h < CONFIG_T::n_head; h++) {
                        for (int ii = 0; ii < tf_T; ii++) {
                            for (int jj = 0; jj < tf_H; jj++) {
                                O_ff[h*tf_T*tf_H + ii*tf_H + jj] = P_exp_aggr_block_read[CONFIG_T::n_head*tf_T*tf_T + h*tf_T + ii]*O_ff[h*tf_T*tf_H + ii*tf_H + jj] + PV[h*tf_T*tf_H + ii*tf_H + jj];
                            }
                        }
                    }
                STORE_O_from_FF_to_RAM_or_SoB:
                    for (int h = 0; h < CONFIG_T::n_head; h++) {
                        for (int ii = 0; ii < tf_T; ii++) {
                            for (int jj = 0; jj < tf_H; jj++) {
                                if (j == T-1){
                                    inv_rowsum[h*tf_T + ii] = lookup_inv<CONFIG_T>(prev_rowsum[h*tf_T + ii]);
                                    O_write_block[hd*tf_T*tf_H*CONFIG_T::n_head + h*tf_T*tf_H + ii*tf_H + jj] = O_ff[h*tf_T*tf_H + ii*tf_H + jj]*inv_rowsum[h*tf_T + ii];
                                    
                                    // 儲存未乘上inv_row_sum的O至O.txt
                                    // std::ofstream O_file("O.txt", std::ios::app);
                                    // O_file << O_ff[h*tf_T*tf_H + ii*tf_H + jj] << " ";
                                    // if (jj == tf_H - 1) O_file << "\n";
                                    // O_file.close();
                                }
                                else{
                                    O_ram[hd*tf_T*tf_H*CONFIG_T::n_head + h*tf_T*tf_H + ii*tf_H + jj] = O_ff[h*tf_T*tf_H + ii*tf_H + jj];
                                }
                            }
                        }
                    }
            }
        }
    }
    
    typename CONFIG_T::accum_t tile_buffer[tf_T*tf_N];
    res_T res_pack;
    int out_proj_weight_offset = 0;
    int out_proj_bias_offset = 0;
    int out_proj_input_offset = 0;
    int out_proj_output_offset = 0;
    #pragma HLS ARRAY_PARTITION variable=tile_buffer complete dim=0                       
    compute_output: 
    for (int i = 0; i < T; i++) {
        hls::read_lock<O_buf> O_block(O_sob);
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < H; k++) {
                #pragma HLS PIPELINE II = 1
                out_proj_weight_offset = (k*N + j)*tf_H*tf_N*CONFIG_T::n_head; //(embed_dim/tf, head_dim/tf, n_head, tf_H, tf_N)
                out_proj_bias_offset = j*tf_N;
                out_proj_input_offset = k*tf_H*tf_T*CONFIG_T::n_head;
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
                                tile_buffer[ii*tf_N + jj] += O_block[out_proj_input_offset + h*tf_T*tf_H + ii*tf_H + kk]*out_proj_weight[out_proj_weight_offset + h*tf_H*tf_N + kk*tf_N + jj];
                            }
                        }
                        res_pack[ii*tf_N + jj] = tile_buffer[ii*tf_N + jj];
                    }
                }
                if (k==H-1){
                    res.write(res_pack);
                    // 儲存res至MHA_res.txt
                    // std::ofstream MHA_res_file("MHA_res.txt", std::ios::app);
                    // for (int ii = 0; ii < tf_T; ii++) {
                    //     for (int jj = 0; jj < tf_N; jj++) {
                    //         MHA_res_file << res_pack[ii*tf_N + jj] << " ";
                    //     }
                    //     MHA_res_file << "\n";
                    // }
                    // MHA_res_file << "\n";
                    // MHA_res_file.close();
                }
            }
        }
    }
}

}

#endif
