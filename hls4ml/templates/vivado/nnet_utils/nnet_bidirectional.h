#ifndef NNET_BIDIRECTIONAL_H_
#define NNET_BIDIRECTIONAL_H_

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_recurrent.h"

namespace nnet{
struct bidirectional_config
{
    // Internal data type definitions
    typedef float weight_t;
    typedef float bias_t;
    typedef float accum_t;

    // Layer Sizes
    static const unsigned n_in =  2;
    static const unsigned n_out = 2;
    static const unsigned n_state = 2;
    static const unsigned n_sequence = 2;
    static const unsigned n_4state = 8;
    static const unsigned table_size = 1024;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    static const bool use_static = true;
    static const unsigned n_zeros = 0;

    template<class x_T, class y_T, class config_T>
    using activation_recr = nnet::activation::relu<x_T, y_T, config_T>;
    template<class x_T, class y_T, class config_T>
    using activation = nnet::activation::relu<x_T, y_T, config_T>;
};

// template<class data_T, class res_T, typename CONFIG_T>
//   void reverse_input(
//       data_T data_in[CONFIG_T::n_sequence*CONFIG_T::n_in],
//       res_T  data_out[CONFIG_T::n_sequence*CONFIG_T::n_in]    
//   ){
//     // for(int i = 0; i < (CONFIG_T::n_sequence*CONFIG_T::n_in); i++){
//     //     data_out[(CONFIG_T::n_sequence*CONFIG_T::n_in)-(1+i)] = data_in[i];
//     // }
//     for (int i=0; i<(CONFIG_T::n_sequence); i++){
//       for(int j=0; j<(CONFIG_T::n_in); j++){
//         data_out[(i*(CONFIG_T::n_in))+j] = data_in[(CONFIG_T::n_sequence-(i+1))*(CONFIG_T::n_in) + j];
//       }
//     }
//   }

template<class data_T, class res_T, typename CONFIG_T>
  void bidirectional(
      data_T data_in[CONFIG_T::n_sequence*CONFIG_T::n_in],
      res_T  data_out[CONFIG_T::n_out],
	    typename CONFIG_T::config_rnn_layer_b::weight_t     bweight     [CONFIG_T::config_rnn_layer_b::n_state*3*CONFIG_T::n_in],
	    typename CONFIG_T::config_rnn_layer_b::weight_t     brecweight  [CONFIG_T::config_rnn_layer_b::n_state*3*CONFIG_T::n_state],
	    typename CONFIG_T::config_rnn_layer_b::bias_t       bbais       [CONFIG_T::config_rnn_layer_b::n_state*3],
      typename CONFIG_T::config_rnn_layer_b::bias_t       bbias_r     [CONFIG_T::config_rnn_layer_b::n_state*3],
      typename CONFIG_T::config_rnn_layer_f::weight_t     fweight     [CONFIG_T::config_rnn_layer_f::n_state*3*CONFIG_T::n_in],
	    typename CONFIG_T::config_rnn_layer_f::weight_t     frecweight  [CONFIG_T::config_rnn_layer_f::n_state*3*CONFIG_T::n_state],
	    typename CONFIG_T::config_rnn_layer_f::bias_t       fbias       [CONFIG_T::config_rnn_layer_f::n_state*3],
      typename CONFIG_T::config_rnn_layer_f::bias_t       fbias_r     [CONFIG_T::config_rnn_layer_f::n_state*3]
  ){

    data_T temp_reverse    [CONFIG_T::n_sequence*CONFIG_T::n_in];
    res_T  forwardgru_out  [CONFIG_T::config_rnn_layer_f::n_sequence_out*CONFIG_T::config_rnn_layer_f::n_state];
    res_T  backwardgru_out [CONFIG_T::config_rnn_layer_b::n_sequence_out*CONFIG_T::config_rnn_layer_b::n_state];
    for (int i=0; i<(CONFIG_T::n_sequence); i++){
      for(int j=0; j<(CONFIG_T::n_in); j++){
        temp_reverse[(i*(CONFIG_T::n_in))+j] = data_in[(CONFIG_T::n_sequence-(i+1))*(CONFIG_T::n_in) + j];
      }
    }
    nnet::gru_stack<data_T, res_T, typename CONFIG_T::config_rnn_layer_f>(data_in, forwardgru_out, fweight, frecweight, fbias, fbias_r);
    nnet::gru_stack<data_T, res_T, typename CONFIG_T::config_rnn_layer_b>(temp_reverse, backwardgru_out, bweight, brecweight, bbais, bbias_r);
    

    for(int j=0; j<(CONFIG_T::n_out); j++){
       data_out[j] = forwardgru_out[(CONFIG_T::n_sequence_out-1)* (CONFIG_T::n_state)+j];
       if(j>=(CONFIG_T::n_state)){
        data_out[j] = backwardgru_out[(CONFIG_T::n_sequence_out-1)* (CONFIG_T::n_state)+j-(CONFIG_T::n_state)];
       }
    }
}

template<class data_T, class res_T, typename CONFIG_T>
  void bidirectional(
      hls::stream<data_T> data_in[CONFIG_T::n_in],
      hls::stream<res_T> data_out[CONFIG_T::n_out],
	    typename CONFIG_T::weight_t     bweight     [CONFIG_T::n_state*3*CONFIG_T::n_in],
	    typename CONFIG_T::weight_t     brecweight  [CONFIG_T::n_state*3*CONFIG_T::n_state],
	    typename CONFIG_T::bias_t       bbais       [CONFIG_T::n_state*3],
      typename CONFIG_T::bias_t       bbias_r     [CONFIG_T::n_state*3],
      typename CONFIG_T::weight_t     fweight     [CONFIG_T::n_state*3*CONFIG_T::n_in],
	    typename CONFIG_T::weight_t     frecweight  [CONFIG_T::n_state*3*CONFIG_T::n_state],
	    typename CONFIG_T::bias_t       fbias       [CONFIG_T::n_state*3],
      typename CONFIG_T::bias_t       fbias_r     [CONFIG_T::n_state*3]
  ){


    data_T temp_normal[CONFIG_T::n_sequence*CONFIG_T::n_in];
    #pragma HLS ARRAY_PARTITION variable=temp_normal cyclic factor=CONFIG_T::n_in
    data_T temp_reverse[CONFIG_T::n_sequence*CONFIG_T::n_in];
    #pragma HLS ARRAY_PARTITION variable=temp_reverse cyclic factor=CONFIG_T::n_in

    res_T forwardgru_out[CONFIG_T::n_sequence_out*CONFIG_T::n_state];
    #pragma HLS ARRAY_PARTITION variable=forwardgru_out cyclic factor=CONFIG_T::n_state
    res_T backwardgru_out[CONFIG_T::n_sequence_out*CONFIG_T::n_state];
    #pragma HLS ARRAY_PARTITION variable=backwardgru_out cyclic factor=CONFIG_T::n_state

    for (int i=0; i<(CONFIG_T::n_sequence); i++){
        #pragma HLS PIPELINE
        for(int j=0; j<(CONFIG_T::n_in); j++){
            #pragma HLS UNROLL
            data_T temp = data_in[j].read();
            temp_normal[i*CONFIG_T::n_in+j] = temp;
            temp_reverse[((CONFIG_T::n_sequence)-i-1)*CONFIG_T::n_in+j] = temp;
        }
    }

    nnet::gru_stack<data_T, res_T, typename CONFIG_T::config_rnn_layer_f>(temp_normal, forwardgru_out, fweight, frecweight, fbias, fbias_r);
    nnet::gru_stack<data_T, res_T, typename CONFIG_T::config_rnn_layer_b>(temp_reverse, backwardgru_out, bweight, brecweight, bbais, bbias_r);


    res_T out_tmpt;
    for(int j=0; j<(CONFIG_T::n_out); j++){
        #pragma HLS UNROLL
        if(j <CONFIG_T::n_state ){
            out_tmpt = forwardgru_out[(CONFIG_T::n_sequence_out-1)* (CONFIG_T::n_state)+j];
        }
        else {
            out_tmpt = backwardgru_out[(CONFIG_T::n_sequence_out-1)* (CONFIG_T::n_state)+j-(CONFIG_T::n_state)];
        }
        data_out[j].write(out_tmpt);
    }

}

  }
#endif
