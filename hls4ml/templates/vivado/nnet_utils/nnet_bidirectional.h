
#ifndef NNET_BIDIRECTIONAL_H_
#define NNET_BIDIRECTIONAL_H_

#include "nnet_common.h"
#include "nnet_activation.h"
#include "nnet_recr_activations.h"
#include "nnet_dense.h"
#include "nnet_recurrent.h"
#include "hls_stream.h"
#include <iostream>


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
      // for external initial_state if needed
      // data_T initial_state[CONFIG_T::n_state],
      res_T  data_out[CONFIG_T::n_out],
	    typename CONFIG_T::weight_t     bweight     [CONFIG_T::n_state*3*CONFIG_T::n_in], //backward_gru_1_gru_cell_1_weight2 13440
	    typename CONFIG_T::weight_t     brecweight  [CONFIG_T::n_state*3*CONFIG_T::n_state], //backward_gru_1_gru_cell_1_recurrent_weight2 12288
	    typename CONFIG_T::bias_t       bbais       [CONFIG_T::n_state*3],
      typename CONFIG_T::bias_t       bbias_r     [CONFIG_T::n_state*3],
      typename CONFIG_T::weight_t     fweight     [CONFIG_T::n_state*3*CONFIG_T::n_in],//forward_gru_gru_cell_2_weight2 13440
	    typename CONFIG_T::weight_t     frecweight  [CONFIG_T::n_state*3*CONFIG_T::n_state], //forward_gru_gru_cell_2_recurrent_weight2 12288
	    typename CONFIG_T::bias_t       fbias       [CONFIG_T::n_state*3],
      typename CONFIG_T::bias_t       fbias_r     [CONFIG_T::n_state*3]
  ){
    data_T initial_state[CONFIG_T::n_state];
    data_T temp_reverse    [CONFIG_T::n_sequence*CONFIG_T::n_in];
    res_T  forwardgru_out  [CONFIG_T::n_sequence_out*CONFIG_T::n_state];
    res_T  backwardgru_out [CONFIG_T::n_sequence_out*CONFIG_T::n_state];

    for (int i=0; i<(CONFIG_T::n_state); i++){
      initial_state[i] = 0;
    }

    for (int i=0; i<(CONFIG_T::n_sequence); i++){
      for(int j=0; j<(CONFIG_T::n_in); j++){
        temp_reverse[(i*(CONFIG_T::n_in))+j] = data_in[(CONFIG_T::n_sequence-(i+1))*(CONFIG_T::n_in) + j];
      }
    }
    // res_T temp_bout [CONFIG_T::n_sequence_out*CONFIG_T::n_state];

    // nnet::reverse_input<data_T, res_T, CONFIG_T>(data_in, reverse_in);

    // for (int i =0; i<(CONFIG_T::n_sequence*CONFIG_T::n_in); i++){
    //   temp_reverse[i] = (data_T) reverse_in[i];
    // }

    nnet::gru_stack<data_T, res_T, CONFIG_T>(data_in, initial_state, forwardgru_out, fweight, frecweight, fbias, fbias_r);
    nnet::gru_stack<data_T, res_T, CONFIG_T>(temp_reverse, initial_state, backwardgru_out, bweight, brecweight, bbais, bbias_r);

    for(int j=0; j<(CONFIG_T::n_out); j++){
       data_out[j] = forwardgru_out[(CONFIG_T::n_sequence_out-1)* (CONFIG_T::n_state)+j];
       if(j>=(CONFIG_T::n_state)){
        data_out[j] = backwardgru_out[(CONFIG_T::n_sequence_out-1)* (CONFIG_T::n_state)+j-(CONFIG_T::n_state)];
       }
    }

    // std::cout << "output from backwardout: " << std::endl;
    // nnet::print_result<res_T, CONFIG_T::n_sequence_out*CONFIG_T::n_state>(backwardgru_out, std::cout);
    // std::cout << " " << std::endl;

    // std::cout << "output from temp bout: " << std::endl;
    // nnet::print_result<res_T, CONFIG_T::n_sequence_out*CONFIG_T::n_state>(temp_bout, std::cout);
    // std::cout << " " << std::endl;

    // std::cout << "output from forwardout: " << std::endl;
    // nnet::print_result<res_T, CONFIG_T::n_sequence_out*CONFIG_T::n_state>(forwardgru_out, std::cout);
    // std::cout << " " << std::endl;

    // std::cout << "output from data_out: " << std::endl;
    // nnet::print_result<res_T, CONFIG_T::n_out>(data_out, std::cout);
    // std::cout << " " << std::endl;

    }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class data_T, class res_T, typename CONFIG_T>
  void bidirectional_single(
      hls::stream<data_T> data_in[1],
      hls::stream<res_T> data_out[1],
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
        for(int j=0; j<(CONFIG_T::n_in); j++){
            #pragma HLS PIPELINE II=1
            data_T temp = data_in[0].read();
            temp_normal[(i*(CONFIG_T::n_in))+j] = temp;
            temp_reverse[(CONFIG_T::n_sequence-(i+1))*(CONFIG_T::n_in) + j] = temp;
        }
    }

    nnet::gru_stack_for_bidirectional<data_T, res_T, typename CONFIG_T::config_rnn_layer_f>(temp_normal, forwardgru_out, fweight, frecweight, fbias, fbias_r);
    nnet::gru_stack_for_bidirectional<data_T, res_T, typename CONFIG_T::config_rnn_layer_b>(temp_reverse, backwardgru_out, bweight, brecweight, bbais, bbias_r);


    res_T out_tmpt;
    for(int j=0; j<(CONFIG_T::n_out); j++){
        #pragma HLS PIPELINE II=1
        if(j <CONFIG_T::n_state ){
            out_tmpt = forwardgru_out[(CONFIG_T::n_sequence_out-1)* (CONFIG_T::n_state)+j];
        }
        else {
            out_tmpt = backwardgru_out[(CONFIG_T::n_sequence_out-1)* (CONFIG_T::n_state)+j-(CONFIG_T::n_state)];
        }
        data_out[0].write(out_tmpt);
    }

}

template<class data_T, class res_T, typename CONFIG_T>
  void bidirectional_array(
      hls::stream<data_T> data_in[CONFIG_T::n_in],
      hls::stream<res_T> data_out[CONFIG_T::n_state*2],
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
            temp_normal[(i*(CONFIG_T::n_in))+j] = temp;
            temp_reverse[(CONFIG_T::n_sequence-(i+1))*(CONFIG_T::n_in) + j] = temp;
        }
    }

    nnet::gru_stack_for_bidirectional<data_T, res_T, typename CONFIG_T::config_rnn_layer_f>(temp_normal, forwardgru_out, fweight, frecweight, fbias, fbias_r);
    nnet::gru_stack_for_bidirectional<data_T, res_T, typename CONFIG_T::config_rnn_layer_b>(temp_reverse, backwardgru_out, bweight, brecweight, bbais, bbias_r);


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

template<class data_T, class res_T, typename CONFIG_T>
  void bidirectional_switch(
      hls::stream<data_T> data_in[CONFIG_T::data_transfer_in],
      hls::stream<res_T> data_out[CONFIG_T::data_transfer_out],
      typename CONFIG_T::weight_t     bweight     [CONFIG_T::n_state*3*CONFIG_T::n_in],
      typename CONFIG_T::weight_t     brecweight  [CONFIG_T::n_state*3*CONFIG_T::n_state],
      typename CONFIG_T::bias_t       bbais       [CONFIG_T::n_state*3],
      typename CONFIG_T::bias_t       bbias_r     [CONFIG_T::n_state*3],
      typename CONFIG_T::weight_t     fweight     [CONFIG_T::n_state*3*CONFIG_T::n_in],
      typename CONFIG_T::weight_t     frecweight  [CONFIG_T::n_state*3*CONFIG_T::n_state],
      typename CONFIG_T::bias_t       fbias       [CONFIG_T::n_state*3],
      typename CONFIG_T::bias_t       fbias_r     [CONFIG_T::n_state*3]
  ){
    #pragma HLS inline region
    if(CONFIG_T::data_transfer_in == 1 && CONFIG_T::data_transfer_out == 1){
        nnet::bidirectional_single<input_t, result_t, CONFIG_T>(data_in, data_out, bweight,brecweight,bbais,bbias_r,fweight,frecweight,fbias,fbias_r);
    }else {
        nnet::bidirectional_array<input_t, result_t, CONFIG_T>(data_in, data_out, bweight,brecweight,bbais,bbias_r,fweight,frecweight,fbias,fbias_r);
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  }
#endif






