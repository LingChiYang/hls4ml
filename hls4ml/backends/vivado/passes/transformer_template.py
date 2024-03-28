from hls4ml.backends.backend import get_backend
from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.layers import (
    MultiheadAttention,
    LayerNorm,
)

transformer_layer_template = """struct config{index} : nnet::transformer_config {{
    typedef {config_mha} mha_config;
    typedef {config_ffn} ffn_config;
    typedef {config_add_mha} add_mha_config;
    typedef {config_add_ffn} add_ffn_config;
    typedef {config_norm_mha} norm_mha_config;
    typedef {config_norm_ffn} norm_ffn_config;
    static const unsigned seq_len  = {seq_len};
    static const unsigned feature_dim = {feature_dim};
    static const unsigned io_type = nnet::{strategy};
    static const unsigned fifo_ram_style = nnet::{ram_style};
    static const unsigned tiling_factor[3] = {tiling_factor};
    static const bool norm_first = {norm_first};
    static const bool ffn_norm_use_bias = {ffn_norm_use_bias};
    typedef {mha_fifo_t.name} mha_fifo_t;
    typedef {ffn_fifo_t.name} ffn_fifo_t;
    typedef {mha_layernorm_fifo_t.name} add_fifo_t;
}};\n"""

mha_template = """struct mha_config{index} : nnet::mha_config {{
    static const unsigned n_head = {n_head};
    static const unsigned head_dim = {head_dim};
    static const unsigned feature_dim = {feature_dim};
    static const unsigned seq_len = {seq_len};
    static const unsigned qkv_ram_style = nnet::block;
    static const unsigned attn_ram_style = nnet::block;
    static const unsigned out_ram_style = nnet::block;
    static const unsigned tiling_factor[{rank}] = {tiling_factor};
}};\n"""

softmax_template = """struct softmax_config{index} : nnet::activ_config {{
    static const unsigned n_in = {n_in};
    static const unsigned table_size = {table_size};
    static const unsigned tiling_factor[{rank}] = {tiling_factor};
    static const nnet::softmax_implementation implementation = nnet::softmax_implementation::{implementation};
    typedef {table_t.name} exp_table_t;
    typedef {table_t.name} inv_table_t;
    typedef {accum_t.name} accum_t;
    static const unsigned inv_range = {inv_range};
    static const unsigned exp_range = {exp_range};
}};\n"""

transformer_ffn_template = """struct ffn_config{index} : nnet::ffn_config {{
    static const unsigned seq_len = {seq_len};
    static const unsigned feature_dim = {feature_dim};
    static const unsigned hidden_dim = {hidden_dim};
    static const unsigned in_ram_style = nnet::{in_ram_style};
    static const unsigned out_ram_style = nnet::{out_ram_style};
    static const unsigned tiling_factor[{rank}] = {tiling_factor};
    typedef {act_t} ACT_CONFIG_T;
}};\n"""

activ_template = """struct act_config{index} : nnet::activ_config {{
    static const unsigned n_in = {n_in};
    static const unsigned table_size = {table_size};
    static const unsigned tiling_factor[{rank}] = {tiling_factor};
    typedef {table_t.name} table_t;
}};\n"""

mha_layernorm_template = """struct mha_layernorm_config{index} : nnet::layernorm_config {{
    static const unsigned seq_len = {seq_len};
    static const unsigned feature_dim = {feature_dim};
    static const unsigned table_size = {table_size};
    static constexpr double table_range = {table_range};
    static const unsigned tiling_factor[{rank}] = {tiling_factor};
    typedef {bias_t.name} bias_t;
    typedef {scale_t.name} scale_t;
    typedef {table_t.name} table_t;
}};\n"""

ffn_layernorm_template = """struct ffn_layernorm_config{index} : nnet::layernorm_config {{
    static const unsigned seq_len = {seq_len};
    static const unsigned feature_dim = {feature_dim};
    static const unsigned table_size = {table_size};
    static constexpr double table_range = {table_range};
    static const unsigned tiling_factor[{rank}] = {tiling_factor};
    typedef {bias_t.name} bias_t;
    typedef {scale_t.name} scale_t;
    typedef {table_t.name} table_t;
}};\n"""

transformer_function_template = 'nnet::transformer<{input_t}, {output_t}, {config}>({input_qkv}, {output}, \
                                                                                    {w_mha_in_proj}, {b_mha_in_proj}, \
                                                                                    {mask_mha}, \
                                                                                    {w_mha_out_proj}, {b_mha_out_proj},\
                                                                                    {w_mha_norm}, {b_mha_norm},\
                                                                                    {w_ffn_in_proj}, {b_ffn_in_Proj},\
                                                                                    {w_ffn_norm}, {b_ffn_norm});'
#print('transformer template',transformer_function_template)
transformer_include_list = ['nnet_utils/nnet_recurrent.h']

class MHAConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__((M))
        self.template = transformer_layer_template
        self.transformer_mha_template  = transformer_mha_template 
        self.activ_template = activ_template
        self.transformer_ffn_template = transformer_ffn_template
        self.softmax_template = softmax_template
        self.mha_layernorm_template = mha_layernorm_template
        self.ffn_layernorm_template = ffn_layernorm_template

    def format(self, node):
        params = self._default_config_params(node)
        params['config_mha'] = f'mha_config{node.index}'
        params['config_ffn'] = f'ffn_config{node.index}'
        #print(node.get_input_variable().dim_names)
        #print(node.get_output_variable().dim_names)
        #print(params)
        mha_params = node.get_attr('self_attn')
        mha_params['index'] = node.index
        from pprint import pprint
        pprint(mha_params.keys())
        mha_config = self.transformer_mha_template.format(**mha_params)
        ffn_config = self.transformer_ffn_template.format(**node.get_attr('linear1'))
        transformer_config = self.template.format(**params)
        return transformer_config