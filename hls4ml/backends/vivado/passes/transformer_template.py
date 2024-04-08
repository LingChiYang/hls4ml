from hls4ml.backends.backend import get_backend
from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.layers import (
    MultiheadAttention,
    LayerNorm,
    FeedForwardNetwork
)

mha_template = """struct config{index} : nnet::mha_config {{
    static const unsigned n_head = {num_heads};
    static const unsigned head_dim = {head_dim};
    static const unsigned embed_dim = {embed_dim};
    static const unsigned seq_len = {seq_len};
    static const unsigned qkv_ram_style = nnet::block;
    static const unsigned attn_ram_style = nnet::block;
    static const unsigned out_ram_style = nnet::block;
    static constexpr unsigned tiling_factor[3] = {tiling_factor};
    static const unsigned table_size = {table_size};
    static constexpr double table_range = {table_range};
    typedef {out_proj_bias_t.name} out_proj_bias_t;
    typedef {out_proj_weight_t.name} out_proj_weight_t;
    typedef {in_proj_bias_t.name} in_proj_bias_t;
    typedef {in_proj_weight_t.name} in_proj_weight_t;
    typedef {table_t.name} table_t;
    
}};\n"""

ffn_template = """struct config{index} : nnet::ffn_config {{
    static const unsigned seq_len = {seq_len};
    static const unsigned embed_dim = {embed_dim};
    static const unsigned hidden_dim = {hidden_dim};
    static const unsigned in_ram_style = nnet::{in_ram_style};
    static const unsigned out_ram_style = nnet::{out_ram_style};
    static constexpr unsigned tiling_factor[3] = {tiling_factor};
    typedef {out_proj_bias_t.name} out_proj_bias_t;
    typedef {out_proj_weight_t.name} out_proj_weight_t;
    typedef {in_proj_bias_t.name} in_proj_bias_t;
    typedef {in_proj_weight_t.name} in_proj_weight_t;
}};\n"""

layernorm_template = """struct config{index} : nnet::layernorm_config {{
    static const unsigned seq_len = {seq_len};
    static const unsigned embed_dim = {embed_dim};
    static const unsigned table_size = {table_size};
    static constexpr double table_range = {table_range};
    static constexpr unsigned tiling_factor[3] = {tiling_factor};
    typedef {mean_t} mean_t;   
    typedef {bias_t.name} bias_t;
    typedef {scale_t.name} scale_t;
    typedef {table_t.name} table_t;
}};\n"""


mha_function_template = 'nnet::MultiHeadAttention<{input_t}, {output_t}, {config}>({input}, {output}, {iprj_w}, {iprj_b}, {oprj_w}, {oprj_b});'
mha_include_list = ["nnet_utils/nnet_multiheadattention_stream.h"]

layernorm_function_template = 'nnet::LayerNormalize<{input_t}, {output_t}, {config}>({input}, {output}, {s}, {b});'
layernorm_include_list = ["nnet_utils/nnet_layernorm_stream.h"]

ffn_function_template = 'nnet::FeedForwardNetwork<{input_t}, {output_t}, {config}>({input}, {output}, {iprj_w}, {iprj_b}, {oprj_w}, {oprj_b});'
ffn_include_list = ["nnet_utils/nnet_feedforwardnetwork_stream.h"]

class MHAConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__((MultiheadAttention))
        self.mha_template  = mha_template 

    def format(self, node):
        params = self._default_config_params(node)
        params['tiling_factor'] = '{'+','.join([str(x) for x in params['tiling_factor']])+'}'
        mha_config = self.mha_template.format(**params)
        return mha_config

class MHAFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__((MultiheadAttention), include_header=mha_include_list)
        self.templates = mha_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['iprj_w'] = node.get_weights('in_proj_weight').name
        params['iprj_b'] = node.get_weights('in_proj_bias').name
        params['oprj_w'] = node.get_weights('out_proj_weight').name
        params['oprj_b'] = node.get_weights('out_proj_bias').name
        return self.templates.format(**params)

class LayerNormConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__((LayerNorm))
        self.layernorm_template  = layernorm_template 

    def format(self, node):
        params = self._default_config_params(node)
        params['tiling_factor'] = '{'+','.join([str(x) for x in params['tiling_factor']])+'}'
        layernorm_config = self.layernorm_template.format(**params)
        return layernorm_config

class LayerNormFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__((LayerNorm), include_header=layernorm_include_list)
        self.templates = layernorm_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['s'] = node.get_weights('scale').name
        params['b'] = node.get_weights('bias').name

        return self.templates.format(**params)

class FFNConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__((FeedForwardNetwork))
        self.ffn_template  = ffn_template 

    def format(self, node):
        params = self._default_config_params(node)
        params['tiling_factor'] = '{'+','.join([str(x) for x in params['tiling_factor']])+'}'
        ffn_config = self.ffn_template.format(**params)
        return ffn_config

class FFNFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__((FeedForwardNetwork), include_header=ffn_include_list)
        self.templates = ffn_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['iprj_w'] = node.get_weights('in_proj_weight').name
        params['iprj_b'] = node.get_weights('in_proj_bias').name
        params['oprj_w'] = node.get_weights('out_proj_weight').name
        params['oprj_b'] = node.get_weights('out_proj_bias').name
        return self.templates.format(**params)