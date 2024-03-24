from hls4ml.backends.backend import get_backend
from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.layers import (
    TransformerEncoderLayer,
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
}};\n"""

transformer_mha_template = """struct config{index} : nnet::mha_config {{
    static const unsigned n_head = {n_head};
    static const unsigned head_dim = {head_dim};
    static const unsigned feature_dim = {feature_dim};
    static const unsigned seq_len = {seq_len};
    static const unsigned io_type = nnet::{strategy};
    static const unsigned qkv_ram_style = nnet::{qkv_ram_style};
    static const unsigned attn_ram_style = nnet::{attn_ram_style};
    static const unsigned out_ram_style = nnet::{out_ram_style};
    static const unsigned tiling_factor[3] = {tiling_factor};
}};\n"""

transformer_ffn_template = """struct config{index} : nnet::ffn_config {{
    static const unsigned n_state = {n_state};
    static const unsigned n_feature = {n_feature};
    static const unsigned io_type = nnet::{strategy};
    static const unsigned in_ram_style = nnet::{in_ram_style};
    static const unsigned out_ram_style = nnet::{out_ram_style};
    static const unsigned tiling_factor[3] = {tiling_factor};
    template<class x_T, class y_T, class config_T>
    using activation = nnet::activation::{activation}<x_T, y_T, config_T>;
}};\n"""