from hls4ml.converters.pytorch_to_hls import pytorch_handler
from hls4ml.converters.utils import compute_padding_1d_pytorch, compute_padding_2d_pytorch, parse_data_format
from hls4ml.converters.pytorch.core import parse_linear_layer

@pytorch_handler('LayerNorm')
def parse_layernrom_layer(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    assert 'LayerNorm' in operation
    layer = {}

    layer['name'] = layer_name
    layer['inputs'] = input_names
    layer['class_name'] = 'LayerNorm'
    layer['data_format'] = 'channels_first'
    #only implemented for in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias
    #TODO: implement for other weights and biases

    output_shapes = input_shapes   
    return layer, output_shapes

@pytorch_handler('MultiheadAttention')
def parse_mha_layer(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    assert 'MultiheadAttention' in operation
    layer = {}

    layer['name'] = layer_name
    layer['inputs'] = input_names
    layer['class_name'] = 'MultiheadAttention'
    layer['data_format'] = 'channels_first'
    #only implemented for in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias
    #TODO: implement for other weights and biases
    from pprint import pprint
    print("class_object.__dict__ = ")
    pprint(class_object.__dict__)
    layer['n_head'] = class_object.num_heads
    layer['head_dim'] = class_object.head_dim
    layer['feature_dim'] = class_object.embed_dim
    print(input_shapes)
    layer['seq_len'] = input_shapes[0][-2]
    layer['in_proj_weight_data'] = class_object.in_proj_weight.data.numpy()
    layer['in_proj_bias_data'] = class_object.in_proj_bias.data.numpy()
    layer['out_proj_weight_data'] = class_object.__dict__['_modules']['out_proj'].weight.data.numpy()
    layer['out_proj_bias_data'] = class_object.__dict__['_modules']['out_proj'].bias.data.numpy()

    output_shapes = input_shapes   
    return layer, output_shapes

@pytorch_handler('TransformerEncoderLayer')
def parse_transenc_layer(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    assert 'TransformerEncoderLayer' in operation
    layer = {}

    layer['name'] = layer_name
    layer['inputs'] = input_names
    layer['class_name'] = 'LayerGroup'
    layer['data_format'] = 'channels_first'  # Pytorch default (can't change)
    
    subclass_object = class_object.__dict__['_modules']['self_attn']
    sub_layer, output_shapes = parse_mha_layer('MultiheadAttention', layer_name + '_self_attn', input_names, input_shapes, node, subclass_object, data_reader, config)
    layer['self_attn'] = sub_layer
    subclass_object = class_object.__dict__['_modules']['linear1']
    sub_layer, output_shapes = parse_linear_layer('Linear', layer_name + '_linear1', input_names, input_shapes, node, subclass_object, data_reader, config)
    layer['linear1'] = sub_layer
    subclass_object = class_object.__dict__['_modules']['linear2']
    sub_layer, output_shapes = parse_linear_layer('Linear', layer_name + '_linear2', input_names, input_shapes, node, subclass_object, data_reader, config)
    layer['linear2'] = sub_layer
    subclass_object = class_object.__dict__['_modules']['norm1']
    sub_layer, output_shapes = parse_layernrom_layer('LayerNorm', layer_name + '_norm1', input_names, input_shapes, node, class_object, data_reader, config)
    layer['norm1'] = sub_layer
    subclass_object = class_object.__dict__['_modules']['norm2']
    sub_layer, output_shapes = parse_layernrom_layer('LayerNorm', layer_name + '_norm2', input_names, input_shapes, node, class_object, data_reader, config)
    layer['norm2'] = sub_layer
    #print("input_shapes = ", input_shapes)
    output_shapes = input_shapes
    return layer, output_shapes

@pytorch_handler('Layers')
def parse_layers(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    assert 'Layers' in operation

    layer = {}

    layer['name'] = layer_name
    layer['inputs'] = input_names
    layer['class_name'] = 'LayerGroup'
    layer_list = []
    for key, subclass_object in class_object.__dict__['_modules'].items():
        sublayer_name = layer_name + '_' + key
        if 'TransformerEncoderLayer' == subclass_object._get_name():
            sublayer, _= parse_transenc_layer('TransformerEncoderLayer', sublayer_name, input_names, input_shapes, node, subclass_object, data_reader, config)
            layer_list.append(sublayer)
        else:
            raise Exception(f'Layer type not supported: {subclass_object._get_name()} in {layer_name}')    

    # LayerGroup info
    layer['output_shape'] = input_shapes
    layer['layer_list'] = layer_list
    layer['input_layers'] = []
    layer['output_layers'] = []
    layer['data_reader'] = data_reader

    output_shape = input_shapes  # Channel first as default

    return layer, output_shape

@pytorch_handler('TransformerEncoder')
def parse_transenc(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    assert 'TransformerEncoder' in operation

    layer = {}

    layer['name'] = layer_name
    layer['inputs'] = input_names
    layer['class_name'] = 'LayerGroup'
    layer_list = []
    for key, subclass_object in class_object.__dict__['_modules'].items():
        if key == 'layers':
            sublayer, _= parse_layers('Layers', key, input_names, input_shapes, node, subclass_object, data_reader, config)
            layer_list.append(sublayer)
        elif key == 'norm':
            sublayer, _= parse_layernrom_layer('LayerNorm', key, input_names, input_shapes, node, subclass_object, data_reader, config)
            layer_list.append(sublayer)

    # LayerGroup info
    layer['output_shape'] = input_shapes
    layer['layer_list'] = layer_list
    layer['input_layers'] = []
    layer['output_layers'] = []
    layer['data_reader'] = data_reader

    output_shape = input_shapes  # Channel first as default

    return layer, output_shape


#@pytorch_handler('Conv2d')
def parse_conv2d_layer(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    assert 'Conv2d' in operation

    layer = {}

    layer['name'] = layer_name
    layer['inputs'] = input_names
    layer['class_name'] = 'Conv2D'
    layer['data_format'] = 'channels_first'  # Pytorch default (can't change)

    layer['weight_data'] = class_object.weight.data.numpy()
    if class_object.bias is not None:
        layer['bias_data'] = class_object.bias.data.numpy()
    else:
        layer['bias_data'] = None

    # Input info
    (layer['in_height'], layer['in_width'], layer['n_chan']) = parse_data_format(
        input_shapes[0], 'channels_first'
    )  # Keras's default is channels_last

    # Additional parameters
    layer['n_filt'] = class_object.out_channels
    layer['filt_height'] = class_object.kernel_size[0]
    layer['filt_width'] = class_object.kernel_size[1]
    layer['stride_height'] = class_object.stride[0]
    layer['stride_width'] = class_object.stride[1]
    layer['dilation'] = class_object.dilation[0]
    layer['pad_top'] = layer['pad_bottom'] = class_object.padding[0]
    layer['pad_left'] = layer['pad_right'] = class_object.padding[1]

    if all(x == 0 for x in class_object.padding):  # No padding, i.e., 'VALID' padding in Keras/Tensorflow
        layer['padding'] = 'valid'
    else:  # Only 'valid' and 'same' padding are available in Keras
        layer['padding'] = 'same'

    # Ouput info
    (layer['out_height'], layer['out_width'], _, _, _, _) = compute_padding_2d_pytorch(
        class_object.padding,
        layer['in_height'],
        layer['in_width'],
        layer['stride_height'],
        layer['stride_width'],
        layer['filt_height'],
        layer['filt_width'],
        class_object.dilation[0],
        class_object.dilation[1],
    )

    output_shape = [input_shapes[0][0], layer['n_filt'], layer['out_height'], layer['out_width']]

    return layer, output_shape
