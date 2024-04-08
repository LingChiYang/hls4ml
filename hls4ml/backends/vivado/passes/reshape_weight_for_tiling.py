import numpy as np

from hls4ml.model.layers import MultiheadAttention, LayerNorm
from hls4ml.model.optimizer import OptimizerPass


class ReshapeWeightForTiling(OptimizerPass):
    '''Reshape the weights of the layers with resource strategy to be compatible with tiling.'''

    def match(self, node):
        node_matches = isinstance(node, (MultiheadAttention, LayerNorm))
        is_io_tile_stream = node.get_attr('iotype', '').lower() == 'io_tile_stream'
        print('iotype:', node.get_attr('iotype', ''))
        print(node.__class__.__name__)

        return node_matches and is_io_tile_stream and not node.get_attr('_weights_reshaped', False)

    def transform(self, model, node):
        #if isinstance(node, (MultiHeadAttention)):
        #    node.weights['weight'].data = np.transpose(node.weights['weight'].data)
        if isinstance(node, (MultiheadAttention)): 
            node.weights['in_proj_weight'].data = np.reshape(node.weights['in_proj_weight'].data, \
                                                            (3, \
                                                            node.get_attr("num_heads"), \
                                                            node.weights['in_proj_weight'].data.shape[0]//3//node.get_attr("tiling_factor")[1]//node.get_attr("num_heads"), \
                                                            node.get_attr("tiling_factor")[1], \
                                                            node.weights['in_proj_weight'].data.shape[1]//node.get_attr("tiling_factor")[1], \
                                                            node.get_attr("tiling_factor")[1]))
            node.weights['in_proj_weight'].data = np.transpose(node.weights['in_proj_weight'].data, axes=[0, 1, 4, 2, 5, 3])
            node.weights['in_proj_weight'].shape = node.weights['in_proj_weight'].data.shape
            node.weights['in_proj_bias'].data = np.reshape(node.weights['in_proj_bias'].data, \
                                                            (3, node.get_attr("num_heads"), node.weights['in_proj_bias'].data.shape[0]//3//node.get_attr("tiling_factor")[1]//node.get_attr("num_heads"), node.get_attr("tiling_factor")[1]))
            node.weights['in_proj_bias'].shape = node.weights['in_proj_bias'].data.shape
            node.weights['out_proj_weight'].data = np.reshape(node.weights['out_proj_weight'].data, \
                                                            (node.get_attr("num_heads"), \
                                                            node.weights['out_proj_weight'].data.shape[0]//node.get_attr("tiling_factor")[1], \
                                                            node.get_attr("tiling_factor")[1], \
                                                            node.weights['out_proj_weight'].data.shape[1]//node.get_attr("tiling_factor")[1]//node.get_attr("num_heads"), \
                                                            node.get_attr("tiling_factor")[1]))
            node.weights['out_proj_weight'].data = np.transpose(node.weights['out_proj_weight'].data, axes=[0, 3, 1, 4, 2])
            node.weights['out_proj_weight'].shape = node.weights['out_proj_weight'].data.shape
            node.weights['out_proj_bias'].data = np.reshape(node.weights['out_proj_bias'].data, \
                                                            (node.weights['out_proj_bias'].data.shape[0]//node.get_attr("tiling_factor")[1], node.get_attr("tiling_factor")[1]))        
            node.weights['out_proj_bias'].shape = node.weights['out_proj_bias'].data.shape
        #elif isinstance(node, (FFN)):
        #    #get iotype
        #    if model.config.get_config_value('IOType') == 'io_array_stream':
        #        node.weights['ffn1_weight'].data = np.reshape(node.weights['ffn1_weight'].data, (node.weights['ffn1_weight'].data.shape[0]//node.get_attr("block_y"), node.get_attr("block_y"), node.weights['ffn1_weight'].data.shape[1]//node.get_attr("block_k"), node.get_attr("block_k")))
        #        node.weights['ffn1_weight'].data = np.transpose(node.weights['ffn1_weight'].data, axes=[0, 2, 1, 3])
        #        node.weights['ffn2_weight'].data = np.reshape(node.weights['ffn2_weight'].data, (node.weights['ffn2_weight'].data.shape[0]//node.get_attr("block_k"), node.get_attr("block_k"), node.weights['ffn2_weight'].data.shape[1]//node.get_attr("block_y"), node.get_attr("block_y")))
        #        node.weights['ffn2_weight'].data = np.transpose(node.weights['ffn2_weight'].data, axes=[0, 2, 1, 3])  
        #        node.weights['ffn1_weight'].shape = node.weights['ffn1_weight'].data.shape
        #        node.weights['ffn2_weight'].shape = node.weights['ffn2_weight'].data.shape
        #        node.weights['ffn1_bias'].data = np.reshape(node.weights['ffn1_bias'].data, (node.weights['ffn1_bias'].data.shape[0]//node.get_attr("block_k"), node.get_attr("block_k")))
        #        node.weights['ffn1_bias'].shape = node.weights['ffn1_bias'].data.shape
        #        node.weights['ffn2_bias'].data = np.reshape(node.weights['ffn2_bias'].data, (node.weights['ffn2_bias'].data.shape[0]//node.get_attr("block_y"), node.get_attr("block_y")))
        #        node.weights['ffn2_bias'].shape = node.weights['ffn2_bias'].data.shape
        #        print("data shape",node.weights['ffn1_weight'].shape)
        elif isinstance(node, (LayerNorm)):
            node.weights['scale'].data = np.reshape(node.weights['scale'].data, (node.weights['scale'].data.shape[0]//node.get_attr("tiling_factor")[1], node.get_attr("tiling_factor")[1]))
            node.weights['scale'].shape = node.weights['scale'].data.shape
            node.weights['bias'].data = np.reshape(node.weights['bias'].data, (node.weights['bias'].data.shape[0]//node.get_attr("tiling_factor")[1], node.get_attr("tiling_factor")[1]))
            node.weights['bias'].shape = node.weights['bias'].data.shape
        else:
            raise Exception(f'Unexpected layer {node.class_name} with resource strategy')

        node.set_attr('_weights_reshaped', True)

        return False
