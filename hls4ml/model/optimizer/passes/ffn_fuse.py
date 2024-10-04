from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.layers import Dense, FeedForwardNetwork, Activation

class FuseFeedForwardNetwork(OptimizerPass):
    def match(self, node):
        if not isinstance(node, Dense):
            return False
        if len(node.get_output_nodes()) != 1: #check if there is only one output node
            return False
        if len(node.get_output_nodes()[0].get_output_nodes()) != 1: #check if there is only one output node
            return False
        is_match = isinstance(node, Dense) and \
            isinstance(node.get_output_nodes()[0], (Activation)) and \
            isinstance(node.get_output_nodes()[0].get_output_nodes()[0], Dense) and \
            node.get_attr('n_in') == node.get_output_nodes()[0].get_output_nodes()[0].get_attr('n_out')
        return is_match

    def transform(self, model, node):
        # Fuse weight and bias of Dense/Conv1D/Conv2D layer with BN values
        #if model.config.get_config_value('IOType') == 'io_array_stream':
        in_proj_weight = node.weights['weight']
        in_proj_bias = node.weights['bias']
        embed_dim = node.get_attr('n_in')
        hidden_dim = node.get_attr('n_out')
        ffn2_node = node.get_output_nodes()[0].get_output_nodes()[0]
        out_proj_weight = ffn2_node.weights['weight']
        out_proj_bias = ffn2_node.weights['bias']
        new_attr = node.attributes
        new_attr['embed_dim'] = embed_dim
        new_attr['seq_len'] = node.get_output_variable().shape[-1]
        new_attr['hidden_dim'] = hidden_dim
        new_attr['in_proj_weight_data'] = in_proj_weight.data.transpose()
        new_attr['in_proj_bias_data'] = in_proj_bias.data
        new_attr['out_proj_weight_data'] = out_proj_weight.data.transpose()
        new_attr['out_proj_bias_data'] = out_proj_bias.data
        new_attr[node.name] = ffn2_node.get_attr(ffn2_node.name)
        new_node = FeedForwardNetwork(model, 'ffn', new_attr, node.inputs.copy())
        out_var = model.get_output_variables()[0]
        model.remove_node(node.get_output_nodes()[0], rewire=True) #remove the activation node
        model.remove_node(node.get_output_nodes()[0], rewire=True) #remove the dense node
        model.replace_node(node, new_node)
        model.register_output_variable(node.name, out_var)
        return True