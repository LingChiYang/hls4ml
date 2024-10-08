import torch

from hls4ml.model import ModelGraph


class PyTorchModelReader:
    """
    PyTorch reader to extract weights data.
    """

    def __init__(self, config):
        self.torch_model = config['PytorchModel']
        self.state_dict = self.torch_model.state_dict()
        self.input_shape = config['InputShape']

    def get_weights_data(self, layer_name, var_name):
        data = None

        tensorName = layer_name + '.' + var_name

        if tensorName in self.state_dict:
            data = self.state_dict[tensorName].numpy()

        return data


class PyTorchFileReader(PyTorchModelReader):  # Inherit get_weights_data method
    def __init__(self, config):
        self.config = config

        if not torch.cuda.is_available():
            self.torch_model = torch.load(config['PytorchModel'], map_location=lambda storage, loc: storage)
        else:
            self.torch_model = torch.load(config['PytorchModel'])

        # Get input tensor's shape
        self.input_shape = config.get('InputShape')

        if self.input_shape is None:
            raise Exception('Must specify input shape ("InputShape") in config!')

        # Convert it to a list
        self.input_shape = self.input_shape.strip('(,)').split(',')
        self.input_shape = [None if n == 'None' else int(n) for n in self.input_shape]

        self.state_dict = self.torch_model.state_dict()


def get_weights_data(data_reader, layer_name, var_name):
    if not isinstance(var_name, (list, tuple)):
        var_name = [var_name]

    data = [data_reader.get_weights_data(layer_name, var) for var in var_name]

    if len(data) == 1:
        return data[0]
    else:
        return (*data,)


# ----------------------Layer handling--------------------- #
layer_handlers = {}


def register_pytorch_layer_handler(layer_name, handler_func):
    if layer_name in layer_handlers:
        raise Exception(f'Layer {layer_name} already registered')
    else:
        layer_handlers[layer_name] = handler_func


def get_supported_pytorch_layers():
    return list(layer_handlers.keys())


def pytorch_handler(*args):
    def decorator(function):
        function.handles = [arg for arg in args]
        return function

    return decorator


# map names of operations between toch.nn and torch.nn.functionals
layer_name_map = {
    'relu': 'ReLU',
    'leaky_relu': 'LeakyReLU',
    'elu': 'ELU',
    'prelu': 'PReLU',
    'sigmoid': 'Sigmoid',
    '_threshold': 'Threshold',
    'softmax': 'Softmax',
    'max_pool1d': 'MaxPool1d',
    'max_pool2d': 'MaxPool2d',
    'avg_pool1d': 'AvgPool1d',
    'avg_pool2d': 'AvgPool2d',
    'flatten': 'Flatten',
}


# ----------------------------------------------------------------


def pytorch_to_hls(config):
    """Convert PyTorch model to hls4ml ModelGraph.

    Args:
        config (dict): The conversion config

    Raises:
        Exception: On unsupported features of the model.

    Returns:
        ModelGraph: hls4ml model object.
    """

    # This is a list of dictionaries to hold all the layer info we need to generate HLS
    layer_list = []

    #print('Interpreting Model ...')

    reader = PyTorchFileReader(config) if isinstance(config['PytorchModel'], str) else PyTorchModelReader(config)
    if type(reader.input_shape) is tuple:
        input_shapes = [list(reader.input_shape)]
    else:
        input_shapes = list(reader.input_shape)
    input_shapes = [list(shape) for shape in input_shapes]

    model = reader.torch_model

    # dict of layer objects in non-traced form for access lateron
    children = {c[0]: c[1] for c in model.named_children()}
    # use symbolic_trace to get a full graph of the model
    from torch.fx import symbolic_trace

    traced_model = symbolic_trace(model)
    # Define layers to skip for conversion to HLS
    skip_layers = ['Dropout', 'Sequential']

    # All supported layers
    supported_layers = get_supported_pytorch_layers() + skip_layers

    # Map inputs of skipped and split (activation) layers
    inputs_map = {}

    input_layers = []

    # Output shape tracking
    output_shapes = {}
    output_shape = None

    # Loop through layers
    #print('Topology:')
    layer_counter = 0

    n_inputs = 0

    for node in traced_model.graph.nodes:
        if node.op == 'call_module':
            # modules that are part of a torch.nn.Sequential with name 'name' have target names 'name.x',
            # where x is an integer numbering the elements of the Sequential
            if '.' in node.target:
                fqn_path = node.target.split('.')
                sub_children = dict(children[fqn_path[0]].named_children())
                for name in fqn_path[1:-1]:
                    sub_children = dict(sub_children[name].named_children())
                sub_children[fqn_path[-1]]
                class_object = sub_children[fqn_path[-1]]
            else:
                class_object = children[node.target]

            pytorch_class = class_object.__class__.__name__
            #print(f'Layer {layer_counter}: {pytorch_class}')
            if pytorch_class not in supported_layers:
                raise Exception(f'Unsupported layer {pytorch_class}')

            if layer_counter != 0:
                input_shapes = [output_shape]  # In case there are multiple inputs

            layer_name = node.name

            # Handle skipped layers
            if pytorch_class in skip_layers:
                if pytorch_class == 'Sequential':  # Ignore the mother module's class name
                    continue

                # Assuming only one input
                parent_input = [str(i) for i in node.args][0]
                inputs_map[layer_name] = inputs_map.get(parent_input, parent_input)

                output_shapes[layer_name] = input_shapes[0]

                continue

            # Increment the layer counter after initial screenings
            if pytorch_class in supported_layers:
                layer_counter += 1

            # parse info from class object
            input_names = [inputs_map.get(str(i), str(i)) for i in node.args]
            input_shapes = [output_shapes[str(i)] for i in node.args]

            # for Conv layers
            if 'Conv' in pytorch_class:
                if not class_object.padding_mode == 'zeros':
                    raise Exception('Padding modes other than "zeros" not implemented yet')
                if not class_object.groups == 1:
                    raise Exception('Non-default options for groups not implemented yet')
                        

            # Process the layer
            layer, output_shape = layer_handlers[pytorch_class](
                pytorch_class, layer_name, input_names, input_shapes, node, class_object, reader, config
            )
            
            #for transformer encoder
            if 'TransformerEncoder' in pytorch_class:
                if len(input_names) > 1:
                    if len(node.kwargs.keys()) == 0:
                        raise Exception('Passing the arguments by position can be ambiguous. Please pass the arguments by name')
                    #if node.kwargs has the key mask, then remove the node from layer_list and add the mask attribute to current node
                if 'mask' in node.kwargs.keys():
                    mask_node = node.kwargs['mask']
                    if '.' not in mask_node.target:
                        obj = getattr(model, mask_node.name)
                    else:
                        obj = getattr(children[mask_node.target.split('.')[0], mask_node.name])
                    mask = obj.numpy() != 0
                    mask = mask.astype(int)
                    for transenc_layer in layer['layer_list'][0]['layer_list']:
                        for transenc_sublayer in transenc_layer['layer_list']:
                            if transenc_sublayer['class_name'] == 'MultiheadAttention':
                                transenc_sublayer['mask_data'] = mask                        
                    #delete the layer that layer['name'] == mask_node.name from layer_list
                    for i in range(len(layer_list)):
                        if layer_list[i]['name'] == mask_node.name:
                            del layer_list[i]
                            break
                    for i in range(len(input_layers)):
                        if input_layers[i] == mask_node.name:
                            del input_layers[i]
                            break
                else:
                    import numpy as np
                    mask = np.zeros((input_shapes[0][1], input_shapes[0][1]))
                    mask = mask.astype(int)
                    for transenc_layer in layer['layer_list'][0]['layer_list']:
                        for transenc_sublayer in transenc_layer['layer_list']:
                            if transenc_sublayer['class_name'] == 'MultiheadAttention':
                                transenc_sublayer['mask_data'] = mask
            #for multiheadattention
            if 'MultiheadAttention' in pytorch_class:
                if len(input_names) > 3:
                    if len(node.kwargs.keys()) == 0:
                        raise Exception('Passing the arguments by position can be ambiguous. Please pass the arguments by name')
                    #if node.kwargs has the key mask, then remove the node from layer_list and add the mask attribute to current node
                if 'attn_mask' in node.kwargs.keys():
                    mask_node = node.kwargs['attn_mask']
                    if '.' not in mask_node.target:
                        obj = getattr(model, mask_node.name)
                    else:
                        obj = getattr(children[mask_node.target.split('.')[0], mask_node.name])
                    mask = obj.numpy() != 0
                    mask = mask.astype(int)
                    layer['mask_data'] = mask
                    #delete the layer that layer['name'] == mask_node.name from layer_list
                    for i in range(len(layer_list)):
                        if layer_list[i]['name'] == mask_node.name:
                            del layer_list[i]
                            break
                    for i in range(len(input_layers)):
                        if input_layers[i] == mask_node.name:
                            del input_layers[i]
                            break
                else:
                    import numpy as np
                    mask = np.zeros((input_shapes[0][1], input_shapes[0][1]))
                    mask = mask.astype(int)
                    layer['mask_data'] = mask
            #print(
            #    'Layer name: {}, layer type: {}, input shape: {}'.format(
            #        layer['name'],
            #        layer['class_name'],
            #        input_shapes,
            #    )
            #)
            layer_list.append(layer)

            assert output_shape is not None
            output_shapes[layer['name']] = output_shape

            layer_counter += 1

        if node.op == 'placeholder':
            # 'placeholder' indicates an input layer. Multiple inputs are supported

            input_layer = {}
            input_layer['name'] = node.name
            input_layer['class_name'] = 'InputLayer'
            input_layer['input_shape'] = list(input_shapes[n_inputs][1:])
            layer_list.insert(n_inputs, input_layer)

            output_shapes[input_layer['name']] = list(input_shapes[n_inputs])
            input_layers.append(input_layer['name'])
            n_inputs += 1

            layer_counter += 1

        if node.op == 'call_function':
            # Function calls in the graph have to be transformed to layers known to hls4ml

            # operations that appear repeatedly have '_n' appended to their name for the nth repetition
            operation = node.name
            if node.name.split('_')[-1].isdigit():
                operation = '_'.join(node.name.split('_')[:-1])

            if operation in layer_name_map:
                operation = layer_name_map[operation]

            # only a limited number of functions are supported
            if operation not in supported_layers:
                raise Exception(f'Unsupported function {operation}')
            if operation == 'PReLU' or operation == 'batch_norm' or operation == 'conv1d' or operation == 'conv2d':
                raise Exception(
                    f'Function {operation} cannot be parsed as torch.nn.functional. Use the torch.nn implementation instead'
                )

            layer_name = node.name

            layer_counter += 1

            input_names = [inputs_map.get(str(i), str(i)) for i in node.all_input_nodes]
            input_shapes = [list(output_shapes[str(i)]) for i in input_names]

            # Process the layer
            layer, output_shape = layer_handlers[operation](
                operation, layer_name, input_names, input_shapes, node, None, reader, config
            )

            #print('Layer name: {}, layer type: {}, input shape: {}'.format(layer['name'], layer['class_name'], input_shapes))
            layer_list.append(layer)

            assert output_shape is not None
            output_shapes[layer['name']] = output_shape

        if node.op == 'get_attr':
            # Deals with tensors that are member variables of the model class
            # We insert these tensors are input layer nodes into the hls4ML model graph
            if '.' not in node.target:
                obj = getattr(model, node.name)
            else:
                obj = getattr(children[node.target.split('.')[0], node.name])

            input_layer = {}
            input_layer['name'] = node.name
            input_layer['class_name'] = 'InputLayer'
            input_layer['input_shape'] = [None] + list(obj.size())
            layer_list.insert(n_inputs, input_layer)

            output_shapes[input_layer['name']] = [None] + list(obj.size())
            input_layers.append(input_layer['name'])
            n_inputs += 1

            layer_counter += 1

        if node.op == 'call_method':
            # Method calls in the graph have to be transformed to layers known to hls4ml

            # operations that appear repeatedly have '_n' appended to their name for the nth repetition
            operation = node.name
            if node.name.split('_')[-1].isdigit():
                operation = '_'.join(node.name.split('_')[:-1])

            if operation in layer_name_map:
                operation = layer_name_map[operation]

            # only a limited number of functions are supported
            if operation not in supported_layers:
                raise Exception(f'Unsupported function {operation}')

            layer_name = node.name

            layer_counter += 1

            input_names = [inputs_map.get(str(i), str(i)) for i in node.all_input_nodes]

            # Process the layer
            input_shapes = [list(output_shapes[str(i)]) for i in input_names]

            layer, output_shape = layer_handlers[operation](
                operation, layer_name, input_names, input_shapes, node, None, reader, config
            )

            #print('Layer name: {}, layer type: {}, input shape: {}'.format(layer['name'], layer['class_name'], input_shapes))
            layer_list.append(layer)

            assert output_shape is not None
            output_shapes[layer['name']] = output_shape

    if len(input_layers) == 0:
        input_layers = None

    #print('Creating HLS model')
    hls_model = ModelGraph(config, layer_list, inputs=input_layers)
    return hls_model
