from hls4ml.backends.fpga.fpga_types import (
    APTypeConverter,
    HLSTypeConverter,
    StaticWeightVariableConverter,
    VivadoArrayVariableConverter,
    VivadoInplaceArrayVariableConverter,
    VivadoInplaceStreamVariableConverter,
    VivadoStreamVariableConverter,
)
from hls4ml.model.optimizer import GlobalOptimizerPass
from hls4ml.model.types import InplaceTensorVariable


class TransformTypes(GlobalOptimizerPass):
    def __init__(self):
        self.type_converter = HLSTypeConverter(precision_converter=APTypeConverter())
        self.array_var_converter = VivadoArrayVariableConverter(type_converter=self.type_converter)
        self.inplace_array_var_converter = VivadoInplaceArrayVariableConverter(type_converter=self.type_converter)
        self.stream_var_converter = VivadoStreamVariableConverter(type_converter=self.type_converter)
        self.inplace_stream_var_converter = VivadoInplaceStreamVariableConverter(type_converter=self.type_converter)
        self.weight_var_converter = StaticWeightVariableConverter(type_converter=self.type_converter)

    def transform(self, model, node):
        io_type = node.model.config.get_config_value('IOType')
        #print all nodes in model
        #print('ssssssssssssssssssss')
        #for n in model.get_layers():
        #    print(n.name)
        #    for k,v in n.get_output_variable().get_shape():
        #        print(k,v) 

        for out_name, var in node.variables.items():
            if io_type == 'io_stream':
                if isinstance(var, InplaceTensorVariable):
                    new_var = self.inplace_stream_var_converter.convert(var)
                else:
                    new_var = self.stream_var_converter.convert(var)
            elif io_type == 'io_tile_stream':
                tf = node.get_attr('tiling_factor')
                if tf is None:
                    raise Exception(f'No tiling factor found in {node.name} ({node.__class__.__name__})')
                tf_T = tf[0]
                tf_N = tf[1]
                var.shape = [var.shape[0]//tf_N, var.shape[1]//tf_T, tf_T*tf_N]
                new_var = self.stream_var_converter.convert(var)
            elif io_type == 'io_serial':
                new_var = self.array_var_converter.convert(var, pragma='stream')
            elif io_type == 'io_parallel':
                if out_name in node.model.inputs:
                    new_var = self.array_var_converter.convert(var, pragma='reshape')
                elif isinstance(var, InplaceTensorVariable):
                    new_var = self.inplace_array_var_converter.convert(var, pragma='')
                else:
                    new_var = self.array_var_converter.convert(var, pragma='partition')
            else:
                raise Exception(f'Unknown IOType {io_type} in {node.name} ({node.__class__.__name__})')

            node.set_attr(out_name, new_var)

        for w_name, weight in node.weights.items():
            new_weight = self.weight_var_converter.convert(weight)
            node.set_attr(w_name, new_weight)

        for t_name, type in node.types.items():
            new_type = self.type_converter.convert(type)
            node.set_attr(t_name, new_type)
        
        #print('aaaaaaaaaaaaaaaaaaaaa')
        #for n in model.get_layers():
        #    print(n.name)
        #    for k,v in n.get_output_variable().get_shape():
        #        print(k,v) 
