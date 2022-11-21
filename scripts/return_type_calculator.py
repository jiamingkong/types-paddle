"""
Calculate the return type of a Paddle API given its implementation details.
"""

from dataclasses import dataclass
import yaml

"""
A typical ops.yaml function declaration is shown below:
- op : unique
  args : (Tensor x, bool return_index, bool return_inverse, bool return_counts, int[] axis, DataType dtype=DataType::INT64)
  output : Tensor(out), Tensor(indices), Tensor(inverse), Tensor(counts)
  infer_meta :
    func : UniqueInferMeta
  kernel :
    func : unique
    data_type : x
"""

def handle_extra_delimiter(return_string):
    """
    There are also some cases where the return type string is not well formatted, such as:
        - unbind: "Tensor[] {axis<0 ? input.dims()[input.dims().size()+axis]:input.dims()[axis]}"
        - update_loss_scaling_: "Tensor[](out){x.size()}, Tensor(loss_scaling), Tensor(out_good_steps), Tensor(out_bad_steps)"
    The idea is to turn Tensor[] into List[Tensor]
    """
    return_string = return_string.replace("Tensor[]", "List[Tensor]")
    return_types = return_string.split(",")
    # then cut by space, parentheses, and brackets
    CUT = [" ", "(", "{"]
    for c in CUT:
        return_types = [rt.strip().split(c)[0] for rt in return_types]
    return return_types



METHODS_NEED_SPECIAL_CARE = ["unbind", "update_loss_scaling_", "split", "split_with_num", "rnn", "broadcast_tensors", "unstack"]


METHODS_KNOWN = {
    "topk": "Tuple[Tensor, Tensor]",
    "T": "Tensor",
    "astype": "Tensor",
    "broadcast_to": "Tensor",
    "broadcast_shape": "Tuple[int]",
    "clone": "Tensor",
    "cond": "None",
    "cov": "None",
    "cpu": "Tensor",
    "cuda": "Tensor",
    "copy_": "None",
    "diff": "Tensor",
    "floor_mod": "Tensor",
    # a scalar is returned for item, but we don't have a type for scalar
    "item": "Any",
    "median": "Tensor",
    "mm": "Tensor",
    "mod": "Tensor",
    "moveaxis": "Tensor",
    "nanmean": "Tensor",
    "nansum": "Tensor",
    "ndimension": "int",
    "neg": "Tensor",
    "nonzero": "Tensor",
    "numel": "int",
    "outer": "Tensor",
    "numpy": "numpy.ndarray",
    "prod": "Tensor",
    "quantile": "Tensor",
    "rad2deg": "Tensor",
    "rank": "Tensor",
    # 文档中写rot90 会返回Tensor或者LoDTensor
    "rot90": "Tensor",
    "sort": "Tensor",
    "stop_gradient": "None",
    "to_dense": "Tensor",
    "to_sparse_coo": "Tensor",
    "values": "Any",
}


class ReturnType:

    def __init__(self, function_name, returns = None):
        self.function_name = function_name
        if returns is None:
            self.returns = []
        else:
            self.returns = returns

    @classmethod
    def from_yaml(cls, yaml_dict):
        # get the output field
        output = yaml_dict["output"]
        # get name
        function_name = yaml_dict["op"]

        # split the output field by comma and remove the variable name in parenthesis
        if function_name in METHODS_NEED_SPECIAL_CARE:
            returns = handle_extra_delimiter(output)
        else:
            returns = output.split(",")
            returns = [ret.split("(")[0].strip() for ret in returns]
        return cls(function_name, returns)

    def to_return_string(self):
        if len(self.returns) == 1:
            return self.returns[0]
        else:
            return f"Tuple[{', '.join(self.returns)}]"

    def to_return_annotation(self):
        return f"def {self.function_name} -> {self.to_return_string()}: ..."

    def __repr__(self):
        return self.to_return_annotation()

    def guess_return_type(self):
        """
        guess the return types from the naming custom of the function
        - if ends with "_", then it's an inplace function, return None
        - if startswith "is_", then it's a bool function, return bool
        """
        if self.function_name.endswith("_"):
            return None
        elif self.function_name.startswith("is_"):
            return "bool"
        elif self.function_name in METHODS_KNOWN:
            return METHODS_KNOWN[self.function_name]
        return self.to_return_string()

def calculate_return_type_ops_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        ops_list = yaml.load(f, Loader=yaml.FullLoader)
        return_types = []
        for op in ops_list:
            rt = ReturnType.from_yaml(op)
            return_types.append(rt)

        return_types_dict= {rt.function_name: rt for rt in return_types}
        return return_types_dict



if __name__ == '__main__':
#     typical_example = """
# - op : unique
#   args : (Tensor x, bool return_index, bool return_inverse, bool return_counts, int[] axis, DataType dtype=DataType::INT64)
#   output : Tensor(out), Tensor(indices), Tensor(inverse), Tensor(counts)
#   infer_meta :
#     func : UniqueInferMeta
#   kernel :
#     func : unique
#     data_type : x
# """

#     yaml_ops = yaml.load(typical_example, Loader=yaml.FullLoader)
#     yaml_dict = yaml_ops[0]
#     return_type = ReturnType.from_yaml(yaml_dict)
#     print(return_type.to_return_annotation())

    # for the full example: 
    result1 = calculate_return_type_ops_yaml("ops.yaml")
    result2 = calculate_return_type_ops_yaml("legacy_ops.yaml")
    result = result1 + result2
    for r in result:
        print(r)