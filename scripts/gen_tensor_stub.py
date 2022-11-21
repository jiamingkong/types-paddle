from __future__ import annotations
import json
import inspect
from typing import Any, List, Tuple
import paddle
from tabulate import tabulate
from return_type_calculator import ReturnType, calculate_return_type_ops_yaml


def get_tensor_memebers(return_types = None, guess = False) -> List[str]:
    """get tensor runtime public method

    Returns:
        List[str]: the generated public method
    """
    tensor = paddle.randn([2,3])

    lines = []
    members = inspect.getmembers(tensor)
    for name, _ in members:
        if name.startswith('_'):
            continue
        try:
            method = getattr(tensor, name)
            arg_spec = inspect.signature(method)
            if return_types is not None and name in return_types:
                return_string = return_types[name].to_return_string()
                print(f"CALC:  {name} -> {return_string}")
            elif guess:
                return_string = ReturnType(name).guess_return_type()
                print(f"GUESS: {name} -> {return_string}")
            else:
                return_string = "Any"
            result = f"def {name}(self, {str(arg_spec)[1:]} -> {return_string}: ..."
            lines.append(result)
        except TypeError:
            print(f"Skipping {name} for it is not callable")
        except ValueError:
            print(f"Skipping {name} for there is no signature")
        # except Exception as e:
        #     e.print
        #     continue
    return lines


def build_tensor_class(output_file: str):
    """build tensor class file
    """
    # 0. calculate the return types of the paddle api
    return_types = calculate_return_type_ops_yaml("./legacy_ops.yaml")
    return_types.update(calculate_return_type_ops_yaml("./ops.yaml"))
    print(len(return_types))
    # 1. get the runtime tensor public method
    lines = get_tensor_memebers(return_types, guess = True)
    
    # 2. build the tensor class file
    tab = '    '
    content = "from typing import Optional, List, Tuple\n\nclass Tensor:\n"
    for line in lines:
        content += f'{tab}{line}\n'
    
    # 3. global replace from config.json file
    with open('./config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)

    for before, after in config.get("terms", []):
        content = content.replace(before, after)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    build_tensor_class("./tensor.pyi")
