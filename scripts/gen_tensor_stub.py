from __future__ import annotations
import json
import inspect
from typing import Any, List, Tuple
import paddle
from tabulate import tabulate


def get_tensor_memebers() -> List[str]:
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
            lines.append(
                f"def {name}(self, {str(arg_spec)[1:]}: ..."
            )
        except:
            continue
    return lines


def build_tensor_class(output_file: str):
    """build tensor class file
    """
    # 1. get the runtime tensor public method
    lines = get_tensor_memebers()
    
    # 2. build the tensor class file
    tab = '    '
    content = "class Tensor:\n"
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
    build_tensor_class("../paddle-stubs/tensor.pyi")
