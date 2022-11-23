from __future__ import annotations
import json
import inspect
from typing import Any, List, Tuple
import paddle
from tabulate import tabulate
from return_type_calculator import ReturnType, calculate_return_type_ops_yaml
from docstring_retrieval import retrieve_inline_documentation, simple_rewriter

declaration = """#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""

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
            result = f"def {name}(self, {str(arg_spec)[1:]} -> {return_string}:\nPLACEHOLDER_FOR_DOC"
            # get documentation as well:
            inline_docstring = simple_rewriter(retrieve_inline_documentation(name))
            
            if inline_docstring is not None:
                result = result.replace("PLACEHOLDER_FOR_DOC", '        """\n        ' + inline_docstring.replace("\n", "\n        ") + '        """\n' + " "*8 + "pass\n\n")
            else:
                result = result.replace("PLACEHOLDER_FOR_DOC", " "* 8 + "pass")
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
    content = declaration
    content += "from typing import Optional, List, Tuple, Union\nfrom __future__ import annotations\n\nclass Tensor:\n"
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
    build_tensor_class("./tensor_proxy.py")
