import os
from random import shuffle
import shutil


PACKAGE_DIR = os.path.dirname(__file__)

PADDLE_DIR = os.path.join(
    PACKAGE_DIR,
    "../",
    "paddle"
)


def replace_tensor_file():
    """replace the source tensor/tensor.py file to make it executable
    """
    with open(os.path.join(PACKAGE_DIR, 'tensor_proxy.py'), "r", encoding='utf-8') as f:
        target_tensor_content = f.read()
    
    with open(os.path.join(PADDLE_DIR, 'tensor', 'tensor.py'), 'w', encoding='utf-8') as f:
        f.write(target_tensor_content)

PATCH = """

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tensor.tensor_proxy import Tensor as Tensor
"""

def patch_init_file():
    """
    patch the __init__.py file at the paddle dir to apply the TYPE_CHECKING trick.
    """
    with open(os.path.join(PADDLE_DIR, '__init__.py'), 'r', encoding='utf-8') as f:
        init_content = f.read()
    if "TYPE_CHECKING" in init_content:
        return
    with open(os.path.join(PADDLE_DIR, '__init__.py'), 'w', encoding='utf-8') as f:
        f.write(init_content + PATCH)
    

def add_tensor_proxy_file():
    """add tensor.pyi file to the target dir to make it intelligence in IDE
    """
    shutil.copyfile(
        os.path.join(PACKAGE_DIR, 'tensor_proxy.py'),
        os.path.join(PADDLE_DIR, 'tensor', 'tensor_proxy.py')
    )


def main():
    replace_tensor_file()
    patch_init_file()
    add_tensor_proxy_file()