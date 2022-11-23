from paddle.tensor.math import *
import paddle
import inspect

def retrieve_inline_documentation(func_name):
    """
    for the given func_name, use inspect to retrieve the docstring defined in paddle.tensor.math
    """
    # func = getattr(paddle.tensor.math, func_name)
    # see if func is indeed in paddle.tensor.math
    for module in [paddle.tensor.math, paddle.tensor.random, paddle.tensor.search, paddle.tensor.stat, paddle.tensor.manipulation, paddle.tensor.creation, paddle.tensor.logic, paddle.tensor.linalg]:
        if hasattr(module, func_name):
            func = getattr(module, func_name)
            return inspect.getdoc(func)
    else:
        return None

def simple_rewriter(docstring):
    """
    The docstring uses x to represent the input tensor, in the Tensor.FUNC_NAME docstring, we replace x with self
    A typical writing will say:
    "
    Args:
        x (Tensor): Input of acos operator
    "
    and we should replace x with self
    "
    Args:
        self (Tensor): Input of acos operator
    "
    """
    if docstring is None:
        return None
    return docstring.replace("x (Tensor)", "self (Tensor)").replace("x(Tensor)", "self (Tensor)")



if __name__ == '__main__':
    print(retrieve_inline_documentation("acos"))