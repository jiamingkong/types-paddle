#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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


from typing import Optional, List, Tuple, Union
from __future__ import annotations

class Tensor:
    def abs(self, name: Optional[str] = None) -> Tensor:
        """
        Abs Operator.
        
        This operator is used to perform elementwise abs for input $X$.
        :math:`out = |x|`
        
        
        Args:
            self (Tensor): (Tensor), The input tensor of abs op.
            with_quant_attr (BOOLEAN): Whether the operator has attributes used by quantization. 
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            out (Tensor): (Tensor), The output tensor of abs op.
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
                out = paddle.abs(x)
                print(out)
                # [0.4 0.2 0.1 0.3]        """
        pass


    def acos(self, name: Optional[str] = None) -> Tensor:
        """
        Arccosine Operator.
        
        :math:`out = \cos^{-1}(x)`
        
        
        Args:
            self (Tensor): Input of acos operator
            with_quant_attr (BOOLEAN): Whether the operator has attributes used by quantization. 
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            out (Tensor): Output of acos operator
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
                out = paddle.acos(x)
                print(out)
                # [1.98231317 1.77215425 1.47062891 1.26610367]        """
        pass


    def acosh(self, name: Optional[str] = None) -> Tensor:
        """
        Acosh Activation Operator.
        
        :math:`out = acosh(x)`
        
        
        Args:
            self (Tensor): Input of Acosh operator, an N-D Tensor, with data type float32, float64 or float16.
            with_quant_attr (BOOLEAN): Whether the operator has attributes used by quantization. 
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            out (Tensor): Output of Acosh operator, a Tensor with shape same as input.
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([1., 3., 4., 5.])
                out = paddle.acosh(x)
                print(out)
                # [0.        , 1.76274729, 2.06343699, 2.29243159]        """
        pass


    def add(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        """
        Elementwise Add Operator.
        
        Add two tensors element-wise
        
        The equation is:
        
        :math:`Out = X + Y`
        
        - $X$: a tensor of any dimension.
        - $Y$: a tensor whose dimensions must be less than or equal to the dimensions of $X$.
        
        There are two cases for this operator:
        
        1. The shape of $Y$ is the same with $X$.
        2. The shape of $Y$ is a continuous subsequence of $X$.
        
        For case 2:
        
        1. Broadcast $Y$ to match the shape of $X$, where $axis$ is the start dimension index
           for broadcasting $Y$ onto $X$.
        2. If $axis$ is -1 (default), $axis = rank(X) - rank(Y)$.
        3. The trailing dimensions of size 1 for $Y$ will be ignored for the consideration of
           subsequence, such as shape(Y) = (2, 1) => (2).
        
        For example:
        
          .. code-block:: text
        
            shape(X) = (2, 3, 4, 5), shape(Y) = (,)
            shape(X) = (2, 3, 4, 5), shape(Y) = (5,)
            shape(X) = (2, 3, 4, 5), shape(Y) = (4, 5), with axis: int = -1(default) or axis: int = 2
            shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis: int = 1
            shape(X) = (2, 3, 4, 5), shape(Y) = (2), with axis: int = 0
            shape(X) = (2, 3, 4, 5), shape(Y) = (2, 1), with axis: int = 0
        
        
        Args:
            self (Tensor): (Variable), Tensor or LoDTensor of any dimensions. Its dtype should be int32, int64, float32, float64.
            y (Tensor): (Variable), Tensor or LoDTensor of any dimensions. Its dtype should be int32, int64, float32, float64.
            with_quant_attr (BOOLEAN): Whether the operator has attributes used by quantization. 
            name (string, optional): Name of the output.         Default is None. It's used to print debug info for developers. Details:         :ref:`api_guide_Name`
        
        Returns:
            out (Tensor): N-dimension tensor. A location into which the result is stored. It's dimension equals with x
        
            Examples:
        
            ..  code-block:: python
        
                import paddle
                x = paddle.to_tensor([2, 3, 4], 'float64')
                y = paddle.to_tensor([1, 5, 2], 'float64')
                z = paddle.add(x, y)
                print(z)  # [3., 8., 6. ]
        
                    """
        pass


    def add_(self, y: Tensor, name: Optional[str] = None) -> None:
        """
        Inplace version of ``add`` API, the output Tensor will be inplaced with input ``x``.
        Please refer to :ref:`api_tensor_add`.        """
        pass


    def add_n(self, name: Optional[str] = None) -> Tensor:
        """
        This OP is used to sum one or more Tensor of the input.
        
        For example:
        
        .. code-block:: text
        
            Case 1:
        
                Input:
                    input.shape = [2, 3]
                    input = [[1, 2, 3],
                             [4, 5, 6]]
        
                Output:
                    output.shape = [2, 3]
                    output = [[1, 2, 3],
                              [4, 5, 6]]
        
            Case 2:
           
                Input:
                    First input:
                        input1.shape = [2, 3]
                        Input1 = [[1, 2, 3],
                                  [4, 5, 6]]
        
                    The second input:
                        input2.shape = [2, 3]
                        input2 = [[7, 8, 9],
                                  [10, 11, 12]]
        
                    Output:
                        output.shape = [2, 3]
                        output = [[8, 10, 12],
                                  [14, 16, 18]]
        
        Args:
            inputs (Tensor|list[Tensor]|tuple[Tensor]):  A Tensor or a list/tuple of Tensors. The shape and data type of the list/tuple elements should be consistent.
                Input can be multi-dimensional Tensor, and data types can be: float32, float64, int32, int64.
            name(str, optional): The default value is None. Normally there is no need for
                user to set this property. For more information, please refer to :ref:`api_guide_Name`
        
        Returns:
            Tensor, the sum of input :math:`inputs` , its shape and data types are consistent with :math:`inputs`.
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                input0 = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype='float32')
                input1 = paddle.to_tensor([[7, 8, 9], [10, 11, 12]], dtype='float32')
                output = paddle.add_n([input0, input1])
                # [[8., 10., 12.], 
                #  [14., 16., 18.]]        """
        pass


    def addmm(self, x, y, beta=1.0, alpha=1.0, name: Optional[str] = None) -> Tensor:
        """
        **addmm**
        
        This operator is used to perform matrix multiplication for input $x$ and $y$.
        $input$ is added to the final result.
        The equation is:
        
        ..  math::
            Out = alpha * x * y + beta * input
        
        $Input$, $x$ and $y$ can carry the LoD (Level of Details) information, or not. But the output only shares the LoD information with input $input$.
        
        Args:
            input (Tensor): The input Tensor to be added to the final result.
            self (Tensor): The first input Tensor for matrix multiplication.
            y (Tensor): The second input Tensor for matrix multiplication.
            beta (float): Coefficient of $input$.
            alpha (float): Coefficient of $x*y$.
            name (str, optional): Name of the output. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`. Default is None.
        
        Returns:
            Tensor: The output Tensor of addmm op.
        
        Examples:
            ..  code-block:: python
                
                import paddle
        
                x = paddle.ones([2,2])
                y = paddle.ones([2,2])
                input = paddle.ones([2,2])
        
                out = paddle.addmm( input=input, x=x, y=y, beta=0.5, alpha=5.0 )
        
                print(out)
                # [[10.5 10.5]
                # [10.5 10.5]]        """
        pass


    def all(self, axis: Optional[int] = None, keepdim: bool = False, name: Optional[str] = None) -> Tensor:
        """
        Computes the the ``logical and`` of tensor elements over the given dimension.
        
        Args:
            self (Tensor): An N-D Tensor, the input data type should be `bool`.
            axis (int|list|tuple, optional): The dimensions along which the ``logical and`` is compute. If
                :attr:`None`, and all elements of :attr:`x` and return a
                Tensor with a single element, otherwise must be in the
                range :math:`[-rank(x), rank(x))`. If :math:`axis[i] < 0`,
                the dimension to reduce is :math:`rank + axis[i]`.
            keepdim (bool, optional): Whether to reserve the reduced dimension in the
                output Tensor. The result Tensor will have one fewer dimension
                than the :attr:`x` unless :attr:`keepdim` is true, default
                value is False.
            name (str, optional): The default value is None. Normally there is no need for
                user to set this property.  For more information, please refer to :ref:`api_guide_Name`
        
        Returns:
            Tensor: Results the ``logical and`` on the specified axis of input Tensor `x`,  it's data type is bool.
        
        Raises:
            ValueError: If the data type of `x` is not bool.
            TypeError: The type of :attr:`axis` must be int, list or tuple.
        
        Examples:
            .. code-block:: python
        
                import paddle
                import numpy as np
                
                # x is a bool Tensor with following elements:
                #    [[True, False]
                #     [True, True]]
                x = paddle.assign(np.array([[1, 0], [1, 1]], dtype='int32'))
                print(x)
                x = paddle.cast(x, 'bool')
                
                # out1 should be [False]
                out1 = paddle.all(x)  # [False]
                print(out1)
                
                # out2 should be [True, False]
                out2 = paddle.all(x, axis: int = 0)  # [True, False]
                print(out2)
                
                # keep_dim=False, out3 should be [False, True], out.shape should be (2,)
                out3 = paddle.all(x, axis: int = -1)  # [False, True]
                print(out3)
                
                # keep_dim=True, out4 should be [[False], [True]], out.shape should be (2,1)
                out4 = paddle.all(x, axis: int = 1, keepdim=True)
                out4 = paddle.cast(out4, 'int32')  # [[False], [True]]
                print(out4)
                        """
        pass


    def allclose(self, y: Tensor, rtol=1e-05, atol=1e-08, equal_nan=False, name: Optional[str] = None) -> Tensor:
        """
        This operator checks if all :math:`x` and :math:`y` satisfy the condition: 
        
        .. math:: \left| x - y \right| \leq atol + rtol \times \left| y \right| 
        
        elementwise, for all elements of :math:`x` and :math:`y`. The behaviour of this operator is analogous to :math:`numpy.allclose`, namely that it returns :math:`True` if two tensors are elementwise equal within a tolerance. 
        
        
        
        Args:
            self (Tensor): The input tensor, it's data type should be float32, float64.
            y(Tensor): The input tensor, it's data type should be float32, float64.
            rtol(rtoltype, optional): The relative tolerance. Default: :math:`1e-5` .
            atol(atoltype, optional): The absolute tolerance. Default: :math:`1e-8` .
            equal_nan(equalnantype, optional): If :math:`True` , then two :math:`NaNs` will be compared as equal. Default: :math:`False` .
            name (str, optional): Name for the operation. For more information, please
                refer to :ref:`api_guide_Name`. Default: None.
        
        Returns:
            Tensor: The output tensor, it's data type is bool.
        
        Raises:
            TypeError: The data type of ``x`` must be one of float32, float64.
            TypeError: The data type of ``y`` must be one of float32, float64.
            TypeError: The type of ``rtol`` must be float.
            TypeError: The type of ``atol`` must be float.
            TypeError: The type of ``equal_nan`` must be bool.
        
        Examples:
            .. code-block:: python
        
              import paddle
        
              x = paddle.to_tensor([10000., 1e-07])
              y = paddle.to_tensor([10000.1, 1e-08])
              result1 = paddle.allclose(x, y, rtol=1e-05, atol=1e-08,
                                      equal_nan=False, name="ignore_nan")
              np_result1 = result1.numpy()
              # [False]
              result2 = paddle.allclose(x, y, rtol=1e-05, atol=1e-08,
                                          equal_nan=True, name="equal_nan")
              np_result2 = result2.numpy()
              # [False]
        
              x = paddle.to_tensor([1.0, float('nan')])
              y = paddle.to_tensor([1.0, float('nan')])
              result1 = paddle.allclose(x, y, rtol=1e-05, atol=1e-08,
                                      equal_nan=False, name="ignore_nan")
              np_result1 = result1.numpy()
              # [False]
              result2 = paddle.allclose(x, y, rtol=1e-05, atol=1e-08,
                                          equal_nan=True, name="equal_nan")
              np_result2 = result2.numpy()
              # [True]        """
        pass


    def amax(self, axis: Optional[int] = None, keepdim: bool = False, name: Optional[str] = None) -> Tensor:
        """
        Computes the maximum of tensor elements over the given axis.
        
        Note:
            The difference between max and amax is: If there are multiple maximum elements,
            amax evenly distributes gradient between these equal values, 
            while max propagates gradient to all of them.
        
        Args:
            self (Tensor): A tensor, the data type is float32, float64, int32, int64,
                the dimension is no more than 4.
            axis(int|list|tuple, optional): The axis along which the maximum is computed.
                If :attr:`None`, compute the maximum over all elements of
                `x` and return a Tensor with a single element,
                otherwise must be in the range :math:`[-x.ndim(x), x.ndim(x))`.
                If :math:`axis[i] < 0`, the axis to reduce is :math:`x.ndim + axis[i]`.
            keepdim(bool, optional): Whether to reserve the reduced dimension in the
                output Tensor. The result tensor will have one fewer dimension
                than the `x` unless :attr:`keepdim` is true, default
                value is False.
            name(str, optional): The default value is None.  Normally there is no need for
                user to set this property.  For more information, please refer to :ref:`api_guide_Name`
        
        Returns:
            Tensor, results of maximum on the specified axis of input tensor,
            it's data type is the same as `x`.
        
        Examples:
            .. code-block:: python
        
                import paddle
                # data_x is a Tensor with shape [2, 4] with multiple maximum elements
                # the axis is a int element
        
                x = paddle.to_tensor([[0.1, 0.9, 0.9, 0.9],
                                      [0.9, 0.9, 0.6, 0.7]], 
                                     dtype='float64', stop_gradient=False)
                # There are 5 maximum elements: 
                # 1) amax evenly distributes gradient between these equal values, 
                #    thus the corresponding gradients are 1/5=0.2;
                # 2) while max propagates gradient to all of them, 
                #    thus the corresponding gradient are 1.
                result1 = paddle.amax(x)
                result1.backward()
                print(result1, x.grad) 
                #[0.9], [[0., 0.2, 0.2, 0.2], [0.2, 0.2, 0., 0.]]
        
                x.clear_grad()
                result1_max = paddle.max(x)
                result1_max.backward()
                print(result1_max, x.grad) 
                #[0.9], [[0., 1.0, 1.0, 1.0], [1.0, 1.0, 0., 0.]]
        
                ###############################
        
                x.clear_grad()
                result2 = paddle.amax(x, axis: int = 0)
                result2.backward()
                print(result2, x.grad) 
                #[0.9, 0.9, 0.9, 0.9], [[0., 0.5, 1., 1.], [1., 0.5, 0., 0.]]
        
                x.clear_grad()
                result3 = paddle.amax(x, axis: int = -1)
                result3.backward()
                print(result3, x.grad) 
                #[0.9, 0.9], [[0., 0.3333, 0.3333, 0.3333], [0.5, 0.5, 0., 0.]]
        
                x.clear_grad()
                result4 = paddle.amax(x, axis: int = 1, keepdim=True)
                result4.backward()
                print(result4, x.grad) 
                #[[0.9], [0.9]], [[0., 0.3333, 0.3333, 0.3333.], [0.5, 0.5, 0., 0.]]
        
                # data_y is a Tensor with shape [2, 2, 2]
                # the axis is list 
                y = paddle.to_tensor([[[0.1, 0.9], [0.9, 0.9]],
                                      [[0.9, 0.9], [0.6, 0.7]]],
                                     dtype='float64', stop_gradient=False)
                result5 = paddle.amax(y, axis: int = [1, 2])
                result5.backward()
                print(result5, y.grad) 
                #[0.9., 0.9], [[[0., 0.3333], [0.3333, 0.3333]], [[0.5, 0.5], [0., 1.]]]
        
                y.clear_grad()
                result6 = paddle.amax(y, axis: int = [0, 1])
                result6.backward()
                print(result6, y.grad) 
                #[0.9., 0.9], [[[0., 0.3333], [0.5, 0.3333]], [[0.5, 0.3333], [1., 1.]]]        """
        pass


    def amin(self, axis: Optional[int] = None, keepdim: bool = False, name: Optional[str] = None) -> Tensor:
        """
        Computes the minimum of tensor elements over the given axis
        
        Note:
            The difference between min and amin is: If there are multiple minimum elements,
            amin evenly distributes gradient between these equal values, 
            while min propagates gradient to all of them.
        
        Args:
            self (Tensor): A tensor, the data type is float32, float64, int32, int64, 
                the dimension is no more than 4.
            axis(int|list|tuple, optional): The axis along which the minimum is computed.
                If :attr:`None`, compute the minimum over all elements of
                `x` and return a Tensor with a single element,
                otherwise must be in the range :math:`[-x.ndim, x.ndim)`.
                If :math:`axis[i] < 0`, the axis to reduce is :math:`x.ndim + axis[i]`.
            keepdim(bool, optional): Whether to reserve the reduced dimension in the
                output Tensor. The result tensor will have one fewer dimension
                than the `x` unless :attr:`keepdim` is true, default
                value is False.
            name(str, optional): The default value is None.  Normally there is no need for 
                user to set this property.  For more information, please refer to :ref:`api_guide_Name`
        
        Returns:
            Tensor, results of minimum on the specified axis of input tensor,
            it's data type is the same as input's Tensor.
        
        Examples:
            .. code-block:: python
        
                import paddle
                # data_x is a Tensor with shape [2, 4] with multiple minimum elements
                # the axis is a int element
        
                x = paddle.to_tensor([[0.2, 0.1, 0.1, 0.1],
                                      [0.1, 0.1, 0.6, 0.7]], 
                                     dtype='float64', stop_gradient=False)
                # There are 5 minimum elements: 
                # 1) amin evenly distributes gradient between these equal values, 
                #    thus the corresponding gradients are 1/5=0.2;
                # 2) while min propagates gradient to all of them, 
                #    thus the corresponding gradient are 1.
                result1 = paddle.amin(x)
                result1.backward()
                print(result1, x.grad) 
                #[0.1], [[0., 0.2, 0.2, 0.2], [0.2, 0.2, 0., 0.]]
        
                x.clear_grad()
                result1_min = paddle.min(x)
                result1_min.backward()
                print(result1_min, x.grad) 
                #[0.1], [[0., 1.0, 1.0, 1.0], [1.0, 1.0, 0., 0.]]
        
                ###############################
        
                x.clear_grad()
                result2 = paddle.amin(x, axis: int = 0)
                result2.backward()
                print(result2, x.grad) 
                #[0.1, 0.1, 0.1, 0.1], [[0., 0.5, 1., 1.], [1., 0.5, 0., 0.]]
        
                x.clear_grad()
                result3 = paddle.amin(x, axis: int = -1)
                result3.backward()
                print(result3, x.grad) 
                #[0.1, 0.1], [[0., 0.3333, 0.3333, 0.3333], [0.5, 0.5, 0., 0.]]
        
                x.clear_grad()
                result4 = paddle.amin(x, axis: int = 1, keepdim=True)
                result4.backward()
                print(result4, x.grad) 
                #[[0.1], [0.1]], [[0., 0.3333, 0.3333, 0.3333.], [0.5, 0.5, 0., 0.]]
        
                # data_y is a Tensor with shape [2, 2, 2]
                # the axis is list 
                y = paddle.to_tensor([[[0.2, 0.1], [0.1, 0.1]],
                                      [[0.1, 0.1], [0.6, 0.7]]],
                                     dtype='float64', stop_gradient=False)
                result5 = paddle.amin(y, axis: int = [1, 2])
                result5.backward()
                print(result5, y.grad) 
                #[0.1., 0.1], [[[0., 0.3333], [0.3333, 0.3333]], [[0.5, 0.5], [0., 1.]]]
        
                y.clear_grad()
                result6 = paddle.amin(y, axis: int = [0, 1])
                result6.backward()
                print(result6, y.grad) 
                #[0.1., 0.1], [[[0., 0.3333], [0.5, 0.3333]], [[0.5, 0.3333], [1., 1.]]]        """
        pass


    def angle(self, name: Optional[str] = None) -> Tensor:
        """
        Element-wise angle of complex numbers. For non-negative real numbers, the angle is 0 while 
        for negative real numbers, the angle is :math:`\pi`.
        
        Equation:
            .. math::
        
                angle(x)=arctan2(x.imag, x.real)
        
        Args:
            self (Tensor): An N-D Tensor, the data type is complex64, complex128, or float32, float64 .
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            Tensor: An N-D Tensor of real data type with the same precision as that of x's data type.
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([-2, -1, 0, 1]).unsqueeze(-1).astype('float32')
                y = paddle.to_tensor([-2, -1, 0, 1]).astype('float32')
                z = x + 1j * y
                print(z.numpy())
                # [[-2.-2.j -2.-1.j -2.+0.j -2.+1.j]
                #  [-1.-2.j -1.-1.j -1.+0.j -1.+1.j]
                #  [ 0.-2.j  0.-1.j  0.+0.j  0.+1.j]
                #  [ 1.-2.j  1.-1.j  1.+0.j  1.+1.j]]
        
                theta = paddle.angle(z)
                print(theta.numpy())
                # [[-2.3561945 -2.6779451  3.1415927  2.6779451]
                #  [-2.0344439 -2.3561945  3.1415927  2.3561945]
                #  [-1.5707964 -1.5707964  0.         1.5707964]
                #  [-1.1071488 -0.7853982  0.         0.7853982]]        """
        pass


    def any(self, axis: Optional[int] = None, keepdim: bool = False, name: Optional[str] = None) -> Tensor:
        """
        Computes the the ``logical or`` of tensor elements over the given dimension.
        
        Args:
            self (Tensor): An N-D Tensor, the input data type should be `bool`.
            axis (int|list|tuple, optional): The dimensions along which the ``logical or`` is compute. If
                :attr:`None`, and all elements of :attr:`x` and return a
                Tensor with a single element, otherwise must be in the
                range :math:`[-rank(x), rank(x))`. If :math:`axis[i] < 0`,
                the dimension to reduce is :math:`rank + axis[i]`.
            keepdim (bool, optional): Whether to reserve the reduced dimension in the
                output Tensor. The result Tensor will have one fewer dimension
                than the :attr:`x` unless :attr:`keepdim` is true, default
                value is False.
            name (str, optional): The default value is None. Normally there is no need for
                user to set this property.  For more information, please refer to :ref:`api_guide_Name`
        
        Returns:
            Tensor: Results the ``logical or`` on the specified axis of input Tensor `x`,  it's data type is bool.
        
        Raises:
            ValueError: If the data type of `x` is not bool.
            TypeError: The type of :attr:`axis` must be int, list or tuple.
        
        Examples:
            .. code-block:: python
        
                import paddle
                import numpy as np
                
                # x is a bool Tensor with following elements:
                #    [[True, False]
                #     [False, False]]
                x = paddle.assign(np.array([[1, 0], [1, 1]], dtype='int32'))
                print(x)
                x = paddle.cast(x, 'bool')
                
                # out1 should be [True]
                out1 = paddle.any(x)  # [True]
                print(out1)
                
                # out2 should be [True, True]
                out2 = paddle.any(x, axis: int = 0)  # [True, True]
                print(out2)
                
                # keep_dim=False, out3 should be [True, True], out.shape should be (2,)
                out3 = paddle.any(x, axis: int = -1)  # [True, True]
                print(out3)
                
                # keep_dim=True, result should be [[True], [True]], out.shape should be (2,1)
                out4 = paddle.any(x, axis: int = 1, keepdim=True)
                out4 = paddle.cast(out4, 'int32')  # [[True], [True]]
                print(out4)
                        """
        pass


    def argmax(self, axis: Optional[int] = None, keepdim: bool = False, dtype='int64', name: Optional[str] = None) -> Tensor:
        """
        This OP computes the indices of the max elements of the input tensor's
        element along the provided axis.
        
        Args:
            self (Tensor): An input N-D Tensor with type float32, float64, int16,
                int32, int64, uint8.
            axis(int, optional): Axis to compute indices along. The effective range
                is [-R, R), where R is x.ndim. when axis < 0, it works the same way
                as axis + R. Default is None, the input `x` will be into the flatten tensor, and selecting the min value index.
            keepdim(bool, optional): Whether to keep the given axis in output. If it is True, the dimensions will be same as input x and with size one in the axis. Otherwise the output dimentions is one fewer than x since the axis is squeezed. Default is False.
            dtype(str|np.dtype, optional): Data type of the output tensor which can
                        be int32, int64. The default value is 'int64', and it will
                        return the int64 indices.
            name(str, optional): The default value is None. Normally there is no
                need for user to set this property. For more information, please
                refer to :ref:`api_guide_Name`.
        
        Returns:
            Tensor, return the tensor of `int32` if set :attr:`dtype` is `int32`, otherwise return the tensor of `int64`
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                x =  paddle.to_tensor([[5,8,9,5],
                                         [0,0,1,7],
                                         [6,9,2,4]])
                out1 = paddle.argmax(x)
                print(out1) # 2
                out2 = paddle.argmax(x, axis: int = 0)
                print(out2) 
                # [2, 2, 0, 1]
                out3 = paddle.argmax(x, axis: int = -1)
                print(out3) 
                # [2, 3, 1]
                out4 = paddle.argmax(x, axis: int = 0, keepdim=True)
                print(out4)
                # [[2, 2, 0, 1]]        """
        pass


    def argmin(self, axis: Optional[int] = None, keepdim: bool = False, dtype='int64', name: Optional[str] = None) -> Tensor:
        """
        This OP computes the indices of the min elements of the input tensor's
        element along the provided axis.
        
        Args:
            self (Tensor): An input N-D Tensor with type float32, float64, int16,
                int32, int64, uint8.
            axis(int, optional): Axis to compute indices along. The effective range
                is [-R, R), where R is x.ndim. when axis < 0, it works the same way
                as axis + R. Default is None, the input `x` will be into the flatten tensor, and selecting the min value index.
            keepdim(bool, optional): Whether to keep the given axis in output. If it is True, the dimensions will be same as input x and with size one in the axis. Otherwise the output dimentions is one fewer than x since the axis is squeezed. Default is False.
            dtype(str): Data type of the output tensor which can
                        be int32, int64. The default value is 'int64', and it will
                        return the int64 indices.
            name(str, optional): The default value is None. Normally there is no
                need for user to set this property. For more information, please
                refer to :ref:`api_guide_Name`.
        
        Returns:
            Tensor, return the tensor of `int32` if set :attr:`dtype` is `int32`, otherwise return the tensor of `int64`
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                x =  paddle.to_tensor([[5,8,9,5],
                                         [0,0,1,7],
                                         [6,9,2,4]])
                out1 = paddle.argmin(x)
                print(out1) # 4
                out2 = paddle.argmin(x, axis: int = 0)
                print(out2) 
                # [1, 1, 1, 2]
                out3 = paddle.argmin(x, axis: int = -1)
                print(out3) 
                # [0, 0, 2]
                out4 = paddle.argmin(x, axis: int = 0, keepdim=True)
                print(out4)
                # [[1, 1, 1, 2]]        """
        pass


    def argsort(self, axis: int = -1, descending=False, name: Optional[str] = None) -> Tuple[Tensor, Tensor]:
        """
        This OP sorts the input along the given axis, and returns the corresponding index tensor for the sorted output values. The default sort algorithm is ascending, if you want the sort algorithm to be descending, you must set the :attr:`descending` as True.
        
        Args:
            self (Tensor): An input N-D Tensor with type float32, float64, int16,
                int32, int64, uint8.
            axis(int, optional): Axis to compute indices along. The effective range
                is [-R, R), where R is Rank(x). when axis<0, it works the same way
                as axis+R. Default is 0.
            descending(bool, optional) : Descending is a flag, if set to true,
                algorithm will sort by descending order, else sort by
                ascending order. Default is false.
            name(str, optional): The default value is None. Normally there is no
                need for user to set this property. For more information, please
                refer to :ref:`api_guide_Name`.
        
        Returns:
            Tensor: sorted indices(with the same shape as ``x``
            and with data type int64).
        
        Examples:
        
            .. code-block:: python
        
                import paddle
                
                x = paddle.to_tensor([[[5,8,9,5],
                                       [0,0,1,7],
                                       [6,9,2,4]],
                                      [[5,2,4,2],
                                       [4,7,7,9],
                                       [1,7,0,6]]], 
                                    dtype='float32')
                out1 = paddle.argsort(x=x, axis: int = -1)
                out2 = paddle.argsort(x=x, axis: int = 0)
                out3 = paddle.argsort(x=x, axis: int = 1)
                print(out1)
                #[[[0 3 1 2]
                #  [0 1 2 3]
                #  [2 3 0 1]]
                # [[1 3 2 0]
                #  [0 1 2 3]
                #  [2 0 3 1]]]
                print(out2)
                #[[[0 1 1 1]
                #  [0 0 0 0]
                #  [1 1 1 0]]
                # [[1 0 0 0]
                #  [1 1 1 1]
                #  [0 0 0 1]]]
                print(out3)
                #[[[1 1 1 2]
                #  [0 0 2 0]
                #  [2 2 0 1]]
                # [[2 0 2 0]
                #  [1 1 0 2]
                #  [0 2 1 1]]]        """
        pass


    def as_complex(self, name: Optional[str] = None) -> Tensor:
        """
        Transform a real tensor to a complex tensor. 
        
        The data type of the input tensor is 'float32' or 'float64', and the data
        type of the returned tensor is 'complex64' or 'complex128', respectively.
        
        The shape of the input tensor is ``(* ,2)``, (``*`` means arbitary shape), i.e. 
        the size of the last axis shoule be 2, which represent the real and imag part
        of a complex number. The shape of the returned tensor is ``(*,)``.
        
        Args:
            self (Tensor): The input tensor. Data type is 'float32' or 'float64'.
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            Tensor: The output. Data type is 'complex64' or 'complex128', with the same precision as the input.
        
        Examples:
            .. code-block:: python
        
                import paddle
                x = paddle.arange(12, dtype=paddle.float32).reshape([2, 3, 2])
                y = paddle.as_complex(x)
                print(y.numpy())
        
                # [[ 0. +1.j  2. +3.j  4. +5.j]
                #  [ 6. +7.j  8. +9.j 10.+11.j]]        """
        pass


    def as_real(self, name: Optional[str] = None) -> Tensor:
        """
        Transform a complex tensor to a real tensor. 
        
        The data type of the input tensor is 'complex64' or 'complex128', and the data 
        type of the returned tensor is 'float32' or 'float64', respectively.
        
        When the shape of the input tensor is ``(*, )``, (``*`` means arbitary shape),
        the shape of the output tensor is ``(*, 2)``, i.e. the shape of the output is
        the shape of the input appended by an extra ``2``.
        
        Args:
            self (Tensor): The input tensor. Data type is 'complex64' or 'complex128'.
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            Tensor: The output. Data type is 'float32' or 'float64', with the same precision as the input.
        
        Examples:
            .. code-block:: python
        
                import paddle
                x = paddle.arange(12, dtype=paddle.float32).reshape([2, 3, 2])
                y = paddle.as_complex(x)
                z = paddle.as_real(y)
                print(z.numpy())
        
                # [[[ 0.  1.]
                #   [ 2.  3.]
                #   [ 4.  5.]]
        
                #  [[ 6.  7.]
                #   [ 8.  9.]
                #   [10. 11.]]]        """
        pass


    def asin(self, name: Optional[str] = None) -> Tensor:
        """
        Arcsine Operator.
        
        :math:`out = \sin^{-1}(x)`
        
        
        Args:
            self (Tensor): Input of asin operator, an N-D Tensor, with data type float32, float64 or float16.
            with_quant_attr (BOOLEAN): Whether the operator has attributes used by quantization. 
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            out (Tensor): Output of asin operator
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
                out = paddle.asin(x)
                print(out)
                # [-0.41151685 -0.20135792  0.10016742  0.30469265]        """
        pass


    def asinh(self, name: Optional[str] = None) -> Tensor:
        """
        Asinh Activation Operator.
        
        :math:`out = asinh(x)`
        
        
        Args:
            self (Tensor): Input of Asinh operator, an N-D Tensor, with data type float32, float64 or float16.
            with_quant_attr (BOOLEAN): Whether the operator has attributes used by quantization. 
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            out (Tensor): Output of Asinh operator, a Tensor with shape same as input.
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
                out = paddle.asinh(x)
                print(out)
                # [-0.39003533, -0.19869010,  0.09983408,  0.29567307]        """
        pass


    def astype(self, dtype) -> Tensor:
        pass
    def atan(self, name: Optional[str] = None) -> Tensor:
        """
        Arctangent Operator.
        
        :math:`out = \tan^{-1}(x)`
        
        
        Args:
            self (Tensor): Input of atan operator, an N-D Tensor, with data type float32, float64 or float16.
            with_quant_attr (BOOLEAN): Whether the operator has attributes used by quantization. 
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            out (Tensor): Output of atan operator
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
                out = paddle.atan(x)
                print(out)
                # [-0.38050638 -0.19739556  0.09966865  0.29145679]        """
        pass


    def atanh(self, name: Optional[str] = None) -> Tensor:
        """
        Atanh Activation Operator.
        
        :math:`out = atanh(x)`
        
        
        Args:
            self (Tensor): Input of Atanh operator, an N-D Tensor, with data type float32, float64 or float16.
            with_quant_attr (BOOLEAN): Whether the operator has attributes used by quantization. 
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            out (Tensor): Output of Atanh operator, a Tensor with shape same as input.
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
                out = paddle.atanh(x)
                print(out)
                # [-0.42364895, -0.20273256,  0.10033535,  0.30951962]        """
        pass


    def backward(self, grad_tensor=None, retain_graph=False) -> Tuple[]:
        pass
    def bincount(self, weights=None, minlength=0, name: Optional[str] = None) -> Tensor:
        """
        Computes frequency of each value in the input tensor. 
        
        Args:
            self (Tensor): A Tensor with non-negative integer. Should be 1-D tensor.
            weights (Tensor, optional): Weight for each value in the input tensor. Should have the same shape as input. Default is None.
            minlength (int, optional): Minimum number of bins. Should be non-negative integer. Default is 0.
            name(str, optional): The default value is None.  Normally there is no need for user to set this
                property.  For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            Tensor: The tensor of frequency.
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([1, 2, 1, 4, 5])
                result1 = paddle.bincount(x)
                print(result1) # [0, 2, 1, 0, 1, 1]
        
                w = paddle.to_tensor([2.1, 0.4, 0.1, 0.5, 0.5])
                result2 = paddle.bincount(x, weights=w)
                print(result2) # [0., 2.19999981, 0.40000001, 0., 0.50000000, 0.50000000]        """
        pass


    def bitwise_and(self, y: Tensor, out=None, name: Optional[str] = None) -> Tensor:
        """
        It operates ``bitwise_and`` on Tensor ``X`` and ``Y`` . 
        
        .. math:: Out = X \& Y 
        
        .. note:: ``paddle.bitwise_and`` supports broadcasting. If you want know more about broadcasting, please refer to :ref:`user_guide_broadcasting`. 
        
        
        
        Args:
            self (Tensor): Input Tensor of ``bitwise_and`` . It is a N-D Tensor of bool, uint8, int8, int16, int32, int64
            y (Tensor): Input Tensor of ``bitwise_and`` . It is a N-D Tensor of bool, uint8, int8, int16, int32, int64
            out(Tensor): Result of ``bitwise_and`` . It is a N-D Tensor with the same data type of input Tensor
        
        Returns:
            Tensor: Result of ``bitwise_and`` . It is a N-D Tensor with the same data type of input Tensor
            
        Examples:
            .. code-block:: python
        
                import paddle
                x = paddle.to_tensor([-5, -1, 1])
                y = paddle.to_tensor([4,  2, -3])
                res = paddle.bitwise_and(x, y)
                print(res)  # [0, 2, 1]        """
        pass


    def bitwise_not(self, out=None, name: Optional[str] = None) -> Tensor:
        """
        It operates ``bitwise_not`` on Tensor ``X`` . 
        
        .. math:: Out = \sim X 
        
        
        
        
        
        Args:
            self (Tensor):  Input Tensor of ``bitwise_not`` . It is a N-D Tensor of bool, uint8, int8, int16, int32, int64
            out(Tensor): Result of ``bitwise_not`` . It is a N-D Tensor with the same data type of input Tensor
        
        Returns:
            Tensor: Result of ``bitwise_not`` . It is a N-D Tensor with the same data type of input Tensor
        
        Examples:
            .. code-block:: python
        
                import paddle
                x = paddle.to_tensor([-5, -1, 1])
                res = paddle.bitwise_not(x)
                print(res) # [4, 0, -2]        """
        pass


    def bitwise_or(self, y: Tensor, out=None, name: Optional[str] = None) -> Tensor:
        """
        It operates ``bitwise_or`` on Tensor ``X`` and ``Y`` . 
        
        .. math:: Out = X | Y 
        
        .. note:: ``paddle.bitwise_or`` supports broadcasting. If you want know more about broadcasting, please refer to :ref:`user_guide_broadcasting`. 
        
        
        
        Args:
            self (Tensor): Input Tensor of ``bitwise_or`` . It is a N-D Tensor of bool, uint8, int8, int16, int32, int64
            y (Tensor): Input Tensor of ``bitwise_or`` . It is a N-D Tensor of bool, uint8, int8, int16, int32, int64
            out(Tensor): Result of ``bitwise_or`` . It is a N-D Tensor with the same data type of input Tensor
        
        Returns:
            Tensor: Result of ``bitwise_or`` . It is a N-D Tensor with the same data type of input Tensor
        
        Examples:
            .. code-block:: python
        
                import paddle
                x = paddle.to_tensor([-5, -1, 1])
                y = paddle.to_tensor([4,  2, -3])
                res = paddle.bitwise_or(x, y)
                print(res)  # [-1, -1, -3]        """
        pass


    def bitwise_xor(self, y: Tensor, out=None, name: Optional[str] = None) -> Tensor:
        """
        It operates ``bitwise_xor`` on Tensor ``X`` and ``Y`` . 
        
        .. math:: Out = X ^\wedge Y 
        
        .. note:: ``paddle.bitwise_xor`` supports broadcasting. If you want know more about broadcasting, please refer to :ref:`user_guide_broadcasting`. 
        
        
        
        Args:
            self (Tensor): Input Tensor of ``bitwise_xor`` . It is a N-D Tensor of bool, uint8, int8, int16, int32, int64
            y (Tensor): Input Tensor of ``bitwise_xor`` . It is a N-D Tensor of bool, uint8, int8, int16, int32, int64
            out(Tensor): Result of ``bitwise_xor`` . It is a N-D Tensor with the same data type of input Tensor
        
        Returns:
            Tensor: Result of ``bitwise_xor`` . It is a N-D Tensor with the same data type of input Tensor
        
        Examples:
            .. code-block:: python
        
                import paddle
                x = paddle.to_tensor([-5, -1, 1])
                y = paddle.to_tensor([4,  2, -3])
                res = paddle.bitwise_xor(x, y)
                print(res) # [-1, -3, -4]        """
        pass


    def bmm(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        """
        Applies batched matrix multiplication to two tensors.
        
        Both of the two input tensors must be three-dementional and share the same batch size.
        
        if x is a (b, m, k) tensor, y is a (b, k, n) tensor, the output will be a (b, m, n) tensor.
        
        Args:
            self (Tensor): The input Tensor.
            y (Tensor): The input Tensor.
            name(str|None): A name for this layer(optional). If set None, the layer
                will be named automatically.
        
        Returns:
            Tensor: The product Tensor.
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                # In imperative mode:
                # size x: (2, 2, 3) and y: (2, 3, 2)
                x = paddle.to_tensor([[[1.0, 1.0, 1.0],
                                    [2.0, 2.0, 2.0]],
                                    [[3.0, 3.0, 3.0],
                                    [4.0, 4.0, 4.0]]])
                y = paddle.to_tensor([[[1.0, 1.0],[2.0, 2.0],[3.0, 3.0]],
                                    [[4.0, 4.0],[5.0, 5.0],[6.0, 6.0]]])
                out = paddle.bmm(x, y)
                #output size: (2, 2, 2)
                #output value:
                #[[[6.0, 6.0],[12.0, 12.0]],[[45.0, 45.0],[60.0, 60.0]]]
                out_np = out.numpy()        """
        pass


    def broadcast_shape(self, y_shape) -> Tuple[int]:
        """
        The function returns the shape of doing operation with broadcasting on tensors of x_shape and y_shape, please refer to :ref:`user_guide_broadcasting` for more details.
        
        Args:
            x_shape (list[int]|tuple[int]): A shape of tensor.
            y_shape (list[int]|tuple[int]): A shape of tensor.
            
        
        Returns:
            list[int], the result shape.
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                shape = paddle.broadcast_shape([2, 1, 3], [1, 3, 1])
                # [2, 3, 3]
                
                # shape = paddle.broadcast_shape([2, 1, 3], [3, 3, 1])
                # ValueError (terminated with error message).        """
        pass


    def broadcast_tensors(self, name: Optional[str] = None) -> List[Tensor]:
        """
        This OP broadcast a list of tensors following broadcast semantics
        
        .. note::
            If you want know more about broadcasting, please refer to :ref:`user_guide_broadcasting`.
        
        Args:
            input(list|tuple): ``input`` is a Tensor list or Tensor tuple which is with data type bool,
                float16, float32, float64, int32, int64. All the Tensors in ``input`` must have same data type.
                Currently we only support tensors with rank no greater than 5.
        
            name (str, optional): The default value is None. Normally there is no need for user to set this property. 
                For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            list(Tensor): The list of broadcasted tensors following the same order as ``input``.
        
        Examples:
            .. code-block:: python
        
                import paddle
                x1 = paddle.rand([1, 2, 3, 4]).astype('float32')
                x2 = paddle.rand([1, 2, 1, 4]).astype('float32')
                x3 = paddle.rand([1, 1, 3, 1]).astype('float32')
                out1, out2, out3 = paddle.broadcast_tensors(input=[x1, x2, x3])
                # out1, out2, out3: tensors broadcasted from x1, x2, x3 with shape [1,2,3,4]        """
        pass


    def broadcast_to(self, shape, name: Optional[str] = None) -> Tensor:
        """
        Broadcast the input tensor to a given shape.
        
        Both the number of dimensions of ``x`` and the number of elements in ``shape`` should be less than or equal to 6. The dimension to broadcast to must have a value 1.
        
        
        Args:
            self (Tensor): The input tensor, its data type is bool, float32, float64, int32 or int64.
            shape (list|tuple|Tensor): The result shape after broadcasting. The data type is int32. If shape is a list or tuple, all its elements
                should be integers or 1-D Tensors with the data type int32. If shape is a Tensor, it should be an 1-D Tensor with the data type int32. 
                The value -1 in shape means keeping the corresponding dimension unchanged.
            name (str, optional): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name` .
        
        Returns:
            N-D Tensor: A Tensor with the given shape. The data type is the same as ``x``.
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                data = paddle.to_tensor([1, 2, 3], dtype='int32')
                out = paddle.broadcast_to(data, shape=[2, 3])
                print(out)
                # [[1, 2, 3], [1, 2, 3]]        """
        pass


    def cast(self, dtype) -> Tensor:
        """
        This OP takes in the Tensor :attr:`x` with :attr:`x.dtype` and casts it
        to the output with :attr:`dtype`. It's meaningless if the output dtype
        equals the input dtype, but it's fine if you do so.
        
        Args:
            self (Tensor): An input N-D Tensor with data type bool, float16,
                float32, float64, int32, int64, uint8.
            dtype(np.dtype|str): Data type of the output:
                bool, float16, float32, float64, int8, int32, int64, uint8.
        
        Returns:
            Tensor: A Tensor with the same shape as input's.
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([2, 3, 4], 'float64')
                y = paddle.cast(x, 'uint8')        """
        pass


    def ceil(self, name: Optional[str] = None) -> Tensor:
        """
        Ceil Operator. Computes ceil of x element-wise.
        
        :math:`out = \\lceil x \\rceil`
        
        
        Args:
            self (Tensor): Input of Ceil operator, an N-D Tensor, with data type float32, float64 or float16.
            with_quant_attr (BOOLEAN): Whether the operator has attributes used by quantization. 
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            out (Tensor): Output of Ceil operator, a Tensor with shape same as input.
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
                out = paddle.ceil(x)
                print(out)
                # [-0. -0.  1.  1.]        """
        pass


    def ceil_(self, name: Optional[str] = None) -> None:
        """
        Inplace version of ``ceil`` API, the output Tensor will be inplaced with input ``x``.
        Please refer to :ref:`api_fluid_layers_ceil`.        """
        pass


    def cholesky(self, upper=False, name: Optional[str] = None) -> Tensor:
        """
        Computes the Cholesky decomposition of one symmetric positive-definite
        matrix or batches of symmetric positive-definite matrice.
        
        If `upper` is `True`, the decomposition has the form :math:`A = U^{T}U` ,
        and the returned matrix :math:`U` is upper-triangular. Otherwise, the
        decomposition has the form  :math:`A = LL^{T}` , and the returned matrix
        :math:`L` is lower-triangular.
        
        Args:
            self (Tensor): The input tensor. Its shape should be `[*, M, M]`,
                where * is zero or more batch dimensions, and matrices on the
                inner-most 2 dimensions all should be symmetric positive-definite.
                Its data type should be float32 or float64.
            upper (bool): The flag indicating whether to return upper or lower
                triangular matrices. Default: False.
        
        Returns:
            Tensor: A Tensor with same shape and data type as `x`. It represents \
                triangular matrices generated by Cholesky decomposition.
        
        Examples:
            .. code-block:: python
        
                import paddle
                import numpy as np
        
                a = np.random.rand(3, 3)
                a_t = np.transpose(a, [1, 0])
                x_data = np.matmul(a, a_t) + 1e-03
                x = paddle.to_tensor(x_data)
                out = paddle.linalg.cholesky(x, upper=False)
                print(out)
                # [[1.190523   0.         0.        ]
                #  [0.9906703  0.27676893 0.        ]
                #  [1.25450498 0.05600871 0.06400121]]        """
        pass


    def cholesky_solve(self, y: Tensor, upper=False, name: Optional[str] = None) -> Tensor:
        """
        Solves a linear system of equations A @ X = B, given A's Cholesky factor matrix u and  matrix B.
        
        Input `x` and `y` is 2D matrices or batches of 2D matrices. If the inputs are batches, the outputs
        is also batches.
        
        Args:
            self (Tensor): The input matrix which is upper or lower triangular Cholesky factor of square matrix A. Its shape should be `[*, M, M]`, where `*` is zero or
                more batch dimensions. Its data type should be float32 or float64.
            y (Tensor): Multiple right-hand sides of system of equations. Its shape should be `[*, M, K]`, where `*` is 
                zero or more batch dimensions. Its data type should be float32 or float64.
            upper (bool, optional): whether to consider the Cholesky factor as a lower or upper triangular matrix. Default: False.
            name(str, optional): Name for the operation (optional, default is None).
                For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            Tensor: The solution of the system of equations. Its data type is the same as that of `x`.
        
        Examples:
        .. code-block:: python
        
            import paddle
        
            u = paddle.to_tensor([[1, 1, 1], 
                                    [0, 2, 1],
                                    [0, 0,-1]], dtype="float64")
            b = paddle.to_tensor([[0], [-9], [5]], dtype="float64")
            out = paddle.linalg.cholesky_solve(b, u, upper=True)
        
            print(out)
            # [-2.5, -7, 9.5]        """
        pass


    def chunk(self, chunks, axis: int = 0, name: Optional[str] = None) -> Tuple[]:
        """
        Split the input tensor into multiple sub-Tensors.
        
        Args:
            self (Tensor): A N-D Tensor. The data type is bool, float16, float32, float64, int32 or int64.
            chunks(int): The number of tensor to be split along the certain axis.
            axis (int|Tensor, optional): The axis along which to split, it can be a scalar with type 
                ``int`` or a ``Tensor`` with shape [1] and data type  ``int32`` or ``int64``.
                If :math::`axis < 0`, the axis to split along is :math:`rank(x) + axis`. Default is 0.
            name (str, optional): The default value is None.  Normally there is no need for user to set this property.
                For more information, please refer to :ref:`api_guide_Name` .
        Returns:
            list(Tensor): The list of segmented Tensors.
        
        Example:
            .. code-block:: python
                
                import numpy as np
                import paddle
                
                # x is a Tensor which shape is [3, 9, 5]
                x_np = np.random.random([3, 9, 5]).astype("int32")
                x = paddle.to_tensor(x_np)
        
                out0, out1, out2 = paddle.chunk(x, chunks=3, axis: int = 1)
                # out0.shape [3, 3, 5]
                # out1.shape [3, 3, 5]
                # out2.shape [3, 3, 5]
        
                
                # axis is negative, the real axis is (rank(x) + axis) which real
                # value is 1.
                out0, out1, out2 = paddle.chunk(x, chunks=3, axis: int = -2)
                # out0.shape [3, 3, 5]
                # out1.shape [3, 3, 5]
                # out2.shape [3, 3, 5]        """
        pass


    def clear_grad(self) -> Tuple[]:
        pass
    def clip(self, min=None, max=None, name: Optional[str] = None) -> Tensor:
        """
        This operator clip all elements in input into the range [ min, max ] and return
        a resulting tensor as the following equation:
        
        .. math::
        
            Out = MIN(MAX(x, min), max)
        
        Args:
            self (Tensor): An N-D Tensor with data type float32, float64, int32 or int64.
            min (float|int|Tensor): The lower bound with type ``float`` , ``int`` or a ``Tensor``
                with shape [1] and type ``int32``, ``float32``, ``float64``.
            max (float|int|Tensor): The upper bound with type ``float``, ``int`` or a ``Tensor``
                with shape [1] and type ``int32``, ``float32``, ``float64``.
            name (str, optional): The default value is None. Normally there is no
                need for user to set this property. For more information, please
                refer to :ref:`api_guide_Name`.
        
        Returns:
            Tensor: A Tensor with the same data type and data shape as input.
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                x1 = paddle.to_tensor([[1.2, 3.5], [4.5, 6.4]], 'float32')
                out1 = paddle.clip(x1, min=3.5, max=5.0)
                out2 = paddle.clip(x1, min=2.5)
                print(out1)
                # [[3.5, 3.5]
                # [4.5, 5.0]]
                print(out2)
                # [[2.5, 3.5]
                # [[4.5, 6.4]        """
        pass


    def clip_(self, min=None, max=None, name: Optional[str] = None) -> None:
        """
        Inplace version of ``clip`` API, the output Tensor will be inplaced with input ``x``.
        Please refer to :ref:`api_tensor_clip`.        """
        pass


    def concat(self, axis: int = 0, name: Optional[str] = None) -> Tensor:
        """
        This OP concatenates the input along the axis.
        
        Args:
            x(list|tuple): ``x`` is a Tensor list or Tensor tuple which is with data type bool, float16,
                float32, float64, int32, int64, uint8. All the Tensors in ``x`` must have same data type.
            axis(int|Tensor, optional): Specify the axis to operate on the input Tensors.
                It's a scalar with data type int or a Tensor with shape [1] and data type int32 
                or int64. The effective range is [-R, R), where R is Rank(x). When ``axis < 0``,
                it works the same way as ``axis+R``. Default is 0.
            name (str, optional): The default value is None. Normally there is no
                need for user to set this property. For more information, please
                refer to :ref:`api_guide_Name`.
        
        Returns:
            Tensor: A Tensor with the same data type as ``x``.
        
        Examples:
            .. code-block:: python
                
                import paddle
                
                x1 = paddle.to_tensor([[1, 2, 3],
                                       [4, 5, 6]])
                x2 = paddle.to_tensor([[11, 12, 13],
                                       [14, 15, 16]])
                x3 = paddle.to_tensor([[21, 22],
                                       [23, 24]])
                zero = paddle.full(shape=[1], dtype='int32', fill_value=0)
                # When the axis is negative, the real axis is (axis + Rank(x))
                # As follow, axis is -1, Rank(x) is 2, the real axis is 1
                out1 = paddle.concat(x=[x1, x2, x3], axis: int = -1)
                out2 = paddle.concat(x=[x1, x2], axis: int = 0)
                out3 = paddle.concat(x=[x1, x2], axis: int = zero)
                # out1
                # [[ 1  2  3 11 12 13 21 22]
                #  [ 4  5  6 14 15 16 23 24]]
                # out2 out3
                # [[ 1  2  3]
                #  [ 4  5  6]
                #  [11 12 13]
                #  [14 15 16]]        """
        pass


    def cond(self, p=None, name: Optional[str] = None) -> None:
        """
        Computes the condition number of a matrix or batches of matrices with respect to a matrix norm ``p``.
        
        Args:
            self (Tensor): The input tensor could be tensor of shape ``(*, m, n)`` where ``*`` is zero or more batch dimensions
                for ``p`` in ``(2, -2)``, or of shape ``(*, n, n)`` where every matrix is invertible for any supported ``p``.
                And the input data type could be ``float32`` or ``float64``.
            p (float|string, optional): Order of the norm. Supported values are `fro`, `nuc`, `1`, `-1`, `2`, `-2`,
                `inf`, `-inf`. Default value is `None`, meaning that the order of the norm is `2`.
            name (str, optional): The default value is `None`. Normally there is no need for
                user to set this property. For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            Tensor: computing results of condition number, its data type is the same as input Tensor ``x``.
        
        Examples:
            .. code-block:: python
        
                import paddle
                import numpy as np
        
                x = paddle.to_tensor([[1., 0, -1], [0, 1, 0], [1, 0, 1]])
        
                # compute conditional number when p is None
                out = paddle.linalg.cond(x)
                # out.numpy() [1.4142135]
        
                # compute conditional number when order of the norm is 'fro'
                out_fro = paddle.linalg.cond(x, p='fro')
                # out_fro.numpy() [3.1622777]
        
                # compute conditional number when order of the norm is 'nuc'
                out_nuc = paddle.linalg.cond(x, p='nuc')
                # out_nuc.numpy() [9.2426405]
        
                # compute conditional number when order of the norm is 1
                out_1 = paddle.linalg.cond(x, p=1)
                # out_1.numpy() [2.]
        
                # compute conditional number when order of the norm is -1
                out_minus_1 = paddle.linalg.cond(x, p=-1)
                # out_minus_1.numpy() [1.]
        
                # compute conditional number when order of the norm is 2
                out_2 = paddle.linalg.cond(x, p=2)
                # out_2.numpy() [1.4142135]
        
                # compute conditional number when order of the norm is -1
                out_minus_2 = paddle.linalg.cond(x, p=-2)
                # out_minus_2.numpy() [0.70710677]
        
                # compute conditional number when order of the norm is inf
                out_inf = paddle.linalg.cond(x, p=np.inf)
                # out_inf.numpy() [2.]
        
                # compute conditional number when order of the norm is -inf
                out_minus_inf = paddle.linalg.cond(x, p=-np.inf)
                # out_minus_inf.numpy() [1.]
        
                a = paddle.to_tensor(np.random.randn(2, 4, 4).astype('float32'))
                # a.numpy()
                # [[[ 0.14063153 -0.996288    0.7996131  -0.02571543]
                #   [-0.16303636  1.5534962  -0.49919784 -0.04402903]
                #   [-1.1341571  -0.6022629   0.5445269   0.29154757]
                #   [-0.16816919 -0.30972657  1.7521842  -0.5402487 ]]
                #  [[-0.58081484  0.12402827  0.7229862  -0.55046535]
                #   [-0.15178485 -1.1604939   0.75810957  0.30971205]
                #   [-0.9669573   1.0940945  -0.27363303 -0.35416734]
                #   [-1.216529    2.0018666  -0.7773689  -0.17556527]]]
                a_cond_fro = paddle.linalg.cond(a, p='fro')
                # a_cond_fro.numpy()  [31.572273 28.120834]
        
                b = paddle.to_tensor(np.random.randn(2, 3, 4).astype('float64'))
                # b.numpy()
                # [[[ 1.61707487  0.46829144  0.38130416  0.82546736]
                #   [-1.72710298  0.08866375 -0.62518804  0.16128892]
                #   [-0.02822879 -1.67764516  0.11141444  0.3220113 ]]
                #  [[ 0.22524372  0.62474921 -0.85503233 -1.03960523]
                #   [-0.76620689  0.56673047  0.85064753 -0.45158196]
                #   [ 1.47595418  2.23646462  1.5701758   0.10497519]]]
                b_cond_2 = paddle.linalg.cond(b, p=2)
                # b_cond_2.numpy()  [3.30064451 2.51976252]        """
        pass


    def conj(self, name: Optional[str] = None) -> Tensor:
        """
        This function computes the conjugate of the Tensor elementwisely.
        
        Args:
            self (Tensor): The input tensor which hold the complex numbers. 
                Optional data types are: complex64, complex128, float32, float64, int32 or int64.
            name (str, optional): The default value is None. Normally there is no need for
                user to set this property.  For more information, please refer to :ref:`api_guide_Name`
        
        Returns:
            out (Tensor): The conjugate of input. The shape and data type is the same with input.
                If the elements of tensor is real type such as float32, float64, int32 or int64, the out is the same with input.
        
        Examples:
            .. code-block:: python
        
              import paddle
              data=paddle.to_tensor([[1+1j, 2+2j, 3+3j], [4+4j, 5+5j, 6+6j]])
              #Tensor(shape=[2, 3], dtype=complex64, place=CUDAPlace(0), stop_gradient=True,
              #       [[(1+1j), (2+2j), (3+3j)],
              #        [(4+4j), (5+5j), (6+6j)]])
        
              conj_data=paddle.conj(data)
              #Tensor(shape=[2, 3], dtype=complex64, place=CUDAPlace(0), stop_gradient=True,
              #       [[(1-1j), (2-2j), (3-3j)],
              #        [(4-4j), (5-5j), (6-6j)]])        """
        pass


    def cos(self, name: Optional[str] = None) -> Tensor:
        """
        Cosine Operator. Computes cosine of x element-wise.
        
        Input range is `(-inf, inf)` and output range is `[-1,1]`.
        
        :math:`out = cos(x)`
        
        
        Args:
            self (Tensor): Input of Cos operator, an N-D Tensor, with data type float32, float64 or float16.
            with_quant_attr (BOOLEAN): Whether the operator has attributes used by quantization. 
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            out (Tensor): Output of Cos operator, a Tensor with shape same as input.
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
                out = paddle.cos(x)
                print(out)
                # [0.92106099 0.98006658 0.99500417 0.95533649]        """
        pass


    def cosh(self, name: Optional[str] = None) -> Tensor:
        """
        Cosh Activation Operator.
        
        :math:`out = cosh(x)`
        
        
        Args:
            self (Tensor): Input of Cosh operator, an N-D Tensor, with data type float32, float64 or float16.
            with_quant_attr (BOOLEAN): Whether the operator has attributes used by quantization. 
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            out (Tensor): Output of Cosh operator, a Tensor with shape same as input.
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
                out = paddle.cosh(x)
                print(out)
                # [1.08107237 1.02006676 1.00500417 1.04533851]        """
        pass


    def cov(self, rowvar=True, ddof=True, fweights=None, aweights=None, name: Optional[str] = None) -> None:
        """
        Estimate the covariance matrix of the input variables, given data and weights.
        
        A covariance matrix is a square matrix, indicate the covariance of each pair variables in the input matrix.
        For example, for an N-dimensional samples X=[x1,x2,…xN]T, then the covariance matrix 
        element Cij is the covariance of xi and xj. The element Cii is the variance of xi itself.
        
        Parameters:
            self (Tensor): A N-D(N<=2) Tensor containing multiple variables and observations. By default, each row of x represents a variable. Also see rowvar below.
            rowvar(Bool, optional): If rowvar is True (default), then each row represents a variable, with observations in the columns. Default: True
            ddof(Bool, optional): If ddof=True will return the unbiased estimate, and ddof=False will return the simple average. Default: True
            fweights(Tensor, optional): 1-D Tensor of integer frequency weights; The number of times each observation vector should be repeated. Default: None
            aweights(Tensor, optional): 1-D Tensor of observation vector weights. How important of the observation vector, larger data means this element is more important. Default: None
            name(str, optional): Name of the output. Default is None. It's used to print debug info for developers. Details: :ref:`api_guide_Name`
        
        Returns:
            Tensor: The covariance matrix Tensor of the variables.
        
        Examples:
        
        .. code-block:: python
        
            import paddle
        
            xt = paddle.rand((3,4))
            paddle.linalg.cov(xt)
        
            '''
            Tensor(shape=[3, 3], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
                [[0.07918842, 0.06127326, 0.01493049],
                    [0.06127326, 0.06166256, 0.00302668],
                    [0.01493049, 0.00302668, 0.01632146]])
            '''        """
        pass


    def cross(self, y: Tensor, axis: int = 9, name: Optional[str] = None) -> Tensor:
        """
        Computes the cross product between two tensors along an axis.
        
        Inputs must have the same shape, and the length of their axes should be equal to 3.
        If `axis` is not given, it defaults to the first axis found with the length 3.
        
        Args:
            self (Tensor): The first input tensor.
            y (Tensor): The second input tensor.
            axis (int, optional): The axis along which to compute the cross product. It defaults to be 9 which indicates using the first axis found with the length 3.
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            Tensor. A Tensor with same data type as `x`.
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([[1.0, 1.0, 1.0],
                                      [2.0, 2.0, 2.0],
                                      [3.0, 3.0, 3.0]])
                y = paddle.to_tensor([[1.0, 1.0, 1.0],
                                      [1.0, 1.0, 1.0],
                                      [1.0, 1.0, 1.0]])
        
                z1 = paddle.cross(x, y)
                # [[-1. -1. -1.]
                #  [ 2.  2.  2.]
                #  [-1. -1. -1.]]
        
                z2 = paddle.cross(x, y, axis: int = 1)
                # [[0. 0. 0.]
                #  [0. 0. 0.]
                #  [0. 0. 0.]]        """
        pass


    def cumprod(self, dim=None, dtype=None, name: Optional[str] = None) -> Tensor:
        """
        Compute the cumulative product of the input tensor x along a given dimension dim.
        
        **Note**:
        The first element of the result is the same as the first element of the input.
        
        Args:
            self (Tensor): the input tensor need to be cumproded.
            dim (int): the dimension along which the input tensor will be accumulated. It need to be in the range of [-x.rank, x.rank), where x.rank means the dimensions of the input tensor x and -1 means the last dimension.
            dtype (str, optional): The data type of the output tensor, can be float32, float64, int32, int64, complex64, complex128. If specified, the input tensor is casted to dtype before the operation is performed. This is useful for preventing data type overflows. The default value is None.
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            Tensor, the result of cumprod operator.
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                data = paddle.arange(12)
                data = paddle.reshape(data, (3, 4))
                # [[ 0  1  2  3 ]
                #  [ 4  5  6  7 ]
                #  [ 8  9  10 11]]
        
                y = paddle.cumprod(data, dim=0)
                # [[ 0  1   2   3]
                #  [ 0  5  12  21]
                #  [ 0 45 120 231]]
        
                y = paddle.cumprod(data, dim=-1)
                # [[ 0   0   0    0]
                #  [ 4  20 120  840]
                #  [ 8  72 720 7920]]
        
                y = paddle.cumprod(data, dim=1, dtype='float64')
                # [[ 0.   0.   0.    0.]
                #  [ 4.  20. 120.  840.]
                #  [ 8.  72. 720. 7920.]]
        
                print(y.dtype)
                # paddle.float64        """
        pass


    def cumsum(self, axis: Optional[int] = None, dtype=None, name: Optional[str] = None) -> Tensor:
        """
        The cumulative sum of the elements along a given axis. 
        
        **Note**:
        The first element of the result is the same of the first element of the input. 
        
        Args:
            self (Tensor): The input tensor needed to be cumsumed.
            axis (int, optional): The dimension to accumulate along. -1 means the last dimension. The default (None) is to compute the cumsum over the flattened array.
            dtype (str, optional): The data type of the output tensor, can be float32, float64, int32, int64. If specified, the input tensor is casted to dtype before the operation is performed. This is useful for preventing data type overflows. The default value is None. 
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            Tensor, the result of cumsum operator. 
        
        Examples:
            .. code-block:: python
                
                import paddle
                
                data = paddle.arange(12)
                data = paddle.reshape(data, (3, 4))
        
                y = paddle.cumsum(data)
                # [ 0  1  3  6 10 15 21 28 36 45 55 66]
        
                y = paddle.cumsum(data, axis: int = 0)
                # [[ 0  1  2  3]
                #  [ 4  6  8 10]
                #  [12 15 18 21]]
                
                y = paddle.cumsum(data, axis: int = -1)
                # [[ 0  1  3  6]
                #  [ 4  9 15 22]
                #  [ 8 17 27 38]]
        
                y = paddle.cumsum(data, dtype='float64')
                print(y.dtype)
                # paddle.float64        """
        pass


    def deg2rad(self, name: Optional[str] = None) -> Tuple[]:
        """
        Convert each of the elements of input x from degrees to angles in radians.
        
        Equation:
            .. math::
        
                deg2rad(x)=\pi * x / 180
        
        Args:
            self (Tensor): An N-D Tensor, the data type is float32, float64, int32, int64.
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            out (Tensor): An N-D Tensor, the shape and data type is the same with input (The output data type is float32 when the input data type is int).
        
        Examples:
            .. code-block:: python
        
                import paddle
                import numpy as np
                
                x1 = paddle.to_tensor([180.0, -180.0, 360.0, -360.0, 90.0, -90.0])
                result1 = paddle.deg2rad(x1)
                print(result1)
                # Tensor(shape=[6], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
                #         [3.14159274, -3.14159274,  6.28318548, -6.28318548,  1.57079637,
                #           -1.57079637])
        
                x2 = paddle.to_tensor(180)
                result2 = paddle.deg2rad(x2)
                print(result2)
                # Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
                #         [3.14159274])        """
        pass


    def diagonal(self, offset=0, axis1=0, axis2=1, name: Optional[str] = None) -> Tensor:
        """
        This OP computes the diagonals of the input tensor x.
        
        If ``x`` is 2D, returns the diagonal.
        If ``x`` has larger dimensions, diagonals be taken from the 2D planes specified by axis1 and axis2. 
        By default, the 2D planes formed by the first and second axis of the input tensor x.
        
        The argument ``offset`` determines where diagonals are taken from input tensor x:
        
        - If offset = 0, it is the main diagonal.
        - If offset > 0, it is above the main diagonal.
        - If offset < 0, it is below the main diagonal.
        
        Args:
            self (Tensor): The input tensor x. Must be at least 2-dimensional. The input data type should be bool, int32, int64, float16, float32, float64.
            offset(int, optional): Which diagonals in input tensor x will be taken. Default: 0 (main diagonals).
            axis1(int, optional): The first axis with respect to take diagonal. Default: 0.
            axis2(int, optional): The second axis with respect to take diagonal. Default: 1.
            name (str, optional): Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`. Default: None.
        
        Returns:
            Tensor: a partial view of input tensor in specify two dimensions, the output data type is the same as input data type.
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.rand([2,2,3],'float32')
                print(x)
                # Tensor(shape=[2, 2, 3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
                #        [[[0.45661032, 0.03751532, 0.90191704],
                #          [0.43760979, 0.86177313, 0.65221709]],
        
                #         [[0.17020577, 0.00259554, 0.28954273],
                #          [0.51795638, 0.27325270, 0.18117726]]])
        
                out1 = paddle.diagonal(x)
                print(out1)
                #Tensor(shape=[3, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
                #       [[0.45661032, 0.51795638],
                #        [0.03751532, 0.27325270],
                #        [0.90191704, 0.18117726]])
        
                out2 = paddle.diagonal(x, offset=0, axis1=2, axis2=1)
                print(out2)
                #Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
                #       [[0.45661032, 0.86177313],
                #        [0.17020577, 0.27325270]])
        
                out3 = paddle.diagonal(x, offset=1, axis1=0, axis2=1)
                print(out3)
                #Tensor(shape=[3, 1], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
                #       [[0.43760979],
                #        [0.86177313],
                #        [0.65221709]])
        
                out4 = paddle.diagonal(x, offset=0, axis1=1, axis2=2)
                print(out4)
                #Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
                #       [[0.45661032, 0.86177313],
                #        [0.17020577, 0.27325270]])
                        """
        pass


    def diff(self, n=1, axis: int = -1, prepend=None, append=None, name: Optional[str] = None) -> Tensor:
        """
        Computes the n-th forward difference along the given axis.
        The first-order differences is computed by using the following formula: 
        
        .. math::
        
            out[i] = x[i+1] - x[i]
        
        Higher-order differences are computed by using paddle.diff() recursively. 
        Only n=1 is currently supported.
        
        Args:
            self (Tensor): The input tensor to compute the forward difference on
            n(int, optional): The number of times to recursively compute the difference. 
                              Only support n=1. Default:1
            axis(int, optional): The axis to compute the difference along. Default:-1
            prepend(Tensor, optional): The tensor to prepend to input along axis before computing the difference.
                                       It's dimensions must be equivalent to that of x, 
                                       and its shapes must match x's shape except on axis.
            append(Tensor, optional): The tensor to append to input along axis before computing the difference, 
                                       It's dimensions must be equivalent to that of x, 
                                       and its shapes must match x's shape except on axis.
            name(str|None): A name for this layer(optional). If set None, 
                            the layer will be named automatically.
        
        Returns:
            Tensor: The output tensor with same dtype with x.
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([1, 4, 5, 2])
                out = paddle.diff(x)
                print(out)
                # out:
                # [3, 1, -3]
        
                y = paddle.to_tensor([7, 9])
                out = paddle.diff(x, append=y)
                print(out)
                # out: 
                # [3, 1, -3, 5, 2]
        
                z = paddle.to_tensor([[1, 2, 3], [4, 5, 6]])
                out = paddle.diff(z, axis: int = 0)
                print(out)
                # out:
                # [[3, 3, 3]]
                out = paddle.diff(z, axis: int = 1)
                print(out)
                # out:
                # [[1, 1], [1, 1]]        """
        pass


    def digamma(self, name: Optional[str] = None) -> Tensor:
        """
        Calculates the digamma of the given input tensor, element-wise.
        
        .. math::
            Out = \Psi(x) = \frac{ \Gamma^{'}(x) }{ \Gamma(x) }
        
        Args:
            self (Tensor): Input Tensor. Must be one of the following types: float32, float64.
            name(str, optional): The default value is None.  Normally there is no need for 
                user to set this property.  For more information, please refer to :ref:`api_guide_Name`
        Returns:
            Tensor, the digamma of the input Tensor, the shape and data type is the same with input.
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                data = paddle.to_tensor([[1, 1.5], [0, -2.2]], dtype='float32')
                res = paddle.digamma(data)
                print(res)
                # Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
                #       [[-0.57721591,  0.03648996],
                #        [ nan       ,  5.32286835]])        """
        pass


    def dim(self) -> Tuple[]:
        pass
    def dist(self, y: Tensor, p=2, name: Optional[str] = None) -> Tensor:
        """
        This OP returns the p-norm of (x - y). It is not a norm in a strict sense, only as a measure
        of distance. The shapes of x and y must be broadcastable. The definition is as follows, for
        details, please refer to the `numpy's broadcasting <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_:
        
        - Each input has at least one dimension.
        - Match the two input dimensions from back to front, the dimension sizes must either be equal, one of them is 1, or one of them does not exist.
        
        Where, z = x - y, the shapes of x and y are broadcastable, then the shape of z can be
        obtained as follows:
        
        1. If the number of dimensions of x and y are not equal, prepend 1 to the dimensions of the
        tensor with fewer dimensions.
        
        For example, The shape of x is [8, 1, 6, 1], the shape of y is [7, 1, 5], prepend 1 to the
        dimension of y.
        
        x (4-D Tensor):  8 x 1 x 6 x 1
        
        y (4-D Tensor):  1 x 7 x 1 x 5
        
        2. Determine the size of each dimension of the output z: choose the maximum value from the
        two input dimensions.
        
        z (4-D Tensor):  8 x 7 x 6 x 5
        
        If the number of dimensions of the two inputs are the same, the size of the output can be
        directly determined in step 2. When p takes different values, the norm formula is as follows:
        
        When p = 0, defining $0^0=0$, the zero-norm of z is simply the number of non-zero elements of z.
        
        .. math::
        
            ||z||_{0}=\lim_{p \\rightarrow 0}\sum_{i=1}^{m}|z_i|^{p}
        
        When p = inf, the inf-norm of z is the maximum element of z.
        
        .. math::
        
            ||z||_\infty=\max_i |z_i|
        
        When p = -inf, the negative-inf-norm of z is the minimum element of z.
        
        .. math::
        
            ||z||_{-\infty}=\min_i |z_i|
        
        Otherwise, the p-norm of z follows the formula,
        
        .. math::
        
            ||z||_{p}=(\sum_{i=1}^{m}|z_i|^p)^{\\frac{1}{p}}
        
        Args:
            self (Tensor): 1-D to 6-D Tensor, its data type is float32 or float64.
            y (Tensor): 1-D to 6-D Tensor, its data type is float32 or float64.
            p (float, optional): The norm to be computed, its data type is float32 or float64. Default: 2.
        
        Returns:
            Tensor: Tensor that is the p-norm of (x - y).
        
        Examples:
            .. code-block:: python
        
                import paddle
                import numpy as np
        
                x = paddle.to_tensor(np.array([[3, 3],[3, 3]]), "float32")
                y = paddle.to_tensor(np.array([[3, 3],[3, 1]]), "float32")
                out = paddle.dist(x, y, 0)
                print(out) # out = [1.]
        
                out = paddle.dist(x, y, 2)
                print(out) # out = [2.]
        
                out = paddle.dist(x, y, float("inf"))
                print(out) # out = [2.]
        
                out = paddle.dist(x, y, float("-inf"))
                print(out) # out = [0.]        """
        pass


    def divide(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        """
        Divide two tensors element-wise. The equation is:
        
        .. math::
            out = x / y
        
        **Note**:
        ``paddle.divide`` supports broadcasting. If you want know more about broadcasting, please refer to :ref:`user_guide_broadcasting` .
        
        Args:
            self (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
            y (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            N-D Tensor. A location into which the result is stored. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape,  its shape is the same as x and y.
        
        Examples:
        
            ..  code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([2, 3, 4], dtype='float64')
                y = paddle.to_tensor([1, 5, 2], dtype='float64')
                z = paddle.divide(x, y)
                print(z)  # [2., 0.6, 2.]        """
        pass


    def dot(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        """
        This operator calculates inner product for vectors.
        
        .. note::
           Support 1-d and 2-d Tensor. When it is 2d, the first dimension of this matrix
           is the batch dimension, which means that the vectors of multiple batches are dotted.
        
        Parameters:
            self (Tensor): 1-D or 2-D ``Tensor``. Its dtype should be ``float32``, ``float64``, ``int32``, ``int64``
            y(Tensor): 1-D or 2-D ``Tensor``. Its dtype soulde be ``float32``, ``float64``, ``int32``, ``int64``
            name(str, optional): Name of the output. Default is None. It's used to print debug info for developers. Details: :ref:`api_guide_Name`
        
        Returns:
            Tensor: the calculated result Tensor.
        
        Examples:
        
        .. code-block:: python
        
            import paddle
            import numpy as np
        
            x_data = np.random.uniform(0.1, 1, [10]).astype(np.float32)
            y_data = np.random.uniform(1, 3, [10]).astype(np.float32)
            x = paddle.to_tensor(x_data)
            y = paddle.to_tensor(y_data)
            z = paddle.dot(x, y)
            print(z)        """
        pass


    def eig(self, name: Optional[str] = None) -> Tuple[Tensor, Tensor]:
        """
        This API performs the eigenvalue decomposition of a square matrix or a batch of square matrices.
        
        .. note::
            If the matrix is a Hermitian or a real symmetric matrix, please use :ref:`paddle.linalg.eigh` instead, which is much faster.
            If only eigenvalues is needed, please use :ref:`paddle.linalg.eigvals` instead.
            If the matrix is of any shape, please use :ref:`paddle.linalg.svd`.
            This API is only supported on CPU device.
            The output datatype is always complex for both real and complex input.
        
        Args:
            self (Tensor): A tensor with shape math:`[*, N, N]`, The data type of the x should be one of ``float32``,
                ``float64``, ``compplex64`` or ``complex128``.
            name (str, optional): The default value is `None`. Normally there is no need for user to set 
                this property. For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            Eigenvalues(Tensors): A tensor with shape math:`[*, N]` refers to the eigen values.
            Eigenvectors(Tensors): A tensor with shape math:`[*, N, N]` refers to the eigen vectors.
        
        Examples:
            .. code-block:: python
        
                import paddle
                import numpy as np
        
                paddle.device.set_device("cpu")
        
                x_data = np.array([[1.6707249, 7.2249975, 6.5045543],
                                   [9.956216,  8.749598,  6.066444 ],
                                   [4.4251957, 1.7983172, 0.370647 ]]).astype("float32")
                x = paddle.to_tensor(x_data)
                w, v = paddle.linalg.eig(x)
                print(w)
                # Tensor(shape=[3, 3], dtype=complex128, place=CPUPlace, stop_gradient=False,
                #       [[(-0.5061363550800655+0j) , (-0.7971760990842826+0j) ,
                #         (0.18518077798279986+0j)],
                #        [(-0.8308237755993192+0j) ,  (0.3463813401919749+0j) ,
                #         (-0.6837005269141947+0j) ],
                #        [(-0.23142567697893396+0j),  (0.4944999840400175+0j) ,
                #         (0.7058765252952796+0j) ]])
        
                print(v)
                # Tensor(shape=[3], dtype=complex128, place=CPUPlace, stop_gradient=False,
                #       [ (16.50471283351188+0j)  , (-5.5034820550763515+0j) ,
                #         (-0.21026087843552282+0j)])        """
        pass


    def eigvals(self, name: Optional[str] = None) -> Tensor:
        """
        Compute the eigenvalues of one or more general matrices.
        
        Warning:
            The gradient kernel of this operator does not yet developed.
            If you need back propagation through this operator, please replace it with paddle.linalg.eig.
        
        Args:
            self (Tensor): A square matrix or a batch of square matrices whose eigenvalues will be computed.
                Its shape should be `[*, M, M]`, where `*` is zero or more batch dimensions.
                Its data type should be float32, float64, complex64, or complex128.
            name (str, optional): Name for the operation (optional, default is None).
                For more information, please refer to :ref:`api_guide_Name`.
                
        Returns:
            Tensor: A tensor containing the unsorted eigenvalues which has the same batch dimensions with `x`.
                The eigenvalues are complex-valued even when `x` is real.
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                paddle.set_device("cpu")
                paddle.seed(1234)
        
                x = paddle.rand(shape=[3, 3], dtype='float64')
                # [[0.02773777, 0.93004224, 0.06911496],
                #  [0.24831591, 0.45733623, 0.07717843],
                #  [0.48016702, 0.14235102, 0.42620817]])
        
                print(paddle.linalg.eigvals(x))
                # [(-0.27078833542132674+0j), (0.29962280156230725+0j), (0.8824477020120244+0j)] #complex128        """
        pass


    def eigvalsh(self, UPLO='L', name: Optional[str] = None) -> Tuple[Tensor, Tensor]:
        """
        Computes the eigenvalues of a 
        complex Hermitian (conjugate symmetric) or a real symmetric matrix.
        
        Args:
            self (Tensor): A tensor with shape :math:`[_, M, M]` , The data type of the input Tensor x
                should be one of float32, float64, complex64, complex128.
            UPLO(str, optional): Lower triangular part of a (‘L’, default) or the upper triangular part (‘U’).
            name(str, optional): The default value is None.  Normally there is no need for user to set this
                property.  For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            Tensor: The tensor eigenvalues in ascending order.
        
        Examples:
            .. code-block:: python
        
                import numpy as np
                import paddle
        
                x_data = np.array([[1, -2j], [2j, 5]])
                x = paddle.to_tensor(x_data)
                out_value = paddle.eigvalsh(x, UPLO='L')
                print(out_value)
                #[0.17157288, 5.82842712]        """
        pass


    def equal(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        """
        This layer returns the truth value of :math:`x == y` elementwise.
        
        **NOTICE**: The output of this OP has no gradient.
        
        Args:
            self (Tensor): Tensor, data type is bool, float32, float64, int32, int64.
            y(Tensor): Tensor, data type is bool, float32, float64, int32, int64.
            name(str, optional): The default value is None.  Normally there is no need for
                user to set this property.  For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            Tensor: output Tensor, it's shape is the same as the input's Tensor,
            and the data type is bool. The result of this op is stop_gradient. 
        
        Examples:
            .. code-block:: python
        
              import paddle
        
              x = paddle.to_tensor([1, 2, 3])
              y = paddle.to_tensor([1, 3, 2])
              result1 = paddle.equal(x, y)
              print(result1)  # result1 = [True False False]        """
        pass


    def equal_all(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        """
        This OP returns the truth value of :math:`x == y`. True if two inputs have the same elements, False otherwise.
        
        **NOTICE**: The output of this OP has no gradient.
        
        Args:
            self (Tensor): Tensor, data type is bool, float32, float64, int32, int64.
            y(Tensor): Tensor, data type is bool, float32, float64, int32, int64.
            name(str, optional): The default value is None.  Normally there is no need for
                user to set this property.  For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            Tensor: output Tensor, data type is bool, value is [False] or [True].
        
        Examples:
            .. code-block:: python
        
              import paddle
        
              x = paddle.to_tensor([1, 2, 3])
              y = paddle.to_tensor([1, 2, 3])
              z = paddle.to_tensor([1, 4, 3])
              result1 = paddle.equal_all(x, y)
              print(result1) # result1 = [True ]
              result2 = paddle.equal_all(x, z)
              print(result2) # result2 = [False ]        """
        pass


    def erf(self, name: Optional[str] = None) -> Tensor:
        """
        :strong:`Erf Operator`
        For more details, see [Error function](https://en.wikipedia.org/wiki/Error_function).
        
        Equation:
            ..  math::
                out = \\frac{2}{\\sqrt{\\pi}} \\int_{0}^{x}e^{- \\eta^{2}}d\\eta
        
        Args:
        
            self (Tensor): The input tensor, it's data type should be float32, float64.
        
        Returns:
        
            Tensor: The output of Erf op, dtype: float32 or float64, the same as the input, shape: the same as the input.
        
        Examples:
            
            .. code-block:: python
            
                import paddle
                x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
                out = paddle.erf(x)
                print(out)
                # [-0.42839236 -0.22270259  0.11246292  0.32862676]        """
        pass


    def erfinv(self, name: Optional[str] = None) -> Tensor:
        """
        The inverse error function of x, .
        
        Equation:
            .. math::
        
                erfinv(erf(x)) = x.
        
        Args:
            self (Tensor): An N-D Tensor, the data type is float32, float64.
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            out (Tensor): An N-D Tensor, the shape and data type is the same with input.
        
        Example:
            .. code-block:: python
        
                import paddle
                
                x = paddle.to_tensor([0, 0.5, -1.], dtype="float32")
                out = paddle.erfinv(x)
                # out: [0, 0.4769, -inf]        """
        pass


    def erfinv_(self, name: Optional[str] = None) -> None:
        """
        Inplace version of ``erfinv`` API, the output Tensor will be inplaced with input ``x``.
        Please refer to :ref:`api_tensor_erfinv`.        """
        pass


    def exp(self, name: Optional[str] = None) -> Tensor:
        """
        Exp Operator. Computes exp of x element-wise with a natural number :math:`e` as the base.
        
        :math:`out = e^x`
        
        
        Args:
            self (Tensor): Input of Exp operator, an N-D Tensor, with data type float32, float64 or float16.
            with_quant_attr (BOOLEAN): Whether the operator has attributes used by quantization. 
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            out (Tensor): Output of Exp operator, a Tensor with shape same as input.
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
                out = paddle.exp(x)
                print(out)
                # [0.67032005 0.81873075 1.10517092 1.34985881]        """
        pass


    def exp_(self, name: Optional[str] = None) -> None:
        """
        Inplace version of ``exp`` API, the output Tensor will be inplaced with input ``x``.
        Please refer to :ref:`api_fluid_layers_exp`.        """
        pass


    def expand(self, shape, name: Optional[str] = None) -> Tensor:
        """
        Expand the input tensor to a given shape.
        
        Both the number of dimensions of ``x`` and the number of elements in ``shape`` should be less than or equal to 6. The dimension to expand must have a value 1.
        
        
        Args:
            self (Tensor): The input tensor, its data type is bool, float32, float64, int32 or int64.
            shape (list|tuple|Tensor): The result shape after expanding. The data type is int32. If shape is a list or tuple, all its elements
                should be integers or 1-D Tensors with the data type int32. If shape is a Tensor, it should be an 1-D Tensor with the data type int32. 
                The value -1 in shape means keeping the corresponding dimension unchanged.
            name (str, optional): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name` .
        
        Returns:
            N-D Tensor: A Tensor with the given shape. The data type is the same as ``x``.
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                data = paddle.to_tensor([1, 2, 3], dtype='int32')
                out = paddle.expand(data, shape=[2, 3])
                print(out)
                # [[1, 2, 3], [1, 2, 3]]        """
        pass


    def expand_as(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        """
        Expand the input tensor ``x`` to the same shape as the input tensor ``y``.
        
        Both the number of dimensions of ``x`` and ``y`` must be less than or equal to 6, and the number of dimensions of ``y`` must be greather than or equal to that of ``x``. The dimension to expand must have a value of 1.
        
        Args:
            self (Tensor): The input tensor, its data type is bool, float32, float64, int32 or int64.
            y (Tensor): The input tensor that gives the shape to expand to.
            name (str, optional): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            N-D Tensor: A Tensor with the same shape as ``y``. The data type is the same as ``x``.
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                data_x = paddle.to_tensor([1, 2, 3], 'int32')
                data_y = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], 'int32')
                out = paddle.expand_as(data_x, data_y)
                np_out = out.numpy()
                # [[1, 2, 3], [1, 2, 3]]        """
        pass


    def exponential_(self, lam=1.0, name: Optional[str] = None) -> Tensor:
        """
        This inplace OP fill input Tensor ``x`` with random number from a Exponential Distribution.
        
        ``lam`` is :math:`\lambda` parameter of Exponential Distribution. 
        
        .. math::
        
            f(x) = \lambda e^{-\lambda x}
        
        Args:
            self (Tensor):  Input tensor. The data type should be float32, float64.
            lam(float, optional): :math:`\lambda` parameter of Exponential Distribution. Default, 1.0.
            name(str, optional): The default value is None. Normally there is no
                need for user to set this property. For more information, please
                refer to :ref:`api_guide_Name`.
        Returns: 
            Tensor: Input Tensor ``x``.
        
        Examples:
            .. code-block:: python
        
                import paddle
                paddle.set_device('cpu')
                paddle.seed(100)
        
                x = paddle.empty([2,3])
                x.exponential_()
                # [[0.80643415, 0.23211166, 0.01169797],
                #  [0.72520673, 0.45208144, 0.30234432]]        """
        pass


    def fill_(self, value) -> None:
        """
        **Notes**:
            **This API is ONLY available in Dygraph mode**
        
        This function fill the Tensor with value inplace.
        
        Args:
            self (Tensor): ``x`` is the Tensor we want to filled data inplace
            value(Scale): ``value`` is the value to be filled in x
        
        Returns:
            self (Tensor): Tensor x filled with value inplace
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                tensor = paddle.to_tensor([0, 1, 2, 3, 4])
        
                tensor.fill_(0)
                print(tensor.tolist())   #[0, 0, 0, 0, 0]        """
        pass


    def fill_diagonal_(self, value, offset=0, wrap=False, name: Optional[str] = None) -> None:
        """
        **Notes**:
            **This API is ONLY available in Dygraph mode**
        This function fill the value into the x Tensor's diagonal inplace.
        Args:
            self (Tensor): ``x`` is the original Tensor
            value(Scale): ``value`` is the value to filled in x
            offset(int,optional): the offset to the main diagonal. Default: 0 (main diagonal).
            wrap(bool,optional): the diagonal 'wrapped' after N columns for tall matrices.
            name(str,optional): Name for the operation (optional, default is None)
        Returns:
            Tensor: Tensor with diagonal filled with value.
        Returns type:
            dtype is same as x Tensor
        Examples:
            .. code-block:: python
                import paddle
                x = paddle.ones((4, 3)) * 2
                x.fill_diagonal_(1.0)
                print(x.tolist())   #[[1.0, 2.0, 2.0], [2.0, 1.0, 2.0], [2.0, 2.0, 1.0], [2.0, 2.0, 2.0]]        """
        pass


    def fill_diagonal_tensor(self, y: Tensor, offset=0, dim1=0, dim2=1, name: Optional[str] = None) -> Tensor:
        """
        This function fill the source Tensor y into the x Tensor's diagonal.
        
        Args:
            self (Tensor): ``x`` is the original Tensor
            y(Tensor): ``y`` is the Tensor to filled in x
            dim1(int,optional): first dimension with respect to which to fill diagonal. Default: 0.
            dim2(int,optional): second dimension with respect to which to fill diagonal. Default: 1.
            offset(int,optional): the offset to the main diagonal. Default: 0 (main diagonal).
            name(str,optional): Name for the operation (optional, default is None)
        
        Returns:
            Tensor: Tensor with diagonal filled with y.
        
        Returns type:
            list: dtype is same as x Tensor
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.ones((4, 3)) * 2
                y = paddle.ones((3,))
                nx = x.fill_diagonal_tensor(y)
                print(nx.tolist())   #[[1.0, 2.0, 2.0], [2.0, 1.0, 2.0], [2.0, 2.0, 1.0], [2.0, 2.0, 2.0]]        """
        pass


    def fill_diagonal_tensor_(self, y: Tensor, offset=0, dim1=0, dim2=1, name: Optional[str] = None) -> None:
        """
        **Notes**:
            **This API is ONLY available in Dygraph mode**
        
        This function fill the source Tensor y into the x Tensor's diagonal inplace.
        
        Args:
            self (Tensor): ``x`` is the original Tensor
            y(Tensor): ``y`` is the Tensor to filled in x
            dim1(int,optional): first dimension with respect to which to fill diagonal. Default: 0.
            dim2(int,optional): second dimension with respect to which to fill diagonal. Default: 1.
            offset(int,optional): the offset to the main diagonal. Default: 0 (main diagonal).
            name(str,optional): Name for the operation (optional, default is None)
        
        Returns:
            Tensor: Tensor with diagonal filled with y.
        
        Returns type:
            list: dtype is same as x Tensor
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.ones((4, 3)) * 2
                y = paddle.ones((3,))
                x.fill_diagonal_tensor_(y)
                print(x.tolist())   #[[1.0, 2.0, 2.0], [2.0, 1.0, 2.0], [2.0, 2.0, 1.0], [2.0, 2.0, 2.0]]        """
        pass


    def flatten(self, start_axis: int = 0, stop_axis: int = -1, name: Optional[str] = None) -> Tuple[Tensor, Tensor]:
        """
        **Flatten op**
        
        Flattens a contiguous range of axes in a tensor according to start_axis and stop_axis.
        
        Note that the output Tensor will share data with origin Tensor and doesn't have a 
        Tensor copy in ``dygraph`` mode. If you want to use the Tensor copy version, please 
        use `Tensor.clone` like ``flatten_clone_x = x.flatten().clone()``.
        
        For Example:
        
        .. code-block:: text
        
            Case 1:
        
              Given
                X.shape = (3, 100, 100, 4)
        
              and
                start_axis = 1
                end_axis = 2
        
              We get:
                Out.shape = (3, 1000 * 100, 2)
        
            Case 2:
        
              Given
                X.shape = (3, 100, 100, 4)
        
              and
                start_axis = 0
                stop_axis = -1
        
              We get:
                Out.shape = (3 * 100 * 100 * 4)
        
        Args:
            self (Tensor): A tensor of number of dimentions >= axis. A tensor with data type float32,
                          float64, int8, int32, int64, uint8.
            start_axis (int): the start axis to flatten
            stop_axis (int): the stop axis to flatten
            name(str, Optional): For details, please refer to :ref:`api_guide_Name`.
                            Generally, no setting is required. Default: None.
        
        Returns:
            Tensor: A tensor with the contents of the input tensor, with input \
                      axes flattened by indicated start axis and end axis. \
                      A Tensor with data type same as input x.
        
        Raises:
            ValueError: If x is not a Tensor.
            ValueError: If start_axis or stop_axis is illegal.
        
        Examples:
        
            .. code-block:: python
        
                import paddle
        
                image_shape=(2, 3, 4, 4)
        
                x = paddle.arange(end=image_shape[0] * image_shape[1] * image_shape[2] * image_shape[3])
                img = paddle.reshape(x, image_shape)
        
                out = paddle.flatten(img, start_axis: int = 1, stop_axis: int = 2)
                # out shape is [2, 12, 4]
        
                # out shares data with img in dygraph mode
                img[0, 0, 0, 0] = -1
                print(out[0, 0, 0]) # [-1]        """
        pass


    def flatten_(self, start_axis: int = 0, stop_axis: int = -1, name: Optional[str] = None) -> None:
        """
        Inplace version of ``flatten`` API, the output Tensor will be inplaced with input ``x``.
        Please refer to :ref:`api_tensor_flatten`.        """
        pass


    def flip(self, axis, name: Optional[str] = None) -> Tensor:
        """
        Reverse the order of a n-D tensor along given axis in axis.
        
        Args:
            self (Tensor): A Tensor(or LoDTensor) with shape :math:`[N_1, N_2,..., N_k]` . The data type of the input Tensor x
                should be float32, float64, int32, int64, bool.
            axis (list|tuple|int): The axis(axes) to flip on. Negative indices for indexing from the end are accepted.
            name (str, optional): The default value is None.  Normally there is no need for user to set this property.
                For more information, please refer to :ref:`api_guide_Name` .
        
        Returns:
            Tensor: Tensor or LoDTensor calculated by flip layer. The data type is same with input x.
        
        Examples:
            .. code-block:: python
        
              import paddle
              import numpy as np
        
              image_shape=(3, 2, 2)
              x = np.arange(image_shape[0] * image_shape[1] * image_shape[2]).reshape(image_shape)
              x = x.astype('float32')
              img = paddle.to_tensor(x)
              tmp = paddle.flip(img, [0,1])
              print(tmp) # [[[10,11],[8, 9]], [[6, 7],[4, 5]], [[2, 3],[0, 1]]]
        
              out = paddle.flip(tmp,-1)
              print(out) # [[[11,10],[9, 8]], [[7, 6],[5, 4]], [[3, 2],[1, 0]]]        """
        pass


    def floor(self, name: Optional[str] = None) -> Tensor:
        """
        Floor Activation Operator. Computes floor of x element-wise.
        
        :math:`out = \\lfloor x \\rfloor`
        
        
        Args:
            self (Tensor): Input of Floor operator, an N-D Tensor, with data type float32, float64 or float16.
            with_quant_attr (BOOLEAN): Whether the operator has attributes used by quantization. 
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            out (Tensor): Output of Floor operator, a Tensor with shape same as input.
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
                out = paddle.floor(x)
                print(out)
                # [-1. -1.  0.  0.]        """
        pass


    def floor_(self, name: Optional[str] = None) -> None:
        """
        Inplace version of ``floor`` API, the output Tensor will be inplaced with input ``x``.
        Please refer to :ref:`api_fluid_layers_floor`.        """
        pass


    def floor_divide(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        """
        Floor divide two tensors element-wise. The equation is:
        
        .. math::
            out = x // y
        
        **Note**:
        ``paddle.floor_divide`` supports broadcasting. If you want know more about broadcasting, please refer to :ref:`user_guide_broadcasting` .
        
        Args:
            self (Tensor): the input tensor, it's data type should be int32, int64.
            y (Tensor): the input tensor, it's data type should be int32, int64.
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            N-D Tensor. A location into which the result is stored. It's dimension equals with $x$.
        
        Examples:
        
            ..  code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([2, 3, 8, 7])
                y = paddle.to_tensor([1, 5, 3, 3])
                z = paddle.floor_divide(x, y)
                print(z)  # [2, 0, 2, 2]        """
        pass


    def floor_mod(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        """
        Mod two tensors element-wise. The equation is:
        
        .. math::
        
            out = x \% y
        
        **Note**:
        ``paddle.remainder`` supports broadcasting. If you want know more about broadcasting, please refer to :ref:`user_guide_broadcasting` .
        
        Args:
            self (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
            y (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            N-D Tensor. A location into which the result is stored. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape,  its shape is the same as x and y.
        
        Examples:
        
            ..  code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([2, 3, 8, 7])
                y = paddle.to_tensor([1, 5, 3, 3])
                z = paddle.remainder(x, y)
                print(z)  # [0, 3, 2, 1]        """
        pass


    def fmax(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        """
        Compares the elements at the corresponding positions of the two tensors and returns a new tensor containing the maximum value of the element.
        If one of them is a nan value, the other value is directly returned, if both are nan values, then the first nan value is returned.
        The equation is:
        
        .. math::
            out = fmax(x, y)
        
        **Note**:
        ``paddle.fmax`` supports broadcasting. If you want know more about broadcasting, please refer to :ref:`user_guide_broadcasting` .
        
        Args:
            self (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
            y (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            N-D Tensor. A location into which the result is stored. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape,  its shape is the same as x and y.
        
        Examples:
        
            .. code-block:: python
        
                import numpy as np
                import paddle
        
                x = paddle.to_tensor([[1, 2], [7, 8]])
                y = paddle.to_tensor([[3, 4], [5, 6]])
                res = paddle.fmax(x, y)
                print(res)
                #    [[3, 4],
                #     [7, 8]]
        
                x = paddle.to_tensor([[1, 2, 3], [1, 2, 3]])
                y = paddle.to_tensor([3, 0, 4])
                res = paddle.fmax(x, y)
                print(res)
                #    [[3, 2, 4],
                #     [3, 2, 4]]
        
                x = paddle.to_tensor([2, 3, 5], dtype='float32')
                y = paddle.to_tensor([1, np.nan, np.nan], dtype='float32')
                res = paddle.fmax(x, y)
                print(res)
                #    [ 2., 3., 5.]
        
                x = paddle.to_tensor([5, 3, np.inf], dtype='float32')
                y = paddle.to_tensor([1, -np.inf, 5], dtype='float32')
                res = paddle.fmax(x, y)
                print(res)
                #    [  5.,   3., inf.]        """
        pass


    def fmin(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        """
        Compares the elements at the corresponding positions of the two tensors and returns a new tensor containing the minimum value of the element.
        If one of them is a nan value, the other value is directly returned, if both are nan values, then the first nan value is returned.
        The equation is:
        
        .. math::
            out = fmin(x, y)
        
        **Note**:
        ``paddle.fmin`` supports broadcasting. If you want know more about broadcasting, please refer to :ref:`user_guide_broadcasting` .
        
        Args:
            self (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
            y (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            N-D Tensor. A location into which the result is stored. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape,  its shape is the same as x and y.
        
        Examples:
        
            .. code-block:: python
        
                import numpy as np
                import paddle
        
                x = paddle.to_tensor([[1, 2], [7, 8]])
                y = paddle.to_tensor([[3, 4], [5, 6]])
                res = paddle.fmin(x, y)
                print(res)
                #       [[1, 2],
                #        [5, 6]]
        
                x = paddle.to_tensor([[[1, 2, 3], [1, 2, 3]]])
                y = paddle.to_tensor([3, 0, 4])
                res = paddle.fmin(x, y)
                print(res)
                #       [[[1, 0, 3],
                #         [1, 0, 3]]]
        
                x = paddle.to_tensor([2, 3, 5], dtype='float32')
                y = paddle.to_tensor([1, np.nan, np.nan], dtype='float32')
                res = paddle.fmin(x, y)
                print(res)
                #       [ 1., 3., 5.]
        
                x = paddle.to_tensor([5, 3, np.inf], dtype='float64')
                y = paddle.to_tensor([1, -np.inf, 5], dtype='float64')
                res = paddle.fmin(x, y)
                print(res)
                #       [   1., -inf.,    5.]        """
        pass


    def gather(self, index, axis: Optional[int] = None, name: Optional[str] = None) -> Tensor:
        """
        Output is obtained by gathering entries of ``axis``
        of ``x`` indexed by ``index`` and concatenate them together.
        
        .. code-block:: text
        
        
                    Given:
        
                    x = [[1, 2],
                         [3, 4],
                         [5, 6]]
        
                    index = [1, 2]
                    axis: int = [0]
        
                    Then:
        
                    out = [[3, 4],
                           [5, 6]] 
        
        Args:
            self (Tensor): The source input tensor with rank>=1. Supported data type is
                int32, int64, float32, float64 and uint8 (only for CPU),
                float16 (only for GPU).
            indeself (Tensor): The index input tensor with rank=1. Data type is int32 or int64.
            axis (Tensor|int, optional): The axis of input to be gathered, it's can be int or a Tensor with data type is int32 or int64. The default value is None, if None, the ``axis`` is 0.
            name (str, optional): The default value is None.  Normally there is no need for user to set this property.
                For more information, please refer to :ref:`api_guide_Name` .
        
        Returns:
            output (Tensor): The output is a tensor with the same rank as ``x``.
        
        Examples:
        
            .. code-block:: python
        
                import paddle
        
                input = paddle.to_tensor([[1,2],[3,4],[5,6]])
                index = paddle.to_tensor([0,1])
                output = paddle.gather(input, index, axis: int = 0)
                # expected output: [[1,2],[3,4]]        """
        pass


    def gather_nd(self, index, name: Optional[str] = None) -> Tensor:
        """
        This function is actually a high-dimensional extension of :code:`gather`
        and supports for simultaneous indexing by multiple axes. :attr:`index` is a
        K-dimensional integer tensor, which is regarded as a (K-1)-dimensional
        tensor of :attr:`index` into :attr:`input`, where each element defines
        a slice of params:
        
        .. math::
        
            output[(i_0, ..., i_{K-2})] = input[index[(i_0, ..., i_{K-2})]]
        
        Obviously, :code:`index.shape[-1] <= input.rank` . And, the output tensor has
        shape :code:`index.shape[:-1] + input.shape[index.shape[-1]:]` .
        
        .. code-block:: text
        
                Given:
                    x =  [[[ 0,  1,  2,  3],
                           [ 4,  5,  6,  7],
                           [ 8,  9, 10, 11]],
                          [[12, 13, 14, 15],
                           [16, 17, 18, 19],
                           [20, 21, 22, 23]]]
                    x.shape = (2, 3, 4)
        
                * Case 1:
                    index = [[1]]
        
                    gather_nd(x, index)
                             = [x[1, :, :]]
                             = [[12, 13, 14, 15],
                                [16, 17, 18, 19],
                                [20, 21, 22, 23]]
        
                * Case 2:
                    index = [[0,2]]
        
                    gather_nd(x, index)
                             = [x[0, 2, :]]
                             = [8, 9, 10, 11]
        
                * Case 3:
                    index = [[1, 2, 3]]
        
                    gather_nd(x, index)
                             = [x[1, 2, 3]]
                             = [23]
        
        Args:
            self (Tensor): The input Tensor which it's data type should be bool, float32, float64, int32, int64.
            indeself (Tensor): The index input with rank > 1, index.shape[-1] <= input.rank.
                            Its dtype should be int32, int64.
            name(str, optional): The default value is None.  Normally there is no need for user to set this property.
                            For more information, please refer to :ref:`api_guide_Name` .
        
        Returns:
            output (Tensor): A tensor with the shape index.shape[:-1] + input.shape[index.shape[-1]:]
        
        Examples:
        
            .. code-block:: python
                
                import paddle
                
                x = paddle.to_tensor([[[1, 2], [3, 4], [5, 6]],
                                      [[7, 8], [9, 10], [11, 12]]])
                index = paddle.to_tensor([[0, 1]])
                
                output = paddle.gather_nd(x, index) #[[3, 4]]        """
        pass


    def gcd(self, y: Tensor, name: Optional[str] = None) -> Tuple[]:
        """
        Computes the element-wise greatest common divisor (GCD) of input |x| and |y|.
        Both x and y must have integer types.
        
        Note:
            gcd(0,0)=0, gcd(0, y)=|y|
        
            If x.shape != y.shape, they must be broadcastable to a common shape (which becomes the shape of the output).
        
        Args:
            self (Tensor): An N-D Tensor, the data type is int32，int64. 
            y (Tensor): An N-D Tensor, the data type is int32，int64. 
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            out (Tensor): An N-D Tensor, the data type is the same with input.
        
        Examples:
            .. code-block:: python
        
                import paddle
                
                x1 = paddle.to_tensor(12)
                x2 = paddle.to_tensor(20)
                paddle.gcd(x1, x2)
                # Tensor(shape=[1], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
                #        [4])
        
                x3 = paddle.arange(6)
                paddle.gcd(x3, x2)
                # Tensor(shape=[6], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
                #        [20, 1 , 2 , 1 , 4 , 5])
        
                x4 = paddle.to_tensor(0)
                paddle.gcd(x4, x2)
                # Tensor(shape=[1], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
                #        [20])
        
                paddle.gcd(x4, x4)
                # Tensor(shape=[1], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
                #        [0])
                
                x5 = paddle.to_tensor(-20)
                paddle.gcd(x1, x5)
                # Tensor(shape=[1], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
                #        [4])        """
        pass


    def gradient(self) -> Tuple[]:
        pass
    def greater_equal(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        """
        This OP returns the truth value of :math:`x >= y` elementwise, which is equivalent function to the overloaded operator `>=`.
        
        **NOTICE**: The output of this OP has no gradient.
        
        Args:
            self (Tensor): First input to compare which is N-D tensor. The input data type should be bool, float32, float64, int32, int64.
            y(Tensor): Second input to compare which is N-D tensor. The input data type should be bool, float32, float64, int32, int64.
            name(str, optional): The default value is None.  Normally there is no need for
                user to set this property.  For more information, please refer to :ref:`api_guide_Name`.
        Returns:
            Tensor, the output data type is bool: The tensor storing the output, the output shape is same as input :attr:`x`.
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([1, 2, 3])
                y = paddle.to_tensor([1, 3, 2])
                result1 = paddle.greater_equal(x, y)
                print(result1)  # result1 = [True False True]        """
        pass


    def greater_than(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        """
        This OP returns the truth value of :math:`x > y` elementwise, which is equivalent function to the overloaded operator `>`.
        
        **NOTICE**: The output of this OP has no gradient.
        
        Args:
            self (Tensor): First input to compare which is N-D tensor. The input data type should be bool, float32, float64, int32, int64.
            y(Tensor): Second input to compare which is N-D tensor. The input data type should be bool, float32, float64, int32, int64.
            name(str, optional): The default value is None.  Normally there is no need for
                user to set this property.  For more information, please refer to :ref:`api_guide_Name`.
        Returns:
            Tensor, the output data type is bool: The tensor storing the output, the output shape is same as input :attr:`x` .
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([1, 2, 3])
                y = paddle.to_tensor([1, 3, 2])
                result1 = paddle.greater_than(x, y)
                print(result1)  # result1 = [False False True]        """
        pass


    def histogram(self, bins=100, min=0, max=0, name: Optional[str] = None) -> Tensor:
        """
        Computes the histogram of a tensor. The elements are sorted into equal width bins between min and max.
        If min and max are both zero, the minimum and maximum values of the data are used.
        
        Args:
            input (Tensor): A Tensor(or LoDTensor) with shape :math:`[N_1, N_2,..., N_k]` . The data type of the input Tensor
                should be float32, float64, int32, int64.
            bins (int): number of histogram bins
            min (int): lower end of the range (inclusive)
            max (int): upper end of the range (inclusive)
        
        Returns:
            Tensor: data type is int64, shape is (nbins,).
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                inputs = paddle.to_tensor([1, 2, 1])
                result = paddle.histogram(inputs, bins=4, min=0, max=3)
                print(result) # [0, 2, 1, 0]        """
        pass


    def imag(self, name: Optional[str] = None) -> Tensor:
        pass
    def increment(self, value=1.0, name: Optional[str] = None) -> Tensor:
        """
        The OP is usually used for control flow to increment the data of :attr:`x` by an amount :attr:`value`.
        Notice that the number of elements in :attr:`x` must be equal to 1.
        
        Args:
            self (Tensor): A tensor that must always contain only one element, its data type supports float32, float64, int32 and int64.
            value(float, optional): The amount to increment the data of :attr:`x`. Default: 1.0.
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            Tensor, the elementwise-incremented tensor with the same shape and data type as :attr:`x`.
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                data = paddle.zeros(shape=[1], dtype='float32')
                counter = paddle.increment(data)
                # [1.]        """
        pass


    def index_sample(self, index) -> Tensor:
        """
        **IndexSample Layer**
        
        IndexSample OP returns the element of the specified location of X, 
        and the location is specified by Index. 
        
        .. code-block:: text
        
        
                    Given:
        
                    X = [[1, 2, 3, 4, 5],
                         [6, 7, 8, 9, 10]]
        
                    Index = [[0, 1, 3],
                             [0, 2, 4]]
        
                    Then:
        
                    Out = [[1, 2, 4],
                           [6, 8, 10]]
        
        Args:
            self (Tensor): The source input tensor with 2-D shape. Supported data type is 
                int32, int64, float32, float64.
            indeself (Tensor): The index input tensor with 2-D shape, first dimension should be same with X. 
                Data type is int32 or int64.
        
        Returns:
            output (Tensor): The output is a tensor with the same shape as index.
        
        Examples:
        
            .. code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0],
                                      [5.0, 6.0, 7.0, 8.0],
                                      [9.0, 10.0, 11.0, 12.0]], dtype='float32')
                index = paddle.to_tensor([[0, 1, 2],
                                          [1, 2, 3],
                                          [0, 0, 0]], dtype='int32')
                target = paddle.to_tensor([[100, 200, 300, 400],
                                           [500, 600, 700, 800],
                                           [900, 1000, 1100, 1200]], dtype='int32')
                out_z1 = paddle.index_sample(x, index)
                print(out_z1)
                #[[1. 2. 3.]
                # [6. 7. 8.]
                # [9. 9. 9.]]
        
                # Use the index of the maximum value by topk op
                # get the value of the element of the corresponding index in other tensors
                top_value, top_index = paddle.topk(x, k=2)
                out_z2 = paddle.index_sample(target, top_index)
                print(top_value)
                #[[ 4.  3.]
                # [ 8.  7.]
                # [12. 11.]]
        
                print(top_index)
                #[[3 2]
                # [3 2]
                # [3 2]]
        
                print(out_z2)
                #[[ 400  300]
                # [ 800  700]
                # [1200 1100]]        """
        pass


    def index_select(self, index, axis: int = 0, name: Optional[str] = None) -> Tensor:
        """
        Returns a new tensor which indexes the ``input`` tensor along dimension ``axis`` using 
        the entries in ``index`` which is a Tensor. The returned tensor has the same number 
        of dimensions as the original ``x`` tensor. The dim-th dimension has the same 
        size as the length of ``index``; other dimensions have the same size as in the ``x`` tensor. 
        
        Args:
            self (Tensor): The input Tensor to be operated. The data of ``x`` can be one of float32, float64, int32, int64.
            indeself (Tensor): The 1-D Tensor containing the indices to index. The data type of ``index`` must be int32 or int64.
            axis (int, optional): The dimension in which we index. Default: if None, the ``axis`` is 0.
            name(str, optional): The default value is None. Normally there is no
                need for user to set this property. For more information, please
                refer to :ref:`api_guide_Name`.
        
        Returns:
            Tensor: A Tensor with same data type as ``x``.
        
        Examples:
            .. code-block:: python
                
                import paddle
        
                x = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0],
                                      [5.0, 6.0, 7.0, 8.0],
                                      [9.0, 10.0, 11.0, 12.0]])
                index = paddle.to_tensor([0, 1, 1], dtype='int32')
                out_z1 = paddle.index_select(x=x, index=index)
                #[[1. 2. 3. 4.]
                # [5. 6. 7. 8.]
                # [5. 6. 7. 8.]]
                out_z2 = paddle.index_select(x=x, index=index, axis: int = 1)
                #[[ 1.  2.  2.]
                # [ 5.  6.  6.]
                # [ 9. 10. 10.]]        """
        pass


    def inner(self, y: Tensor, name: Optional[str] = None) -> Tuple[]:
        """
        Inner product of two input Tensor.
        
        Ordinary inner product for 1-D Tensors, in higher dimensions a sum product over the last axes.
        
        Args:
            self (Tensor): An N-D Tensor or a Scalar Tensor. If its not a scalar Tensor, its last dimensions must match y's.
            y (Tensor): An N-D Tensor or a Scalar Tensor. If its not a scalar Tensor, its last dimensions must match x's.
            name(str, optional): The default value is None. Normally there is no need for
                user to set this property. For more information, please refer to :ref:`api_guide_Name`
        
        Returns:
            Tensor: The inner-product Tensor, the output shape is x.shape[:-1] + y.shape[:-1].
        
        Examples:
            .. code-block:: python
        
                import paddle
                x = paddle.arange(1, 7).reshape((2, 3)).astype('float32')
                y = paddle.arange(1, 10).reshape((3, 3)).astype('float32')
                out = paddle.inner(x, y)
                print(out)
                #        ([[14, 32, 50],
                #         [32, 77, 122]])        """
        pass


    def inverse(self, name: Optional[str] = None) -> Tensor:
        """
        Takes the inverse of the square matrix. A square matrix is a matrix with
        the same number of rows and columns. The input can be a square matrix
        (2-D Tensor) or batches of square matrices.
        
        Args:
            self (Tensor): The input tensor. The last two
                dimensions should be equal. When the number of dimensions is
                greater than 2, it is treated as batches of square matrix. The data
                type can be float32 and float64.
            name (str, optional): The default value is None. Normally there is no need for
                user to set this property. For more information,
                please refer to :ref:`api_guide_Name`
        
        Returns:
            Tensor: A Tensor holds the inverse of x. The shape and data type
                            is the same as x.
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                mat = paddle.to_tensor([[2, 0], [0, 2]], dtype='float32')
                inv = paddle.inverse(mat)
                print(inv) # [[0.5, 0], [0, 0.5]]        """
        pass


    def is_complex(self) -> bool:
        pass
    def is_empty(self, name: Optional[str] = None) -> Tensor:
        """
        Test whether a Tensor is empty.
        
        Args:
            self (Tensor): The Tensor to be tested.
            name (str, optional): The default value is ``None`` . Normally users
                                don't have to set this parameter. For more information,
                                please refer to :ref:`api_guide_Name` .
        
        Returns:
            Tensor: A bool scalar Tensor. True if 'x' is an empty Tensor.
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                input = paddle.rand(shape=[4, 32, 32], dtype='float32')
                res = paddle.is_empty(x=input)
                print("res:", res)
                # ('res:', Tensor: eager_tmp_1
                #    - place: CPUPlace
                #    - shape: [1]
                #    - layout: NCHW
                #    - dtype: bool
                #    - data: [0])        """
        pass


    def is_floating_point(self) -> bool:
        pass
    def is_integer(self) -> bool:
        pass
    def is_tensor(self) -> bool:
        """
        This function tests whether input object is a paddle.Tensor.
        
        Args:
            x (object): Object to test.
        
        Returns:
            A boolean value. True if 'x' is a paddle.Tensor, otherwise False.
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                input1 = paddle.rand(shape=[2, 3, 5], dtype='float32')
                check = paddle.is_tensor(input1)
                print(check)  #True
        
                input3 = [1, 4]
                check = paddle.is_tensor(input3)
                print(check)  #False
                        """
        pass


    def isclose(self, y: Tensor, rtol=1e-05, atol=1e-08, equal_nan=False, name: Optional[str] = None) -> Tensor:
        """
        This operator checks if all :math:`x` and :math:`y` satisfy the condition: 
        
        .. math:: \left| x - y \right| \leq atol + rtol \times \left| y \right| 
        
        elementwise, for all elements of :math:`x` and :math:`y`. The behaviour of this operator is analogous to :math:`numpy.isclose`, namely that it returns :math:`True` if two tensors are elementwise equal within a tolerance. 
        
        
        
        Args:
            self (Tensor): The input tensor, it's data type should be float32, float64.
            y(Tensor): The input tensor, it's data type should be float32, float64.
            rtol(rtoltype, optional): The relative tolerance. Default: :math:`1e-5` .
            atol(atoltype, optional): The absolute tolerance. Default: :math:`1e-8` .
            equal_nan(equalnantype, optional): If :math:`True` , then two :math:`NaNs` will be compared as equal. Default: :math:`False` .
            name (str, optional): Name for the operation. For more information, please
                refer to :ref:`api_guide_Name`. Default: None.
        
        Returns:
            Tensor: The output tensor, it's data type is bool.
        
        Raises:
            TypeError: The data type of ``x`` must be one of float32, float64.
            TypeError: The data type of ``y`` must be one of float32, float64.
            TypeError: The type of ``rtol`` must be float.
            TypeError: The type of ``atol`` must be float.
            TypeError: The type of ``equal_nan`` must be bool.
        
        Examples:
            .. code-block:: python
        
              import paddle
        
              x = paddle.to_tensor([10000., 1e-07])
              y = paddle.to_tensor([10000.1, 1e-08])
              result1 = paddle.isclose(x, y, rtol=1e-05, atol=1e-08,
                                      equal_nan=False, name="ignore_nan")
              np_result1 = result1.numpy()
              # [True, False]
              result2 = paddle.isclose(x, y, rtol=1e-05, atol=1e-08,
                                          equal_nan=True, name="equal_nan")
              np_result2 = result2.numpy()
              # [True, False]
        
              x = paddle.to_tensor([1.0, float('nan')])
              y = paddle.to_tensor([1.0, float('nan')])
              result1 = paddle.isclose(x, y, rtol=1e-05, atol=1e-08,
                                      equal_nan=False, name="ignore_nan")
              np_result1 = result1.numpy()
              # [True, False]
              result2 = paddle.isclose(x, y, rtol=1e-05, atol=1e-08,
                                          equal_nan=True, name="equal_nan")
              np_result2 = result2.numpy()
              # [True, True]        """
        pass


    def isfinite(self, name: Optional[str] = None) -> Tensor:
        """
        Return whether every element of input tensor is finite number or not.
        
        Args:
            self (Tensor): The input tensor, it's data type should be float16, float32, float64, int32, int64.
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            `Tensor`, the bool result which shows every element of `x` whether it is finite number or not.
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([float('-inf'), -2, 3.6, float('inf'), 0, float('-nan'), float('nan')])
                out = paddle.tensor.isfinite(x)
                print(out)  # [False  True  True False  True False False]        """
        pass


    def isinf(self, name: Optional[str] = None) -> Tensor:
        """
        Return whether every element of input tensor is `+/-INF` or not.
        
        Args:
            self (Tensor): The input tensor, it's data type should be float16, float32, float64, int32, int64.
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            `Tensor`, the bool result which shows every element of `x` whether it is `+/-INF` or not.
        
        Examples:
            .. code-block:: python
        
                import paddle
                x = paddle.to_tensor([float('-inf'), -2, 3.6, float('inf'), 0, float('-nan'), float('nan')])
                out = paddle.tensor.isinf(x)
                print(out)  # [ True False False  True False False False]        """
        pass


    def isnan(self, name: Optional[str] = None) -> Tensor:
        """
        Return whether every element of input tensor is `NaN` or not.
        
        Args:
            self (Tensor): The input tensor, it's data type should be float16, float32, float64, int32, int64.
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            `Tensor`, the bool result which shows every element of `x` whether it is `NaN` or not.
        
        Examples:
            .. code-block:: python
        
                import paddle
                x = paddle.to_tensor([float('-inf'), -2, 3.6, float('inf'), 0, float('-nan'), float('nan')])
                out = paddle.tensor.isnan(x)
                print(out)  # [False False False False False  True  True]        """
        pass


    def item(self, *args) -> Any:
        pass
    def kron(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        """
        Kron Operator. 
            
            This operator computes the Kronecker product of two tensors, a composite tensor made of blocks of the second tensor scaled by the first. 
            
            This operator assumes that the rank of the two tensors, $X$ and $Y$ are the same, if necessary prepending the smallest with ones. If the shape of $X$ is [$r_0$, $r_1$, ..., $r_N$] and the shape of $Y$ is [$s_0$, $s_1$, ..., $s_N$], then the shape of the output tensor is [$r_{0}s_{0}$, $r_{1}s_{1}$, ..., $r_{N}s_{N}$]. The elements are products of elements from $X$ and $Y$. 
            
            The equation is: $$ output[k_{0}, k_{1}, ..., k_{N}] = X[i_{0}, i_{1}, ..., i_{N}] * Y[j_{0}, j_{1}, ..., j_{N}] $$ 
            
            where $$ k_{t} = i_{t} * s_{t} + j_{t}, t = 0, 1, ..., N $$ 
            
            
        
            Args:
                self (Tensor): the fist operand of kron op, data type: float16, float32,
                    float64, int32 or int64.
                y (Tensor): the second operand of kron op, data type: float16,
                    float32, float64, int32 or int64. Its data type should be the same
                    with x.
                name(str, optional): The default value is None.  Normally there is no
                    need for user to set this property.  For more information, please
                    refer to :ref:`api_guide_Name`.
        
            Returns:
                Tensor: The output of kron op, data type: float16, float32, float64, int32 or int64. Its data is the same with x.
        
            Examples:
                .. code-block:: python
        
                    import paddle
                    x = paddle.to_tensor([[1, 2], [3, 4]], dtype='int64')
                    y = paddle.to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype='int64')
                    out = paddle.kron(x, y)
                    print(out)
                    #        [[1, 2, 3, 2, 4, 6],
                    #         [ 4,  5,  6,  8, 10, 12],
                    #         [ 7,  8,  9, 14, 16, 18],
                    #         [ 3,  6,  9,  4,  8, 12],
                    #         [12, 15, 18, 16, 20, 24],
                    #         [21, 24, 27, 28, 32, 36]])
                    """
        pass


    def kthvalue(self, k, axis: Optional[int] = None, keepdim: bool = False, name: Optional[str] = None) -> Tuple[Tensor, Tensor]:
        """
        This OP is used to find values and indices of the k-th smallest at the axis.
        
        Args:
            self (Tensor): A N-D Tensor with type float32, float64, int32, int64.
            k(int): The k for the k-th smallest number to look for along the axis.
            axis(int, optional): Axis to compute indices along. The effective range
                is [-R, R), where R is x.ndim. when axis < 0, it works the same way
                as axis + R. The default is None. And if the axis is None, it will computed as -1 by default.
            keepdim(bool, optional): Whether to keep the given axis in output. If it is True, the dimensions will be same as input x and with size one in the axis. Otherwise the output dimentions is one fewer than x since the axis is squeezed. Default is False.
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            tuple(Tensor), return the values and indices. The value data type is the same as the input `x`. The indices data type is int64.
        
        Examples:
        
            .. code-block:: python
        
                import paddle
                
                x = paddle.randn((2,3,2))
                # Tensor(shape=[2, 3, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
                #       [[[ 0.22954939, -0.01296274],
                #         [ 1.17135799, -0.34493217],
                #         [-0.19550551, -0.17573971]],
                #
                #        [[ 0.15104349, -0.93965352],
                #         [ 0.14745511,  0.98209465],
                #         [ 0.10732264, -0.55859774]]])           
                y = paddle.kthvalue(x, 2, 1)    
                # (Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
                # [[ 0.22954939, -0.17573971],
                #  [ 0.14745511, -0.55859774]]), Tensor(shape=[2, 2], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
                #  [[0, 2],
                #  [1, 2]]))        """
        pass


    def lcm(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        """
        Computes the element-wise least common multiple (LCM) of input |x| and |y|.
        Both x and y must have integer types.
        
        Note:
            lcm(0,0)=0, lcm(0, y)=0
        
            If x.shape != y.shape, they must be broadcastable to a common shape (which becomes the shape of the output).
        
        Args:
            self (Tensor): An N-D Tensor, the data type is int32，int64. 
            y (Tensor): An N-D Tensor, the data type is int32，int64. 
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            out (Tensor): An N-D Tensor, the data type is the same with input.
        
        Examples:
            .. code-block:: python
        
                import paddle
                
                x1 = paddle.to_tensor(12)
                x2 = paddle.to_tensor(20)
                paddle.lcm(x1, x2)
                # Tensor(shape=[1], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
                #        [60])
        
                x3 = paddle.arange(6)
                paddle.lcm(x3, x2)
                # Tensor(shape=[6], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
                #        [0, 20, 20, 60, 20, 20])
        
                x4 = paddle.to_tensor(0)
                paddle.lcm(x4, x2)
                # Tensor(shape=[1], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
                #        [0])
        
                paddle.lcm(x4, x4)
                # Tensor(shape=[1], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
                #        [0])
                
                x5 = paddle.to_tensor(-20)
                paddle.lcm(x1, x5)
                # Tensor(shape=[1], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
                #        [60])        """
        pass


    def lerp(self, y: Tensor, weight, name: Optional[str] = None) -> Tensor:
        """
        Does a linear interpolation between x and y based on weight.
        
        Equation:
            .. math::
        
                lerp(x, y, weight) = x + weight * (y - x).
        
        Args:
            self (Tensor): An N-D Tensor with starting points, the data type is float32, float64.
            y (Tensor): An N-D Tensor with ending points, the data type is float32, float64.
            weight (float|Tensor): The weight for the interpolation formula. When weight is Tensor, the data type is float32, float64.
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            out (Tensor): An N-D Tensor, the shape and data type is the same with input.
        
        Example:
            .. code-block:: python
        
                import paddle
                
                x = paddle.arange(1., 5., dtype='float32')
                y = paddle.empty([4], dtype='float32')
                y.fill_(10.)
                out = paddle.lerp(start, end, 0.5)
                # out: [5.5., 6., 6.5, 7.]        """
        pass


    def lerp_(self, y: Tensor, weight, name: Optional[str] = None) -> None:
        """
        Inplace version of ``lerp`` API, the output Tensor will be inplaced with input ``x``.
        Please refer to :ref:`api_tensor_lerp`.        """
        pass


    def less_equal(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        """
        This OP returns the truth value of :math:`x <= y` elementwise, which is equivalent function to the overloaded operator `<=`.
        
        **NOTICE**: The output of this OP has no gradient.
        
        Args:
            self (Tensor): First input to compare which is N-D tensor. The input data type should be bool, float32, float64, int32, int64.
            y(Tensor): Second input to compare which is N-D tensor. The input data type should be bool, float32, float64, int32, int64.
            name(str, optional): The default value is None.  Normally there is no need for
                user to set this property.  For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            Tensor, the output data type is bool: The tensor storing the output, the output shape is same as input :attr:`x`.
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([1, 2, 3])
                y = paddle.to_tensor([1, 3, 2])
                result1 = paddle.less_equal(x, y)
                print(result1)  # result1 = [True True False]        """
        pass


    def less_than(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        """
        This OP returns the truth value of :math:`x < y` elementwise, which is equivalent function to the overloaded operator `<`.
        
        **NOTICE**: The output of this OP has no gradient.
        
        Args:
            self (Tensor): First input to compare which is N-D tensor. The input data type should be bool, float32, float64, int32, int64.
            y(Tensor): Second input to compare which is N-D tensor. The input data type should be bool, float32, float64, int32, int64.
            name(str, optional): The default value is None.  Normally there is no need for
                user to set this property.  For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            Tensor, the output data type is bool: The tensor storing the output, the output shape is same as input :attr:`x`.
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([1, 2, 3])
                y = paddle.to_tensor([1, 3, 2])
                result1 = paddle.less_than(x, y)
                print(result1)  # result1 = [False True False]        """
        pass


    def lgamma(self, name: Optional[str] = None) -> Tensor:
        """
        Lgamma Operator.
        
        This operator performs elementwise lgamma for input $X$.
        :math:`out = log\Gamma(x)`
        
        
        Args:
            self (Tensor): (Tensor), The input tensor of lgamma op.
            with_quant_attr (BOOLEAN): Whether the operator has attributes used by quantization. 
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            out (Tensor): (Tensor), The output tensor of lgamma op.
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
                out = paddle.lgamma(x)
                print(out)
                # [1.31452441, 1.76149750, 2.25271273, 1.09579802]        """
        pass


    def log(self, name: Optional[str] = None) -> Tensor:
        """
        Calculates the natural log of the given input tensor, element-wise.
        
        .. math::
        
            Out = \\ln(x)
        
        Args:
            self (Tensor): Input Tensor. Must be one of the following types: float32, float64.
            name (str|None): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`
        
        
        Returns:
            Tensor: The natural log of the input Tensor computed element-wise.
        
        Examples:
        
            .. code-block:: python
        
                import paddle
        
                x = [[2,3,4], [7,8,9]]
                x = paddle.to_tensor(x, dtype='float32')
                res = paddle.log(x)
                # [[0.693147, 1.09861, 1.38629], [1.94591, 2.07944, 2.19722]]        """
        pass


    def log10(self, name: Optional[str] = None) -> Tensor:
        """
        Calculates the log to the base 10 of the given input tensor, element-wise.
        
        .. math::
        
            Out = \\log_10_x
        
        Args:
            self (Tensor): Input tensor must be one of the following types: float32, float64.
            name (str|None): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`
        
        
        Returns:
            Tensor: The log to the base 10 of the input Tensor computed element-wise.
        
        Examples:
        
            .. code-block:: python
            
                import paddle
        
                # example 1: x is a float
                x_i = paddle.to_tensor([[1.0], [10.0]])
                res = paddle.log10(x_i) # [[0.], [1.0]]
        
                # example 2: x is float32
                x_i = paddle.full(shape=[1], fill_value=10, dtype='float32')
                paddle.to_tensor(x_i)
                res = paddle.log10(x_i)
                print(res) # [1.0]
        
                # example 3: x is float64
                x_i = paddle.full(shape=[1], fill_value=10, dtype='float64')
                paddle.to_tensor(x_i)
                res = paddle.log10(x_i)
                print(res) # [1.0]        """
        pass


    def log1p(self, name: Optional[str] = None) -> Tensor:
        """
        Calculates the natural log of the given input tensor, element-wise.
        
        .. math::
            Out = \\ln(x+1)
        
        Args:
            self (Tensor): Input Tensor. Must be one of the following types: float32, float64.
            name(str, optional): The default value is None.  Normally there is no need for 
                user to set this property.  For more information, please refer to :ref:`api_guide_Name`
        Returns:
            Tensor, the natural log of the input Tensor computed element-wise.
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                data = paddle.to_tensor([[0], [1]], dtype='float32')
                res = paddle.log1p(data)
                # [[0.], [0.6931472]]        """
        pass


    def log2(self, name: Optional[str] = None) -> Tensor:
        """
        Calculates the log to the base 2 of the given input tensor, element-wise.
        
        .. math::
        
            Out = \\log_2x
        
        Args:
            self (Tensor): Input tensor must be one of the following types: float32, float64.
            name (str|None): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`
        
        
        Returns:
            Tensor: The log to the base 2 of the input Tensor computed element-wise.
        
        Examples:
        
            .. code-block:: python
            
                import paddle
        
                # example 1: x is a float
                x_i = paddle.to_tensor([[1.0], [2.0]])
                res = paddle.log2(x_i) # [[0.], [1.0]]
        
                # example 2: x is float32
                x_i = paddle.full(shape=[1], fill_value=2, dtype='float32')
                paddle.to_tensor(x_i)
                res = paddle.log2(x_i)
                print(res) # [1.0]
        
                # example 3: x is float64
                x_i = paddle.full(shape=[1], fill_value=2, dtype='float64')
                paddle.to_tensor(x_i)
                res = paddle.log2(x_i)
                print(res) # [1.0]        """
        pass


    def logical_and(self, y: Tensor, out=None, name: Optional[str] = None) -> Tensor:
        """
        ``logical_and`` operator computes element-wise logical AND on ``x`` and ``y``, and returns ``out``. ``out`` is N-dim boolean ``Tensor``.
        Each element of ``out`` is calculated by
        
        .. math::
        
            out = x \&\& y
        
        .. note::
            ``paddle.logical_and`` supports broadcasting. If you want know more about broadcasting, please refer to :ref:`user_guide_broadcasting`.
        
        Args:
            self (Tensor): the input tensor, it's data type should be one of bool, int8, int16, in32, in64, float32, float64.
            y (Tensor): the input tensor, it's data type should be one of bool, int8, int16, in32, in64, float32, float64.
            out(Tensor): The ``Tensor`` that specifies the output of the operator, which can be any ``Tensor`` that has been created in the program. The default value is None, and a new ``Tensor`` will be created to save the output.
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            N-D Tensor. A location into which the result is stored. It's dimension equals with ``x``.
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([True])
                y = paddle.to_tensor([True, False, True, False])
                res = paddle.logical_and(x, y)
                print(res) # [True False True False]        """
        pass


    def logical_not(self, out=None, name: Optional[str] = None) -> Tensor:
        """
        ``logical_not`` operator computes element-wise logical NOT on ``x``, and returns ``out``. ``out`` is N-dim boolean ``Variable``.
        Each element of ``out`` is calculated by
        
        .. math::
        
            out = !x
        
        Args:
            self (Tensor):  Operand of logical_not operator. Must be a Tensor of type bool, int8, int16, in32, in64, float32, or float64.
            out(Tensor): The ``Tensor`` that specifies the output of the operator, which can be any ``Tensor`` that has been created in the program. The default value is None, and a new ``Tensor` will be created to save the output.
            name(str|None): The default value is None. Normally there is no need for users to set this property. For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            Tensor: n-dim bool LoDTensor or Tensor
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([True, False, True, False])
                res = paddle.logical_not(x)
                print(res) # [False  True False  True]        """
        pass


    def logical_or(self, y: Tensor, out=None, name: Optional[str] = None) -> Tensor:
        """
        ``logical_or`` operator computes element-wise logical OR on ``x`` and ``y``, and returns ``out``. ``out`` is N-dim boolean ``Tensor``.
        Each element of ``out`` is calculated by
        
        .. math::
        
            out = x || y
        
        .. note::
            ``paddle.logical_or`` supports broadcasting. If you want know more about broadcasting, please refer to :ref:`user_guide_broadcasting`.
        
        Args:
            self (Tensor): the input tensor, it's data type should be one of bool, int8, int16, in32, in64, float32, float64.
            y (Tensor): the input tensor, it's data type should be one of bool, int8, int16, in32, in64, float32, float64.
            out(Tensor): The ``Variable`` that specifies the output of the operator, which can be any ``Tensor`` that has been created in the program. The default value is None, and a new ``Tensor`` will be created to save the output.
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            N-D Tensor. A location into which the result is stored. It's dimension equals with ``x``.
        
        Examples:
            .. code-block:: python
        
                import paddle
                import numpy as np
        
                x_data = np.array([True, False], dtype=np.bool).reshape(2, 1)
                y_data = np.array([True, False, True, False], dtype=np.bool).reshape(2, 2)
                x = paddle.to_tensor(x_data)
                y = paddle.to_tensor(y_data)
                res = paddle.logical_or(x, y)
                print(res) # [[ True  True] [ True False]]        """
        pass


    def logical_xor(self, y: Tensor, out=None, name: Optional[str] = None) -> Tensor:
        """
        ``logical_xor`` operator computes element-wise logical XOR on ``x`` and ``y``, and returns ``out``. ``out`` is N-dim boolean ``Tensor``.
        Each element of ``out`` is calculated by
        
        .. math::
        
            out = (x || y) \&\& !(x \&\& y)
        
        .. note::
            ``paddle.logical_xor`` supports broadcasting. If you want know more about broadcasting, please refer to :ref:`user_guide_broadcasting`.
        
        Args:
            self (Tensor): the input tensor, it's data type should be one of bool, int8, int16, in32, in64, float32, float64.
            y (Tensor): the input tensor, it's data type should be one of bool, int8, int16, in32, in64, float32, float64.
            out(Tensor): The ``Tensor`` that specifies the output of the operator, which can be any ``Tensor`` that has been created in the program. The default value is None, and a new ``Tensor`` will be created to save the output.
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            N-D Tensor. A location into which the result is stored. It's dimension equals with ``x``.
        
        Examples:
            .. code-block:: python
        
                import paddle
                import numpy as np
        
                x_data = np.array([True, False], dtype=np.bool).reshape([2, 1])
                y_data = np.array([True, False, True, False], dtype=np.bool).reshape([2, 2])
                x = paddle.to_tensor(x_data)
                y = paddle.to_tensor(y_data)
                res = paddle.logical_xor(x, y)
                print(res) # [[False,  True], [ True, False]]        """
        pass


    def logit(self, eps=None, name: Optional[str] = None) -> Tensor:
        """
        This function generates a new tensor with the logit of the elements of input x. x is clamped to [eps, 1-eps] when eps is not zero. When eps is zero and x < 0 or x > 1, the function will yields NaN.
        
        .. math::
        
            logit(x) = ln(\frac{x}{1 - x})
        
        where
        
        .. math::
        
            x_i=
                \left\{\begin{array}{rcl}
                    x_i & &\text{if } eps == Default \\
                    eps & &\text{if } x_i < eps \\
                    x_i & &\text{if } eps <= x_i <= 1-eps \\
                    1-eps & &\text{if } x_i > 1-eps
                \end{array}\right.
        
        Args:
            self (Tensor): The input Tensor with data type float32, float64.
            eps (float, optional):  the epsilon for input clamp bound. Default is None.
            name (str, optional): Name for the operation (optional, default is None).
                For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            out(Tensor): A Tensor with the same data type and shape as ``x`` .
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([0.2635, 0.0106, 0.2780, 0.2097, 0.8095])
                out1 = paddle.logit(x)
                print(out1)
                # [-1.0277, -4.5365, -0.9544, -1.3269,  1.4468]          """
        pass


    def logsumexp(self, axis: Optional[int] = None, keepdim: bool = False, name: Optional[str] = None) -> Tensor:
        """
        This OP calculates the log of the sum of exponentials of ``x`` along ``axis`` .
        
        .. math::
           logsumexp(x) = \\log\\sum exp(x)
        
        Args:
            self (Tensor): The input Tensor with data type float32 or float64, which 
                have no more than 4 dimensions.
            axis (int|list|tuple, optional): The axis along which to perform
                logsumexp calculations. ``axis`` should be int, list(int) or
                tuple(int). If ``axis`` is a list/tuple of dimension(s), logsumexp
                is calculated along all element(s) of ``axis`` . ``axis`` or
                element(s) of ``axis`` should be in range [-D, D), where D is the
                dimensions of ``x`` . If ``axis`` or element(s) of ``axis`` is
                less than 0, it works the same way as :math:`axis + D` . If
                ``axis`` is None, logsumexp is calculated along all elements of
                ``x``. Default is None.
            keepdim (bool, optional): Whether to reserve the reduced dimension(s)
                in the output Tensor. If ``keep_dim`` is True, the dimensions of
                the output Tensor is the same as ``x`` except in the reduced
                dimensions(it is of size 1 in this case). Otherwise, the shape of
                the output Tensor is squeezed in ``axis`` . Default is False.
            name (str, optional): Name for the operation (optional, default is None).
                For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            Tensor, results of logsumexp along ``axis`` of ``x``, with the same data
            type as ``x``.
        
        Examples:
        
        .. code-block:: python
        
            import paddle
        
            x = paddle.to_tensor([[-1.5, 0., 2.], [3., 1.2, -2.4]])
            out1 = paddle.logsumexp(x) # [3.4691226]
            out2 = paddle.logsumexp(x, 1) # [2.15317821, 3.15684602]        """
        pass


    def lstsq(self, y: Tensor, rcond=None, driver=None, name: Optional[str] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Computes a solution to
        the least squares problem of a system of linear equations.
        
        Args:
            self (Tensor): A tensor with shape ``(*, M, N)`` , the data type of the input Tensor ``x``
                should be one of float32, float64.
            y (Tensor): A tensor with shape ``(*, M, K)`` , the data type of the input Tensor ``y`` 
                should be one of float32, float64.
            rcond(float, optional): The default value is None. A float pointing number used to determine 
                the effective rank of ``x``. If ``rcond`` is None, it will be set to max(M, N) times the 
                machine precision of x_dtype.
            driver(str, optional): The default value is None. The name of LAPACK method to be used. For 
                CPU inputs the valid values are ‘gels’, ‘gelsy’, ‘gelsd, ‘gelss’. For CUDA input, the only 
                valid driver is ‘gels’. If ``driver`` is None, ‘gelsy’ is used for CPU inputs and ‘gels’ 
                for CUDA inputs.
            name(str, optional): The default value is None. Normally there is no need for user to set 
                this property. For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            Tuple: A tuple of 4 Tensors which is (``solution``, ``residuals``, ``rank``, ``singular_values``). 
            ``solution`` is a tensor with shape ``(*, N, K)``, meaning the least squares solution. ``residuals`` 
            is a tensor with shape ``(*, K)``, meaning the squared residuals of the solutions, which is computed 
            when M > N and every matrix in ``x`` is full-rank, otherwise return an empty tensor. ``rank`` is a tensor 
            with shape ``(*)``, meaning the ranks of the matrices in ``x``, which is computed when ``driver`` in 
            (‘gelsy’, ‘gelsd’, ‘gelss’), otherwise return an empty tensor. ``singular_values`` is a tensor with 
            shape ``(*, min(M, N))``, meaning singular values of the matrices in ``x``, which is computed when 
            ``driver`` in (‘gelsd’, ‘gelss’), otherwise return an empty tensor.
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                paddle.set_device("cpu")
                x = paddle.to_tensor([[1, 3], [3, 2], [5, 6.]])
                y = paddle.to_tensor([[3, 4, 6], [5, 3, 4], [1, 2, 1.]])
                results = paddle.linalg.lstsq(x, y, driver="gelsd")
                print(results[0])
                # [[ 0.78350395, -0.22165027, -0.62371236],
                # [-0.11340097,  0.78866047,  1.14948535]]
                print(results[1])
                # [19.81443405, 10.43814468, 30.56185532])
                print(results[2])
                # 2
                print(results[3])
                # [9.03455734, 1.54167950]
        
                x = paddle.to_tensor([[10, 2, 3], [3, 10, 5], [5, 6, 12.]])
                y = paddle.to_tensor([[4, 2, 9], [2, 0, 3], [2, 5, 3.]])
                results = paddle.linalg.lstsq(x, y, driver="gels")
                print(results[0])
                # [[ 0.39386186,  0.10230173,  0.93606132],
                # [ 0.10741687, -0.29028133,  0.11892585],
                # [-0.05115091,  0.51918161, -0.19948854]]
                print(results[1])
                # []        """
        pass


    def lu(self, pivot=True, get_infos=False, name: Optional[str] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Computes the LU factorization of an N-D(N>=2) matrix x. 
        
        Returns the LU factorization(inplace x) and Pivots. low triangular matrix L and 
        upper triangular matrix U are combined to a single LU matrix.
        
        Pivoting is done if pivot is set to True.
        P mat can be get by pivots:
        # ones = eye(rows) #eye matrix of rank rows
        # for i in range(cols):
        #     swap(ones[i], ones[pivots[i]])
        # return ones
        
        Args:
        
            X (Tensor): the tensor to factor of N-dimensions(N>=2).
        
            pivot (bool, optional): controls whether pivoting is done. Default: True.
        
            get_infos (bool, optional): if set to True, returns an info IntTensor. Default: False.
        
            name (str, optional): Name for the operation (optional, default is None).
                For more information, please refer to :ref:`api_guide_Name`.
                
        Returns:
            factorization (Tensor): LU matrix, the factorization of input X.
        
            pivots (IntTensor): the pivots of size(∗(N-2), min(m,n)). `pivots` stores all the 
                        intermediate transpositions of rows. The final permutation `perm` could be 
                        reconstructed by this, details refer to upper example.
        
            infos (IntTensor, optional): if `get_infos` is `True`, this is a tensor of size (∗(N-2)) 
                        where non-zero values indicate whether factorization for the matrix or each minibatch 
                        has succeeded or failed.
        
            
        Examples:            
            .. code-block:: python
        
                import paddle 
        
                x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).astype('float64')
                lu,p,info = paddle.linalg.lu(x, get_infos=True)
        
                # >>> lu:
                # Tensor(shape=[3, 2], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
                #    [[5.        , 6.        ],
                #        [0.20000000, 0.80000000],
                #        [0.60000000, 0.50000000]])
                # >>> p
                # Tensor(shape=[2], dtype=int32, place=CUDAPlace(0), stop_gradient=True,
                #    [3, 3])
                # >>> info
                # Tensor(shape=[], dtype=int32, place=CUDAPlace(0), stop_gradient=True,
                #    0)
                
                P,L,U = paddle.linalg.lu_unpack(lu,p)
        
                # >>> P
                # (Tensor(shape=[3, 3], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
                # [[0., 1., 0.],
                # [0., 0., 1.],
                # [1., 0., 0.]]), 
                # >>> L
                # Tensor(shape=[3, 2], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
                # [[1.        , 0.        ],
                # [0.20000000, 1.        ],
                # [0.60000000, 0.50000000]]), 
                # >>> U
                # Tensor(shape=[2, 2], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
                # [[5.        , 6.        ],
                # [0.        , 0.80000000]]))
                
        
                # one can verify : X = P @ L @ U ;             """
        pass


    def lu_unpack(self, y: Tensor, unpack_ludata=True, unpack_pivots=True, name: Optional[str] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Unpack L U and P to single matrix tensor . 
        unpack L and U matrix from LU, unpack permutation matrix P from Pivtos .
        
        P mat can be get by pivots:
        # ones = eye(rows) #eye matrix of rank rows
        # for i in range(cols):
        #     swap(ones[i], ones[pivots[i]])
        
        
        Args:
            self (Tensor): The LU tensor get from paddle.linalg.lu, which is combined by L and U.
        
            y (Tensor): Pivots get from paddle.linalg.lu.
        
            unpack_ludata (bool,optional): whether to unpack L and U from x. Default: True.
        
            unpack_pivots (bool, optional): whether to unpack permutation matrix P from Pivtos. Default: True.
        
            name (str, optional): Name for the operation (optional, default is None).
                For more information, please refer to :ref:`api_guide_Name`.
                
        Returns:
            P (Tensor): Permutation matrix P of lu factorization.
        
            L (Tensor): The lower triangular matrix tensor of lu factorization.
        
            U (Tensor): The upper triangular matrix tensor of lu factorization.
        
            
        Examples:            
            .. code-block:: python
        
                import paddle 
        
                x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).astype('float64')
                lu,p,info = paddle.linalg.lu(x, get_infos=True)
        
                # >>> lu:
                # Tensor(shape=[3, 2], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
                #    [[5.        , 6.        ],
                #        [0.20000000, 0.80000000],
                #        [0.60000000, 0.50000000]])
                # >>> p
                # Tensor(shape=[2], dtype=int32, place=CUDAPlace(0), stop_gradient=True,
                #    [3, 3])
                # >>> info
                # Tensor(shape=[], dtype=int32, place=CUDAPlace(0), stop_gradient=True,
                #    0)
                
                P,L,U = paddle.linalg.lu_unpack(lu,p)
        
                # >>> P
                # (Tensor(shape=[3, 3], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
                # [[0., 1., 0.],
                # [0., 0., 1.],
                # [1., 0., 0.]]), 
                # >>> L
                # Tensor(shape=[3, 2], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
                # [[1.        , 0.        ],
                # [0.20000000, 1.        ],
                # [0.60000000, 0.50000000]]), 
                # >>> U
                # Tensor(shape=[2, 2], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
                # [[5.        , 6.        ],
                # [0.        , 0.80000000]]))
        
                # one can verify : X = P @ L @ U ;           """
        pass


    def masked_select(self, mask, name: Optional[str] = None) -> Tensor:
        """
        This OP Returns a new 1-D tensor which indexes the input tensor according to the ``mask``
        which is a tensor with data type of bool.
        
        Args:
            self (Tensor): The input Tensor, the data type can be int32, int64, float32, float64. 
            mask (Tensor): The Tensor containing the binary mask to index with, it's data type is bool.
            name(str, optional): The default value is None. Normally there is no
                need for user to set this property. For more information, please
                refer to :ref:`api_guide_Name`.
        
        Returns: A 1-D Tensor which is the same data type  as ``x``.
        
        Examples:
        
            .. code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0],
                                      [5.0, 6.0, 7.0, 8.0],
                                      [9.0, 10.0, 11.0, 12.0]])
                mask = paddle.to_tensor([[True, False, False, False],
                                         [True, True, False, False],
                                         [True, False, False, False]])
                out = paddle.masked_select(x, mask)
                #[1.0 5.0 6.0 9.0]        """
        pass


    def matmul(self, y: Tensor, transpose_x=False, transpose_y=False, name: Optional[str] = None) -> Tensor:
        """
        Applies matrix multiplication to two tensors. `matmul` follows
        the complete broadcast rules,
        and its behavior is consistent with `np.matmul`.
        
        Currently, the input tensors' number of dimensions can be any, `matmul` can be used to
        achieve the `dot`, `matmul` and `batchmatmul`.
        
        The actual behavior depends on the shapes of :math:`x`, :math:`y` and the
        flag values of :attr:`transpose_x`, :attr:`transpose_y`. Specifically:
        
        - If a transpose flag is specified, the last two dimensions of the tensor
          are transposed. If the tensor is ndim-1 of shape, the transpose is invalid. If the tensor
          is ndim-1 of shape :math:`[D]`, then for :math:`x` it is treated as :math:`[1, D]`, whereas
          for :math:`y` it is the opposite: It is treated as :math:`[D, 1]`.
        
        The multiplication behavior depends on the dimensions of `x` and `y`. Specifically:
        
        - If both tensors are 1-dimensional, the dot product result is obtained.
        
        - If both tensors are 2-dimensional, the matrix-matrix product is obtained.
        
        - If the `x` is 1-dimensional and the `y` is 2-dimensional,
          a `1` is prepended to its dimension in order to conduct the matrix multiply.
          After the matrix multiply, the prepended dimension is removed.
        
        - If the `x` is 2-dimensional and `y` is 1-dimensional,
          the matrix-vector product is obtained.
        
        - If both arguments are at least 1-dimensional and at least one argument
          is N-dimensional (where N > 2), then a batched matrix multiply is obtained.
          If the first argument is 1-dimensional, a 1 is prepended to its dimension
          in order to conduct the batched matrix multiply and removed after.
          If the second argument is 1-dimensional, a 1 is appended to its
          dimension for the purpose of the batched matrix multiple and removed after.
          The non-matrix (exclude the last two dimensions) dimensions are
          broadcasted according the broadcast rule.
          For example, if input is a (j, 1, n, m) tensor and the other is a (k, m, p) tensor,
          out will be a (j, k, n, p) tensor.
        
        Args:
            self (Tensor): The input tensor which is a Tensor.
            y (Tensor): The input tensor which is a Tensor.
            transpose_x (bool): Whether to transpose :math:`x` before multiplication.
            transpose_y (bool): Whether to transpose :math:`y` before multiplication.
            name(str|None): A name for this layer(optional). If set None, the layer
                will be named automatically.
        
        Returns:
            Tensor: The output Tensor.
        
        Examples:
        
        .. code-block:: python
        
            import paddle
            import numpy as np
        
            # vector * vector
            x_data = np.random.random([10]).astype(np.float32)
            y_data = np.random.random([10]).astype(np.float32)
            x = paddle.to_tensor(x_data)
            y = paddle.to_tensor(y_data)
            z = paddle.matmul(x, y)
            print(z.numpy().shape)
            # [1]
        
            # matrix * vector
            x_data = np.random.random([10, 5]).astype(np.float32)
            y_data = np.random.random([5]).astype(np.float32)
            x = paddle.to_tensor(x_data)
            y = paddle.to_tensor(y_data)
            z = paddle.matmul(x, y)
            print(z.numpy().shape)
            # [10]
        
            # batched matrix * broadcasted vector
            x_data = np.random.random([10, 5, 2]).astype(np.float32)
            y_data = np.random.random([2]).astype(np.float32)
            x = paddle.to_tensor(x_data)
            y = paddle.to_tensor(y_data)
            z = paddle.matmul(x, y)
            print(z.numpy().shape)
            # [10, 5]
        
            # batched matrix * batched matrix
            x_data = np.random.random([10, 5, 2]).astype(np.float32)
            y_data = np.random.random([10, 2, 5]).astype(np.float32)
            x = paddle.to_tensor(x_data)
            y = paddle.to_tensor(y_data)
            z = paddle.matmul(x, y)
            print(z.numpy().shape)
            # [10, 5, 5]
        
            # batched matrix * broadcasted matrix
            x_data = np.random.random([10, 1, 5, 2]).astype(np.float32)
            y_data = np.random.random([1, 3, 2, 5]).astype(np.float32)
            x = paddle.to_tensor(x_data)
            y = paddle.to_tensor(y_data)
            z = paddle.matmul(x, y)
            print(z.numpy().shape)
            # [10, 3, 5, 5]        """
        pass


    def matrix_power(self, n, name: Optional[str] = None) -> Tensor:
        """
        Computes the n-th power of a square matrix or a batch of square matrices.
        
        Let :math:`X` be a sqaure matrix or a batch of square matrices, :math:`n` be
        an exponent, the equation should be:
        
        .. math::
            Out = X ^ {n}
        
        Specifically,
        
        - If `n > 0`, it returns the matrix or a batch of matrices raised to the power
        of `n`.
        
        - If `n = 0`, it returns the identity matrix or a batch of identity matrices.
        
        - If `n < 0`, it returns the inverse of each matrix (if invertible) raised to
        the power of `abs(n)`.
        
        Args:
            self (Tensor): A square matrix or a batch of square matrices to be raised
                to power `n`. Its shape should be `[*, M, M]`, where `*` is zero or
                more batch dimensions. Its data type should be float32 or float64.
            n (int): The exponent. It can be any positive, negative integer or zero.
            name (str, optional): Name for the operation (optional, default is None).
                For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            Tensor: The n-th power of the matrix (or the batch of matrices) `x`. Its
                data type should be the same as that of `x`.
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([[1, 2, 3],
                                      [1, 4, 9],
                                      [1, 8, 27]], dtype='float64')
                print(paddle.linalg.matrix_power(x, 2))
                # [[6.  , 34. , 102.],
                #  [14. , 90. , 282.],
                #  [36. , 250., 804.]]
        
                print(paddle.linalg.matrix_power(x, 0))
                # [[1., 0., 0.],
                #  [0., 1., 0.],
                #  [0., 0., 1.]]
        
                print(paddle.linalg.matrix_power(x, -2))
                # [[ 12.91666667, -12.75000000,  2.83333333 ],
                #  [-7.66666667 ,  8.         , -1.83333333 ],
                #  [ 1.80555556 , -1.91666667 ,  0.44444444 ]]        """
        pass


    def max(self, axis: Optional[int] = None, keepdim: bool = False, name: Optional[str] = None) -> Tensor:
        """
        Computes the maximum of tensor elements over the given axis.
        
        Note:
            The difference between max and amax is: If there are multiple maximum elements,
            amax evenly distributes gradient between these equal values, 
            while max propagates gradient to all of them.
        
        
        Args:
            self (Tensor): A tensor, the data type is float32, float64, int32, int64.
            axis(int|list|tuple, optional): The axis along which the maximum is computed.
                If :attr:`None`, compute the maximum over all elements of
                `x` and return a Tensor with a single element,
                otherwise must be in the range :math:`[-x.ndim(x), x.ndim(x))`.
                If :math:`axis[i] < 0`, the axis to reduce is :math:`x.ndim + axis[i]`.
            keepdim(bool, optional): Whether to reserve the reduced dimension in the
                output Tensor. The result tensor will have one fewer dimension
                than the `x` unless :attr:`keepdim` is true, default
                value is False.
            name(str, optional): The default value is None.  Normally there is no need for
                user to set this property.  For more information, please refer to :ref:`api_guide_Name`
        
        Returns:
            Tensor, results of maximum on the specified axis of input tensor,
            it's data type is the same as `x`.
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                # data_x is a Tensor with shape [2, 4]
                # the axis is a int element
                x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
                                      [0.1, 0.2, 0.6, 0.7]], 
                                     dtype='float64', stop_gradient=False)
                result1 = paddle.max(x)
                result1.backward()
                print(result1, x.grad) 
                #[0.9], [[0., 0., 0., 1.], [0., 0., 0., 0.]]
        
                x.clear_grad()
                result2 = paddle.max(x, axis: int = 0)
                result2.backward()
                print(result2, x.grad) 
                #[0.2, 0.3, 0.6, 0.9], [[1., 1., 0., 1.], [0., 0., 1., 0.]]
        
                x.clear_grad()
                result3 = paddle.max(x, axis: int = -1)
                result3.backward()
                print(result3, x.grad) 
                #[0.9, 0.7], [[0., 0., 0., 1.], [0., 0., 0., 1.]]
        
                x.clear_grad()
                result4 = paddle.max(x, axis: int = 1, keepdim=True)
                result4.backward()
                print(result4, x.grad) 
                #[[0.9], [0.7]], [[0., 0., 0., 1.], [0., 0., 0., 1.]]
        
                # data_y is a Tensor with shape [2, 2, 2]
                # the axis is list 
                y = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0]],
                                      [[5.0, 6.0], [7.0, 8.0]]],
                                     dtype='float64', stop_gradient=False)
                result5 = paddle.max(y, axis: int = [1, 2])
                result5.backward()
                print(result5, y.grad) 
                #[4., 8.], [[[0., 0.], [0., 1.]], [[0., 0.], [0., 1.]]]
        
                y.clear_grad()
                result6 = paddle.max(y, axis: int = [0, 1])
                result6.backward()
                print(result6, y.grad) 
                #[7., 8.], [[[0., 0.], [0., 0.]], [[0., 0.], [1., 1.]]]        """
        pass


    def maximum(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        """
        Compare two tensors and returns a new tensor containing the element-wise maxima. The equation is:
        
        .. math::
            out = max(x, y)
        
        **Note**:
        ``paddle.maximum`` supports broadcasting. If you want know more about broadcasting, please refer to :ref:`user_guide_broadcasting` .
        
        Args:
            self (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
            y (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            N-D Tensor. A location into which the result is stored. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape,  its shape is the same as x and y.
        
        Examples:
        
            .. code-block:: python
        
                import numpy as np
                import paddle
        
                x = paddle.to_tensor([[1, 2], [7, 8]])
                y = paddle.to_tensor([[3, 4], [5, 6]])
                res = paddle.maximum(x, y)
                print(res)
                #    [[3, 4],
                #     [7, 8]]
        
                x = paddle.to_tensor([[1, 2, 3], [1, 2, 3]])
                y = paddle.to_tensor([3, 0, 4])
                res = paddle.maximum(x, y)
                print(res)
                #    [[3, 2, 4],
                #     [3, 2, 4]]
        
                x = paddle.to_tensor([2, 3, 5], dtype='float32')
                y = paddle.to_tensor([1, np.nan, np.nan], dtype='float32')
                res = paddle.maximum(x, y)
                print(res)
                #    [ 2., nan, nan]
        
                x = paddle.to_tensor([5, 3, np.inf], dtype='float32')
                y = paddle.to_tensor([1, -np.inf, 5], dtype='float32')
                res = paddle.maximum(x, y)
                print(res)
                #    [  5.,   3., inf.]        """
        pass


    def mean(self, axis: Optional[int] = None, keepdim: bool = False, name: Optional[str] = None) -> Tensor:
        """
        Computes the mean of the input tensor's elements along ``axis``.
        
        Args:
            self (Tensor): The input Tensor with data type float32, float64.
            axis (int|list|tuple, optional): The axis along which to perform mean
                calculations. ``axis`` should be int, list(int) or tuple(int). If
                ``axis`` is a list/tuple of dimension(s), mean is calculated along
                all element(s) of ``axis`` . ``axis`` or element(s) of ``axis``
                should be in range [-D, D), where D is the dimensions of ``x`` . If
                ``axis`` or element(s) of ``axis`` is less than 0, it works the
                same way as :math:`axis + D` . If ``axis`` is None, mean is
                calculated over all elements of ``x``. Default is None.
            keepdim (bool, optional): Whether to reserve the reduced dimension(s)
                in the output Tensor. If ``keepdim`` is True, the dimensions of
                the output Tensor is the same as ``x`` except in the reduced
                dimensions(it is of size 1 in this case). Otherwise, the shape of
                the output Tensor is squeezed in ``axis`` . Default is False.
            name (str, optional): Name for the operation (optional, default is None).
                For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            Tensor, results of average along ``axis`` of ``x``, with the same data
            type as ``x``.
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([[[1., 2., 3., 4.],
                                       [5., 6., 7., 8.],
                                       [9., 10., 11., 12.]],
                                      [[13., 14., 15., 16.],
                                       [17., 18., 19., 20.],
                                       [21., 22., 23., 24.]]])
                out1 = paddle.mean(x)
                # [12.5]
                out2 = paddle.mean(x, axis: int = -1)
                # [[ 2.5  6.5 10.5]
                #  [14.5 18.5 22.5]]
                out3 = paddle.mean(x, axis: int = -1, keepdim=True)
                # [[[ 2.5]
                #   [ 6.5]
                #   [10.5]]
                #  [[14.5]
                #   [18.5]
                #   [22.5]]]
                out4 = paddle.mean(x, axis: int = [0, 2])
                # [ 8.5 12.5 16.5]        """
        pass


    def median(self, axis: Optional[int] = None, keepdim: bool = False, name: Optional[str] = None) -> Tensor:
        """
        Compute the median along the specified axis.
        
        Args:
            self (Tensor): The input Tensor, it's data type can be bool, float16, float32, float64, int32, int64.
            axis (int, optional): The axis along which to perform median calculations ``axis`` should be int.
                ``axis`` should be in range [-D, D), where D is the dimensions of ``x`` .
                If ``axis`` is less than 0, it works the same way as :math:`axis + D`.
                If ``axis`` is None, median is calculated over all elements of ``x``. Default is None.
            keepdim (bool, optional): Whether to reserve the reduced dimension(s)
                in the output Tensor. If ``keepdim`` is True, the dimensions of
                the output Tensor is the same as ``x`` except in the reduced
                dimensions(it is of size 1 in this case). Otherwise, the shape of
                the output Tensor is squeezed in ``axis`` . Default is False.
            name (str, optional): Name for the operation (optional, default is None).
                For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            Tensor, results of median along ``axis`` of ``x``. If data type of ``x`` is float64, data type of results will be float64, otherwise data type will be float32.
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.arange(12).reshape([3, 4])
                # x is [[0 , 1 , 2 , 3 ],
                #       [4 , 5 , 6 , 7 ],
                #       [8 , 9 , 10, 11]]
        
                y1 = paddle.median(x)
                # y1 is [5.5]
        
                y2 = paddle.median(x, axis: int = 0)
                # y2 is [4., 5., 6., 7.]
        
                y3 = paddle.median(x, axis: int = 1)
                # y3 is [1.5, 5.5, 9.5]
        
                y4 = paddle.median(x, axis: int = 0, keepdim=True)
                # y4 is [[4., 5., 6., 7.]]        """
        pass


    def min(self, axis: Optional[int] = None, keepdim: bool = False, name: Optional[str] = None) -> Tensor:
        """
        Computes the minimum of tensor elements over the given axis
        
        Note:
            The difference between min and amin is: If there are multiple minimum elements,
            amin evenly distributes gradient between these equal values, 
            while min propagates gradient to all of them.
        
        Args:
            self (Tensor): A tensor, the data type is float32, float64, int32, int64.
            axis(int|list|tuple, optional): The axis along which the minimum is computed.
                If :attr:`None`, compute the minimum over all elements of
                `x` and return a Tensor with a single element,
                otherwise must be in the range :math:`[-x.ndim, x.ndim)`.
                If :math:`axis[i] < 0`, the axis to reduce is :math:`x.ndim + axis[i]`.
            keepdim(bool, optional): Whether to reserve the reduced dimension in the
                output Tensor. The result tensor will have one fewer dimension
                than the `x` unless :attr:`keepdim` is true, default
                value is False.
            name(str, optional): The default value is None.  Normally there is no need for 
                user to set this property.  For more information, please refer to :ref:`api_guide_Name`
        
        Returns:
            Tensor, results of minimum on the specified axis of input tensor,
            it's data type is the same as input's Tensor.
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                # data_x is a Tensor with shape [2, 4]
                # the axis is a int element
                x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
                                      [0.1, 0.2, 0.6, 0.7]], 
                                     dtype='float64', stop_gradient=False)
                result1 = paddle.min(x)
                result1.backward()
                print(result1, x.grad) 
                #[0.1], [[0., 0., 0., 0.], [1., 0., 0., 0.]]
        
                x.clear_grad()
                result2 = paddle.min(x, axis: int = 0)
                result2.backward()
                print(result2, x.grad) 
                #[0.1, 0.2, 0.5, 0.7], [[0., 0., 1., 0.], [1., 1., 0., 1.]]
        
                x.clear_grad()
                result3 = paddle.min(x, axis: int = -1)
                result3.backward()
                print(result3, x.grad) 
                #[0.2, 0.1], [[1., 0., 0., 0.], [1., 0., 0., 0.]]
        
                x.clear_grad()
                result4 = paddle.min(x, axis: int = 1, keepdim=True)
                result4.backward()
                print(result4, x.grad) 
                #[[0.2], [0.1]], [[1., 0., 0., 0.], [1., 0., 0., 0.]]
        
                # data_y is a Tensor with shape [2, 2, 2]
                # the axis is list 
                y = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0]],
                                      [[5.0, 6.0], [7.0, 8.0]]],
                                     dtype='float64', stop_gradient=False)
                result5 = paddle.min(y, axis: int = [1, 2])
                result5.backward()
                print(result5, y.grad) 
                #[1., 5.], [[[1., 0.], [0., 0.]], [[1., 0.], [0., 0.]]]
        
                y.clear_grad()
                result6 = paddle.min(y, axis: int = [0, 1])
                result6.backward()
                print(result6, y.grad) 
                #[1., 2.], [[[1., 1.], [0., 0.]], [[0., 0.], [0., 0.]]]        """
        pass


    def minimum(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        """
        Compare two tensors and returns a new tensor containing the element-wise minima. The equation is:
        
        .. math::
            out = min(x, y)
        
        **Note**:
        ``paddle.minimum`` supports broadcasting. If you want know more about broadcasting, please refer to :ref:`user_guide_broadcasting` .
        
        Args:
            self (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
            y (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            N-D Tensor. A location into which the result is stored. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape,  its shape is the same as x and y.
        
        Examples:
        
            .. code-block:: python
        
                import numpy as np
                import paddle
        
                x = paddle.to_tensor([[1, 2], [7, 8]])
                y = paddle.to_tensor([[3, 4], [5, 6]])
                res = paddle.minimum(x, y)
                print(res)
                #       [[1, 2],
                #        [5, 6]]
        
                x = paddle.to_tensor([[[1, 2, 3], [1, 2, 3]]])
                y = paddle.to_tensor([3, 0, 4])
                res = paddle.minimum(x, y)
                print(res)
                #       [[[1, 0, 3],
                #         [1, 0, 3]]]
        
                x = paddle.to_tensor([2, 3, 5], dtype='float32')
                y = paddle.to_tensor([1, np.nan, np.nan], dtype='float32')
                res = paddle.minimum(x, y)
                print(res)
                #       [ 1., nan, nan]
        
                x = paddle.to_tensor([5, 3, np.inf], dtype='float64')
                y = paddle.to_tensor([1, -np.inf, 5], dtype='float64')
                res = paddle.minimum(x, y)
                print(res)
                #       [   1., -inf.,    5.]        """
        pass


    def mm(self, mat2, name: Optional[str] = None) -> Tensor:
        """
        Applies matrix multiplication to two tensors.
        
        Currently, the input tensors' rank can be any, but when the rank of any
        inputs is bigger than 3, this two inputs' rank should be equal.
        
        
        Also note that if the raw tensor :math:`x` or :math:`mat2` is rank-1 and
        nontransposed, the prepended or appended dimension :math:`1` will be
        removed after matrix multiplication.
        
        Args:
            input (Tensor): The input tensor which is a Tensor.
            mat2 (Tensor): The input tensor which is a Tensor.
            name(str, optional): The default value is None. Normally there is no need for
                user to set this property. For more information, please refer to :ref:`api_guide_Name`
        
        Returns:
            Tensor: The product Tensor.
        
        ::
        
            * example 1:
        
            input: [B, ..., M, K], mat2: [B, ..., K, N]
            out: [B, ..., M, N]
        
            * example 2:
        
            input: [B, M, K], mat2: [B, K, N]
            out: [B, M, N]
        
            * example 3:
        
            input: [B, M, K], mat2: [K, N]
            out: [B, M, N]
        
            * example 4:
        
            input: [M, K], mat2: [K, N]
            out: [M, N]
        
            * example 5:
        
            input: [B, M, K], mat2: [K]
            out: [B, M]
        
            * example 6:
        
            input: [K], mat2: [K]
            out: [1]
        
        Examples:
            .. code-block:: python
        
                import paddle
                input = paddle.arange(1, 7).reshape((3, 2)).astype('float32')
                mat2 = paddle.arange(1, 9).reshape((2, 4)).astype('float32')
                out = paddle.mm(input, mat2)
                print(out)
                #        [[11., 14., 17., 20.],
                #         [23., 30., 37., 44.],
                #         [35., 46., 57., 68.]])        """
        pass


    def mod(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        """
        Mod two tensors element-wise. The equation is:
        
        .. math::
        
            out = x \% y
        
        **Note**:
        ``paddle.remainder`` supports broadcasting. If you want know more about broadcasting, please refer to :ref:`user_guide_broadcasting` .
        
        Args:
            self (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
            y (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            N-D Tensor. A location into which the result is stored. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape,  its shape is the same as x and y.
        
        Examples:
        
            ..  code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([2, 3, 8, 7])
                y = paddle.to_tensor([1, 5, 3, 3])
                z = paddle.remainder(x, y)
                print(z)  # [0, 3, 2, 1]        """
        pass


    def mode(self, axis: int = -1, keepdim: bool = False, name: Optional[str] = None) -> Tuple[Tensor, Tensor]:
        """
        This OP is used to find values and indices of the modes at the optional axis.
        
        Args:
            self (Tensor): Tensor, an input N-D Tensor with type float32, float64, int32, int64.
            axis(int, optional): Axis to compute indices along. The effective range
                is [-R, R), where R is x.ndim. when axis < 0, it works the same way
                as axis + R. Default is -1.
            keepdim(bool, optional): Whether to keep the given axis in output. If it is True, the dimensions will be same as input x and with size one in the axis. Otherwise the output dimentions is one fewer than x since the axis is squeezed. Default is False.
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            tuple(Tensor), return the values and indices. The value data type is the same as the input `x`. The indices data type is int64.
        
        Examples:
        
            .. code-block:: python
        
               import paddle
               
               tensor = paddle.to_tensor([[[1,2,2],[2,3,3]],[[0,5,5],[9,9,0]]], dtype=paddle.float32)
               res = paddle.mode(tensor, 2)
               print(res)
               # (Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
               #   [[2., 3.],
               #    [5., 9.]]), Tensor(shape=[2, 2], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
               #   [[1, 1],
               #    [1, 0]]))
                       """
        pass


    def moveaxis(self, source, destination, name: Optional[str] = None) -> Tensor:
        """
        Move the axis of tensor from ``source`` position to ``destination`` position.
        
        Other axis that have not been moved remain their original order.
        
        Args:
            self (Tensor): The input Tensor. It is a N-D Tensor of data types bool, int32, int64, float32, float64, complex64, complex128.
            source(int|tuple|list): ``source`` position of axis that will be moved. Each element must be unique and integer.
            destination(int|tuple|list(int)): ``destination`` position of axis that has been moved. Each element must be unique and integer.
            name(str, optional): The default value is None.  Normally there is no need for user to set this
                property. For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            Tensor: A new tensor whose axis have been moved.
        
        Examples:
            .. code-block:: python
            
                import paddle
        
                x = paddle.ones([3, 2, 4])
                paddle.moveaxis(x, [0, 1], [1, 2]).shape
                # [4, 3, 2]
        
                x = paddle.ones([2, 3])
                paddle.moveaxis(x, 0, 1) # equivalent to paddle.t(x)
                # [3, 2]          """
        pass


    def multi_dot(self, name: Optional[str] = None) -> Tensor:
        """
        Multi_dot is an operator that calculates multiple matrix multiplications.
        
        Supports inputs of float16(only GPU support), float32 and float64 dtypes. This function does not
        support batched inputs.
        
        The input tensor in [x] must be 2-D except for the first and last can be 1-D.
        If the first tensor is a 1-D vector of shape(n, ) it is treated as row vector
        of shape(1, n), similarly if the last tensor is a 1D vector of shape(n, ), it
        is treated as a column vector of shape(n, 1).
        
        If the first and last tensor are 2-D matrix, then the output is also 2-D matrix,
        otherwise the output is a 1-D vector.
        
        Multi_dot will select the lowest cost multiplication order for calculation. The
        cost of multiplying two matrices with shapes (a, b) and (b, c) is a * b * c.
        Given matrices A, B, C with shapes (20, 5), (5, 100), (100, 10) respectively,
        we can calculate the cost of different multiplication orders as follows:
        - Cost((AB)C) = 20x5x100 + 20x100x10 = 30000
        - Cost(A(BC)) = 5x100x10 + 20x5x10 = 6000
        
        In this case, multiplying B and C first, then multiply A, which is 5 times faster
        than sequential calculation.
        
        Args:
            x ([Tensor]): The input tensors which is a list Tensor.
            name(str|None): A name for this layer(optional). If set None, the layer
                will be named automatically.
        
        Returns:
            Tensor: The output Tensor.
        
        
        Examples:
        
        .. code-block:: python
        
            import paddle
            import numpy as np
        
            # A * B
            A_data = np.random.random([3, 4]).astype(np.float32)
            B_data = np.random.random([4, 5]).astype(np.float32)
            A = paddle.to_tensor(A_data)
            B = paddle.to_tensor(B_data)
            out = paddle.linalg.multi_dot([A, B])
            print(out.numpy().shape)
            # [3, 5]
        
            # A * B * C
            A_data = np.random.random([10, 5]).astype(np.float32)
            B_data = np.random.random([5, 8]).astype(np.float32)
            C_data = np.random.random([8, 7]).astype(np.float32)
            A = paddle.to_tensor(A_data)
            B = paddle.to_tensor(B_data)
            C = paddle.to_tensor(C_data)
            out = paddle.linalg.multi_dot([A, B, C])
            print(out.numpy().shape)
            # [10, 7]        """
        pass


    def multiplex(self, index, name: Optional[str] = None) -> Tensor:
        """
        Based on the given index parameter, the OP selects a specific row from each input Tensor to construct the output Tensor.
        
        If the input of this OP contains :math:`m` Tensors, where :math:`I_{i}` means the i-th input Tensor, :math:`i` between :math:`[0,m)` .
        
        And :math:`O` means the output, where :math:`O[i]` means the i-th row of the output, then the output satisfies that :math:`O[i] = I_{index[i]}[i]` .
        
        For Example:
        
                .. code-block:: text
        
                    Given:
        
                    inputs = [[[0,0,3,4], [0,1,3,4], [0,2,4,4], [0,3,3,4]],
                              [[1,0,3,4], [1,1,7,8], [1,2,4,2], [1,3,3,4]],
                              [[2,0,3,4], [2,1,7,8], [2,2,4,2], [2,3,3,4]],
                              [[3,0,3,4], [3,1,7,8], [3,2,4,2], [3,3,3,4]]]
        
                    index = [[3],[0],[1],[2]]
        
                    out = [[3,0,3,4],    # out[0] = inputs[index[0]][0] = inputs[3][0] = [3,0,3,4]
                           [0,1,3,4],    # out[1] = inputs[index[1]][1] = inputs[0][1] = [0,1,3,4]
                           [1,2,4,2],    # out[2] = inputs[index[2]][2] = inputs[1][2] = [1,2,4,2]
                           [2,3,3,4]]    # out[3] = inputs[index[3]][3] = inputs[2][3] = [2,3,3,4]
        
        
        Args:
            inputs (list): The input Tensor list. The list elements are N-D Tensors of data types float32, float64, int32, int64. All input Tensor shapes should be the same and rank must be at least 2.
            indeself (Tensor): Used to select some rows in the input Tensor to construct an index of the output Tensor. It is a 2-D Tensor with data type int32 or int64 and shape [M, 1], where M is the number of input Tensors.
            name(str, optional): The default value is None. Normally there is no
                need for user to set this property. For more information, please
                refer to :ref:`api_guide_Name`.
        Returns:
            Tensor: Output of multiplex OP, with data type being float32, float64, int32, int64.
        
        Examples:
        
            .. code-block:: python
        
                import paddle
                import numpy as np
                img1 = np.array([[1, 2], [3, 4]]).astype(np.float32)
                img2 = np.array([[5, 6], [7, 8]]).astype(np.float32)
                inputs = [paddle.to_tensor(img1), paddle.to_tensor(img2)]
                index = paddle.to_tensor(np.array([[1], [0]]).astype(np.int32))
                res = paddle.multiplex(inputs, index)
                print(res) # [array([[5., 6.], [3., 4.]], dtype=float32)]        """
        pass


    def multiply(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        """
        Elementwise Mul Operator.
        
        Multiply two tensors element-wise
        
        The equation is:
        
        :math:`Out = X \\odot Y`
        
        - $X$: a tensor of any dimension.
        - $Y$: a tensor whose dimensions must be less than or equal to the dimensions of $X$.
        
        There are two cases for this operator:
        
        1. The shape of $Y$ is the same with $X$.
        2. The shape of $Y$ is a continuous subsequence of $X$.
        
        For case 2:
        
        1. Broadcast $Y$ to match the shape of $X$, where $axis$ is the start dimension index
           for broadcasting $Y$ onto $X$.
        2. If $axis$ is -1 (default), $axis = rank(X) - rank(Y)$.
        3. The trailing dimensions of size 1 for $Y$ will be ignored for the consideration of
           subsequence, such as shape(Y) = (2, 1) => (2).
        
        For example:
        
          .. code-block:: text
        
            shape(X) = (2, 3, 4, 5), shape(Y) = (,)
            shape(X) = (2, 3, 4, 5), shape(Y) = (5,)
            shape(X) = (2, 3, 4, 5), shape(Y) = (4, 5), with axis: int = -1(default) or axis: int = 2
            shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis: int = 1
            shape(X) = (2, 3, 4, 5), shape(Y) = (2), with axis: int = 0
            shape(X) = (2, 3, 4, 5), shape(Y) = (2, 1), with axis: int = 0
        
        
        Args:
            self (Tensor): (Variable), Tensor or LoDTensor of any dimensions. Its dtype should be int32, int64, float32, float64.
            y (Tensor): (Variable), Tensor or LoDTensor of any dimensions. Its dtype should be int32, int64, float32, float64.
            with_quant_attr (BOOLEAN): Whether the operator has attributes used by quantization. 
            name (string, optional): Name of the output.         Default is None. It's used to print debug info for developers. Details:         :ref:`api_guide_Name`
        
        Returns:
            out (Tensor): N-dimension tensor. A location into which the result is stored. It's dimension equals with x
        
            multiply two tensors element-wise. The equation is:
        
            .. math::
                out = x * y
        
            **Note**:
            ``paddle.multiply`` supports broadcasting. If you would like to know more about broadcasting, please refer to :ref:`user_guide_broadcasting` .
        
            Args:
                self (Tensor): the input tensor, its data type should be one of float32, float64, int32, int64, bool.
                y (Tensor): the input tensor, its data type should be one of float32, float64, int32, int64, bool.
                name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
            Returns:
                N-D Tensor. A location into which the result is stored. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape,  its shape is the same as x and y.
        
            Examples:
        
                ..  code-block:: python
        
                    import paddle
        
                    x = paddle.to_tensor([[1, 2], [3, 4]])
                    y = paddle.to_tensor([[5, 6], [7, 8]])
                    res = paddle.multiply(x, y)
                    print(res) # [[5, 12], [21, 32]]
        
                    x = paddle.to_tensor([[[1, 2, 3], [1, 2, 3]]])
                    y = paddle.to_tensor([2])
                    res = paddle.multiply(x, y)
                    print(res) # [[[2, 4, 6], [2, 4, 6]]]
        
                    """
        pass


    def mv(self, vec, name: Optional[str] = None) -> Tensor:
        """
        Performs a matrix-vector product of the matrix x and the vector vec.
        
        Args:
            self (Tensor): A tensor with shape :math:`[M, N]` , The data type of the input Tensor x
                should be one of float32, float64.
            vec (Tensor): A tensor with shape :math:`[N]` , The data type of the input Tensor x
                should be one of float32, float64.
            name(str, optional): The default value is None.  Normally there is no need for user to set this
                property.  For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            Tensor: The tensor which is producted by x and vec.
        
        Examples:
            .. code-block:: python
        
                # x: [M, N], vec: [N]
                # paddle.mv(x, vec)  # out: [M]
        
                import numpy as np
                import paddle
        
                x_data = np.array([[2, 1, 3], [3, 0, 1]]).astype("float64")
                x = paddle.to_tensor(x_data)
                vec_data = np.array([3, 5, 1])
                vec = paddle.to_tensor(vec_data).astype("float64")
                out = paddle.mv(x, vec)        """
        pass


    def nanmean(self, axis: Optional[int] = None, keepdim: bool = False, name: Optional[str] = None) -> Tensor:
        """
        Compute the arithmetic mean along the specified axis, ignoring NaNs.
        
        Args:
            self (Tensor): The input Tensor with data type uint16, float16, float32, float64.
            axis (int|list|tuple, optional):The axis along which to perform nanmean
                calculations. ``axis`` should be int, list(int) or tuple(int). If
                ``axis`` is a list/tuple of dimension(s), nanmean is calculated along
                all element(s) of ``axis`` . ``axis`` or element(s) of ``axis``
                should be in range [-D, D), where D is the dimensions of ``x`` . If
                ``axis`` or element(s) of ``axis`` is less than 0, it works the
                same way as :math:`axis + D` . If ``axis`` is None, nanmean is
                calculated over all elements of ``x``. Default is None.
            keepdim (bool, optional): Whether to reserve the reduced dimension(s)
                in the output Tensor. If ``keepdim`` is True, the dimensions of
                the output Tensor is the same as ``x`` except in the reduced
                dimensions(it is of size 1 in this case). Otherwise, the shape of
                the output Tensor is squeezed in ``axis`` . Default is False.
            name (str, optional): Name for the operation (optional, default is None).
                For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            Tensor, results of arithmetic mean along ``axis`` of ``x``, with the same data
            type as ``x``.
        
        Examples:
        
            .. code-block:: python
                :name: code-example1
        
                import paddle
                # x is a 2-D Tensor:
                x = paddle.to_tensor([[float('nan'), 0.3, 0.5, 0.9],
                                      [0.1, 0.2, float('-nan'), 0.7]])
                out1 = paddle.nanmean(x)
                # [0.44999996]
                out2 = paddle.nanmean(x, axis: int = 0)
                # [0.1, 0.25, 0.5, 0.79999995]
                out3 = paddle.nanmean(x, axis: int = 0, keepdim=True)
                # [[0.1, 0.25, 0.5, 0.79999995]]
                out4 = paddle.nanmean(x, axis: int = 1)
                # [0.56666666 0.33333334]
                out5 = paddle.nanmean(x, axis: int = 1, keepdim=True)
                # [[0.56666666]
                #  [0.33333334]]
        
                # y is a 3-D Tensor:
                y = paddle.to_tensor([[[1, float('nan')], [3, 4]],
                                       [[5, 6], [float('-nan'), 8]]])
                out6 = paddle.nanmean(y, axis: int = [1, 2])
                # [2.66666675, 6.33333349]
                out7 = paddle.nanmean(y, axis: int = [0, 1])
                # [3., 6.]        """
        pass


    def nansum(self, axis: Optional[int] = None, dtype=None, keepdim: bool = False, name: Optional[str] = None) -> Tensor:
        """
        Computes the sum of tensor elements over the given axis, treating Not a Numbers (NaNs) as zero.
        
        Args:
            self (Tensor): An N-D Tensor, the data type is float32, float64, int32 or int64.
            axis (int|list|tuple, optional): The dimensions along which the nansum is performed. If
                :attr:`None`, nansum all elements of :attr:`x` and return a
                Tensor with a single element, otherwise must be in the
                range :math:`[-rank(x), rank(x))`. If :math:`axis[i] < 0`,
                the dimension to reduce is :math:`rank + axis[i]`.
            dtype (str, optional): The dtype of output Tensor. The default value is None, the dtype
                of output is the same as input Tensor `x`.
            keepdim (bool, optional): Whether to reserve the reduced dimension in the
                output Tensor. The result Tensor will have one fewer dimension
                than the :attr:`x` unless :attr:`keepdim` is true, default
                value is False.
            name (str, optional): The default value is None. Normally there is no need for
                user to set this property.  For more information, please refer to :ref:`api_guide_Name`
        
        Returns:
            Tensor: Results of summation operation on the specified axis of input Tensor `x`,
        
        Examples:
            .. code-block:: python
        
                import paddle
                import numpy as np
        
                # x is a Tensor with following elements:
                #    [[nan, 0.3, 0.5, 0.9]
                #     [0.1, 0.2, -nan, 0.7]]
                # Each example is followed by the corresponding output tensor.
                x = np.array([[float('nan'), 0.3, 0.5, 0.9],
                                [0.1, 0.2, float('-nan'), 0.7]]).astype(np.float32)
                x = paddle.to_tensor(x)
                out1 = paddle.nansum(x)  # [2.7]
                out2 = paddle.nansum(x, axis: int = 0)  # [0.1, 0.5, 0.5, 1.6]
                out3 = paddle.nansum(x, axis: int = -1)  # [1.7, 1.0]
                out4 = paddle.nansum(x, axis: int = 1, keepdim=True)  # [[1.7], [1.0]]
        
                # y is a Tensor with shape [2, 2, 2] and elements as below:
                #      [[[1, nan], [3, 4]],
                #      [[5, 6], [-nan, 8]]]
                # Each example is followed by the corresponding output tensor.
                y = np.array([[[1, float('nan')], [3, 4]], 
                                [[5, 6], [float('-nan'), 8]]])
                y = paddle.to_tensor(y)
                out5 = paddle.nansum(y, axis: int = [1, 2]) # [8, 19]
                out6 = paddle.nansum(y, axis: int = [0, 1]) # [9, 18]        """
        pass


    def ndimension(self) -> int:
        pass
    def neg(self, name: Optional[str] = None) -> Tensor:
        """
        This function computes the negative of the Tensor elementwisely.
        
        Args:
            self (Tensor): Input of neg operator, an N-D Tensor, with data type float32, float64, int8, int16, int32, or int64.
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            out (Tensor): The negative of input Tensor. The shape and data type are the same with input Tensor.
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
                out = paddle.neg(x)
                print(out)
                # [0.4 0.2 -0.1 -0.3]        """
        pass


    def nonzero(self, as_tuple=False) -> Tensor:
        """
        Return a tensor containing the indices of all non-zero elements of the `input` 
        tensor. If as_tuple is True, return a tuple of 1-D tensors, one for each dimension 
        in `input`, each containing the indices (in that dimension) of all non-zero elements 
        of `input`. Given a n-Dimensional `input` tensor with shape [x_1, x_2, ..., x_n], If 
        as_tuple is False, we can get a output tensor with shape [z, n], where `z` is the 
        number of all non-zero elements in the `input` tensor. If as_tuple is True, we can get 
        a 1-D tensor tuple of length `n`, and the shape of each 1-D tensor is [z, 1].
        
        Args:
            self (Tensor): The input tensor variable.
            as_tuple (bool): Return type, Tensor or tuple of Tensor.
        
        Returns:
            Tensor. The data type is int64.
        
        Examples:
        
            .. code-block:: python
        
                import paddle
        
                x1 = paddle.to_tensor([[1.0, 0.0, 0.0],
                                       [0.0, 2.0, 0.0],
                                       [0.0, 0.0, 3.0]])
                x2 = paddle.to_tensor([0.0, 1.0, 0.0, 3.0])
                out_z1 = paddle.nonzero(x1)
                print(out_z1)
                #[[0 0]
                # [1 1]
                # [2 2]]
                out_z1_tuple = paddle.nonzero(x1, as_tuple=True)
                for out in out_z1_tuple:
                    print(out)
                #[[0]
                # [1]
                # [2]]
                #[[0]
                # [1]
                # [2]]
                out_z2 = paddle.nonzero(x2)
                print(out_z2)
                #[[1]
                # [3]]
                out_z2_tuple = paddle.nonzero(x2, as_tuple=True)
                for out in out_z2_tuple:
                    print(out)
                #[[1]
                # [3]]        """
        pass


    def norm(self, p='fro', axis: Optional[int] = None, keepdim: bool = False, name: Optional[str] = None) -> Tuple[Tensor, Tensor]:
        """
        Returns the matrix norm (Frobenius) or vector norm (the 1-norm, the Euclidean
        or 2-norm, and in general the p-norm for p > 0) of a given tensor.
        
        .. note::
            This norm API is different from `numpy.linalg.norm`.
            This api supports high-order input tensors (rank >= 3), and certain axis need to be pointed out to calculate the norm.
            But `numpy.linalg.norm` only supports 1-D vector or 2-D matrix as input tensor.
            For p-order matrix norm, this api actually treats matrix as a flattened vector to calculate the vector norm, NOT REAL MATRIX NORM.
        
        Args:
            self (Tensor): The input tensor could be N-D tensor, and the input data
                type could be float32 or float64.
            p (float|string, optional): Order of the norm. Supported values are `fro`, `0`, `1`, `2`,
                `inf`, `-inf` and any positive real number yielding the corresponding p-norm. Not supported: ord < 0 and nuclear norm.
                Default value is `fro`.
            axis (int|list|tuple, optional): The axis on which to apply norm operation. If axis is int
                or list(int)/tuple(int)  with only one element, the vector norm is computed over the axis.
                If `axis < 0`, the dimension to norm operation is rank(input) + axis.
                If axis is a list(int)/tuple(int) with two elements, the matrix norm is computed over the axis.
                Defalut value is `None`.
            keepdim (bool, optional): Whether to reserve the reduced dimension in the
                output Tensor. The result tensor will have fewer dimension
                than the :attr:`input` unless :attr:`keepdim` is true, default
                value is False.
            name (str, optional): The default value is None. Normally there is no need for
                user to set this property. For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            Tensor: results of norm operation on the specified axis of input tensor,
            it's data type is the same as input's Tensor.
        
        Examples:
            .. code-block:: python
        
                import paddle
                import numpy as np
                shape=[2, 3, 4]
                np_input = np.arange(24).astype('float32') - 12
                np_input = np_input.reshape(shape)
                x = paddle.to_tensor(np_input)
                #[[[-12. -11. -10.  -9.] [ -8.  -7.  -6.  -5.] [ -4.  -3.  -2.  -1.]]
                # [[  0.   1.   2.   3.] [  4.   5.   6.   7.] [  8.   9.  10.  11.]]]
        
                # compute frobenius norm along last two dimensions.
                out_fro = paddle.linalg.norm(x, p='fro', axis: int = [0,1])
                # out_fro.numpy() [17.435596 16.911535 16.7332   16.911535]
        
                # compute 2-order vector norm along last dimension.
                out_pnorm = paddle.linalg.norm(x, p=2, axis: int = -1)
                #out_pnorm.numpy(): [[21.118711  13.190906   5.477226]
                #                    [ 3.7416575 11.224972  19.131126]]
        
                # compute 2-order  norm along [0,1] dimension.
                out_pnorm = paddle.linalg.norm(x, p=2, axis: int = [0,1])
                #out_pnorm.numpy(): [17.435596 16.911535 16.7332   16.911535]
        
                # compute inf-order  norm
                out_pnorm = paddle.linalg.norm(x, p=np.inf)
                #out_pnorm.numpy()  = [12.]
                out_pnorm = paddle.linalg.norm(x, p=np.inf, axis: int = 0)
                #out_pnorm.numpy(): [[12. 11. 10. 9.] [8. 7. 6. 7.] [8. 9. 10. 11.]]
        
                # compute -inf-order  norm
                out_pnorm = paddle.linalg.norm(x, p=-np.inf)
                #out_pnorm.numpy(): [0.]
                out_pnorm = paddle.linalg.norm(x, p=-np.inf, axis: int = 0)
                #out_pnorm.numpy(): [[0. 1. 2. 3.] [4. 5. 6. 5.] [4. 3. 2. 1.]]        """
        pass


    def not_equal(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        """
        This OP returns the truth value of :math:`x != y` elementwise, which is equivalent function to the overloaded operator `!=`.
        
        **NOTICE**: The output of this OP has no gradient.
        
        Args:
            self (Tensor): First input to compare which is N-D tensor. The input data type should be bool, float32, float64, int32, int64.
            y(Tensor): Second input to compare which is N-D tensor. The input data type should be bool, float32, float64, int32, int64.
            name(str, optional): The default value is None.  Normally there is no need for
                user to set this property.  For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            Tensor, the output data type is bool: The tensor storing the output, the output shape is same as input :attr:`x`.
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([1, 2, 3])
                y = paddle.to_tensor([1, 3, 2])
                result1 = paddle.not_equal(x, y)
                print(result1)  # result1 = [False True True]        """
        pass


    def numel(self, name: Optional[str] = None) -> int:
        """
        Returns the number of elements for a tensor, which is a int64 Tensor with shape [1] in static mode
        or a scalar value in imperative mode
        
        Args:
            self (Tensor): The input Tensor, it's data type can be bool, float16, float32, float64, int32, int64.
        
        Returns:
            Tensor: The number of elements for the input Tensor.
        
        Examples:
            .. code-block:: python
        
                import paddle
                
                x = paddle.full(shape=[4, 5, 7], fill_value=0, dtype='int32')
                numel = paddle.numel(x) # 140        """
        pass


    def outer(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        """
        Outer product of two Tensors.
        
        Input is flattened if not already 1-dimensional.
        
        Args:
            self (Tensor): An N-D Tensor or a Scalar Tensor. 
            y (Tensor): An N-D Tensor or a Scalar Tensor. 
            name(str, optional): The default value is None. Normally there is no need for
                user to set this property. For more information, please refer to :ref:`api_guide_Name`
        
        Returns:
            Tensor: The outer-product Tensor.
        
        Examples:
            .. code-block:: python
        
                import paddle
                x = paddle.arange(1, 4).astype('float32')
                y = paddle.arange(1, 6).astype('float32')
                out = paddle.outer(x, y)
                print(out)
                #        ([[1, 2, 3, 4, 5],
                #         [2, 4, 6, 8, 10],
                #         [3, 6, 9, 12, 15]])        """
        pass


    def pow(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        """
        Compute the power of tensor elements. The equation is:
        
        .. math::
            out = x^{y} 
        
        **Note**:
        ``paddle.pow`` supports broadcasting. If you want know more about broadcasting, please refer to :ref:`user_guide_broadcasting` .
        
        
        Args:
            self (Tensor): An N-D Tensor, the data type is float32, float64, int32 or int64.
            y (float|int|Tensor): If it is an N-D Tensor, its data type should be the same as `x`.
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            N-D Tensor. A location into which the result is stored. Its dimension and data type are the same as `x`.
        
        Examples:
        
            ..  code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([1, 2, 3], dtype='float32')
        
                # example 1: y is a float or int
                res = paddle.pow(x, 2)
                print(res)
                # Tensor(shape=[3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
                #        [1., 4., 9.])
                res = paddle.pow(x, 2.5)
                print(res)
                # Tensor(shape=[3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
                #        [1.         , 5.65685415 , 15.58845711])
        
                # example 2: y is a Tensor
                y = paddle.to_tensor([2], dtype='float32')
                res = paddle.pow(x, y)
                print(res)
                # Tensor(shape=[3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
                #        [1., 4., 9.])        """
        pass


    def prod(self, axis: Optional[int] = None, keepdim: bool = False, dtype=None, name: Optional[str] = None) -> Tensor:
        """
        Compute the product of tensor elements over the given axis.
        
        Args:
            self (Tensor): The input tensor, its data type should be float32, float64, int32, int64.
            axis(int|list|tuple, optional): The axis along which the product is computed. If :attr:`None`, 
                multiply all elements of `x` and return a Tensor with a single element, 
                otherwise must be in the range :math:`[-x.ndim, x.ndim)`. If :math:`axis[i]<0`, 
                the axis to reduce is :math:`x.ndim + axis[i]`. Default is None.
            dtype(str|np.dtype, optional): The desired date type of returned tensor, can be float32, float64, 
                int32, int64. If specified, the input tensor is casted to dtype before operator performed. 
                This is very useful for avoiding data type overflows. The default value is None, the dtype 
                of output is the same as input Tensor `x`.
            keepdim(bool, optional): Whether to reserve the reduced dimension in the output Tensor. The result 
                tensor will have one fewer dimension than the input unless `keepdim` is true. Default is False.
            name(string, optional): The default value is None. Normally there is no need for user to set this property.
                For more information, please refer to :ref:`api_guide_Name` .
        
        Returns:
            Tensor, result of product on the specified dim of input tensor.
        
        Raises:
            ValueError: The :attr:`dtype` must be float32, float64, int32 or int64.
            TypeError: The type of :attr:`axis` must be int, list or tuple.
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                # the axis is a int element
                x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
                                      [0.1, 0.2, 0.6, 0.7]])
                out1 = paddle.prod(x)
                # [0.0002268]
        
                out2 = paddle.prod(x, -1)
                # [0.027  0.0084]
        
                out3 = paddle.prod(x, 0)
                # [0.02 0.06 0.3  0.63]
        
                out4 = paddle.prod(x, 0, keepdim=True)
                # [[0.02 0.06 0.3  0.63]]
        
                out5 = paddle.prod(x, 0, dtype='int64')
                # [0 0 0 0]
        
                # the axis is list
                y = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0]],
                                      [[5.0, 6.0], [7.0, 8.0]]])
                out6 = paddle.prod(y, [0, 1])
                # [105. 384.]
        
                out7 = paddle.prod(y, (1, 2))
                # [  24. 1680.]        """
        pass


    def put_along_axis(self, indices, values, axis, reduce='assign') -> Tensor:
        """
        Put values into the destination array by given indices matrix along the designated axis.
        
        Args:
            arr (Tensor) : The Destination Tensor. Supported data types are float32 and float64.
            indices (Tensor) : Indices to put along each 1d slice of arr. This must match the dimension of arr,
                and need to broadcast against arr. Supported data type are int and int64.
            axis (int) : The axis to put 1d slices along. 
            reduce (string | optinal) : The reduce operation, default is 'assign', support 'add', 'assign', 'mul' and 'multiply'.
        Returns : 
            Tensor: The indexed element, same dtype with arr
        
        Examples:
            .. code-block:: python
        
                import paddle
                import numpy as np
        
                x_np = np.array([[10, 30, 20], [60, 40, 50]])
                index_np = np.array([[0]])
                x = paddle.to_tensor(x_np)
                index = paddle.to_tensor(index_np)
                value = 99
                axis = 0
                result = paddle.put_along_axis(x, index, value, axis)
                print(result)
                # [[99, 99, 99],
                # [60, 40, 50]]        """
        pass


    def put_along_axis_(self, indices, values, axis, reduce='assign') -> None:
        """
        Inplace version of ``put_along_axis`` API, the output Tensor will be inplaced with input ``arr``.
        Please refer to :ref:`api_tensor_put_along_axis`.        """
        pass


    def qr(self, mode='reduced', name: Optional[str] = None) -> Tuple[Tensor, Tensor]:
        """
        Computes the QR decomposition of one matrix or batches of matrice (backward is unsupported now).
        
        Args:
            self (Tensor): The input tensor. Its shape should be `[..., M, N]`,
                where ... is zero or more batch dimensions. M and N can be arbitrary
                positive number. The data type of x should be float32 or float64. 
            mode (str, optional): A flag to control the behavior of qr, the default is "reduced". 
                Suppose x's shape is `[..., M, N]` and denoting `K = min(M, N)`:
                If mode = "reduced", qr op will return reduced Q and R matrices, 
                which means Q's shape is `[..., M, K]` and R's shape is `[..., K, N]`.
                If mode = "complete", qr op will return complete Q and R matrices, 
                which means Q's shape is `[..., M, M]` and R's shape is `[..., M, N]`.
                If mode = "r", qr op will only return reduced R matrix, which means
                R's shape is `[..., K, N]`.
            name (str, optional): Name for the operation (optional, default is None).
                For more information, please refer to :ref:`api_guide_Name`.
                
        Returns:
            If mode = "reduced" or mode = "complete", qr will return a two tensor-tuple, which represents Q and R. 
            If mode = "r", qr will return a tensor which represents R.
            
        Examples:            
            .. code-block:: python
        
                import paddle 
        
                x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).astype('float64')
                q, r = paddle.linalg.qr(x)
                print (q)
                print (r)
        
                # Q = [[-0.16903085,  0.89708523],
                #      [-0.50709255,  0.27602622],
                #      [-0.84515425, -0.34503278]])
        
                # R = [[-5.91607978, -7.43735744],
                #      [ 0.        ,  0.82807867]])
                
                # one can verify : X = Q * R ;             """
        pass


    def quantile(self, q, axis: Optional[int] = None, keepdim: bool = False) -> Tensor:
        """
        Compute the quantile of the input along the specified axis.
        
        Args:
            self (Tensor): The input Tensor, it's data type can be float32, float64.
            q (int|float|list): The q for calculate quantile, which should be in range [0, 1]. If q is a list, 
                each q will be calculated and the first dimension of output is same to the number of ``q`` .
            axis (int|list, optional): The axis along which to calculate quantile. ``axis`` should be int or list of int.
                ``axis`` should be in range [-D, D), where D is the dimensions of ``x`` .
                If ``axis`` is less than 0, it works the same way as :math:`axis + D`.
                If ``axis`` is a list, quantile is calculated over all elements of given axises.
                If ``axis`` is None, quantile is calculated over all elements of ``x``. Default is None.
            keepdim (bool, optional): Whether to reserve the reduced dimension(s)
                in the output Tensor. If ``keepdim`` is True, the dimensions of
                the output Tensor is the same as ``x`` except in the reduced
                dimensions(it is of size 1 in this case). Otherwise, the shape of
                the output Tensor is squeezed in ``axis`` . Default is False.
            name (str, optional): Name for the operation (optional, default is None).
                For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            Tensor, results of quantile along ``axis`` of ``x``. If data type of ``x`` is float64, data type of results will be float64, otherwise data type will be float32.
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.randn((2,3))
                #[[-1.28740597,  0.49533170, -1.00698614],
                # [-1.11656201, -1.01010525, -2.23457789]])
        
                y1 = paddle.quantile(x, q=0.5, axis: int = [0, 1])
                # y1 = -1.06333363
        
                y2 = paddle.quantile(x, q=0.5, axis: int = 1)
                # y2 = [-1.00698614, -1.11656201]
        
                y3 = paddle.quantile(x, q=[0.3, 0.5], axis: int = 1)
                # y3 =[[-1.11915410, -1.56376839],
                #      [-1.00698614, -1.11656201]]
        
                y4 = paddle.quantile(x, q=0.8, axis: int = 1, keepdim=True)
                # y4 = [[-0.10559537],
                #       [-1.05268800]])        """
        pass


    def rad2deg(self, name: Optional[str] = None) -> Tensor:
        """
        Convert each of the elements of input x from angles in radians to degrees.
        
        Equation:
            .. math::
        
                rad2deg(x)=180/ \pi * x
        
        Args:
            self (Tensor): An N-D Tensor, the data type is float32, float64, int32, int64.
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            out (Tensor): An N-D Tensor, the shape and data type is the same with input (The output data type is float32 when the input data type is int).
        
        Examples:
            .. code-block:: python
        
                import paddle
                import numpy as np
                
                x1 = paddle.to_tensor([3.142, -3.142, 6.283, -6.283, 1.570, -1.570])
                result1 = paddle.rad2deg(x1)
                print(result1)
                # Tensor(shape=[6], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
                #         [180.02334595, -180.02334595,  359.98937988, -359.98937988,
                #           9.95437622 , -89.95437622])
        
                x2 = paddle.to_tensor(np.pi/2)
                result2 = paddle.rad2deg(x2)
                print(result2)
                # Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
                #         [90.])
                         
                x3 = paddle.to_tensor(1)
                result3 = paddle.rad2deg(x3)
                print(result3)
                # Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
                #         [57.29578018])        """
        pass


    def rank(self) -> Tensor:
        pass
    def real(self, name: Optional[str] = None) -> Tensor:
        pass
    def reciprocal(self, name: Optional[str] = None) -> Tensor:
        """
        Reciprocal Activation Operator.
        
        :math:`out = \\frac{1}{x}`
        
        
        Args:
            self (Tensor): Input of Reciprocal operator, an N-D Tensor, with data type float32, float64 or float16.
            with_quant_attr (BOOLEAN): Whether the operator has attributes used by quantization. 
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            out (Tensor): Output of Reciprocal operator, a Tensor with shape same as input.
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
                out = paddle.reciprocal(x)
                print(out)
                # [-2.5        -5.         10.          3.33333333]        """
        pass


    def reciprocal_(self, name: Optional[str] = None) -> None:
        """
        Inplace version of ``reciprocal`` API, the output Tensor will be inplaced with input ``x``.
        Please refer to :ref:`api_fluid_layers_reciprocal`.        """
        pass


    def register_hook(self, hook) -> None:
        pass
    def remainder(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        """
        Mod two tensors element-wise. The equation is:
        
        .. math::
        
            out = x \% y
        
        **Note**:
        ``paddle.remainder`` supports broadcasting. If you want know more about broadcasting, please refer to :ref:`user_guide_broadcasting` .
        
        Args:
            self (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
            y (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            N-D Tensor. A location into which the result is stored. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape,  its shape is the same as x and y.
        
        Examples:
        
            ..  code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([2, 3, 8, 7])
                y = paddle.to_tensor([1, 5, 3, 3])
                z = paddle.remainder(x, y)
                print(z)  # [0, 3, 2, 1]        """
        pass


    def repeat_interleave(self, repeats, axis: Optional[int] = None, name: Optional[str] = None) -> Tensor:
        """
        Returns a new tensor which repeats the ``x`` tensor along dimension ``axis`` using
        the entries in ``repeats`` which is a int or a Tensor.
        
        Args:
            self (Tensor): The input Tensor to be operated. The data of ``x`` can be one of float32, float64, int32, int64.
            repeats (Tensor or int): The number of repetitions for each element. repeats is broadcasted to fit the shape of the given axis.
            axis (int, optional): The dimension in which we manipulate. Default: None, the output tensor is flatten.
            name(str, optional): The default value is None. Normally there is no
                need for user to set this property. For more information, please
                refer to :ref:`api_guide_Name`.
        
        Returns:
            Tensor: A Tensor with same data type as ``x``.
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([[1, 2, 3], [4, 5, 6]])
                repeats  = paddle.to_tensor([3, 2, 1], dtype='int32')
        
                paddle.repeat_interleave(x, repeats, 1)
                # [[1, 1, 1, 2, 2, 3],
                #  [4, 4, 4, 5, 5, 6]]
        
                paddle.repeat_interleave(x, 2, 0)
                # [[1, 2, 3], [1, 2, 3], [4, 5, 6], [4, 5, 6]]
        
                paddle.repeat_interleave(x, 2, None)
                # [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]        """
        pass


    def reshape(self, shape, name: Optional[str] = None) -> Tuple[Tensor, Tensor]:
        """
        This operator changes the shape of ``x`` without changing its data.
        
        Note that the output Tensor will share data with origin Tensor and doesn't
        have a Tensor copy in ``dygraph`` mode. 
        If you want to use the Tensor copy version, please use `Tensor.clone` like 
        ``reshape_clone_x = x.reshape([-1]).clone()``.
        
        Some tricks exist when specifying the target shape.
        
        1. -1 means the value of this dimension is inferred from the total element
        number of x and remaining dimensions. Thus one and only one dimension can
        be set -1.
        
        2. 0 means the actual dimension value is going to be copied from the
        corresponding dimension of x. The index of 0s in shape can not exceed
        the dimension of x.
        
        Here are some examples to explain it.
        
        1. Given a 3-D tensor x with a shape [2, 4, 6], and the target shape
        is [6, 8], the reshape operator will transform x into a 2-D tensor with
        shape [6, 8] and leaving x's data unchanged.
        
        2. Given a 3-D tensor x with a shape [2, 4, 6], and the target shape
        specified is [2, 3, -1, 2], the reshape operator will transform x into a
        4-D tensor with shape [2, 3, 4, 2] and leaving x's data unchanged. In this
        case, one dimension of the target shape is set to -1, the value of this
        dimension is inferred from the total element number of x and remaining
        dimensions.
        
        3. Given a 3-D tensor x with a shape [2, 4, 6], and the target shape
        is [-1, 0, 3, 2], the reshape operator will transform x into a 4-D tensor
        with shape [2, 4, 3, 2] and leaving x's data unchanged. In this case,
        besides -1, 0 means the actual dimension value is going to be copied from
        the corresponding dimension of x.
        
        Args:
            self (Tensor): An N-D Tensor. The data type is ``float32``, ``float64``, ``int32``, ``int64`` or ``bool``
            shape(list|tuple|Tensor): Define the target shape. At most one dimension of the target shape can be -1.
                            The data type is ``int32`` . If ``shape`` is a list or tuple, the elements of it should be integers or Tensors with shape [1].
                            If ``shape`` is an Tensor, it should be an 1-D Tensor .
            name(str, optional): The default value is None. Normally there is no need for user to set this property.
                                For more information, please refer to :ref:`api_guide_Name` .
        
        Returns:
            Tensor: A reshaped Tensor with the same data type as ``x``.
        
        Examples:
            .. code-block:: python
        
                import numpy as np
                import paddle
        
                x = paddle.rand([2, 4, 6], dtype="float32")
                positive_four = paddle.full([1], 4, "int32")
        
                out = paddle.reshape(x, [-1, 0, 3, 2])
                print(out)
                # the shape is [2,4,3,2].
        
                out = paddle.reshape(x, shape=[positive_four, 12])
                print(out)
                # the shape of out_2 is [4, 12].
        
                shape_tensor = paddle.to_tensor(np.array([8, 6]).astype("int32"))
                out = paddle.reshape(x, shape=shape_tensor)
                print(out)
                # the shape is [8, 6].
                # out shares data with x in dygraph mode
                x[0, 0, 0] = 10.
                print(out[0, 0])
                # the value is [10.]        """
        pass


    def reshape_(self, shape, name: Optional[str] = None) -> None:
        """
        Inplace version of ``reshape`` API, the output Tensor will be inplaced with input ``x``.
        Please refer to :ref:`api_paddle_tensor_reshape`.        """
        pass


    def reverse(self, axis, name: Optional[str] = None) -> Tensor:
        pass
    def roll(self, shifts, axis: Optional[int] = None, name: Optional[str] = None) -> Tensor:
        """
        Roll the `x` tensor along the given axis(axes). With specific 'shifts', Elements that 
        roll beyond the last position are re-introduced at the first according to 'shifts'. 
        If a axis is not specified, 
        the tensor will be flattened before rolling and then restored to the original shape.
        
        Args:
            self (Tensor): The x tensor as input.
            shifts (int|list|tuple): The number of places by which the elements
                               of the `x` tensor are shifted.
            axis (int|list|tuple|None): axis(axes) along which to roll.
        
        Returns:
            Tensor: A Tensor with same data type as `x`.
        
        Examples:
            .. code-block:: python
                
                import paddle
        
                x = paddle.to_tensor([[1.0, 2.0, 3.0],
                                      [4.0, 5.0, 6.0],
                                      [7.0, 8.0, 9.0]])
                out_z1 = paddle.roll(x, shifts=1)
                print(out_z1)
                #[[9. 1. 2.]
                # [3. 4. 5.]
                # [6. 7. 8.]]
                out_z2 = paddle.roll(x, shifts=1, axis: int = 0)
                print(out_z2)
                #[[7. 8. 9.]
                # [1. 2. 3.]
                # [4. 5. 6.]]        """
        pass


    def rot90(self, k=1, axes=[0, 1], name: Optional[str] = None) -> Tensor:
        """
        Rotate a n-D tensor by 90 degrees. The rotation direction and times are specified by axes. Rotation direction is from axes[0] towards axes[1] if k > 0, and from axes[1] towards axes[0] for k < 0.
        
        Args:
            self (Tensor): The input Tensor(or LoDTensor). The data type of the input Tensor x
                should be float16, float32, float64, int32, int64, bool. float16 is only supported on gpu.
            k (int, optional): Direction and number of times to rotate, default value: 1.
            axes (list|tuple, optional): Axes to rotate, dimension must be 2. default value: [0, 1].
            name (str, optional): The default value is None.  Normally there is no need for user to set this property.
                For more information, please refer to :ref:`api_guide_Name` .
        
        Returns:
            Tensor: Tensor or LoDTensor calculated by rot90 layer. The data type is same with input x.
        
        Examples:
            .. code-block:: python
        
              import paddle
        
              data = paddle.arange(4)
              data = paddle.reshape(data, (2, 2))
              print(data) 
              #[[0, 1],
              # [2, 3]]
        
              y = paddle.rot90(data, 1, [0, 1])
              print(y) 
              #[[1, 3],
              # [0, 2]]
        
              y= paddle.rot90(data, -1, [0, 1])
              print(y) 
              #[[2, 0],
              # [3, 1]]
        
              data2 = paddle.arange(8)
              data2 = paddle.reshape(data2, (2,2,2))
              print(data2) 
              #[[[0, 1],
              #  [2, 3]],
              # [[4, 5],
              #  [6, 7]]]
        
              y = paddle.rot90(data2, 1, [1, 2])
              print(y)
              #[[[1, 3],
              #  [0, 2]],
              # [[5, 7],
              #  [4, 6]]]        """
        pass


    def round(self, name: Optional[str] = None) -> Tensor:
        """
        The OP rounds the values in the input to the nearest integer value.
        
        .. code-block:: text
        
          input:
            x.shape = [4]
            x.data = [1.2, -0.9, 3.4, 0.9]
        
          output:
            out.shape = [4]
            out.data = [1., -1., 3., 1.]
        
        
        Args:
            self (Tensor): Input of Round operator, an N-D Tensor, with data type float32, float64 or float16.
            with_quant_attr (BOOLEAN): Whether the operator has attributes used by quantization. 
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            out (Tensor): Output of Round operator, a Tensor with shape same as input.
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([-0.5, -0.2, 0.6, 1.5])
                out = paddle.round(x)
                print(out)
                # [-1. -0.  1.  2.]        """
        pass


    def round_(self, name: Optional[str] = None) -> None:
        """
        Inplace version of ``round`` API, the output Tensor will be inplaced with input ``x``.
        Please refer to :ref:`api_fluid_layers_round`.        """
        pass


    def rsqrt(self, name: Optional[str] = None) -> Tensor:
        """
        Rsqrt Activation Operator.
        
        Please make sure input is legal in case of numeric errors.
        
        :math:`out = \\frac{1}{\\sqrt{x}}`
        
        
        Args:
            self (Tensor): Input of Rsqrt operator, an N-D Tensor, with data type float32, float64 or float16.
            with_quant_attr (BOOLEAN): Whether the operator has attributes used by quantization. 
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            out (Tensor): Output of Rsqrt operator, a Tensor with shape same as input.
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([0.1, 0.2, 0.3, 0.4])
                out = paddle.rsqrt(x)
                print(out)
                # [3.16227766 2.23606798 1.82574186 1.58113883]        """
        pass


    def rsqrt_(self, name: Optional[str] = None) -> None:
        """
        Inplace version of ``rsqrt`` API, the output Tensor will be inplaced with input ``x``.
        Please refer to :ref:`api_fluid_layers_rsqrt`.        """
        pass


    def scale(self, scale=1.0, bias=0.0, bias_after_scale=True, act=None, name: Optional[str] = None) -> Tensor:
        """
        Scale operator.
        
        Putting scale and bias to the input Tensor as following:
        
        ``bias_after_scale`` is True:
        
        .. math::
                                Out=scale*X+bias
        
        ``bias_after_scale`` is False:
        
        .. math::
                                Out=scale*(X+bias)
        
        Args:
            self (Tensor): Input N-D Tensor of scale operator. Data type can be float32, float64, int8, int16, int32, int64, uint8.
            scale(float|Tensor): The scale factor of the input, it should be a float number or a Tensor with shape [1] and data type as float32.
            bias(float): The bias to be put on the input.
            bias_after_scale(bool): Apply bias addition after or before scaling. It is useful for numeric stability in some circumstances.
            act(str, optional): Activation applied to the output such as tanh, softmax, sigmoid, relu.
            name(str, optional): The default value is None. Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name`
        
        Returns:
            Tensor: Output tensor of scale operator, with shape and data type same as input.
        
        Examples:
            .. code-block:: python
                
                # scale as a float32 number
                import paddle
        
                data = paddle.randn(shape=[2,3], dtype='float32')
                res = paddle.scale(data, scale=2.0, bias=1.0)
        
            .. code-block:: python
        
                # scale with parameter scale as a Tensor
                import paddle
        
                data = paddle.randn(shape=[2, 3], dtype='float32')
                factor = paddle.to_tensor([2], dtype='float32')
                res = paddle.scale(data, scale=factor, bias=1.0)        """
        pass


    def scale_(self, scale=1.0, bias=0.0, bias_after_scale=True, act=None, name: Optional[str] = None) -> None:
        """
        Inplace version of ``scale`` API, the output Tensor will be inplaced with input ``x``.
        Please refer to :ref:`api_tensor_scale`.        """
        pass


    def scatter(self, index, updates, overwrite=True, name: Optional[str] = None) -> Tensor:
        """
        **Scatter Layer**
        Output is obtained by updating the input on selected indices based on updates.
        
        .. code-block:: python
        
            import numpy as np
            #input:
            x = np.array([[1, 1], [2, 2], [3, 3]])
            index = np.array([2, 1, 0, 1])
            # shape of updates should be the same as x
            # shape of updates with dim > 1 should be the same as input
            updates = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
            overwrite = False
            # calculation:
            if not overwrite:
                for i in range(len(index)):
                    x[index[i]] = np.zeros((2))
            for i in range(len(index)):
                if (overwrite):
                    x[index[i]] = updates[i]
                else:
                    x[index[i]] += updates[i]
            # output:
            out = np.array([[3, 3], [6, 6], [1, 1]])
            out.shape # [3, 2]
        
        **NOTICE**: The order in which updates are applied is nondeterministic, 
        so the output will be nondeterministic if index contains duplicates.
        
        Args:
            self (Tensor): The input N-D Tensor with ndim>=1. Data type can be float32, float64.
            indeself (Tensor): The index 1-D Tensor. Data type can be int32, int64. The length of index cannot exceed updates's length, and the value in index cannot exceed input's length.
            updates (Tensor): update input with updates parameter based on index. shape should be the same as input, and dim value with dim > 1 should be the same as input.
            overwrite (bool): The mode that updating the output when there are same indices. 
                
                If True, use the overwrite mode to update the output of the same index,
                    if False, use the accumulate mode to update the output of the same index.Default value is True.
            
            name(str, optional): The default value is None. Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name` .
        
        Returns:
            Tensor: The output is a Tensor with the same shape as x.
        
        Examples:
            .. code-block:: python
                
                import paddle
        
                x = paddle.to_tensor([[1, 1], [2, 2], [3, 3]], dtype='float32')
                index = paddle.to_tensor([2, 1, 0, 1], dtype='int64')
                updates = paddle.to_tensor([[1, 1], [2, 2], [3, 3], [4, 4]], dtype='float32')
        
                output1 = paddle.scatter(x, index, updates, overwrite=False)
                # [[3., 3.],
                #  [6., 6.],
                #  [1., 1.]]
        
                output2 = paddle.scatter(x, index, updates, overwrite=True)
                # CPU device:
                # [[3., 3.],
                #  [4., 4.],
                #  [1., 1.]]
                # GPU device maybe have two results because of the repeated numbers in index
                # result 1:
                # [[3., 3.],
                #  [4., 4.],
                #  [1., 1.]]
                # result 2:
                # [[3., 3.],
                #  [2., 2.],
                #  [1., 1.]]        """
        pass


    def scatter_(self, index, updates, overwrite=True, name: Optional[str] = None) -> None:
        """
        Inplace version of ``scatter`` API, the output Tensor will be inplaced with input ``x``.
        Please refer to :ref:`api_paddle_tensor_scatter`.        """
        pass


    def scatter_nd(self, updates, shape, name: Optional[str] = None) -> Tensor:
        """
        **Scatter_nd Layer**
        
        Output is obtained by scattering the :attr:`updates` in a new tensor according
        to :attr:`index` . This op is similar to :code:`scatter_nd_add`, except the
        tensor of :attr:`shape` is zero-initialized. Correspondingly, :code:`scatter_nd(index, updates, shape)`
        is equal to :code:`scatter_nd_add(paddle.zeros(shape, updates.dtype), index, updates)` .
        If :attr:`index` has repeated elements, then the corresponding updates are accumulated.
        Because of the numerical approximation issues, the different order of repeated elements
        in :attr:`index` may cause different results. The specific calculation method can be
        seen :code:`scatter_nd_add` . This op is the inverse of the :code:`gather_nd` op.
        
        Args:
            indeself (Tensor): The index input with ndim > 1 and index.shape[-1] <= len(shape).
                              Its dtype should be int32 or int64 as it is used as indexes.
            updates (Tensor): The updated value of scatter_nd op. Its dtype should be float32, float64.
                                It must have the shape index.shape[:-1] + shape[index.shape[-1]:]
            shape(tuple|list): Shape of output tensor.
            name (str|None): The output Tensor name. If set None, the layer will be named automatically.
        
        Returns:
            output (Tensor): The output is a tensor with the same type as :attr:`updates` .
        
        Examples:
        
            .. code-block:: python
        
                import paddle
                import numpy as np
        
                index_data = np.array([[1, 1],
                                       [0, 1],
                                       [1, 3]]).astype(np.int64)
                index = paddle.to_tensor(index_data)
                updates = paddle.rand(shape=[3, 9, 10], dtype='float32')
                shape = [3, 5, 9, 10]
        
                output = paddle.scatter_nd(index, updates, shape)        """
        pass


    def scatter_nd_add(self, index, updates, name: Optional[str] = None) -> Tensor:
        """
        **Scatter_nd_add Layer**
        
        Output is obtained by applying sparse addition to a single value
        or slice in a Tensor.
        
        :attr:`x` is a Tensor with ndim :math:`R`
        and :attr:`index` is a Tensor with ndim :math:`K` . Thus, :attr:`index`
        has shape :math:`[i_0, i_1, ..., i_{K-2}, Q]` where :math:`Q \leq R` . :attr:`updates`
        is a Tensor with ndim :math:`K - 1 + R - Q` and its
        shape is :math:`index.shape[:-1] + x.shape[index.shape[-1]:]` .
        
        According to the :math:`[i_0, i_1, ..., i_{K-2}]` of :attr:`index` ,
        add the corresponding :attr:`updates` slice to the :attr:`x` slice
        which is obtained by the last one dimension of :attr:`index` .
        
        .. code-block:: text
        
            Given:
        
            * Case 1:
                x = [0, 1, 2, 3, 4, 5]
                index = [[1], [2], [3], [1]]
                updates = [9, 10, 11, 12]
        
              we get:
        
                output = [0, 22, 12, 14, 4, 5]
        
            * Case 2:
                x = [[65, 17], [-14, -25]]
                index = [[], []]
                updates = [[[-1, -2], [1, 2]],
                           [[3, 4], [-3, -4]]]
                x.shape = (2, 2)
                index.shape = (2, 0)
                updates.shape = (2, 2, 2)
        
              we get:
        
                output = [[67, 19], [-16, -27]]
        
        Args:
            self (Tensor): The x input. Its dtype should be int32, int64, float32, float64.
            indeself (Tensor): The index input with ndim > 1 and index.shape[-1] <= x.ndim.
                              Its dtype should be int32 or int64 as it is used as indexes.
            updates (Tensor): The updated value of scatter_nd_add op, and it must have the same dtype
                                as x. It must have the shape index.shape[:-1] + x.shape[index.shape[-1]:].
            name (str|None): The output tensor name. If set None, the layer will be named automatically.
        
        Returns:
            output (Tensor): The output is a tensor with the same shape and dtype as x.
        
        Examples:
        
            .. code-block:: python
        
                import paddle
                import numpy as np
        
                x = paddle.rand(shape=[3, 5, 9, 10], dtype='float32')
                updates = paddle.rand(shape=[3, 9, 10], dtype='float32')
                index_data = np.array([[1, 1],
                                       [0, 1],
                                       [1, 3]]).astype(np.int64)
                index = paddle.to_tensor(index_data)
                output = paddle.scatter_nd_add(x, index, updates)        """
        pass


    def set_value(self, value) -> Tuple[]:
        pass
    def shard_index(self, index_num, nshards, shard_id, ignore_value=-1) -> Tensor:
        """
        Reset the values of `input` according to the shard it beloning to.
        Every value in `input` must be a non-negative integer, and
        the parameter `index_num` represents the integer above the maximum
        value of `input`. Thus, all values in `input` must be in the range
        [0, index_num) and each value can be regarded as the offset to the beginning
        of the range. The range is further split into multiple shards. Specifically,
        we first compute the `shard_size` according to the following formula,
        which represents the number of integers each shard can hold. So for the
        i'th shard, it can hold values in the range [i*shard_size, (i+1)*shard_size).
        ::
        
            shard_size = (index_num + nshards - 1) // nshards
        
        For each value `v` in `input`, we reset it to a new value according to the
        following formula:
        ::
        
            v = v - shard_id * shard_size if shard_id * shard_size <= v < (shard_id+1) * shard_size else ignore_value
        
        That is, the value `v` is set to the new offset within the range represented by the shard `shard_id`
        if it in the range. Otherwise, we reset it to be `ignore_value`.
        
        Args:
            input (Tensor): Input tensor with data type int64 or int32. It's last dimension must be 1.
            index_num (int): An integer represents the integer above the maximum value of `input`.
            nshards (int): The number of shards.
            shard_id (int): The index of the current shard.
            ignore_value (int): An integer value out of sharded index range.
        
        Returns:
            Tensor.
        
        Examples:
            .. code-block:: python
        
                import paddle
                label = paddle.to_tensor([[16], [1]], "int64")
                shard_label = paddle.shard_index(input=label,
                                                 index_num=20,
                                                 nshards=2,
                                                 shard_id=0)
                print(shard_label)
                # [[-1], [1]]        """
        pass


    def sign(self, name: Optional[str] = None) -> Tensor:
        """
        This OP returns sign of every element in `x`: 1 for positive, -1 for negative and 0 for zero.
        
        Args:
            self (Tensor): The input tensor. The data type can be float16, float32 or float64.
            name (str, optional): The default value is None. Normally there is no need for user to
                set this property. For more information, please refer to :ref:`api_guide_Name`
        
        Returns:
            Tensor: The output sign tensor with identical shape and data type to the input :attr:`x`.
        
        Examples:
            .. code-block:: python
        
              import paddle
        
              x = paddle.to_tensor([3.0, 0.0, -2.0, 1.7], dtype='float32')
              out = paddle.sign(x=x)
              print(out)  # [1.0, 0.0, -1.0, 1.0]        """
        pass


    def sin(self, name: Optional[str] = None) -> Tensor:
        """
        Sine Activation Operator.
        
        :math:`out = sin(x)`
        
        
        Args:
            self (Tensor): Input of Sin operator, an N-D Tensor, with data type float32, float64 or float16.
            with_quant_attr (BOOLEAN): Whether the operator has attributes used by quantization. 
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            out (Tensor): Output of Sin operator, a Tensor with shape same as input.
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
                out = paddle.sin(x)
                print(out)
                # [-0.38941834 -0.19866933  0.09983342  0.29552021]        """
        pass


    def sinh(self, name: Optional[str] = None) -> Tensor:
        """
        Sinh Activation Operator.
        
        :math:`out = sinh(x)`
        
        
        Args:
            self (Tensor): Input of Sinh operator, an N-D Tensor, with data type float32, float64 or float16.
            with_quant_attr (BOOLEAN): Whether the operator has attributes used by quantization. 
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            out (Tensor): Output of Sinh operator, a Tensor with shape same as input.
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
                out = paddle.sinh(x)
                print(out)
                # [-0.41075233 -0.201336    0.10016675  0.30452029]        """
        pass


    def slice(self, axes, starts, ends) -> Tensor:
        """
        This operator produces a slice of ``input`` along multiple axes. Similar to numpy:
        https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
        Slice uses ``axes``, ``starts`` and ``ends`` attributes to specify the start and
        end dimension for each axis in the list of axes and Slice uses this information
        to slice the input data tensor. If a negative value is passed to
        ``starts`` or ``ends`` such as :math:`-i`,  it represents the reverse position of the
        axis :math:`i-1` (here 0 is the initial position).
        If the value passed to ``starts`` or ``ends`` is greater than n
        (the number of elements in this dimension), it represents n.
        For slicing to the end of a dimension with unknown size, it is recommended
        to pass in INT_MAX. The size of ``axes`` must be equal to ``starts`` and ``ends``.
        Following examples will explain how slice works:
        
        .. code-block:: text
        
            Case1:
                Given:
                    data = [ [1, 2, 3, 4], [5, 6, 7, 8], ]
                    axes = [0, 1]
                    starts = [1, 0]
                    ends = [2, 3]
                Then:
                    result = [ [5, 6, 7], ]
        
            Case2:
                Given:
                    data = [ [1, 2, 3, 4], [5, 6, 7, 8], ]
                    axes = [0, 1]
                    starts = [0, 1]
                    ends = [-1, 1000]       # -1 denotes the reverse 0th position of dimension 0.
                Then:
                    result = [ [2, 3, 4], ] # result = data[0:1, 1:4]
        
        Args:
            input (Tensor): A ``Tensor`` . The data type is ``float16``, ``float32``, ``float64``, ``int32`` or ``int64``.
            axes (list|tuple): The data type is ``int32`` . Axes that `starts` and `ends` apply to .
            starts (list|tuple|Tensor): The data type is ``int32`` . If ``starts`` is a list or tuple, the elements of
                    it should be integers or Tensors with shape [1]. If ``starts`` is an Tensor, it should be an 1-D Tensor.
                    It represents starting indices of corresponding axis in ``axes``.
            ends (list|tuple|Tensor): The data type is ``int32`` . If ``ends`` is a list or tuple, the elements of
                    it should be integers or Tensors with shape [1]. If ``ends`` is an Tensor, it should be an 1-D Tensor .
                    It represents ending indices of corresponding axis in ``axes``.
        
        Returns:
            Tensor:  A ``Tensor``. The data type is same as ``input``.
        
        Raises:
            TypeError: The type of ``starts`` must be list, tuple or Tensor.
            TypeError: The type of ``ends`` must be list, tuple or Tensor.
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                input = paddle.rand(shape=[4, 5, 6], dtype='float32')
                # example 1:
                # attr starts is a list which doesn't contain tensor.
                axes = [0, 1, 2]
                starts = [-3, 0, 2]
                ends = [3, 2, 4]
                sliced_1 = paddle.slice(input, axes=axes, starts=starts, ends=ends)
                # sliced_1 is input[0:3, 0:2, 2:4].
        
                # example 2:
                # attr starts is a list which contain tensor.
                minus_3 = paddle.full([1], -3, "int32")
                sliced_2 = paddle.slice(input, axes=axes, starts=[minus_3, 0, 2], ends=ends)
                # sliced_2 is input[0:3, 0:2, 2:4].        """
        pass


    def solve(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        """
        Computes the solution of a square system of linear equations with a unique solution for input 'X' and 'Y'.
        Let :math: `X` be a sqaure matrix or a batch of square matrices, :math:`Y` be
        a vector/matrix or a batch of vectors/matrices, the equation should be:
        
        .. math::
            Out = X^-1 * Y
        Specifically,
        - This system of linear equations has one solution if and only if input 'X' is invertible.
        
        Args:
            self (Tensor): A square matrix or a batch of square matrices. Its shape should be `[*, M, M]`, where `*` is zero or
                more batch dimensions. Its data type should be float32 or float64.
            y (Tensor): A vector/matrix or a batch of vectors/matrices. Its shape should be `[*, M, K]`, where `*` is zero or
                more batch dimensions. Its data type should be float32 or float64.
            name(str, optional): Name for the operation (optional, default is None).
                For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            Tensor: The solution of a square system of linear equations with a unique solution for input 'x' and 'y'.
            Its data type should be the same as that of `x`.
        
        Examples:
        .. code-block:: python
        
            # a square system of linear equations:
            # 2*X0 + X1 = 9
            # X0 + 2*X1 = 8
        
            import paddle
            import numpy as np
        
            np_x = np.array([[3, 1],[1, 2]])
            np_y = np.array([9, 8])
            x = paddle.to_tensor(np_x, dtype="float64")
            y = paddle.to_tensor(np_y, dtype="float64")
            out = paddle.linalg.solve(x, y)
        
            print(out)
            # [2., 3.])        """
        pass


    def sort(self, axis: int = -1, descending=False, name: Optional[str] = None) -> Tensor:
        """
        This OP sorts the input along the given axis, and returns the sorted output tensor. The default sort algorithm is ascending, if you want the sort algorithm to be descending, you must set the :attr:`descending` as True.
        
        Args:
            self (Tensor): An input N-D Tensor with type float32, float64, int16,
                int32, int64, uint8.
            axis(int, optional): Axis to compute indices along. The effective range
                is [-R, R), where R is Rank(x). when axis<0, it works the same way
                as axis+R. Default is 0.
            descending(bool, optional) : Descending is a flag, if set to true,
                algorithm will sort by descending order, else sort by
                ascending order. Default is false.
            name(str, optional): The default value is None. Normally there is no
                need for user to set this property. For more information, please
                refer to :ref:`api_guide_Name`.
        Returns:
            Tensor: sorted tensor(with the same shape and data type as ``x``).
        Examples:
        
            .. code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([[[5,8,9,5],
                                       [0,0,1,7],
                                       [6,9,2,4]],
                                      [[5,2,4,2],
                                       [4,7,7,9],
                                       [1,7,0,6]]], 
                                     dtype='float32')
                out1 = paddle.sort(x=x, axis: int = -1)
                out2 = paddle.sort(x=x, axis: int = 0)
                out3 = paddle.sort(x=x, axis: int = 1)
                print(out1)
                #[[[5. 5. 8. 9.]
                #  [0. 0. 1. 7.]
                #  [2. 4. 6. 9.]]
                # [[2. 2. 4. 5.]
                #  [4. 7. 7. 9.]
                #  [0. 1. 6. 7.]]]
                print(out2)
                #[[[5. 2. 4. 2.]
                #  [0. 0. 1. 7.]
                #  [1. 7. 0. 4.]]
                # [[5. 8. 9. 5.]
                #  [4. 7. 7. 9.]
                #  [6. 9. 2. 6.]]]
                print(out3)
                #[[[0. 0. 1. 4.]
                #  [5. 8. 2. 5.]
                #  [6. 9. 9. 7.]]
                # [[1. 2. 0. 2.]
                #  [4. 7. 4. 6.]
                #  [5. 7. 7. 9.]]]        """
        pass


    def split(self, num_or_sections, axis: int = 0, name: Optional[str] = None) -> List[Tensor]:
        """
        Split the input tensor into multiple sub-Tensors.
        
        Args:
            self (Tensor): A N-D Tensor. The data type is bool, float16, float32, float64, int32 or int64.
            num_or_sections (int|list|tuple): If ``num_or_sections`` is an int, then ``num_or_sections`` 
                indicates the number of equal sized sub-Tensors that the ``x`` will be divided into.
                If ``num_or_sections`` is a list or tuple, the length of it indicates the number of
                sub-Tensors and the elements in it indicate the sizes of sub-Tensors'  dimension orderly.
                The length of the list must not  be larger than the ``x`` 's size of specified ``axis``.
            axis (int|Tensor, optional): The axis along which to split, it can be a scalar with type 
                ``int`` or a ``Tensor`` with shape [1] and data type  ``int32`` or ``int64``.
                If :math::`axis < 0`, the axis to split along is :math:`rank(x) + axis`. Default is 0.
            name (str, optional): The default value is None.  Normally there is no need for user to set this property.
                For more information, please refer to :ref:`api_guide_Name` .
        Returns:
            list(Tensor): The list of segmented Tensors.
        
        Example:
            .. code-block:: python
                
                import paddle
                
                # x is a Tensor of shape [3, 9, 5]
                x = paddle.rand([3, 9, 5])
        
                out0, out1, out2 = paddle.split(x, num_or_sections=3, axis: int = 1)
                print(out0.shape)  # [3, 3, 5]
                print(out1.shape)  # [3, 3, 5]
                print(out2.shape)  # [3, 3, 5]
        
                out0, out1, out2 = paddle.split(x, num_or_sections=[2, 3, 4], axis: int = 1)
                print(out0.shape)  # [3, 2, 5]
                print(out1.shape)  # [3, 3, 5]
                print(out2.shape)  # [3, 4, 5]
        
                out0, out1, out2 = paddle.split(x, num_or_sections=[2, 3, -1], axis: int = 1)
                print(out0.shape)  # [3, 2, 5]
                print(out1.shape)  # [3, 3, 5]
                print(out2.shape)  # [3, 4, 5]
                
                # axis is negative, the real axis is (rank(x) + axis)=1
                out0, out1, out2 = paddle.split(x, num_or_sections=3, axis: int = -2)
                print(out0.shape)  # [3, 3, 5]
                print(out1.shape)  # [3, 3, 5]
                print(out2.shape)  # [3, 3, 5]        """
        pass


    def sqrt(self, name: Optional[str] = None) -> Tensor:
        """
        Sqrt Activation Operator.
        
        :math:`out=\\sqrt{x}=x^{1/2}`
        
        **Note**:
          input value must be greater than or equal to zero.
        
        
        Args:
            self (Tensor): Input of Sqrt operator, an N-D Tensor, with data type float32, float64 or float16.
            with_quant_attr (BOOLEAN): Whether the operator has attributes used by quantization. 
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            out (Tensor): Output of Sqrt operator, a Tensor with shape same as input.
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([0.1, 0.2, 0.3, 0.4])
                out = paddle.sqrt(x)
                print(out)
                # [0.31622777 0.4472136  0.54772256 0.63245553]        """
        pass


    def sqrt_(self, name: Optional[str] = None) -> None:
        """
        Inplace version of ``sqrt`` API, the output Tensor will be inplaced with input ``x``.
        Please refer to :ref:`api_fluid_layers_sqrt`.        """
        pass


    def square(self, name: Optional[str] = None) -> Tensor:
        """
        The OP square each elements of the inputs.
        
        :math:`out = x^2`
        
        
        Args:
            self (Tensor): Input of Square operator, an N-D Tensor, with data type float32, float64 or float16.
            with_quant_attr (BOOLEAN): Whether the operator has attributes used by quantization. 
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            out (Tensor): Output of Square operator, a Tensor with shape same as input.
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
                out = paddle.square(x)
                print(out)
                # [0.16 0.04 0.01 0.09]        """
        pass


    def squeeze(self, axis: Optional[int] = None, name: Optional[str] = None) -> Tuple[Tensor, Tensor]:
        """
        This OP will squeeze the dimension(s) of size 1 of input tensor x's shape. 
        
        Note that the output Tensor will share data with origin Tensor and doesn't have a 
        Tensor copy in ``dygraph`` mode. If you want to use the Tensor copy version, 
        please use `Tensor.clone` like ``squeeze_clone_x = x.squeeze().clone()``.
        
        If axis is provided, it will remove the dimension(s) by given axis that of size 1. 
        If the dimension of given axis is not of size 1, the dimension remain unchanged. 
        If axis is not provided, all dims equal of size 1 will be removed.
        
        .. code-block:: text
        
            Case1:
        
              Input:
                x.shape = [1, 3, 1, 5]  # If axis is not provided, all dims equal of size 1 will be removed.
                axis = None
              Output:
                out.shape = [3, 5]
        
            Case2:
        
              Input:
                x.shape = [1, 3, 1, 5]  # If axis is provided, it will remove the dimension(s) by given axis that of size 1.
                axis = 0
              Output:
                out.shape = [3, 1, 5]
            
            Case4:
        
              Input:
                x.shape = [1, 3, 1, 5]  # If the dimension of one given axis (3) is not of size 1, the dimension remain unchanged. 
                axis = [0, 2, 3]
              Output:
                out.shape = [3, 5]
        
            Case4:
        
              Input:
                x.shape = [1, 3, 1, 5]  # If axis is negative, axis = axis + ndim (number of dimensions in x). 
                axis = [-2]
              Output:
                out.shape = [1, 3, 5]
        
        Args:
            self (Tensor): The input Tensor. Supported data type: float32, float64, bool, int8, int32, int64.
            axis (int|list|tuple, optional): An integer or list/tuple of integers, indicating the dimensions to be squeezed. Default is None.
                              The range of axis is :math:`[-ndim(x), ndim(x))`.
                              If axis is negative, :math:`axis = axis + ndim(x)`.
                              If axis is None, all the dimensions of x of size 1 will be removed.
            name (str, optional): Please refer to :ref:`api_guide_Name`, Default None.
        
        Returns:
            Tensor: Squeezed Tensor with the same data type as input Tensor.
        
        Examples:
            .. code-block:: python
        
                import paddle
                
                x = paddle.rand([5, 1, 10])
                output = paddle.squeeze(x, axis: int = 1)
        
                print(x.shape)  # [5, 1, 10]
                print(output.shape)  # [5, 10]
        
                # output shares data with x in dygraph mode
                x[0, 0, 0] = 10.
                print(output[0, 0]) # [10.]        """
        pass


    def squeeze_(self, axis: Optional[int] = None, name: Optional[str] = None) -> None:
        """
        Inplace version of ``squeeze`` API, the output Tensor will be inplaced with input ``x``.
        Please refer to :ref:`api_paddle_tensor_squeeze`.        """
        pass


    def stack(self, axis: int = 0, name: Optional[str] = None) -> Tensor:
        """
        This OP stacks all the input tensors ``x`` along ``axis`` dimemsion. 
        All tensors must be of the same shape and same dtype.
        
        For example, given N tensors of shape [A, B], if ``axis == 0``, the shape of stacked 
        tensor is [N, A, B]; if ``axis == 1``, the shape of stacked 
        tensor is [A, N, B], etc.
        
        
        .. code-block:: text
        
            Case 1:
        
              Input:
                x[0].shape = [1, 2]
                x[0].data = [ [1.0 , 2.0 ] ]
                x[1].shape = [1, 2]
                x[1].data = [ [3.0 , 4.0 ] ]
                x[2].shape = [1, 2]
                x[2].data = [ [5.0 , 6.0 ] ]
        
              Attrs:
                axis = 0
        
              Output:
                Out.dims = [3, 1, 2]
                Out.data =[ [ [1.0, 2.0] ],
                            [ [3.0, 4.0] ],
                            [ [5.0, 6.0] ] ]
        
        
            Case 2:
        
              Input:
                x[0].shape = [1, 2]
                x[0].data = [ [1.0 , 2.0 ] ]
                x[1].shape = [1, 2]
                x[1].data = [ [3.0 , 4.0 ] ]
                x[2].shape = [1, 2]
                x[2].data = [ [5.0 , 6.0 ] ]
        
        
              Attrs:
                axis = 1 or axis = -2  # If axis = -2, axis = axis+ndim(x[0])+1 = -2+2+1 = 1.
        
              Output:
                Out.shape = [1, 3, 2]
                Out.data =[ [ [1.0, 2.0]
                              [3.0, 4.0]
                              [5.0, 6.0] ] ]
        
        Args:
            x (list[Tensor]|tuple[Tensor]): Input ``x`` can be a ``list`` or ``tuple`` of tensors, the Tensors in ``x``
                                         must be of the same shape and dtype. Supported data types: float32, float64, int32, int64.
            axis (int, optional): The axis along which all inputs are stacked. ``axis`` range is ``[-(R+1), R+1)``,
                                  where ``R`` is the number of dimensions of the first input tensor ``x[0]``. 
                                  If ``axis < 0``, ``axis = axis+R+1``. The default value of axis is 0.
            name (str, optional): Please refer to :ref:`api_guide_Name`, Default None.
            
        Returns:
            Tensor: The stacked tensor with same data type as input.
        
        Example:    
            .. code-block:: python
        
                import paddle
                
                x1 = paddle.to_tensor([[1.0, 2.0]])
                x2 = paddle.to_tensor([[3.0, 4.0]])
                x3 = paddle.to_tensor([[5.0, 6.0]])
                
                out = paddle.stack([x1, x2, x3], axis: int = 0)
                print(out.shape)  # [3, 1, 2]
                print(out)
                # [[[1., 2.]],
                #  [[3., 4.]],
                #  [[5., 6.]]]
                
                out = paddle.stack([x1, x2, x3], axis: int = -2)
                print(out.shape)  # [1, 3, 2]
                print(out)
                # [[[1., 2.],
                #   [3., 4.],
                #   [5., 6.]]]        """
        pass


    def stanh(self, scale_a=0.67, scale_b=1.7159, name: Optional[str] = None) -> Tensor:
        """
        stanh activation.
        
        .. math::
        
            out = b * \frac{e^{a * x} - e^{-a * x}}{e^{a * x} + e^{-a * x}}
        
        Parameters:
            self (Tensor): The input Tensor with data type float32, float64.
            scale_a (float, optional): The scale factor a of the input. Default is 0.67.
            scale_b (float, optional): The scale factor b of the output. Default is 1.7159.
            name (str, optional): Name for the operation (optional, default is None).
                For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            A Tensor with the same data type and shape as ``x`` .
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
                out = paddle.stanh(x, scale_a=0.67, scale_b=1.72) # [1.00616539, 1.49927628, 1.65933108, 1.70390463]        """
        pass


    def std(self, axis: Optional[int] = None, unbiased=True, keepdim: bool = False, name: Optional[str] = None) -> Tensor:
        """
        Computes the standard-deviation of ``x`` along ``axis`` .
        
        Args:
            self (Tensor): The input Tensor with data type float32, float64.
            axis (int|list|tuple, optional): The axis along which to perform
                standard-deviation calculations. ``axis`` should be int, list(int)
                or tuple(int). If ``axis`` is a list/tuple of dimension(s),
                standard-deviation is calculated along all element(s) of ``axis`` .
                ``axis`` or element(s) of ``axis`` should be in range [-D, D),
                where D is the dimensions of ``x`` . If ``axis`` or element(s) of
                ``axis`` is less than 0, it works the same way as :math:`axis + D` .
                If ``axis`` is None, standard-deviation is calculated over all
                elements of ``x``. Default is None.
            unbiased (bool, optional): Whether to use the unbiased estimation. If
                ``unbiased`` is True, the standard-deviation is calculated via the
                unbiased estimator. If ``unbiased`` is True,  the divisor used in
                the computation is :math:`N - 1`, where :math:`N` represents the
                number of elements along ``axis`` , otherwise the divisor is
                :math:`N`. Default is True.
            keepdim (bool, optional): Whether to reserve the reduced dimension(s)
                in the output Tensor. If ``keepdim`` is True, the dimensions of
                the output Tensor is the same as ``x`` except in the reduced
                dimensions(it is of size 1 in this case). Otherwise, the shape of
                the output Tensor is squeezed in ``axis`` . Default is False.
            name (str, optional): Name for the operation (optional, default is None).
                For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            Tensor, results of standard-deviation along ``axis`` of ``x``, with the
            same data type as ``x``.
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([[1.0, 2.0, 3.0], [1.0, 4.0, 5.0]])
                out1 = paddle.std(x)
                # [1.63299316]
                out2 = paddle.std(x, axis: int = 1)
                # [1.       2.081666]        """
        pass


    def strided_slice(self, axes, starts, ends, strides, name: Optional[str] = None) -> Tensor:
        """
        This operator produces a slice of ``x`` along multiple axes. Similar to numpy:
        https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
        Slice uses ``axes``, ``starts`` and ``ends`` attributes to specify the start and
        end dimension for each axis in the list of axes and Slice uses this information
        to slice the input data tensor. If a negative value is passed to
        ``starts`` or ``ends`` such as :math:`-i`,  it represents the reverse position of the
        axis :math:`i-1` th(here 0 is the initial position). The ``strides`` represents steps of
        slicing and if the ``strides`` is negative, slice operation is in the opposite direction.
        If the value passed to ``starts`` or ``ends`` is greater than n
        (the number of elements in this dimension), it represents n.
        For slicing to the end of a dimension with unknown size, it is recommended
        to pass in INT_MAX. The size of ``axes`` must be equal to ``starts`` , ``ends`` and ``strides``.
        Following examples will explain how strided_slice works:
        
        .. code-block:: text
        
            Case1:
                Given:
                    data = [ [1, 2, 3, 4], [5, 6, 7, 8], ]
                    axes = [0, 1]
                    starts = [1, 0]
                    ends = [2, 3]
                    strides = [1, 1]
                Then:
                    result = [ [5, 6, 7], ]
        
            Case2:
                Given:
                    data = [ [1, 2, 3, 4], [5, 6, 7, 8], ]
                    axes = [0, 1]
                    starts = [0, 1]
                    ends = [2, 0]
                    strides = [1, -1]
                Then:
                    result = [ [8, 7, 6], ]
            Case3:
                Given:
                    data = [ [1, 2, 3, 4], [5, 6, 7, 8], ]
                    axes = [0, 1]
                    starts = [0, 1]
                    ends = [-1, 1000]
                    strides = [1, 3]
                Then:
                    result = [ [2], ]
        
        Args:
            self (Tensor): An N-D ``Tensor``. The data type is ``float32``, ``float64``, ``int32`` or ``int64``.
            axes (list|tuple): The data type is ``int32`` . Axes that `starts` and `ends` apply to.
                                It's optional. If it is not provides, it will be treated as :math:`[0,1,...,len(starts)-1]`.
            starts (list|tuple|Tensor): The data type is ``int32`` . If ``starts`` is a list or tuple, the elements of                                                                                          it should be integers or Tensors with shape [1]. If ``starts`` is an Tensor, it should be an 1-D Tensor.                                                                                    It represents starting indices of corresponding axis in ``axes``.
            ends (list|tuple|Tensor): The data type is ``int32`` . If ``ends`` is a list or tuple, the elements of
                    it should be integers or Tensors with shape [1]. If ``ends`` is an Tensor, it should be an 1-D Tensor .                                                                                     It represents ending indices of corresponding axis in ``axes``.
            strides (list|tuple|Tensor): The data type is ``int32`` . If ``strides`` is a list or tuple, the elements of
                    it should be integers or Tensors with shape [1]. If ``strides`` is an Tensor, it should be an 1-D Tensor .                                                                                  It represents slice step of corresponding axis in ``axes``.
            name(str, optional): The default value is None.  Normally there is no need for user to set this property.
                            For more information, please refer to :ref:`api_guide_Name` .
        
        Returns:
            Tensor:  A ``Tensor`` with the same dimension as ``x``. The data type is same as ``x``.
        
        Examples:
            .. code-block:: python
        
                import paddle
                x = paddle.zeros(shape=[3,4,5,6], dtype="float32")
                # example 1:
                # attr starts is a list which doesn't contain Tensor.
                axes = [1, 2, 3]
                starts = [-3, 0, 2]
                ends = [3, 2, 4]
                strides_1 = [1, 1, 1]
                strides_2 = [1, 1, 2]
                sliced_1 = paddle.strided_slice(x, axes=axes, starts=starts, ends=ends, strides=strides_1)
                # sliced_1 is x[:, 1:3:1, 0:2:1, 2:4:1].                                
                # example 2:
                # attr starts is a list which contain tensor Tensor.
                minus_3 = paddle.full(shape=[1], fill_value=-3, dtype='int32')
                sliced_2 = paddle.strided_slice(x, axes=axes, starts=[minus_3, 0, 2], ends=ends, strides=strides_2)
                # sliced_2 is x[:, 1:3:1, 0:2:1, 2:4:2].        """
        pass


    def subtract(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        """
        Substract two tensors element-wise. The equation is:
        
        .. math::
            out = x - y
        
        **Note**:
        ``paddle.subtract`` supports broadcasting. If you want know more about broadcasting, please refer to :ref:`user_guide_broadcasting` .
        
        Args:
            self (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
            y (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            N-D Tensor. A location into which the result is stored. If x, y have different shapes and are "broadcastable", the resulting tensor shape is the shape of x and y after broadcasting. If x, y have the same shape,  its shape is the same as x and y.
        
        Examples:
        
            .. code-block:: python
        
                import numpy as np
                import paddle
        
                x = paddle.to_tensor([[1, 2], [7, 8]])
                y = paddle.to_tensor([[5, 6], [3, 4]])
                res = paddle.subtract(x, y)
                print(res)
                #       [[-4, -4],
                #        [4, 4]]
        
                x = paddle.to_tensor([[[1, 2, 3], [1, 2, 3]]])
                y = paddle.to_tensor([1, 0, 4])
                res = paddle.subtract(x, y)
                print(res)
                #       [[[ 0,  2, -1],
                #         [ 0,  2, -1]]]
        
                x = paddle.to_tensor([2, np.nan, 5], dtype='float32')
                y = paddle.to_tensor([1, 4, np.nan], dtype='float32')
                res = paddle.subtract(x, y)
                print(res)
                #       [ 1., nan, nan]
        
                x = paddle.to_tensor([5, np.inf, -np.inf], dtype='float64')
                y = paddle.to_tensor([1, 4, 5], dtype='float64')
                res = paddle.subtract(x, y)
                print(res)
                #       [   4.,  inf., -inf.]        """
        pass


    def subtract_(self, y: Tensor, name: Optional[str] = None) -> None:
        """
        Inplace version of ``subtract`` API, the output Tensor will be inplaced with input ``x``.
        Please refer to :ref:`api_tensor_subtract`.        """
        pass


    def sum(self, axis: Optional[int] = None, dtype=None, keepdim: bool = False, name: Optional[str] = None) -> Tensor:
        """
        Computes the sum of tensor elements over the given dimension.
        
        Args:
            self (Tensor): An N-D Tensor, the data type is bool, float16, float32, float64, int32 or int64.
            axis (int|list|tuple, optional): The dimensions along which the sum is performed. If
                :attr:`None`, sum all elements of :attr:`x` and return a
                Tensor with a single element, otherwise must be in the
                range :math:`[-rank(x), rank(x))`. If :math:`axis[i] < 0`,
                the dimension to reduce is :math:`rank + axis[i]`.
            dtype (str, optional): The dtype of output Tensor. The default value is None, the dtype
                of output is the same as input Tensor `x`.
            keepdim (bool, optional): Whether to reserve the reduced dimension in the
                output Tensor. The result Tensor will have one fewer dimension
                than the :attr:`x` unless :attr:`keepdim` is true, default
                value is False.
            name (str, optional): The default value is None. Normally there is no need for
                user to set this property.  For more information, please refer to :ref:`api_guide_Name`
        
        Returns:
            Tensor: Results of summation operation on the specified axis of input Tensor `x`,
            if `x.dtype='bool'`, `x.dtype='int32'`, it's data type is `'int64'`, 
            otherwise it's data type is the same as `x`.
        
        Raises:
            TypeError: The type of :attr:`axis` must be int, list or tuple.
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                # x is a Tensor with following elements:
                #    [[0.2, 0.3, 0.5, 0.9]
                #     [0.1, 0.2, 0.6, 0.7]]
                # Each example is followed by the corresponding output tensor.
                x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
                                      [0.1, 0.2, 0.6, 0.7]])
                out1 = paddle.sum(x)  # [3.5]
                out2 = paddle.sum(x, axis: int = 0)  # [0.3, 0.5, 1.1, 1.6]
                out3 = paddle.sum(x, axis: int = -1)  # [1.9, 1.6]
                out4 = paddle.sum(x, axis: int = 1, keepdim=True)  # [[1.9], [1.6]]
        
                # y is a Tensor with shape [2, 2, 2] and elements as below:
                #      [[[1, 2], [3, 4]],
                #      [[5, 6], [7, 8]]]
                # Each example is followed by the corresponding output tensor.
                y = paddle.to_tensor([[[1, 2], [3, 4]], 
                                      [[5, 6], [7, 8]]])
                out5 = paddle.sum(y, axis: int = [1, 2]) # [10, 26]
                out6 = paddle.sum(y, axis: int = [0, 1]) # [16, 20]
                
                # x is a Tensor with following elements:
                #    [[True, True, True, True]
                #     [False, False, False, False]]
                # Each example is followed by the corresponding output tensor.
                x = paddle.to_tensor([[True, True, True, True],
                                      [False, False, False, False]])
                out7 = paddle.sum(x)  # [4]
                out8 = paddle.sum(x, axis: int = 0)  # [1, 1, 1, 1]
                out9 = paddle.sum(x, axis: int = 1)  # [4, 0]        """
        pass


    def t(self, name: Optional[str] = None) -> Tuple[]:
        """
        Transpose <=2-D tensor.
        0-D and 1-D tensors are returned as it is and 2-D tensor is equal to
        the paddle.transpose function which perm dimensions set 0 and 1.
        
        Args:
            input (Tensor): The input Tensor. It is a N-D (N<=2) Tensor of data types float32, float64, int32, int64.
            name(str, optional): The default value is None.  Normally there is no need for
                user to set this property.  For more information, please refer to :ref:`api_guide_Name`
        Returns:
            Tensor: A transposed n-D Tensor, with data type being float16, float32, float64, int32, int64.
        
        Examples:
        
            .. code-block:: python
               :name: code-example
                 import paddle
                 
                 # Example 1 (0-D tensor)
                 x = paddle.to_tensor([0.79])
                 paddle.t(x) # [0.79]
                 
                 # Example 2 (1-D tensor)
                 x = paddle.to_tensor([0.79, 0.84, 0.32])
                 paddle.t(x) # [0.79000002, 0.83999997, 0.31999999]
                 paddle.t(x).shape # [3]
        
                 # Example 3 (2-D tensor)
                 x = paddle.to_tensor([[0.79, 0.84, 0.32],
                                      [0.64, 0.14, 0.57]])
                 x.shape # [2, 3]
                 paddle.t(x)
                 # [[0.79000002, 0.63999999],
                 #  [0.83999997, 0.14000000],
                 #  [0.31999999, 0.56999999]]
                 paddle.t(x).shape # [3, 2]        """
        pass


    def take_along_axis(self, indices, axis) -> Tensor:
        """
        Take values from the input array by given indices matrix along the designated axis.
        
        Args:
            arr (Tensor) : The input Tensor. Supported data types are float32 and float64.
            indices (Tensor) : Indices to take along each 1d slice of arr. This must match the dimension of arr,
                and need to broadcast against arr. Supported data type are int and int64.
            axis (int) : The axis to take 1d slices along. 
        
        Returns: 
            Tensor: The indexed element, same dtype with arr
        
        Examples:
            .. code-block:: python
        
                import paddle
                import numpy as np
        
                x_np = np.array([[1, 2, 3], [4, 5, 6], [7,8,9]])
                index_np = np.array([[0]])
                x = paddle.to_tensor(x_np)
                index = paddle.to_tensor(index_np)
                axis = 0
                result = paddle.take_along_axis(x, index, axis)
                print(result)
                # [[1, 2, 3]]        """
        pass


    def tanh(self, name: Optional[str] = None) -> Tensor:
        """
        Tanh Activation Operator.
        
        .. math::
            out = \\frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}
        
        Args:
            self (Tensor): Input of Tanh operator, an N-D Tensor, with data type float32, float64 or float16.
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            Output of Tanh operator, a Tensor with same data type and shape as input.
        
        Examples:
        
            .. code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
                out = paddle.tanh(x)
                print(out)
                # [-0.37994896 -0.19737532  0.09966799  0.29131261]        """
        pass


    def tanh_(self, name: Optional[str] = None) -> None:
        """
        Inplace version of ``tanh`` API, the output Tensor will be inplaced with input ``x``.
        Please refer to :ref:`api_tensor_tanh`.        """
        pass


    def tensordot(self, y: Tensor, axes=2, name: Optional[str] = None) -> Tuple[]:
        """
        This function computes a contraction, which sum the product of elements from two tensors along the given axes. 
        
        Args:
            self (Tensor): The left tensor for contraction with data type ``float32`` or ``float64``.
            y (Tensor): The right tensor for contraction with the same data type as ``x``.
            axes (int|tuple|list|Tensor, optional):  The axes to contract for ``x`` and ``y``, defaulted to integer ``2``.
        
                1. It could be a non-negative integer ``n``, 
                   in which the function will sum over the last ``n`` axes of ``x`` and the first ``n`` axes of ``y`` in order.
            
                2. It could be a 1-d tuple or list with data type ``int``, in which ``x`` and ``y`` will be contracted along the same given axes. 
                   For example, ``axes`` =[0, 1] applies contraction along the first two axes for ``x`` and the first two axes for ``y``.
            
                3. It could be a tuple or list containing one or two 1-d tuple|list|Tensor with data type ``int``. 
                   When containing one tuple|list|Tensor, the data in tuple|list|Tensor specified the same axes for ``x`` and ``y`` to contract. 
                   When containing two tuple|list|Tensor, the first will be applied to ``x`` and the second to ``y``. 
                   When containing more than two tuple|list|Tensor, only the first two axis sequences will be used while the others will be ignored.
            
                4. It could be a tensor, in which the ``axes`` tensor will be translated to a python list 
                   and applied the same rules described above to determine the contraction axes. 
                   Note that the ``axes`` with Tensor type is ONLY available in Dygraph mode.
            name(str, optional): The default value is None.  Normally there is no need for user to set this property. 
                                 For more information, please refer to :ref:`api_guide_Name` .
        
        Return: 
            Output (Tensor): The contraction result with the same data type as ``x`` and ``y``. 
            In general, :math:`output.ndim = x.ndim + y.ndim - 2 \times n_{axes}`, where :math:`n_{axes}` denotes the number of axes to be contracted.
        
        NOTES:
            1. This function supports tensor broadcast, 
               the size in the corresponding dimensions of ``x`` and ``y`` should be equal, or applies to the broadcast rules.
            2. This function also supports axes expansion, 
               when the two given axis sequences for ``x`` and ``y`` are of different lengths, 
               the shorter sequence will expand the same axes as the longer one at the end. 
               For example, if ``axes`` =[[0, 1, 2, 3], [1, 0]], 
               the axis sequence for ``x`` is [0, 1, 2, 3], 
               while the corresponding axis sequences for ``y`` will be expanded from [1, 0] to [1, 0, 2, 3].
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                data_type = 'float64'
        
                # For two 2-d tensor x and y, the case axes=0 is equivalent to outer product.
                # Note that tensordot supports empty axis sequence, so all the axes=0, axes=[], axes=[[]], and axes=[[],[]] are equivalent cases.   
                x = paddle.arange(4, dtype=data_type).reshape([2, 2])
                y = paddle.arange(4, dtype=data_type).reshape([2, 2])
                z = paddle.tensordot(x, y, axes=0)
                # z = [[[[0., 0.],
                #        [0., 0.]],
                #
                #       [[0., 1.],
                #        [2., 3.]]],
                #
                #
                #      [[[0., 2.],
                #        [4., 6.]],
                #
                #       [[0., 3.],
                #        [6., 9.]]]]
        
        
                # For two 1-d tensor x and y, the case axes=1 is equivalent to inner product.
                x = paddle.arange(10, dtype=data_type)
                y = paddle.arange(10, dtype=data_type)
                z1 = paddle.tensordot(x, y, axes=1)
                z2 = paddle.dot(x, y)
                # z1 = z2 = [285.]
        
        
                # For two 2-d tensor x and y, the case axes=1 is equivalent to matrix multiplication.
                x = paddle.arange(6, dtype=data_type).reshape([2, 3])
                y = paddle.arange(12, dtype=data_type).reshape([3, 4])
                z1 = paddle.tensordot(x, y, axes=1)
                z2 = paddle.matmul(x, y)
                # z1 = z2 =  [[20., 23., 26., 29.],
                #             [56., 68., 80., 92.]]
        
        
                # When axes is a 1-d int list, x and y will be contracted along the same given axes.
                # Note that axes=[1, 2] is equivalent to axes=[[1, 2]], axes=[[1, 2], []], axes=[[1, 2], [1]], and axes=[[1, 2], [1, 2]].
                x = paddle.arange(24, dtype=data_type).reshape([2, 3, 4])
                y = paddle.arange(36, dtype=data_type).reshape([3, 3, 4])
                z = paddle.tensordot(x, y, axes=[1, 2])
                # z =  [[506. , 1298., 2090.],
                #       [1298., 3818., 6338.]]
        
        
                # When axes is a list containing two 1-d int list, the first will be applied to x and the second to y.
                x = paddle.arange(60, dtype=data_type).reshape([3, 4, 5])
                y = paddle.arange(24, dtype=data_type).reshape([4, 3, 2])
                z = paddle.tensordot(x, y, axes=([1, 0], [0, 1]))
                # z =  [[4400., 4730.],
                #       [4532., 4874.],
                #       [4664., 5018.],
                #       [4796., 5162.],
                #       [4928., 5306.]]
        
        
                # Thanks to the support of axes expansion, axes=[[0, 1, 3, 4], [1, 0, 3, 4]] can be abbreviated as axes= [[0, 1, 3, 4], [1, 0]].
                x = paddle.arange(720, dtype=data_type).reshape([2, 3, 4, 5, 6])
                y = paddle.arange(720, dtype=data_type).reshape([3, 2, 4, 5, 6])
                z = paddle.tensordot(x, y, axes=[[0, 1, 3, 4], [1, 0]])
                # z = [[23217330., 24915630., 26613930., 28312230.],
                #      [24915630., 26775930., 28636230., 30496530.],
                #      [26613930., 28636230., 30658530., 32680830.],
                #      [28312230., 30496530., 32680830., 34865130.]]         """
        pass


    def tile(self, repeat_times, name: Optional[str] = None) -> Tensor:
        """
        Construct a new Tensor by repeating ``x`` the number of times given by ``repeat_times``.
        After tiling, the value of the i'th dimension of the output is equal to ``x.shape[i]*repeat_times[i]``.
        
        Both the number of dimensions of ``x`` and the number of elements in ``repeat_times`` should be less than or equal to 6.
        
        Args:
            self (Tensor): The input tensor, its data type should be bool, float32, float64, int32 or int64.
            repeat_times (list|tuple|Tensor): The number of repeating times. If repeat_times is a list or tuple, all its elements
                should be integers or 1-D Tensors with the data type int32. If repeat_times is a Tensor, it should be an 1-D Tensor with the data type int32.
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            N-D Tensor. The data type is the same as ``x``. The size of the i-th dimension is equal to ``x[i] * repeat_times[i]``.
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                data = paddle.to_tensor([1, 2, 3], dtype='int32')
                out = paddle.tile(data, repeat_times=[2, 1])
                np_out = out.numpy()
                # [[1, 2, 3]
                #  [1, 2, 3]]
        
                out = paddle.tile(data, repeat_times=(2, 2))
                np_out = out.numpy()
                # [[1, 2, 3, 1, 2, 3]
                #  [1, 2, 3, 1, 2, 3]]
        
                repeat_times = paddle.to_tensor([1, 2], dtype='int32')
                out = paddle.tile(data, repeat_times=repeat_times)
                np_out = out.numpy()
                # [[1, 2, 3, 1, 2, 3]]        """
        pass


    def to_dense(self) -> Tensor:
        pass
    def to_sparse_coo(self, sparse_dim) -> Tensor:
        pass
    def tolist(self) -> Tuple[]:
        """
        **Notes**:
            **This API is ONLY available in Dygraph mode**
        
        This function translate the paddle.Tensor to python list.
        
        Args:
            self (Tensor): ``x`` is the Tensor we want to translate to list
        
        Returns:
            list: A list that contain the same value of current Tensor.
        
        Returns type:
            list: dtype is same as current Tensor
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                t = paddle.to_tensor([0,1,2,3,4])
                expectlist = t.tolist()
                print(expectlist)   #[0, 1, 2, 3, 4]
        
                expectlist = paddle.tolist(t)
                print(expectlist)   #[0, 1, 2, 3, 4]        """
        pass


    def topk(self, k, axis: Optional[int] = None, largest=True, sorted=True, name: Optional[str] = None) -> Tuple[Tensor, Tensor]:
        """
        This OP is used to find values and indices of the k largest or smallest at the optional axis.
        If the input is a 1-D Tensor, finds the k largest or smallest values and indices.
        If the input is a Tensor with higher rank, this operator computes the top k values and indices along the :attr:`axis`.
        
        Args:
            self (Tensor): Tensor, an input N-D Tensor with type float32, float64, int32, int64.
            k(int, Tensor): The number of top elements to look for along the axis.
            axis(int, optional): Axis to compute indices along. The effective range
                is [-R, R), where R is x.ndim. when axis < 0, it works the same way
                as axis + R. Default is -1.
            largest(bool, optional) : largest is a flag, if set to true,
                algorithm will sort by descending order, otherwise sort by
                ascending order. Default is True.
            sorted(bool, optional): controls whether to return the elements in sorted order, default value is True. In gpu device, it always return the sorted value. 
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            tuple(Tensor), return the values and indices. The value data type is the same as the input `x`. The indices data type is int64.
        
        Examples:
        
            .. code-block:: python
        
               import paddle
        
               tensor_1 = paddle.to_tensor([1, 4, 5, 7])
               value_1, indices_1 = paddle.topk(tensor_1, k=1)
               print(value_1)
               # [7]
               print(indices_1)
               # [3] 
               tensor_2 = paddle.to_tensor([[1, 4, 5, 7], [2, 6, 2, 5]])
               value_2, indices_2 = paddle.topk(tensor_2, k=1)
               print(value_2)
               # [[7]
               #  [6]]
               print(indices_2)
               # [[3]
               #  [1]]
               value_3, indices_3 = paddle.topk(tensor_2, k=1, axis: int = -1)
               print(value_3)
               # [[7]
               #  [6]]
               print(indices_3)
               # [[3]
               #  [1]]
               value_4, indices_4 = paddle.topk(tensor_2, k=1, axis: int = 0)
               print(value_4)
               # [[2 6 5 7]]
               print(indices_4)
               # [[1 1 0 0]]        """
        pass


    def trace(self, offset=0, axis1=0, axis2=1, name: Optional[str] = None) -> Tensor:
        """
        **trace**
        
        This OP computes the sum along diagonals of the input tensor x.
        
        If ``x`` is 2D, returns the sum of diagonal.
        
        If ``x`` has larger dimensions, then returns an tensor of diagonals sum, diagonals be taken from
        the 2D planes specified by axis1 and axis2. By default, the 2D planes formed by the first and second axes
        of the input tensor x.
        
        The argument ``offset`` determines where diagonals are taken from input tensor x:
        
        - If offset = 0, it is the main diagonal.
        - If offset > 0, it is above the main diagonal.
        - If offset < 0, it is below the main diagonal.
        - Note that if offset is out of input's shape indicated by axis1 and axis2, 0 will be returned.
        
        Args:
            self (Tensor): The input tensor x. Must be at least 2-dimensional. The input data type should be float32, float64, int32, int64.
            offset(int, optional): Which diagonals in input tensor x will be taken. Default: 0 (main diagonals).
            axis1(int, optional): The first axis with respect to take diagonal. Default: 0.
            axis2(int, optional): The second axis with respect to take diagonal. Default: 1.
            name (str, optional): Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`. Default: None.
        
        Returns:
            Tensor: the output data type is the same as input data type.
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                case1 = paddle.randn([2, 3])
                case2 = paddle.randn([3, 10, 10])
                case3 = paddle.randn([3, 10, 5, 10])
                data1 = paddle.trace(case1) # data1.shape = [1]
                data2 = paddle.trace(case2, offset=1, axis1=1, axis2=2) # data2.shape = [3]
                data3 = paddle.trace(case3, offset=-3, axis1=1, axis2=-1) # data2.shape = [3, 5]        """
        pass


    def transpose(self, perm, name: Optional[str] = None) -> Tensor:
        """
        Permute the data dimensions of `input` according to `perm`.
        
        The `i`-th dimension  of the returned tensor will correspond to the
        perm[i]-th dimension of `input`.
        
        Args:
            self (Tensor): The input Tensor. It is a N-D Tensor of data types bool, float32, float64, int32.
            perm (list|tuple): Permute the input according to the data of perm.
            name (str): The name of this layer. It is optional.
        
        Returns:
            Tensor: A transposed n-D Tensor, with data type being bool, float32, float64, int32, int64.
        
        For Example:
        
            .. code-block:: text
        
             x = [[[ 1  2  3  4] [ 5  6  7  8] [ 9 10 11 12]]
                 [[13 14 15 16] [17 18 19 20] [21 22 23 24]]]
             shape(x) =  [2,3,4]
        
             # Example 1
             perm0 = [1,0,2]
             y_perm0 = [[[ 1  2  3  4] [13 14 15 16]]
                       [[ 5  6  7  8]  [17 18 19 20]]
                       [[ 9 10 11 12]  [21 22 23 24]]]
             shape(y_perm0) = [3,2,4]
        
             # Example 2
             perm1 = [2,1,0]
             y_perm1 = [[[ 1 13] [ 5 17] [ 9 21]]
                       [[ 2 14] [ 6 18] [10 22]]
                       [[ 3 15]  [ 7 19]  [11 23]]
                       [[ 4 16]  [ 8 20]  [12 24]]]
             shape(y_perm1) = [4,3,2]
        
        Examples:
        
            .. code-block:: python
        
                import paddle
        
                x = paddle.randn([2, 3, 4])
                x_transposed = paddle.transpose(x, perm=[1, 0, 2])
                print(x_transposed.shape)
                # [3L, 2L, 4L]        """
        pass


    def trunc(self, name: Optional[str] = None) -> Tensor:
        """
        This API is used to returns a new tensor with the truncated integer values of input.
        
        Args:
            input (Tensor): The input tensor, it's data type should be int32, int64, float32, float64.
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            Tensor: The output Tensor of trunc.
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                input = paddle.rand([2,2],'float32')
                print(input)
                # Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
                #         [[0.02331470, 0.42374918],
                #         [0.79647720, 0.74970269]])
        
                output = paddle.trunc(input)
                print(output)
                # Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
                #         [[0., 0.],
                #         [0., 0.]]))        """
        pass


    def unbind(self, axis: int = 0) -> List[Tensor]:
        """
        Removes a tensor dimension, then split the input tensor into multiple sub-Tensors.
        
        Args:
            input (Tensor): The input variable which is an N-D Tensor, data type being float32, float64, int32 or int64.
            axis (int32|int64, optional): A scalar with type ``int32|int64`` shape [1]. The dimension along which to unbind. 
                If :math:`axis < 0`, the dimension to unbind along is :math:`rank(input) + axis`. Default is 0.
        Returns:
            list(Tensor): The list of segmented Tensor variables.
        
        Example:
            .. code-block:: python
        
                import paddle
                import numpy as np
                # input is a variable which shape is [3, 4, 5]
                np_input = np.random.rand(3, 4, 5).astype('float32')
                input = paddle.to_tensor(np_input)
                [x0, x1, x2] = paddle.unbind(input, axis: int = 0)
                # x0.shape [4, 5]
                # x1.shape [4, 5]
                # x2.shape [4, 5]
                [x0, x1, x2, x3] = paddle.unbind(input, axis: int = 1)
                # x0.shape [3, 5]
                # x1.shape [3, 5]
                # x2.shape [3, 5]
                # x3.shape [3, 5]        """
        pass


    def uniform_(self, min=-1.0, max=1.0, seed=0, name: Optional[str] = None) -> None:
        """
        This is the inplace version of OP ``uniform``, which returns a Tensor filled 
        with random values sampled from a uniform distribution. The output Tensor will
        be inplaced with input ``x``. Please refer to :ref:`api_tensor_uniform`.
        
        Args:
            self (Tensor): The input tensor to be filled with random values.
            min(float|int, optional): The lower bound on the range of random values
                to generate, ``min`` is included in the range. Default is -1.0.
            max(float|int, optional): The upper bound on the range of random values
                to generate, ``max`` is excluded in the range. Default is 1.0.
            seed(int, optional): Random seed used for generating samples. If seed is 0, 
                it will use the seed of the global default generator (which can be set by paddle.seed). 
                Note that if seed is not 0, this operator will always generate the same random numbers every
                time. Default is 0.
            name(str, optional): The default value is None. Normally there is no
                need for user to set this property. For more information, please
                refer to :ref:`api_guide_Name`.
        Returns:
            Tensor: The input tensor x filled with random values sampled from a uniform
            distribution in the range [``min``, ``max``).
        Examples:
            .. code-block:: python
                
                import paddle
                # example:
                x = paddle.ones(shape=[3, 4])
                x.uniform_()
                print(x)
                # [[ 0.84524226,  0.6921872,   0.56528175,  0.71690357], # random
                #  [-0.34646994, -0.45116323, -0.09902662, -0.11397249], # random
                #  [ 0.433519,    0.39483607, -0.8660099,   0.83664286]] # random        """
        pass


    def unique(self, return_index=False, return_inverse=False, return_counts=False, axis: Optional[int] = None, dtype='int64', name: Optional[str] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Returns the unique elements of `x` in ascending order.
        
        Args:
            self (Tensor): The input tensor, it's data type should be float32, float64, int32, int64.
            return_index(bool, optional): If True, also return the indices of the input tensor that
                result in the unique Tensor.
            return_inverse(bool, optional): If True, also return the indices for where elements in
                the original input ended up in the returned unique tensor.
            return_counts(bool, optional): If True, also return the counts for each unique element.
            axis(int, optional): The axis to apply unique. If None, the input will be flattened.
                Default: None.
            dtype(np.dtype|str, optional): The date type of `indices` or `inverse` tensor: int32 or int64.
                Default: int64.
            name(str, optional): Name for the operation. For more information, please refer to
                :ref:`api_guide_Name`. Default: None.
        
        Returns: 
            tuple: (out, indices, inverse, counts). `out` is the unique tensor for `x`. `indices` is \
                provided only if `return_index` is True. `inverse` is provided only if `return_inverse` \
                is True. `counts` is provided only if `return_counts` is True.
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([2, 3, 3, 1, 5, 3])
                unique = paddle.unique(x)
                np_unique = unique.numpy() # [1 2 3 5]
                _, indices, inverse, counts = paddle.unique(x, return_index=True, return_inverse=True, return_counts=True)
                np_indices = indices.numpy() # [3 0 1 4]
                np_inverse = inverse.numpy() # [1 2 2 0 3 2]
                np_counts = counts.numpy() # [1 1 3 1]
        
                x = paddle.to_tensor([[2, 1, 3], [3, 0, 1], [2, 1, 3]])
                unique = paddle.unique(x)
                np_unique = unique.numpy() # [0 1 2 3]
        
                unique = paddle.unique(x, axis: int = 0)
                np_unique = unique.numpy() 
                # [[2 1 3]
                #  [3 0 1]]        """
        pass


    def unique_consecutive(self, return_inverse=False, return_counts=False, axis: Optional[int] = None, dtype='int64', name: Optional[str] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Eliminates all but the first element from every consecutive group of equivalent elements.
        
        .. note:: This function is different from :func:`paddle.unique` in the sense that this function
            only eliminates consecutive duplicate values. This semantics is similar to `std::unique` in C++.
        
        Args:
            self (Tensor): the input tensor, it's data type should be float32, float64, int32, int64.
            return_inverse(bool, optional): If True, also return the indices for where elements in
                the original input ended up in the returned unique consecutive tensor. Default is False.
            return_counts(bool, optional): If True, also return the counts for each unique consecutive element.
                Default is False.
            axis(int, optional): The axis to apply unique consecutive. If None, the input will be flattened.
                Default is None.
            dtype(np.dtype|str, optional): The data type `inverse` tensor: int32 or int64.
                Default: int64.
            name(str, optional): Name for the operation. For more information, please refer to
                :ref:`api_guide_Name`. Default is None.
        
        Returns:
            tuple: (out, inverse, counts). `out` is the unique consecutive tensor for `x`. `inverse` is provided only if `return_inverse` is True. `counts` is provided only if `return_counts` is True.
        
        Example:
            .. code-block:: python
        
                import paddle 
        
                x = paddle.to_tensor([1, 1, 2, 2, 3, 1, 1, 2])
                output = paddle.unique_consecutive(x) # 
                np_output = output.numpy() # [1 2 3 1 2]
                _, inverse, counts = paddle.unique_consecutive(x, return_inverse=True, return_counts=True)
                np_inverse = inverse.numpy() # [0 0 1 1 2 3 3 4]
                np_counts = inverse.numpy() # [2 2 1 2 1]
        
                x = paddle.to_tensor([[2, 1, 3], [3, 0, 1], [2, 1, 3], [2, 1, 3]])
                output = paddle.unique_consecutive(x, axis: int = 0) # 
                np_output = output.numpy() # [2 1 3 0 1 2 1 3 2 1 3]
        
                x = paddle.to_tensor([[2, 1, 3], [3, 0, 1], [2, 1, 3], [2, 1, 3]])
                output = paddle.unique_consecutive(x, axis: int = 0) # 
                np_output = output.numpy()
                # [[2 1 3]
                #  [3 0 1]
                #  [2 1 3]]        """
        pass


    def unsqueeze(self, axis, name: Optional[str] = None) -> Tuple[Tensor, Tensor]:
        """
        Insert single-dimensional entries to the shape of input Tensor ``x``. Takes one
        required argument axis, a dimension or list of dimensions that will be inserted.
        Dimension indices in axis are as seen in the output tensor.
        
        Note that the output Tensor will share data with origin Tensor and doesn't have a 
        Tensor copy in ``dygraph`` mode. If you want to use the Tensor copy version, 
        please use `Tensor.clone` like ``unsqueeze_clone_x = x.unsqueeze(-1).clone()``.
        
        Args:
            self (Tensor): The input Tensor to be unsqueezed. Supported data type: float32, float64, bool, int8, int32, int64.
            axis (int|list|tuple|Tensor): Indicates the dimensions to be inserted. The data type is ``int32`` . 
                                        If ``axis`` is a list or tuple, the elements of it should be integers or Tensors with shape [1]. 
                                        If ``axis`` is a Tensor, it should be an 1-D Tensor .
                                        If ``axis`` is negative, ``axis = axis + ndim(x) + 1``.
            name (str|None): Name for this layer. Please refer to :ref:`api_guide_Name`, Default None.
        
        Returns:
            Tensor: Unsqueezed Tensor with the same data type as input Tensor.
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.rand([5, 10])
                print(x.shape)  # [5, 10]
                
                out1 = paddle.unsqueeze(x, axis: int = 0)
                print(out1.shape)  # [1, 5, 10]
                
                out2 = paddle.unsqueeze(x, axis: int = [0, 2]) 
                print(out2.shape)  # [1, 5, 1, 10]
        
                axis = paddle.to_tensor([0, 1, 2])
                out3 = paddle.unsqueeze(x, axis: int = axis) 
                print(out3.shape)  # [1, 1, 1, 5, 10]
        
                # out1, out2, out3 share data with x in dygraph mode
                x[0, 0] = 10.
                print(out1[0, 0, 0]) # [10.]
                print(out2[0, 0, 0, 0]) # [10.]
                print(out3[0, 0, 0, 0, 0]) # [10.]
                        """
        pass


    def unsqueeze_(self, axis, name: Optional[str] = None) -> None:
        """
        Inplace version of ``unsqueeze`` API, the output Tensor will be inplaced with input ``x``.
        Please refer to :ref:`api_paddle_tensor_unsqueeze`.        """
        pass


    def unstack(self, axis: int = 0, num=None) -> List[Tensor]:
        """
        :alias_main: paddle.unstack
            :alias: paddle.unstack,paddle.tensor.unstack,paddle.tensor.manipulation.unstack
            :old_api: paddle.fluid.layers.unstack
        
        **UnStack Layer**
        
        This layer unstacks input Tensor :code:`x` into several Tensors along :code:`axis`.
        
        If :code:`axis` < 0, it would be replaced with :code:`axis+rank(x)`.
        If :code:`num` is None, it would be inferred from :code:`x.shape[axis]`,
        and if :code:`x.shape[axis]` <= 0 or is unknown, :code:`ValueError` is
        raised.
        
        Args:
            self (Tensor): Input Tensor. It is a N-D Tensors of data types float32, float64, int32, int64.
            axis (int): The axis along which the input is unstacked.
            num (int|None): The number of output variables.
        
        Returns:
            list(Tensor): The unstacked Tensors list. The list elements are N-D Tensors of data types float32, float64, int32, int64.
        
        Raises:
            ValueError: If x.shape[axis] <= 0 or axis is not in range [-D, D).
        
        Examples:
            .. code-block:: python
        
                import paddle
                x = paddle.ones(name='x', shape=[2, 3, 5], dtype='float32')  # create a tensor with shape=[2, 3, 5]
                y = paddle.unstack(x, axis: int = 1)  # unstack with second axis, which results 3 tensors with shape=[2, 5]        """
        pass


    def values(self) -> Any:
        pass
    def var(self, axis: Optional[int] = None, unbiased=True, keepdim: bool = False, name: Optional[str] = None) -> Tuple[]:
        """
        Computes the variance of ``x`` along ``axis`` .
        
        Args:
            self (Tensor): The input Tensor with data type float32, float64.
            axis (int|list|tuple, optional): The axis along which to perform
                variance calculations. ``axis`` should be int, list(int) or
                tuple(int). If ``axis`` is a list/tuple of dimension(s), variance
                is calculated along all element(s) of ``axis`` . ``axis`` or
                element(s) of ``axis`` should be in range [-D, D), where D is the
                dimensions of ``x`` . If ``axis`` or element(s) of ``axis`` is less
                than 0, it works the same way as :math:`axis + D` . If ``axis`` is
                None, variance is calculated over all elements of ``x``. Default
                is None.
            unbiased (bool, optional): Whether to use the unbiased estimation. If
                ``unbiased`` is True, the divisor used in the computation is
                :math:`N - 1`, where :math:`N` represents the number of elements
                along ``axis`` , otherwise the divisor is :math:`N`. Default is True.
            keepdim (bool, optional): Whether to reserve the reduced dimension(s)
                in the output Tensor. If ``keepdim`` is True, the dimensions of
                the output Tensor is the same as ``x`` except in the reduced
                dimensions(it is of size 1 in this case). Otherwise, the shape of
                the output Tensor is squeezed in ``axis`` . Default is False.
            name (str, optional): Name for the operation (optional, default is None).
                For more information, please refer to :ref:`api_guide_Name`.
        
        Returns:
            Tensor, results of variance along ``axis`` of ``x``, with the same data
            type as ``x``.
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                x = paddle.to_tensor([[1.0, 2.0, 3.0], [1.0, 4.0, 5.0]])
                out1 = paddle.var(x)
                # [2.66666667]
                out2 = paddle.var(x, axis: int = 1)
                # [1.         4.33333333]        """
        pass


    def where(self, x=None, y=None, name: Optional[str] = None) -> Tensor:
        """
        Return a tensor of elements selected from either $x$ or $y$, depending on $condition$.
        
        **Note**:
            ``paddle.where(condition)`` is identical to ``paddle.nonzero(condition, as_tuple=True)``.
        
        .. math::
        
          out_i =
          \begin{cases}
          x_i, \quad  \text{if}  \ condition_i \  is \ True \\
          y_i, \quad  \text{if}  \ condition_i \  is \ False \\
          \end{cases}
        
        
        Args:
            condition(Tensor): The condition to choose x or y. When True(nonzero), yield x, otherwise yield y.
            x(Tensor or Scalar, optional): x is a Tensor or Scalar with data type float32, float64, int32, int64. Either both or neither of x and y should be given.
            y(Tensor or Scalar, optional): y is a Tensor or Scalar with data type float32, float64, int32, int64. Either both or neither of x and y should be given.
        
            name(str, optional): The default value is None. Normally there is no
                need for user to set this property. For more information, please
                refer to :ref:`api_guide_Name`.
        
        Returns:
            Tensor: A Tensor with the same data dype as x. 
        
        Examples:
            .. code-block:: python
        
              import paddle
        
              x = paddle.to_tensor([0.9383, 0.1983, 3.2, 1.2])
              y = paddle.to_tensor([1.0, 1.0, 1.0, 1.0])
              out = paddle.where(x>1, x, y)
        
              print(out)
              #out: [1.0, 1.0, 3.2, 1.2]
        
              out = paddle.where(x>1)
              print(out)
              #out: (Tensor(shape=[2, 1], dtype=int64, place=CPUPlace, stop_gradient=True,
              #            [[2],
              #             [3]]),)        """
        pass


    def zero_(self) -> None:
        """
        **Notes**:
            **This API is ONLY available in Dygraph mode**
        
        This function fill the Tensor with zero inplace.
        
        Args:
            self (Tensor): ``x`` is the Tensor we want to filled with zero inplace
        
        Returns:
            self (Tensor): Tensor x filled with zero inplace
        
        Examples:
            .. code-block:: python
        
                import paddle
        
                tensor = paddle.to_tensor([0, 1, 2, 3, 4])
        
                tensor.zero_()
                print(tensor.tolist())   #[0, 0, 0, 0, 0]        """
        pass


