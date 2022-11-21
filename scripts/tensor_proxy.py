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
        pass
    def acos(self, name: Optional[str] = None) -> Tensor:
        pass
    def acosh(self, name: Optional[str] = None) -> Tensor:
        pass
    def add(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        pass
    def add_(self, y: Tensor, name: Optional[str] = None) -> None:
        pass
    def add_n(self, name: Optional[str] = None) -> Tensor:
        pass
    def addmm(self, x, y, beta=1.0, alpha=1.0, name: Optional[str] = None) -> Tensor:
        pass
    def all(self, axis: Optional[int] = None, keepdim: bool = False, name: Optional[str] = None) -> Tensor:
        pass
    def allclose(self, y: Tensor, rtol=1e-05, atol=1e-08, equal_nan=False, name: Optional[str] = None) -> Tensor:
        pass
    def amax(self, axis: Optional[int] = None, keepdim: bool = False, name: Optional[str] = None) -> Tensor:
        pass
    def amin(self, axis: Optional[int] = None, keepdim: bool = False, name: Optional[str] = None) -> Tensor:
        pass
    def angle(self, name: Optional[str] = None) -> Tensor:
        pass
    def any(self, axis: Optional[int] = None, keepdim: bool = False, name: Optional[str] = None) -> Tensor:
        pass
    def argmax(self, axis: Optional[int] = None, keepdim: bool = False, dtype='int64', name: Optional[str] = None) -> Tensor:
        pass
    def argmin(self, axis: Optional[int] = None, keepdim: bool = False, dtype='int64', name: Optional[str] = None) -> Tensor:
        pass
    def argsort(self, axis: int = -1, descending=False, name: Optional[str] = None) -> Tuple[Tensor, Tensor]:
        pass
    def as_complex(self, name: Optional[str] = None) -> Tensor:
        pass
    def as_real(self, name: Optional[str] = None) -> Tensor:
        pass
    def asin(self, name: Optional[str] = None) -> Tensor:
        pass
    def asinh(self, name: Optional[str] = None) -> Tensor:
        pass
    def astype(self, dtype) -> Tensor:
        pass
    def atan(self, name: Optional[str] = None) -> Tensor:
        pass
    def atanh(self, name: Optional[str] = None) -> Tensor:
        pass
    def backward(self, grad_tensor=None, retain_graph=False) -> Tuple[]:
        pass
    def bincount(self, weights=None, minlength=0, name: Optional[str] = None) -> Tensor:
        pass
    def bitwise_and(self, y: Tensor, out=None, name: Optional[str] = None) -> Tensor:
        pass
    def bitwise_not(self, out=None, name: Optional[str] = None) -> Tensor:
        pass
    def bitwise_or(self, y: Tensor, out=None, name: Optional[str] = None) -> Tensor:
        pass
    def bitwise_xor(self, y: Tensor, out=None, name: Optional[str] = None) -> Tensor:
        pass
    def bmm(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        pass
    def broadcast_shape(self, y_shape) -> Tuple[int]:
        pass
    def broadcast_tensors(self, name: Optional[str] = None) -> List[Tensor]:
        pass
    def broadcast_to(self, shape, name: Optional[str] = None) -> Tensor:
        pass
    def cast(self, dtype) -> Tensor:
        pass
    def ceil(self, name: Optional[str] = None) -> Tensor:
        pass
    def ceil_(self, name: Optional[str] = None) -> None:
        pass
    def cholesky(self, upper=False, name: Optional[str] = None) -> Tensor:
        pass
    def cholesky_solve(self, y: Tensor, upper=False, name: Optional[str] = None) -> Tensor:
        pass
    def chunk(self, chunks, axis: int = 0, name: Optional[str] = None) -> Tuple[]:
        pass
    def clear_grad(self) -> Tuple[]:
        pass
    def clip(self, min=None, max=None, name: Optional[str] = None) -> Tensor:
        pass
    def clip_(self, min=None, max=None, name: Optional[str] = None) -> None:
        pass
    def concat(self, axis: int = 0, name: Optional[str] = None) -> Tensor:
        pass
    def cond(self, p=None, name: Optional[str] = None) -> None:
        pass
    def conj(self, name: Optional[str] = None) -> Tensor:
        pass
    def cos(self, name: Optional[str] = None) -> Tensor:
        pass
    def cosh(self, name: Optional[str] = None) -> Tensor:
        pass
    def cov(self, rowvar=True, ddof=True, fweights=None, aweights=None, name: Optional[str] = None) -> None:
        pass
    def cross(self, y: Tensor, axis: int = 9, name: Optional[str] = None) -> Tensor:
        pass
    def cumprod(self, dim=None, dtype=None, name: Optional[str] = None) -> Tensor:
        pass
    def cumsum(self, axis: Optional[int] = None, dtype=None, name: Optional[str] = None) -> Tensor:
        pass
    def deg2rad(self, name: Optional[str] = None) -> Tuple[]:
        pass
    def diagonal(self, offset=0, axis1=0, axis2=1, name: Optional[str] = None) -> Tensor:
        pass
    def diff(self, n=1, axis: int = -1, prepend=None, append=None, name: Optional[str] = None) -> Tensor:
        pass
    def digamma(self, name: Optional[str] = None) -> Tensor:
        pass
    def dim(self) -> Tuple[]:
        pass
    def dist(self, y: Tensor, p=2, name: Optional[str] = None) -> Tensor:
        pass
    def divide(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        pass
    def dot(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        pass
    def eig(self, name: Optional[str] = None) -> Tuple[Tensor, Tensor]:
        pass
    def eigvals(self, name: Optional[str] = None) -> Tensor:
        pass
    def eigvalsh(self, UPLO='L', name: Optional[str] = None) -> Tuple[Tensor, Tensor]:
        pass
    def equal(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        pass
    def equal_all(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        pass
    def erf(self, name: Optional[str] = None) -> Tensor:
        pass
    def erfinv(self, name: Optional[str] = None) -> Tensor:
        pass
    def erfinv_(self, name: Optional[str] = None) -> None:
        pass
    def exp(self, name: Optional[str] = None) -> Tensor:
        pass
    def exp_(self, name: Optional[str] = None) -> None:
        pass
    def expand(self, shape, name: Optional[str] = None) -> Tensor:
        pass
    def expand_as(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        pass
    def exponential_(self, lam=1.0, name: Optional[str] = None) -> Tensor:
        pass
    def fill_(self, value) -> None:
        pass
    def fill_diagonal_(self, value, offset=0, wrap=False, name: Optional[str] = None) -> None:
        pass
    def fill_diagonal_tensor(self, y: Tensor, offset=0, dim1=0, dim2=1, name: Optional[str] = None) -> Tensor:
        pass
    def fill_diagonal_tensor_(self, y: Tensor, offset=0, dim1=0, dim2=1, name: Optional[str] = None) -> None:
        pass
    def flatten(self, start_axis: int = 0, stop_axis: int = -1, name: Optional[str] = None) -> Tuple[Tensor, Tensor]:
        pass
    def flatten_(self, start_axis: int = 0, stop_axis: int = -1, name: Optional[str] = None) -> None:
        pass
    def flip(self, axis, name: Optional[str] = None) -> Tensor:
        pass
    def floor(self, name: Optional[str] = None) -> Tensor:
        pass
    def floor_(self, name: Optional[str] = None) -> None:
        pass
    def floor_divide(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        pass
    def floor_mod(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        pass
    def fmax(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        pass
    def fmin(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        pass
    def gather(self, index, axis: Optional[int] = None, name: Optional[str] = None) -> Tensor:
        pass
    def gather_nd(self, index, name: Optional[str] = None) -> Tensor:
        pass
    def gcd(self, y: Tensor, name: Optional[str] = None) -> Tuple[]:
        pass
    def gradient(self) -> Tuple[]:
        pass
    def greater_equal(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        pass
    def greater_than(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        pass
    def histogram(self, bins=100, min=0, max=0, name: Optional[str] = None) -> Tensor:
        pass
    def imag(self, name: Optional[str] = None) -> Tensor:
        pass
    def increment(self, value=1.0, name: Optional[str] = None) -> Tensor:
        pass
    def index_sample(self, index) -> Tensor:
        pass
    def index_select(self, index, axis: int = 0, name: Optional[str] = None) -> Tensor:
        pass
    def inner(self, y: Tensor, name: Optional[str] = None) -> Tuple[]:
        pass
    def inverse(self, name: Optional[str] = None) -> Tensor:
        pass
    def is_complex(self) -> bool:
        pass
    def is_empty(self, name: Optional[str] = None) -> Tensor:
        pass
    def is_floating_point(self) -> bool:
        pass
    def is_integer(self) -> bool:
        pass
    def is_tensor(self) -> bool:
        pass
    def isclose(self, y: Tensor, rtol=1e-05, atol=1e-08, equal_nan=False, name: Optional[str] = None) -> Tensor:
        pass
    def isfinite(self, name: Optional[str] = None) -> Tensor:
        pass
    def isinf(self, name: Optional[str] = None) -> Tensor:
        pass
    def isnan(self, name: Optional[str] = None) -> Tensor:
        pass
    def item(self, *args) -> Any:
        pass
    def kron(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        pass
    def kthvalue(self, k, axis: Optional[int] = None, keepdim: bool = False, name: Optional[str] = None) -> Tuple[Tensor, Tensor]:
        pass
    def lcm(self, y: Tensor, name: Optional[str] = None) -> Tuple[]:
        pass
    def lerp(self, y: Tensor, weight, name: Optional[str] = None) -> Tensor:
        pass
    def lerp_(self, y: Tensor, weight, name: Optional[str] = None) -> None:
        pass
    def less_equal(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        pass
    def less_than(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        pass
    def lgamma(self, name: Optional[str] = None) -> Tensor:
        pass
    def log(self, name: Optional[str] = None) -> Tensor:
        pass
    def log10(self, name: Optional[str] = None) -> Tensor:
        pass
    def log1p(self, name: Optional[str] = None) -> Tensor:
        pass
    def log2(self, name: Optional[str] = None) -> Tensor:
        pass
    def logical_and(self, y: Tensor, out=None, name: Optional[str] = None) -> Tensor:
        pass
    def logical_not(self, out=None, name: Optional[str] = None) -> Tensor:
        pass
    def logical_or(self, y: Tensor, out=None, name: Optional[str] = None) -> Tensor:
        pass
    def logical_xor(self, y: Tensor, out=None, name: Optional[str] = None) -> Tensor:
        pass
    def logit(self, eps=None, name: Optional[str] = None) -> Tensor:
        pass
    def logsumexp(self, axis: Optional[int] = None, keepdim: bool = False, name: Optional[str] = None) -> Tensor:
        pass
    def lstsq(self, y: Tensor, rcond=None, driver=None, name: Optional[str] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        pass
    def lu(self, pivot=True, get_infos=False, name: Optional[str] = None) -> Tuple[Tensor, Tensor, Tensor]:
        pass
    def lu_unpack(self, y: Tensor, unpack_ludata=True, unpack_pivots=True, name: Optional[str] = None) -> Tuple[Tensor, Tensor, Tensor]:
        pass
    def masked_select(self, mask, name: Optional[str] = None) -> Tensor:
        pass
    def matmul(self, y: Tensor, transpose_x=False, transpose_y=False, name: Optional[str] = None) -> Tensor:
        pass
    def matrix_power(self, n, name: Optional[str] = None) -> Tensor:
        pass
    def max(self, axis: Optional[int] = None, keepdim: bool = False, name: Optional[str] = None) -> Tensor:
        pass
    def maximum(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        pass
    def mean(self, axis: Optional[int] = None, keepdim: bool = False, name: Optional[str] = None) -> Tensor:
        pass
    def median(self, axis: Optional[int] = None, keepdim: bool = False, name: Optional[str] = None) -> Tensor:
        pass
    def min(self, axis: Optional[int] = None, keepdim: bool = False, name: Optional[str] = None) -> Tensor:
        pass
    def minimum(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        pass
    def mm(self, mat2, name: Optional[str] = None) -> Tensor:
        pass
    def mod(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        pass
    def mode(self, axis: int = -1, keepdim: bool = False, name: Optional[str] = None) -> Tuple[Tensor, Tensor]:
        pass
    def moveaxis(self, source, destination, name: Optional[str] = None) -> Tensor:
        pass
    def multi_dot(self, name: Optional[str] = None) -> Tensor:
        pass
    def multiplex(self, index, name: Optional[str] = None) -> Tensor:
        pass
    def multiply(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        pass
    def mv(self, vec, name: Optional[str] = None) -> Tensor:
        pass
    def nanmean(self, axis: Optional[int] = None, keepdim: bool = False, name: Optional[str] = None) -> Tensor:
        pass
    def nansum(self, axis: Optional[int] = None, dtype=None, keepdim: bool = False, name: Optional[str] = None) -> Tensor:
        pass
    def ndimension(self) -> int:
        pass
    def neg(self, name: Optional[str] = None) -> Tensor:
        pass
    def nonzero(self, as_tuple=False) -> Tensor:
        pass
    def norm(self, p='fro', axis: Optional[int] = None, keepdim: bool = False, name: Optional[str] = None) -> Tuple[Tensor, Tensor]:
        pass
    def not_equal(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        pass
    def numel(self, name: Optional[str] = None) -> int:
        pass
    def outer(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        pass
    def pow(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        pass
    def prod(self, axis: Optional[int] = None, keepdim: bool = False, dtype=None, name: Optional[str] = None) -> Tensor:
        pass
    def put_along_axis(self, indices, values, axis, reduce='assign') -> Tensor:
        pass
    def put_along_axis_(self, indices, values, axis, reduce='assign') -> None:
        pass
    def qr(self, mode='reduced', name: Optional[str] = None) -> Tuple[Tensor, Tensor]:
        pass
    def quantile(self, q, axis: Optional[int] = None, keepdim: bool = False) -> Tensor:
        pass
    def rad2deg(self, name: Optional[str] = None) -> Tensor:
        pass
    def rank(self) -> Tensor:
        pass
    def real(self, name: Optional[str] = None) -> Tensor:
        pass
    def reciprocal(self, name: Optional[str] = None) -> Tensor:
        pass
    def reciprocal_(self, name: Optional[str] = None) -> None:
        pass
    def register_hook(self, hook) -> Tuple[]:
        pass
    def remainder(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        pass
    def repeat_interleave(self, repeats, axis: Optional[int] = None, name: Optional[str] = None) -> Tensor:
        pass
    def reshape(self, shape, name: Optional[str] = None) -> Tuple[Tensor, Tensor]:
        pass
    def reshape_(self, shape, name: Optional[str] = None) -> None:
        pass
    def reverse(self, axis, name: Optional[str] = None) -> Tensor:
        pass
    def roll(self, shifts, axis: Optional[int] = None, name: Optional[str] = None) -> Tensor:
        pass
    def rot90(self, k=1, axes=[0, 1], name: Optional[str] = None) -> Tensor:
        pass
    def round(self, name: Optional[str] = None) -> Tensor:
        pass
    def round_(self, name: Optional[str] = None) -> None:
        pass
    def rsqrt(self, name: Optional[str] = None) -> Tensor:
        pass
    def rsqrt_(self, name: Optional[str] = None) -> None:
        pass
    def scale(self, scale=1.0, bias=0.0, bias_after_scale=True, act=None, name: Optional[str] = None) -> Tensor:
        pass
    def scale_(self, scale=1.0, bias=0.0, bias_after_scale=True, act=None, name: Optional[str] = None) -> None:
        pass
    def scatter(self, index, updates, overwrite=True, name: Optional[str] = None) -> Tensor:
        pass
    def scatter_(self, index, updates, overwrite=True, name: Optional[str] = None) -> None:
        pass
    def scatter_nd(self, updates, shape, name: Optional[str] = None) -> Tuple[]:
        pass
    def scatter_nd_add(self, index, updates, name: Optional[str] = None) -> Tensor:
        pass
    def set_value(self, value) -> Tuple[]:
        pass
    def shard_index(self, index_num, nshards, shard_id, ignore_value=-1) -> Tensor:
        pass
    def sign(self, name: Optional[str] = None) -> Tensor:
        pass
    def sin(self, name: Optional[str] = None) -> Tensor:
        pass
    def sinh(self, name: Optional[str] = None) -> Tensor:
        pass
    def slice(self, axes, starts, ends) -> Tensor:
        pass
    def solve(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        pass
    def sort(self, axis: int = -1, descending=False, name: Optional[str] = None) -> Tensor:
        pass
    def split(self, num_or_sections, axis: int = 0, name: Optional[str] = None) -> List[Tensor]:
        pass
    def sqrt(self, name: Optional[str] = None) -> Tensor:
        pass
    def sqrt_(self, name: Optional[str] = None) -> None:
        pass
    def square(self, name: Optional[str] = None) -> Tensor:
        pass
    def squeeze(self, axis: Optional[int] = None, name: Optional[str] = None) -> Tuple[Tensor, Tensor]:
        pass
    def squeeze_(self, axis: Optional[int] = None, name: Optional[str] = None) -> None:
        pass
    def stack(self, axis: int = 0, name: Optional[str] = None) -> Tensor:
        pass
    def stanh(self, scale_a=0.67, scale_b=1.7159, name: Optional[str] = None) -> Tuple[]:
        pass
    def std(self, axis: Optional[int] = None, unbiased=True, keepdim: bool = False, name: Optional[str] = None) -> Tuple[]:
        pass
    def strided_slice(self, axes, starts, ends, strides, name: Optional[str] = None) -> Tensor:
        pass
    def subtract(self, y: Tensor, name: Optional[str] = None) -> Tensor:
        pass
    def subtract_(self, y: Tensor, name: Optional[str] = None) -> None:
        pass
    def sum(self, axis: Optional[int] = None, dtype=None, keepdim: bool = False, name: Optional[str] = None) -> Tensor:
        pass
    def t(self, name: Optional[str] = None) -> Tuple[]:
        pass
    def take_along_axis(self, indices, axis) -> Tensor:
        pass
    def tanh(self, name: Optional[str] = None) -> Tensor:
        pass
    def tanh_(self, name: Optional[str] = None) -> None:
        pass
    def tensordot(self, y: Tensor, axes=2, name: Optional[str] = None) -> Tuple[]:
        pass
    def tile(self, repeat_times, name: Optional[str] = None) -> Tensor:
        pass
    def to_dense(self) -> Tensor:
        pass
    def to_sparse_coo(self, sparse_dim) -> Tensor:
        pass
    def tolist(self) -> Tuple[]:
        pass
    def topk(self, k, axis: Optional[int] = None, largest=True, sorted=True, name: Optional[str] = None) -> Tuple[Tensor, Tensor]:
        pass
    def trace(self, offset=0, axis1=0, axis2=1, name: Optional[str] = None) -> Tensor:
        pass
    def transpose(self, perm, name: Optional[str] = None) -> Tensor:
        pass
    def trunc(self, name: Optional[str] = None) -> Tensor:
        pass
    def unbind(self, axis: int = 0) -> List[Tensor]:
        pass
    def uniform_(self, min=-1.0, max=1.0, seed=0, name: Optional[str] = None) -> None:
        pass
    def unique(self, return_index=False, return_inverse=False, return_counts=False, axis: Optional[int] = None, dtype='int64', name: Optional[str] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        pass
    def unique_consecutive(self, return_inverse=False, return_counts=False, axis: Optional[int] = None, dtype='int64', name: Optional[str] = None) -> Tuple[Tensor, Tensor, Tensor]:
        pass
    def unsqueeze(self, axis, name: Optional[str] = None) -> Tuple[Tensor, Tensor]:
        pass
    def unsqueeze_(self, axis, name: Optional[str] = None) -> None:
        pass
    def unstack(self, axis: int = 0, num=None) -> List[Tensor]:
        pass
    def values(self) -> Any:
        pass
    def var(self, axis: Optional[int] = None, unbiased=True, keepdim: bool = False, name: Optional[str] = None) -> Tuple[]:
        pass
    def where(self, x=None, y=None, name: Optional[str] = None) -> Tensor:
        pass
    def zero_(self) -> None:
        pass
