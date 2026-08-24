"""
Build Your Own teenygrad: A Tiny Tensor Autograd Engine

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - prod
def prod(shape):
    # TODO: Multiply together the elements of a shape tuple to get the total number of elements.
    product = 1
    for i in shape:
        product *= i

    return product

# Step 2 - argsort
def argsort(values):
    # TODO: Return the indices that would sort values in ascending order.
    return sorted(range(len(values)), key=lambda x: values[x])

# Step 3 - make_op_enums
from enum import Enum

def make_op_enums():
    # TODO: create four enum classes naming every supported operation kind
    UnaryOps = Enum('UnaryOps', ['NEG', 'RELU', 'LOG', 'EXP', 'SQRT', 'SIGMOID'])
    BinaryOps = Enum('BinaryOps', ['ADD', 'SUB', 'MUL', 'DIV', 'CMPLT', 'MAX'])
    ReduceOps = Enum('ReduceOps', ['SUM', 'MAX'])
    MovementOps = Enum('MovementOps', ['RESHAPE', 'EXPAND', 'PERMUTE'])

    return UnaryOps, BinaryOps, ReduceOps, MovementOps

# Step 4 - LazyBuffer
class LazyBuffer:
    def __init__(self, np_array):
        # TODO: wrap np_array as an ndarray and expose shape and dtype
        self._np = np.asarray(np_array)
        self.shape = self._np.shape
        self.dtype = self._np.dtype

# Step 5 - lazybuffer_const
def const(value, shape):
    # TODO: Create a new LazyBuffer of the given shape filled with a constant value.
    return LazyBuffer(np.full(shape, value, dtype=np.float32))
LazyBuffer.const = staticmethod(const)

# Step 6 - rand
def rand(shape, seed=None):
    # TODO: return a LazyBuffer of uniform random floats in [0, 1) with given shape
    rng = np.random.default_rng(seed)
    return LazyBuffer(rng.random(shape, dtype=np.float32))

# Step 7 - lazybuffer_unary_e
def e(self, op):
    # TODO: apply a unary elementwise op (NEG, RELU, LOG, EXP, SQRT, SIGMOID)
    match op.name:
        case 'NEG':
            out = -self._np
        case 'RELU':
            out = np.maximum(self._np, 0.0)
        case 'LOG':
            out = np.log(self._np)
        case 'EXP':
            out = np.exp(self._np)
        case 'SQRT':
            out = np.sqrt(self._np)
        case 'SIGMOID':
            out = 1.0/(1.0 + np.exp(-self._np))
        case _:
            raise ValueError

    return LazyBuffer(out)

LazyBuffer.e = e

# Step 8 - lazybuffer_binary_e
def lazybuffer_binary_e(self, op, other):
    # TODO: apply a binary elementwise op between two LazyBuffers, return a new LazyBuffer
    match op.name:
        case 'ADD':
            out = self._np + other._np
        case 'SUB':
            out = self._np - other._np
        case 'MUL':
            out = self._np * other._np
        case 'DIV':
            out = self._np / other._np
        case 'CMPLT':
            out = (self._np < other._np).astype(self.dtype)
        case 'MAX':
            out = np.maximum(self._np, other._np)
        case _:
            raise ValueError

    return LazyBuffer(out)

LazyBuffer.lazybuffer_binary_e = lazybuffer_binary_e

# Step 9 - lazybuffer_r
def r(self, op, axis):
    # TODO: reduce the underlying array along axis (SUM or MAX), keeping reduced dims as size 1
    match op.name:
        case 'SUM':
            out = self._np.sum(axis=axis, keepdims=True)
        case 'MAX':
            out = self._np.max(axis=axis, keepdims=True)
        case _:
            raise ValueError

    return LazyBuffer(out)

LazyBuffer.r = r

# Step 10 - lazybuffer_reshape
def reshape(self, new_shape):
    # TODO: return a new LazyBuffer with the array reshaped to new_shape
    out = self._np.reshape(new_shape)
    return LazyBuffer(out)

LazyBuffer.reshape = reshape

# Step 11 - lazybuffer_expand
def expand(self, new_shape):
    # TODO: broadcast this buffer's size-1 dims out to new_shape
    shape = tuple(int(d) for d in new_shape)
    out = np.array(np.broadcast_to(self._np, shape))
    return LazyBuffer(out)

LazyBuffer.expand = expand

# Step 12 - lazybuffer_permute
def permute(self, order):
    # TODO: return a new LazyBuffer with axes reordered according to order
    return LazyBuffer(self._np.transpose(order))

LazyBuffer.permute = permute

# Step 13 - Function
class Function:
    def __init__(self, *tensors):
        # TODO: record needs_input_grad, requires_grad, and parents for backprop
        self.needs_input_grad = [t.requires_grad for t in tensors]
        is_none = None in self.needs_input_grad
        self.requires_grad = True if any(self.needs_input_grad) and not is_none else None if is_none else False

        if self.requires_grad:
            self.parents = tensors

# Step 14 - function_forward_backward_stubs
def function_forward_backward_stubs():
    # TODO: attach forward and backward stubs to Function that raise NotImplementedError
    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"forward not implemented for {type(self).__name__}")

    def backward(self, *args, **kwargs):
        raise NotImplementedError(f"backward not implemented for {type(self).__name__}")

    Function.forward = forward
    Function.backward = backward

# Step 15 - apply
@classmethod
def apply(cls, *tensors, **kwargs):
    # TODO: build the Function, run forward on the input buffers, wrap in a
    # Tensor, and link out._ctx when a gradient is needed.
    ctx = cls(*tensors)
    out = Tensor(ctx.forward(*[t.lazydata for t in tensors], **kwargs), requires_grad=ctx.requires_grad)
    if ctx.requires_grad:
        out._ctx = ctx
    
    return out


# Provided: attaches apply onto the Function base class. Leave this as-is.
for _obj in list(globals().values()):
    if isinstance(_obj, type):
        for _k in _obj.__mro__:
            if _k.__name__ == 'Function':
                _k.apply = apply

# Step 16 - Neg
import numpy as np

class Neg(Function):
    def forward(self, x):
        # TODO: return a LazyBuffer holding the elementwise negation of x
        return LazyBuffer(-x._np)

    def backward(self, grad_output):
        # TODO: return the negated incoming gradient
        return LazyBuffer(-grad_output._np)

# Step 17 - Relu
UnaryOps, BinaryOps, ReduceOps, MovementOps = make_op_enums()

class Relu(Function):
    def forward(self, x):
        # TODO: apply the rectified linear unit to lazy buffer x and cache the result
        self.ret = x.e(UnaryOps.RELU)
        return self.ret


    def backward(self, grad_output):
        # TODO: route the upstream gradient only through positions that were positive
        grad = const(0, grad_output.shape)
        mask = grad.lazybuffer_binary_e(BinaryOps.CMPLT, self.ret)
        return grad_output.lazybuffer_binary_e(BinaryOps.MUL, mask)

# Step 18 - Log
class Log(Function):
    def forward(self, x):
        # TODO: return the natural log of x and save x for backward
        self.x = x
        return x.e(UnaryOps.LOG)


    def backward(self, grad_output):
        # TODO: return the gradient of log with respect to its input
        return grad_output.lazybuffer_binary_e(BinaryOps.DIV, self.x)

# Step 19 - Exp
class Exp(Function):
    def forward(self, x):
        # TODO: compute the elementwise exponential and keep what backward needs
        self.ret = x.e(UnaryOps.EXP)
        return self.ret


    def backward(self, grad_output):
        # TODO: turn the upstream gradient into the gradient w.r.t. the input
        return grad_output.lazybuffer_binary_e(BinaryOps.MUL, self.ret)

# Step 20 - Sqrt
class Sqrt(Function):
    def forward(self, x):
        # TODO: compute the elementwise square root and cache it for backward
        self.ret = x.e(UnaryOps.SQRT)
        return self.ret


    def backward(self, grad_output):
        factor = const(0.5, grad_output.shape)
        grad = factor.lazybuffer_binary_e(BinaryOps.DIV, self.ret)
        return grad_output.lazybuffer_binary_e(BinaryOps.MUL, grad)

# Step 21 - Sigmoid
class Sigmoid(Function):
    def forward(self, x):
        # TODO: return the elementwise logistic activation of LazyBuffer x
        self.ret = x.e(UnaryOps.SIGMOID)
        return self.ret


    def backward(self, grad_output):
        # TODO: return grad_output times the sigmoid derivative
        grad = const(1.0, grad_output.shape)
        grad = grad.lazybuffer_binary_e(BinaryOps.SUB, self.ret)
        grad = self.ret.lazybuffer_binary_e(BinaryOps.MUL, grad)

        return grad_output.lazybuffer_binary_e(BinaryOps.MUL, grad)

# Step 22 - Add
class Add(Function):
    def forward(self, x, y):
        # TODO: return the elementwise sum of LazyBuffers x and y
        return x.lazybuffer_binary_e(BinaryOps.ADD, y)


    def backward(self, grad_output):
        # TODO: route grad_output to each input that requires a gradient
        return (grad_output if rg else None for rg in self.needs_input_grad)

# Step 23 - Sub
class Sub(Function):
    def forward(self, x, y):
        # TODO: return the elementwise difference x - y as a LazyBuffer
        return x.lazybuffer_binary_e(BinaryOps.SUB, y)


    def backward(self, grad_output):
        # TODO: return gradients for x and y (None where grad is not needed)
        return (grad_output if self.needs_input_grad[0] else None,
        grad_output.e(UnaryOps.NEG) if self.needs_input_grad[1] else None)

# Step 24 - Mul
class Mul(Function):
    def forward(self, x, y):
        # TODO: compute the elementwise product and save what backward needs
        self.x = x
        self.y = y
        return x.lazybuffer_binary_e(BinaryOps.MUL, y)


    def backward(self, grad_output):
        # TODO: return the gradient w.r.t. each input (None if not needed)
        return (grad_output.lazybuffer_binary_e(BinaryOps.MUL, self.y) if self.needs_input_grad[0] else None,
        grad_output.lazybuffer_binary_e(BinaryOps.MUL, self.x) if self.needs_input_grad[1] else None)

# Step 25 - Div
class Div(Function):
    def forward(self, x, y):
        # TODO: divide LazyBuffer x by y and cache inputs for backward
        self.x = x
        self.y = y
        return x.lazybuffer_binary_e(BinaryOps.DIV, y)


    def backward(self, grad_output):
        # TODO: return gradients w.r.t. x and y via the quotient rule
        if self.needs_input_grad[0]:
            ones = const(1.0, grad_output.shape)
            grad_x = ones.lazybuffer_binary_e(BinaryOps.DIV, self.y)
            grad_x = grad_x.lazybuffer_binary_e(BinaryOps.MUL, grad_output)
        else:
            grad_x = None

        if self.needs_input_grad[1]:
            ones = const(-1.0, grad_output.shape)
            grad_y = self.y.lazybuffer_binary_e(BinaryOps.MUL, self.y)
            grad_y = self.x.lazybuffer_binary_e(BinaryOps.DIV, grad_y)
            grad_y = grad_y.lazybuffer_binary_e(BinaryOps.MUL, ones)
            grad_y = grad_y.lazybuffer_binary_e(BinaryOps.MUL, grad_output)
        else:
            grad_y = None

        return (grad_x, grad_y)

# Step 26 - sum_function_forward
class Sum(Function):
    def forward(self, x, axis):
        # TODO: Reduce x with ReduceOps.SUM over axis (keepdims) and cache shape/axis.
        self.input_shape = x._np.shape
        self.axis = axis

        return x.r(ReduceOps.SUM, axis)

# Step 27 - sum_function_backward
def backward(self, grad_output):
    # TODO: broadcast the summed gradient back to the original input shape
    return grad_output.expand(self.input_shape)

# Step 28 - max_function_forward
class Max(Function):
    def forward(self, x, axis):
        # TODO: reduce x with the MAX reduce op along axis and cache for backward
        self.x = x
        self.axis = axis
        self.ret = x.r(ReduceOps.MAX, axis)
        return self.ret

# Step 29 - max_function_backward
def backward(self, grad_output):
    # TODO: route grad_output back to the input elements that were the maximum
    ones = const(1.0, self.x.shape)
    mask = self.x.lazybuffer_binary_e(BinaryOps.CMPLT, self.ret)
    mask = ones.lazybuffer_binary_e(BinaryOps.SUB, mask)
    counts = mask.r(ReduceOps.SUM, self.axis)
    counts = counts.expand(self.x.shape)
    mask = mask.lazybuffer_binary_e(BinaryOps.DIV, counts)
    mask = grad_output.expand(self.x.shape).lazybuffer_binary_e(BinaryOps.MUL, mask)
    return mask


Max.backward = backward

# Step 30 - Reshape
class Reshape(Function):
    def forward(self, x, shape):
        # TODO: cache the input shape and return x reshaped to shape
        self.input_shape = x.shape
        return x.reshape(shape)


    def backward(self, grad_output):
        # TODO: reshape the gradient back to the cached input shape
        return grad_output.reshape(self.input_shape)

# Step 31 - expand_function_forward
def expand_function_forward(ctx, x, shape):
    # TODO: cache x.shape on ctx, then broadcast x to the target shape
    ctx.input_shape = x.shape
    shape = tuple(int(d) for d in shape)
    return x.expand(shape)

# Step 32 - expand_function_backward
def expand_function_backward(ctx, grad_output):
    # TODO: Sum grad_output over the broadcast axes back to ctx.input_shape...
    axis = tuple(i for i in range(len(ctx.input_shape)) if ctx.input_shape[i]==1 and grad_output.shape[i]!=1)
    return grad_output.r(ReduceOps.SUM, axis)

# Step 33 - permute_function_forward_backward
def permute_function_forward_backward():
    # TODO: return (forward, backward); forward reorders axes, backward inverts the order
    def forward(self, x, order):
        self.order = order
        return x.permute(order)


    def backward(self, grad_output):
        order = argsort(self.order)
        return grad_output.permute(order)

    
    return (forward, backward)

# Step 34 - Tensor
class Tensor:
    def __init__(self, data, requires_grad=False, _ctx=None):
        # TODO: wrap data in a LazyBuffer and store grad/ctx bookkeeping
        if isinstance(data, LazyBuffer):
            self.lazydata = data
        else:
            self.lazydata = LazyBuffer(np.asarray(data, dtype=np.float32))

        self.requires_grad = requires_grad
        self.grad = None
        self._ctx = _ctx


    @property
    def data(self):
        # TODO: return the underlying LazyBuffer
        return self.lazydata

    @data.setter
    def data(self, value):
        # TODO: replace the underlying LazyBuffer
        if isinstance(value, LazyBuffer):
            self.lazydata = value
        else:
            self.lazydata = LazyBuffer(np.asarray(value, dtype=np.float32))

    @property
    def shape(self):
        return self.lazydata.shape

    @property
    def dtype(self):
        return self.lazydata.dtype

    def numpy(self):
        return self.lazydata._np

# Step 35 - tensor_from_data
def tensor_from_data(data, requires_grad=False):
    # TODO: wrap a number, list, or numpy array in a LazyBuffer held by a Tensor
    if isinstance(data, LazyBuffer):
        buf = data
    else:
        buf = LazyBuffer(np.asarray(data, dtype=np.float32))
    return Tensor(buf, requires_grad)

# Step 36 - tensor_creation_helpers
def tensor_creation_helpers():
    # TODO: return (zeros_fn, ones_fn, full_fn) building constant-filled Tensors
    def zeros_fn(shape):
        return Tensor(const(0.0, shape))


    def ones_fn(shape):
        return Tensor(const(1.0, shape))

    
    def full_fn(shape, value):
        return Tensor(const(value, shape))


    return (zeros_fn, ones_fn, full_fn)

# Step 37 - tensor_randn
def tensor_randn(shape, seed=None, requires_grad=False):
    # TODO: Create a Tensor of standard-normal samples for the given shape.
    shape = tuple(int(d) for d in shape)
    u = np.random.RandomState(seed).rand(2, *shape)
    u1 = np.clip(u[0], 1e-12, 1.0)
    u2 = u[1]
    z = (np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0*np.pi*u2)).astype(np.float32)
    buf = LazyBuffer(z)

    return Tensor(buf, requires_grad)

# Step 38 - build_topological_order
def build_topological_order(tensor):
    # TODO: DFS over each node's _ctx.parents, append a node after its parents
    visited = set()
    def dfs(tensor):
        order = []
        if tensor not in visited:
            visited.add(tensor)
            if tensor._ctx is not None:
                for parent in tensor._ctx.parents:
                    order += dfs(parent)
            order += [tensor]

        return order

    return dfs(tensor)

# Step 39 - tensor_backward
def tensor_backward(tensor):
    # TODO: seed root grad with ones, run each backward in reverse topo order
    tensor.grad = Tensor(const(1.0, tensor.shape))
    grad_output = const(1.0, tensor.shape)

    for node in reversed(build_topological_order(tensor)):
        if node._ctx is None:
            continue
        grads = node._ctx.backward(grad_output)
        if isinstance(grads, LazyBuffer):
            grads = [grads]
        for parent, grad in zip(node._ctx.parents, grads):
            if grad is None or not parent.requires_grad:
                continue
            if parent.grad is None:
                parent.grad = Tensor(grad)
            else:
                parent.grad = tensor_from_data(parent.grad.lazydata._np + grad._np)

# Step 40 - bind_unary_tensor_methods
def bind_unary_tensor_methods():
    # TODO: map neg/relu/log/exp/sqrt/sigmoid names to callables using function_apply
    def _make(F):
        def method(*t, **kwargs):
            return F.apply(*t, **kwargs)
        return method

    methods = {
        'neg' : _make(Neg),
        'relu' : _make(Relu),
        'log' : _make(Log),
        'exp' : _make(Exp),
        'sqrt' : _make(Sqrt),
        'sigmoid' : _make(Sigmoid)
    }

    return methods

# Step 41 - broadcasted
def broadcasted(x, y):
    # TODO: align two tensors to one common shape so an elementwise op can run
    if x.data._np.shape == y.data._np.shape:
        return x, y
    
    bx, by = np.broadcast_arrays(x.data._np, y.data._np)
    bx = np.array(bx, dtype=np.float32)
    by = np.array(by, dtype=np.float32)

    return tensor_from_data(bx), tensor_from_data(by)

# Step 42 - bind_binary_tensor_methods
def bind_binary_tensor_methods():
    # TODO: attach broadcasting add/sub/mul/div methods onto the Tensor class
    def add(self, other):
        x, y = broadcasted(self, other)
        return Add.apply(x, y)
    Tensor.add = add
    Tensor.__add__ = add

    def sub(self, other):
        x, y = broadcasted(self, other)
        return Sub.apply(x, y)
    Tensor.sub = sub
    Tensor.__sub__ = sub

    def mul(self, other):
        x, y = broadcasted(self, other)
        return Mul.apply(x, y)
    Tensor.mul = mul
    Tensor.__mul__ = mul

    def div(self, other):
        x, y = broadcasted(self, other)
        return Div.apply(x, y)
    Tensor.div = div
    Tensor.__truediv__ = div

# Step 43 - bind_movement_tensor_methods
def bind_movement_tensor_methods():
    def _get_lazydata(t):
        for attr in ('lazydata', 'data', '_lazydata'):
            if hasattr(t, attr):
                val = getattr(t, attr)
                return val if isinstance(val, LazyBuffer) else LazyBuffer(val)
        raise AttributeError("no lazybuffer found")

    def _wrap(out, requires_grad, ctx):
        t = Tensor.__new__(Tensor)
        t.lazydata = out
        t.requires_grad = requires_grad
        t.grad = None
        t._ctx = ctx if requires_grad else None
        return t

    Expand = type('Expand', (Function,), {'forward': expand_function_forward, 'backward': expand_function_backward})
    permute_fwd, permute_bwd = permute_function_forward_backward()
    Permute = type('Permute', (Function,), {'forward': permute_fwd, 'backward': permute_bwd})

    def _apply(Cls, x, **kwargs):
        ctx = object.__new__(Cls)
        ctx.needs_input_grad = [x.requires_grad]
        ctx.requires_grad = x.requires_grad
        ctx.parents = (x,)
        buf = _get_lazydata(x)
        out = ctx.forward(buf, **kwargs)
        return _wrap(out, ctx.requires_grad, ctx)

    def reshape(self, shape):
        return _apply(Reshape, self, shape=tuple(shape))

    def expand(self, shape):
        return _apply(Expand, self, shape=tuple(shape))

    def permute(self, order):
        return _apply(Permute, self, order=tuple(order))

    return {
        'reshape': reshape,
        'expand': expand,
        'permute': permute
    }

# Step 44 - bind_reduce_tensor_methods (not yet solved)
# TODO: implement

# Step 45 - tensor_mean (not yet solved)
# TODO: implement

# Step 46 - tensor_transpose (not yet solved)
# TODO: implement

# Step 47 - tensor_matmul_2d (not yet solved)
# TODO: implement

# Step 48 - tensor_softmax (not yet solved)
# TODO: implement

# Step 49 - tensor_log_softmax (not yet solved)
# TODO: implement

# Step 50 - sparse_categorical_cross_entropy (not yet solved)
# TODO: implement

# Step 51 - Linear (not yet solved)
# TODO: implement

# Step 52 - MLP (not yet solved)
# TODO: implement

# Step 53 - sgd_step (not yet solved)
# TODO: implement

# Step 54 - zero_grad (not yet solved)
# TODO: implement

# Step 55 - make_toy_digit_dataset (not yet solved)
# TODO: implement

# Step 56 - accuracy (not yet solved)
# TODO: implement

# Step 57 - train_mlp (not yet solved)
# TODO: implement

# Step 58 - evaluate_mlp (not yet solved)
# TODO: implement

