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
        return x.r(ReduceOps.MAX, axis)

# Step 29 - max_function_backward (not yet solved)
# TODO: implement

# Step 30 - Reshape (not yet solved)
# TODO: implement

# Step 31 - expand_function_forward (not yet solved)
# TODO: implement

# Step 32 - expand_function_backward (not yet solved)
# TODO: implement

# Step 33 - permute_function_forward_backward (not yet solved)
# TODO: implement

# Step 34 - Tensor (not yet solved)
# TODO: implement

# Step 35 - tensor_from_data (not yet solved)
# TODO: implement

# Step 36 - tensor_creation_helpers (not yet solved)
# TODO: implement

# Step 37 - tensor_randn (not yet solved)
# TODO: implement

# Step 38 - build_topological_order (not yet solved)
# TODO: implement

# Step 39 - tensor_backward (not yet solved)
# TODO: implement

# Step 40 - bind_unary_tensor_methods (not yet solved)
# TODO: implement

# Step 41 - broadcasted (not yet solved)
# TODO: implement

# Step 42 - bind_binary_tensor_methods (not yet solved)
# TODO: implement

# Step 43 - bind_movement_tensor_methods (not yet solved)
# TODO: implement

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

