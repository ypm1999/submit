#!/user/bin/env python3
# -*- coding:utf-8 -*-

from tensorlow.base import *
from tensorlow.topo import *


class Node(object):
	def __init__(self):
		self.input = []
		self.value = None
		self.op = None
		self.dtype = None
		self.shape = ()
		self.name = ""

	def __add__(self, other):
		if not isinstance(other, Node):
			other = constant(other)
		return add(self, other)

	def __sub__(self, other):
		if  not isinstance(other, Node):
			other = constant(other)
		return sub(self, other)

	def __rsub__(self, other):
		if  not isinstance(other, Node):
			other = constant(other)
		return sub(other, self)

	def __mul__(self, other):
		if  not isinstance(other, Node):
			other = constant(other)
		return mul(self, other)

	def __truediv__(self, other):
		if not isinstance(other, Node):
			other = constant(other)
		return div(self, other)

	def __rtruediv__(self, other):
		if not isinstance(other, Node):
			other = constant(other)
		return div(other, self)

	def __neg__(self):
		return neg(self)

	__radd__ = __add__
	__rmul__ = __mul__

	def __str__(self):
		return self.name

	__repr__ = __str__

	def run(self, feed_dict = None):
		from tensorlow.Session import default_session
		return default_session.run(self, feed_dict)


	eval = run


class Op(object):

	def __call__(self):
		new_node = Node()
		new_node.op = self
		return new_node

	def compute(self, node, input_vals):
		raise NotImplementedError

	def gradient(self, node, grad):
		raise NotImplementedError


class PlaceHolderOp(Op):
	placeholder_list = []
	value_list = {}

	def __call__(self, dtype, shape = None, name = "Plh"):
		new_node = Node()
		new_node.dtype = dtype
		new_node.shape = shape
		new_node.name = name
		new_node.op = self
		self.placeholder_list.append(new_node)
		self.value_list[new_node] = None
		return new_node


class VariableOp(Op):
	node_list = []

	def __call__(self, initial_value=None, name="Var", dtype=float32):
		new_node = placeholder(dtype, name = name)

		if isinstance(initial_value, Node):
			new_node.input = [initial_value]
		else:
			if isinstance(initial_value, list):
				new_node.value = np.array(initial_value, dtype = dtype)
			else:
				new_node.value = dtype(initial_value)
			new_node.shape = new_node.value.shape

		self.node_list.append(new_node)
		return new_node


class ConstantOp(Op):
	def __call__(self, value, dtype = float32, shape = None, name = "Const"):
		new_node = placeholder(dtype, name = name)

		if isinstance(value, list):
			new_node.value = np.array(value, dtype = dtype)
		else:
			new_node.value = dtype(value)

		if shape:
			new_node.value = new_node.value + zeros(shape, dtype = float32)
			new_node.shape = shape
		else:
			new_node.shape = new_node.value.shape

		placeholder.value_list[new_node] = new_node.value
		return new_node


class Global_Variables_InitializerOp(Op):

	def __call__(self):
		new_node = Node()
		new_node.op = self
		new_node.name = "Global_Variables_Initializer"
		return new_node

	def compute(self, node, input_vals):
		for i in Variable.node_list:
			if type(i.value) != type(None):
				placeholder.value_list[i] = i.value
			else:
				placeholder.value_list[i] = i.input[0].run()
		return None


class AssignOp(Op):

	def __call__(self, ref, value, name="Assign"):
		new_node = Node()
		new_node.op = self
		new_node.name = name
		if isinstance(value, Node):
			new_node.input = [value]
		else:
			new_node.value = value
		new_node.ref = ref
		new_node.name = "assign(%s,%s)" % (ref.name, value)
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == len(node.input), "\033[1;31mNode number not suit!\033[0m"
		ref = node.ref
		if len(node.input) == 1:
			ref.value = input_vals[0]
		else:
			if isinstance(node.value, list):
				ref.value = np.array(node.value)
			else:
				ref.value = node.value

		placeholder.value_list[ref] = ref.value
		return ref


class AddOp(Op):
	def __call__(self, node1, node2):
		new_node = Node()
		new_node.op = self
		new_node.input = [node1, node2]
		new_node.name = "%s+%s" % (node1.name, node2.name)
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 2, "\033[1;31mNode number not suit at add!\033[0m"
		return input_vals[0] + input_vals[1]

	def gradient(self, node, grad):
		return [reduce_shape_to(grad, node.input[0]), reduce_shape_to(grad, node.input[1])]


class SubOp(Op):
	def __call__(self, node1, node2):
		new_node = Node()
		new_node.op = self
		new_node.input = [node1, node2]
		new_node.name = "%s-%s" % (node1.name, node2.name)
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 2, "\033[1;31mNode number not suit at sub!\033[0m"
		return input_vals[0] - input_vals[1]

	def gradient(self, node, grad):
		return [reduce_shape_to(grad, node.input[0]), reduce_shape_to(-grad, node.input[1])]


class NegOp(Op):
	def __call__(self, node1):
		new_node = Node()
		new_node.op = self
		new_node.input = [node1]
		new_node.name = "(-%s)" % node1.name
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 1, "\033[1;31mNode number not suit at neg!\033[0m"
		return -input_vals[0]

	def gradient(self, node, grad):
		return [reduce_shape_to(-grad, node.input[0])]


class MulOp(Op):
	def __call__(self, node1, node2):
		new_node = Node()
		new_node.op = self
		new_node.input = [node1, node2]
		new_node.name = "(%s*%s)" % (node1.name, node2.name)
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 2, "\033[1;31mNode number not suit at mul!\033[0m"
		return input_vals[0] * input_vals[1]

	def gradient(self, node, grad):
		resl = node.input[1] * grad
		resr = node.input[0] * grad
		return [reduce_shape_to(resl, node.input[0]), reduce_shape_to(resr, node.input[1])]


class DivOp(Op):
	def __call__(self, node1, node2):
		new_node = Node()
		new_node.op = self
		new_node.input = [node1, node2]
		new_node.name = "%s/%s" % (node1.name, node2.name)
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 2, "\033[1;31mNode number not suit!\033[0m"
		return input_vals[0] / input_vals[1]

	def gradient(self, node, grad):
		resl = grad / node.input[1]
		resr = (-grad * node.input[0]) / (node.input[1] * node.input[1])
		return [reduce_shape_to(resl, node.input[0]), reduce_shape_to(resr, node.input[1])]


class MatmulOp(Op):
	cnt = 0
	def __call__(self, node1, node2, trans1 = False, trans2 = False):
		new_node = Node()
		new_node.op = self
		new_node.transA = trans1
		new_node.transB = trans2
		new_node.input = [node1, node2]
		new_node.name = "MatMul(%s,%s,%s,%s)" % (node1.name, node2.name, str(trans1), str(trans2))
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 2, "\033[1;31mNode number not suit at matmul!\033[0m"
		a, b = input_vals
		na, ma = a.shape
		nb, mb = b.shape
		if node.transA:
			if node.transB:
				result = np.ndarray(shape = (ma, nb), dtype = float32)
			else:
				result = np.ndarray(shape = (ma, mb), dtype = float32)
		else:
			if node.transB:
				result = np.ndarray(shape = (na, nb), dtype = float32)
			else:
				result = np.ndarray(shape = (na, mb), dtype = float32)
		matmul_c(a, b, result, na, ma, nb, mb, node.transA, node.transB)
		return result

	def gradient(self, node, grad):
		resl = matmul(grad, node.input[1], False, True ^ node.transB)
		resr = matmul(node.input[0], grad, True ^ node.transA, False)
		return [reduce_shape_to(resl, node.input[0]), reduce_shape_to(resr, node.input[1])]


class ExpOp(Op):
	def __call__(self, node1):
		new_node = Node()
		new_node.op = self
		new_node.input = [node1]
		new_node.name = "Exp(%s)" % node1.name
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 1, "\033[1;31mNode number not suit at exp!\033[0m"
		return np.exp(input_vals[0])

	def gradient(self, node, grad):
		return [grad * exp(node.input[0])]


class LogOp(Op):
	def __call__(self, node1):
		new_node = Node()
		new_node.op = self
		new_node.input = [node1]
		new_node.name = "Log(%s)" % node1.name
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 1, "\033[1;31mNode number not suit at log!\033[0m"
		return np.log(input_vals[0])

	def gradient(self, node, grad):
		return [grad / node.input[0]]


class ZerosLikeOp(Op):
	def __call__(self, node1):
		new_node = Node()
		new_node.input = [node1]
		new_node.op = self
		new_node.name = "Zeroslike(%s)" % node1.name
		return new_node

	def compute(self, node, input_vals):
		if isinstance(input_vals[0], np.ndarray):
			return np.zeros(input_vals[0].shape, dtype = np.float32)
		else:
			return float32(0)

	def gradient(self, node, output_grad):
		return [zeroslike(node.input[0])]


class OnesLikeOp(Op):
	def __call__(self, node1):
		new_node = Node()
		new_node.input = [node1]
		new_node.op = self
		new_node.name = "Oneslike(%s)" % node1.name
		return new_node

	def compute(self, node, input_vals):
		if isinstance(input_vals[0], np.ndarray):
			return np.ones(input_vals[0].shape, dtype = np.float32)
		else:
			return float32(1)

	def gradient(self, node, output_grad):
		return [zeroslike(node.input[0])]


class Reduce_Shape_ToOp(Op):

	def __call__(self, node1, node2):
		new_node = Node()
		new_node.op = self
		new_node.input = [node1, node2]
		new_node.name = "reduceShapeTo(%s,%s)" % (node1.name, node2.name)
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 2, "\033[1;31mNode number not suit at reduce_shape_to!\033[0m"
		if not isinstance(input_vals[1], np.ndarray):
			return np.sum(input_vals[0])
		shape = list(np.shape(input_vals[0]))
		newshape = list(np.shape(input_vals[1]))
		if shape == newshape:
			return input_vals[0]
		shape.reverse()
		newshape.reverse()

		while len(shape) > len(newshape):
			newshape = newshape + [1,]
		for i in range(len(shape)):
			if shape[i] != newshape[i]:
				assert newshape[i] == 1
				pos = len(shape) - i - 1
				input_vals[0] = np.sum(input_vals[0], axis = pos, keepdims = True)

		return np.reshape(input_vals[0], input_vals[1].shape)

	def gradient(self, node, grad):
		assert False, "\033[1;31mBroadcast don't have gradient!\033[0m"


class Reduce_MeanOp(Op):

	def __call__(self, input_tensor, axis=None, keepdims=None, name=None, reduction_indices=None, keep_dims=None):
		new_node = Node()
		new_node.op = self
		new_node.input = [input_tensor]
		new_node.name = "reduce_mean(%s)" % input_tensor.name
		if reduction_indices:
			new_node.axis = reduction_indices[0]
		else:
			new_node.axis = axis
		if keep_dims:
			new_node.keepdims = keep_dims
		else:
			new_node.keepdims = keepdims
		new_node.name = "reduce_mean(%s,axis=%s,keepdim=%s)" % (input_tensor.name, new_node.axis, new_node.keepdims)
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 1, "\033[1;31mNode number not suit at reduce_mean!\033[0m"
		if node.keepdims:
			return np.mean(input_vals[0], axis = node.axis, keepdims = node.keepdims)
		else:
			return np.mean(input_vals[0], axis = node.axis)

	def gradient(self, node, grad):
		return [expand_mean(grad, node.input[0], node.axis, node.keepdims)]


class Expand_MeanOp(Op):
	def __call__(self, node1, node2, axis, keepdims):
		new_node = Node()
		new_node.op = self
		new_node.input = [node1, node2]
		new_node.axis = axis
		new_node.keepdims = keepdims
		new_node.name = "expand_mean(%s, %s, axis=%s, keepdims=%s)" % (node1.name, node2.name, axis, keepdims)
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 2, "\033[1;31mNode number not suit at expand_mean!\033[0m"
		if node.axis and not node.keepdims:
			input_vals[0] = np.expand_dims(input_vals[0], node.axis)
		new_shape = np.shape(input_vals[1])
		res = 1
		if node.axis:
			res = new_shape[node.axis]
		else:
			for i in new_shape:
				res = res * i
		return np.array(np.broadcast_to(input_vals[0] / float32(res), np.shape(input_vals[1])))

	def gradient(self, node, grad):
		assert False, "\033[1;31mExpand_mean don't have gradient!\033[0m"


class Reduce_SumOp(Op):

	def __call__(self, input_tensor, axis=None, keepdims=None, name=None, reduction_indices=None, keep_dims=None):
		new_node = Node()
		new_node.op = self
		new_node.input = [input_tensor]
		new_node.name = "reduce_sum(%s)" % input_tensor.name
		if reduction_indices:
			new_node.axis = reduction_indices[0]
		else:
			new_node.axis = axis
		if keep_dims:
			new_node.keepdims = keep_dims
		else:
			new_node.keepdims = keepdims
		new_node.name = "reduce_sum(%s,axis=%s,keepdim=%s)" % (input_tensor.name, new_node.axis, new_node.keepdims)
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 1, "\033[1;31mNode number not suit at reduce_sum!\033[0m"
		if node.keepdims:
			return np.sum(input_vals[0], axis = node.axis, keepdims = node.keepdims)
		else:
			return np.sum(input_vals[0], axis = node.axis)

	def gradient(self, node, grad):
		return [expand_sum(grad, node.input[0], node.axis, node.keepdims)]


class Expand_SumOp(Op):
	def __call__(self, node1, node2, axis, keepdims):
		new_node = Node()
		new_node.op = self
		new_node.input = [node1, node2]
		new_node.axis = axis
		new_node.keepdims = keepdims
		new_node.name = "expand_sum(%s, %s, axis=%s, keepdims=%s)" % (node1.name, node2.name, axis, keepdims)
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 2, "\033[1;31mNode number not suit at expand_sum!\033[0m"
		if node.axis and not node.keepdims:
			input_vals[0] = np.expand_dims(input_vals[0], node.axis)
		return np.array(np.broadcast_to(input_vals[0], np.shape(input_vals[1])))

	def gradient(self, node, grad):
		assert False, "\033[1;31mExpand_sum don't have gradient!\033[0m"


class EqualOp(Op):

	def __call__(self, node1, node2):
		new_node = Node()
		new_node.op = self
		new_node.input = [node1, node2]
		new_node.name = "equal(%s,%s)" % (node1.name, node2.name)
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 2, "\033[1;31mNode number not suit at equal!\033[0m"
		return np.equal(input_vals[0], input_vals[1])

	def gradient(self, node, grad):
		assert False, "\033[1;31mEqual don't have gradient!\033[0m"


class ArgmaxOp(Op):

	def __call__(self, node1, axis = None, dimension=None):
		new_node = Node()
		new_node.op = self
		new_node.input = [node1]
		if dimension:
			new_node.axis = dimension
		else:
			new_node.axis = axis
		new_node.name = "argmax(%s,axis=%s)" % (node1.name, new_node.axis)
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 1, "\033[1;31mNode number not suit at argmax!\033[0m"
		return np.argmax(input_vals[0], axis = node.axis)

	def gradient(self, node, grad):
		assert False, "\033[1;31mArgmax don't have gradient!\033[0m"


class ReshapeOp(Op):

	def __call__(self, node1, shape):
		new_node = Node()
		new_node.op = self
		new_node.input = [node1]
		new_node.shape = shape
		new_node.name = "reshape(%s,%s)" % (node1.name, shape)
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 1, "\033[1;31mNode number not suit at reshape!\033[0m"
		return np.reshape(input_vals[0], node.shape)

	def gradient(self, node, grad):
		return [reshape_to(grad, node.input[0])]


class Reshape_ToOp(Op):

	def __call__(self, node1, node2):
		new_node = Node()
		new_node.op = self
		new_node.input = [node1, node2]
		new_node.name = "reshape_to(%s,%s)" % (node1.name, node2.name)
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 2, "\033[1;31mNode number not suit at reshape_to!\033[0m"
		return np.reshape(input_vals[0], np.shape(input_vals[1]))

	def gradient(self, node, grad):
		return reshape_to(grad, node.input[0])


class CastOp(Op):
	type_name_map = {
		"float": float32, "float16": float16, "float32": float32, "float64": float64, "float128": float128,
		"int": int32, "int8": int8, "int16": int16, "int32": int32, "int64": int64,
		"uint8": uint8, "uint16": uint16, "uint32": uint32, "uint64": uint64
	}
	def __call__(self, node1, dtype):
		new_node = Node()
		new_node.op = self
		new_node.input = [node1]
		if type(dtype) == str:
			dtype = self.type_name_map[dtype]
		new_node.dtype = dtype
		new_node.name = "cast(%s,dtype=%s)" % (node1.name, dtype)
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 1, "\033[1;31mNode number not suit at cast!\033[0m"
		return node.dtype(input_vals[0])

	def gradient(self, node, grad):
		assert False, "\033[1;31mCast don't have gradient!\033[0m"


class Gradients:

	def __call__(self, output_node, node_list):
		node_to_output_grads_list = dict()
		node_to_output_grads_list[output_node] = [oneslike(output_node)]
		node_to_output_grad = {}
		reverse_topo_order = reversed(find_topo_sort([output_node]))

		for node in reverse_topo_order:
			grad = sum_node_list(node_to_output_grads_list.get(node))
			node_to_output_grad[node] = grad
			if isinstance(node.op, PlaceHolderOp):
				continue
			assert grad != None
			input_grads = node.op.gradient(node, grad)
			for i in range(len(node.input)):
				if node_to_output_grads_list.get(node.input[i]):
					node_to_output_grads_list[node.input[i]].append(input_grads[i])
				else:
					node_to_output_grads_list[node.input[i]] = [input_grads[i]]

		grad_node_list = []
		for node in node_list:
			it = node_to_output_grad.get(node)
			if it:
				grad_node_list.append(it)
			else:
				grad_node_list.append(zeroslike(node))

		return grad_node_list



add = AddOp()
sub = SubOp()
neg = NegOp()
mul = MulOp()
div = DivOp()
reduce_shape_to = Reduce_Shape_ToOp()
expand_mean = Expand_MeanOp()
expand_sum = Expand_SumOp()
oneslike = OnesLikeOp()
zeroslike = ZerosLikeOp()
reshape_to = Reshape_ToOp()

placeholder = PlaceHolderOp()
Variable = VariableOp()
constant = ConstantOp()
global_variables_initializer = Global_Variables_InitializerOp()
assign = AssignOp()
reduce_mean = Reduce_MeanOp()
reduce_sum = Reduce_SumOp()
gradients = Gradients()
equal = EqualOp()
argmax = ArgmaxOp()
reshape = ReshapeOp()
cast = CastOp()

matmul = MatmulOp()
exp = ExpOp()
log = LogOp()


if __name__  == "__main__":
	a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype = float32)
	b = np.array([[1, 3, 2, 4], [5, 7, 6, 8]], dtype = float32)
	#c = np.ndarray(shape = (3, 5), dtype = float32)
	Mul = matmul(None, None, False, True)
	c = Mul.op.compute(Mul, [a, b])
	print(c)
