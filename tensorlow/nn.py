#!/user/bin/env python3
# -*- coding:utf-8 -*-

from tensorlow.ops import *


class SoftmaxOp(Op):

	def __call__(self, node, axis = -1):
		tmpNode = exp(node)
		return tmpNode / reduce_sum(tmpNode, axis = axis, keepdims = True)

	def compute(self, node, input_vals):
		assert False, "\033[1;31mSofrmaxOp can't compute!\033[0m"

	def gradient(self, node, grad):
		assert False, "\033[1;31mSofrmaxOp don't have gradient!\033[0m"


class Softmax_Cross_Entropy_With_LogitsOp(Op):
	def __call__(self, labels, logits):
		new_node = Node()
		new_node.op = self
		new_node.input = [labels, logits]
		new_node.name = "Softmax_Cross_Entropy_With_Logits(%s,%s)" % (labels.name, logits.name)
		return new_node


	def compute(self, node, input_vals):
		label = input_vals[0]
		logit = input_vals[1]
		tmp = np.exp(logit)
		return -np.sum(label * (np.log(tmp) - np.log(np.sum(tmp, axis = -1, keepdims = True))), axis = 1, keepdims = True)

	def gradient(self, node, grad):
		return [zeroslike(node.input[0]), grad * (nn.softmax(node.input[1]) - node.input[0])]



class SingOp(Op):

	def __call__(self, node1):
		new_node = Node()
		new_node.op = self
		new_node.input = [node1]
		new_node.name = "sing(%s)" % node1.name
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 1, "\033[1;31mNode number not suit at sign!\033[0m"
		a = input_vals[0]
		n = 1
		for i in a.shape:
			n  *= i
		result = np.ndarray(shape = a.shape, dtype = float32)

		sign_c(a, result, n)
		return result

	def gradient(self, node, grad):
		assert False, "\033[1;31mSignOp don't have gradient!\033[0m"


class ReluOp(Op):
	def __call__(self, features):
		new_node = Node()
		new_node.op = self
		new_node.input = [features]
		new_node.name = "relu(%s)" % features.name
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 1, "\033[1;31mNode number not suit at relu!\033[0m"
		a = input_vals[0]
		shape = a.shape
		n = 1
		for i in shape:
			n *= i
		result = np.ndarray(shape = shape, dtype = float32)

		relu_c(a, result, n)
		return result

	def gradient(self, node, grad):
		return [sign(node.input[0]) * grad]


class Conv2dOp(Op):
	def __call__(self, image, filter, strides, padding):
		new_node = Node()
		new_node.op = self
		new_node.input = [image, filter]
		new_node.strides = strides
		new_node.padding = padding
		new_node.name = "conv2d(%s,%s)" % (image.name, filter.name)
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 2, "\033[1;31mNode number not suit at nn.conv2d!\033[0m"
		img = input_vals[0]
		flt = input_vals[1]
		num, n, m, ins = img.shape
		fn, fm, fin, fout = flt.shape
		assert ins == fin, "\033[1;31mThe number of channels is not same for img and filter!\033[0m"
		if(node.padding == 'SAME'):
			result = np.ndarray(shape = (num, n, m, fout), dtype = np.float32)
		else:
			result = np.ndarray(shape = (num, n - fn + 1, m - fm + 1, fout), dtype = np.float32)
		conv2d_c(img, flt, result, num, n, m, fn, fm, fin, fout, node.padding == 'SAME')
		return result


	def gradient(self, node, grad):
		return [grad_of_conv2d(node.input[0], node.input[1], grad, node.strides, node.padding),
		        grad_toW_ofconv2d(node.input[0], node.input[1], grad, node.strides, node.padding)]


class Grad_Of_conv2dOp(Op):
	def __call__(self, img, filter, grad, strides, padding):
		new_node = Node()
		new_node.op = self
		new_node.input = [img, filter, grad]
		new_node.strides = strides
		new_node.padding = padding
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 3, "\033[1;31mNode number not suit at nn.conv2d!\033[0m"
		img = input_vals[0]
		flt = input_vals[1]
		grad = input_vals[2]
		fn, fm, fin, fout = flt.shape
		num, n, m, ins = img.shape
		result = np.ndarray(shape = (num, n, m, fin), dtype = np.float32)

		backup_conv2d_image_c(grad, flt, result, num, n, m, fn, fm, fin, fout, node.padding == 'SAME')
		return result

	def gradient(self, node, grad):
		assert False, "\033[1;31mgradient of conv2d don't have gradient!\033[0m"


class Grad_toW_Of_conv2dOp():
	def __call__(self, img, filter, grad, strides, padding):
		new_node = Node()
		new_node.op = self
		new_node.input = [img, filter, grad]
		new_node.strides = strides
		new_node.padding = padding
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 3, "\033[1;31mNode number not suit at nn.conv2d!\033[0m"
		img = input_vals[0]
		flt = input_vals[1]
		grad  = input_vals[2]
		num, n, m, ins = img.shape
		fn, fm, fin, fout = flt.shape
		result = np.zeros(shape = (fn, fm, fin, fout), dtype = np.float32)
		backup_conv2d_filter_c(img, grad, result, num, n, m, fn, fm, fin, fout, node.padding == "SAME")
		return result

	def gradient(self, node, grad):
		assert False, "\033[1;31mgradient of conv2d don't have gradient!\033[0m"


class MaxpoolOp(Op):
	def __call__(self, value, ksize, strides, padding):
		assert ksize == strides, "\033[1;31mksize != strides at max_pool, not support!\033[0m"
		assert strides[0] == 1 and strides[3] == 1, "\033[1;31mstrides are not be as [1, x, x, 1] at max_pool, notsupport!\033[0m"
		new_node = Node()
		new_node.op = self
		new_node.input = [value]
		new_node.ksize = ksize
		new_node.strides = strides
		new_node.padding = padding.lower()
		new_node.maxpos = None
		new_node.name = "maxpool(%s)" % value.name
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 1, "\033[1;31mNode number not suit at nn.max_pool!\033[0m"
		img = input_vals[0]
		num, n, m, ins = np.shape(img)
		result = np.ndarray(shape = (num, n // node.ksize[1], m // node.ksize[2], ins), dtype = float32)
		node.maxpos = np.ndarray(shape = result.shape, dtype = int32)

		max_pool_c(img, result, node.maxpos, num, n, m, node.ksize[1], node.ksize[2], ins)
		return result

	def gradient(self, node, grad):
		return [grad_of_maxpool(node, node.input[0], grad, node.ksize)]


class Grad_Of_MaxpoolOp(Op):
	def __call__(self, node1, node2, node3, ksize):
		new_node = Node()
		new_node.op = self
		new_node.input = [node1, node2, node3]
		new_node.ksize = ksize
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 3, "\033[1;31mNode number not suit at max_pool's gradient!\033[0m"
		grad = input_vals[2]
		num, n, m, ins = input_vals[1].shape
		result = np.zeros(shape = (num, n, m, ins), dtype = float32)

		backup_max_pool_c(node.input[0].maxpos, grad, result, num, n, m, node.ksize[1], node.ksize[2], ins)
		return result

	def gradient(self, node, grad):
		assert False, "\033[1;31mgradient of max_pool don't have gradient!\033[0m"


class DropoutOp(Op):
	def __call__(self, x, keep_prob):
		new_node = Node()
		new_node.op = self
		new_node.input = [x, keep_prob]
		new_node.data = None
		new_node.name = "dropout(%s,%s)" % (x.name, keep_prob.name)
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 2, "\033[1;31mNode number not suit at nn.max_pool!\033[0m"
		x = input_vals[0]
		keep_prob = input_vals[1]
		shape = np.shape(x)
		node.data = np.random.rand(*shape)
		node.data = node.data < keep_prob
		return x * node.data

	def gradient(self, node, grad):
		return [grad_of_dropout(node, grad), 0]


class Grad_Of_DropoutOp(Op):
	def __call__(self, node1, node2):
		new_node = Node()
		new_node.op = self
		new_node.input = [node1, node2]
		new_node.name = "grad_of_dropout(%s,%s)" % (node1.name, node2.name)
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 2, "\033[1;31mNode number not suit at nn.max_pool!\033[0m"
		return input_vals[1] * node.input[0].data

	def gradient(self, node, grad):
		assert False, "\033[1;31mgradient of dropout don't have gradient!\033[0m"


class nn(object):
	softmax = SoftmaxOp()
	softmax_cross_entropy_with_logits = Softmax_Cross_Entropy_With_LogitsOp()
	conv2d = Conv2dOp()
	max_pool = MaxpoolOp()
	relu = ReluOp()
	dropout = DropoutOp()


sign = SingOp()
grad_of_maxpool = Grad_Of_MaxpoolOp()
grad_of_conv2d = Grad_Of_conv2dOp()
grad_toW_ofconv2d = Grad_toW_Of_conv2dOp()
grad_of_dropout = Grad_Of_DropoutOp()

#
if __name__  == "__main__":
	img = np.array([[[[1], [2], [4], [8]],
#			        [[0], [1], [5], [1], [0]],
			        [[3], [5], [7], [9]],
			        [[4], [8], [1], [4]],
			        [[8], [1], [1], [5]]]]*1, dtype = float32)

	o = 1
	flt = np.array([ [ [[1] * o], [[0] * o], [[4] * o] ],
			        [ [[0] * o], [[2] * o], [[0] * o] ],
			        [ [[5] * o], [[0] * o], [[3] * o] ]], dtype = float32)

	fflt = flt[0:2, 0:2]
	a = nn.conv2d(None, None, [1, 1, 1, 1], 'SAME')
	res = a.op.compute(a, [img, flt])
	# print(res.shape)
	# print(res[0, :, :, 0])
	b = nn.max_pool(None, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
	mx = b.op.compute(b, [res])
	# print(mx.shape)
	# print(mx[0, :, :, 0])
	c = grad_of_maxpool(b, None, None, [1, 2, 2, 1])
	grad = c.op.compute(c, [mx, res, ones(mx.shape)])
	# print(grad.shape)
	print(img[0, :, :, 0])
	print(grad[0, :, :, 0])
	d = grad_of_conv2d(a.input[0], a.input[1], None, [1, 1, 1, 1], a.padding)
	e = grad_toW_ofconv2d(a.input[0], a.input[1], None, [1, 1, 1, 1], a.padding)
	grad1 = d.op.compute(d, [img, flt, grad])
	# print("grad:\n", grad1.shape)
	# print(grad1[0, :, :, 0])
	grad2 = e.op.compute(e, [img, flt, grad])
	print("grad:", grad2.shape, end = "\n")
	print(grad2[:, :, 0, 0])
