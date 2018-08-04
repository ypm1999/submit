#！/user/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import ctypes


if __name__  == "__main__":
	img = np.array([[[[1], [3], [8], [0], [3]],
			         [[0], [1], [5], [1], [0]],
			         [[9], [0], [1], [7], [1]],
			         [[0], [8], [1], [4], [0]],
			         [[8], [1], [1], [0], [1]]]]*2, dtype = np.float32)

	o = 2
	flt = np.array([ [ [[1] * o], [[0] * o], [[1] * o] ],
			         [ [[0] * o], [[1] * o], [[0] * o] ],
			         [ [[1] * o], [[0] * o], [[1] * o] ]], dtype = np.float32)

	fflt = flt[0:2, 0:2]
	test = ctypes.cdll.LoadLibrary("./nn.so")
	num, n, m, ins = img.shape
	fn, fm, fin, fout = flt.shape
	result = np.ones(shape = (num * n * m * fout), dtype = np.float32)
	img = img.reshape((-1))
	flt = flt.reshape((-1))
	#result = result.reshape((-1))
	test.run.argtypes = [np.ctypeslib.ndpointer(ctypes.c_float)] * 3 + [ctypes.c_int] * 7
	test.run(img, flt, result, num, n, m, fn, fm, fin, fout)
	print("---------result---------")
	result = result.reshape((num, n, m, fout))
	for i in range(num):
		for j in range(fout):
			print(result[i, :, :, j])


