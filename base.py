#！/user/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import ctypes

ndpointer = np.ctypeslib.ndpointer
c_float32 = ctypes.c_float
c_int32 = ctypes.c_int32
c_bool = ctypes.c_bool

clib = ctypes.cdll.LoadLibrary("tensorlow/nn_c.so")
# clib = ctypes.cdll.LoadLibrary("./nn_c.so")
max_pool_c = clib.max_pool
backup_max_pool_c = clib.backup_max_pool
max_pool_c.argtypes = [ndpointer(c_float32)] * 2 + [ndpointer(c_int32)] + [c_int32] * 6
backup_max_pool_c.argtypes = [ndpointer(c_int32)] + [ndpointer(c_float32)] * 2 + [c_int32] * 6

conv2d_c = clib.conv2d
backup_conv2d_filter_c = clib.backup_conv2d_filter
backup_conv2d_image_c = clib.backup_conv2d_image
conv2d_c.argtypes = [ndpointer(c_float32)] * 3 + [c_int32] * 7 + [c_bool]
backup_conv2d_filter_c.argtypes = backup_conv2d_image_c.argtypes = conv2d_c.argtypes

matmul_c = clib.Matmul
matmul_c.argtypes = [ndpointer(c_float32)] * 3 + [c_int32] * 3

sign_c = clib.sgn
relu_c = clib.relu
relu_c.argtypes = sign_c.argtypes = [ndpointer(c_float32)] * 2 + [c_int32]



float16 = np.float16
float32 = np.float32
float64 = np.float64
float128 = np.float128

int8 = np.int8
int16 = np.int16
int32 = np.int32
int64 = np.int64

uint8 = np.uint8
uint16 = np.uint16
uint32 = np.uint32
uint64 = np.uint64

zeros = np.zeros
ones = np.ones

def random_normal(shape,mean=0.0,stddev=1.0,dtype=float32):
	return np.random.normal(mean, stddev, shape).astype(dtype)





