#！/user/bin/env python3
# -*- coding:utf-8 -*-

from tensorlow.base import float16, float32, float64, float128, \
                            int8, int16, int32, int64, \
                            uint8, uint16, uint32, uint64, \
                            random_normal, zeros, ones

from tensorlow.ops import placeholder, Variable, constant, \
                            global_variables_initializer, assign, \
                            reduce_mean, reduce_sum, reshape, \
                            matmul, exp, log, \
                            gradients, equal, argmax, cast

from tensorlow.Session import Session
from tensorlow.nn import nn as nn
import tensorlow.train


