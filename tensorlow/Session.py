#!/user/bin/env python3
# -*- coding:utf-8 -*-

from tensorlow.ops import *


class Session:

	def __init__(self, target = '', graph = None, config = None):
		self.target = target
		self.graph = graph
		self.config = config

	def __enter__(self):
		return default_session

	def __exit__(self, exc_type, exc_val, exc_tb):
		if exc_type :
			print(exc_type)
			print(exc_val)
			print(exc_tb)

	@staticmethod
	def _run(output, node_value):
		topo_order = find_topo_sort(output)
		for node in topo_order:
			if isinstance(node.op, PlaceHolderOp):
				continue
			val = []
			for inputs in node.input:
				val.append(node_value[inputs])
			node_value[node] = node.op.compute(node, val)
		return [node_value[node] for node in output]

	def run(self, fetches, feed_dict = None):

		# value_list = placeholder.value_list.copy()
		value_list = placeholder.value_list
		if feed_dict:
			for i, j in feed_dict.items():
				if not i in placeholder.placeholder_list:
					raise NameError
				if isinstance(j, (list, np.ndarray)):
					value_list[i] = np.array(j, dtype = i.dtype)
				else:
					value_list[i] = i.dtype(j)


		if isinstance(fetches, (list, tuple)):
			result = self._run(fetches, value_list)
		elif isinstance(fetches, dict):
			result = dict(zip(fetches.keys(), self.run(fetches.values(), value_list)))
		else:
			result = self._run([fetches], value_list)[0]

		return result



default_session = Session()
