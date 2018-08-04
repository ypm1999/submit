#！/user/bin/env python3
# -*- coding:utf-8 -*-


def find_topo_sort(node_list):
	visited = set()
	topo_order = []
	for node in node_list:
		topo_sort_dfs(node, visited, topo_order)
	return topo_order

def topo_sort_dfs(node, visited, topo_order):
	if node in visited:
		return
	visited.add(node)
	for n in node.input:
		topo_sort_dfs(n, visited, topo_order)
	topo_order.append(node)


from operator import add
from functools import reduce
def sum_node_list(node_list):
	if node_list == None:
		return None
	return reduce(add, node_list)