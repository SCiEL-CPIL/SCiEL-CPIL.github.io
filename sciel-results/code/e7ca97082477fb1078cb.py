"""
Construct K_{8,8} (complete bipartite graph) on 16 vertices.
This is optimal: 64 edges, diameter 2, every edge critical.
"""

import time
from collections import deque

P = {'n': 16}
START_TIME = time.time()
TIME_LIMIT = 85


def construct():
    n = P['n']
    
    # Construct K_{8,8}: partition vertices into two sets of 8
    # Part A: vertices 0-7, Part B: vertices 8-15
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
    
    return edges
