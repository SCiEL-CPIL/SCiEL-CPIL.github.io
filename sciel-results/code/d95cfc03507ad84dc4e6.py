import time
import random
from itertools import combinations, product

def _count_distinct_cycle_lengths(adj, n):
    """
    Enumerate all simple cycles in an undirected graph and return
    the set of distinct cycle lengths.
    For speed, we only explore cycles where the start vertex is the
    smallest vertex in the cycle (to avoid counting each cycle multiple times).
    This is a standard trick: only start DFS from vertex v, and only visit
    vertices with index > v.
    """
    lengths = set()
    for start in range(n):
        # DFS: path is a list of vertices, stack-based for efficiency
        # We require all vertices in the path to be > start (except start itself)
        # and the last vertex must connect back to start.
        # Use iterative DFS to avoid recursion depth issues, though n=12 is small.
        
        # State: (current, path)
        # We start with path = [start]
        stack = [(start, [start])]
        while stack:
            curr, path = stack.pop()
            for nb in adj[curr]:
                if nb == start:
                    if len(path) >= 3:  # Cycle must have length >= 3 (actually >=4 due to triangle-free)
                        lengths.add(len(path))
                elif nb > start:
                    # Only visit vertices with index > start to ensure uniqueness
                    if nb not in path:
                        stack.append((nb, path + [nb]))
    return lengths

def construct():
    n = 12
    
    # Theoretical minimum for distinct cycle lengths in a triangle-free graph
    # with minimum degree 3 is 2.
    # A known construction achieving this is the disjoint union of two K_{3,3} graphs.
    # K_{3,3} has cycle lengths 4 and 6.
    # Thus, the set of cycle lengths is {4, 6}, size 2.
    # Score = seed / value. If seed is 5, score is 2.5.
    
    edges = []
    
    # First K_{3,3} on vertices 0, 1, 2 and 3, 4, 5
    for u in range(3):
        for v in range(3, 6):
            edges.append([u, v])
            
    # Second K_{3,3} on vertices 6, 7, 8 and 9, 10, 11
    for u in range(6, 9):
        for v in range(9, 12):
            edges.append([u, v])
            
    return edges
