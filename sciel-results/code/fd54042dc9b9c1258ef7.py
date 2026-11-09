import time
import itertools
import random

N = 12
MIN_DEG = 3
GIRTH = 4

def construct():
    start_time = time.time()
    deadline = start_time + 45  # 45s safety margin
    
    # The optimal solution for this problem is two disjoint K_{3,3} graphs.
    # Each K_{3,3} is bipartite, hence triangle-free (girth >= 4).
    # The simple cycles in K_{3,3} have lengths 4 and 6 only.
    # Thus the set of distinct cycle lengths is {4, 6}, size 2.
    # It is impossible to have size 1 because a graph with a single cycle length L 
    # would need all cycles to be length L. If L=4, it's a disjoint union of C4s, 
    # but min degree 3 requires more edges, creating other cycles.
    # So 2 is the theoretical minimum.
    
    edges = []
    
    # First K_{3,3}: partition {0,1,2} and {3,4,5}
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            if u < v:
                edges.append([u, v])
            else:
                edges.append([v, u])
                
    # Second K_{3,3}: partition {6,7,8} and {9,10,11}
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            if u < v:
                edges.append([u, v])
            else:
                edges.append([v, u])
                
    # Sort edges for determinism
    edges.sort()
    
    # Validation checks (quick, to ensure no bugs)
    n = 12
    deg = [0] * n
    adj = {i: set() for i in range(n)}
    
    for e in edges:
        u, v = e[0], e[1]
        if not isinstance(u, int) or not isinstance(v, int):
            return []
        if u == v or u < 0 or u >= n or v < 0 or v >= n:
            return []
        deg[u] += 1
        deg[v] += 1
        adj[u].add(v)
        adj[v].add(u)
        
    # Min degree check
    for i in range(n):
        if deg[i] < 3:
            return []
            
    # Triangle check (girth >= 4)
    for u in range(n):
        for v in adj[u]:
            if u < v:
                if adj[u].intersection(adj[v]):
                    return []
                    
    # Ensure it's not a tree (must contain a cycle)
    if len(edges) < n:
        return []

    if time.time() - start_time > deadline:
        return []

    return edges
