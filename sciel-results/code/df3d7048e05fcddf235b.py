import time
import random
import math
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s buffer
    
    q = 5
    n = 31
    
    # 1. Generate 31 points of PG(2,5)
    points = []
    points.append((1, 0, 0))
    points.append((0, 1, 0))
    points.append((0, 0, 1))
    for y in range(1, q):
        points.append((1, y, 0))
    for z in range(1, q):
        points.append((1, 0, z))
    for z in range(1, q):
        points.append((0, 1, z))
    for y in range(1, q):
        for z in range(1, q):
            points.append((1, y, z))
            
    # 2. Construct Polarity Graph
    # Adjacency as list of sets for fast intersection
    adj = [set() for _ in range(n)]
    base_edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            p_i = points[i]
            p_j = points[j]
            if (p_i[0]*p_j[0] + p_i[1]*p_j[1] + p_i[2]*p_j[2]) % q == 0:
                adj[i].add(j)
                adj[j].add(i)
                base_edges.append((i, j))
                
    target_edges = 91
    edges_to_add = target_edges - len(base_edges) # Should be 1
    
    # If edges_to_add is not 1, we might need a more complex search.
    # But for q=5, it is exactly 1.
    
    if edges_to_add != 1:
        # Fallback to greedy + SA if the assumption breaks
        # But for this specific task, it should be 1.
        pass

    # Identify non-edges
    non_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if j not in adj[i]:
                non_edges.append((i, j))
                
    best_edge = None
    min_cost = float('inf')
    
    # Precompute neighbor sets for speed (already in adj)
    # For each non-edge (u, v), calculate the number of new C4s
    # Cost = number of edges (x, y) in G such that x in N(v) and y in N(u)
    
    for u, v in non_edges:
        A = adj[v]
        B = adj[u]
        
        # S = sum_{x in A} |N(x) intersect B|
        S = 0
        for x in A:
            S += len(adj[x] & B)
            
        C = A & B
        # E_CC = number of edges within C
        # Count degrees in induced subgraph on C
        deg_sum = 0
        for x in C:
            deg_sum += len(adj[x] & C)
        E_CC = deg_sum // 2
        
        cost = S - E_CC
        
        if cost < min_cost:
            min_cost = cost
            best_edge = (u, v)
            
    # If no non-edge found (should not happen), return base edges + any
    if best_edge is None:
        best_edge = non_edges[0] if non_edges else (0, 1)
        
    final_edges = []
    for u, v in base_edges:
        final_edges.append([u, v])
    final_edges.append([best_edge[0], best_edge[1]])
        
    return final_edges
