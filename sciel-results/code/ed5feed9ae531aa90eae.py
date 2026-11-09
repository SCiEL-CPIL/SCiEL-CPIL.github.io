import random
import itertools
from functools import lru_cache

def construct():
    n = 18
    edges = []
    
    # Define Fano plane incidence
    # Points 0-6, Lines 0-6
    # We will use points 0-5 for the 6 vertices in each part
    fano_lines = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # Check if point p is on line l
    def is_on_line(p, l):
        return p in fano_lines[l]
        
    # Parts
    A = list(range(0, 6))
    B = list(range(6, 12))
    C = list(range(12, 18))
    
    # Internal edges: Perfect matchings
    # A: (0,1), (2,3), (4,5) -> mapped to 0-5
    # B: (6,7), (8,9), (10,11) -> mapped to 0-5
    # C: (12,13), (14,15), (16,17) -> mapped to 0-5
    
    internal_edges = []
    for base in [0, 6, 12]:
        for i in range(3):
            u = base + 2*i
            v = base + 2*i + 1
            internal_edges.append([u, v])
            
    edges.extend(internal_edges)
    
    # Cross edges using Fano structure
    # Between A and B:
    # Vertex a in A (index a_idx in 0..5) is connected to b in B (index b_idx in 0..5)
    # if a_idx is on line b_idx in Fano
    for a_idx in range(6):
        for b_idx in range(6):
            if is_on_line(a_idx, b_idx):
                edges.append([a_idx, 6 + b_idx])
                
    # Between B and C:
    for b_idx in range(6):
        for c_idx in range(6):
            if is_on_line(b_idx, c_idx):
                edges.append([6 + b_idx, 12 + c_idx])
                
    # Between C and A:
    for c_idx in range(6):
        for a_idx in range(6):
            if is_on_line(c_idx, a_idx):
                edges.append([12 + c_idx, a_idx])
                
    # Build adjacency
    adj = [[False]*n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        
    # Verify K4-free
    def has_k4():
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i][j]:
                    continue
                for k in range(j+1, n):
                    if not adj[i][k] or not adj[j][k]:
                        continue
                    for l in range(k+1, n):
                        if adj[i][l] and adj[j][l] and adj[k][l]:
                            return True
        return False
        
    if has_k4():
        # Fallback: Return K6,6,6 (Score 12) if K4 found
        parts = [range(0, 6), range(6, 12), range(12, 18)]
        return [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts[i] for b in parts[j]]
        
    # Compute max triangle-free induced subgraph size
    def is_triangle_free(vertex_set):
        vs = list(vertex_set)
        for i in range(len(vs)):
            for j in range(i+1, len(vs)):
                for k in range(j+1, len(vs)):
                    u, v, w = vs[i], vs[j], vs[k]
                    if adj[u][v] and adj[v][w] and adj[u][w]:
                        return False
        return True
        
    def max_triangle_free():
        best = 0
        # Check sizes from 18 down to 1
        # Optimization: Start from 12, since K6,6,6 has 12.
        # If we find a triangle-free set of size 12, return 12.
        # If not, check 11, etc.
        for size in range(12, 0, -1):
            for comb in itertools.combinations(range(n), size):
                if is_triangle_free(comb):
                    return size
        return 0
        
    score = max_triangle_free()
    
    return edges
