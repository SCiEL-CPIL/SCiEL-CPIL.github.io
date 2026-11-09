import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    N = 180
    
    # Generate all forbidden triples (a, b, c) with a < b < c
    # a = kuv, b = ku(u+v), c = kv(u+v)
    # u, v coprime, u != v
    # a, b, c <= N
    
    triples = []
    # Max u, v such that k u v <= N
    # k >= 1, so u v <= N
    for u in range(1, N + 1):
        for v in range(1, N + 1):
            if u == v:
                continue
            if u > v:
                continue # WLOG u < v to avoid duplicates? 
                         # Actually (u,v) and (v,u) generate same set {a,b,c} but swapped b,c.
                         # We just need the set of vertices.
                         # Let's just iterate all u, v coprime u != v.
            
            # Check coprime
            if math.gcd(u, v) != 1:
                continue
                
            # Max k
            # a = k u v <= N
            # b = k u (u+v) <= N
            # c = k v (u+v) <= N
            
            k_max_a = N // (u * v)
            k_max_b = N // (u * (u + v))
            k_max_c = N // (v * (u + v))
            
            k_max = min(k_max_a, k_max_b, k_max_c)
            
            for k in range(1, k_max + 1):
                a = k * u * v
                b = k * u * (u + v)
                c = k * v * (u + v)
                
                if a <= N and b <= N and c <= N:
                    # Ensure distinct
                    if a != b and a != c and b != c:
                        triples.append((a, b, c))
                        
    # Remove duplicates
    triples = list(set(triples))
    
    # Map vertex to list of triple indices
    vertex_triples = {i: [] for i in range(1, N + 1)}
    for idx, t in enumerate(triples):
        for v in t:
            vertex_triples[v].append(idx)
            
    # Greedy Hitting Set
    # Start with all vertices
    selected = set(range(1, N + 1))
    active_triples = set(range(len(triples)))
    
    # Count occurrences in active triples
    counts = {v: len(vertex_triples[v]) for v in range(1, N + 1)}
    
    while active_triples and time.time() < deadline:
        # Find vertex with max count
        # Optimization: use heap? For 180 vertices, linear scan is fine.
        max_count = -1
        max_vertex = -1
        for v in selected:
            if counts[v] > max_count:
                max_count = counts[v]
                max_vertex = v
                
        if max_vertex == -1:
            break
            
        # Remove max_vertex
        selected.remove(max_vertex)
        
        # Update active_triples and counts
        to_remove_triples = set(vertex_triples[max_vertex])
        # Filter active_triples
        new_active = active_triples - to_remove_triples
        removed = active_triples - new_active
        
        for t_idx in removed:
            t = triples[t_idx]
            for v in t:
                if v in selected:
                    counts[v] -= 1
                    
        active_triples = new_active
        
    # Convert to sorted list
    result = sorted(list(selected))
    
    return result

import math
