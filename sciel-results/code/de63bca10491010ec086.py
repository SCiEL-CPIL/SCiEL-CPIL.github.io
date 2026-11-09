"""Rewritten program: construct() returns 91 edges on vertices 0..30 minimizing C4s.
Uses a finite-field construction based on q=5 (n = q^2+q+1 = 31) to get a C4-free core,
then adds one edge chosen to minimize new C4s.
"""
import numpy as np
import time

def construct():
    t0 = time.time()
    deadline = t0 + 110.0

    q = 5
    mod = 5
    
    # 1. Generate points of PG(2, q)
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: first non-zero is 1
                if x != 0:
                    inv_x = pow(x, mod - 2, mod)
                    nx, ny, nz = 1, (y * inv_x) % mod, (z * inv_x) % mod
                elif y != 0:
                    inv_y = pow(y, mod - 2, mod)
                    nx, ny, nz = 0, 1, (z * inv_y) % mod
                else:
                    nx, ny, nz = 0, 0, 1
                
                key = (nx, ny, nz)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    n = len(points)
    if n != 31:
        return _fallback_construct()

    # 2. Build Polarity Graph
    # Bilinear form: x1*x2 + y1*y2 - z1*z2 = 0
    adj = [set() for _ in range(n)]
    edges = []
    
    for i in range(n):
        x1, y1, z1 = points[i]
        for j in range(i + 1, n):
            x2, y2, z2 = points[j]
            val = (x1 * x2 + y1 * y2 - z1 * z2) % mod
            if val == 0:
                adj[i].add(j)
                adj[j].add(i)
                edges.append([i, j])
    
    # 3. Add 1 edge to minimize C4s
    best_edge = None
    min_paths = float('inf')
    
    for u in range(n):
        if time.time() > deadline - 10:
            break
        for v in range(u + 1, n):
            if v in adj[u]:
                continue
            count = 0
            for a in adj[u]:
                for b in adj[v]:
                    if b in adj[a]:
                        count += 1
            if count < min_paths:
                min_paths = count
                best_edge = [u, v]
                if min_paths == 0:
                    break
        if min_paths == 0:
            break
    
    if best_edge is None:
        for u in range(n):
            for v in range(u + 1, n):
                if v not in adj[u]:
                    best_edge = [u, v]
                    break
            if best_edge:
                break
        
    if best_edge:
        edges.append(best_edge)
        adj[best_edge[0]].add(best_edge[1])
        adj[best_edge[1]].add(best_edge[0])
        
    if len(edges) != 91:
        return _fallback_construct()
        
    return edges

def _fallback_construct():
    import random
    random.seed(42)
    n = 31
    edges = []
    adj = [set() for _ in range(n)]
    
    candidates = [[i, j] for i in range(n) for j in range(i + 1, n)]
    random.shuffle(candidates)
    
    for _ in range(91):
        best_e = None
        best_add = float('inf')
        for e in candidates:
            u, v = e
            if v in adj[u]:
                continue
            add = 0
            for a in adj[u]:
                for b in adj[v]:
                    if b in adj[a]:
                        add += 1
            if add < best_add:
                best_add = add
                best_e = e
            if add == 0:
                break
        if best_e:
            edges.append(best_e)
            adj[best_e[0]].add(best_e[1])
            adj[best_e[1]].add(best_e[0])
            candidates.remove(best_e)
        else:
            break
            
    while len(edges) < 91:
        for e in candidates:
            u, v = e
            if v not in adj[u]:
                edges.append(e)
                adj[u].add(v)
                adj[v].add(u)
                candidates.remove(e)
                break
        if not candidates:
            break
            
    return edges[:91]
