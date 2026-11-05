import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0
    
    n = 31
    q = 5
    mod = q
    
    # Enumerate all points in PG(2, q)
    # Points are [x:y:z] with not all zero, modulo scaling by nonzero elements
    # Canonical form: first nonzero coordinate is 1
    points = []
    # Case 1: x != 0, canonical x = 1
    for y in range(q):
        for z in range(q):
            points.append((1, y, z))
    # Case 2: x = 0, y != 0, canonical y = 1
    for z in range(q):
        points.append((0, 1, z))
    # Case 3: x = 0, y = 0, z != 0, canonical z = 1
    points.append((0, 0, 1))
    
    assert len(points) == q*q + q + 1, f"Expected {q*q+q+1}, got {len(points)}"
    
    # Build adjacency: (x1,y1,z1) ~ (x2,y2,z2) if x1*x2 + y1*y2 + z1*z2 == 0 mod q
    adj = [set() for _ in range(n)]
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, z1 = points[i]
            x2, y2, z2 = points[j]
            if (x1*x2 + y1*y2 + z1*z2) % mod == 0:
                adj[i].add(j)
                adj[j].add(i)
                edges.append([i, j])
    
    m = len(edges)
    # print(f"Polarity graph has {m} edges")
    
    # We need exactly 91 edges
    if m < 91:
        # Need to add edges. Find the non-edge that creates the fewest new 4-cycles.
        # A new 4-cycle containing (u,v) is formed by a path u-a-b-v where a~u, b~v, a~b.
        # So we count edges between N(u) and N(v).
        
        def count_new_c4s(u, v, adj):
            # u and v are not adjacent
            # Count pairs (a, b) such that a in N(u), b in N(v), and a ~ b
            count = 0
            nu = adj[u]
            nv = adj[v]
            # Iterate over smaller set for efficiency? Or just nested loop.
            # n=31, degrees are small (~6).
            for a in nu:
                for b in nv:
                    if a == b:
                        continue
                    if b in adj[a]:
                        count += 1
            return count

        # Find all non-edges
        edge_set = set()
        for e in edges:
            edge_set.add((e[0], e[1]) if e[0] < e[1] else (e[1], e[0]))
        
        best_cost = float('inf')
        best_edge = None
        for i in range(n):
            for j in range(i+1, n):
                if (i, j) not in edge_set:
                    cost = count_new_c4s(i, j, adj)
                    if cost < best_cost:
                        best_cost = cost
                        best_edge = [i, j]
        
        if best_edge is not None:
            edges.append(best_edge)
            u, v = best_edge
            adj[u].add(v)
            adj[v].add(u)
            
        # If we still need more edges (unlikely for this specific graph size), repeat
        while len(edges) < 91:
             # Re-evaluate non-edges based on current adj
             edge_set = set()
             for e in edges:
                 edge_set.add((e[0], e[1]) if e[0] < e[1] else (e[1], e[0]))
             
             best_cost = float('inf')
             best_edge = None
             for i in range(n):
                 for j in range(i+1, n):
                     if (i, j) not in edge_set:
                         cost = count_new_c4s(i, j, adj)
                         if cost < best_cost:
                             best_cost = cost
                             best_edge = [i, j]
             
             if best_edge is not None:
                 edges.append(best_edge)
                 u, v = best_edge
                 adj[u].add(v)
                 adj[v].add(u)
             else:
                 break # No non-edges left? Should not happen for n=31, m=91

    elif m > 91:
        # Remove edges that are in the fewest 4-cycles
        # Calculate C4 participation for each edge
        edge_c4_count = {}
        for e in edges:
            u, v = e[0], e[1]
            count = 0
            # 4-cycle u-v-a-b-u. a~v, b~u, a~b.
            for a in adj[v]:
                if a == u: continue
                for b in adj[u]:
                    if b == v or b == a: continue
                    if b in adj[a]:
                        count += 1
            edge_c4_count[(u,v) if u<v else (v,u)] = count
        
        sorted_edges = sorted(edges, key=lambda e: edge_c4_count[(e[0],e[1]) if e[0]<e[1] else (e[1],e[0])])
        edges_to_remove = sorted_edges[:m-91]
        remove_set = set()
        for e in edges_to_remove:
            remove_set.add((e[0], e[1]) if e[0]<e[1] else (e[1], e[0]))
        
        edges = []
        for e in edges:
            key = (e[0], e[1]) if e[0] < e[1] else (e[1], e[0])
            if key not in remove_set:
                edges.append(e)
                # Update adj if we were to use it further, but we just return edges
                # So we don't strictly need to update adj for the return value,
                # but if we did local search, we would.
                
    assert len(edges) == 91, f"Expected 91 edges, got {len(edges)}"
    
    return edges
