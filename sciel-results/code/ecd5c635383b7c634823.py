import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n = 31
    target_edges = 91
    
    # Points of PG(2,5) are 1-dim subspaces of GF(5)^3.
    # We represent them as vectors (x, y, z) in GF(5)^3, normalized such that the first non-zero component is 1.
    # There are 31 such points.
    
    points = []
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize
                if x != 0:
                    points.append((1, (y * pow(x, 3, 5)) % 5, (z * pow(x, 3, 5)) % 5))
                elif y != 0:
                    points.append((0, 1, (z * pow(y, 3, 5)) % 5))
                else:
                    points.append((0, 0, 1))
    
    # Map points to indices 0..30
    point_to_idx = {p: i for i, p in enumerate(points)}
    
    # Polarity matrix for q=5
    # M = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    # Two points P=(x1,y1,z1) and Q=(x2,y2,z2) are adjacent if P^T M Q = 0
    # P^T M Q = x1*y2 + y1*x2 + y1*z2 + z1*y2 = 0 (mod 5)
    
    adj = [0] * n
    edge_list = []
    
    for i in range(n):
        p1 = points[i]
        for j in range(i + 1, n):
            p2 = points[j]
            # Calculate P1^T M P2
            val = (p1[0] * p2[1] + p1[1] * p2[0] + p1[1] * p2[2] + p1[2] * p2[1]) % 5
            if val == 0:
                # Add edge
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                edge_list.append([i, j])
    
    # Check if we have 90 edges
    if len(edge_list) != 90:
        # Fallback to random greedy if construction fails
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_pairs)
        adj = [0] * n
        edge_list = []
        for u, v in all_pairs:
            if len(edge_list) >= 90:
                break
            creates_c4 = False
            au = adj[u]
            while au:
                lsb = au & (-au)
                a_idx = lsb.bit_length() - 1
                if adj[a_idx] & adj[v]:
                    creates_c4 = True
                    break
                au ^= lsb
            if not creates_c4:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_list.append([u, v])
        
        if len(edge_list) < 90:
             # Should not happen
             return [[i, j] for i in range(n) for j in range(i + 1, n)][:91]

    # Find the best non-edge to add
    min_cost = float('inf')
    best_edge = None
    
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    for u, v in all_pairs:
        if adj[u] & (1 << v):
            continue
        
        # Count C4s created by adding (u, v)
        # C4 is u-v-w-x-u.
        # w in N(v), x in N(u), and (w, x) is an edge.
        # Cost = number of edges between N(u) and N(v).
        cost = 0
        au = adj[u]
        while au:
            lsb = au & (-au)
            x_idx = lsb.bit_length() - 1
            # Count common neighbors of x_idx and v? No.
            # We need edges (w, x) where w in N(v) and x in N(u).
            # So for each x in N(u), count how many w in N(v) are connected to x.
            # That is popcount(adj[x] & adj[v]).
            cost += (adj[x_idx] & adj[v]).bit_count()
            au ^= lsb
        
        if cost < min_cost:
            min_cost = cost
            best_edge = [u, v]
            if min_cost == 1: # Theoretical minimum is 1
                break
    
    if best_edge:
        edge_list.append(best_edge)
    
    return edge_list
