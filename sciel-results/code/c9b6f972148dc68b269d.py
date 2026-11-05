import time
import numpy as np
import random

def construct():
    n = 40
    deadline = 115.0  # Safety margin from 120s limit (evaluator kills at 150s)
    start_time = time.time()
    
    # Helper to check if adding edge (u,v) creates a C6
    # We maintain a bitset representation of the graph
    # A[u] is a set of neighbors of u
    # To check if adding (u,v) creates a C6, we need to ensure there is no path of length 5 between u and v.
    # Equivalently, in the current graph G, dist(u,v) >= 6.
    # If we add (u,v), we create a cycle of length L+1 where L is the length of the shortest path between u and v in G.
    # We need L+1 != 6, so L != 5.
    # So we must ensure there is no path of length 5 between u and v.
    # Actually, if there is a path of length 5, adding the edge creates a C6.
    # If there is a path of length < 5, adding the edge creates a shorter cycle (C3, C4, C5), which is allowed (only C6 is forbidden).
    # Wait, the task says "containing no six-cycle". It doesn't say "containing no cycles of length <= 6".
    # It says "Hexagon-free". So C3, C4, C5 are fine.
    # So the condition is: there is no path of length exactly 5 between u and v?
    # No. If there is a path of length 5, adding the edge creates a C6.
    # If there is a path of length 4, adding the edge creates a C5. This is allowed.
    # So we just need to check if there is a path of length 5.
    
    # Efficient check:
    # Let N(u) be neighbors of u.
    # Let N2(u) be neighbors of N(u).
    # ...
    # Let P5(u) be vertices reachable from u by a path of length exactly 5? No, that's hard to compute exactly without counting.
    # Actually, we can just do a BFS/DFS limited to depth 5 from u and see if v is reached at exactly depth 5?
    # No, if v is reached at depth 3, and also at depth 5, adding the edge creates a C4 and a C6.
    # So we MUST ensure v is NOT reachable by ANY path of length 5.
    
    # Given n=40, we can compute A^5 quickly using numpy int64.
    # But doing it for every candidate edge is expensive.
    # Alternative: Maintain the adjacency matrix A.
    # When considering adding (u,v), check if (A^5)[u,v] > 0.
    
    # Let's try a different structural approach first to get a good starting point.
    # The current seed uses 8 disjoint K5s. This has 8 * 10 = 80 edges.
    # Can we do better?
    # A known construction for C6-free graphs is the polarity graph of a finite projective plane.
    # For n=40, this is tricky.
    
    # Let's stick to local search but optimize the check.
    # Instead of recomputing A^5 from scratch for every edge addition, we can update it incrementally?
    # A^5 update is complex.
    
    # Let's try a randomized restart strategy with a faster check.
    # Precompute all-pairs shortest paths? No, we need path of length 5 specifically.
    
    # Let's use the numpy matrix multiplication approach but optimize the loop.
    # The bottleneck in the current program is likely `compute_A5` being called too often.
    
    def compute_A5(mat):
        # mat is 40x40 int64
        A2 = mat @ mat
        A3 = A2 @ mat
        A4 = A3 @ mat
        A5 = A4 @ mat
        return A5

    def initial_random():
        # Start with a random sparse graph
        A = np.zeros((n, n), dtype=np.int64)
        # Add some random edges
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.1:
                    A[i, j] = 1
                    A[j, i] = 1
        return A

    best_A = initial_random()
    best_count = np.sum(best_A) // 2
    
    # Try a few different initializations
    # 1. Disjoint K5s
    A_k5 = np.zeros((n, n), dtype=np.int64)
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u = 5 * k + a
                v = 5 * k + b
                A_k5[u, v] = 1
                A_k5[v, u] = 1
    if np.sum(A_k5) // 2 > best_count:
        best_A = A_k5
        best_count = np.sum(best_A) // 2

    # 2. Random dense
    A_rand = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.2:
                A_rand[i, j] = 1
                A_rand[j, i] = 1
    if np.sum(A_rand) // 2 > best_count:
        best_A = A_rand
        best_count = np.sum(best_A) // 2

    # Local Search
    while time.time() < deadline:
        A = best_A.copy()
        A5 = compute_A5(A)
        
        # Greedy Add
        improved = True
        while improved and time.time() < deadline:
            improved = False
            non_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if A[u, v] == 0:
                        non_edges.append((u, v))
            random.shuffle(non_edges)
            
            for u, v in non_edges:
                if time.time() > deadline:
                    break
                if A[u, v] == 1:
                    continue
                if A5[u, v] == 0:
                    A[u, v] = 1
                    A[v, u] = 1
                    A5 = compute_A5(A)
                    improved = True
                    current_count = np.sum(A) // 2
                    if current_count > best_count:
                        best_count = current_count
                        best_A = A.copy()
        
        # Kicks (Swaps)
        if time.time() > deadline:
            break
        
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if A[u, v] == 1:
                    edges.append((u, v))
        
        if not edges:
            continue
            
        for _ in range(10): # Fewer swaps per restart to allow more restarts
            if time.time() > deadline:
                break
            rem_u, rem_v = random.choice(edges)
            A_old = A.copy()
            A[rem_u, rem_v] = 0
            A[rem_v, rem_u] = 0
            A5 = compute_A5(A)
            
            added = 0
            non_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if A[u, v] == 0:
                        non_edges.append((u, v))
            random.shuffle(non_edges)
            
            for u, v in non_edges:
                if time.time() > deadline:
                    break
                if added >= 2:
                    break
                if A[u, v] == 1:
                    continue
                if A5[u, v] == 0:
                    A[u, v] = 1
                    A[v, u] = 1
                    A5 = compute_A5(A)
                    added += 1
            
            current_count = np.sum(A) // 2
            if current_count > best_count:
                best_count = current_count
                best_A = A.copy()
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if A[u, v] == 1:
                            edges.append((u, v))
            else:
                A = A_old

    # Return best found
    edges_out = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_A[u, v] == 1:
                edges_out.append([u, v])
    return edges_out
