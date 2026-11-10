import numpy as np
import time
import random

def count_c4(A):
    """
    Calculate the number of 4-cycles in the graph represented by adjacency matrix A.
    A is a square numpy array of 0s and 1s.
    """
    A2 = A @ A
    # Sum of C(A2[i,j], 2) for all i,j
    # Each 4-cycle is counted 4 times (once for each pair of opposite vertices)
    # C(x, 2) = x * (x - 1) / 2
    total = np.sum(A2 * (A2 - 1))
    return total // 8

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    n = 31
    m = 91
    
    best_edges = None
    best_cost = float('inf')
    
    # Greedy construction with random tie-breaking
    for _ in range(200):
        if time.time() > deadline:
            break
            
        A = np.zeros((n, n), dtype=int)
        edges = []
        
        # Precompute list of all possible edges
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # We need to add m edges
        # We maintain A and update costs
        # To make it faster, we just pick randomly from non-edges with low cost
        # But calculating cost for all non-edges is O(n^2)
        # n=31, n^2=961. Very fast.
        
        for k in range(m):
            min_cost = float('inf')
            candidates = []
            
            # Calculate cost for all non-edges
            # Cost of adding (u, v) is C(c_uv, 2)
            # c_uv is the number of common neighbors of u and v
            # This is A[u] @ A[v]
            
            # Efficient way:
            # A2 = A @ A
            # c_uv = A2[u, v]
            A2 = A @ A
            
            for u in range(n):
                for v in range(u + 1, n):
                    if A[u, v] == 0:
                        c_uv = A2[u, v]
                        cost = c_uv * (c_uv - 1) // 2
                        if cost < min_cost:
                            min_cost = cost
                            candidates = [(u, v)]
                        elif cost == min_cost:
                            candidates.append((u, v))
            
            if not candidates:
                break # Should not happen
            
            # Randomly pick from candidates
            u, v = random.choice(candidates)
            A[u, v] = 1
            A[v, u] = 1
            edges.append([u, v])
            
        # Local search polish
        # Try to swap edges to reduce cost
        current_cost = count_c4(A)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_edges = [e[:] for e in edges]
            
        # Simple local search: random swaps
        # Time limit for local search
        ls_start = time.time()
        ls_limit = 0.5  # 0.5 seconds per greedy run
        
        while time.time() - ls_start < ls_limit and time.time() < deadline:
            # Pick random edge (u, v) to remove
            e_idx = random.randint(0, m - 1)
            u, v = edges[e_idx]
            
            # Pick random non-edge (x, y) to add
            # Ensure (x, y) is not an edge
            while True:
                x = random.randint(0, n - 1)
                y = random.randint(0, n - 1)
                if x == y:
                    continue
                if x > y:
                    x, y = y, x
                if A[x, y] == 0:
                    break
            
            # Perform swap
            A[u, v] = 0
            A[v, u] = 0
            A[x, y] = 1
            A[y, x] = 1
            
            new_cost = count_c4(A)
            
            if new_cost <= current_cost:
                # Accept
                current_cost = new_cost
                # Update edges list
                edges[e_idx] = [x, y]
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_edges = [e[:] for e in edges]
            else:
                # Revert
                A[u, v] = 1
                A[v, u] = 1
                A[x, y] = 0
                A[y, x] = 0
        
    if best_edges is None:
        # Fallback: return first 91 lexicographic edges
        best_edges = [[i, j] for i in range(n) for j in range(i + 1, n)][:m]
        
    return best_edges
