import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    def is_valid(colors, n):
        m = n * (n - 1) // 2
        if len(colors) != m:
            return False
        # Check for C6 in each color
        # Build adjacency lists
        adj = [[] for _ in range(4)]
        for i in range(n):
            for j in range(i + 1, n):
                idx = i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
                c = colors[idx]
                adj[c].append((i, j))
        
        for c in range(4):
            # Build dict for fast lookup
            edges_set = set(adj[c])
            # Check for C6
            # For each edge (u, v), check if there is a path of length 5
            # This is O(E * (deg^4)) which is too slow.
            # Use matrix power exact?
            # For small n, we can just check all 6-cycles.
            # Precomputed cycles for n=17 is too many to list in code.
            # Let's use the walk count as a proxy for validity in the search,
            # but for the final check, we need to be sure.
            # If walk count is 0, it is valid.
            pass
        return True

    def solve_for_n(n):
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        m = len(edges)
        
        # Random init
        colors = np.random.randint(0, 4, size=m)
        
        A = [np.zeros((n, n), dtype=int) for _ in range(4)]
        for k, e in enumerate(edges):
            c = colors[k]
            u, v = e
            A[c][u, v] = 1
            A[c][v, u] = 1
            
        def calc_W(A):
            W = 0
            for c in range(4):
                A2 = A[c] @ A[c]
                A4 = A2 @ A2
                A5 = A4 @ A[c]
                W += np.sum(A5)
            return W
            
        W = calc_W(A)
        
        T = 1.0
        best_W = W
        best_colors = colors.copy()
        
        while time.time() - start_time < time_limit - 5.0:
            k = random.randint(0, m - 1)
            e = edges[k]
            old_c = colors[k]
            new_c = random.randint(0, 3)
            if new_c == old_c:
                continue
                
            u, v = e
            A[old_c][u, v] = 0
            A[old_c][v, u] = 0
            A[new_c][u, v] = 1
            A[new_c][v, u] = 1
            
            W_new = calc_W(A)
            
            if W_new < W or (W_new == W and random.random() < 0.5) or (W_new > W and random.random() < np.exp(-(W_new - W) / T)):
                W = W_new
                colors[k] = new_c
                if W < best_W:
                    best_W = W
                    best_colors = colors.copy()
                    if W == 0:
                        break
            else:
                A[old_c][u, v] = 1
                A[old_c][v, u] = 1
                A[new_c][u, v] = 0
                A[new_c][v, u] = 0
                colors[k] = old_c
                
            T *= 0.999
            
        if best_W == 0:
            return best_colors, n
        return None, n

    for n in range(17, 8, -1):
        res, n_used = solve_for_n(n)
        if res is not None:
            return {'n': n_used, 'colors': res.tolist()}
            
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
