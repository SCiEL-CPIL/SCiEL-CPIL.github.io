"""Combined program: structured random search with local optimization for rank-minimal tensor decomposition."""
import time
import random
import math
import numpy as np

DIMENSIONS = [2, 3, 7]

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14
    
    # Start with schoolbook as fallback
    best = _schoolbook()
    best_rank = len(best["U"][0]) if best["U"] else 0
    
    # Target support: for valid (i,j,k), the tensor value is 1
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Local search approach with error minimization
    def compute_error(U, V, W, rank):
        """Compute sum of squared errors for all tensor entries."""
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    err += (s - 1.0) ** 2
        return err
    
    def local_search(U, V, W, rank, values, max_iters=10000):
        """Perform local search to minimize error."""
        current_err = compute_error(U, V, W, rank)
        if current_err < 1e-9:
            return U, V, W, True
        
        stuck_counter = 0
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick random matrix and entry to modify
            mat_idx = random.choice([0, 1, 2])
            if mat_idx == 0:
                mat = U
                dims = nm
            elif mat_idx == 1:
                mat = V
                dims = mp
            else:
                mat = W
                dims = pn
            
            idx = random.randrange(dims)
            col = random.randrange(rank)
            
            old_val = mat[idx][col]
            new_val = random.choice(values)
            
            if new_val == old_val:
                continue
            
            mat[idx][col] = new_val
            new_err = compute_error(U, V, W, rank)
            
            if new_err < current_err:
                current_err = new_err
                stuck_counter = 0
                if current_err < 1e-9:
                    return U, V, W, True
            else:
                mat[idx][col] = old_val
                stuck_counter += 1
            
            if stuck_counter > 50:
                break
        
        return U, V, W, current_err < 1e-9
    
    # Try different ranks from lower to higher
    # Minimum possible rank is at least max(nm, mp, pn) = 21
    # Try ranks from 21 upward
    
    values_int = [0, 1, -1]
    values_half = [0, 0.5, -0.5, 1, -1, 1.5, -1.5, 2, -2]
    
    for rank in range(21, best_rank):
        if time.time() > deadline:
            break
        
        # Try integer values first
        found = False
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            U = [[random.choice(values_int) for _ in range(rank)] for _ in range(nm)]
            V = [[random.choice(values_int) for _ in range(rank)] for _ in range(mp)]
            W = [[random.choice(values_int) for _ in range(rank)] for _ in range(pn)]
            
            U, V, W, success = local_search(U, V, W, rank, values_int, max_iters=2000)
            if success:
                if _verify(U, V, W):
                    best = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    found = True
                    break
        
        if found:
            break
        
        # If integer failed, try half-integers
        if time.time() > deadline:
            break
        
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            U = [[random.choice(values_half) for _ in range(rank)] for _ in range(nm)]
            V = [[random.choice(values_half) for _ in range(rank)] for _ in range(mp)]
            W = [[random.choice(values_half) for _ in range(rank)] for _ in range(pn)]
            
            U, V, W, success = local_search(U, V, W, rank, values_half, max_iters=2000)
            if success:
                if _verify(U, V, W):
                    best = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    found = True
                    break
        
        if found:
            break
    
    return best

# EVOLVE-BLOCK-END
