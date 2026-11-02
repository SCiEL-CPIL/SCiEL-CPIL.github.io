"""Rewritten program for matrix multiplication 3,5,6 rank-minimal decomposition."""
import time
import numpy as np
import random

DIMENSIONS = [3, 5, 6]

def construct():
    start_time = time.time()
    deadline = 110.0  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor
    T_target = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                T_target[a, b, d] = 1.0
    
    def verify(U, V, W, rank):
        """Verify exact decomposition using numpy einsum."""
        if time.time() > deadline:
            return False
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        T_computed = np.einsum('ac,bc,dc->abd', U_np, V_np, W_np)
        if not np.allclose(T_computed, T_target, atol=1e-9):
            return False
        return True
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, rank
    
    best_U, best_V, best_W, best_rank = schoolbook()
    
    if time.time() > deadline - 5:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Try to find lower rank using random search with integer/half-integer entries
    def random_entry():
        """Generate a random integer or half-integer entry."""
        return random.choice([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
    
    def try_random_rank(rank, num_trials=20):
        """Try random initializations."""
        if time.time() > deadline - 10:
            return None
        
        for trial in range(num_trials):
            if time.time() > deadline - 10:
                break
            
            U = [[random_entry() for _ in range(rank)] for _ in range(nm)]
            V = [[random_entry() for _ in range(rank)] for _ in range(mp)]
            W = [[random_entry() for _ in range(rank)] for _ in range(pn)]
            
            if verify(U, V, W, rank):
                return U, V, W, rank
        
        return None
    
    # Try ranks from 85 down to 70
    for rank in range(85, 69, -1):
        if time.time() > deadline - 15:
            break
        result = try_random_rank(rank, num_trials=10)
        if result is not None:
            best_U, best_V, best_W, best_rank = result
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
