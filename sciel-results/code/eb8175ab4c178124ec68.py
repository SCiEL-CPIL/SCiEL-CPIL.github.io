import numpy as np
import time
from itertools import product

def construct():
    n, m, p = 2, 3, 4
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s limit
    
    # Try ranks from low to high
    for target_rank in range(12, 25):
        if time.time() > deadline:
            break
        result = _try_rank(target_rank, n, m, p, deadline)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _try_rank(target_rank, n, m, p, deadline):
    """
    Try to find a rank-target_rank decomposition using structured search.
    """
    rows_U = n * m  # 6
    rows_V = m * p  # 12
    rows_W = p * n  # 8
    
    # Grid of allowed values (integers and half-integers)
    grid = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    
    def compute_error(U, V, W):
        """Compute sum of squared errors over all (i,j,k) triples."""
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(target_rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    diff = s - 1.0
                    err += diff * diff
        return err
    
    def try_local_search(U, V, W, max_iters=2000):
        """Local search to improve the decomposition."""
        current_err = compute_error(U, V, W)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random variable
            var_idx = np.random.randint(0, rows_U * target_rank + rows_V * target_rank + rows_W * target_rank)
            
            if var_idx < rows_U * target_rank:
                idx = var_idx
                mat = U
                r, c = divmod(idx, target_rank)
            elif var_idx < rows_U * target_rank + rows_V * target_rank:
                idx = var_idx - rows_U * target_rank
                mat = V
                r, c = divmod(idx, target_rank)
            else:
                idx = var_idx - rows_U * target_rank - rows_V * target_rank
                mat = W
                r, c = divmod(idx, target_rank)
                
            old_val = mat[r, c]
            
            # Try changing to a random other grid value
            new_val = np.random.choice(grid)
            if new_val == old_val:
                continue
                
            mat[r, c] = new_val
            new_err = compute_error(U, V, W)
            
            if new_err < current_err:
                current_err = new_err
                if current_err < 1e-10:
                    return U, V, W, current_err
            else:
                mat[r, c] = old_val
                
        return U, V, W, current_err
    
    best_result = None
    best_err = float('inf')
    
    # Multiple restarts with different initialization strategies
    n_restarts = 50
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            # Structured initialization: try a pattern based on modular arithmetic
            U = np.zeros((rows_U, target_rank))
            V = np.zeros((rows_V, target_rank))
            W = np.zeros((rows_W, target_rank))
            
            # Assign values based on c mod n, c mod m, c mod p
            for c in range(target_rank):
                ci = c % n
                cj = c % m
                ck = c % p
                
                # Set values to create constructive interference
                for i in range(n):
                    for j in range(m):
                        U[i*m+j, c] = 1.0 if (i == ci and j == cj) else 0.0
                        
                for j in range(m):
                    for k in range(p):
                        V[j*p+k, c] = 1.0 if (j == cj and k == ck) else 0.0
                        
                for k in range(p):
                    for i in range(n):
                        W[k*n+i, c] = 1.0 if (k == ck and i == ci) else 0.0
                        
        elif restart == 1:
            # Another structured pattern
            U = np.zeros((rows_U, target_rank))
            V = np.zeros((rows_V, target_rank))
            W = np.zeros((rows_W, target_rank))
            
            for c in range(target_rank):
                # Use different assignment
                U[0, c] = 1.0 if c < target_rank // 2 else -1.0
                U[1, c] = 1.0
                U[2, c] = 1.0 if c % 2 == 0 else -1.0
                U[3, c] = 1.0
                U[4, c] = 1.0 if c % 3 == 0 else 0.0
                U[5, c] = 1.0
                
                V[0, c] = 1.0
                V[1, c] = 1.0 if c % 2 == 0 else -1.0
                V[2, c] = 1.0
                V[3, c] = 1.0 if c % 3 == 0 else 0.0
                V[4, c] = 1.0
                V[5, c] = 1.0 if c % 2 == 0 else -1.0
                V[6, c] = 1.0
                V[7, c] = 1.0 if c % 3 == 0 else 0.0
                V[8, c] = 1.0
                V[9, c] = 1.0 if c % 2 == 0 else -1.0
                V[10, c] = 1.0
                V[11, c] = 1.0 if c % 3 == 0 else 0.0
                
                W[0, c] = 1.0 if c < target_rank // 2 else -1.0
                W[1, c] = 1.0
                W[2, c] = 1.0 if c % 2 == 0 else -1.0
                W[3, c] = 1.0
                W[4, c] = 1.0 if c % 3 == 0 else 0.0
                W[5, c] = 1.0
                W[6, c] = 1.0 if c < target_rank // 2 else -1.0
                W[7, c] = 1.0
                
        else:
            # Random initialization from grid
            U = np.random.choice(grid, size=(rows_U, target_rank))
            V = np.random.choice(grid, size=(rows_V, target_rank))
            W = np.random.choice(grid, size=(rows_W, target_rank))
        
        U, V, W, err = try_local_search(U, V, W, max_iters=1000)
        
        if err < best_err:
            best_err = err
            best_result = (U.copy(), V.copy(), W.copy())
            
        if best_err < 1e-10:
            break
    
    if best_result is not None and best_err < 1e-10:
        U, V, W = best_result
        # Round to nearest grid value
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        # Final verification
        if compute_error(U, V, W) < 1e-10:
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    return None
