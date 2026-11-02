import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin
    
    # Try to find a rank 12 solution
    rank = 12
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    best_error = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    for attempt in range(50):
        # Random init
        U[:] = np.random.choice([-1, 0, 1, 0.5, -0.5], size=(n * m, rank))
        V[:] = np.random.choice([-1, 0, 1, 0.5, -0.5], size=(m * p, rank))
        W[:] = np.random.choice([-1, 0, 1, 0.5, -0.5], size=(p * n, rank))
        
        error = _calc_error(U, V, W, n, m, p, rank)
        if error < 1e-6:
            return _to_dict(U, V, W)
        
        if error < best_error:
            best_error = error
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            
        # Local search
        for step in range(1000):
            if time.time() > deadline:
                break
            # Pick random variable
            var_type = np.random.randint(3)
            idx = np.random.randint(0, rank)
            if var_type == 0:
                row = np.random.randint(0, n * m)
                old_val = U[row, idx]
                new_val = old_val + np.random.choice([-1, 1, 0.5, -0.5])
                U[row, idx] = new_val
                new_error = _calc_error(U, V, W, n, m, p, rank)
                if new_error < error:
                    error = new_error
                    if error < 1e-6:
                        return _to_dict(U, V, W)
                else:
                    U[row, idx] = old_val
            elif var_type == 1:
                row = np.random.randint(0, m * p)
                old_val = V[row, idx]
                new_val = old_val + np.random.choice([-1, 1, 0.5, -0.5])
                V[row, idx] = new_val
                new_error = _calc_error(U, V, W, n, m, p, rank)
                if new_error < error:
                    error = new_error
                    if error < 1e-6:
                        return _to_dict(U, V, W)
                else:
                    V[row, idx] = old_val
            else:
                row = np.random.randint(0, p * n)
                old_val = W[row, idx]
                new_val = old_val + np.random.choice([-1, 1, 0.5, -0.5])
                W[row, idx] = new_val
                new_error = _calc_error(U, V, W, n, m, p, rank)
                if new_error < error:
                    error = new_error
                    if error < 1e-6:
                        return _to_dict(U, V, W)
                else:
                    W[row, idx] = old_val
                    
        if time.time() > deadline:
            break

    # Fallback to schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    return {"U": U_school, "V": V_school, "W": W_school}

def _calc_error(U, V, W, n, m, p, rank):
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                error += (s - 1.0) ** 2
    return error

def _to_dict(U, V, W):
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
