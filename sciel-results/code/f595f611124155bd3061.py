import numpy as np
import random
import time
import math

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    def calculate_error(U, V, W, R):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(R):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    err += (val - 1.0) ** 2
        return err

    def random_init(R):
        vals = [-1.0, 0.0, 1.0]
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        for i in range(n * m):
            for r in range(R):
                U[i, r] = random.choice(vals)
        for j in range(m * p):
            for r in range(R):
                V[j, r] = random.choice(vals)
        for k in range(p * n):
            for r in range(R):
                W[k, r] = random.choice(vals)
        return U, V, W

    def simulated_annealing(U, V, W, R, start_temp=1.0, end_temp=0.01, max_iters=5000, timeout=10):
        start_time = time.time()
        curr_err = calculate_error(U, V, W, R)
        best_U = U.copy()
        best_V = V.copy()
        best_W = W.copy()
        best_err = curr_err
        
        total_vars = (n * m + m * p + p * n) * R
        rows_u, rows_v, rows_w = n * m, m * p, p * n
        
        for i in range(max_iters):
            if time.time() - start_time > timeout:
                break
            
            temp = start_temp * (end_temp / start_temp) ** (i / max_iters)
            
            # Pick random variable
            idx = random.randint(0, total_vars - 1)
            
            if idx < rows_u * R:
                row = idx // R
                col = idx % R
                mat = U
                rows = rows_u
            elif idx < (rows_u + rows_v) * R:
                idx_local = idx - rows_u * R
                row = idx_local // R
                col = idx_local % R
                mat = V
                rows = rows_v
            else:
                idx_local = idx - (rows_u + rows_v) * R
                row = idx_local // R
                col = idx_local % R
                mat = W
                rows = rows_w
                
            old_val = mat[row, col]
            
            # Generate a random neighbor
            # Steps of 0.5
            step = random.choice([-0.5, 0.5, -1.0, 1.0])
            new_val = old_val + step
            
            # Optional: clamp to reasonable range?
            # Let's allow it, but maybe reject if abs(new_val) > 5
            
            mat[row, col] = new_val
            new_err = calculate_error(U, V, W, R)
            
            delta = new_err - curr_err
            
            if delta < 0 or random.random() < math.exp(-delta / temp):
                curr_err = new_err
                if curr_err < best_err:
                    best_err = curr_err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
            else:
                mat[row, col] = old_val
            
            if best_err < 1e-10:
                break

        return best_U, best_V, best_W, best_err

    # Try ranks from 15 to 20
    for R in range(15, 21):
        for attempt in range(5):
            random.seed(attempt * 100 + R)
            np.random.seed(attempt * 100 + R)
            
            U, V, W = random_init(R)
            U, V, W, err = simulated_annealing(U, V, W, R, max_iters=2000, timeout=5)
            
            if err < 1e-10:
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
        
    # Fallback: Schoolbook
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
# EVOLVE-BLOCK-END
