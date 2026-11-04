import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    # Precompute target tensor T
    # Dimensions: (n*m, m*p, p*n)
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    T = np.zeros((dim_U, dim_V, dim_W), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k  # Note: V index in schoolbook was j*p+k. 
                                # In check: V[l*p+k]. So if j==l, index is j*p+k.
                c = k * n + i
                T[a, b, c] = 1

    def compute_error(U, V, W, R):
        # U: (dim_U, R), V: (dim_V, R), W: (dim_W, R)
        # S = sum_r U_r outer V_r outer W_r
        # Use einsum: S[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        S = np.einsum('ar,br,cr->abc', U, V, W)
        E = S - T
        # Return sum of squares for annealing
        return np.sum(E * E)

    def check_decomposition(U, V, W, R):
        S = np.einsum('ar,br,cr->abc', U, V, W)
        return np.array_equal(S, T)

    def try_find_decomposition(R, time_limit=30):
        start_time = time.time()
        best_error = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        # Simulated Annealing
        T_start = 1.0
        T_end = 0.001
        alpha = 0.999
        current_T = T_start
        
        # Initialize randomly with small integers
        U = np.array([[random.choice([-1, 0, 1]) for _ in range(R)] for _ in range(dim_U)], dtype=float)
        V = np.array([[random.choice([-1, 0, 1]) for _ in range(R)] for _ in range(dim_V)], dtype=float)
        W = np.array([[random.choice([-1, 0, 1]) for _ in range(R)] for _ in range(dim_W)], dtype=float)
        
        current_error = compute_error(U, V, W, R)
        if current_error < best_error:
            best_error = current_error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if best_error == 0:
                return {"U": best_U.astype(int).tolist(), "V": best_V.astype(int).tolist(), "W": best_W.astype(int).tolist()}

        while time.time() - start_time < time_limit and current_T > T_end:
            # Pick a random variable to change
            matrix = random.choice(['U', 'V', 'W'])
            if matrix == 'U':
                row = random.randint(0, dim_U - 1)
                col = random.randint(0, R - 1)
                old_val = U[row, col]
                new_val = random.choice([-1, 0, 1])
                if new_val == old_val: continue
                U[row, col] = new_val
                new_error = compute_error(U, V, W, R)
                delta = new_error - current_error
                if delta < 0 or random.random() < np.exp(-delta / current_T):
                    current_error = new_error
                    if current_error < best_error:
                        best_error = current_error
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        if best_error == 0:
                            return {"U": best_U.astype(int).tolist(), "V": best_V.astype(int).tolist(), "W": best_W.astype(int).tolist()}
                else:
                    U[row, col] = old_val
            elif matrix == 'V':
                row = random.randint(0, dim_V - 1)
                col = random.randint(0, R - 1)
                old_val = V[row, col]
                new_val = random.choice([-1, 0, 1])
                if new_val == old_val: continue
                V[row, col] = new_val
                new_error = compute_error(U, V, W, R)
                delta = new_error - current_error
                if delta < 0 or random.random() < np.exp(-delta / current_T):
                    current_error = new_error
                    if current_error < best_error:
                        best_error = current_error
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        if best_error == 0:
                            return {"U": best_U.astype(int).tolist(), "V": best_V.astype(int).tolist(), "W": best_W.astype(int).tolist()}
                else:
                    V[row, col] = old_val
            else:
                row = random.randint(0, dim_W - 1)
                col = random.randint(0, R - 1)
                old_val = W[row, col]
                new_val = random.choice([-1, 0, 1])
                if new_val == old_val: continue
                W[row, col] = new_val
                new_error = compute_error(U, V, W, R)
                delta = new_error - current_error
                if delta < 0 or random.random() < np.exp(-delta / current_T):
                    current_error = new_error
                    if current_error < best_error:
                        best_error = current_error
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        if best_error == 0:
                            return {"U": best_U.astype(int).tolist(), "V": best_V.astype(int).tolist(), "W": best_W.astype(int).tolist()}
                else:
                    W[row, col] = old_val
            
            current_T *= alpha

        return None

    # Try ranks from 20 to 23
    # We allocate time budget carefully
    time_budget = 100
    ranks_to_try = [20, 21, 22, 23]
    time_per_rank = time_budget / len(ranks_to_try)
    
    for R in ranks_to_try:
        result = try_find_decomposition(R, time_limit=time_per_rank)
        if result:
            return result

    # Fallback to schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
