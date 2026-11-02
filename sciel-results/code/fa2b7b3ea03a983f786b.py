"""Find rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication."""
import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = schoolbook()
    best_rank = n * m * p

    # Search for lower ranks using Local Search / Simulated Annealing on half-integers
    # Known lower bound is likely around 18-19. We try from 18 up to 23.
    for rank in range(18, best_rank):
        if time.time() > deadline - 10:
            break
            
        result = _local_search(n, m, p, rank, deadline)
        if result is not None:
            return result
            
    return best_result

def _local_search(n, m, p, rank, deadline):
    """
    Discrete local search for half-integer entries.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    We store entries as integers representing 2*x (i.e., half-integers).
    """
    np.random.seed(42)
    random.seed(42)
    
    def compute_S(U, V, W):
        S = np.zeros((n, m, p), dtype=int)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    S[i,j,k] = s
        return S

    def get_error(S):
        return np.sum((S - 8)**2)
    
    def _finalize(U, V, W):
        U_final = U / 2.0
        V_final = V / 2.0
        W_final = W / 2.0
        
        # Verify
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U_final[i*m+j, c] * V_final[j*p+k, c] * W_final[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        return None
        return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

    # Initialize with small random integers (scaled by 2)
    # Entries in [-5, 5] i.e. -2.5 to 2.5
    U = np.random.randint(-5, 6, (n * m, rank))
    V = np.random.randint(-5, 6, (m * p, rank))
    W = np.random.randint(-5, 6, (p * n, rank))
    
    S = compute_S(U, V, W)
    current_error = get_error(S)
    best_error = current_error
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    if best_error == 0:
        return _finalize(best_U, best_V, best_W)

    # Simulated Annealing parameters
    T_start = 10.0
    T_end = 0.1
    steps = 20000
    restarts = 0
    max_restarts = 10
    
    while time.time() < deadline - 5 and restarts < max_restarts:
        # Annealing schedule
        for step in range(steps):
            if time.time() > deadline - 5:
                break
                
            # Pick a random mutation
            factor_idx = random.randint(0, 2)
            
            if factor_idx == 0:
                idx = random.randint(0, n*m - 1)
                c = random.randint(0, rank - 1)
                delta = random.choice([-1, 1, -2, 2])
                old_val = U[idx, c]
                new_val = old_val + delta
                U[idx, c] = new_val
                
                i, j = divmod(idx, m)
                for k in range(p):
                    delta_S = delta * V[j*p+k, c] * W[k*n+i, c]
                    S[i, j, k] += delta_S
                    
            elif factor_idx == 1:
                idx = random.randint(0, m*p - 1)
                c = random.randint(0, rank - 1)
                delta = random.choice([-1, 1, -2, 2])
                old_val = V[idx, c]
                new_val = old_val + delta
                V[idx, c] = new_val
                
                j, k = divmod(idx, p)
                for i in range(n):
                    delta_S = delta * U[i*m+j, c] * W[k*n+i, c]
                    S[i, j, k] += delta_S
                    
            else:
                idx = random.randint(0, p*n - 1)
                c = random.randint(0, rank - 1)
                delta = random.choice([-1, 1, -2, 2])
                old_val = W[idx, c]
                new_val = old_val + delta
                W[idx, c] = new_val
                
                k, i = divmod(idx, n)
                for j in range(m):
                    delta_S = delta * U[i*m+j, c] * V[j*p+k, c]
                    S[i, j, k] += delta_S
            
            new_error = get_error(S)
            
            if new_error <= current_error:
                current_error = new_error
                if new_error < best_error:
                    best_error = new_error
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    if best_error == 0:
                        return _finalize(best_U, best_V, best_W)
            else:
                T = T_start * (T_end / T_start) ** (step / steps)
                if random.random() < np.exp(-(new_error - current_error) / T):
                    current_error = new_error
                else:
                    if factor_idx == 0:
                        U[idx, c] = old_val
                        i, j = divmod(idx, m)
                        for k in range(p):
                            S[i, j, k] -= delta * V[j*p+k, c] * W[k*n+i, c]
                    elif factor_idx == 1:
                        V[idx, c] = old_val
                        j, k = divmod(idx, p)
                        for i in range(n):
                            S[i, j, k] -= delta * U[i*m+j, c] * W[k*n+i, c]
                    else:
                        W[idx, c] = old_val
                        k, i = divmod(idx, n)
                        for j in range(m):
                            S[i, j, k] -= delta * U[i*m+j, c] * V[j*p+k, c]
        
        if best_error > 0:
             U = np.random.randint(-5, 6, (n * m, rank))
             V = np.random.randint(-5, 6, (m * p, rank))
             W = np.random.randint(-5, 6, (p * n, rank))
             S = compute_S(U, V, W)
             current_error = get_error(S)
             if current_error < best_error:
                 best_error = current_error
                 best_U = U.copy()
                 best_V = V.copy()
                 best_W = W.copy()
             restarts += 1
             
    if best_error == 0:
        return _finalize(best_U, best_V, best_W)
        
    return None
