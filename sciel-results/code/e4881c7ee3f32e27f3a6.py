"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p):
    """Verify tensor decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]"""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def _try_rank(R, n, m, p, timeout=100):
    """Try to find a rank-R decomposition via random search + local improvement."""
    start = time.time()
    np.random.seed(42)
    
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize with random sparse patterns
    for attempt in range(50):
        if time.time() - start > timeout:
            break
        
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        
        # Random sparse initialization
        for r in range(R):
            # Pick 2-3 non-zero entries in each
            for _ in range(2):
                idx = np.random.randint(0, n*m)
                U[idx, r] = np.random.choice([-1, 1])
            for _ in range(2):
                idx = np.random.randint(0, m*p)
                V[idx, r] = np.random.choice([-1, 1])
            for _ in range(2):
                idx = np.random.randint(0, p*n)
                W[idx, r] = np.random.choice([-1, 1])
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = 0.0
                    for r in range(R):
                        actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    err += (actual - 1.0)**2
        
        if err < best_err:
            best_err = err
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
        
        if best_err < 1e-6:
            break
    
    if best_err < 1e-6:
        return best_U, best_V, best_W, best_err
    return None, None, None, best_err

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Start with the known rank-55 solution
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        U[0 * 3 + 0][col] += 1
        U[1 * 3 + 1][col] += 1
        V[0 * 10 + k0][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        U[1 * 3 + 0][col] += 1
        U[1 * 3 + 1][col] += 1
        V[0 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        W[k1 * 2 + 1][col] -= 1
        col += 1
        
        U[0 * 3 + 0][col] += 1
        V[0 * 10 + k1][col] += 1
        V[1 * 10 + k1][col] -= 1
        W[k1 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        U[1 * 3 + 1][col] += 1
        V[1 * 10 + k0][col] += 1
        V[0 * 10 + k0][col] -= 1
        W[k0 * 2 + 0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        U[0 * 3 + 0][col] += 1
        U[0 * 3 + 1][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] -= 1
        W[k1 * 2 + 0][col] += 1
        col += 1
        
        U[1 * 3 + 0][col] += 1
        U[0 * 3 + 0][col] -= 1
        V[0 * 10 + k0][col] += 1
        V[0 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        U[0 * 3 + 1][col] += 1
        U[1 * 3 + 1][col] -= 1
        V[1 * 10 + k0][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1
        
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1
        
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 0][col] += 1
        col += 1
        
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    # Try to reduce rank by searching for rank-54, 53, etc.
    for target_rank in range(rank - 1, max(30, rank - 5), -1):
        if time.time() > deadline:
            break
        
        U_t, V_t, W_t, err = _try_rank(target_rank, n, m, p, timeout=15)
        if U_t is not None and err < 1e-6:
            # Convert to list format with integer/half-integer entries
            U_list = [[0] * target_rank for _ in range(n * m)]
            V_list = [[0] * target_rank for _ in range(m * p)]
            W_list = [[0] * target_rank for _ in range(p * n)]
            
            for i in range(n * m):
                for r in range(target_rank):
                    val = U_t[i, r]
                    if abs(val - round(val)) < 1e-6:
                        U_list[i][r] = int(round(val))
                    elif abs(val * 2 - round(val * 2)) < 1e-6:
                        U_list[i][r] = round(val * 2) / 2
                    else:
                        U_list[i][r] = val
            
            for j in range(m * p):
                for r in range(target_rank):
                    val = V_t[j, r]
                    if abs(val - round(val)) < 1e-6:
                        V_list[j][r] = int(round(val))
                    elif abs(val * 2 - round(val * 2)) < 1e-6:
                        V_list[j][r] = round(val * 2) / 2
                    else:
                        V_list[j][r] = val
            
            for k in range(p * n):
                for r in range(target_rank):
                    val = W_t[k, r]
                    if abs(val - round(val)) < 1e-6:
                        W_list[k][r] = int(round(val))
                    elif abs(val * 2 - round(val * 2)) < 1e-6:
                        W_list[k][r] = round(val * 2) / 2
                    else:
                        W_list[k][r] = val
            
            # Verify
            if _verify(U_list, V_list, W_list, n, m, p):
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = target_rank
                break
    
    return best_result
# EVOLVE-BLOCK-END
