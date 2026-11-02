"""Program for 3x3x6 matrix multiplication tensor decomposition."""
import time
import sys
import numpy as np

DIMENSIONS = [3, 3, 6]

def _schoolbook():
    n, m, p = 3, 3, 6
    rank = n * m * p  # 54
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

def _verify_decomposition(U, V, W, n=3, m=3, p=6):
    """Verify that U, V, W form an exact decomposition of the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U_arr[i*m + j, r] * V_arr[j*p + k, r] * W_arr[k*n + i, r]
                if abs(actual - expected) > 1e-10:
                    return False
    return True

def _local_search(U_init, V_init, W_init, target_rank, n=3, m=3, p=6, max_iter=50, seed=42):
    """Local search around an initial point to find a half-integer solution."""
    rng = np.random.RandomState(seed)
    U = np.array(U_init, dtype=float)
    V = np.array(V_init, dtype=float)
    W = np.array(W_init, dtype=float)
    
    for it in range(max_iter):
        # Randomly perturb one entry by +/-0.5
        mat = rng.randint(3)
        delta = 0.5 * rng.choice([-1, 1])
        if mat == 0:
            idx = (rng.randint(n*m), rng.randint(target_rank))
            U[idx] += delta
        elif mat == 1:
            idx = (rng.randint(m*p), rng.randint(target_rank))
            V[idx] += delta
        else:
            idx = (rng.randint(p*n), rng.randint(target_rank))
            W[idx] += delta
        
        # Verify
        if _verify_decomposition(U.tolist(), V.tolist(), W.tolist()):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 115.0
    
    # Start with schoolbook
    schoolbook = _schoolbook()
    U, V, W = schoolbook['U'], schoolbook['V'], schoolbook['W']
    rank = len(U[0])
    
    # Try to find a solution at lower ranks by removing columns
    for target_rank in [53, 52]:
        if time.time() > deadline - 10:
            break
        
        for col_to_remove in range(rank):
            if time.time() > deadline - 5:
                break
            
            U_sub = [row[:col_to_remove] + row[col_to_remove+1:] for row in U]
            V_sub = [row[:col_to_remove] + row[col_to_remove+1:] for row in V]
            W_sub = [row[:col_to_remove] + row[col_to_remove+1:] for row in W]
            
            result = _local_search(U_sub, V_sub, W_sub, target_rank, max_iter=30)
            if result is not None:
                U, V, W = result
                return {"U": U, "V": V, "W": W}
    
    return schoolbook
# EVOLVE-BLOCK-END
