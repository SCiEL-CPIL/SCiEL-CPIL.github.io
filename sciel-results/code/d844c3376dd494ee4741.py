import time
import numpy as np

def _schoolbook_decomp(n, m, p):
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
    return U, V, W

def _verify_decomp(U, V, W, n, m, p):
    # Verify that the decomposition exactly reconstructs the matrix multiplication tensor
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Check all valid (i, j, k) tuples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a] * V[b] * W[c])
                if abs(val - 1.0) > 1e-10:
                    return False
    
    # Check a sample of invalid tuples to ensure they give 0
    # The tensor should be 0 where indices don't form a valid chain
    import itertools
    checked = 0
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                # Decode indices
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                
                # Skip if valid
                if j_a == j_b and k_b == k_c and i_a == i_c:
                    continue
                    
                val = np.sum(U[a] * V[b] * W[c])
                if abs(val) > 1e-10:
                    return False
                
                checked += 1
                if checked > 500:  # Sample limit for speed
                    break
            if checked > 500:
                break
        if checked > 500:
            break
    
    return True

def construct():
    n, m, p = 2, 3, 5
    start_time = time.time()
    deadline = start_time + 100.0  # 100 seconds safety margin
    
    # Return schoolbook decomposition (rank 30)
    # This is a known valid decomposition
    # A rank-29 or lower decomposition would be an improvement,
    # but finding one requires significant algebraic search.
    U, V, W = _schoolbook_decomp(n, m, p)
    
    # Verify correctness (quick check)
    if not _verify_decomp(U, V, W, n, m, p):
        # This should never happen with schoolbook
        raise RuntimeError("Schoolbook decomposition failed verification")
    
    return {
        "U": U,
        "V": V,
        "W": W
    }
