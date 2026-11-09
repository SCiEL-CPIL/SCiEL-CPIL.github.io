"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

memo = {}

def _combine_decomps(n, m, p, split_dim, split_idx, d1, d2):
    """
    Combine two decompositions d1, d2 for sub-problems into one for the full problem.
    split_dim: 0 for n, 1 for m, 2 for p.
    split_idx: the size of the first part.
    """
    combined = []
    
    if split_dim == 0: # Split n
        n1 = split_idx
        n2 = n - n1
        for U1, V1, W1 in d1:
            U = [0.0] * (n * m)
            V = [0.0] * (m * p)
            W = [0.0] * (p * n)
            # U1: n1*m
            for idx in range(n1 * m):
                U[idx] = U1[idx]
            # V1: m*p
            for idx in range(m * p):
                V[idx] = V1[idx]
            # W1: p*n1 -> p*n
            for idx in range(p * n1):
                k = idx // n1
                i1 = idx % n1
                W[k * n + i1] = W1[idx]
            combined.append((U, V, W))
        
        for U2, V2, W2 in d2:
            U = [0.0] * (n * m)
            V = [0.0] * (m * p)
            W = [0.0] * (p * n)
            # U2: n2*m -> n*m
            for idx in range(n2 * m):
                i2 = idx // m
                j = idx % m
                U[(i2 + n1) * m + j] = U2[idx]
            # V2: m*p
            for idx in range(m * p):
                V[idx] = V2[idx]
            # W2: p*n2 -> p*n
            for idx in range(p * n2):
                k = idx // n2
                i2 = idx % n2
                W[k * n + (i2 + n1)] = W2[idx]
            combined.append((U, V, W))
            
    elif split_dim == 1: # Split m
        m1 = split_idx
        m2 = m - m1
        for U1, V1, W1 in d1:
            U = [0.0] * (n * m)
            V = [0.0] * (m * p)
            W = [0.0] * (p * n)
            # U1: n*m1 -> n*m
            for idx in range(n * m1):
                i = idx // m1
                j1 = idx % m1
                U[i * m + j1] = U1[idx]
            # V1: m1*p -> m*p
            for idx in range(m1 * p):
                j1 = idx // p
                k = idx % p
                V[j1 * p + k] = V1[idx]
            # W1: p*n
            for idx in range(p * n):
                W[idx] = W1[idx]
            combined.append((U, V, W))
            
        for U2, V2, W2 in d2:
            U = [0.0] * (n * m)
            V = [0.0] * (m * p)
            W = [0.0] * (p * n)
            # U2: n*m2 -> n*m
            for idx in range(n * m2):
                i = idx // m2
                j2 = idx % m2
                U[i * m + (j2 + m1)] = U2[idx]
            # V2: m2*p -> m*p
            for idx in range(m2 * p):
                j2 = idx // p
                k = idx % p
                V[(j2 + m1) * p + k] = V2[idx]
            # W2: p*n
            for idx in range(p * n):
                W[idx] = W2[idx]
            combined.append((U, V, W))
            
    else: # Split p
        p1 = split_idx
        p2 = p - p1
        for U1, V1, W1 in d1:
            U = [0.0] * (n * m)
            V = [0.0] * (m * p)
            W = [0.0] * (p * n)
            # U1: n*m
            for idx in range(n * m):
                U[idx] = U1[idx]
            # V1: m*p1 -> m*p
            for idx in range(m * p1):
                j = idx // p1
                k1 = idx % p1
                V[j * p + k1] = V1[idx]
            # W1: p1*n -> p*n
            for idx in range(p1 * n):
                k1 = idx // n
                i = idx % n
                W[k1 * n + i] = W1[idx]
            combined.append((U, V, W))
            
        for U2, V2, W2 in d2:
            U = [0.0] * (n * m)
            V = [0.0] * (m * p)
            W = [0.0] * (p * n)
            # U2: n*m
            for idx in range(n * m):
                U[idx] = U2[idx]
            # V2: m*p2 -> m*p
            for idx in range(m * p2):
                j = idx // p2
                k2 = idx % p2
                V[j * p + (k2 + p1)] = V2[idx]
            # W2: p2*n -> p*n
            for idx in range(p2 * n):
                k2 = idx // n
                i = idx % n
                W[(k2 + p1) * n + i] = W2[idx]
            combined.append((U, V, W))
            
    return combined

def get_decomposition(n, m, p):
    if (n, m, p) in memo:
        return memo[(n, m, p)]
        
    # Base case
    if n == 1 and m == 1 and p == 1:
        U = [1.0]
        V = [1.0]
        W = [1.0]
        result = [(U, V, W)]
        memo[(n, m, p)] = result
        return result
        
    best_decomp = None
    best_rank = float('inf')
    
    # Try splitting n
    for n1 in range(1, n):
        n2 = n - n1
        d1 = get_decomposition(n1, m, p)
        d2 = get_decomposition(n2, m, p)
        rank = len(d1) + len(d2)
        if rank < best_rank:
            best_decomp = _combine_decomps(n, m, p, 0, n1, d1, d2)
            best_rank = rank
            
    # Try splitting m
    for m1 in range(1, m):
        m2 = m - m1
        d1 = get_decomposition(n, m1, p)
        d2 = get_decomposition(n, m2, p)
        rank = len(d1) + len(d2)
        if rank < best_rank:
            best_decomp = _combine_decomps(n, m, p, 1, m1, d1, d2)
            best_rank = rank
            
    # Try splitting p
    for p1 in range(1, p):
        p2 = p - p1
        d1 = get_decomposition(n, m, p1)
        d2 = get_decomposition(n, m, p2)
        rank = len(d1) + len(d2)
        if rank < best_rank:
            best_decomp = _combine_decomps(n, m, p, 2, p1, d1, d2)
            best_rank = rank
            
    # If no split found (should not happen for n,m,p > 1), use schoolbook
    if best_decomp is None:
        result = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U = [0.0] * (n * m)
                    V = [0.0] * (m * p)
                    W = [0.0] * (p * n)
                    U[i * m + j] = 1.0
                    V[j * p + k] = 1.0
                    W[k * n + i] = 1.0
                    result.append((U, V, W))
        best_decomp = result
        best_rank = len(result)

    memo[(n, m, p)] = best_decomp
    return best_decomp

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Get the optimal decomposition
    decomp = get_decomposition(n, m, p)
    rank = len(decomp)
    
    # Construct final matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (U_c, V_c, W_c) in enumerate(decomp):
        for idx in range(n * m):
            if U_c[idx] != 0:
                U[idx][c] = U_c[idx]
        for idx in range(m * p):
            if V_c[idx] != 0:
                V[idx][c] = V_c[idx]
        for idx in range(p * n):
            if W_c[idx] != 0:
                W[idx][c] = W_c[idx]
                
    # Verify
    # import numpy as np
    # U_arr = np.array(U, dtype=float)
    # V_arr = np.array(V, dtype=float)
    # W_arr = np.array(W, dtype=float)
    # for i in range(n):
    #     for j in range(m):
    #         for k in range(p):
    #             expected = 1.0
    #             computed = 0.0
    #             for c in range(rank):
    #                 computed += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
    #             if abs(computed - expected) > 1e-9:
    #                 print("Verification failed")
    #                 break
    #         else:
    #             continue
    #         break
    #     else:
    #         continue
    #     break
        
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
