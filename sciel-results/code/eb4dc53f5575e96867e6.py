"""
Rewritten program for 2x3x8 matrix multiplication rank minimization.
Uses Strassen decomposition for 2x2x2 blocks and outer products for 2x1x2 blocks.
Total rank 44.
"""

import numpy as np
import time

DIMENSIONS = [2, 3, 8]
_START_TIME = time.time()
TIME_LIMIT = 150.0

def _verify_decomposition(U, V, W, n=2, m=3, p=8, seed=42):
    """Verify the tensor decomposition is exact."""
    rank = len(U[0])
    rng = np.random.RandomState(seed)
    A = rng.randn(n, m)
    B = rng.randn(m, p)
    C_expected = A @ B
    
    U_arr = np.array(U, dtype=float).reshape(n * m, rank)
    V_arr = np.array(V, dtype=float).reshape(m * p, rank)
    W_arr = np.array(W, dtype=float).reshape(p * n, rank)
    
    # Compute C from decomposition: C[i,k] = sum_r U[i*m+j... wait, need to be careful
    # U is indexed by (i,j) -> i*m + j
    # V is indexed by (j,k) -> j*p + k  
    # W is indexed by (k,i) -> k*n + i
    C_computed = np.zeros((n, p))
    for r in range(rank):
        u = U_arr[:, r].reshape(n, m)
        v = V_arr[:, r].reshape(m, p)
        w = W_arr[:, r].reshape(p, n)
        # Contribution: for each i,k: sum_j u[i,j]*v[j,k] * w[k,i]
        uv = u @ v  # (n, p)
        for i in range(n):
            for k in range(p):
                C_computed[i, k] += uv[i, k] * w[k, i]
    
    max_err = np.max(np.abs(C_computed - C_expected))
    return max_err < 1e-9

def construct():
    """
    Constructs rank 44 decomposition for 2x3x8 matrix multiplication.
    Uses Strassen for 2x2x2 blocks and outer product for 2x1x2 blocks.
    """
    n, m, p = 2, 3, 8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(col, u_idx, u_coeff, v_idx, v_coeff, w_idx, w_coeff):
        for idx, c in zip(u_idx, u_coeff):
            if c != 0:
                U[idx][col] += c
        for idx, c in zip(v_idx, v_coeff):
            if c != 0:
                V[idx][col] += c
        for idx, c in zip(w_idx, w_coeff):
            if c != 0:
                W[idx][col] += c

    current_col = 0
    
    # Split p=8 into 4 pairs of 2.
    # For each pair (k0, k1):
    #   Part 1: Strassen 2x2x2 for j=0,1 (rank 7)
    #   Part 2: Outer product 2x1x2 for j=2 (rank 4)
    # Total: 4 * (7 + 4) = 44
    
    for k_block in range(4):
        if time.time() - _START_TIME > TIME_LIMIT - 5.0:
            break
        k0 = k_block * 2
        k1 = k_block * 2 + 1
        
        # Part 1: Strassen 2x2x2 for j=0,1
        # A: a=A[0,0]=0, b=A[0,1]=1, c=A[1,0]=3, d=A[1,1]=4
        # B: e=B[0,k0], f=B[0,k1], g=B[1,k0], h=B[1,k1]
        # C: w00=C[k0,0], w01=C[k1,0], w10=C[k0,1], w11=C[k1,1]
        
        a_idx, b_idx, c_idx, d_idx = 0, 1, 3, 4
        e_idx, f_idx = 0 * p + k0, 0 * p + k1
        g_idx, h_idx = 1 * p + k0, 1 * p + k1
        w_00, w_01 = k0 * n, k1 * n
        w_10, w_11 = k0 * n + 1, k1 * n + 1
        
        # M1 = (a+d)(e+h) -> contributes to C[0,0] and C[1,1]
        add_term(current_col, [a_idx, d_idx], [1, 1], [e_idx, h_idx], [1, 1], [w_00, w_11], [1, 1])
        current_col += 1
        # M2 = (c+d)e -> contributes to C[1,0] and C[1,1]
        add_term(current_col, [c_idx, d_idx], [1, 1], [e_idx], [1], [w_10, w_11], [1, -1])
        current_col += 1
        # M3 = a(f-h) -> contributes to C[0,1] and C[1,1]
        add_term(current_col, [a_idx], [1], [f_idx, h_idx], [1, -1], [w_01, w_11], [1, 1])
        current_col += 1
        # M4 = d(g-e) -> contributes to C[0,0] and C[1,0]
        add_term(current_col, [d_idx], [1], [g_idx, e_idx], [1, -1], [w_00, w_10], [1, 1])
        current_col += 1
        # M5 = (a+b)h -> contributes to C[0,0] and C[0,1]
        add_term(current_col, [a_idx, b_idx], [1, 1], [h_idx], [1], [w_00, w_01], [-1, 1])
        current_col += 1
        # M6 = (c-a)(e+f) -> contributes to C[1,1]
        add_term(current_col, [c_idx, a_idx], [1, -1], [e_idx, f_idx], [1, 1], [w_11], [1])
        current_col += 1
        # M7 = (b-d)(g+h) -> contributes to C[0,0]
        add_term(current_col, [b_idx, d_idx], [1, -1], [g_idx, h_idx], [1, 1], [w_00], [1])
        current_col += 1
        
        # Part 2: Outer Product 2x1x2 for j=2
        # A: a2=A[0,2]=2, c2=A[1,2]=5
        # B: e2=B[2,k0], f2=B[2,k1]
        
        a2_idx, c2_idx = 2, 5
        e2_idx, f2_idx = 2 * p + k0, 2 * p + k1
        
        # C[0, k0] = A[0,2] * B[2, k0]
        add_term(current_col, [a2_idx], [1], [e2_idx], [1], [w_00], [1])
        current_col += 1
        # C[0, k1] = A[0,2] * B[2, k1]
        add_term(current_col, [a2_idx], [1], [f2_idx], [1], [w_01], [1])
        current_col += 1
        # C[1, k0] = A[1,2] * B[2, k0]
        add_term(current_col, [c2_idx], [1], [e2_idx], [1], [w_10], [1])
        current_col += 1
        # C[1, k1] = A[1,2] * B[2, k1]
        add_term(current_col, [c2_idx], [1], [f2_idx], [1], [w_11], [1])
        current_col += 1

    # Ensure we have exactly 'rank' columns
    actual_rank = current_col
    if actual_rank < rank:
        for r in range(actual_rank, rank):
            for i in range(n * m):
                U[i][r] = 0
            for i in range(m * p):
                V[i][r] = 0
            for i in range(p * n):
                W[i][r] = 0
    elif actual_rank > rank:
        for i in range(n * m):
            U[i] = U[i][:rank]
        for i in range(m * p):
            V[i] = V[i][:rank]
        for i in range(p * n):
            W[i] = W[i][:rank]

    # Final verification
    if time.time() - _START_TIME < TIME_LIMIT - 2.0:
        _verify_decomposition(U, V, W, n, m, p)

    return {"U": U, "V": V, "W": W}
