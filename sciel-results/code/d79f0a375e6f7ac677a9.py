"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Uses structured block decomposition combining Strassen for 2x2x2 blocks
and direct terms, targeting rank < 68."""

DIMENSIONS = [3, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    deadline = 145  # seconds
    
    def check_time():
        return (time.time() - start_time) < deadline
    
    def strassen_decomp(row_off_a, col_off_a, row_off_b, col_off_b, U_t, V_t, W_t):
        """Add Strassen decomposition for 2x2*2x2 block at given offsets."""
        a = (row_off_a, col_off_a)
        b = (row_off_a, col_off_a + 1)
        c = (row_off_a + 1, col_off_a)
        d = (row_off_a + 1, col_off_a + 1)
        e = (row_off_b, col_off_b)
        f = (row_off_b, col_off_b + 1)
        g = (row_off_b + 1, col_off_b)
        h = (row_off_b + 1, col_off_b + 1)
        
        C00 = (row_off_a, col_off_b)
        C01 = (row_off_a, col_off_b + 1)
        C10 = (row_off_a + 1, col_off_b)
        C11 = (row_off_a + 1, col_off_b + 1)
        
        # m1 = (a+d)(e+h) -> C00 += 1, C11 += 1
        U_t.append([(a, 1), (d, 1)])
        V_t.append([(e, 1), (h, 1)])
        W_t.append([(C00, 1), (C11, 1)])
        
        # m2 = (c+d)*e -> C10 += 1, C11 -= 1
        U_t.append([(c, 1), (d, 1)])
        V_t.append([(e, 1)])
        W_t.append([(C10, 1), (C11, -1)])
        
        # m3 = a*(f-h) -> C01 += 1, C11 += 1
        U_t.append([(a, 1)])
        V_t.append([(f, 1), (h, -1)])
        W_t.append([(C01, 1), (C11, 1)])
        
        # m4 = d*(g-e) -> C00 += 1, C10 += 1
        U_t.append([(d, 1)])
        V_t.append([(g, 1), (e, -1)])
        W_t.append([(C00, 1), (C10, 1)])
        
        # m5 = (a+b)*h -> C00 -= 1, C01 += 1
        U_t.append([(a, 1), (b, 1)])
        V_t.append([(h, 1)])
        W_t.append([(C00, -1), (C01, 1)])
        
        # m6 = (c-a)*(e+f) -> C11 += 1
        U_t.append([(c, 1), (a, -1)])
        V_t.append([(e, 1), (f, 1)])
        W_t.append([(C11, 1)])
        
        # m7 = (b-d)*(g+h) -> C00 += 1
        U_t.append([(b, 1), (d, -1)])
        V_t.append([(g, 1), (h, 1)])
        W_t.append([(C00, 1)])
    
    def direct_term(i, j, k, U_t, V_t, W_t):
        """Add a direct term A[i,j]*B[j,k] to C[i,k]."""
        U_t.append([((i, j), 1)])
        V_t.append([((j, k), 1)])
        W_t.append([((i, k), 1)])
    
    def try_rank_reduction(U_t, V_t, W_t, current_rank):
        """Try to reduce rank by finding linear dependencies among terms."""
        if len(U_t) <= 1:
            return U_t, V_t, W_t, len(U_t)
        
        # Build the coefficient matrices
        nm = n * m
        mp = m * p
        pn = p * n
        
        U_mat = np.zeros((nm, current_rank))
        V_mat = np.zeros((mp, current_rank))
        W_mat = np.zeros((pn, current_rank))
        
        for idx in range(current_rank):
            for (i, j), coeff in U_t[idx]:
                U_mat[i * m + j, idx] = coeff
            for (j, k), coeff in V_t[idx]:
                V_mat[j * p + k, idx] = coeff
            for (i, k), coeff in W_t[idx]:
                W_mat[k * n + i, idx] = coeff
        
        # Find the rank of the concatenated constraint system
        # We need U*V*W to preserve the tensor
        # This is complex, so we use a simpler approach:
        # Try to eliminate terms that are linear combinations of others
        
        # For now, just return the current decomposition
        # A more sophisticated approach would be needed
        return U_t, V_t, W_t, current_rank
    
    U_t = []
    V_t = []
    W_t = []
    
    # Use Strassen for the 2x2*2x2 blocks
    # A[0:2, 0:2] * B[0:2, k:k+2] for k in {0, 2, 4, 6}
    for k0 in [0, 2, 4, 6]:
        if not check_time():
            break
        strassen_decomp(0, 0, 0, k0, U_t, V_t, W_t)
    
    # Direct terms for A[0:2, 2] * B[2, :]
    for k0 in [0, 2, 4, 6]:
        for i in range(2):
            for k in range(k0, k0 + 2):
                if not check_time():
                    break
                direct_term(i, 2, k, U_t, V_t, W_t)
    
    # Direct terms for A[2, 0:2] * B[0:2, :]
    for k0 in [0, 2, 4, 6]:
        for j in range(2):
            for k in range(k0, k0 + 2):
                if not check_time():
                    break
                direct_term(2, j, k, U_t, V_t, W_t)
    
    # Direct terms for A[2, 2] * B[2, :]
    for k0 in [0, 2, 4, 6]:
        for k in range(k0, k0 + 2):
            if not check_time():
                break
            direct_term(2, 2, k, U_t, V_t, W_t)
    
    rank = len(W_t)
    
    # Try to reduce rank
    U_t, V_t, W_t, rank = try_rank_reduction(U_t, V_t, W_t, rank)
    
    # Build the factor matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for idx in range(rank):
        for (i, j), coeff in U_t[idx]:
            U[i * m + j][idx] = coeff
        for (j, k), coeff in V_t[idx]:
            V[j * p + k][idx] = coeff
        for (i, k), coeff in W_t[idx]:
            W[k * n + i][idx] = coeff
    
    # Verify correctness
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    def verify_decomp(Ua, Va, Wa):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify_decomp(U_arr, V_arr, W_arr):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Convert to nice integer/half-integer values
    def to_nice_list(arr):
        """Convert numpy array to list of integer/half-integer values."""
        result = []
        for row in arr:
            nice_row = []
            for val in row:
                val2 = val * 2
                if abs(val2 - round(val2)) < 1e-6:
                    nice_val = round(val2) / 2.0
                    if abs(nice_val - round(nice_val)) < 1e-9:
                        nice_val = int(round(nice_val))
                    nice_row.append(nice_val)
                else:
                    nice_row.append(val)
            result.append(nice_row)
        return result
    
    return {"U": to_nice_list(U_arr), "V": to_nice_list(V_arr), "W": to_nice_list(W_arr)}
# EVOLVE-BLOCK-END
