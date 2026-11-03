"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Uses Strassen's rank-7 decomposition for 2x2x2 blocks combined with
direct terms for remaining dimensions, targeting rank < 72.
Additionally attempts to remove linearly dependent columns to reduce rank.
"""

DIMENSIONS = [3, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 130.0  # safety margin before 150s kill

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

    U_t = []
    V_t = []
    W_t = []
    
    # A[0:2, 0:2] * B[0:2, 0:2]: Strassen
    strassen_decomp(0, 0, 0, 0, U_t, V_t, W_t)
    
    # A[0:2, 0:2] * B[0:2, 2:4]: Strassen
    strassen_decomp(0, 0, 0, 2, U_t, V_t, W_t)
    
    # A[0:2, 0:2] * B[0:2, 4:6]: Strassen
    strassen_decomp(0, 0, 0, 4, U_t, V_t, W_t)
    
    # A[0:2, 0:2] * B[0:2, 6:8]: Strassen
    strassen_decomp(0, 0, 0, 6, U_t, V_t, W_t)
    
    # A[0:2, 2] * B[2, 0:2]: direct
    for i in range(2):
        for k in range(2):
            direct_term(i, 2, k, U_t, V_t, W_t)
    
    # A[0:2, 2] * B[2, 2:4]: direct
    for i in range(2):
        for k in range(2, 4):
            direct_term(i, 2, k, U_t, V_t, W_t)
    
    # A[0:2, 2] * B[2, 4:6]: direct
    for i in range(2):
        for k in range(4, 6):
            direct_term(i, 2, k, U_t, V_t, W_t)
    
    # A[0:2, 2] * B[2, 6:8]: direct
    for i in range(2):
        for k in range(6, 8):
            direct_term(i, 2, k, U_t, V_t, W_t)
    
    # A[2, 0:2] * B[0:2, 0:2]: direct
    for j in range(2):
        for k in range(2):
            direct_term(2, j, k, U_t, V_t, W_t)
    
    # A[2, 0:2] * B[0:2, 2:4]: direct
    for j in range(2):
        for k in range(2, 4):
            direct_term(2, j, k, U_t, V_t, W_t)
    
    # A[2, 0:2] * B[0:2, 4:6]: direct
    for j in range(2):
        for k in range(4, 6):
            direct_term(2, j, k, U_t, V_t, W_t)
    
    # A[2, 0:2] * B[0:2, 6:8]: direct
    for j in range(2):
        for k in range(6, 8):
            direct_term(2, j, k, U_t, V_t, W_t)
    
    # A[2, 2] * B[2, 0:2]: direct
    for k in range(2):
        direct_term(2, 2, k, U_t, V_t, W_t)
    
    # A[2, 2] * B[2, 2:4]: direct
    for k in range(2, 4):
        direct_term(2, 2, k, U_t, V_t, W_t)
    
    # A[2, 2] * B[2, 4:6]: direct
    for k in range(4, 6):
        direct_term(2, 2, k, U_t, V_t, W_t)
    
    # A[2, 2] * B[2, 6:8]: direct
    for k in range(6, 8):
        direct_term(2, 2, k, U_t, V_t, W_t)
    
    rank = len(W_t)
    
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
    
    # Rank reduction: iteratively try to remove columns that are linear
    # combinations of others while preserving the tensor identity.
    # For column r, the "interaction vector" is the vector over (i,j,k) of
    # U[i,j,r] * V[j,k,r] * W[i,k,r]. If this can be expressed as a linear
    # combination of interaction vectors of other columns, we can absorb it.
    # 
    # More precisely: column r is removable if there exist coefficients c_s
    # for s != r such that for all (i,j,k):
    #   U[i,j,r]*V[j,k,r]*W[i,k,r] = sum_{s!=r} c_s * U[i,j,s]*V[j,k,s]*W[i,k,s]
    # But that's not quite right either. The correct condition is:
    #   For all (i,j,k): sum_r U[i,j,r]*V[j,k,r]*W[i,k,r] = 1
    # If we remove column r, we need:
    #   sum_{s!=r} U[i,j,s]*V[j,k,s]*W[i,k,s] = 1 - U[i,j,r]*V[j,k,r]*W[i,k,r]
    # This is generally not achievable by just dropping a column.
    #
    # Instead, we use a different approach: we can try to find a linear
    # transformation of the columns that zeros out one column. If columns
    # U[:,r], V[:,r], W[:,r] are linearly dependent in a compatible way,
    # we can eliminate r.
    #
    # Simpler approach: check for exact duplicate or negated columns.
    # Also try: for each pair of columns, check if their contribution
    # vectors are proportional.
    
    # Build interaction vectors: for each column r, the vector over all (i,j,k)
    # of the "contribution" is not simply U*V*W element-wise since U, V, W
    # have different indices. Let's think differently.
    #
    # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For column r to be removable, we need T without r to still equal 1 for
    # all (i,j,k). That means:
    #   U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] must be expressible as
    #   sum_{s != r} delta_s * U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
    # for some delta_s, for ALL (i,j,k). This is a very strong condition.
    #
    # A more practical approach: try all pairs of columns and check if
    # one can be written as a linear combination of the other in a way
    # that preserves the sum. This is complex.
    #
    # Let's try a simpler heuristic: check if any column has all zeros in
    # one of U, V, or W (making its contribution zero), or if two columns
    # are identical or negatives.
    
    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
    best_rank = rank
    
    # Check for zero columns
    for r in range(rank):
        if np.allclose(U_arr[:, r], 0) or np.allclose(V_arr[:, r], 0) or np.allclose(W_arr[:, r], 0):
            # This column contributes nothing, remove it
            mask = [s for s in range(rank) if s != r]
            new_U = U_arr[:, mask]
            new_V = V_arr[:, mask]
            new_W = W_arr[:, mask]
            if verify_decomp(new_U, new_V, new_W):
                best_U, best_V, best_W = new_U, new_V, new_W
                best_rank = len(mask)
                break
    
    # Check for duplicate or negated columns
    if best_rank == rank:
        for r in range(rank):
            for s in range(r + 1, rank):
                if np.allclose(U_arr[:, r], U_arr[:, s]) and np.allclose(V_arr[:, r], V_arr[:, s]) and np.allclose(W_arr[:, r], W_arr[:, s]):
                    # Identical columns: can we combine them?
                    # If we have two identical columns, the sum doubles their
                    # contribution. We can't just remove one. But if the
                    # contribution is such that 2*x = x, that only works if x=0.
                    # So identical columns don't help unless they cancel.
                    pass
                elif np.allclose(U_arr[:, r], -U_arr[:, s]) and np.allclose(V_arr[:, r], V_arr[:, s]) and np.allclose(W_arr[:, r], W_arr[:, s]):
                    # U_r = -U_s, V_r = V_s, W_r = W_s
                    # Contribution_r = -U_s * V_s * W_s = -contribution_s
                    # So they cancel! Remove both.
                    mask = [t for t in range(rank) if t != r and t != s]
                    new_U = U_arr[:, mask]
                    new_V = V_arr[:, mask]
                    new_W = W_arr[:, mask]
                    if verify_decomp(new_U, new_V, new_W):
                        best_U, best_V, best_W = new_U, new_V, new_W
                        best_rank = len(mask)
                        break
            if best_rank < rank:
                break
    
    # Try a more systematic approach: for each column r, solve for whether
    # its "signature" can be replicated by others. We use the fact that the
    # tensor equation must hold. For column r, define the vector f_r over
    # (i,j,k) as f_r[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    # Then sum_r f_r = 1 for all (i,j,k). Column r is removable if
    # f_r is in the span of {f_s : s != r} AND the coefficients work out
    # such that the remaining sum equals 1. But that's not sufficient;
    # we need the actual tensor decomposition to work.
    #
    # Let's try: for each column r, check if there exist coefficients c_s
    # such that for all (i,j,k):
    #   U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = sum_{s!=r} c_s * U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s]
    # If so, we can replace column r's contribution. But we can't just
    # modify other columns arbitrarily.
    #
    # Given time constraints, let's just use the simple checks above.
    
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
    
    return {"U": to_nice_list(best_U), "V": to_nice_list(best_V), "W": to_nice_list(best_W)}
# EVOLVE-BLOCK-END
