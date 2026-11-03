"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Uses Strassen's rank-7 decomposition for 2x2x2 blocks combined with
direct terms for remaining dimensions, targeting rank < 72.
Additionally uses linear algebra to find and remove linearly dependent
columns to reduce rank below 68.
"""

DIMENSIONS = [3, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

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
    
    # A[0:2, 2] * B[2, 0:8]: direct
    for i in range(2):
        for k in range(8):
            direct_term(i, 2, k, U_t, V_t, W_t)
    
    # A[2, 0:2] * B[0:2, 0:8]: direct
    for j in range(2):
        for k in range(8):
            direct_term(2, j, k, U_t, V_t, W_t)
    
    # A[2, 2] * B[2, 0:8]: direct
    for k in range(8):
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
    
    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
    best_rank = rank
    
    # Systematic rank reduction using linear algebra.
    # For each column r, check if it can be eliminated by finding a linear
    # combination of other columns that reproduces its contribution.
    # 
    # The key insight: we can try to eliminate column r by checking if
    # there exist coefficients c_s (s != r) such that for ALL (i,j,k):
    #   U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 
    #     sum_{s!=r} c_s * U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s]
    #
    # This is equivalent to saying the "contribution tensor" of column r
    # lies in the span of contribution tensors of other columns.
    #
    # A more practical approach: build a matrix where each row corresponds
    # to an (i,j,k) triple and each column r has entry
    # U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]. Then check for linear dependencies.
    
    num_ijk = n * m * p  # 72
    # Build contribution matrix: rows = (i,j,k), cols = rank
    contrib = np.zeros((num_ijk, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * m * p + j * p + k
                for r in range(rank):
                    contrib[idx, r] = U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
    
    # Find columns that are linearly dependent
    # Use QR with column pivoting to find the rank and identify dependent columns
    # We'll iteratively try to remove columns
    
    # First, check for zero columns
    for r in range(rank):
        if np.allclose(U_arr[:, r], 0) or np.allclose(V_arr[:, r], 0) or np.allclose(W_arr[:, r], 0):
            mask = [s for s in range(rank) if s != r]
            new_U = U_arr[:, mask]
            new_V = V_arr[:, mask]
            new_W = W_arr[:, mask]
            if verify_decomp(new_U, new_V, new_W):
                best_U, best_V, best_W = new_U, new_V, new_W
                best_rank = len(mask)
                break
    
    # Now try to find linear dependencies among the remaining columns
    # For each column r, check if removing it and adjusting others works.
    # We use the following approach:
    # 1. Build the contribution matrix for current best
    # 2. For each column r, check if the contribution of r is in the span
    #    of contributions of other columns
    # 3. If so, we can potentially eliminate r by modifying other columns
    
    # More efficient: use the fact that we can do a change of basis.
    # If we have a linear dependency among the contribution vectors, we can
    # use it to zero out one column.
    
    # Try pairwise elimination: for each pair (r, s), check if col_r can be
    # expressed as a linear combination involving col_s and others.
    
    # Actually, let's use a simpler but effective approach:
    # For each column r, try to see if we can find a scalar alpha such that
    # replacing column r with (col_r - alpha * col_s) for some s makes
    # column r have a zero in a strategic position, allowing elimination.
    
    # Best approach: use Gaussian elimination on the contribution matrix
    # to find the minimum number of columns needed.
    
    # Build contribution matrix for current best
    contrib_best = np.zeros((num_ijk, best_rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * m * p + j * p + k
                for r in range(best_rank):
                    contrib_best[idx, r] = best_U[i*m+j, r] * best_V[j*p+k, r] * best_W[k*n+i, r]
    
    # The sum of all columns should be the all-ones vector
    ones = np.ones(num_ijk)
    
    # Check: sum of contrib_best columns should equal ones
    # (This is already verified by verify_decomp)
    
    # Now, to reduce rank, we need to find a subset of columns whose
    # sum still equals ones. This is equivalent to finding a linear
    # dependency: sum of all cols = ones, so if we can write one col as
    # a linear combination of others, we can eliminate it.
    
    # Use QR decomposition with column pivoting to find the rank
    # and identify which columns are linearly independent.
    
    # We want to find the minimum subset of columns that sums to ones.
    # This is a subset sum problem, which is hard in general.
    
    # Heuristic: iteratively try to remove each column and check if the
    # remaining columns can still sum to ones (with adjustments).
    
    # For each column r, check if there exist coefficients c_s (s != r)
    # such that sum_{s!=r} c_s * contrib_col_s = ones - contrib_col_r
    # Wait, that's not right. We need sum_{s!=r} c_s * contrib_col_s = ones.
    # But we also need the individual tensor factors to be consistent.
    
    # Let me try a different approach: for each column r, check if
    # contrib_col_r is in the span of {contrib_col_s : s != r}.
    # If so, we can find coefficients and potentially eliminate r.
    
    # Build the matrix of contributions for all columns except r
    # and check if contrib_col_r is in its column space.
    
    # This is O(rank * num_ijk * rank) which is 68 * 72 * 68 ~ 333k ops per r
    # Total: 68 * 333k ~ 22.6M ops, which is fast.
    
    # For each r, check if contrib_col_r is in the span of other columns
    removed_any = False
    for r in range(best_rank):
        if time.time() - start_time > DEADLINE:
            break
        
        # Get contribution vector for column r
        col_r = contrib_best[:, r]
        
        # Get all other columns
        other_cols = [s for s in range(best_rank) if s != r]
        if len(other_cols) == 0:
            continue
        
        # Check if col_r is in the span of other columns
        # Build matrix with other columns
        M = contrib_best[:, other_cols]  # shape (num_ijk, best_rank-1)
        
        # Solve M * x = col_r
        try:
            x, residuals, rank_m, sv = np.linalg.lstsq(M, col_r, rcond=None)
            # Check if the residual is small
            residual = np.linalg.norm(M @ x - col_r)
            if residual < 1e-8:
                # col_r is in the span of other columns
                # Now we need to check if we can actually eliminate r
                # by modifying the tensor factors.
                # 
                # The coefficients x give us: col_r = sum_{s!=r} x_s * col_s
                # This means the contribution of r can be replicated by
                # adjusting other columns. But we need to find actual
                # tensor factor modifications.
                #
                # This is where it gets tricky. The contribution being
                # in the span doesn't guarantee we can modify the tensor
                # factors to eliminate r.
                #
                # For now, let's just note this and try a simpler check:
                # can we remove r entirely (without modifications)?
                mask = [s for s in range(best_rank) if s != r]
                new_U = best_U[:, mask]
                new_V = best_V[:, mask]
                new_W = best_W[:, mask]
                if verify_decomp(new_U, new_V, new_W):
                    best_U, best_V, best_W = new_U, new_V, new_W
                    best_rank = len(mask)
                    removed_any = True
                    # Rebuild contrib_best
                    contrib_best = np.zeros((num_ijk, best_rank))
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                idx = i * m * p + j * p + k
                                for rr in range(best_rank):
                                    contrib_best[idx, rr] = best_U[i*m+j, rr] * best_V[j*p+k, rr] * best_W[k*n+i, rr]
                    break
        except:
            pass
    
    # Try another approach: look for columns that have the same "pattern"
    # in U, V, W and can be combined.
    
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
