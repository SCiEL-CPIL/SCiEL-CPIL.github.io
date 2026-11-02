import numpy as np
import time
import itertools

DIMENSIONS = [2, 2, 4]

def _build_rank14():
    """
    Constructs the rank-14 decomposition for 2x2x4 matrix multiplication.
    This is the direct sum of two rank-7 Strassen decompositions for 2x2x2.
    """
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 products for 2x2 matrices
    # Left Block (cols 0,1) -> C cols 0,1
    # M1_L
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2_L
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3_L
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4_L
    U[3][3] = 1
    V[4][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5_L
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6_L
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7_L
    U[1][6] = 1; U[3][6] = -1
    V[4][6] = 1; V[5][6] = 1
    W[0][6] = 1

    # Right Block (cols 2,3) -> C cols 2,3
    # M1_R
    U[0][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1
    
    # M2_R
    U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1
    W[5][8] = 1; W[7][8] = -1
    
    # M3_R
    U[0][9] = 1
    V[3][9] = 1; V[7][9] = -1
    W[6][9] = 1; W[7][9] = 1
    
    # M4_R
    U[3][10] = 1
    V[6][10] = 1; V[2][10] = -1
    W[4][10] = 1; W[5][10] = 1
    
    # M5_R
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[4][11] = -1; W[6][11] = 1
    
    # M6_R
    U[2][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[7][12] = 1
    
    # M7_R
    U[1][13] = 1; U[3][13] = -1
    V[6][13] = 1; V[7][13] = 1
    W[4][13] = 1

    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W):
    """
    Verifies the decomposition exactly.
    """
    n, m, p = 2, 2, 4
    rank = len(U[0])
    
    # Check non-zero terms
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                t_val = 0.0
                for r in range(rank):
                    t_val += U[i][r] * V[j][r] * W[k][r]
                if abs(t_val - 1.0) > 1e-9:
                    return False
    
    # Check zero terms
    for i in range(n * m):
        for j in range(m * p):
            for k in range(p * n):
                a = i // m
                b = i % m
                b_prime = j // p
                c = j % p
                c_prime = k // n
                a_prime = k % n
                
                if a == a_prime and b == b_prime:
                    continue
                
                t_val = 0.0
                for r in range(rank):
                    t_val += U[i][r] * V[j][r] * W[k][r]
                if abs(t_val) > 1e-9:
                    return False
    
    return True

def _search_rank13():
    """
    Attempts to find a rank-13 decomposition by trying to merge one right-block 
    Strassen term into a linear combination of left-block terms plus one new term.
    """
    start_time = time.time()
    deadline = start_time + 120.0  # Leave margin for verification and return
    
    n, m, p = 2, 2, 4
    rank = 13
    
    # Get the base rank-14 decomposition
    base = _build_rank14()
    U_base = np.array(base["U"], dtype=float)
    V_base = np.array(base["V"], dtype=float)
    W_base = np.array(base["W"], dtype=float)
    
    # Left block terms: indices 0-6
    # Right block terms: indices 7-13
    
    # For each right-block term r_R in {7,...,13}, try to express it as:
    #   term_r_R = sum_{r_L in {0,...,6}} alpha[r_L] * term_r_L + beta * term_new
    # where term_new is a single rank-1 tensor with entries in {-1, 0, 1}
    
    # A rank-1 term is defined by (u, v, w) where u in R^4, v in R^8, w in R^8
    # The contribution is u[i]*v[j]*w[k]
    
    # For a given r_R, the "residual" after subtracting the left-block combination
    # must be a rank-1 tensor.
    
    # Let's enumerate: for each r_R (7 choices), for each choice of which left terms
    # to use (2^7 = 128 subsets), solve for alpha coefficients, then check if
    # residual is rank-1.
    
    # Actually, let's be more systematic. For each r_R:
    #   residual = term_r_R - sum_{r_L} alpha[r_L] * term_r_L
    # We want residual to be rank-1.
    
    # The space of tensors is 4*8*8 = 256 dimensional.
    # term_r_R is a specific tensor.
    # We have 7 free parameters (alpha[0]..alpha[6]).
    # We need the residual to be rank-1, which is a 256-7 = 249 dimensional constraint.
    # This is very restrictive, so we need to be clever.
    
    # Alternative: try small integer coefficients for alpha.
    
    # Let's try: for each r_R, try all combinations of alpha in {-1, 0, 1}^7
    # That's 3^7 = 2187 per r_R, times 7 = 15309 total.
    # For each, compute residual and check if it's rank-1.
    
    # A tensor T is rank-1 iff all 2x2 minors of any flattening are zero.
    # Equivalently, the matrix T[i, (j,k)] has rank 1.
    
    # Let's implement a fast rank-1 check.
    
    def _is_rank1_residual(residual):
        """Check if a 4x8x8 tensor is rank-1."""
        # Reshape to 4 x 64
        mat = residual.reshape(4, 64)
        # Check if all rows are scalar multiples of each other
        # Find first non-zero row
        nonzero_rows = []
        for i in range(4):
            if np.any(np.abs(mat[i]) > 1e-9):
                nonzero_rows.append(i)
        
        if len(nonzero_rows) == 0:
            return True, None  # Zero tensor is rank 0 (which is <= 1)
        
        ref_row = mat[nonzero_rows[0]]
        
        for i in nonzero_rows[1:]:
            row = mat[i]
            # Check if row = c * ref_row for some c
            # Find a non-zero element in ref_row
            ref_nonzero = np.where(np.abs(ref_row) > 1e-9)[0]
            if len(ref_nonzero) == 0:
                # ref_row is zero but we said it's nonzero, contradiction
                return False, None
            
            c = row[ref_nonzero[0]] / ref_row[ref_nonzero[0]]
            if np.max(np.abs(row - c * ref_row)) > 1e-9:
                return False, None
        
        return True, ref_row
    
    # For each right-block term, try to express it using left-block terms
    for r_R in range(7, 14):
        if time.time() > deadline:
            break
        
        # Get the tensor for term r_R
        T_R = np.outer(np.outer(U_base[:, r_R], V_base[:, r_R]), W_base[:, r_R]).reshape(4, 8, 8)
        
        # Try all combinations of alpha in {-1, 0, 1}^7
        for alpha_combo in itertools.product([-1, 0, 1], repeat=7):
            if time.time() > deadline:
                break
            
            # Compute residual = T_R - sum alpha[r_L] * T_L
            residual = T_R.copy()
            for r_L in range(7):
                if alpha_combo[r_L] != 0:
                    T_L = np.outer(np.outer(U_base[:, r_L], V_base[:, r_L]), W_base[:, r_L]).reshape(4, 8, 8)
                    residual -= alpha_combo[r_L] * T_L
            
            # Check if residual is rank-1
            is_rank1, ref_row = _is_rank1_residual(residual)
            
            if is_rank1:
                # We found a valid decomposition!
                # Build the rank-13 decomposition
                U_new = [[0] * rank for _ in range(n * m)]
                V_new = [[0] * rank for _ in range(m * p)]
                W_new = [[0] * rank for _ in range(p * n)]
                
                # Copy left block terms (0-6)
                for r_L in range(7):
                    for i in range(n * m):
                        U_new[i][r_L] = U_base[i][r_L]
                    for j in range(m * p):
                        V_new[j][r_L] = V_base[j][r_L]
                    for k in range(p * n):
                        W_new[k][r_L] = W_base[k][r_L]
                
                # Copy right block terms except r_R
                new_idx = 7
                for r_R2 in range(7, 14):
                    if r_R2 == r_R:
                        continue
                    for i in range(n * m):
                        U_new[i][new_idx] = U_base[i][r_R2]
                    for j in range(m * p):
                        V_new[j][new_idx] = V_base[j][r_R2]
                    for k in range(p * n):
                        W_new[k][new_idx] = W_base[k][r_R2]
                    new_idx += 1
                
                # Add the new rank-1 term at index 12
                # residual = u * v * w where u is the reference row pattern
                # We need to factor residual into u, v, w
                # residual[i,j,k] = u[i] * v[j] * w[k]
                
                # Find u from the first non-zero row
                mat = residual.reshape(4, 64)
                u = np.zeros(4)
                nonzero_rows = []
                for i in range(4):
                    if np.any(np.abs(mat[i]) > 1e-9):
                        nonzero_rows.append(i)
                        u[i] = 1.0  # We'll normalize later
                
                if len(nonzero_rows) > 0:
                    ref_row = mat[nonzero_rows[0]]
                    # Find v and w from ref_row
                    # ref_row[j*8+k] = u[ref] * v[j] * w[k] = v[j] * w[k]
                    # So ref_row is an outer product of v and w
                    ref_2d = ref_row.reshape(8, 8)
                    
                    # Find v and w
                    v = np.zeros(8)
                    w = np.zeros(8)
                    
                    # Find first non-zero column
                    non_zero_cols = np.where(np.any(np.abs(ref_2d) > 1e-9, axis=0))[0]
                    if len(non_zero_cols) > 0:
                        col_idx = non_zero_cols[0]
                        w[col_idx] = 1.0
                        # v[j] = ref_2d[j, col_idx] / w[col_idx]
                        for j in range(8):
                            if np.abs(ref_2d[j, col_idx]) > 1e-9:
                                v[j] = ref_2d[j, col_idx]
                        
                        # Check if this factorization works
                        if np.max(np.abs(v[:, None] * w[None, :] - ref_2d)) < 1e-9:
                            # Valid factorization
                            pass
                        else:
                            # Try another approach
                            non_zero_rows_2d = np.where(np.any(np.abs(ref_2d) > 1e-9, axis=1))[0]
                            if len(non_zero_rows_2d) > 0:
                                row_idx = non_zero_rows_2d[0]
                                v[row_idx] = 1.0
                                for k in range(8):
                                    if np.abs(ref_2d[row_idx, k]) > 1e-9:
                                        w[k] = ref_2d[row_idx, k]
                                if np.max(np.abs(v[:, None] * w[None, :] - ref_2d)) > 1e-9:
                                    continue  # This combination doesn't work
                
                # Scale u, v, w to make entries small integers if possible
                # For now, just use the computed values
                for i in range(4):
                    U_new[i][12] = u[i]
                for j in range(8):
                    V_new[j][12] = v[j]
                for k in range(8):
                    W_new[k][12] = w[k]
                
                # Verify
                if _verify_decomposition(U_new, V_new, W_new):
                    return {"U": U_new, "V": V_new, "W": W_new}
    
    return None

def construct():
    # Try to find rank 13
    result = _search_rank13()
    if result is not None:
        if _verify_decomposition(result["U"], result["V"], result["W"]):
            return result
    
    # Fallback to rank 14
    result = _build_rank14()
    if _verify_decomposition(result["U"], result["V"], result["W"]):
        return result
    
    # Final fallback to schoolbook (rank 16)
    n, m, p = 2, 2, 4
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    idx = 0
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                U[i][idx] = 1
                V[j][idx] = 1
                W[k][idx] = 1
                idx += 1
    return {"U": U, "V": V, "W": W}
