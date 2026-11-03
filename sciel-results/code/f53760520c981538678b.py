import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    deadline = time.time() + 140  # Leave 10s safety margin

    # Generate the Strassen rank-28 solution
    U28, V28, W28 = _strassen_28(n, m, p)
    
    # Verify the rank-28 solution
    if _verify(U28, V28, W28, n, m, p):
        best_rank = 28
    else:
        # Fallback: naive rank-32
        U32, V32, W32 = _naive_32(n, m, p)
        best_rank = 32
        U28, V28, W28 = U32, V32, W32

    # Try to find rank-27 decomposition (very limited budget)
    time_budget = min(30.0, deadline - time.time())
    if time_budget > 5:
        U27, V27, W27 = _try_rank_27(n, m, p, time_budget)
        if U27 is not None and _verify(U27, V27, W27, n, m, p):
            best_rank = 27
            U28, V28, W28 = U27, V27, W27

    # Convert to proper format
    return {"U": np.array(U28), "V": np.array(V28), "W": np.array(W28)}


def _naive_32(n, m, p):
    """Generate the naive rank-32 decomposition."""
    rank = n * m * p  # 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for a_row in range(n):
        for a_col in range(m):
            for b_row in range(m):
                for b_col in range(p):
                    idx = a_row * m * p + a_col * p + b_col
                    a_idx = a_row * m + a_col
                    b_idx = b_row * p + b_col
                    c_idx = b_col * n + a_row
                    U[a_idx, idx] = 1
                    V[b_idx, idx] = 1
                    W[c_idx, idx] = 1
    
    return U, V, W


def _strassen_28(n, m, p):
    """Generate the standard Strassen rank-28 decomposition for 2x4x4."""
    rank = 28
    U = np.zeros((n * m, rank))  # 8 x 28
    V = np.zeros((m * p, rank))  # 16 x 28
    W = np.zeros((p * n, rank))  # 8 x 28

    # Strassen 2x2x2 decomposition: 7 multiplications
    # For matrices A, B (2x2), C = A*B:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6

    strassen = [
        # (A_indices, B_indices, C_indices) with coefficients
        # M1: (A00+A11)(B00+B11) -> C00+=1, C11+=1
        ({(0,0):1, (1,1):1}, {(0,0):1, (0,1):1}, {(0,0):1, (1,1):1}),
        # M2: (A10+A11)B00 -> C10+=1, C11-=1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3: A00(B10-B11) -> C01+=1, C10+=1
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,0):1}),
        # M4: A11(B01-B10) -> C00+=1, C10+=1
        ({(1,1):1}, {(0,1):1, (1,0):-1}, {(0,0):1, (1,0):1}),
        # M5: (A00+A01)B11 -> C00-=1, C01+=1
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6: (A10-A00)(B00+B10) -> C11+=1
        ({(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}, {(1,1):1}),
        # M7: (A01-A11)(B01+B11) -> C00+=1
        ({(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}, {(0,0):1}),
    ]

    def emit_block(a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off, col_off):
        for s in range(7):
            col = col_off + s
            a_dict, b_dict, c_dict = strassen[s]
            for (ar, ac), coeff in a_dict.items():
                U[(a_row_off + ar) * m + (a_col_off + ac), col] += coeff
            for (br, bc), coeff in b_dict.items():
                V[(b_row_off + br) * p + (b_col_off + bc), col] += coeff
            for (cr, cc), coeff in c_dict.items():
                W[(c_row_off + cr) * n + (c_col_off + cc), col] += coeff

    # Four 2x2 blocks: A is 2x4, B is 4x4, C is 4x2
    # Block (0,0): A rows 0-1, cols 0-1; B rows 0-1, cols 0-1; C rows 0-1, cols 0-1
    emit_block(0, 0, 0, 0, 0, 0, 0)
    # Block (0,1): A rows 0-1, cols 0-1; B rows 0-1, cols 2-3; C rows 2-3, cols 0-1
    emit_block(0, 0, 0, 2, 2, 0, 7)
    # Block (1,0): A rows 0-1, cols 2-3; B rows 2-3, cols 0-1; C rows 0-1, cols 0-1
    emit_block(0, 2, 2, 0, 0, 0, 14)
    # Block (1,1): A rows 0-1, cols 2-3; B rows 2-3, cols 2-3; C rows 2-3, cols 0-1
    emit_block(0, 2, 2, 2, 2, 0, 21)

    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    nm = n * m
    mp = m * p
    pn = p * n
    rank = U.shape[1]
    
    # Build the target tensor
    T = np.zeros((nm, mp, pn), dtype=np.int64)
    for c_row in range(p):
        for c_col in range(n):
            for b_row in range(m):
                a_idx = c_col * m + b_row
                b_idx = b_row * p + c_row
                c_idx = c_row * n + c_col
                T[a_idx, b_idx, c_idx] = 1

    # Compute the decomposition
    # T_decomp[i,j,k] = sum_l U[i,l] * V[j,l] * W[k,l]
    # Efficient computation: for each l, add outer(U[:,l], V[:,l]) * W[k,l]
    T_decomp = np.zeros((nm, mp, pn), dtype=np.int64)
    for l in range(rank):
        uv = np.outer(U[:, l], V[:, l])
        T_decomp += uv[:, :, np.newaxis] * W[:, l][np.newaxis, np.newaxis, :]

    return np.array_equal(T_decomp, T)


def _try_rank_27(n, m, p, time_budget):
    """
    Attempt to find a rank-27 decomposition using structured search.
    
    Strategy: Start from the rank-28 Strassen solution and try to find
    a linear combination of two rank-1 terms that can be replaced by one,
    or use a perturbation approach.
    
    For rank 27, we need: T[i,j,k] = sum_{l=1}^{27} U[i,l]*V[j,l]*W[k,l]
    
    We use the following approach:
    1. Take the Strassen rank-28 solution
    2. Try to find a non-trivial linear relation among the 28 rank-1 tensors
    3. If found, we can eliminate one term
    """
    nm = n * m  # 8
    mp = m * p  # 16
    pn = p * n  # 8
    r = 27

    # Build the matrix multiplication tensor
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for c_row in range(p):
        for c_col in range(n):
            for b_row in range(m):
                a_idx = c_col * m + b_row
                b_idx = b_row * p + c_row
                c_idx = c_row * n + c_col
                T[a_idx, b_idx, c_idx] = 1

    U28, V28, W28 = _strassen_28(n, m, p)
    
    # Approach: Try to find if any two columns can be merged
    # For rank 27, we need to find U', V', W' of size 27 such that
    # sum_{l=1}^{27} U'[i,l]*V'[j,l]*W'[k,l] = T[i,j,k]
    
    # Strategy: Use the fact that if we have a rank-28 decomposition,
    # we can try to find a rank-27 by looking for a "flat" direction.
    
    # More practical: Try random perturbations of the rank-28 solution
    # and solve for a rank-27 fit
    
    rng = np.random.default_rng(12345)
    
    # Try a different structured approach:
    # The 2x4x4 tensor can be viewed as 2x2x2 tensor with 2x2 blocks.
    # Rank of 2x2x2 is 7 (Strassen). For 2x4x4, we get 4*7=28.
    # Can we do better by sharing across blocks?
    
    # Let's try: parameterize U, V, W with 27 columns and use
    # a greedy/iterative approach
    
    # Actually, let's try a more direct approach:
    # Fix 27 random rank-1 tensors and check if their span contains T
    
    # For a rank-r decomposition, T must be in the span of {u_l outer v_l outer w_l}
    # This is a nonlinear condition.
    
    # Alternative: Use the "bilinear" structure.
    # For fixed W, the problem becomes: T[:,:,k] = sum_l W[k,l] * (U[:,l] outer V[:,l])
    # This is linear in U and V for fixed W, but bilinear overall.
    
    # Let's try: fix W randomly, then solve for U, V
    # For each k: T[:,:,k] = sum_l W[k,l] * (U[:,l] outer V[:,l])
    # vec(T[:,:,k]) = sum_l W[k,l] * (V[:,l] kron U[:,l])
    # This is a linear system in the variables U[:,l]*V[:,l] for each l
    
    # Actually, this is still nonlinear. Let me try a different approach.
    
    # Approach: Start with rank-28, try to find a null vector in the
    # space of "rank-1 tensor combinations"
    
    # The 28 rank-1 tensors from Strassen are:
    # T_l = U[:,l] outer V[:,l] outer W[:,l] for l=0,...,27
    # If there exists alpha_0,...,alpha_27 (not all zero) such that
    # sum_l alpha_l * T_l = 0, then we have a linear dependency.
    # But that would mean rank < 28, which we know is not the case for exact rank.
    
    # For rank 27, we need a different kind of relation.
    
    # Let's try a numerical optimization approach with limited budget
    # This is unlikely to succeed but worth a brief attempt
    
    if time_budget < 10:
        return None, None, None
    
    # Try: random initialization + Gauss-Seidel style iteration
    # Initialize U, V, W randomly with 27 columns
    # Iterate: fix two, solve for the third
    
    best_residual = np.inf
    best_U, best_V, best_W = None, None, None
    
    n_attempts = min(50, int(time_budget / 0.5))  # At least 0.5s per attempt
    
    for attempt in range(n_attempts):
        if time.time() > time.time() + time_budget:
            break
        
        # Random initialization
        U_init = rng.standard_normal((nm, r))
        V_init = rng.standard_normal((mp, r))
        W_init = rng.standard_normal((pn, r))
        
        # Normalize
        for l in range(r):
            norm_u = np.linalg.norm(U_init[:, l])
            norm_v = np.linalg.norm(V_init[:, l])
            norm_w = np.linalg.norm(W_init[:, l])
            if norm_u > 1e-10 and norm_v > 1e-10 and norm_w > 1e-10:
                U_init[:, l] /= norm_u
                V_init[:, l] /= norm_v
                W_init[:, l] /= norm_w
        
        # Iterate: fix U and V, solve for W
        # For each k: T[:,:,k] = sum_l W[k,l] * (U[:,l] outer V[:,l])
        # This is a linear system: M * w_k = b_k
        # where M[l, (i,j)] = U[i,l] * V[j,l], b_k[(i,j)] = T[i,j,k]
        
        M = np.zeros((r, nm * mp))
        for l in range(r):
            M[l, :] = np.outer(U_init[:, l], V_init[:, l]).flatten()
        
        W_sol = np.zeros((pn, r))
        for k in range(pn):
            b_k = T[:, :, k].flatten()
            # Solve M * w_k = b_k (least squares)
            try:
                w_k, residuals, rank_m, sv = np.linalg.lstsq(M, b_k, rcond=None)
                W_sol[k, :] = w_k
            except:
                continue
        
        # Now fix U and W, solve for V
        # For each j: T[:,j,:] = sum_l V[j,l] * (U[:,l] outer W[:,l])
        # This is: for each (i,k): T[i,j,k] = sum_l V[j,l] * U[i,l] * W[k,l]
        # M2[l, (i,k)] = U[i,l] * W[k,l], b_j[(i,k)] = T[i,j,k]
        
        M2 = np.zeros((r, nm * pn))
        for l in range(r):
            M2[l, :] = np.outer(U_init[:, l], W_sol[:, l]).flatten()
        
        V_sol = np.zeros((mp, r))
        for j in range(mp):
            b_j = T[:, j, :].flatten()
            try:
                v_j, residuals, rank_m, sv = np.linalg.lstsq(M2, b_j, rcond=None)
                V_sol[j, :] = v_j
            except:
                continue
        
        # Compute residual
        T_decomp = np.zeros((nm, mp, pn))
        for l in range(r):
            T_decomp += np.outer(np.outer(U_init[:, l], V_sol[:, l]), W_sol[:, l])
        
        residual = np.linalg.norm(T_decomp - T)
        
        if residual < best_residual:
            best_residual = residual
            best_U, best_V, best_W = U_init.copy(), V_sol.copy(), W_sol.copy()
        
        if residual < 1e-10:
            break
    
    # Check if we found an exact solution
    if best_U is not None and best_residual < 1e-8:
        # Round to nearest integers/half-integers
        best_U = np.round(best_U * 2) / 2
        best_V = np.round(best_V * 2) / 2
        best_W = np.round(best_W * 2) / 2
        
        # Verify
        T_decomp = np.zeros((nm, mp, pn))
        for l in range(r):
            T_decomp += np.outer(np.outer(best_U[:, l], best_V[:, l]), best_W[:, l])
        
        if np.allclose(T_decomp, T, atol=1e-10):
            return best_U, best_V, best_W
    
    return None, None, None
# EVOLVE-BLOCK-END
