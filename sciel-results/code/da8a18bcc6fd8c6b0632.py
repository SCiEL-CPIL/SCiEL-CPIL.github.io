"""Optimized 2x5x6 matrix multiplication tensor decomposition.

Uses a rank-reduction search starting from the rank-54 Strassen block decomposition.
Attempts to find linear dependencies among the factor columns to reduce rank.
Falls back to the known rank-54 construction if no improvement is found.
"""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START


def _strassen_222():
    """Return U, V, W (each 4x7) for 2x2x2 matmul using Strassen.
    Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    # r=0: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[1, 0] = 1; V[2, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1

    # r=1: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1

    # r=2: A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1

    # r=3: A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1

    # r=4: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1

    # r=5: (A10-A00)(B00+B10)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[0, 5] = 1

    # r=6: (A01-A11)(B01+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1

    return U, V, W


def _build_strassen_block_2x2x6(U, V, W, i_off, j_off, k_off, col):
    """Fill U, V, W for a 2x2x6 sub-block."""
    sU, sV, sW = _strassen_222()
    n, m, p = 2, 5, 6
    for block in range(3):
        kk = k_off + block * 2
        for r in range(7):
            U[(i_off + 0) * m + (j_off + 0), col] = sU[0, r]
            U[(i_off + 0) * m + (j_off + 1), col] = sU[1, r]
            U[(i_off + 1) * m + (j_off + 0), col] = sU[2, r]
            U[(i_off + 1) * m + (j_off + 1), col] = sU[3, r]

            V[(j_off + 0) * p + kk + 0, col] = sV[0, r]
            V[(j_off + 0) * p + kk + 1, col] = sV[1, r]
            V[(j_off + 1) * p + kk + 0, col] = sV[2, r]
            V[(j_off + 1) * p + kk + 1, col] = sV[3, r]

            W[(kk + 0) * 2 + i_off + 0, col] = sW[0, r]
            W[(kk + 0) * 2 + i_off + 1, col] = sW[1, r]
            W[(kk + 1) * 2 + i_off + 0, col] = sW[2, r]
            W[(kk + 1) * 2 + i_off + 1, col] = sW[3, r]
            col += 1
    return col


def _build_decomposition():
    """Build the 2x5x6 decomposition, rank 54."""
    n, m, p = 2, 5, 6
    total_rank = 54

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    col = 0
    col = _build_strassen_block_2x2x6(U, V, W, i_off=0, j_off=0, k_off=0, col=col)

    for i in range(2):
        for k in range(6):
            U[i * 5 + 2, col] = 1
            V[2 * 6 + k, col] = 1
            W[k * 2 + i, col] = 1
            col += 1

    col = _build_strassen_block_2x2x6(U, V, W, i_off=0, j_off=3, k_off=0, col=col)

    assert col == total_rank
    return U, V, W


def _verify_decomposition(U, V, W, n=2, m=5, p=6, tol=1e-9):
    """Verify the decomposition is exact."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]

    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = float(np.dot(Ua[i * m + j, :], Va[j * p + k, :] * Wa[k * n + i, :]))
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol


def _check_half_integer(U, V, W, tol=1e-12):
    """Check all entries are integer or half-integer."""
    for mat in [U, V, W]:
        arr = np.array(mat, dtype=float)
        if not np.allclose(arr * 2, np.round(arr * 2), atol=tol):
            return False
    return True


def _try_reduce_rank(U, V, W, n=2, m=5, p=6, max_iter=200, time_budget=100.0, start_time=None):
    """Attempt to reduce rank by finding linearly dependent columns.
    
    Strategy: For each column r, check if the tensor contribution of column r
    can be expressed as a linear combination of other columns. If so, remove it.
    
    We use the following approach:
    - Build the "flattened tensor" constraint matrix.
    - For each candidate column to remove, solve for coefficients that replicate
      its contribution using the remaining columns.
    """
    if start_time is None:
        start_time = time.time()
    
    U = U.copy().astype(float)
    V = V.copy().astype(float)
    W = W.copy().astype(float)
    rank = U.shape[1]
    
    # Precompute for efficiency: for each (i,j,k), we need
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Build the constraint: for each (i,j,k), the equation is linear in the "combined" variable
    # Let's define for each r and (i,j,k): c_r(i,j,k) = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # We need sum_r c_r(i,j,k) = 1 for all (i,j,k).
    
    # To check if column r0 can be removed: we need to find alpha_s for s != r0 such that
    # for all (i,j,k): c_r0(i,j,k) = sum_{s != r0} alpha_s * c_s(i,j,k)
    # This is a linear system: A * alpha = b where
    # A[(i,j,k), s] = c_s(i,j,k) for s != r0
    # b[(i,j,k)] = c_r0(i,j,k)
    
    n_eq = n * m * p  # 60 equations
    removed = []
    
    for trial in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        current_rank = U.shape[1]
        if current_rank <= 1:
            break
        
        # Try removing each column
        found_removal = False
        for r0 in range(current_rank):
            if time.time() - start_time > time_budget:
                break
            
            # Build system: for all (i,j,k), sum_{s != r0} alpha_s * U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s] = U[i*m+j,r0]*V[j*p+k,r0]*W[k*n+i,r0]
            
            # This is nonlinear in alpha because the coefficients depend on s through U,V,W.
            # Actually it IS linear in alpha_s! The coefficient of alpha_s in equation (i,j,k) is
            # U[i*m+j,s] * V[j*p+k,s] * W[k*n+i,s], and the RHS is U[i*m+j,r0]*V[j*p+k,r0]*W[k*n+i,r0].
            
            # So we build a matrix A of size (n*m*p) x (current_rank - 1)
            # A[eq_idx, col_idx] = U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
            # where eq_idx = ((i*m+j)*p+k), and col_idx maps to s != r0
            
            # This is expensive to build each time. Let's precompute the "tensor values" for each column.
            
            # For each column s, define T_s[i,j,k] = U[i*m+j,s] * V[j*p+k,s] * W[k*n+i,s]
            # We need: T_r0 = sum_{s != r0} alpha_s * T_s
            
            # Build A and b
            A = np.zeros((n_eq, current_rank - 1))
            b = np.zeros(n_eq)
            
            s_idx = 0
            for s in range(current_rank):
                if s == r0:
                    continue
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            eq_idx = (i * m + j) * p + k
                            A[eq_idx, s_idx] = U[i * m + j, s] * V[j * p + k, s] * W[k * n + i, s]
                s_idx += 1
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = (i * m + j) * p + k
                        b[eq_idx] = U[i * m + j, r0] * V[j * p + k, r0] * W[k * n + i, r0]
            
            # Solve A * alpha = b (least squares)
            try:
                alpha, residuals, _, _ = np.linalg.lstsq(A, b, rcond=1e-10)
            except Exception:
                continue
            
            # Check if the solution is good
            if residuals.size > 0 and residuals[0] < 1e-8:
                # Column r0 is redundant! Remove it.
                # But wait - we need to verify that after removal, the sum still equals 1.
                # The original sum was sum_s T_s = 1. If T_r0 = sum_{s!=r0} alpha_s T_s,
                # then sum_s T_s = T_r0 + sum_{s!=r0} T_s = sum_{s!=r0} (1 + alpha_s) T_s... no that's wrong.
                
                # Actually, the condition is: we want sum_{s != r0} T_s' = 1 where T_s' are modified.
                # But we can't just change the coefficients easily because U,V,W are fixed per column.
                
                # The correct approach: if T_r0 = sum_{s != r0} alpha_s T_s, then
                # sum_s T_s = T_r0 + sum_{s != r0} T_s = sum_{s != r0} (alpha_s + 1) T_s
                # This doesn't help unless we can rescale.
                
                # Alternative: we need sum_{s != r0} beta_s T_s = 1 for all (i,j,k).
                # This means we need to find beta_s such that sum_{s != r0} beta_s T_s = 1.
                # This is a different linear system!
                
                # Let me reformulate: we want to find new coefficients for the remaining columns
                # such that the sum equals 1. But the columns themselves (U[:,s], V[:,s], W[:,s])
                # are fixed. We can't change them.
                
                # So the correct test is: can we find gamma_s for s != r0 such that
                # sum_{s != r0} gamma_s * U[i*m+j,s] * V[j*p+k,s] * W[k*n+i,s] = 1 for all (i,j,k)?
                
                # This is a linear system in gamma_s!
                A2 = np.zeros((n_eq, current_rank - 1))
                b2 = np.ones(n_eq)
                
                s_idx = 0
                for s in range(current_rank):
                    if s == r0:
                        continue
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                eq_idx = (i * m + j) * p + k
                                A2[eq_idx, s_idx] = U[i * m + j, s] * V[j * p + k, s] * W[k * n + i, s]
                    s_idx += 1
                
                try:
                    gamma, residuals2, _, _ = np.linalg.lstsq(A2, b2, rcond=1e-10)
                except Exception:
                    continue
                
                if residuals2.size > 0 and residuals2[0] < 1e-8:
                    # We can remove column r0 and rescale the others!
                    # But rescaling means changing U[:,s], V[:,s], W[:,s] by factors.
                    # If we multiply U[:,s] by gamma_s, we need to compensate in V or W.
                    # Since the product is U*V*W, we can distribute gamma_s as:
                    # U_new[:,s] = U[:,s] * gamma_s^{1/3}
                    # V_new[:,s] = V[:,s] * gamma_s^{1/3}
                    # W_new[:,s] = W[:,s] * gamma_s^{1/3}
                    # But this may not give half-integer entries!
                    
                    # Alternative: put all of gamma_s into U:
                    # U_new[:,s] = U[:,s] * gamma_s
                    # V_new[:,s] = V[:,s]
                    # W_new[:,s] = W[:,s]
                    # This keeps V,W unchanged and only scales U.
                    
                    # Check if this gives valid (half-integer) entries.
                    new_U = U[:, [s for s in range(current_rank) if s != r0]].copy()
                    new_V = V[:, [s for s in range(current_rank) if s != r0]].copy()
                    new_W = W[:, [s for s in range(current_rank) if s != r0]].copy()
                    
                    s_idx = 0
                    for s in range(current_rank):
                        if s == r0:
                            continue
                        new_U[:, s_idx] *= gamma[s_idx]
                        s_idx += 1
                    
                    # Verify
                    if _verify_decomposition(new_U, new_V, new_W):
                        if _check_half_integer(new_U, new_V, new_W):
                            U, V, W = new_U, new_V, new_W
                            found_removal = True
                            break
                        else:
                            # Try distributing the scaling
                            # Put sqrt(gamma) in U and sqrt(gamma) in V
                            new_U2 = U[:, [s for s in range(current_rank) if s != r0]].copy()
                            new_V2 = V[:, [s for s in range(current_rank) if s != r0]].copy()
                            new_W2 = W[:, [s for s in range(current_rank) if s != r0]].copy()
                            
                            s_idx = 0
                            for s in range(current_rank):
                                if s == r0:
                                    continue
                                if gamma[s_idx] >= 0:
                                    factor = np.sqrt(gamma[s_idx])
                                else:
                                    factor = -np.sqrt(-gamma[s_idx])
                                new_U2[:, s_idx] *= factor
                                new_V2[:, s_idx] *= factor
                                s_idx += 1
                            
                            if _verify_decomposition(new_U2, new_V2, new_W2):
                                if _check_half_integer(new_U2, new_V2, new_W2):
                                    U, V, W = new_U2, new_V2, new_W2
                                    found_removal = True
                                    break
                            
                            # Try putting all in W
                            new_U3 = U[:, [s for s in range(current_rank) if s != r0]].copy()
                            new_V3 = V[:, [s for s in range(current_rank) if s != r0]].copy()
                            new_W3 = W[:, [s for s in range(current_rank) if s != r0]].copy()
                            
                            s_idx = 0
                            for s in range(current_rank):
                                if s == r0:
                                    continue
                                new_W3[:, s_idx] *= gamma[s_idx]
                                s_idx += 1
                            
                            if _verify_decomposition(new_U3, new_V3, new_W3):
                                if _check_half_integer(new_U3, new_V3, new_W3):
                                    U, V, W = new_U3, new_V3, new_W3
                                    found_removal = True
                                    break
            # End if residuals check
        # End for r0
        
        if not found_removal:
            break
    
    return U, V, W


def _schoolbook():
    """Schoolbook decomposition, rank = n*m*p = 60."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1.0
                V[j * p + k][c] = 1.0
                W[k * n + i][c] = 1.0
                c += 1
    return {"U": U, "V": V, "W": W}


def construct():
    t0 = time.time()
    deadline = t0 + 120
    
    # Build initial rank-54 decomposition
    try:
        U, V, W = _build_decomposition()
        
        if not _verify_decomposition(U, V, W):
            return _schoolbook()
        
        if not _check_half_integer(U, V, W):
            return _schoolbook()
        
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = U.shape[1]
        
        # Try rank reduction
        time_left = deadline - time.time()
        if time_left > 10:
            U_red, V_red, W_red = _try_reduce_rank(
                U, V, W, 
                max_iter=100,
                time_budget=min(time_left - 5, 100.0),
                start_time=t0
            )
            
            if U_red.shape[1] < best_rank:
                if _verify_decomposition(U_red, V_red, W_red):
                    if _check_half_integer(U_red, V_red, W_red):
                        best_U, best_V, best_W = U_red, V_red, W_red
                        best_rank = U_red.shape[1]
        
        if time.time() < deadline:
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
        
    except Exception:
        pass
    
    # Fallback
    return _schoolbook()


# EVOLVE-BLOCK-END
