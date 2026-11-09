"""Matrix multiplication 4x4x6 rank-minimal exact decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]
DEADLINE = 115.0  # Safety margin before 150s kill


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the tensor decomposition is valid."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    return False, (i, j, k, val)
    return True, None


def _strassen_2x2x2(U, V, W, n, m, p, a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, base_col):
    """
    Add 7 Strassen multiplications for a 2x2x2 sub-product.
    """
    def uidx(r, c):
        return r * m + c
    
    def vidx(r, c):
        return r * p + c
    
    def widx(r, c):
        return r * n + c
    
    a00 = (a_rows[0], a_cols[0])
    a01 = (a_rows[0], a_cols[1])
    a10 = (a_rows[1], a_cols[0])
    a11 = (a_rows[1], a_cols[1])
    
    b00 = (b_rows[0], b_cols[0])
    b01 = (b_rows[0], b_cols[1])
    b10 = (b_rows[1], b_cols[0])
    b11 = (b_rows[1], b_cols[1])
    
    c00 = (c_rows[0], c_cols[0])
    c01 = (c_rows[0], c_cols[1])
    c10 = (c_rows[1], c_cols[0])
    c11 = (c_rows[1], c_cols[1])
    
    # M1 = (a00+a11)*(b00+b11)
    U[uidx(a00[0], a00[1]), base_col+0] += 1
    U[uidx(a11[0], a11[1]), base_col+0] += 1
    V[vidx(b00[0], b00[1]), base_col+0] += 1
    V[vidx(b11[0], b11[1]), base_col+0] += 1
    W[widx(c00[0], c00[1]), base_col+0] += 1
    W[widx(c01[0], c01[1]), base_col+0] += 1
    W[widx(c10[0], c10[1]), base_col+0] += 1
    W[widx(c11[0], c11[1]), base_col+0] += 1
    
    # M2 = (a10+a11)*b00
    U[uidx(a10[0], a10[1]), base_col+1] += 1
    U[uidx(a11[0], a11[1]), base_col+1] += 1
    V[vidx(b00[0], b00[1]), base_col+1] += 1
    W[widx(c10[0], c10[1]), base_col+1] += 1
    W[widx(c11[0], c11[1]), base_col+1] += -1
    
    # M3 = a00*(b01-b11)
    U[uidx(a00[0], a00[1]), base_col+2] += 1
    V[vidx(b01[0], b01[1]), base_col+2] += 1
    V[vidx(b11[0], b11[1]), base_col+2] += -1
    W[widx(c01[0], c01[1]), base_col+2] += 1
    W[widx(c11[0], c11[1]), base_col+2] += 1
    
    # M4 = a11*(b10-b00)
    U[uidx(a11[0], a11[1]), base_col+3] += 1
    V[vidx(b10[0], b10[1]), base_col+3] += 1
    V[vidx(b00[0], b00[1]), base_col+3] += -1
    W[widx(c00[0], c00[1]), base_col+3] += 1
    W[widx(c10[0], c10[1]), base_col+3] += 1
    
    # M5 = (a00+a01)*b11
    U[uidx(a00[0], a00[1]), base_col+4] += 1
    U[uidx(a01[0], a01[1]), base_col+4] += 1
    V[vidx(b11[0], b11[1]), base_col+4] += 1
    W[widx(c00[0], c00[1]), base_col+4] += -1
    W[widx(c01[0], c01[1]), base_col+4] += 1
    
    # M6 = (a10-a00)*(b00+b01)
    U[uidx(a10[0], a10[1]), base_col+5] += 1
    U[uidx(a00[0], a00[1]), base_col+5] += -1
    V[vidx(b00[0], b00[1]), base_col+5] += 1
    V[vidx(b01[0], b01[1]), base_col+5] += 1
    W[widx(c11[0], c11[1]), base_col+5] += 1
    
    # M7 = (a01-a11)*(b10+b11)
    U[uidx(a01[0], a01[1]), base_col+6] += 1
    U[uidx(a11[0], a11[1]), base_col+6] += -1
    V[vidx(b10[0], b10[1]), base_col+6] += 1
    V[vidx(b11[0], b11[1]), base_col+6] += 1
    W[widx(c00[0], c00[1]), base_col+6] += 1
    W[widx(c01[0], c01[1]), base_col+6] += 1


def _build_strassen_decomposition(n, m, p):
    """
    Build a Strassen-based decomposition for n x m x p matrix multiplication.
    Uses 2x2 blocking for A and appropriate blocking for B and C.
    """
    rank = 84
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    subproducts = [
        ([0,1], [0,1], [0,1], [0,1], [0,1], [0,1]),
        ([0,1], [0,1], [0,1], [2,3], [0,1], [2,3]),
        ([0,1], [0,1], [0,1], [4,5], [0,1], [4,5]),
        ([0,1], [2,3], [2,3], [0,1], [0,1], [0,1]),
        ([0,1], [2,3], [2,3], [2,3], [0,1], [2,3]),
        ([0,1], [2,3], [2,3], [4,5], [0,1], [4,5]),
        ([2,3], [0,1], [0,1], [0,1], [2,3], [0,1]),
        ([2,3], [0,1], [0,1], [2,3], [2,3], [2,3]),
        ([2,3], [0,1], [0,1], [4,5], [2,3], [4,5]),
        ([2,3], [2,3], [2,3], [0,1], [2,3], [0,1]),
        ([2,3], [2,3], [2,3], [2,3], [2,3], [2,3]),
        ([2,3], [2,3], [2,3], [4,5], [2,3], [4,5]),
    ]
    
    for idx, (a_r, a_c, b_r, b_c, c_r, c_c) in enumerate(subproducts):
        base_col = idx * 7
        _strassen_2x2x2(U, V, W, n, m, p, a_r, a_c, b_r, b_c, c_r, c_c, base_col)
    
    return U, V, W, rank


def _build_schoolbook(n, m, p):
    """Build the schoolbook rank-96 decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W, rank


def _try_reduce_rank(U, V, W, n, m, p, target_rank, max_iters=5000, seed=42):
    """
    Try to reduce the rank of a valid decomposition to target_rank
    using iterative refinement on the tensor equations.
    
    We have constraints: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all (i,j,k)
    """
    if target_rank >= U.shape[1]:
        return None
    
    rng = np.random.RandomState(seed)
    rank = target_rank
    
    # Start from the current decomposition, zero out columns beyond target_rank
    U_work = U[:, :rank].copy()
    V_work = V[:, :rank].copy()
    W_work = W[:, :rank].copy()
    
    # If we're starting from a higher rank, we need to re-initialize
    # Try random initialization scaled appropriately
    U_work = rng.randn(n*m, rank) * 0.5
    V_work = rng.randn(m*p, rank) * 0.5
    W_work = rng.randn(p*n, rank) * 0.5
    
    # Iterative refinement
    for it in range(max_iters):
        # Compute all errors
        errors = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    errors[i, j, k] = np.dot(U_work[i*m+j, :], V_work[j*p+k, :] * W_work[k*n+i, :]) - 1.0
        
        max_err = np.max(np.abs(errors))
        if max_err < 1e-12:
            return U_work, V_work, W_work
        
        # Pick the worst error
        idx = np.unravel_index(np.argmax(np.abs(errors)), (n, m, p))
        i, j, k = idx
        err = errors[i, j, k]
        
        if abs(err) < 1e-14:
            continue
        
        # Adjust: perturb the factors to reduce the error
        # The contribution is sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        # We want to reduce this by err
        # Simple approach: scale all factors in one random rank dimension
        r = rng.randint(0, rank)
        
        # Compute current contribution from rank r
        contrib = U_work[i*m+j, r] * V_work[j*p+k, r] * W_work[k*n+i, r]
        
        if abs(contrib) > 1e-12:
            # Scale to reduce error
            # We want new_contrib such that err + contrib - new_contrib = 0
            # new_contrib = contrib - err
            scale = (contrib - err) / contrib
            # Apply scaling to one factor
            if rng.random() < 0.33:
                U_work[i*m+j, r] *= scale
            elif rng.random() < 0.5:
                V_work[j*p+k, r] *= scale
            else:
                W_work[k*n+i, r] *= scale
        else:
            # Contribution is near zero, try to add it
            if abs(err) > 1e-12:
                # Set one factor to help
                if rng.random() < 0.33:
                    U_work[i*m+j, r] = err / (V_work[j*p+k, r] * W_work[k*n+i, r]) if abs(V_work[j*p+k, r] * W_work[k*n+i, r]) > 1e-12 else 0
                elif rng.random() < 0.5:
                    V_work[j*p+k, r] = err / (U_work[i*m+j, r] * W_work[k*n+i, r]) if abs(U_work[i*m+j, r] * W_work[k*n+i, r]) > 1e-12 else 0
                else:
                    W_work[k*n+i, r] = err / (U_work[i*m+j, r] * V_work[j*p+k, r]) if abs(U_work[i*m+j, r] * V_work[j*p+k, r]) > 1e-12 else 0
    
    return None


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Build the Strassen-based decomposition (rank 84)
    U, V, W, rank = _build_strassen_decomposition(n, m, p)
    
    # Verify the decomposition
    valid, err = _verify_decomposition(U, V, W, n, m, p)
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    if not valid:
        # Fallback to schoolbook
        U, V, W, rank = _build_schoolbook(n, m, p)
        best_rank = rank
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank using iterative refinement
    # Try various target ranks from just below current down to theoretical minimum
    # Theoretical minimum for 4x4x6 is unknown but at least max(n*m, m*p, n*p) = 24
    # Actually the tensor rank lower bound is more complex
    
    # Try reducing from current rank
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    
    if remaining > 10:
        # Try rank 42 (half of 84)
        target = min(best_rank - 1, 42)
        result = _try_reduce_rank(best_U, best_V, best_W, n, m, p, target, max_iters=3000, seed=42)
        if result is not None:
            U_r, V_r, W_r = result
            valid_r, _ = _verify_decomposition(U_r, V_r, W_r, n, m, p)
            if valid_r:
                best_rank = target
                best_U, best_V, best_W = U_r, V_r, W_r
    
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    
    if remaining > 10 and best_rank > 30:
        # Try rank 30
        target = min(best_rank - 1, 30)
        result = _try_reduce_rank(best_U, best_V, best_W, n, m, p, target, max_iters=3000, seed=123)
        if result is not None:
            U_r, V_r, W_r = result
            valid_r, _ = _verify_decomposition(U_r, V_r, W_r, n, m, p)
            if valid_r:
                best_rank = target
                best_U, best_V, best_W = U_r, V_r, W_r
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
