"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy import linalg

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n=2, m=4, p=4):
    """Verify the decomposition. Returns max error."""
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err


def _build_strassen_28():
    """Build rank-28 block Strassen decomposition for 2x4 * 4x4."""
    n, m, p = 2, 4, 4
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen terms for 2x2 * 2x2 -> 2x2
    # Each term: (U_coeffs, V_coeffs, W_coeffs)
    # Local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen_terms = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        ([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]),
        ([(0,1)], [(2,1),(3,-1)], [(1,1),(3,1)]),
        ([(3,1)], [(1,1),(2,-1)], [(0,1),(2,1)]),
        ([(0,1),(2,1)], [(3,1)], [(0,-1),(1,1)]),
        ([(1,1),(2,-1)], [(0,1),(2,1)], [(0,1),(1,-1),(2,1),(3,-1)]),
        ([(1,1),(3,-1)], [(1,1),(3,1)], [(0,1),(1,1),(2,-1),(3,-1)]),
    ]
    
    def map_u(a_row, a_col):
        return a_row * 4 + a_col
    
    def map_v(b_row, b_col):
        return b_row * 4 + b_col
    
    def map_w(c_col, c_row):
        return c_col * 2 + c_row
    
    def apply_block(a_col_offset, b_row_offset, b_col_offset, c_col_offset, start):
        for t in strassen_terms:
            u_c, v_c, w_c = t
            for (li, coeff) in u_c:
                lr, lc = li // 2, li % 2
                U[map_u(lr, a_col_offset + lc), start] += coeff
            for (li, coeff) in v_c:
                lr, lc = li // 2, li % 2
                V[map_v(b_row_offset + lr, b_col_offset + lc), start] += coeff
            for (li, coeff) in w_c:
                lr, lc = li // 2, li % 2
                W[map_w(c_col_offset + lc, lr), start] += coeff
            start += 1
        return start
    
    # Four 2x2 blocks:
    col = apply_block(0, 0, 0, 0, 0)
    col = apply_block(2, 2, 0, 0, col)
    col = apply_block(0, 0, 2, 2, col)
    col = apply_block(2, 2, 2, 2, col)
    
    return U, V, W


def _build_schoolbook():
    """Build the rank-32 schoolbook decomposition."""
    n, m, p = 2, 4, 4
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W


def _try_reduce_to_rank27(U, V, W):
    """Try to find a rank-27 decomposition by exploiting linear dependencies.
    
    The 28 rank-1 tensors from Strassen may have linear dependencies.
    If we find one, we can potentially eliminate one term.
    """
    n, m, p = 2, 4, 4
    rank = U.shape[1]
    N = n * m * p  # 32
    
    # Build tensor vectors for each rank-1 component
    T = np.zeros((rank, N))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[r, i*m*p + j*p + k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # SVD to find null space
    try:
        U_svd, S_svd, Vt_svd = np.linalg.svd(T, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    
    tol = 1e-10
    num_rank = np.sum(S_svd > tol)
    
    if num_rank >= rank:
        return None  # No dependency found
    
    # Try to use the dependency to reduce rank
    # The null space vectors give linear combinations that sum to zero
    # We can use this to perturb the decomposition
    
    # For each null vector, try to eliminate the term with largest coefficient
    for null_vec in Vt_svd[num_rank:]:  # All null space vectors
        r_elim = int(np.argmax(np.abs(null_vec)))
        if abs(null_vec[r_elim]) < 1e-15:
            continue
        
        alpha = null_vec / null_vec[r_elim]
        alpha[r_elim] = 1.0
        
        # Verify the dependency
        recon = np.zeros(N)
        for r in range(rank):
            if abs(alpha[r]) > 1e-15:
                recon += alpha[r] * T[r, :]
        
        if np.max(np.abs(recon)) > 1e-8:
            continue
        
        # Found a valid dependency. Now try to construct rank-27 decomposition.
        # This is non-trivial. The dependency tells us that term r_elim can be
        # expressed as a combination of other terms, but we need to find new
        # U', V', W' matrices of rank 27 that produce the same tensor.
        
        # One approach: use the dependency to create a new term that replaces
        # the eliminated one. But this requires solving a system of equations.
        
        # For now, just return None to indicate we found a dependency but
        # haven't implemented the reduction yet.
        return None
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Build rank-28 Strassen decomposition
    U, V, W = _build_strassen_28()
    
    # Verify
    err = _verify(U, V, W)
    
    if err < 1e-9:
        # Try to reduce to rank 27 (may not succeed)
        reduced = _try_reduce_to_rank27(U, V, W)
        if reduced is not None:
            U, V, W = reduced
            err2 = _verify(U, V, W)
            if err2 < 1e-9:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Return rank-28 as the safe answer
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        U, V, W = _build_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
