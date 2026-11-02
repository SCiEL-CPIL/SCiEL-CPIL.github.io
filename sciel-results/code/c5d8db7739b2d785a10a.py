"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_strassen_decomposition(n, m, p):
    """Build block-based Strassen decomposition for n×m × m×p.
    Splits into 2×2 blocks and applies Strassen (rank 7) to each subproduct.
    Returns (U, V, W) as numpy arrays."""
    rank_terms = []
    
    for i1 in range(n // 2):
        for j1 in range(m // 2):
            for k1 in range(p // 2):
                terms = [
                    # M1: (X00+X11)(Y00+Y11) -> Z00(+1), Z11(+1)
                    ({(i1*2, j1*2): 1, (i1*2+1, j1*2+1): 1},
                     {(j1*2, k1*2): 1, (j1*2+1, k1*2+1): 1},
                     {(i1*2, k1*2): 1, (i1*2+1, k1*2+1): 1}),
                    # M2: (X10+X11)Y00 -> Z10(+1), Z11(-1)
                    ({(i1*2+1, j1*2): 1, (i1*2+1, j1*2+1): 1},
                     {(j1*2, k1*2): 1},
                     {(i1*2+1, k1*2): 1, (i1*2+1, k1*2+1): -1}),
                    # M3: X00(Y01-Y11) -> Z01(+1), Z11(+1)
                    ({(i1*2, j1*2): 1},
                     {(j1*2, k1*2+1): 1, (j1*2+1, k1*2+1): -1},
                     {(i1*2, k1*2+1): 1, (i1*2+1, k1*2+1): 1}),
                    # M4: X11(Y10-Y00) -> Z00(+1), Z10(+1)
                    ({(i1*2+1, j1*2+1): 1},
                     {(j1*2+1, k1*2): 1, (j1*2, k1*2): -1},
                     {(i1*2, k1*2): 1, (i1*2+1, k1*2): 1}),
                    # M5: (X00+X01)Y11 -> Z00(-1), Z01(+1)
                    ({(i1*2, j1*2): 1, (i1*2, j1*2+1): 1},
                     {(j1*2+1, k1*2+1): 1},
                     {(i1*2, k1*2): -1, (i1*2, k1*2+1): 1}),
                    # M6: (X10-X00)(Y00+Y01) -> Z11(+1)
                    ({(i1*2+1, j1*2): 1, (i1*2, j1*2): -1},
                     {(j1*2, k1*2): 1, (j1*2, k1*2+1): 1},
                     {(i1*2+1, k1*2+1): 1}),
                    # M7: (X01-X11)(Y10+Y11) -> Z00(+1)
                    ({(i1*2, j1*2+1): 1, (i1*2+1, j1*2+1): -1},
                     {(j1*2+1, k1*2): 1, (j1*2+1, k1*2+1): 1},
                     {(i1*2, k1*2): 1}),
                ]
                for u_dict, v_dict, w_dict in terms:
                    rank_terms.append((u_dict, v_dict, w_dict))
    
    rank = len(rank_terms)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for r, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j, r] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k, r] = coeff
        for (i, k), coeff in w_dict.items():
            W[k * n + i, r] = coeff
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, seed=42):
    """Verify the decomposition is correct."""
    np.random.seed(seed)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for r in range(U.shape[1]):
        u_vec = U[:, r].reshape(n, m)
        v_vec = V[:, r].reshape(m, p)
        w_vec = W[:, r].reshape(p, n)
        C_computed += np.einsum('ij,jk,ki->ik', u_vec, v_vec, w_vec)
    
    return np.allclose(C_computed, C_expected, atol=1e-10)

def _remove_redundant_terms(U, V, W, n, m, p, tol=1e-9):
    """Remove linearly dependent rank-1 terms from the decomposition.
    Two terms r1 and r2 are redundant if the tensor T_r1 is a scalar multiple of T_r2,
    i.e., U[:,r1] ⊗ V[:,r1] ⊗ W[:,r1] = c * (U[:,r2] ⊗ V[:,r2] ⊗ W[:,r2]).
    We check this by seeing if the outer product matrices are proportional."""
    rank = U.shape[1]
    keep = list(range(rank))
    
    for i in range(len(keep)):
        if i >= len(keep):
            break
        ri = keep[i]
        for j in range(i + 1, len(keep)):
            rj = keep[j]
            # Check if term rj is a scalar multiple of term ri
            # Compare U vectors
            u_i = U[:, ri]
            u_j = U[:, rj]
            v_i = V[:, ri]
            v_j = V[:, rj]
            w_i = W[:, ri]
            w_j = W[:, rj]
            
            # Find scaling factor from U vectors
            mask_u = np.abs(u_i) > tol
            if not np.any(mask_u):
                # u_i is zero, check if u_j is also zero
                if np.allclose(u_j, 0, atol=tol):
                    # Both U vectors are zero, check V and W
                    mask_v = np.abs(v_i) > tol
                    if not np.any(mask_v):
                        if np.allclose(v_j, 0, atol=tol) and np.allclose(w_j, 0, atol=tol):
                            keep.remove(rj)
                            j -= 1
                        continue
                    # v_i is zero but v_j is not -> not redundant
                    continue
                else:
                    continue
            
            c_u = u_j[mask_u] / u_i[mask_u]
            # Check consistency
            if not np.allclose(c_u, c_u[0], rtol=1e-7, atol=tol):
                continue
            c = c_u[0]
            
            # Check V vectors
            if not np.allclose(v_j, c * v_i, rtol=1e-7, atol=tol):
                continue
            # Check W vectors
            if not np.allclose(w_j, c * w_i, rtol=1e-7, atol=tol):
                continue
            
            # Term rj is redundant
            keep.remove(rj)
            j -= 1
    
    if len(keep) < rank:
        U_new = U[:, keep]
        V_new = V[:, keep]
        W_new = W[:, keep]
        return U_new, V_new, W_new, True
    return U, V, W, False

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Build the block-based Strassen decomposition
    U, V, W = _build_strassen_decomposition(n, m, p)
    
    # Verify correctness
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to remove redundant terms
    U, V, W, reduced = _remove_redundant_terms(U, V, W, n, m, p)
    
    # Final verification
    if not _verify_decomposition(U, V, W, n, m, p):
        # If reduction broke it, use the original
        U, V, W = _build_strassen_decomposition(n, m, p)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        # Shouldn't happen, but safety check
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
