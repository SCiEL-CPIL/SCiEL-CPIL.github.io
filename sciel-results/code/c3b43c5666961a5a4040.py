"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_decomposition(n, m, p):
    """
    Builds a rank-81 decomposition for 4x4 x 4x6 matrix multiplication.
    Uses tensor product of Strassen 2x2x2 (rank 7) for the 4x4x4 part (rank 49)
    and Schoolbook for the 4x4x2 part (rank 32).
    """
    # 1. Define Strassen 2x2x2 terms as sparse dicts
    strassen_terms_2x2 = [
        ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
        ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(1, 0): 1, (1, 1): -1}),
        ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(0, 1): 1, (1, 1): 1}),
        ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (1, 0): 1}),
        ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 0): -1, (0, 1): 1}),
        ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}),
        ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1}),
    ]
    
    rank_terms = []
    
    # 2. Generate 49 terms for 4x4x4 part via tensor product
    for s in range(7):
        u_s, v_s, w_s = strassen_terms_2x2[s]
        for t in range(7):
            u_t, v_t, w_t = strassen_terms_2x2[t]
            
            U_global = {}
            V_global = {}
            W_global = {}
            
            for (i1, j1), cu in u_s.items():
                for (i2, j2), cu2 in u_t.items():
                    gi = 2 * i1 + i2
                    gj = 2 * j1 + j2
                    coeff = cu * cu2
                    key = (gi, gj)
                    U_global[key] = U_global.get(key, 0) + coeff
                    
            for (j1, k1), cv in v_s.items():
                for (j2, k2), cv2 in v_t.items():
                    gj = 2 * j1 + j2
                    gk = 2 * k1 + k2
                    coeff = cv * cv2
                    key = (gj, gk)
                    V_global[key] = V_global.get(key, 0) + coeff
                    
            for (i1, k1), cw in w_s.items():
                for (i2, k2), cw2 in w_t.items():
                    gi = 2 * i1 + i2
                    gk = 2 * k1 + k2
                    coeff = cw * cw2
                    key = (gk, gi)
                    W_global[key] = W_global.get(key, 0) + coeff
                    
            rank_terms.append((U_global, V_global, W_global))
            
    # 3. Add 32 terms for 4x4x2 part (Schoolbook)
    for i in range(4):
        for j in range(4):
            for k in range(2):
                gk = 4 + k
                U_global = {(i, j): 1}
                V_global = {(j, gk): 1}
                W_global = {(gk, i): 1}
                rank_terms.append((U_global, V_global, W_global))
                
    rank = len(rank_terms)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for r, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j, r] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k, r] = coeff
        for (k, i), coeff in w_dict.items():
            W[k * n + i, r] = coeff
            
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify the decomposition is correct."""
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for r in range(U.shape[1]):
        alpha_r = np.sum(A_test * U[:, r].reshape(n, m))
        beta_r = np.sum(B_test * V[:, r].reshape(m, p))
        C_computed += alpha_r * beta_r * W[:, r].reshape(p, n).T
        
    return np.allclose(C_computed, C_expected, atol=1e-10)

def _remove_redundant(U, V, W, n, m, p, tol=1e-10):
    """
    Try to remove redundant rank terms by checking linear dependencies.
    Returns (U_reduced, V_reduced, W_reduced) with potentially lower rank.
    """
    rank = U.shape[1]
    if rank <= 1:
        return U, V, W
    
    # Check for linearly dependent columns
    # Use SVD to find rank
    U_svd, _, _ = np.linalg.svd(U, full_matrices=False)
    V_svd, _, _ = np.linalg.svd(V, full_matrices=False)
    W_svd, _, _ = np.linalg.svd(W, full_matrices=False)
    
    # Find tolerance for numerical rank
    U_rank = np.sum(U_svd > tol * U_svd[0])
    V_rank = np.sum(V_svd > tol * V_svd[0])
    W_rank = np.sum(W_svd > tol * W_svd[0])
    
    # If any matrix has lower numerical rank, we might be able to reduce
    # But this is complex; for now, just return as is if no clear redundancy
    # A more sophisticated approach would identify specific redundant terms
    
    return U, V, W

def _try_rank_reduction(U, V, W, n, m, p, tol=1e-8):
    """
    Attempt to reduce rank by identifying and removing linearly dependent terms.
    This is a heuristic approach that checks if removing a term still maintains correctness.
    """
    rank = U.shape[1]
    if rank <= 1:
        return U, V, W
    
    # Try to find terms that can be removed
    # We'll do a greedy approach: try removing each term and check if decomposition still holds
    # This is expensive, so we'll limit the number of attempts
    
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    
    # Try to remove terms one by one
    # Start from the end (schoolbook terms) as they're more likely to be redundant
    for r in range(rank - 1, -1, -1):
        # Skip if we've already tried this
        if current_U.shape[1] <= 1:
            break
            
        # Try removing term r
        mask = np.ones(current_U.shape[1], dtype=bool)
        mask[r] = False
        
        U_test = current_U[:, mask]
        V_test = current_V[:, mask]
        W_test = current_W[:, mask]
        
        # Quick check: verify with a small random test
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((n, p))
        for r_idx in range(U_test.shape[1]):
            alpha_r = np.sum(A_test * U_test[:, r_idx].reshape(n, m))
            beta_r = np.sum(B_test * V_test[:, r_idx].reshape(m, p))
            C_computed += alpha_r * beta_r * W_test[:, r_idx].reshape(p, n).T
            
        if np.allclose(C_computed, C_expected, atol=1e-8):
            # Term r is redundant, remove it
            current_U = U_test
            current_V = V_test
            current_W = W_test
    
    return current_U, current_V, current_W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    deadline = start_time + 140.0 # Safety margin for 150s limit
    
    # Build decomposition
    U, V, W = _build_decomposition(n, m, p)
    
    # Verify
    if not _verify(U, V, W, n, m, p):
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
        
    # Try to reduce rank
    if time.time() < deadline - 10:
        U, V, W = _try_rank_reduction(U, V, W, n, m, p)
        
        # Verify after reduction
        if not _verify(U, V, W, n, m, p):
            # Reduction failed, revert to original
            U, V, W = _build_decomposition(n, m, p)
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
