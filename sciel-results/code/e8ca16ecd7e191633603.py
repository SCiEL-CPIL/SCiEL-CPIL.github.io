import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 4, 7]

def _build_trivial(n, m, p):
    """Construct the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U[a_idx, r] = 1
                V[b_idx, r] = 1
                W[c_idx, r] = 1
                r += 1
    return U, V, W

def _build_strassen_block(n, m, p, block_size):
    """
    Build a block decomposition using Strassen-like algorithm for 2x2 sub-blocks.
    This works when m and p are even, allowing us to split into 2x2 blocks of size block_size.
    
    For 2x4 by 4x7: we can split the 4x4 multiplication part into Strassen.
    But 7 is odd, so we need to be careful.
    
    Let me think about this differently. For 2x4 * 4x7:
    A is 2x4, B is 4x7, C is 2x7.
    
    We can view this as multiplying a 2x4 matrix by a 4x7 matrix.
    One approach: use the fact that 4 = 2*2, so we can use Strassen for the 4x4 part.
    
    Actually, let me try a different structural approach:
    Use the identity that for small matrices, we can find explicit low-rank decompositions.
    
    Let me try to construct a rank-49 or lower decomposition by combining known results.
    """
    # This is a placeholder - the actual construction is complex.
    # Let's fall back to trivial for now and use optimization.
    return _build_trivial(n, m, p)

def _tensor_target(n, m, p):
    """Build the target multiplication tensor."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _als_decompose(n, m, p, R, n_restarts=3, time_limit=8.0, seed=0):
    """
    Alternating Least Squares for tensor decomposition.
    Returns (U, V, W, is_valid).
    """
    A_dim = n * m
    B_dim = m * p
    C_dim = p * n
    T = _tensor_target(n, m, p)
    T_vec = T.flatten()
    
    T_mat_a = T.reshape(A_dim, B_dim * C_dim)
    T_mat_b = T.reshape(B_dim, A_dim * C_dim)
    T_mat_c = T.reshape(C_dim, A_dim * B_dim)
    
    start_time = time.time()
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
        
        rng = np.random.RandomState(seed + restart * 1000)
        
        # Initialize with random values, but scale appropriately
        U = rng.randn(A_dim, R) * 0.5
        V = rng.randn(B_dim, R) * 0.5
        W = rng.randn(C_dim, R) * 0.5
        
        best_residual = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(200):
            if time.time() - start_time > time_limit:
                break
            
            # Solve for U
            M_a = np.zeros((R, B_dim * C_dim))
            for r in range(R):
                M_a[r, :] = np.outer(V[:, r], W[:, r]).flatten()
            try:
                U_new = np.linalg.lstsq(M_a, T_mat_a.T, rcond=None)[0].T
            except:
                break
            U = U_new
            
            # Solve for V
            M_b = np.zeros((R, A_dim * C_dim))
            for r in range(R):
                M_b[r, :] = np.outer(U[:, r], W[:, r]).flatten()
            try:
                V_new = np.linalg.lstsq(M_b, T_mat_b.T, rcond=None)[0].T
            except:
                break
            V = V_new
            
            # Solve for W
            M_c = np.zeros((R, A_dim * B_dim))
            for r in range(R):
                M_c[r, :] = np.outer(U[:, r], V[:, r]).flatten()
            try:
                W_new = np.linalg.lstsq(M_c, T_mat_c.T, rcond=None)[0].T
            except:
                break
            W = W_new
            
            # Compute residual
            T_comp = np.einsum('ar,br,cr->abc', U, V, W)
            residual = np.linalg.norm(T_comp - T)
            if residual < best_residual:
                best_residual = residual
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if residual < 1e-10:
                break
        
        if best_residual < 1e-9:
            return best_U, best_V, best_W, True
    
    return best_U, best_V, best_W, False

def _verify_decomposition(U, V, W, n, m, p, atol=1e-9):
    """Verify that the decomposition exactly matches the target tensor."""
    T_comp = np.einsum('ar,br,cr->abc', U, V, W)
    T_target = _tensor_target(n, m, p)
    return np.allclose(T_comp, T_target, atol=atol)

def _try_rank(n, m, p, R, time_budget=10.0):
    """Try to find a valid decomposition of rank R."""
    start = time.time()
    
    # Try multiple restarts
    for restart in range(5):
        if time.time() - start > time_budget:
            break
        
        U, V, W, is_valid = _als_decompose(n, m, p, R, n_restarts=1, 
                                            time_limit=min(5.0, time_budget - (time.time() - start)),
                                            seed=restart * 7 + 123)
        
        if is_valid:
            return U, V, W, True
    
    return None, None, None, False

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin from 150s limit
    
    n, m, p = DIMENSIONS
    
    # Get trivial solution as fallback
    U_triv, V_triv, W_triv = _build_trivial(n, m, p)
    best_U, best_V, best_W = U_triv, V_triv, W_triv
    best_rank = U_triv.shape[1]
    
    # The trivial rank is 56. We want to find something lower.
    # Known results suggest rank around 49-50 might be achievable.
    # Let's try ranks from 49 down to 40, with more time for higher ranks.
    
    # Strategy: try ranks in order of likelihood of success
    # Start with 49 (known to be close to optimal for some similar problems)
    # Then 48, 47, etc.
    
    ranks_to_try = [49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28]
    
    for R in ranks_to_try:
        if time.time() - start_time > deadline - 5:
            break
        
        # Allocate time based on rank: lower ranks get less time
        # Higher ranks (closer to trivial) are more likely to succeed
        remaining = deadline - 5 - (time.time() - start_time)
        if remaining < 2:
            break
            
        # Give more time to higher ranks (more likely to work)
        time_for_rank = min(8.0, max(2.0, remaining / len(ranks_to_try)))
        
        U, V, W, found = _try_rank(n, m, p, R, time_budget=time_for_rank)
        
        if found and _verify_decomposition(U, V, W, n, m, p):
            best_U, best_V, best_W = U, V, W
            best_rank = R
            break
    
    # If we didn't find anything, try a few more ranks with whatever time is left
    if best_rank == 56:
        remaining = deadline - 3 - (time.time() - start_time)
        if remaining > 3:
            for R in [50, 51, 52, 53, 54, 55]:
                if time.time() - start_time > deadline - 3:
                    break
                U, V, W, found = _try_rank(n, m, p, R, time_budget=min(3.0, remaining))
                if found and _verify_decomposition(U, V, W, n, m, p):
                    best_U, best_V, best_W = U, V, W
                    best_rank = R
                    break
    
    # Final safety check
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p):
        best_U, best_V, best_W = U_triv, V_triv, W_triv
        best_rank = 56
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    T = np.einsum('ar,br,cr->abc', U, V, W)
    Expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Expected[i*m + j, j*p + k, k*n + i] = 1
    
    if np.allclose(T, Expected, atol=1e-9):
        print(f"Valid decomposition with rank {U.shape[1]}")
    else:
        print("Invalid decomposition")
        diff = np.abs(T - Expected)
        print(f"Max diff: {diff.max()}")
