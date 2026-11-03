"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i."""
    nm, mp, pn = n*m, m*p, p*n
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T


def _compute_error(U, V, W, T, tol=1e-10):
    """Compute max absolute error of the decomposition."""
    R = U.shape[1]
    T_dec = np.zeros_like(T)
    for r in range(R):
        T_dec += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.max(np.abs(T - T_dec))


def _als_decompose(T, nm, mp, pn, target_rank, max_iter=100, tol=1e-10, seed=None,
                   U_init=None, V_init=None, W_init=None):
    """Alternating least squares to find a rank-`target_rank` decomposition of T.
    Uses proper matrix reshaping for efficiency."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    if U_init is not None:
        U = U_init.astype(np.float64).copy()
    else:
        U = rng.randn(nm, target_rank) * 0.1
    
    if V_init is not None:
        V = V_init.astype(np.float64).copy()
    else:
        V = rng.randn(mp, target_rank) * 0.1
    
    if W_init is not None:
        W = W_init.astype(np.float64).copy()
    else:
        W = rng.randn(pn, target_rank) * 0.1
    
    # Precompute flattened tensors
    T_a = T.reshape(nm, mp*pn)       # T[a, b, c] -> [a, (b,c)]
    T_b = np.transpose(T, (1, 0, 2)).reshape(mp, nm*pn)  # T[b, a, c] -> [b, (a,c)]
    T_c = np.transpose(T, (2, 0, 1)).reshape(pn, nm*mp)  # T[c, a, b] -> [c, (a,b)]
    
    prev_err = np.inf
    for it in range(max_iter):
        # Update U: minimize ||T_a - U @ (V ⊙ W)_rows||
        # M_U[r, (b,c)] = V[b,r] * W[c,r]
        M_U = np.einsum('br,cr->bcr', V, W).reshape(target_rank, mp*pn)
        try:
            U_new = T_a @ np.linalg.pinv(M_U)
        except np.linalg.LinAlgError:
            break
        U = U_new
        
        # Update V: minimize ||T_b - V @ (U ⊙ W)_rows||
        M_V = np.einsum('ar,cr->acr', U, W).reshape(target_rank, nm*pn)
        try:
            V_new = T_b @ np.linalg.pinv(M_V)
        except np.linalg.LinAlgError:
            break
        V = V_new
        
        # Update W: minimize ||T_c - W @ (U ⊙ V)_rows||
        M_W = np.einsum('ar,br->abr', U, V).reshape(target_rank, nm*mp)
        try:
            W_new = T_c @ np.linalg.pinv(M_W)
        except np.linalg.LinAlgError:
            break
        W = W_new
        
        # Check convergence
        err = _compute_error(U, V, W, T)
        if abs(prev_err - err) < tol * max(1.0, err):
            break
        prev_err = err
    
    err = _compute_error(U, V, W, T)
    
    # Try to snap to half-integers if close
    if err < 1e-6:
        for scale in [2, 4, 8]:
            U_snap = np.round(U * scale) / scale
            V_snap = np.round(V * scale) / scale
            W_snap = np.round(W * scale) / scale
            err_snap = _compute_error(U_snap, V_snap, W_snap, T)
            if err_snap < tol:
                return U_snap, V_snap, W_snap, err_snap
    
    return U, V, W, err


def _strassen_block_init(n, m, p):
    """Build a Strassen-based structured initialization for the tensor decomposition.
    Uses 2x2 block Strassen where applicable, schoolbook for remainder."""
    nm, mp, pn = n*m, m*p, p*n
    
    # For 4x4x7: split A into 2x2 blocks, B into 2x2 blocks for first 4 cols + 2x1 for last 3
    # We'll use a hybrid approach: Strassen for the square parts, schoolbook for the rest
    
    # Start with schoolbook as base
    R = n * m * p
    U = np.zeros((nm, R))
    V = np.zeros((mp, R))
    W = np.zeros((pn, R))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    # Build target tensor
    T = _build_target_tensor(n, m, p)
    
    # Schoolbook baseline
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1.0
                V_school[j*p + k, col] = 1.0
                W_school[k*n + i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Known lower bounds: rank(4x4x4) = 21 (Strassen), so 4x4x7 >= ~30
    # Upper bound from Strassen blocks: ~100
    # Search range: try from 28 upward
    
    # Use binary search to find the minimum rank
    lo, hi = 28, rank_school
    found_rank = None
    
    # First, verify schoolbook works
    err_school = _compute_error(U_school, V_school, W_school, T)
    assert err_school < 1e-10, f"Schoolbook error too large: {err_school}"
    
    # Try a set of candidate ranks with multiple restarts
    # Focus on the range where we expect the answer
    candidate_ranks = list(range(28, min(50, rank_school + 1)))
    
    for target_rank in candidate_ranks:
        if time.time() > deadline - 15:
            break
        
        found = False
        n_restarts = 5 if target_rank <= 40 else 3
        
        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                break
            
            seed = 42 + restart * 1000 + target_rank
            U, V, W, err = _als_decompose(T, nm, mp, pn, target_rank,
                                           max_iter=60, tol=1e-10, seed=seed)
            if err < 1e-10:
                # Verify
                err_verify = _compute_error(U, V, W, T)
                if err_verify < 1e-10:
                    best_U, best_V, best_W = U, V, W
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            # Try to go lower
            if target_rank > 28:
                continue
            else:
                break
    
    # If we found a rank, try to polish with more iterations
    if best_rank < rank_school and time.time() < deadline - 30:
        for polish_iter in range(3):
            if time.time() > deadline - 20:
                break
            U, V, W, err = _als_decompose(T, nm, mp, pn, best_rank,
                                           max_iter=100, tol=1e-12,
                                           seed=999 + polish_iter,
                                           U_init=best_U, V_init=best_V, W_init=best_W)
            if err < 1e-10:
                best_U, best_V, best_W = U, V, W
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
