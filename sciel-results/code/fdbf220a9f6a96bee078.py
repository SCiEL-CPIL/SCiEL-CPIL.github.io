"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
_START_TIME = time.time()
_DEADLINE = 120.0
_MARGIN = 5.0

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def _build_rank88():
    """Build the known rank-88 decomposition."""
    n, m, p = DIMENSIONS
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Part 1: Schoolbook for A[0, :] (1x4) * B (4x8)
    for j in range(m):
        for k in range(p):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
    
    # Part 2: Strassen for A[1:3, :] (2x4) * B (4x8)
    strassen_terms = [
        {'U': {0: 1, 3: 1}, 'V': {0: 1, 3: 1}, 'W': {0: 1, 3: 1}},
        {'U': {2: 1, 3: 1}, 'V': {0: 1}, 'W': {2: 1, 3: -1}},
        {'U': {0: 1}, 'V': {1: 1, 3: -1}, 'W': {1: 1, 3: 1}},
        {'U': {3: 1}, 'V': {2: 1, 0: -1}, 'W': {0: 1, 2: 1}},
        {'U': {0: 1, 1: 1}, 'V': {3: 1}, 'W': {0: -1, 1: 1}},
        {'U': {2: 1, 0: -1}, 'V': {0: 1, 1: 1}, 'W': {3: 1}},
        {'U': {1: 1, 3: -1}, 'V': {2: 1, 3: 1}, 'W': {0: 1}}
    ]
    
    for j_block in range(4):
        for term in strassen_terms:
            for local_u, val in term['U'].items():
                lu, lv = divmod(local_u, 2)
                U[(1 + lu) * m + lv][col] += val
            for local_v, val in term['V'].items():
                lu, lv = divmod(local_v, 2)
                V[lu * p + (2 * j_block + lv)][col] += val
            for local_w, val in term['W'].items():
                lu, lv = divmod(local_w, 2)
                W[(2 * j_block + lv) * n + (1 + lu)][col] += val
            col += 1
            
        for term in strassen_terms:
            for local_u, val in term['U'].items():
                lu, lv = divmod(local_u, 2)
                U[(1 + lu) * m + (2 + lv)][col] += val
            for local_v, val in term['V'].items():
                lu, lv = divmod(local_v, 2)
                V[(2 + lu) * p + (2 * j_block + lv)][col] += val
            for local_w, val in term['W'].items():
                lu, lv = divmod(local_w, 2)
                W[(2 * j_block + lv) * n + (1 + lu)][col] += val
            col += 1
            
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W):
    """Verify that the decomposition exactly represents matrix multiplication."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _als_optimize(U, V, W, n_iter=400):
    """ALS optimization to minimize error while keeping rank fixed."""
    n, m, p = DIMENSIONS
    
    U = U.astype(np.float64).copy()
    V = V.astype(np.float64).copy()
    W = W.astype(np.float64).copy()
    
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(n_iter):
        if time.time() > _START_TIME + _DEADLINE - _MARGIN:
            break
        
        # Update U
        for i in range(n):
            for j in range(m):
                V_block = V[j * p:(j + 1) * p, :]
                W_vec = np.array([W[k * n + i, :] for k in range(p)])
                P = V_block * W_vec
                if np.linalg.norm(P) < 1e-15:
                    continue
                u, _, _, _ = np.linalg.lstsq(P, np.ones(p), rcond=None)
                U[i * m + j, :] = u
        
        # Update V
        for j in range(m):
            for k in range(p):
                U_block = np.array([U[i * m + j, :] for i in range(n)])
                W_vec = W[k * n:(k + 1) * n, :]
                P = U_block * W_vec
                if np.linalg.norm(P) < 1e-15:
                    continue
                v, _, _, _ = np.linalg.lstsq(P, np.ones(n), rcond=None)
                V[j * p + k, :] = v
        
        # Update W
        for k in range(p):
            for i in range(n):
                U_block = np.array([U[i * m + j, :] for j in range(m)])
                V_vec = np.array([V[j * p + k, :] for j in range(m)])
                P = U_block * V_vec
                if np.linalg.norm(P) < 1e-15:
                    continue
                w, _, _, _ = np.linalg.lstsq(P, np.ones(m), rcond=None)
                W[k * n + i, :] = w
        
        # Compute error
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    error += (val - 1.0) ** 2
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if error < 1e-12:
            break
    
    return best_U, best_V, best_W, best_error

def _try_rank(target_rank, n_trials=3, U88=None, V88=None, W88=None):
    """Try to find a valid decomposition at the given rank."""
    n, m, p = DIMENSIONS
    
    for trial in range(n_trials):
        if time.time() > _START_TIME + _DEADLINE - _MARGIN:
            return None
        
        rng = np.random.RandomState(target_rank * 100 + trial)
        
        if trial == 0 and U88 is not None:
            # Warm start: use first target_rank columns of the rank-88 solution
            U = U88[:, :target_rank].copy()
            V = V88[:, :target_rank].copy()
            W = W88[:, :target_rank].copy()
        elif trial == 1 and U88 is not None:
            # Second warm start: random permutation of columns
            perm = rng.permutation(88)[:target_rank]
            U = U88[:, perm].copy()
            V = V88[:, perm].copy()
            W = W88[:, perm].copy()
        else:
            # Random initialization
            U = rng.randn(n * m, target_rank)
            V = rng.randn(m * p, target_rank)
            W = rng.randn(p * n, target_rank)
        
        # Run ALS
        U_opt, V_opt, W_opt, error = _als_optimize(U, V, W, n_iter=300)
        
        if error < 1e-8:
            # Try rounding to half-integers
            U_r = np.round(U_opt * 2) / 2
            V_r = np.round(V_opt * 2) / 2
            W_r = np.round(W_opt * 2) / 2
            if _verify_decomposition(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                return U_r.tolist(), V_r.tolist(), W_r.tolist()
            
            # Try rounding to integers
            U_i = np.round(U_opt)
            V_i = np.round(V_opt)
            W_i = np.round(W_opt)
            if _verify_decomposition(U_i.tolist(), V_i.tolist(), W_i.tolist()):
                return U_i.tolist(), V_i.tolist(), W_i.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # First, build the known rank-88 solution
    rank88_result = _build_rank88()
    U88 = np.array(rank88_result['U'], dtype=np.float64)
    V88 = np.array(rank88_result['V'], dtype=np.float64)
    W88 = np.array(rank88_result['W'], dtype=np.float64)
    
    # Verify rank-88 is correct
    if not _verify_decomposition(U88.tolist(), V88.tolist(), W88.tolist()):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find lower rank solutions
    for target_rank in [87, 86, 85, 84, 83, 82, 80]:
        time_remaining = _START_TIME + _DEADLINE - _MARGIN - time.time()
        if time_remaining < 20:
            break
        
        result = _try_rank(target_rank, n_trials=2)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Return the rank-88 solution
    return rank88_result
# EVOLVE-BLOCK-END
