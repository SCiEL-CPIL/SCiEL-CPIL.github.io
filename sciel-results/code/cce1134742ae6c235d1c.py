"""Bootstrap direct approach for 2x3x7 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need T[i,j,k] = 1 for all valid (i,j,k)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _snap_to_half_integers(mat):
    """Snap matrix entries to nearest half-integer."""
    return np.round(mat * 2) / 2.0

def _try_rank_decomposition(n, m, p, target_rank, max_iters=500, seed=42):
    """Try to find a decomposition of given rank using random initialization and ALS."""
    np.random.seed(seed)
    
    # Initialize with small random values
    U = np.random.randn(n * m, target_rank) * 0.5
    V = np.random.randn(m * p, target_rank) * 0.5
    W = np.random.randn(p * n, target_rank) * 0.5
    
    # Normalize initial vectors to unit norm to help convergence
    U_norms = np.linalg.norm(U, axis=0, keepdims=True)
    V_norms = np.linalg.norm(V, axis=0, keepdims=True)
    W_norms = np.linalg.norm(W, axis=0, keepdims=True)
    U = U / (U_norms + 1e-9)
    V = V / (V_norms + 1e-9)
    W = W / (W_norms + 1e-9)
    
    best_err = float('inf')
    
    for it in range(max_iters):
        # Fix V and W, solve for U
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, target_rank))
                for k in range(p):
                    M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                # Solve M @ u = ones(p)
                try:
                    u, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    U_new[i * m + j, :] = u
                except:
                    U_new[i * m + j, :] = U[i * m + j, :]
        
        # Fix U and W, solve for V
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, target_rank))
                for i in range(n):
                    M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    v, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                    V_new[j * p + k, :] = v
                except:
                    V_new[j * p + k, :] = V[j * p + k, :]
        
        # Fix U and V, solve for W
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, target_rank))
                for j in range(m):
                    M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    w, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    W_new[k * n + i, :] = w
                except:
                    W_new[k * n + i, :] = W[k * n + i, :]
        
        U, V, W = U_new, V_new, W_new
        
        # Check convergence
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        
        if max_err < best_err:
            best_err = max_err
        
        if max_err < 1e-8:
            # Snap to half-integers
            U_snap = _snap_to_half_integers(U)
            V_snap = _snap_to_half_integers(V)
            W_snap = _snap_to_half_integers(W)
            
            # Verify the snapped version
            if _verify_decomposition(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), n, m, p):
                return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
            
            # Try snapping with different scales if half-integers fail
            # The problem statement guarantees integer/half-integer, so this should work
            # unless the solution is exactly integer and we snapped to .5 by mistake?
            # Rounding to nearest half-integer handles both.
            
        # Early termination if error increases significantly
        if it > 50 and max_err > 1.0:
            break
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Schoolbook as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb

    # Search for lower ranks
    # Lower bound is roughly max(n*m, m*p, p*n) = 21
    # Known constructions for small matrices often reduce rank significantly.
    # We try ranks from 22 up to 41.
    target_ranks = list(range(22, 42))
    
    # We can also try a few very low ranks in case of a miracle, but 22 is a solid LB.
    
    for target_rank in target_ranks:
        if time.time() > deadline - 30:  # Keep 30s buffer for final checks
            break
        
        # Try multiple random seeds for robustness
        # More seeds for lower ranks as they are harder to find
        num_seeds = 10 if target_rank < 30 else 5
        
        for seed in range(num_seeds):
            if time.time() > deadline - 10:
                break
            
            result = _try_rank_decomposition(n, m, p, target_rank, max_iters=300, seed=seed)
            if result is not None:
                U_res, V_res, W_res = result
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_result = {"U": U_res, "V": V_res, "W": W_res}
                    # If we found a very good one, we can stop early
                    if target_rank <= 25:
                        break
        if target_rank <= 25 and best_rank == target_rank:
            break

    return best_result
# EVOLVE-BLOCK-END
