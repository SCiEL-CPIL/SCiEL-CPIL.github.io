"""Matrix multiplication 2,3,9 tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 130  # safety margin before 150s limit
    n, m, p = DIMENSIONS  # 2, 3, 9
    
    # Trivial decomposition: rank = n*m*p = 54
    base_rank = n * m * p
    best_rank = base_rank
    best_U = [[0] * base_rank for _ in range(n * m)]
    best_V = [[0] * base_rank for _ in range(m * p)]
    best_W = [[0] * base_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that the decomposition is exact."""
        rank = len(U[0]) if isinstance(U, list) else U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        if isinstance(U, list):
                            u_val = U[i*m + j][r]
                            v_val = V[j*p + k][r]
                            w_val = W[k*n + i][r]
                        else:
                            u_val = U[i*m + j, r]
                            v_val = V[j*p + k, r]
                            w_val = W[k*n + i, r]
                        total += u_val * v_val * w_val
                    if abs(total - 1.0) > tol:
                        return False
        return True
    
    def try_random_search(target_rank, n_trials=50, seed=42):
        """Try random search for a decomposition of given rank."""
        if target_rank >= best_rank:
            return None
        rng = np.random.RandomState(seed)
        
        for trial in range(n_trials):
            if time.time() > deadline:
                break
            
            U = np.zeros((n * m, target_rank), dtype=float)
            V = np.zeros((m * p, target_rank), dtype=float)
            W = np.zeros((p * n, target_rank), dtype=float)
            
            # Initialize with small random values
            for r in range(target_rank):
                U[:, r] = rng.choice([-1, -0.5, 0, 0.5, 1, 2], size=n*m)
                V[:, r] = rng.choice([-1, -0.5, 0, 0.5, 1, 2], size=m*p)
                W[:, r] = rng.choice([-1, -0.5, 0, 0.5, 1, 2], size=p*n)
            
            if verify_decomposition(U, V, W):
                return (U.tolist(), V.tolist(), W.tolist())
        return None
    
    # Try structured approach: split the 9 columns into three groups of 3
    # For each group, we're doing 2x3 * 3x3 multiplication
    # If we can find a rank-r decomposition for 2x3*3x3, total rank = 3*r
    
    # Let's try to find good decompositions for the sub-problem 2x3 * 3x3
    # The trivial rank is 2*3*3 = 18
    # Lower bound: max(2*3, 3*3, 3*2) = max(6, 9, 6) = 9
    
    # Try random search for the full problem at various ranks
    # Start from lower bounds and go up
    lower_bound = max(n*m, m*p, p*n)  # 27
    
    # Try a few specific ranks with more trials
    for target_rank in [lower_bound, lower_bound+1, lower_bound+2, 
                         lower_bound+3, lower_bound+4, lower_bound+5,
                         30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
        
        result = try_random_search(target_rank, n_trials=200, seed=target_rank)
        if result is not None:
            best_rank = target_rank
            best_U, best_V, best_W = result
    
    # Also try a more systematic approach: use the structure
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # Group k into blocks of 3: k = 3*q + qk where q in {0,1,2}, qk in {0,1,2}
    # For each block q, C[i, 3*q+qk] = sum_j A[i,j]*B[j, 3*q+qk]
    # This is a 2x3 * 3x3 multiplication for each q
    
    # Try to find a decomposition where the W factors are structured
    # W[k*n+i, r] where k = 3*q+qk
    # If we can make W[3*q+qk, n+i, r] depend only on qk and r (not q),
    # then we can share the same rank-r decomposition across all 3 blocks
    
    # This would give total rank = r (shared) instead of 3*r
    # But we need the V factors to differ per block
    
    # Let's try: for each rank r, try to find U, V_q (for q=0,1,2), W
    # such that sum_r U[i,j,r]*V_q[j,3*q+qk,r]*W[3*q+qk,i,r] = 1 for all i,j,qk
    
    # Actually this is just the original problem with structure imposed on W
    # Let's try random search with this structure
    
    for target_rank in range(lower_bound, min(lower_bound + 15, best_rank)):
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
        
        rng = np.random.RandomState(target_rank * 7 + 13)
        for trial in range(100):
            if time.time() > deadline:
                break
            
            U = np.zeros((n * m, target_rank), dtype=float)
            V = np.zeros((m * p, target_rank), dtype=float)
            W = np.zeros((p * n, target_rank), dtype=float)
            
            # Impose structure: W[3*q+qk, n+i, r] = W_base[qk, n+i, r] (independent of q)
            W_base = rng.choice([-1, -0.5, 0, 0.5, 1], size=(3, n, target_rank))
            for q in range(3):
                for qk in range(3):
                    for i in range(n):
                        W[(3*q+qk)*n + i, :] = W_base[qk, i, :]
            
            U[:, :] = rng.choice([-1, -0.5, 0, 0.5, 1, 2], size=(n*m, target_rank))
            V[:, :] = rng.choice([-1, -0.5, 0, 0.5, 1, 2], size=(m*p, target_rank))
            
            if verify_decomposition(U, V, W):
                best_rank = target_rank
                best_U, best_V, best_W = U.tolist(), V.tolist(), W.tolist()
                break
    
    # Convert to list format if needed
    if isinstance(best_U, np.ndarray):
        U_final = best_U.tolist()
        V_final = best_V.tolist()
        W_final = best_W.tolist()
    else:
        U_final = best_U
        V_final = best_V
        W_final = best_W
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
