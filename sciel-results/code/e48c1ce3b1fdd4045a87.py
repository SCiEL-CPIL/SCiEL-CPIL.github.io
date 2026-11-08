"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

DIMENSIONS = [2, 5, 6]

def _verify(U, V, W, n, m, p):
    """Verify the tensor decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 if valid matmul term"""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    total = np.zeros((n, m, p))
    for r in range(rank):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total[i,j,k] += u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
    expected = np.ones((n, m, p))
    if not np.allclose(total, expected, atol=1e-6):
        diff = np.max(np.abs(total - expected))
        return False, diff
    return True, 0.0

def _build_rank54(n, m, p):
    """Constructs the known rank-54 decomposition using block Strassen."""
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # j_pairs: (0,1) and (2,3)
    # k_pairs: (0,1), (2,3), (4,5)
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            # P1 = (A_0,j0 + A_1,j1)(B_j0,k0 + B_j1,k1)
            U[0*m + j0][col] += 1
            U[1*m + j1][col] += 1
            V[j0*p + k0][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P2 = (A_1,j0 + A_1,j1) B_j0,k0
            U[1*m + j0][col] += 1
            U[1*m + j1][col] += 1
            V[j0*p + k0][col] += 1
            W[k0*n + 1][col] += 1
            W[k1*n + 1][col] -= 1
            col += 1
            
            # P3 = A_0,j0 (B_j0,k1 - B_j1,k1)
            U[0*m + j0][col] += 1
            V[j0*p + k1][col] += 1
            V[j1*p + k1][col] -= 1
            W[k1*n + 0][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P4 = A_1,j1 (B_j1,k0 - B_j0,k0)
            U[1*m + j1][col] += 1
            V[j1*p + k0][col] += 1
            V[j0*p + k0][col] -= 1
            W[k0*n + 0][col] += 1
            W[k0*n + 1][col] += 1
            col += 1
            
            # P5 = (A_0,j0 + A_0,j1) B_j1,k1
            U[0*m + j0][col] += 1
            U[0*m + j1][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] -= 1
            W[k1*n + 0][col] += 1
            col += 1
            
            # P6 = (A_1,j0 - A_0,j0)(B_j0,k0 + B_j0,k1)
            U[1*m + j0][col] += 1
            U[0*m + j0][col] -= 1
            V[j0*p + k0][col] += 1
            V[j0*p + k1][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P7 = (A_0,j1 - A_1,j1)(B_j1,k0 + B_j1,k1)
            U[0*m + j1][col] += 1
            U[1*m + j1][col] -= 1
            V[j1*p + k0][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] += 1
            col += 1
            
    # Remaining j=4
    for k in range(p):
        U[0*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 0][col] += 1
        col += 1
        U[1*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 1][col] += 1
        col += 1
        
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # 1. Build and verify the baseline rank-54 solution
    baseline = _build_rank54(n, m, p)
    valid, diff = _verify(baseline["U"], baseline["V"], baseline["W"], n, m, p)
    
    best = baseline
    best_rank = 54
    
    if not valid:
        # Fallback: should not happen, but return anyway
        return best

    # 2. Attempt to find a lower rank solution (53, 52, 51)
    # Strategy: Fix U and V randomly (dense, well-conditioned), solve for W linearly.
    # For each (k, i), we have a system M_ki @ w_ki = 1_m
    # where M_ki[j, r] = U[i*m+j, r] * V[j*p+k, r]
    
    import random
    random.seed(42)
    np.random.seed(42)

    def _try_lower_rank(target_rank, max_attempts=20, time_limit=5.0):
        """
        Attempts to find a decomposition of target_rank.
        Returns dict if successful, None otherwise.
        """
        start = time.time()
        for attempt in range(max_attempts):
            if time.time() - start > time_limit:
                break
            
            # Initialize U and V with random normal distributions
            # Scaling down helps with condition number
            scale = 1.0
            U = np.random.randn(n * m, target_rank) * scale
            V = np.random.randn(m * p, target_rank) * scale
            
            W = np.zeros((p * n, target_rank))
            feasible = True
            
            for k in range(p):
                for i in range(n):
                    # Construct M_ki (m x target_rank)
                    M = np.zeros((m, target_rank))
                    for j in range(m):
                        # M[j, r] = U[i*m+j, r] * V[j*p+k, r]
                        M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    
                    # Solve M @ w = ones(m)
                    try:
                        # Use lstsq for robustness
                        w, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        
                        # Check if solution is exact (within tolerance)
                        # and if the solution is numerically stable (not too large)
                        if np.allclose(M @ w, np.ones(m), atol=1e-8):
                            # Heuristic: reject if W entries are too large (precision risk)
                            if np.max(np.abs(w)) < 1000:
                                W[k*n+i, :] = w
                            else:
                                feasible = False
                                break
                        else:
                            feasible = False
                            break
                    except np.linalg.LinAlgError:
                        feasible = False
                        break
                if not feasible:
                    break
            
            if not feasible:
                continue
            
            # Verify the full decomposition
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            # Quick check before full verify
            valid, diff = _verify(U_list, V_list, W_list, n, m, p)
            if valid:
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Try rank 53
    if time.time() < deadline - 30:
        result = _try_lower_rank(53, max_attempts=30, time_limit=10.0)
        if result:
            best = result
            best_rank = 53

    # Try rank 52
    if time.time() < deadline - 20 and best_rank > 52:
        result = _try_lower_rank(52, max_attempts=30, time_limit=10.0)
        if result:
            best = result
            best_rank = 52

    # Try rank 51
    if time.time() < deadline - 10 and best_rank > 51:
        result = _try_lower_rank(51, max_attempts=30, time_limit=10.0)
        if result:
            best = result
            best_rank = 51

    return best
# EVOLVE-BLOCK-END
