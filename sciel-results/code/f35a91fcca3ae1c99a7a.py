"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

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

def _try_rank(n, m, p, target_rank, timeout=10):
    """Try to find a decomposition of given rank (placeholder for future optimization)"""
    pass

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s kill
    
    def _build_rank54():
        rank = 54
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
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
    
    # Build baseline
    baseline = _build_rank54()
    valid, diff = _verify(baseline["U"], baseline["V"], baseline["W"], n, m, p)
    
    best = baseline
    best_rank = 54
    
    if time.time() > deadline - 30:
        return best
    
    # Try rank 53 and 52 using linear algebra
    # The condition is: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # This is a system of n*m*p = 60 equations in rank^3 unknowns (but structured).
    # For rank r, we have r*(n*m + m*p + p*n) = r*(10+30+12) = 52r unknowns.
    # For r=53: 52*53 = 2756 unknowns, 60 equations. Very underdetermined.
    # But the equations are nonlinear (products of 3 variables per term).
    
    # Alternative: fix U and V, solve for W linearly, or vice versa.
    # For fixed U and V, each equation is linear in W entries.
    # Equation (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # This is linear in W[k*n+i,r] for each r, but the coefficient depends on r.
    # Actually for fixed (i,j,k), we get: sum_r (U[i*m+j,r]*V[j*p+k,r]) * W[k*n+i,r] = 1
    # This is one linear equation in the W variables W[0,0],W[0,1],...,W[5,1] (12 vars)
    # with coefficients c_r = U[i*m+j,r]*V[j*p+k,r] for r=0..rank-1.
    # But we only use W[k*n+i,r] for the specific (k,i) pair!
    # So equation (i,j,k) only involves W[k*n+i, r] for all r.
    # That means for each (k,i) pair, we have m equations (one for each j) all involving
    # the same W[k*n+i, :].
    
    # So: fix U and V. For each (k,i) in range(p*n):
    #   For each j in range(m):
    #     sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # This is a system of m linear equations in rank unknowns W[k*n+i,0..rank-1].
    # If rank >= m, this is solvable (generically).
    
    # Strategy: 
    # 1. Pick random U and V with rank columns
    # 2. For each (k,i), solve the linear system for W[k*n+i,:]
    # 3. Verify the full decomposition
    
    import random
    random.seed(42)
    
    def _try_random_decomposition(target_rank, n, m, p, max_attempts=50):
        """Try to find a decomposition of given rank by fixing U,V randomly and solving for W."""
        for attempt in range(max_attempts):
            # Initialize U and V with random values
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            
            # Use sparse random initialization: each row has a few nonzeros
            for _ in range(target_rank):
                # Pick a few entries to be nonzero
                for _ in range(random.randint(1, 3)):
                    row = random.randint(0, n*m - 1)
                    U[row, _] = random.choice([-1, 1])
                for _ in range(random.randint(1, 3)):
                    row = random.randint(0, m*p - 1)
                    V[row, _] = random.choice([-1, 1])
            
            # For each (k, i), solve for W[k*n + i, :]
            W = np.zeros((p*n, target_rank))
            feasible = True
            for k in range(p):
                for i in range(n):
                    # Build system: for each j, sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
                    # Let c_r = U[i*m+j,r]*V[j*p+k,r] for the specific j
                    # System: M @ W[k*n+i,:] = 1 where M[j,r] = U[i*m+j,r]*V[j*p+k,r]
                    M = np.zeros((m, target_rank))
                    for j in range(m):
                        for r in range(target_rank):
                            M[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    
                    # Solve M @ w = ones(m)
                    try:
                        # Use least squares
                        w, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        # Check if it's an exact solution
                        if np.allclose(M @ w, np.ones(m), atol=1e-8):
                            W[k*n+i, :] = w
                        else:
                            # Try exact solution if possible
                            if rank_m == m and rank_m <= target_rank:
                                # Use pinv
                                w = np.linalg.pinv(M) @ np.ones(m)
                                if np.allclose(M @ w, np.ones(m), atol=1e-8):
                                    W[k*n+i, :] = w
                                else:
                                    feasible = False
                                    break
                            else:
                                feasible = False
                                break
                    except:
                        feasible = False
                        break
                if not feasible:
                    break
            
            if not feasible:
                continue
            
            # Verify
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            valid, diff = _verify(U_list, V_list, W_list, n, m, p)
            if valid:
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try rank 52
    if time.time() < deadline - 15:
        result = _try_random_decomposition(52, n, m, p, max_attempts=20)
        if result:
            best = result
            best_rank = 52
    
    # Try rank 51
    if time.time() < deadline - 10 and best_rank > 51:
        result = _try_random_decomposition(51, n, m, p, max_attempts=20)
        if result:
            best = result
            best_rank = 51
    
    # Try rank 50
    if time.time() < deadline - 5 and best_rank > 50:
        result = _try_random_decomposition(50, n, m, p, max_attempts=20)
        if result:
            best = result
            best_rank = 50
    
    return best
# EVOLVE-BLOCK-END
