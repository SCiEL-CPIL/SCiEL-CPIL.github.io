import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
                
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Strategy: Decompose 2x3x6 into two 2x3x3 blocks
    # Rank(2x3x6) <= 2 * Rank(2x3x3)
    # Try to find minimal rank for 2x3x3
    
    n2, m2, p2 = 2, 3, 3
    
    # Tensor for 2x3x3
    # T_flat shape (n2*m2, m2*p2, p2*n2) -> (6, 9, 6)
    # Non-zero entries: T[i*m2+j, j*p2+k, k*n2+i] = 1
    T_233 = np.zeros((n2 * m2, m2 * p2, p2 * n2))
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                T_233[i * m2 + j, j * p2 + k, k * n2 + i] = 1
                
    T_flat = T_233.flatten()
    
    def solve_233(rank):
        if time.time() > deadline - 10:
            return None
            
        # Variables: U (n2*m2 * rank), V (m2*p2 * rank), W (p2*n2 * rank)
        # Total vars: (6 + 9 + 6) * rank = 21 * rank
        
        def objective(params):
            U = params[0:n2*m2*rank].reshape(n2*m2, rank)
            V = params[n2*m2*rank:n2*m2*rank + m2*p2*rank].reshape(m2*p2, rank)
            W = params[n2*m2*rank + m2*p2*rank:].reshape(p2*n2, rank)
            
            # Compute tensor T_comp[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
            # i in 0..5, j in 0..8, k in 0..5
            T_comp = np.einsum('ir,jr,kr->ijk', U, V, W)
            diff = T_comp.flatten() - T_flat
            return diff
            
        best_params = None
        best_err = float('inf')
        
        # Try multiple restarts
        # Number of restarts depends on time
        max_restarts = 20
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            np.random.seed(42 + restart)
            size = 21 * rank
            params0 = np.random.randn(size) * 0.5
            
            try:
                from scipy.optimize import least_squares
                # Use 'lm' for small problems
                res = least_squares(objective, params0, method='lm', max_nfev=2000)
                err = np.linalg.norm(res.fun)
                if err < best_err:
                    best_err = err
                    best_params = res.x
            except Exception:
                pass
                
        if best_params is None or best_err > 1e-6:
            return None
            
        # Extract U, V, W
        U = best_params[0:n2*m2*rank].reshape(n2*m2, rank)
        V = best_params[n2*m2*rank:n2*m2*rank + m2*p2*rank].reshape(m2*p2, rank)
        W = best_params[n2*m2*rank + m2*p2*rank:].reshape(p2*n2, rank)
        
        # Snap to half-integers
        def snap_matrix(M):
            M_snap = np.round(M * 2) / 2.0
            T_comp = np.einsum('ir,jr,kr->ijk', M_snap[0:n2*m2], M_snap[n2*m2:n2*m2+m2*p2], M_snap[n2*m2+m2*p2:])
            diff = T_comp.flatten() - T_flat
            if np.linalg.norm(diff) < 1e-6:
                return M_snap
            return None # Return None if snapping fails
            
        U_snap = snap_matrix(U)
        V_snap = snap_matrix(V)
        W_snap = snap_matrix(W)
        
        if U_snap is None or V_snap is None or W_snap is None:
            # Try snapping individually or jointly?
            # If joint snapping fails, maybe try snapping with different tolerances?
            # Or just return None
            return None
            
        return U_snap, V_snap, W_snap

    # Try ranks 12 to 17
    for r in range(12, 18):
        if time.time() > deadline - 15:
            break
            
        res = solve_233(r)
        if res is not None:
            U_233, V_233, W_233 = res
            rank_236 = 2 * r
            U_236 = np.zeros((n*m, rank_236))
            V_236 = np.zeros((m*p, rank_236))
            W_236 = np.zeros((p*n, rank_236))
            
            # Block 0: k=0,1,2
            # Block 1: k=3,4,5
            
            # U is the same for both blocks
            U_236[:, 0:r] = U_233
            U_236[:, r:2*r] = U_233
            
            # V:
            # For block 0, V uses cols 0,1,2 of B (indices 0..2)
            # For block 1, V uses cols 3,4,5 of B (indices 3..5)
            # V_233 has rows j*3 + k (j=0..2, k=0..2)
            # V_236 has rows j*6 + k (j=0..2, k=0..5)
            
            for j in range(m2):
                for k in range(p2):
                    # Block 0
                    V_236[j*6 + k, 0:r] = V_233[j*3 + k, 0:r]
                    # Block 1
                    V_236[j*6 + (k+3), r:2*r] = V_233[j*3 + k, 0:r]
                    
            # W:
            # For block 0, W uses rows 0,1,2 of C (indices 0..2)
            # For block 1, W uses rows 3,4,5 of C (indices 3..5)
            # W_233 has rows k*2 + i (k=0..2, i=0..1)
            # W_236 has rows k*2 + i (k=0..5, i=0..1)
            
            for k in range(p2):
                for i in range(n2):
                    # Block 0
                    W_236[k*2 + i, 0:r] = W_233[k*2 + i, 0:r]
                    # Block 1
                    W_236[(k+3)*2 + i, r:2*r] = W_233[k*2 + i, 0:r]
                    
            # Verify
            # Construct full tensor and check
            U_full = U_236
            V_full = V_236
            W_full = W_236
            T_comp = np.einsum('ir,jr,kr->ijk', U_full, V_full, W_full)
            
            T_target = np.zeros((n*m, m*p, p*n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_target[i*m + j, j*p + k, k*n + i] = 1
                        
            if np.allclose(T_comp, T_target, atol=1e-6):
                return {"U": U_236.tolist(), "V": V_236.tolist(), "W": W_236.tolist()}
            else:
                # Verification failed, continue searching
                pass
                
    return {"U": best_U, "V": best_V, "W": best_W}
