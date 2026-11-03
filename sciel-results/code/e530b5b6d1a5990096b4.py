import time
import numpy as np
import random
from itertools import product

DIMENSIONS = [3, 3, 4]
START_TIME = time.time()
DEADLINE = START_TIME + 140

def construct():
    n, m, p = DIMENSIONS
    U_rows = n * m  # 9
    V_rows = m * p  # 12
    W_rows = p * n  # 12
    
    def schoolbook():
        U = np.zeros((U_rows, n*m*p))
        V = np.zeros((V_rows, n*m*p))
        W = np.zeros((W_rows, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    def compute_error(U, V, W):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += (s - 1.0) ** 2
        return err
    
    def alternating_solve(U, V, W, max_iter=100, tol=1e-12):
        U = U.copy()
        V = V.copy()
        W = W.copy()
        r = len(U[0])
        
        prev_err = compute_error(U, V, W)
        
        for it in range(max_iter):
            if time.time() > DEADLINE - 5:
                break
            
            # Solve for U: for each (i,j), solve linear system in U[i*m+j,:]
            for i in range(n):
                for j in range(m):
                    A_mat = np.zeros((p, r))
                    b_vec = np.ones(p)
                    for k in range(p):
                        A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    U[i*m+j, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            
            # Solve for V: for each (j,k), solve linear system in V[j*p+k,:]
            for j in range(m):
                for k in range(p):
                    A_mat = np.zeros((n, r))
                    b_vec = np.ones(n)
                    for i in range(n):
                        A_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    V[j*p+k, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            
            # Solve for W: for each (k,i), solve linear system in W[k*n+i,:]
            for k in range(p):
                for i in range(n):
                    A_mat = np.zeros((m, r))
                    b_vec = np.ones(m)
                    for j in range(m):
                        A_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    W[k*n+i, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            
            cur_err = compute_error(U, V, W)
            if abs(prev_err - cur_err) < tol:
                break
            prev_err = cur_err
        
        return U, V, W, cur_err
    
    def verify(U, V, W, tol=1e-8):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def snap_to_half(U, V, W):
        U = np.round(2 * U) / 2
        V = np.round(2 * V) / 2
        W = np.round(2 * W) / 2
        return U, V, W

    def try_snap_and_verify(U, V, W):
        # Try snapping to nearest 0.5
        U_s, V_s, W_s = snap_to_half(U, V, W)
        if verify(U_s, V_s, W_s, tol=1e-6):
            return U_s, V_s, W_s, True
        
        # Try snapping to nearest integer (subset of 0.5 grid)
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if verify(U_i, V_i, W_i, tol=1e-6):
            return U_i, V_i, W_i, True
            
        # If raw error is extremely small, maybe it's just floating point noise on valid halves?
        # But verify uses exact dot products. 
        # Let's check if the raw error is small enough that snapping *might* work if we tweak?
        # No, snapping is discrete. If verify fails, it fails.
        
        return U, V, W, False

    U_sb, V_sb, W_sb = schoolbook()
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p
    
    # We know rank 36 works.
    # Let's try to find lower ranks.
    # Strategy:
    # 1. Try ranks from 30 down to 18.
    # 2. For each rank, try several random inits.
    # 3. If a rank R is found, use it to warm start R-1.
    
    # To save time, we can start with a higher rank and go down.
    # But random search for 30 is hard.
    # Let's use a "greedy" approach:
    # Start with rank 36 (schoolbook).
    # Try to merge columns? No, that's not how ALS works.
    
    # Alternative: Just do random search but with better init.
    # Init: Take schoolbook, randomly select R columns, scale them?
    # Or: Random init, but run for many iterations.
    
    ranks_to_try = [30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18]
    
    # We will store the best solution found for each rank to use for warm starting
    last_solution = None
    
    for target_rank in ranks_to_try:
        if time.time() > DEADLINE - 15:
            break
        
        found = False
        attempts = 0
        max_attempts = 10 # Reduce attempts per rank to allow trying more ranks
        
        while attempts < max_attempts and time.time() < DEADLINE - 15:
            attempts += 1
            np.random.seed(attempts * 7919 + target_rank * 1000)
            
            # Initialization Strategy:
            # If we have a solution from a higher rank, use it.
            # We can't just drop a column, but we can initialize with the previous solution 
            # and add a small random perturbation to the dropped column?
            # Actually, if last_solution has rank R_prev > target_rank, we can't directly use it.
            # But we can initialize with random.
            
            # Let's try a "mixed" init:
            # If last_solution exists and its rank is target_rank + 1, 
            # we can take the first target_rank columns and randomize the last one?
            # No, that's not standard.
            
            # Let's stick to random but with a bias towards small values?
            # Or use the schoolbook structure?
            
            U = np.random.randn(U_rows, target_rank) * 0.5
            V = np.random.randn(V_rows, target_rank) * 0.5
            W = np.random.randn(W_rows, target_rank) * 0.5
            
            U_opt, V_opt, W_opt, err = alternating_solve(U, V, W, max_iter=300)
            
            if err < 1e-6:
                U_snap, V_snap, W_snap, success = try_snap_and_verify(U_opt, V_opt, W_opt)
                if success:
                    best_U, best_V, best_W = U_snap, V_snap, W_snap
                    best_rank = target_rank
                    last_solution = (U_snap, V_snap, W_snap)
                    found = True
                    break
            elif err < 0.01:
                # Refine
                U_opt, V_opt, W_opt, err = alternating_solve(U_opt, V_opt, W_opt, max_iter=500)
                if err < 1e-6:
                    U_snap, V_snap, W_snap, success = try_snap_and_verify(U_opt, V_opt, W_opt)
                    if success:
                        best_U, best_V, best_W = U_snap, V_snap, W_snap
                        best_rank = target_rank
                        last_solution = (U_snap, V_snap, W_snap)
                        found = True
                        break
    
    if not verify(best_U, best_V, best_W, tol=1e-6):
        U_sb, V_sb, W_sb = schoolbook()
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
