import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145  # Leave 5s safety margin before 150s kill
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    def build_A_mat(U, V, rank):
        """Build the linear system matrix A such that A @ W_flat = target_flat"""
        A_mat = np.zeros((n * m * p, p * n * rank))
        for r in range(rank):
            for k in range(p):
                for i in range(n):
                    col_base = k * n * rank + i * rank + r
                    for j in range(m):
                        row = i * m * p + j * p + k
                        A_mat[row, col_base] = U[i * m + j, r] * V[j * p + k, r]
        return A_mat
    
    def try_rank(rank, max_trials=200):
        if time.time() > deadline - 20:
            return None
        
        target = np.ones((n, m, p))
        target_flat = target.flatten()
        
        for trial in range(max_trials):
            if time.time() > deadline - 20:
                return None
            
            # Random initialization with structured patterns
            U = np.random.randint(-1, 2, (n * m, rank))
            V = np.random.randint(-1, 2, (m * p, rank))
            
            # Build A_mat
            A_mat = build_A_mat(U, V, rank)
            
            # Solve least squares
            try:
                W_flat, residuals, rank_sv, sv = np.linalg.lstsq(A_mat, target_flat, rcond=None)
            except:
                continue
            
            # Check if solution is close to integer
            W_int = np.round(W_flat)
            error = np.linalg.norm(A_mat @ W_int - target_flat)
            
            if error < 1e-9:
                W = W_int.reshape(p * n, rank)
                return U, V, W
        
        return None
    
    def try_rank_with_polish(rank, max_trials=100):
        """Try rank with iterative polishing of near-integer solutions"""
        if time.time() > deadline - 20:
            return None
        
        target = np.ones((n, m, p))
        target_flat = target.flatten()
        
        for trial in range(max_trials):
            if time.time() > deadline - 20:
                return None
            
            # Random initialization
            U = np.random.randint(-1, 2, (n * m, rank))
            V = np.random.randint(-1, 2, (m * p, rank))
            
            # Build A_mat
            A_mat = build_A_mat(U, V, rank)
            
            # Solve least squares
            try:
                W_flat, residuals, rank_sv, sv = np.linalg.lstsq(A_mat, target_flat, rcond=None)
            except:
                continue
            
            # Check if solution is close to integer
            W_int = np.round(W_flat)
            error = np.linalg.norm(A_mat @ W_int - target_flat)
            
            if error < 1e-9:
                W = W_int.reshape(p * n, rank)
                return U, V, W
            
            # If close but not exact, try polishing
            if error < 0.1:
                # Try adjusting U and V slightly
                for polish_iter in range(10):
                    if time.time() > deadline - 20:
                        return None
                    
                    # Random small perturbation
                    U_pert = U.copy()
                    V_pert = V.copy()
                    
                    # Perturb a few entries
                    for _ in range(5):
                        if np.random.random() < 0.5:
                            idx = np.random.randint(0, n * m)
                            col = np.random.randint(0, rank)
                            U_pert[idx, col] += np.random.choice([-1, 1])
                        else:
                            idx = np.random.randint(0, m * p)
                            col = np.random.randint(0, rank)
                            V_pert[idx, col] += np.random.choice([-1, 1])
                    
                    A_mat_pert = build_A_mat(U_pert, V_pert, rank)
                    
                    try:
                        W_flat_pert, residuals_pert, rank_sv_pert, sv_pert = np.linalg.lstsq(A_mat_pert, target_flat, rcond=None)
                    except:
                        continue
                    
                    W_int_pert = np.round(W_flat_pert)
                    error_pert = np.linalg.norm(A_mat_pert @ W_int_pert - target_flat)
                    
                    if error_pert < 1e-9:
                        W = W_int_pert.reshape(p * n, rank)
                        return U_pert, V_pert, W
        
        return None
    
    # Try ranks from lower to higher
    # Known lower bounds suggest rank >= 30 for 3x4x4
    # Try from 35 upward with decreasing trial counts
    for rank in range(35, 48):
        if time.time() > deadline - 25:
            break
        
        # First pass: direct search
        trials = 150 if rank < 40 else 80
        result = try_rank(rank, max_trials=trials)
        if result is not None:
            U, V, W = result
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
        
        # Second pass: with polishing
        if time.time() > deadline - 25:
            break
        trials_polish = 50 if rank < 40 else 30
        result = try_rank_with_polish(rank, max_trials=trials_polish)
        if result is not None:
            U, V, W = result
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    return schoolbook()
