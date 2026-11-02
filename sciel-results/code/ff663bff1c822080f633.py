"""Rewrite for matrix multiplication 3x4 x 4x8 rank minimization."""
import time
import numpy as np
from scipy.optimize import minimize
import itertools
import random

DIMENSIONS = [3, 4, 8]
TIME_LIMIT = 120  # seconds
SAFETY_MARGIN = 5  # seconds


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify tensor decomposition correctness."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build the tensor T_{i,j,k} = 1 if valid indices
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    rank = U.shape[1]
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = 0.0
                for r in range(rank):
                    val += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                err += abs(val - 1.0)
    return err


def _round_halfint(x):
    """Round to nearest half-integer."""
    return round(2 * x) / 2.0


def construct():
    """Construct rank-minimal exact tensor decomposition for 3x4 x 4x8 matrix multiplication."""
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions using optimization
    # Strategy: start with random initialization and optimize
    
    random.seed(42)
    np.random.seed(42)
    
    # Target ranks to try (from lower to higher)
    # Known bounds: rank is at least max(n*m, m*p, p*n) = max(12, 32, 24) = 32
    # Upper bound is 96 (schoolbook)
    # Try several ranks in between
    
    candidate_ranks = []
    # Try ranks from a reasonable range
    for r in range(35, 60):
        if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN - 30:
            break
        candidate_ranks.append(r)
    
    if not candidate_ranks:
        candidate_ranks = [40, 50, 60]
    
    # For each candidate rank, try to find a valid decomposition
    for target_rank in sorted(candidate_ranks, reverse=True):  # Try lower ranks first is harder, so go high to low? No, try to minimize
        # Actually, try from lower to upper to find the minimum
        pass
    
    # Better approach: try a few specific ranks with multiple restarts
    ranks_to_try = [38, 40, 42, 45, 50, 55, 60]
    
    for target_rank in sorted(ranks_to_try):
        if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        if target_rank >= best_rank:
            continue
        
        # Try multiple random initializations
        max_restarts = 3
        found = False
        
        for restart in range(max_restarts):
            if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            # Random initialization
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            # Scale to encourage non-zero values
            U /= np.max(np.abs(U)) if np.max(np.abs(U)) > 0 else 1
            V /= np.max(np.abs(V)) if np.max(np.abs(V)) > 0 else 1
            W /= np.max(np.abs(W)) if np.max(np.abs(W)) > 0 else 1
            
            # Define objective: sum of squared errors
            def objective(params):
                if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                    return 1e10
                # Reshape params
                U_flat = params[:n * m * target_rank]
                V_flat = params[n * m * target_rank:n * m * target_rank + m * p * target_rank]
                W_flat = params[n * m * target_rank + m * p * target_rank:]
                
                U_mat = U_flat.reshape(n * m, target_rank)
                V_mat = V_flat.reshape(m * p, target_rank)
                W_mat = W_flat.reshape(p * n, target_rank)
                
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            val = np.dot(U_mat[u_idx, :], V_mat[v_idx, :])
                            # This isn't right - we need element-wise product summed over r
                            val = 0.0
                            for r in range(target_rank):
                                val += U_mat[u_idx, r] * V_mat[v_idx, r] * W_mat[w_idx, r]
                            err += (val - 1.0) ** 2
                return err
            
            def objective_grad(params):
                # Numerical gradient approximation (expensive)
                if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                    return 1e10
                U_flat = params[:n * m * target_rank]
                V_flat = params[n * m * target_rank:n * m * target_rank + m * p * target_rank]
                W_flat = params[n * m * target_rank + m * p * target_rank:]
                
                U_mat = U_flat.reshape(n * m, target_rank)
                V_mat = V_flat.reshape(m * p, target_rank)
                W_mat = W_flat.reshape(p * n, target_rank)
                
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            val = 0.0
                            for r in range(target_rank):
                                val += U_mat[u_idx, r] * V_mat[v_idx, r] * W_mat[w_idx, r]
                            err += (val - 1.0) ** 2
                return err
            
            # Use a simpler approach: direct coordinate descent
            # For each element, update to minimize error
            
            U_mat = U.copy()
            V_mat = V.copy()
            W_mat = W.copy()
            
            # Coordinate descent iterations
            max_iter = 200
            converged = False
            
            for it in range(max_iter):
                if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                    break
                
                # Update U
                for idx in range(n * m):
                    best_val = U_mat[idx, :].copy()
                    best_err = _verify(U_mat.tolist(), V_mat.tolist(), W_mat.tolist(), n, m, p)
                    
                    for r in range(target_rank):
                        for delta in [-1.0, -0.5, 0.5, 1.0]:
                            U_mat[idx, r] = best_val[r] + delta
                            err = _verify(U_mat.tolist(), V_mat.tolist(), W_mat.tolist(), n, m, p)
                            if err < best_err:
                                best_err = err
                                best_val = U_mat[idx, :].copy()
                    U_mat[idx, :] = best_val
                
                # Update V
                for idx in range(m * p):
                    best_val = V_mat[idx, :].copy()
                    best_err = _verify(U_mat.tolist(), V_mat.tolist(), W_mat.tolist(), n, m, p)
                    
                    for r in range(target_rank):
                        for delta in [-1.0, -0.5, 0.5, 1.0]:
                            V_mat[idx, r] = best_val[r] + delta
                            err = _verify(U_mat.tolist(), V_mat.tolist(), W_mat.tolist(), n, m, p)
                            if err < best_err:
                                best_err = err
                                best_val = V_mat[idx, :].copy()
                    V_mat[idx, :] = best_val
                
                # Update W
                for idx in range(p * n):
                    best_val = W_mat[idx, :].copy()
                    best_err = _verify(U_mat.tolist(), V_mat.tolist(), W_mat.tolist(), n, m, p)
                    
                    for r in range(target_rank):
                        for delta in [-1.0, -0.5, 0.5, 1.0]:
                            W_mat[idx, r] = best_val[r] + delta
                            err = _verify(U_mat.tolist(), V_mat.tolist(), W_mat.tolist(), n, m, p)
                            if err < best_err:
                                best_err = err
                                best_val = W_mat[idx, :].copy()
                    W_mat[idx, :] = best_val
                
                # Check convergence
                if best_err < 1e-8:
                    converged = True
                    break
                
                if it % 10 == 0 and time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                    break
            
            if converged:
                # Round to half-integers and verify
                U_rounded = [[_round_halfint(U_mat[i, r]) for r in range(target_rank)] for i in range(n * m)]
                V_rounded = [[_round_halfint(V_mat[i, r]) for r in range(target_rank)] for i in range(m * p)]
                W_rounded = [[_round_halfint(W_mat[i, r]) for r in range(target_rank)] for i in range(p * n)]
                
                err_rounded = _verify(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-4)
                if err_rounded < 1e-3:
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                    best_rank = target_rank
                    found = True
                    break
            elif best_err < 1e-6:
                # Close enough, try to round
                U_rounded = [[_round_halfint(U_mat[i, r]) for r in range(target_rank)] for i in range(n * m)]
                V_rounded = [[_round_halfint(V_mat[i, r]) for r in range(target_rank)] for i in range(m * p)]
                W_rounded = [[_round_halfint(W_mat[i, r]) for r in range(target_rank)] for i in range(p * n)]
                
                err_rounded = _verify(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-4)
                if err_rounded < 1e-3:
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            # Try to go lower
            continue
        else:
            # This rank didn't work, try next higher
            continue
    
    # Final verification
    if best_rank < rank_schoolbook:
        final_err = _verify(best_U, best_V, best_W, n, m, p, tol=1e-4)
        if final_err > 1e-3:
            # Fall back to schoolbook
            best_U, best_V, best_W = U_school, V_school, W_school
            best_rank = rank_schoolbook
    
    return {"U": best_U, "V": best_V, "W": best_W}
