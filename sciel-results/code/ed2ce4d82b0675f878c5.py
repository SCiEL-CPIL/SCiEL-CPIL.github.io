import numpy as np
import time
import sys
import random
from itertools import product

DIMENSIONS = [3, 4, 6]

def _verify(U, V, W, n, m, p):
    """Verify that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p):
    """Vectorized verification."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(Ur[i,j,:], Vr[j,k,:]*Wr[k,i,:])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 72
    
    # Build schoolbook decomposition as fallback
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    best_rank = schoolbook_rank
    
    start_time = time.time()
    time_limit = 110  # Leave 10s safety margin
    
    def try_rank_fixed_VW(R, n, m, p, num_trials=200, time_budget=8.0):
        """
        Try to find a rank-R decomposition by fixing V and W randomly,
        then solving for U via least squares. Verify the result.
        """
        start = time.time()
        for trial in range(num_trials):
            if time.time() - start > time_budget:
                break
            
            # Random V and W
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # Build the linear system for U
            # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # For fixed (i,j): sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for all k
            # This gives a system: A[i,j] @ U[i*m+j, :] = ones(p)
            # where A[i,j][k, r] = V[j*p+k, r] * W[k*n+i, r]
            
            # Build A as (n*m*p, R) and b as (n*m*p,)
            A = np.zeros((n * m * p, R))
            b = np.ones(n * m * p)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        A[idx, :] = V[j * p + k, :] * W[k * n + i, :]
                        idx += 1
            
            # Solve least squares: min ||A @ U_flat - b||^2
            # U_flat is (n*m*R,)
            try:
                U_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                continue
            
            U = U_flat.reshape(n * m, R)
            
            # Verify
            if _verify_fast(U, V, W, n, m, p):
                return U, V, W
        
        return None
    
    def try_rank_fixed_UW(R, n, m, p, num_trials=200, time_budget=8.0):
        """
        Fix U and W randomly, solve for V.
        For fixed (j,k): sum_r V[j*p+k, r] * (U[i*m+j, r] * W[k*n+i, r]) = 1 for all i
        A[j,k][i, r] = U[i*m+j, r] * W[k*n+i, r]
        """
        start = time.time()
        for trial in range(num_trials):
            if time.time() - start > time_budget:
                break
            
            U = np.random.randn(n * m, R)
            W = np.random.randn(p * n, R)
            
            A = np.zeros((n * m * p, R))
            b = np.ones(n * m * p)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        A[idx, :] = U[i * m + j, :] * W[k * n + i, :]
                        idx += 1
            
            try:
                V_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                continue
            
            V = V_flat.reshape(m * p, R)
            
            if _verify_fast(U, V, W, n, m, p):
                return U, V, W
        
        return None
    
    def try_rank_fixed_UV(R, n, m, p, num_trials=200, time_budget=8.0):
        """
        Fix U and V randomly, solve for W.
        For fixed (k,i): sum_r W[k*n+i, r] * (U[i*m+j, r] * V[j*p+k, r]) = 1 for all j
        A[k,i][j, r] = U[i*m+j, r] * V[j*p+k, r]
        """
        start = time.time()
        for trial in range(num_trials):
            if time.time() - start > time_budget:
                break
            
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            
            A = np.zeros((n * m * p, R))
            b = np.ones(n * m * p)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        A[idx, :] = U[i * m + j, :] * V[j * p + k, :]
                        idx += 1
            
            try:
                W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                continue
            
            W = W_flat.reshape(p * n, R)
            
            if _verify_fast(U, V, W, n, m, p):
                return U, V, W
        
        return None
    
    # Try ranks from 71 down to 50
    # For each rank, try all three "fix-two-solve-one" strategies
    for R in range(schoolbook_rank - 1, max(schoolbook_rank - 25, 45), -1):
        if time.time() - start_time > time_limit - 15:
            break
        
        remaining_time = time_limit - 15 - (time.time() - start_time)
        if remaining_time < 2:
            break
        
        per_strategy_time = max(1.0, remaining_time / 6.0)
        per_strategy_trials = max(10, int(per_strategy_time * 50))
        
        # Try each strategy
        for strategy in [try_rank_fixed_VW, try_rank_fixed_UW, try_rank_fixed_UV]:
            if time.time() - start_time > time_limit - 15:
                break
            
            result = strategy(R, n, m, p, num_trials=per_strategy_trials, time_budget=per_strategy_time)
            if result is not None:
                U, V, W = result
                best_U, best_V, best_W = U, V, W
                best_rank = R
                break
        
        if best_rank < R:
            break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
