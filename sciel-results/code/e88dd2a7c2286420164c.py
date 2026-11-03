"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

import numpy as np
from scipy.optimize import minimize
import itertools

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target tensor T[i*m+j, j*p+k, k*n+i] = 1
    # We look for U, V, W such that sum_r U[:,r] V[:,r] W[:,r] = T
    
    # Try ranks from a lower bound up to schoolbook rank
    # Lower bound is max(n*m, m*p, n*p) = 20
    # But typically much higher. Let's start from 60 (schoolbook) and go down?
    # Or start from a reasonable guess and go up?
    # If we find a solution at rank R, we can try R-1, etc.
    # But finding a solution is hard.
    # Let's try to find a solution for rank 60 first to ensure we have a valid baseline, 
    # then try to reduce.
    # Actually, the schoolbook is rank 60.
    # If we find rank 59, score improves.
    
    # Strategy:
    # 1. Try to find a decomposition for rank < 60 using optimization.
    # 2. If found, return it.
    # 3. If not, return schoolbook.
    
    max_rank_to_try = 60
    # We will try ranks from 60 down to 20? 
    # Optimization is expensive. Let's try a few specific ranks likely to be optimal.
    # Or just try to find *any* decomposition with rank < 60.
    
    # Let's try ranks 59, 58, ... down to 40.
    # If we don't find any, we return schoolbook.
    
    for rank in range(59, 39, -1):
        if _find_decomposition_optimization(n, m, p, rank):
            return _get_best_decomposition(n, m, p, rank)
            
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
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

def _find_decomposition_optimization(n, m, p, rank):
    """
    Use continuous optimization to find a real-valued decomposition,
    then round to nearest half-integer and verify.
    """
    global _best_decomposition
    _best_decomposition = None
    
    # Target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
                
    # Number of trials
    num_trials = 5
    
    for trial in range(num_trials):
        # Initialize U, V, W with small random values
        # Scaling is important. If we expect integers, start around 1.
        U0 = np.random.randn(n*m, rank) * 0.5
        V0 = np.random.randn(m*p, rank) * 0.5
        W0 = np.random.randn(p*n, rank) * 0.5
        
        # Flatten for optimization
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        def objective(x):
            U = x[:n*m*rank].reshape(n*m, rank)
            V = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
            
            # Compute tensor
            # T = sum_r U[:,r] outer V[:,r] outer W[:,r]
            # Efficient computation:
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # We can compute this as:
            # M1 = U @ V.T  (n*m x m*p) ? No, that's not right.
            # T is 3rd order.
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # Let's just use einsum or loops for small sizes.
            # n*m=12, m*p=20, p*n=15. Rank <= 60.
            # 12*20*15*60 = 216,000 operations. Very fast.
            
            T_comp = np.einsum('ar,br,cr->abc', U, V, W)
            
            # Error
            err = np.sum((T_comp - T_target)**2)
            
            # Regularization to push towards integers/half-integers
            # We want 2*x to be close to integer.
            # Penalty = sum (2*x - round(2*x))^2
            # This is non-smooth. We can use a smooth approximation or just rely on rounding after.
            # For now, just minimize tensor error.
            return err
        
        # Optimize
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10})
        except Exception as e:
            continue
            
        if res.success or res.fun < 1e-8:
            # Extract solution
            U_opt = res.x[:n*m*rank].reshape(n*m, rank)
            V_opt = res.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W_opt = res.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
            
            # Round to nearest half-integer
            # half_int = 0.5 * round(2 * val)
            U_round = 0.5 * np.round(2 * U_opt)
            V_round = 0.5 * np.round(2 * V_opt)
            W_round = 0.5 * np.round(2 * W_opt)
            
            # Verify exactness
            T_verify = np.einsum('ar,br,cr->abc', U_round, V_round, W_round)
            if np.allclose(T_verify, T_target, atol=1e-9):
                # Check if entries are actually integer or half-integer
                # They are by construction.
                _best_decomposition = (U_round, V_round, W_round)
                return True
                
    return False

def _get_best_decomposition(n, m, p, rank):
    """Get the best decomposition found"""
    global _best_decomposition
    if _best_decomposition is not None:
        U, V, W = _best_decomposition
        # Convert to lists
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    return _schoolbook(n, m, p)

# Global variable to store best decomposition
_best_decomposition = None
# EVOLVE-BLOCK-END
