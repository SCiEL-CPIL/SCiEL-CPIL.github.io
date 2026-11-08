import time
import sys
import itertools
from fractions import Fraction

DIMENSIONS = [3, 4, 4]
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    global START_TIME, TIME_LIMIT
    
    # Trivial rank decomposition (baseline)
    rank_trivial = n * m * p
    U_trivial = [[0] * rank_trivial for _ in range(n * m)]
    V_trivial = [[0] * rank_trivial for _ in range(m * p)]
    W_trivial = [[0] * rank_trivial for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j][col] = 1
                V_trivial[j * p + k][col] = 1
                W_trivial[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_trivial
    best_result = {"U": U_trivial, "V": V_trivial, "W": W_trivial}
    
    # Search for lower rank decompositions
    # Use NumPy for faster verification
    
    def verify_decomposition_numpy(U, V, W, n, m, p):
        """Verify using NumPy broadcasting."""
        try:
            U_reshaped = np.array(U).reshape(n, m, -1)
            V_reshaped = np.array(V).reshape(m, p, -1)
            W_reshaped = np.array(W).reshape(p, n, -1)
            
            # Sum over r (last dimension)
            # U[i,j,r] * V[j,k,r] * W[k,i,r]
            result = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
            return np.all(result == 1)
        except Exception:
            return False

    def try_rank_numpy(r, n, m, p, max_iterations=5000):
        """Try to find a rank-r decomposition with coefficients in {-1, 0, 1} using NumPy."""
        import random
        random.seed(42)
        
        num_u = n * m
        num_v = m * p
        num_w = p * n
        
        for _ in range(max_iterations):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                return None
            
            # Generate random U, V, W with entries in {-1, 0, 1}
            U = np.random.choice([-1, 0, 1], size=(num_u, r))
            V = np.random.choice([-1, 0, 1], size=(num_v, r))
            W = np.random.choice([-1, 0, 1], size=(num_w, r))
            
            if verify_decomposition_numpy(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None

    # Try ranks from 30 upward (closer to trivial bound 48)
    # Higher ranks are more likely to have solutions if any exist
    for r in range(30, rank_trivial):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        result = try_rank_numpy(r, n, m, p, max_iterations=5000)
        if result is not None:
            best_rank = r
            best_result = result
            break
    
    return best_result
# EVOLVE-BLOCK-END
