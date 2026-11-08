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
    
    def verify_decomposition_numpy(U, V, W, n, m, p):
        """Verify using NumPy broadcasting."""
        try:
            U_arr = np.array(U, dtype=np.int64).reshape(n, m, -1)
            V_arr = np.array(V, dtype=np.int64).reshape(m, p, -1)
            W_arr = np.array(W, dtype=np.int64).reshape(p, n, -1)
            result = np.einsum('ijr,jkr,kir->ijk', U_arr, V_arr, W_arr)
            return np.all(result == 1)
        except Exception:
            return False

    def verify_decomposition_f2(U, V, W, n, m, p):
        """Verify over F2."""
        try:
            U_arr = np.array(U, dtype=np.int64).reshape(n, m, -1) % 2
            V_arr = np.array(V, dtype=np.int64).reshape(m, p, -1) % 2
            W_arr = np.array(W, dtype=np.int64).reshape(p, n, -1) % 2
            result = np.einsum('ijr,jkr,kir->ijk', U_arr, V_arr, W_arr) % 2
            return np.all(result == 1)
        except Exception:
            return False

    # F2-based support search: find sparse supports over F2, then lift to integers
    def f2_support_search(r, n, m, p, max_iterations=20000):
        """Search for F2 decomposition with sparse supports, then try to lift."""
        np.random.seed(12345)
        
        num_u = n * m
        num_v = m * p
        num_w = p * n
        
        for _ in range(max_iterations):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                return None
            
            # Generate sparse random matrices over F2
            U = (np.random.rand(num_u, r) < 0.3).astype(int)
            V = (np.random.rand(num_v, r) < 0.3).astype(int)
            W = (np.random.rand(num_w, r) < 0.3).astype(int)
            
            if verify_decomposition_f2(U, V, W, n, m, p):
                # Found F2 decomposition - check if it lifts to integers
                if verify_decomposition_numpy(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                # Try sign flips for lifting
                for su in [1, -1]:
                    for sv in [1, -1]:
                        for sw in [1, -1]:
                            U2 = (U * su).tolist()
                            V2 = (V * sv).tolist()
                            W2 = (W * sw).tolist()
                            if verify_decomposition_numpy(U2, V2, W2, n, m, p):
                                return {"U": U2, "V": V2, "W": W2}
        
        return None

    # Direct integer search with {-1, 0, 1} coefficients
    def integer_search(r, n, m, p, max_iterations=15000):
        """Direct search for integer decomposition."""
        np.random.seed(67890)
        
        num_u = n * m
        num_v = m * p
        num_w = p * n
        
        for _ in range(max_iterations):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                return None
            
            U = np.random.choice([-1, 0, 1], size=(num_u, r))
            V = np.random.choice([-1, 0, 1], size=(num_v, r))
            W = np.random.choice([-1, 0, 1], size=(num_w, r))
            
            if verify_decomposition_numpy(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None

    # Try F2 search for ranks close to trivial bound
    for r in range(44, rank_trivial + 1):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        result = f2_support_search(r, n, m, p, max_iterations=10000)
        if result is not None:
            best_rank = r
            best_result = result
            break
    
    # If F2 search didn't find anything, try direct integer search
    if best_rank == rank_trivial:
        for r in range(44, rank_trivial + 1):
            if time.time() - START_TIME > TIME_LIMIT - 10:
                break
            
            result = integer_search(r, n, m, p, max_iterations=8000)
            if result is not None:
                best_rank = r
                best_result = result
                break
    
    return best_result
# EVOLVE-BLOCK-END
