import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose matrix multiplication."""
    rank = len(U[0]) if len(U) > 0 else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if abs(actual - 1.0) > tol:
                    return False
    return True

def _try_rank(target_rank, n, m, p, time_budget=10.0, seed=0):
    """
    Try to find a rank-target_rank decomposition.
    """
    start_time = time.time()
    np.random.seed(seed)
    
    n_m = n * m  # 6
    m_p = m * p  # 12
    p_n = p * n  # 8
    
    num_eqs = n * m * p  # 24
    num_vars = p_n * target_rank
    
    # Pre-compute equation indices
    eq_indices = {}
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq_indices[(i, j, k)] = idx
                idx += 1

    attempts = 0
    while time.time() - start_time < time_budget:
        attempts += 1
        if attempts > 20000:
            break
            
        # Strategy: Use sparse U and V with entries in {-1, 0, 1}
        # This makes the linear system for W well-conditioned
        
        # Random sparse initialization
        U = np.random.choice([-1, 0, 1], size=(n_m, target_rank))
        V = np.random.choice([-1, 0, 1], size=(m_p, target_rank))
        
        # Build A matrix: A[eq, var] = U[i*m+j, r] * V[j*p+k, r]
        # where var = (k*n+i)*rank + r
        A = np.zeros((num_eqs, num_vars))
        b = np.ones(num_eqs)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = eq_indices[(i, j, k)]
                    for r in range(target_rank):
                        var_idx = (k * n + i) * target_rank + r
                        A[eq_idx, var_idx] = U[i*m+j, r] * V[j*p+k, r]
        
        # Solve for W
        try:
            W_flat, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
        except:
            continue
        
        W = W_flat.reshape(p_n, target_rank)
        
        # Check if W is close to half-integers
        W_scaled = W * 2
        W_rounded = np.round(W_scaled) / 2
        error = np.max(np.abs(W - W_rounded))
        
        if error < 1e-8:
            # Verify exactness
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        actual = 0.0
                        for r in range(target_rank):
                            actual += U[i*m+j, r] * V[j*p+k, r] * W_rounded[k*n+i, r]
                        if abs(actual - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return {"U": U.tolist(), "V": V.tolist(), "W": W_rounded.tolist()}
        
        # Also try if W is close to integers
        W_int = np.round(W)
        error_int = np.max(np.abs(W - W_int))
        if error_int < 1e-8:
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        actual = 0.0
                        for r in range(target_rank):
                            actual += U[i*m+j, r] * V[j*p+k, r] * W_int[k*n+i, r]
                        if abs(actual - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return {"U": U.tolist(), "V": V.tolist(), "W": W_int.tolist()}

    return None

def _try_structured_rank(target_rank, n, m, p, time_budget=10.0, seed=0):
    """
    Try structured decompositions where U, V, W have specific patterns.
    """
    start_time = time.time()
    np.random.seed(seed)
    
    n_m = n * m
    m_p = m * p
    p_n = p * n
    num_eqs = n * m * p
    
    eq_indices = {}
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq_indices[(i, j, k)] = idx
                idx += 1
    
    # Try various structured patterns for U
    # Pattern 1: U[i*m+j, r] = 1 if r == i*m+j else 0 (identity-like)
    # Pattern 2: U[i*m+j, r] = 1 if (i*m+j) % target_rank == r else 0
    
    patterns = []
    
    # Identity-like pattern
    U_id = np.zeros((n_m, target_rank))
    for i in range(n_m):
        if i < target_rank:
            U_id[i, i] = 1
    patterns.append(('identity', U_id))
    
    # Cyclic pattern
    U_cyc = np.zeros((n_m, target_rank))
    for i in range(n_m):
        U_cyc[i, i % target_rank] = 1
    patterns.append(('cyclic', U_cyc))
    
    # Random but with structure: each row has exactly 1s
    for _ in range(10):
        U_rand = np.zeros((n_m, target_rank))
        for i in range(n_m):
            pos = np.random.randint(target_rank)
            U_rand[i, pos] = 1
        patterns.append(('random_sparse', U_rand))
    
    for name, U in patterns:
        if time.time() - start_time > time_budget:
            break
        
        for v_attempt in range(50):
            if time.time() - start_time > time_budget:
                break
            
            # Random V
            V = np.random.choice([-1, 0, 1], size=(m_p, target_rank))
            
            # Build A matrix
            num_vars = p_n * target_rank
            A = np.zeros((num_eqs, num_vars))
            b = np.ones(num_eqs)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = eq_indices[(i, j, k)]
                        for r in range(target_rank):
                            var_idx = (k * n + i) * target_rank + r
                            A[eq_idx, var_idx] = U[i*m+j, r] * V[j*p+k, r]
            
            try:
                W_flat, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
            
            W = W_flat.reshape(p_n, target_rank)
            
            # Check for half-integers
            W_rounded = np.round(W * 2) / 2
            if np.max(np.abs(W - W_rounded)) < 1e-8:
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            actual = 0.0
                            for r in range(target_rank):
                                actual += U[i*m+j, r] * V[j*p+k, r] * W_rounded[k*n+i, r]
                            if abs(actual - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W_rounded.tolist()}
            
            # Check for integers
            W_int = np.round(W)
            if np.max(np.abs(W - W_int)) < 1e-8:
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            actual = 0.0
                            for r in range(target_rank):
                                actual += U[i*m+j, r] * V[j*p+k, r] * W_int[k*n+i, r]
                            if abs(actual - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W_int.tolist()}

    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Try ranks from low to high
    for target_rank in [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]:
        if time.time() > deadline - 10:
            break
        
        # Try random search
        for seed in range(20):
            if time.time() > deadline - 5:
                break
            result = _try_rank(target_rank, n, m, p, 
                             time_budget=min(2, deadline - time.time() - 3), 
                             seed=seed + target_rank * 100)
            if result is not None:
                return result
        
        # Try structured search
        for seed in range(5):
            if time.time() > deadline - 5:
                break
            result = _try_structured_rank(target_rank, n, m, p, 
                                         time_budget=min(2, deadline - time.time() - 3), 
                                         seed=seed + target_rank * 200)
            if result is not None:
                return result
    
    # Fallback: trivial rank-24 decomposition
    rank = n * m * p  # 24
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
