import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 4]
_SEARCH_TIMEOUT = 110.0  # 10s margin before 120s limit

def _verify_exact(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact and entries are half-integers."""
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    W_perm = np.transpose(W3, (1, 0, 2))
    
    T_calc = np.einsum('ijr,jkr,ikr->ijk', U3, V3, W_perm, optimize=True)
    
    if not np.allclose(T_calc, np.ones((n, m, p)), atol=tol):
        return False
        
    def is_half_int(arr):
        return np.allclose(arr * 2, np.round(arr * 2), atol=1e-9)
        
    return is_half_int(U) and is_half_int(V) and is_half_int(W)

def _residual(U, V, W, n, m, p):
    """Compute the residual norm."""
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    W_perm = np.transpose(W3, (1, 0, 2))
    T_calc = np.einsum('ijr,jkr,ikr->ijk', U3, V3, W_perm, optimize=True)
    return np.linalg.norm(T_calc - 1.0)

def _schoolbook():
    """Return the schoolbook decomposition with rank 36."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _try_rank_with_patterns(n, m, p, R, deadline, seed=42):
    """
    Try to find a rank-R decomposition using structured patterns.
    Exploit the fact that T[i,j,k]=1 for all i,j,k.
    """
    np.random.seed(seed)
    
    # Strategy: Try different density patterns and value ranges
    # For small ranks, we need dense matrices with small values
    
    best_res = float('inf')
    best_U, best_V, best_W = None, None, None
    
    # Try multiple restarts with different strategies
    for restart in range(30):
        if time.time() > deadline:
            break
        
        # Choose density based on rank
        density = min(0.5, 12.0 / R)  # Higher density for lower rank
        
        # Use values in {-1, -0.5, 0, 0.5, 1}
        half_int_values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        
        for r in range(R):
            # Each column should have some structure
            # Try to make each "term" contribute to multiple outputs
            u_nonzero = np.random.choice(n*m, size=max(1, int(n*m*density)), replace=False)
            v_nonzero = np.random.choice(m*p, size=max(1, int(m*p*density)), replace=False)
            w_nonzero = np.random.choice(p*n, size=max(1, int(p*n*density)), replace=False)
            
            for idx in u_nonzero:
                U[idx, r] = np.random.choice(half_int_values)
            for idx in v_nonzero:
                V[idx, r] = np.random.choice(half_int_values)
            for idx in w_nonzero:
                W[idx, r] = np.random.choice(half_int_values)
        
        res = _residual(U, V, W, n, m, p)
        
        if res < best_res:
            best_res = res
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Local refinement: try to fix individual entries
        for step in range(50):
            if time.time() > deadline:
                break
            
            if res < 1e-9:
                if _verify_exact(U, V, W, n, m, p):
                    return U, V, W
                break
            
            # Try changing a few entries
            for _ in range(3):
                if time.time() > deadline:
                    break
                
                mat_choice = np.random.randint(0, 3)
                if mat_choice == 0:
                    mat = U; rows = n*m
                elif mat_choice == 1:
                    mat = V; rows = m*p
                else:
                    mat = W; rows = p*n
                
                row = np.random.randint(0, rows)
                col = np.random.randint(0, R)
                
                old_val = mat[row, col]
                new_val = np.random.choice(half_int_values)
                
                mat[row, col] = new_val
                new_res = _residual(U, V, W, n, m, p)
                
                if new_res < res:
                    res = new_res
                else:
                    mat[row, col] = old_val
        
        if best_U is not None and _verify_exact(best_U, best_V, best_W, n, m, p):
            return best_U, best_V, best_W
    
    return None

def _try_rank_gradient_snap(n, m, p, R, deadline, seed=42):
    """
    Gradient descent in continuous space, then snap to half-integers.
    """
    np.random.seed(seed)
    
    U = np.random.randn(n*m, R) * 0.3
    V = np.random.randn(m*p, R) * 0.3
    W = np.random.randn(p*n, R) * 0.3
    
    lr = 0.005
    best_res = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for step in range(3000):
        if time.time() > deadline:
            break
            
        res = _residual(U, V, W, n, m, p)
        if res < best_res:
            best_res = res
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if res < 1e-7:
            break
        
        # Compute gradient via finite differences on a few random coordinates
        for _ in range(5):
            if time.time() > deadline:
                break
                
            mat_choice = np.random.randint(0, 3)
            if mat_choice == 0:
                mat = U; rows = n*m
            elif mat_choice == 1:
                mat = V; rows = m*p
            else:
                mat = W; rows = p*n
                
            row = np.random.randint(0, rows)
            col = np.random.randint(0, R)
            
            eps = 0.005
            old_val = mat[row, col]
            mat[row, col] = old_val + eps
            res_plus = _residual(U, V, W, n, m, p)
            mat[row, col] = old_val - eps
            res_minus = _residual(U, V, W, n, m, p)
            mat[row, col] = old_val
            
            grad = (res_plus - res_minus) / (2 * eps)
            mat[row, col] = old_val - lr * grad
            mat[row, col] = np.clip(mat[row, col], -1.5, 1.5)
    
    # Snap to half-integers
    U_s = np.round(best_U * 2) / 2
    V_s = np.round(best_V * 2) / 2
    W_s = np.round(best_W * 2) / 2
    
    if _verify_exact(U_s, V_s, W_s, n, m, p):
        return U_s, V_s, W_s
    
    return None

def _try_rank_combined(n, m, p, R, deadline, seed=42):
    """
    Combined approach: try multiple strategies.
    """
    np.random.seed(seed)
    
    # Strategy 1: Pattern-based
    result = _try_rank_with_patterns(n, m, p, R, deadline, seed)
    if result is not None:
        return result
    
    # Strategy 2: Gradient + snap
    result = _try_rank_gradient_snap(n, m, p, R, deadline, seed)
    if result is not None:
        return result
    
    return None

def construct():
    t0 = time.time()
    hard_deadline = t0 + _SEARCH_TIMEOUT
    n, m, p = DIMENSIONS
    
    best_result = None
    
    # Try ranks from lower bound up to 35
    # Lower bound: 12 (max of n*m, m*p, p*n)
    # But realistically, we may not find anything below 30
    
    # Time budget: ~110s total
    # Try ranks in order, with more time for lower ranks
    
    # First, try a few key ranks with more time
    key_ranks = [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34]
    
    for R in key_ranks:
        if time.time() > hard_deadline:
            break
            
        # Allocate time based on rank: lower ranks get more time
        time_budget = min(6.0, 110.0 / (36 - R + 1))
        rank_deadline = min(time.time() + time_budget, hard_deadline)
        
        for seed in [42, 123, 456]:
            if time.time() > rank_deadline:
                break
                
            result = _try_rank_combined(n, m, p, R, rank_deadline, seed)
            if result is not None:
                best_result = result
                break
                
        if best_result is not None:
            break
    
    # If not found in key ranks, try remaining
    if best_result is None:
        remaining_ranks = [R for R in range(13, 36) if R not in key_ranks]
        
        for R in remaining_ranks:
            if time.time() > hard_deadline:
                break
                
            rank_deadline = min(time.time() + 2.0, hard_deadline)
            
            for seed in [42, 123]:
                if time.time() > rank_deadline:
                    break
                    
                result = _try_rank_combined(n, m, p, R, rank_deadline, seed)
                if result is not None:
                    best_result = result
                    break
                    
            if best_result is not None:
                break
    
    # Fallback to schoolbook
    if best_result is None:
        U, V, W = _schoolbook()
    else:
        U, V, W = best_result
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
