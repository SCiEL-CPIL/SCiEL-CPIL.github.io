import numpy as np
import itertools
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will solve for M_{n, m, p'} where p' divides p
    # If we find rank r for M_{n, m, p'}, we can repeat p/p' times to get rank r*(p/p') for M_{n, m, p}
    
    # Target: M_{2, 3, 2}
    n_sub, m_sub, p_sub = 2, 3, 2
    
    # Check if p is divisible by p_sub
    if p % p_sub == 0:
        repeats = p // p_sub
        # Try to find minimal rank for M_{2,3,2}
        best_rank = None
        best_U = None
        best_V = None
        best_W = None
        
        deadline = time.time() + 100 # 100 seconds budget for search
        
        # Try ranks 6, 7, 8...
        # Lower bound is max(n*m, m*p, p*n) = max(6, 6, 4) = 6
        for r in range(6, 13): # Up to 12 (schoolbook for sub-problem)
            if time.time() > deadline:
                break
            if _find_decomposition(n_sub, m_sub, p_sub, r, deadline):
                best_rank = r
                best_U, best_V, best_W = _last_solution
                break
        
        if best_rank is not None:
            # Construct full U, V, W
            full_rank = best_rank * repeats
            U_full = [[0] * full_rank for _ in range(n * m)]
            V_full = [[0] * full_rank for _ in range(m * p)]
            W_full = [[0] * full_rank for _ in range(p * n)]
            
            for rep in range(repeats):
                offset = rep * best_rank
                # Map indices
                # U indices: i*m + j (0..5) -> same in sub
                # V indices: j*p + k. In sub, k is 0..1. In full, k = rep*2 + k_sub
                # W indices: k*n + i. In sub, k is 0..1. In full, k = rep*2 + k_sub
                
                for i in range(n_sub):
                    for j in range(m_sub):
                        u_idx = i * m_sub + j
                        u_full_idx = i * m + j
                        for c in range(best_rank):
                            U_full[u_full_idx][offset + c] = best_U[u_idx][c]
                
                for j in range(m_sub):
                    for k_sub in range(p_sub):
                        v_idx = j * p_sub + k_sub
                        k_full = rep * p_sub + k_sub
                        v_full_idx = j * p + k_full
                        for c in range(best_rank):
                            V_full[v_full_idx][offset + c] = best_V[v_idx][c]
                
                for k_sub in range(p_sub):
                    for i in range(n_sub):
                        w_idx = k_sub * n_sub + i
                        k_full = rep * p_sub + k_sub
                        w_full_idx = k_full * n + i
                        for c in range(best_rank):
                            W_full[w_full_idx][offset + c] = best_W[w_idx][c]
                
            return {"U": U_full, "V": V_full, "W": W_full}
    
    # Fallback: Schoolbook
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

_last_solution = None

def _find_decomposition(n, m, p, r, deadline):
    global _last_solution
    import random
    import numpy as np
    
    # Strategy: 
    # 1. Try structured patterns first
    # 2. Then random search with wider value ranges
    
    # First, try structured decompositions
    if _try_structured(n, m, p, r):
        return True
    
    # Random search with multiple seeds and wider ranges
    for seed in range(50):
        if time.time() > deadline:
            return False
        
        random.seed(seed)
        for attempt in range(200):
            if time.time() > deadline:
                return False
            
            # Use wider range for better coverage
            val_range = 3
            U = np.array([[random.randint(-val_range, val_range) for _ in range(r)] for _ in range(n*m)], dtype=float)
            W = np.array([[random.randint(-val_range, val_range) for _ in range(r)] for _ in range(p*n)], dtype=float)
            
            V_sol = np.zeros((m*p, r), dtype=float)
            valid = True
            
            for j in range(m):
                for k in range(p):
                    # Build linear system for V[j*p+k, c]
                    # Eq for each i: sum_c U[i*m+j, c] W[k*n+i, c] V[j*p+k, c] = 1
                    A_sys = np.zeros((n, r), dtype=float)
                    for i in range(n):
                        for c in range(r):
                            A_sys[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    
                    # Solve A_sys @ V_col = ones(n)
                    try:
                        V_col = np.linalg.lstsq(A_sys, np.ones(n), rcond=None)[0]
                    except:
                        valid = False
                        break
                        
                    # Check if V_col has integer or half-integer entries
                    V_col_check = V_col * 2
                    if not np.allclose(V_col_check, np.round(V_col_check), atol=1e-6):
                        valid = False
                        break
                    V_sol[j*p+k, :] = V_col
                    
            if valid:
                # Verify the solution
                ok = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for c in range(r):
                                val += U[i*m+j, c] * V_sol[j*p+k, c] * W[k*n+i, c]
                            if not np.isclose(val, 1, atol=1e-6):
                                ok = False
                                break
                        if not ok: break
                    if not ok: break
                
                if ok:
                    _last_solution = (U.tolist(), V_sol.tolist(), W.tolist())
                    return True
    
    return False

def _try_structured(n, m, p, r):
    """Try structured decompositions with known patterns"""
    global _last_solution
    import numpy as np
    
    # Try pattern 1: U and W as permutation-like matrices
    # For rank r = n*m*p (schoolbook), this always works
    if r == n * m * p:
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        _last_solution = (U.tolist(), V.tolist(), W.tolist())
        return True
    
    # Try pattern 2: Kronecker-like structure
    # For M_{2,3,2}, try to use rank 6 with specific structure
    if n == 2 and m == 3 and p == 2 and r == 6:
        # Try a specific structured approach
        # U: 6x6, V: 6x6, W: 4x6
        # We need sum_c U[i*3+j, c] * V[j*2+k, c] * W[k*2+i, c] = 1
        
        # Try using orthogonal-like patterns
        # Let's try a few specific patterns
        
        # Pattern: use identity in some sense
        U = np.zeros((6, 6))
        W = np.zeros((4, 6))
        
        # Try to set up so that for each (j,k), the system is well-conditioned
        # Use different patterns for different (j,k)
        
        # Attempt 1: Simple alternating pattern
        for c in range(6):
            for i in range(2):
                for j in range(3):
                    U[i*3+j, c] = 1 if (i + j + c) % 2 == 0 else -1
            for k in range(2):
                for i in range(2):
                    W[k*2+i, c] = 1 if (k + i + c) % 2 == 0 else -1
        
        V_sol = np.zeros((6, 6))
        valid = True
        for j in range(3):
            for k in range(2):
                A_sys = np.zeros((2, 6))
                for i in range(2):
                    for c in range(6):
                        A_sys[i, c] = U[i*3+j, c] * W[k*2+i, c]
                try:
                    V_col = np.linalg.lstsq(A_sys, np.ones(2), rcond=None)[0]
                    V_col_check = V_col * 2
                    if not np.allclose(V_col_check, np.round(V_col_check), atol=1e-6):
                        valid = False
                        break
                    V_sol[j*2+k, :] = V_col
                except:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Verify
            ok = True
            for i in range(2):
                for j in range(3):
                    for k in range(2):
                        val = sum(U[i*3+j, c] * V_sol[j*2+k, c] * W[k*2+i, c] for c in range(6))
                        if not np.isclose(val, 1, atol=1e-6):
                            ok = False
                            break
                    if not ok: break
                if not ok: break
            
            if ok:
                _last_solution = (U.tolist(), V_sol.tolist(), W.tolist())
                return True
    
    return False
# EVOLVE-BLOCK-END
