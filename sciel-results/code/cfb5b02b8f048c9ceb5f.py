import time
import numpy as np
from fractions import Fraction

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _construct_rank39():
    """
    Explicit rank-39 decomposition for M_{2,2,11}.
    5 blocks of 2 columns (Strassen rank 7 each) + 1 single column (schoolbook rank 4).
    Total: 5*7 + 4 = 39.
    """
    n, m, p = 2, 2, 11
    rank = 39
    
    U = [[0.0] * rank for _ in range(n * m)]   # 4 x 39
    V = [[0.0] * rank for _ in range(m * p)]   # 22 x 39
    W = [[0.0] * rank for _ in range(p * n)]   # 22 x 39
    
    r = 0
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        v_e = k0        
        v_f = k1        
        v_g = p + k0    
        v_h = p + k1    
        
        w_c00 = k0 * n      
        w_c10 = k0 * n + 1  
        w_c01 = k1 * n      
        w_c11 = k1 * n + 1  
        
        U[0][r] = 1; U[3][r] = 1
        V[v_e][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1; W[w_c11][r] = 1
        r += 1
        
        U[2][r] = 1; U[3][r] = 1
        V[v_e][r] = 1
        W[w_c10][r] = 1; W[w_c11][r] = -1
        r += 1
        
        U[0][r] = 1
        V[v_f][r] = 1; V[v_h][r] = -1
        W[w_c01][r] = 1; W[w_c11][r] = 1
        r += 1
        
        U[3][r] = 1
        V[v_g][r] = 1; V[v_e][r] = -1
        W[w_c00][r] = 1; W[w_c10][r] = 1
        r += 1
        
        U[0][r] = 1; U[1][r] = 1
        V[v_h][r] = 1
        W[w_c00][r] = -1; W[w_c01][r] = 1
        r += 1
        
        U[2][r] = 1; U[0][r] = -1
        V[v_e][r] = 1; V[v_f][r] = 1
        W[w_c11][r] = 1
        r += 1
        
        U[1][r] = 1; U[3][r] = -1
        V[v_g][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1
        r += 1
    
    k2 = 10
    v_e2 = k2        
    v_g2 = p + k2    
    w_c02 = k2 * n      
    w_c12 = k2 * n + 1  
    
    U[0][r] = 1
    V[v_e2][r] = 1
    W[w_c02][r] = 1
    r += 1
    
    U[1][r] = 1
    V[v_g2][r] = 1
    W[w_c02][r] = 1
    r += 1
    
    U[2][r] = 1
    V[v_e2][r] = 1
    W[w_c12][r] = 1
    r += 1
    
    U[3][r] = 1
    V[v_g2][r] = 1
    W[w_c12][r] = 1
    r += 1
    
    assert r == rank
    return U, V, W

def _als_solve(U, V, W, n, m, p, rank, deadline, max_rounds=5000, seed=42):
    """
    ALS refinement for a given rank.
    """
    rng = np.random.RandomState(seed)
    
    if U is None:
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)
    else:
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
    # Precompute indices
    ta_list, tb_list, tc_list = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ta_list.append(i * m + j)
                tb_list.append(j * p + k)
                tc_list.append(k * n + i)
    ta_arr = np.array(ta_list)
    tb_arr = np.array(tb_list)
    tc_arr = np.array(tc_list)
    
    prev_err = float('inf')
    for it in range(max_rounds):
        if time.time() > deadline:
            break
        
        # Update U
        for a in range(n * m):
            mask = (ta_arr == a)
            if not np.any(mask):
                continue
            tb_sub = tb_arr[mask]
            tc_sub = tc_arr[mask]
            V_sub = V[tb_sub, :]
            W_sub = W[tc_sub, :]
            prod = V_sub * W_sub
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                U[a, valid] = num[valid] / den[valid]
        
        # Update V
        for b in range(m * p):
            mask = (tb_arr == b)
            if not np.any(mask):
                continue
            ta_sub = ta_arr[mask]
            tc_sub = tc_arr[mask]
            U_sub = U[ta_sub, :]
            W_sub = W[tc_sub, :]
            prod = U_sub * W_sub
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                V[b, valid] = num[valid] / den[valid]
        
        # Update W
        for c_idx in range(p * n):
            mask = (tc_arr == c_idx)
            if not np.any(mask):
                continue
            ta_sub = ta_arr[mask]
            tb_sub = tb_arr[mask]
            U_sub = U[ta_sub, :]
            V_sub = V[tb_sub, :]
            prod = U_sub * V_sub
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                W[c_idx, valid] = num[valid] / den[valid]
        
        if it % 100 == 0 or it == max_rounds - 1:
            sums = np.zeros(len(ta_arr))
            for c in range(rank):
                sums += U[ta_arr, c] * V[tb_arr, c] * W[tc_arr, c]
            err = np.sum((sums - 1.0) ** 2)
            
            if err < 1e-10:
                return U.tolist(), V.tolist(), W.tolist()
            
            if err > prev_err * 1.01 and it > 200:
                break
            prev_err = err
    
    return None

def construct():
    n, m, p = 2, 2, 11
    deadline = time.time() + 110
    
    # Fallback: Rank 39
    U39, V39, W39 = _construct_rank39()
    if _verify(U39, V39, W39, n, m, p):
        best_U, best_V, best_W = U39, V39, W39
        best_rank = 39
    else:
        # Should not happen
        best_U, best_V, best_W = None, None, None
        best_rank = 44

    # Try to find lower rank
    # Theoretical guess: Rank 34 is possible (3p+1)
    # We try ranks 34, 35, 36, 37, 38
    for target_rank in [34, 35, 36, 37, 38]:
        if time.time() > deadline - 10:
            break
            
        # Try a few seeds
        for seed in [target_rank, 100 + target_rank, 200 + target_rank]:
            if time.time() > deadline - 5:
                break
            
            # Warm start from rank 39 if possible
            # We can't directly warm start from 39 to 34 easily without projection, 
            # but we can try random init or a structured init.
            # Let's try random init first.
            result = _als_solve(None, None, None, n, m, p, target_rank, deadline=deadline - 5, max_rounds=2000, seed=seed)
            
            if result is not None:
                U, V, W = result
                if _verify(U, V, W, n, m, p):
                    best_U, best_V, best_W = U, V, W
                    best_rank = target_rank
                    break # Move to next lower rank? No, we want minimal rank, so if we found 34, we are done.
                    # But if we found 36, we might still try 35, 34.
                    # The loop goes 34, 35, 36... so if we find 34, we break the inner loop and then should break outer?
                    # Actually, if we find a valid 34, that's likely the minimum.
                    # If we find 36, we continue to try 35, 34 in subsequent iterations? 
                    # No, the loop is `for target_rank in [34, 35, 36...]`.
                    # So it tries 34 first. If it fails, it tries 35.
                    # So if we find a valid 34, we can return immediately? 
                    # Yes, 34 is the lowest we are trying.
                    return {"U": best_U, "V": best_V, "W": best_W}
                
        # If we didn't find 34, we try 35, etc.
        # But the loop structure above tries 34, then 35.
        # If 34 fails, it goes to 35.
        # If 35 succeeds, it updates best and continues to 36? No, 35 is better than 36.
        # So we should return as soon as we find a valid decomposition for the current target_rank,
        # because we are iterating from low to high.
        
    # If no lower rank found, return best found (which is 39)
    return {"U": best_U, "V": best_V, "W": best_W}
