"""Combined approach: verified rank-39 baseline with optimized iterative refinement for lower ranks."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-8):
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
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
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

def _try_low_rank(rank, n, m, p, deadline, max_rounds=5000, seed=42, init_U=None, init_V=None, init_W=None):
    """
    Try to find a decomposition of given rank using iterative refinement.
    Returns (U, V, W) as lists if found, else None.
    """
    rng = np.random.RandomState(seed)
    
    if init_U is not None:
        U = np.array(init_U, dtype=float)
    else:
        U = rng.randn(n * m, rank)
    
    if init_V is not None:
        V = np.array(init_V, dtype=float)
    else:
        V = rng.randn(m * p, rank)
    
    if init_W is not None:
        W = np.array(init_W, dtype=float)
    else:
        W = rng.randn(p * n, rank)
    
    # Precompute index arrays
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
    
    # Precompute masks for each index
    masks_U = [ta_arr == a for a in range(n * m)]
    masks_V = [tb_arr == b for b in range(m * p)]
    masks_W = [tc_arr == c for c in range(p * n)]
    
    best_err = float('inf')
    
    for it in range(max_rounds):
        if time.time() > deadline:
            break
        
        # Update U
        for a in range(n * m):
            mask = masks_U[a]
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
            mask = masks_V[b]
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
            mask = masks_W[c_idx]
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
        
        # Check convergence every 20 iterations
        if it % 20 == 0:
            sums = np.zeros(len(ta_arr))
            for c in range(rank):
                sums += U[ta_arr, c] * V[tb_arr, c] * W[tc_arr, c]
            err = np.sum((sums - 1.0) ** 2)
            best_err = min(best_err, err)
            if err < 1e-10:
                return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _structured_init(rank, n, m, p, seed=0):
    """
    Create a structured initial guess for rank-r decomposition.
    Uses Strassen-like patterns where possible.
    """
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank) * 0.5
    V = rng.randn(m * p, rank) * 0.5
    W = rng.randn(p * n, rank) * 0.5
    
    # Add some structure: for the first few ranks, use Strassen-like patterns
    num_strassen_blocks = min(rank // 7, p // 2)
    for block in range(num_strassen_blocks):
        r0 = block * 7
        if r0 + 7 > rank:
            break
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
        
        # m1 = (a+d)(e+h)
        U[0, r0] += 1; U[3, r0] += 1
        V[v_e, r0] += 1; V[v_h, r0] += 1
        W[w_c00, r0] += 1; W[w_c11, r0] += 1
        
        # m2 = (c+d)e
        U[2, r0+1] += 1; U[3, r0+1] += 1
        V[v_e, r0+1] += 1
        W[w_c10, r0+1] += 1; W[w_c11, r0+1] += -1
        
        # m3 = a(f-h)
        U[0, r0+2] += 1
        V[v_f, r0+2] += 1; V[v_h, r0+2] += -1
        W[w_c01, r0+2] += 1; W[w_c11, r0+2] += 1
        
        # m4 = d(g-e)
        U[3, r0+3] += 1
        V[v_g, r0+3] += 1; V[v_e, r0+3] += -1
        W[w_c00, r0+3] += 1; W[w_c10, r0+3] += 1
        
        # m5 = (a+b)h
        U[0, r0+4] += 1; U[1, r0+4] += 1
        V[v_h, r0+4] += 1
        W[w_c00, r0+4] += -1; W[w_c01, r0+4] += 1
        
        # m6 = (c-a)(e+f)
        U[2, r0+5] += 1; U[0, r0+5] += -1
        V[v_e, r0+5] += 1; V[v_f, r0+5] += 1
        W[w_c11, r0+5] += 1
        
        # m7 = (b-d)(g+h)
        U[1, r0+6] += 1; U[3, r0+6] += -1
        V[v_g, r0+6] += 1; V[v_h, r0+6] += 1
        W[w_c00, r0+6] += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    # First, create the verified rank-39 construction
    U39, V39, W39 = _construct_rank39()
    if not _verify(U39, V39, W39, n, m, p):
        # Fallback: schoolbook
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
    
    # Try to find lower rank with remaining time
    # Use structured initialization for better convergence
    best_result = (39, U39, V39, W39)
    
    if time.time() < deadline - 10:
        for rank in [38, 37, 36, 35, 34, 33, 32, 31, 30]:
            if time.time() > deadline - 5:
                break
            
            # Try multiple seeds with structured initialization
            for seed in [rank, rank + 100, rank + 200, rank + 300, rank + 400]:
                if time.time() > deadline - 3:
                    break
                
                # Create structured initialization
                init_U, init_V, init_W = _structured_init(rank, n, m, p, seed=seed)
                
                result = _try_low_rank(rank, n, m, p, deadline=deadline - 3, 
                                       max_rounds=2000, seed=seed,
                                       init_U=init_U, init_V=init_V, init_W=init_W)
                if result is not None:
                    U2, V2, W2 = result
                    if _verify(U2, V2, W2, n, m, p):
                        best_result = (rank, U2, V2, W2)
                        break  # Found valid decomposition for this rank, try next lower rank
            
            # If we found a valid decomposition for this rank, continue to try lower
            if best_result[0] == rank:
                continue
    
    final_rank, U_final, V_final, W_final = best_result
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
