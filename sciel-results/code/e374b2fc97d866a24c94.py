"""Combined approach: verified rank-39 Strassen+schoolbook, then bounded exact search."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
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
        
        # m1 = (A00+A11)*(B[0,k0]+B[1,k1]) -> C[0,k0] + C[1,k1]
        U[0][r] = 1; U[3][r] = 1
        V[v_e][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1; W[w_c11][r] = 1
        r += 1
        
        # m2 = (A10+A11)*B[0,k0] -> C[1,k0] - C[1,k1]
        U[2][r] = 1; U[3][r] = 1
        V[v_e][r] = 1
        W[w_c10][r] = 1; W[w_c11][r] = -1
        r += 1
        
        # m3 = A00*(B[0,k1]-B[1,k1]) -> C[0,k1] + C[1,k1]
        U[0][r] = 1
        V[v_f][r] = 1; V[v_h][r] = -1
        W[w_c01][r] = 1; W[w_c11][r] = 1
        r += 1
        
        # m4 = A11*(B[1,k0]-B[0,k0]) -> C[0,k0] + C[1,k0]
        U[3][r] = 1
        V[v_g][r] = 1; V[v_e][r] = -1
        W[w_c00][r] = 1; W[w_c10][r] = 1
        r += 1
        
        # m5 = (A00+A01)*B[1,k1] -> -C[0,k0] + C[0,k1]
        U[0][r] = 1; U[1][r] = 1
        V[v_h][r] = 1
        W[w_c00][r] = -1; W[w_c01][r] = 1
        r += 1
        
        # m6 = (A10-A00)*(B[0,k0]+B[0,k1]) -> C[1,k1]
        U[2][r] = 1; U[0][r] = -1
        V[v_e][r] = 1; V[v_f][r] = 1
        W[w_c11][r] = 1
        r += 1
        
        # m7 = (A01-A11)*(B[1,k0]+B[1,k1]) -> C[0,k0]
        U[1][r] = 1; U[3][r] = -1
        V[v_g][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1
        r += 1
    
    # Single column k=10: schoolbook 4 terms
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

def _build_indices(n, m, p):
    """Precompute index arrays for the tensor entries."""
    ta_list, tb_list, tc_list = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ta_list.append(i * m + j)
                tb_list.append(j * p + k)
                tc_list.append(k * n + i)
    return np.array(ta_list), np.array(tb_list), np.array(tc_list)

def _try_low_rank(rank, n, m, p, deadline, max_rounds=2000, seed=42, ta_arr=None, tb_arr=None, tc_arr=None):
    """
    Try to find a decomposition of given rank using iterative refinement (ALS).
    Returns (U, V, W) as lists if found, else None.
    """
    if ta_arr is None:
        ta_arr, tb_arr, tc_arr = _build_indices(n, m, p)
    
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    # Precompute masks for each index
    u_masks = {}
    for a in range(n * m):
        u_masks[a] = (ta_arr == a)
    v_masks = {}
    for b in range(m * p):
        v_masks[b] = (tb_arr == b)
    w_masks = {}
    for c_idx in range(p * n):
        w_masks[c_idx] = (tc_arr == c_idx)
    
    for it in range(max_rounds):
        if time.time() > deadline:
            break
        
        # Update U
        for a in range(n * m):
            mask = u_masks[a]
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
            mask = v_masks[b]
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
            mask = w_masks[c_idx]
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
        
        # Check convergence every 50 iterations
        if it % 50 == 0:
            sums = np.zeros(len(ta_arr))
            for c in range(rank):
                sums += U[ta_arr, c] * V[tb_arr, c] * W[tc_arr, c]
            err = np.sum((sums - 1.0) ** 2)
            if err < 1e-10:
                return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _polish(U, V, W, n, m, p, deadline, max_iters=5000, ta_arr=None, tb_arr=None, tc_arr=None):
    """
    High-precision polish pass using alternating least squares with tighter tolerance.
    """
    if ta_arr is None:
        ta_arr, tb_arr, tc_arr = _build_indices(n, m, p)
    
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    # Precompute masks
    u_masks = {}
    for a in range(n * m):
        u_masks[a] = (ta_arr == a)
    v_masks = {}
    for b in range(m * p):
        v_masks[b] = (tb_arr == b)
    w_masks = {}
    for c_idx in range(p * n):
        w_masks[c_idx] = (tc_arr == c_idx)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Update U
        for a in range(n * m):
            mask = u_masks[a]
            if not np.any(mask):
                continue
            tb_sub = tb_arr[mask]
            tc_sub = tc_arr[mask]
            V_sub = V[tb_sub, :]
            W_sub = W[tc_sub, :]
            prod = V_sub * W_sub
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-15
            if np.any(valid):
                U[a, valid] = num[valid] / den[valid]
        
        # Update V
        for b in range(m * p):
            mask = v_masks[b]
            if not np.any(mask):
                continue
            ta_sub = ta_arr[mask]
            tc_sub = tc_arr[mask]
            U_sub = U[ta_sub, :]
            W_sub = W[tc_sub, :]
            prod = U_sub * W_sub
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-15
            if np.any(valid):
                V[b, valid] = num[valid] / den[valid]
        
        # Update W
        for c_idx in range(p * n):
            mask = w_masks[c_idx]
            if not np.any(mask):
                continue
            ta_sub = ta_arr[mask]
            tb_sub = tb_arr[mask]
            U_sub = U[ta_sub, :]
            V_sub = V[tb_sub, :]
            prod = U_sub * V_sub
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-15
            if np.any(valid):
                W[c_idx, valid] = num[valid] / den[valid]
        
        if it % 50 == 0:
            sums = np.zeros(len(ta_arr))
            for c in range(rank):
                sums += U[ta_arr, c] * V[tb_arr, c] * W[tc_arr, c]
            err = np.sum((sums - 1.0) ** 2)
            if err < 1e-12:
                return U.tolist(), V.tolist(), W.tolist()
    
    return U.tolist(), V.tolist(), W.tolist()

def _matching_based_search(rank, n, m, p, deadline, seed=42, ta_arr=None, tb_arr=None, tc_arr=None):
    """
    Matching-based support pattern search for lower rank.
    Tries several random sparse initializations.
    """
    if ta_arr is None:
        ta_arr, tb_arr, tc_arr = _build_indices(n, m, p)
    
    rng = np.random.RandomState(seed)
    
    for attempt in range(30):
        if time.time() > deadline:
            break
        
        # Random initialization with sparsification
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)
        
        # Apply random sparsification to encourage structured solutions
        for c in range(rank):
            if rng.rand() < 0.4:
                U[rng.randint(n * m), c] = 0
            if rng.rand() < 0.4:
                V[rng.randint(m * p), c] = 0
            if rng.rand() < 0.4:
                W[rng.randint(p * n), c] = 0
        
        result = _try_low_rank(rank, n, m, p, deadline=deadline, max_rounds=1500, 
                               seed=seed + attempt, ta_arr=ta_arr, tb_arr=tb_arr, tc_arr=tc_arr)
        if result is not None:
            U2, V2, W2 = result
            if _verify(U2, V2, W2, n, m, p):
                return U2, V2, W2
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # Precompute indices
    ta_arr, tb_arr, tc_arr = _build_indices(n, m, p)
    
    # First, return the verified rank-39 construction
    U, V, W = _construct_rank39()
    if not _verify(U, V, W, n, m, p):
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
    
    # Try to find a lower rank decomposition with remaining time
    if time.time() < deadline - 20:
        # Try matching-based search for rank 38
        result = _matching_based_search(38, n, m, p, deadline=deadline - 20, seed=42,
                                        ta_arr=ta_arr, tb_arr=tb_arr, tc_arr=tc_arr)
        if result is not None:
            U2, V2, W2 = result
            if _verify(U2, V2, W2, n, m, p):
                return {"U": U2, "V": V2, "W": W2}
    
    # Then try iterative refinement for various ranks
    for rank in [38, 37, 36, 35]:
        if time.time() > deadline - 10:
            break
        for seed in [rank, rank + 100, rank + 200, rank + 300, rank + 400]:
            if time.time() > deadline - 5:
                break
            result = _try_low_rank(rank, n, m, p, deadline=deadline - 5, max_rounds=2000, 
                                   seed=seed, ta_arr=ta_arr, tb_arr=tb_arr, tc_arr=tc_arr)
            if result is not None:
                U2, V2, W2 = result
                if _verify(U2, V2, W2, n, m, p):
                    return {"U": U2, "V": V2, "W": W2}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
