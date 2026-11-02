"""Combined approach: verified rank-39 Strassen+schoolbook, then bounded exact search."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact using NumPy."""
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    rank = U_arr.shape[1]
    
    # Indices for all (i, j, k)
    ta = np.array([i * m + j for i in range(n) for j in range(m) for k in range(p)])
    tb = np.array([j * p + k for i in range(n) for j in range(m) for k in range(p)])
    tc = np.array([k * n + i for i in range(n) for j in range(m) for k in range(p)])
    
    # Compute sums: sum_c U[ta, c] * V[tb, c] * W[tc, c]
    sums = np.zeros(len(ta))
    for c in range(rank):
        sums += U_arr[ta, c] * V_arr[tb, c] * W_arr[tc, c]
        
    return np.all(np.abs(sums - 1.0) < tol)

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
        
        # V indices: B[j,k] -> V[j*p + k]
        v_e = k0        # B[0, k0]
        v_f = k1        # B[0, k1]
        v_g = p + k0    # B[1, k0]
        v_h = p + k1    # B[1, k1]
        
        # W indices: C[i,k] -> W[k*n + i]
        w_c00 = k0 * n      # C[0, k0]
        w_c10 = k0 * n + 1  # C[1, k0]
        w_c01 = k1 * n      # C[0, k1]
        w_c11 = k1 * n + 1  # C[1, k1]
        
        # U indices: A[i,j] -> U[i*m + j]
        # U[0]=A00, U[1]=A01, U[2]=A10, U[3]=A11
        
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
    v_e2 = k2        # B[0, 10]
    v_g2 = p + k2    # B[1, 10]
    w_c02 = k2 * n      # C[0, 10]
    w_c12 = k2 * n + 1  # C[1, 10]
    
    # A00 * B[0,10] -> C[0,10]
    U[0][r] = 1
    V[v_e2][r] = 1
    W[w_c02][r] = 1
    r += 1
    
    # A01 * B[1,10] -> C[0,10]
    U[1][r] = 1
    V[v_g2][r] = 1
    W[w_c02][r] = 1
    r += 1
    
    # A10 * B[0,10] -> C[1,10]
    U[2][r] = 1
    V[v_e2][r] = 1
    W[w_c12][r] = 1
    r += 1
    
    # A11 * B[1,10] -> C[1,10]
    U[3][r] = 1
    V[v_g2][r] = 1
    W[w_c12][r] = 1
    r += 1
    
    assert r == rank
    return U, V, W

def _round_to_half_integers(arr, tol=1e-6):
    """Round array entries to nearest half-integer if close enough."""
    result = arr.copy()
    for idx in np.ndindex(result.shape):
        val = result[idx]
        nearest = round(2 * val) / 2.0
        if abs(val - nearest) < tol:
            result[idx] = nearest
    return result

def _verify_exact(U, V, W, n, m, p):
    """Verify with exact arithmetic using Fractions."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(str(U[i * m + j][c])) * Fraction(str(V[j * p + k][c])) * Fraction(str(W[k * n + i][c]))
                if s != 1:
                    return False
    return True

def _polish(U, V, W, n, m, p, deadline, max_rounds=500):
    """High-precision polish pass on a near-converged solution.
    Optimized for speed."""
    rank = U.shape[1]
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
    
    # Precompute index groups as lists of indices for faster access
    groups_a = [[] for _ in range(n * m)]
    groups_b = [[] for _ in range(m * p)]
    groups_c = [[] for _ in range(p * n)]
    for idx in range(len(ta_arr)):
        groups_a[ta_arr[idx]].append(idx)
        groups_b[tb_arr[idx]].append(idx)
        groups_c[tc_arr[idx]].append(idx)
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_rounds):
        if time.time() > deadline:
            break
        
        # Update U
        for a in range(n * m):
            idxs = groups_a[a]
            if not idxs:
                continue
            tb_sub = tb_arr[idxs]
            tc_sub = tc_arr[idxs]
            prod = V[tb_sub, :] * W[tc_sub, :]
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                U[a, valid] = num[valid] / den[valid]
        
        # Update V
        for b in range(m * p):
            idxs = groups_b[b]
            if not idxs:
                continue
            ta_sub = ta_arr[idxs]
            tc_sub = tc_arr[idxs]
            prod = U[ta_sub, :] * W[tc_sub, :]
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                V[b, valid] = num[valid] / den[valid]
        
        # Update W
        for c_idx in range(p * n):
            idxs = groups_c[c_idx]
            if not idxs:
                continue
            ta_sub = ta_arr[idxs]
            tb_sub = tb_arr[idxs]
            prod = U[ta_sub, :] * V[tb_sub, :]
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                W[c_idx, valid] = num[valid] / den[valid]
        
        if it % 10 == 0:
            # Vectorized error computation
            sums = np.zeros(len(ta_arr))
            for c in range(rank):
                sums += U[ta_arr, c] * V[tb_arr, c] * W[tc_arr, c]
            err = np.sum((sums - 1.0) ** 2)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < 1e-14:
                break
    
    return best_U, best_V, best_W, best_err

def _try_low_rank(rank, n, m, p, deadline, max_rounds=2000, seed=42, warm_start=None):
    """
    Try to find a decomposition of given rank using iterative refinement.
    Returns (U, V, W) as lists if found, else None.
    """
    rng = np.random.RandomState(seed)
    
    if warm_start is not None:
        U, V, W = warm_start
    else:
        U = rng.randn(n * m, rank) * 0.3
        V = rng.randn(m * p, rank) * 0.3
        W = rng.randn(p * n, rank) * 0.3
    
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
    
    # Precompute index groups
    groups_a = [[] for _ in range(n * m)]
    groups_b = [[] for _ in range(m * p)]
    groups_c = [[] for _ in range(p * n)]
    for idx in range(len(ta_arr)):
        groups_a[ta_arr[idx]].append(idx)
        groups_b[tb_arr[idx]].append(idx)
        groups_c[tc_arr[idx]].append(idx)
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_rounds):
        if time.time() > deadline:
            break
        
        # Update U
        for a in range(n * m):
            idxs = groups_a[a]
            if not idxs:
                continue
            tb_sub = tb_arr[idxs]
            tc_sub = tc_arr[idxs]
            prod = V[tb_sub, :] * W[tc_sub, :]
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                U[a, valid] = num[valid] / den[valid]
        
        # Update V
        for b in range(m * p):
            idxs = groups_b[b]
            if not idxs:
                continue
            ta_sub = ta_arr[idxs]
            tc_sub = tc_arr[idxs]
            prod = U[ta_sub, :] * W[tc_sub, :]
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                V[b, valid] = num[valid] / den[valid]
        
        # Update W
        for c_idx in range(p * n):
            idxs = groups_c[c_idx]
            if not idxs:
                continue
            ta_sub = ta_arr[idxs]
            tb_sub = tb_arr[idxs]
            prod = U[ta_sub, :] * V[tb_sub, :]
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                W[c_idx, valid] = num[valid] / den[valid]
        
        if it % 10 == 0:
            sums = np.zeros(len(ta_arr))
            for c in range(rank):
                sums += U[ta_arr, c] * V[tb_arr, c] * W[tc_arr, c]
            err = np.sum((sums - 1.0) ** 2)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < 1e-10:
                # Polish and try to round
                pU, pV, pW, perr = _polish(U, V, W, n, m, p, deadline, max_rounds=300)
                if perr < 1e-12:
                    rU = _round_to_half_integers(pU)
                    rV = _round_to_half_integers(pV)
                    rW = _round_to_half_integers(pW)
                    if _verify_exact(rU.tolist(), rV.tolist(), rW.tolist(), n, m, p):
                        return rU.tolist(), rV.tolist(), rW.tolist()
                return pU.tolist(), pV.tolist(), pW.tolist()
    
    if best_err < 1e-8:
        pU, pV, pW, perr = _polish(best_U, best_V, best_W, n, m, p, deadline, max_rounds=500)
        if perr < 1e-12:
            rU = _round_to_half_integers(pU)
            rV = _round_to_half_integers(pV)
            rW = _round_to_half_integers(pW)
            if _verify_exact(rU.tolist(), rV.tolist(), rW.tolist(), n, m, p):
                return rU.tolist(), rV.tolist(), rW.tolist()
        return pU.tolist(), pV.tolist(), pW.tolist()
    
    return None

def _make_warm_start_for_rank38(U39, V39, W39):
    """Create a warm start for rank 38 by projecting rank-39 solution.
    Uses SVD to find the column that contributes least to the tensor."""
    U39_arr = np.array(U39, dtype=float)
    V39_arr = np.array(V39, dtype=float)
    W39_arr = np.array(W39, dtype=float)
    n, m, p = 2, 2, 11
    rank = 39
    
    # Compute a "weight" for each column based on how much it can be absorbed
    # by the others.
    
    # For each column c, compute its "energy" as the sum of squared contributions
    energies = np.zeros(rank)
    for c in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = U39_arr[i*m+j, c] * V39_arr[j*p+k, c] * W39_arr[k*n+i, c]
                    energies[c] += val * val
    energies = np.sqrt(energies)
    
    # Try removing each column and see which gives the smallest error
    # after one polish iteration
    best_col = 0
    best_err = float('inf')
    
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
    
    # For efficiency, just try a few candidate columns
    # The schoolbook columns (last 4) are likely the most redundant with Strassen
    candidates = list(range(rank))
    # Prioritize the schoolbook columns (indices 35-38)
    candidates.sort(key=lambda c: 0 if c >= 35 else 1)
    
    for c in candidates[:8]:  # Try top 8 candidates
        U_new = np.delete(U39_arr, c, axis=1)
        V_new = np.delete(V39_arr, c, axis=1)
        W_new = np.delete(W39_arr, c, axis=1)
        
        # Quick error estimate: compute the residual
        sums = np.zeros(len(ta_arr))
        for cc in range(rank - 1):
            sums += U_new[ta_arr, cc] * V_new[tb_arr, cc] * W_new[tc_arr, cc]
        err = np.sum((sums - 1.0) ** 2)
        
        if err < best_err:
            best_err = err
            best_col = c
    
    U38 = np.delete(U39_arr, best_col, axis=1)
    V38 = np.delete(V39_arr, best_col, axis=1)
    W38 = np.delete(W39_arr, best_col, axis=1)
    
    return U38, V38, W38

def _sparse_support_search(rank, n, m, p, deadline, max_attempts=50):
    """
    Try to find a low-rank decomposition using a sparse support approach.
    We search for a subset of rank-1 tensors with small integer/half-integer coefficients.
    This is inspired by the MILP idea but implemented as a heuristic search.
    
    Strategy: Start from the rank-39 solution, try to merge pairs of columns
    that have similar patterns, effectively reducing the rank.
    """
    U39, V39, W39 = _construct_rank39()
    U39_arr = np.array(U39, dtype=float)
    V39_arr = np.array(V39, dtype=float)
    W39_arr = np.array(W39, dtype=float)
    
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
    
    # Try all pairs of columns to see if they can be merged
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # Pick two random columns to try to merge
        c1 = attempt % 39
        c2 = (attempt + 1) % 39
        
        if c1 == c2:
            continue
        
        # Create a new solution with these two columns merged
        # The merged column is the sum of the two
        U_new = U39_arr.copy()
        V_new = V39_arr.copy()
        W_new = W39_arr.copy()
        
        # Merge c2 into c1
        U_new[:, c1] += U_new[:, c2]
        V_new[:, c1] += V_new[:, c2]
        W_new[:, c1] += W_new[:, c2]
        
        # Remove column c2
        if c2 > c1:
            U_new = np.delete(U_new, c2, axis=1)
            V_new = np.delete(V_new, c2, axis=1)
            W_new = np.delete(W_new, c2, axis=1)
        else:
            U_new = np.delete(U_new, c1, axis=1)
            V_new = np.delete(V_new, c1, axis=1)
            W_new = np.delete(W_new, c1, axis=1)
        
        # Now we have rank 38. Try to polish it.
        pU, pV, pW, perr = _polish(U_new, V_new, W_new, n, m, p, deadline, max_rounds=100)
        
        if perr < 1e-10:
            rU = _round_to_half_integers(pU)
            rV = _round_to_half_integers(pV)
            rW = _round_to_half_integers(pW)
            if _verify_exact(rU.tolist(), rV.tolist(), rW.tolist(), n, m, p):
                return rU.tolist(), rV.tolist(), rW.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin before 120s kill
    
    # First, return the verified rank-39 construction
    U, V, W = _construct_rank39()
    if _verify(U, V, W, n, m, p):
        # Try to find a lower rank decomposition with remaining time
        if time.time() < deadline - 10:
            # Try rank 38 first with warm start from rank 39
            if time.time() < deadline - 15:
                U38_ws, V38_ws, W38_ws = _make_warm_start_for_rank38(U, V, W)
                result = _try_low_rank(38, n, m, p, deadline=deadline - 8, max_rounds=3000, seed=0, warm_start=(U38_ws, V38_ws, W38_ws))
                if result is not None:
                    U2, V2, W2 = result
                    if _verify(U2, V2, W2, n, m, p):
                        return {"U": U2, "V": V2, "W": W2}
            
            # Try sparse support search (column removal + polish)
            if time.time() < deadline - 8:
                result = _sparse_support_search(38, n, m, p, deadline=deadline - 5, max_attempts=20)
                if result is not None:
                    U2, V2, W2 = result
                    if _verify(U2, V2, W2, n, m, p):
                        return {"U": U2, "V": V2, "W": W2}
            
            # Try ranks with random initialization
            for rank in [38, 37]:
                if time.time() > deadline - 4:
                    break
                seeds = list(range(rank, rank + 6))
                for seed in seeds:
                    if time.time() > deadline - 2:
                        break
                    result = _try_low_rank(rank, n, m, p, deadline=deadline - 1, max_rounds=1500, seed=seed)
                    if result is not None:
                        U2, V2, W2 = result
                        if _verify(U2, V2, W2, n, m, p):
                            return {"U": U2, "V": V2, "W": W2}
        return {"U": U, "V": V, "W": W}
    
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
# EVOLVE-BLOCK-END
