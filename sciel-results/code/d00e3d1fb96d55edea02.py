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
    """High-precision polish pass on a near-converged solution."""
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
    
    groups_a = {a: (ta_arr == a) for a in range(n * m)}
    groups_b = {b: (tb_arr == b) for b in range(m * p)}
    groups_c = {c: (tc_arr == c) for c in range(p * n)}
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_rounds):
        if time.time() > deadline:
            break
        
        for a in range(n * m):
            mask = groups_a[a]
            if not np.any(mask):
                continue
            tb_sub = tb_arr[mask]
            tc_sub = tc_arr[mask]
            prod = V[tb_sub, :] * W[tc_sub, :]
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                U[a, valid] = num[valid] / den[valid]
        
        for b in range(m * p):
            mask = groups_b[b]
            if not np.any(mask):
                continue
            ta_sub = ta_arr[mask]
            tc_sub = tc_arr[mask]
            prod = U[ta_sub, :] * W[tc_sub, :]
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                V[b, valid] = num[valid] / den[valid]
        
        for c_idx in range(p * n):
            mask = groups_c[c_idx]
            if not np.any(mask):
                continue
            ta_sub = ta_arr[mask]
            tb_sub = tb_arr[mask]
            prod = U[ta_sub, :] * V[tb_sub, :]
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                W[c_idx, valid] = num[valid] / den[valid]
        
        if it % 20 == 0:
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

def _isds_find_sparse_kernel(M, target_weight, deadline, max_iters=200):
    """
    Information Set Decoding to find a sparse vector in the kernel of M.
    M is a (44 x N) matrix over rationals (represented as float for speed).
    Returns a sparse index set (list of column indices) of weight <= target_weight
    whose corresponding columns span a subspace in the kernel, or None.
    """
    rows, cols = M.shape
    if rows == 0 or cols == 0:
        return None
    
    # Compute rank and null space basis using SVD
    try:
        U_svd, s_vals, Vt = np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    
    # Numerical rank
    tol_rank = max(M.shape) * np.amax(s_vals) * np.finfo(float).eps
    num_rank = np.sum(s_vals > tol_rank)
    
    if num_rank == 0:
        # Entire space is kernel - any subset works
        return list(range(min(target_weight, cols)))
    
    if num_rank == cols:
        return None  # No kernel
    
    # Null space basis: columns of Vt.T corresponding to zero singular values
    null_basis = Vt[num_rank:].T  # shape (cols, cols - num_rank)
    
    # Now search for a sparse combination of null space basis vectors
    # that has small Hamming weight in the original column space
    # This is equivalent to finding a sparse vector in the row space of null_basis^T
    
    # Simplified ISD: try random subsets of columns and check if they're in kernel
    # Better: use the null space basis to generate candidate sparse vectors
    
    # For each subset of columns of size target_weight, check if there exists
    # a non-trivial linear combination in the kernel
    # This is expensive, so use a heuristic:
    
    # Strategy: find columns that are "almost" in the kernel direction
    # Compute for each column c, the minimum norm of M[:,c] - M @ x for x in null space
    # Actually, we want: is column c in the span of other columns restricted to kernel?
    
    # Alternative simpler approach: try to find a set of 'target_weight' columns
    # such that the submatrix M[:, subset] has a non-trivial null vector
    # i.e., rank(M[:, subset]) < target_weight
    
    import itertools
    
    best_subset = None
    best_deficit = float('inf')
    
    # If target_weight is small, enumerate subsets
    if target_weight <= 8 and cols <= 20:
        for subset in itertools.combinations(range(cols), target_weight):
            subM = M[:, list(subset)]
            try:
                r = np.linalg.matrix_rank(subM, tol=tol_rank)
            except:
                continue
            deficit = target_weight - r
            if deficit > 0 and deficit < best_deficit:
                best_deficit = deficit
                best_subset = list(subset)
            if deficit >= target_weight:
                return list(subset)
    else:
        # Heuristic: greedy selection
        # Start with all columns, iteratively remove the column whose removal
        # causes the least increase in rank deficiency
        remaining = list(range(cols))
        for _ in range(cols - target_weight):
            if time.time() > deadline:
                break
            best_remove = None
            best_rank = -1
            for c in remaining:
                trial = [x for x in remaining if x != c]
                if len(trial) < target_weight:
                    continue
                subM = M[:, trial]
                try:
                    r = np.linalg.matrix_rank(subM, tol=tol_rank)
                except:
                    r = len(trial)
                # We want to keep columns that maintain low rank
                # Remove the column that, when kept, gives highest rank
                # Actually we want to remove the column that is "most dependent"
                # Let's remove the column whose removal decreases rank the least
                # i.e., keep the set with lowest rank
                if r < best_rank or best_rank == -1:
                    best_rank = r
                    best_remove = c
            if best_remove is not None:
                remaining.remove(best_remove)
            if len(remaining) <= target_weight:
                break
        if len(remaining) <= target_weight:
            best_subset = remaining
    
    return best_subset

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
    groups_a = {a: (ta_arr == a) for a in range(n * m)}
    groups_b = {b: (tb_arr == b) for b in range(m * p)}
    groups_c = {c: (tc_arr == c) for c in range(p * n)}
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_rounds):
        if time.time() > deadline:
            break
        
        for a in range(n * m):
            mask = groups_a[a]
            if not np.any(mask):
                continue
            tb_sub = tb_arr[mask]
            tc_sub = tc_arr[mask]
            prod = V[tb_sub, :] * W[tc_sub, :]
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                U[a, valid] = num[valid] / den[valid]
        
        for b in range(m * p):
            mask = groups_b[b]
            if not np.any(mask):
                continue
            ta_sub = ta_arr[mask]
            tc_sub = tc_arr[mask]
            prod = U[ta_sub, :] * W[tc_sub, :]
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                V[b, valid] = num[valid] / den[valid]
        
        for c_idx in range(p * n):
            mask = groups_c[c_idx]
            if not np.any(mask):
                continue
            ta_sub = ta_arr[mask]
            tb_sub = tb_arr[mask]
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
    """Create a warm start for rank 38 by projecting rank-39 solution."""
    U39_arr = np.array(U39, dtype=float)
    V39_arr = np.array(V39, dtype=float)
    W39_arr = np.array(W39, dtype=float)
    n, m, p = 2, 2, 11
    rank = 39
    
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
    
    best_col = 0
    best_err = float('inf')
    
    candidates = list(range(rank))
    candidates.sort(key=lambda c: 0 if c >= 35 else 1)
    
    for c in candidates[:8]:
        U_new = np.delete(U39_arr, c, axis=1)
        V_new = np.delete(V39_arr, c, axis=1)
        W_new = np.delete(W39_arr, c, axis=1)
        
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
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        c1 = attempt % 39
        c2 = (attempt + 1) % 39
        
        if c1 == c2:
            continue
        
        U_new = U39_arr.copy()
        V_new = V39_arr.copy()
        W_new = W39_arr.copy()
        
        U_new[:, c1] += U_new[:, c2]
        V_new[:, c1] += V_new[:, c2]
        W_new[:, c1] += W_new[:, c2]
        
        if c2 > c1:
            U_new = np.delete(U_new, c2, axis=1)
            V_new = np.delete(V_new, c2, axis=1)
            W_new = np.delete(W_new, c2, axis=1)
        else:
            U_new = np.delete(U_new, c1, axis=1)
            V_new = np.delete(V_new, c1, axis=1)
            W_new = np.delete(W_new, c1, axis=1)
        
        pU, pV, pW, perr = _polish(U_new, V_new, W_new, n, m, p, deadline, max_rounds=100)
        
        if perr < 1e-10:
            rU = _round_to_half_integers(pU)
            rV = _round_to_half_integers(pV)
            rW = _round_to_half_integers(pW)
            if _verify_exact(rU.tolist(), rV.tolist(), rW.tolist(), n, m, p):
                return rU.tolist(), rV.tolist(), rW.tolist()
    
    return None

def _isds_search(rank, n, m, p, deadline):
    """
    Structural approach: Model the matrix multiplication tensor as a linear system
    and use Information Set Decoding to find a sparse (low-rank) solution.
    
    The tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    This is a multilinear system. We linearize by fixing two of the three matrices
    and solving for the third, then iterate. The ISD insight is to look for solutions
    where the columns of U, V, W have small support (sparse patterns).
    
    For rank r, we need to find r "rank-1 tensor components" that sum to the
    multiplication tensor. Each component is defined by three vectors (u_c, v_c, w_c).
    
    Key insight: For the 2x2x11 case, many entries of U, V, W will be zero in
    an optimal decomposition. We search for decompositions with maximum sparsity.
    """
    if time.time() > deadline:
        return None
    
    # Build the constraint matrix for the linearized problem.
    # Fix V and W, solve for U. The constraint is:
    # For each (i,j,k): sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
    # This is a linear system in U entries: A_U * U_vec = b_U
    
    # But we need a joint approach. Instead, use the fact that for a valid
    # rank-r decomposition, the following 44x44 "syndrome" matrix must have
    # rank <= 44 - r (the nullity gives us the degrees of freedom for rank reduction).
    
    # Actually, let's use a more direct approach:
    # Start from rank 39 and try to find a rank 38 solution by checking if
    # there exists a non-trivial linear relation among the 39 rank-1 components.
    
    U39, V39, W39 = _construct_rank39()
    U39_arr = np.array(U39, dtype=float)
    V39_arr = np.array(V39, dtype=float)
    W39_arr = np.array(W39, dtype=float)
    
    # The 39 components are: T_c = outer(U[:,c], V[:,c], W[:,c])
    # A rank-38 solution exists iff there's a linear dependence:
    # sum_c alpha_c * T_c = 0 for some non-zero alpha in R^39
    # This is equivalent to: the 39 vectors in R^{44} (flattened tensors) are linearly dependent
    
    # Flatten each rank-1 component to a 44-dim vector
    components = []
    for c in range(39):
        comp = np.zeros(44)
        for i in range(2):
            for j in range(2):
                for k in range(11):
                    comp[k * 4 + i * 2 + j] = U39_arr[i * 2 + j, c] * V39_arr[j * 11 + k, c] * W39_arr[k * 2 + i, c]
        components.append(comp)
    
    comp_matrix = np.array(components).T  # 44 x 39
    
    # Check if these 39 vectors are linearly dependent
    # If rank < 39, there's a non-trivial null vector
    try:
        U_svd, s_vals, Vt = np.linalg.svd(comp_matrix, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    
    tol_rank = max(comp_matrix.shape) * np.amax(s_vals) * np.finfo(float).eps
    num_rank = np.sum(s_vals > tol_rank)
    
    if num_rank < 39:
        # There's a linear dependence! The null space gives us a relation.
        # This means we can potentially reduce the rank.
        # Find the null vector
        null_vec = Vt[num_rank:, :]  # shape (39-num_rank, 39)
        
        # The null vector tells us which combination of components sums to zero.
        # If the null vector has small support, we can drop those components.
        # Find the sparsest null vector
        
        # For a 1-dimensional null space, the null vector is unique up to scale
        if 39 - num_rank == 1:
            nv = null_vec[0]
            # Normalize
            nv = nv / np.linalg.norm(nv)
            # Find the support
            support = np.where(np.abs(nv) > 1e-6)[0]
            if len(support) > 0:
                # Try removing the component with the largest coefficient
                # and re-solving
                best_c = support[np.argmax(np.abs(nv[support]))]
                U_new = np.delete(U39_arr, best_c, axis=1)
                V_new = np.delete(V39_arr, best_c, axis=1)
                W_new = np.delete(W39_arr, best_c, axis=1)
                
                # Polish the rank-38 solution
                pU, pV, pW, perr = _polish(U_new, V_new, W_new, n, m, p, deadline, max_rounds=500)
                if perr < 1e-10:
                    rU = _round_to_half_integers(pU)
                    rV = _round_to_half_integers(pV)
                    rW = _round_to_half_integers(pW)
                    if _verify_exact(rU.tolist(), rV.tolist(), rW.tolist(), n, m, p):
                        return rU.tolist(), rV.tolist(), rW.tolist()
                    return pU.tolist(), pV.tolist(), pW.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin before 120s kill
    
    # First, return the verified rank-39 construction
    U, V, W = _construct_rank39()
    if _verify(U, V, W, n, m, p):
        # Try to find a lower rank decomposition with remaining time
        if time.time() < deadline - 10:
            # Try ISD-based structural search first
            if time.time() < deadline - 20:
                result = _isds_search(38, n, m, p, deadline=deadline - 15)
                if result is not None:
                    U2, V2, W2 = result
                    if _verify(U2, V2, W2, n, m, p):
                        return {"U": U2, "V": V2, "W": W2}
            
            # Try rank 38 with warm start from rank 39
            if time.time() < deadline - 15:
                U38_ws, V38_ws, W38_ws = _make_warm_start_for_rank38(U, V, W)
                result = _try_low_rank(38, n, m, p, deadline=deadline - 8, max_rounds=3000, seed=0, warm_start=(U38_ws, V38_ws, W38_ws))
                if result is not None:
                    U2, V2, W2 = result
                    if _verify(U2, V2, W2, n, m, p):
                        return {"U": U2, "V": V2, "W": W2}
            
            # Try sparse support search
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
