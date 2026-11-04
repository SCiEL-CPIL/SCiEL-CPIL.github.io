import time
import numpy as np
import random

DIMENSIONS = [4, 4, 9]

def _build_schoolbook(n, m, p):
    """Guaranteed rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _build_rank114():
    """Fallback: the current Strassen-based rank-114 decomposition."""
    # Re-implement the existing rank-114 construction
    _STRASSEN_222 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        ([0, 0, 0, 1], [0, -1, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    def _generate_strassen_444():
        terms = []
        for u1, v1, w1 in _STRASSEN_222:
            for u2, v2, w2 in _STRASSEN_222:
                u = [0]*16
                v = [0]*16
                w = [0]*16
                for i1 in range(4):
                    r1 = i1 // 2
                    c1 = i1 % 2
                    for i2 in range(4):
                        r2 = i2 // 2
                        c2 = i2 % 2
                        i4 = (r1*2 + r2)*4 + (c1*2 + c2)
                        u[i4] = u1[i1] * u2[i2]
                        v[i4] = v1[i1] * v2[i2]
                        w[i4] = w1[i1] * w2[i2]
                terms.append((u, v, w))
        return terms
    
    _STRASSEN_444 = _generate_strassen_444()
    
    def _add_444(U, V, W, col, a_base, b_base, c_base):
        for u, v, w in _STRASSEN_444:
            for t in range(16):
                if u[t]:
                    U[a_base + t][col] += u[t]
                if v[t]:
                    V[b_base + t][col] += v[t]
                if w[t]:
                    W[c_base + t][col] += w[t]
            col += 1
        return col
    
    def _add_mv_block(U, V, W, col, a_base, b_base, c_base, ar, ac, bc):
        for i in range(ar):
            for j in range(ac):
                U[a_base + i * ac + j][col] += 1
                V[b_base + j * bc][col] += 1
                W[c_base + i * bc][col] += 1
                col += 1
        return col
    
    n, m, p = 4, 4, 9
    rank = 114
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = _add_444(U, V, W, 0, 0, 0, 0)
    col = _add_444(U, V, W, col, 0, 4, 4)
    col = _add_mv_block(U, V, W, col, 0, 8, 8, 4, 4, 1)
    if col != rank:
        rank = col
        U = [row[:rank] for row in U]
        V = [row[:rank] for row in V]
        W = [row[:rank] for row in W]
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    """Return True iff the decomposition is exact."""
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    T = np.einsum('ar,br,cr->abc', Ua, Va, Wa)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i * m + j, j * p + k, k * n + i] - 1.0) > 1e-9:
                    return False
    T2 = T.copy()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T2[i * m + j, j * p + k, k * n + i] = 0.0
    if np.max(np.abs(T2)) > 1e-9:
        return False
    return True

def _random_sparse_search(n, m, p, rank, num_trials, deadline):
    """
    Random sparse support search for rank-r decomposition.
    For each trial, pick random sparse supports for U, V, W, then solve the linear system.
    """
    n, m, p = n, m, p
    dim_u = n * m  # 16
    dim_v = m * p  # 36
    dim_w = p * n  # 36
    
    # The tensor has n*m*p = 144 non-zero entries (all 1s at positions (i*m+j, j*p+k, k*n+i))
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    # And = 0 for all other (a,b,c) positions
    
    # Total number of (a,b,c) positions = dim_u * dim_v * dim_w = 16 * 36 * 36 = 20736
    # But we only care about positions where the tensor is defined.
    # Actually, the tensor T[a,b,c] must equal 1 if (a,b,c) = (i*m+j, j*p+k, k*n+i) for some (i,j,k),
    # and 0 otherwise.
    
    # Let's build the linear system. For each rank term r, we have variables:
    # u_r[a] for a in support_U[r], v_r[b] for b in support_V[r], w_r[c] for c in support_W[r]
    # The constraint is: for each (a,b,c) in the full space,
    #   sum_r u_r[a] * v_r[b] * w_r[c] = T[a,b,c]
    
    # This is a system of 20736 equations. But most T[a,b,c] = 0.
    # The non-zero constraints are 144. The zero constraints are 20736 - 144 = 20592.
    
    # For a given sparse support pattern, we can set up a linear system.
    # Let's denote the support for term r as S_U[r], S_V[r], S_W[r].
    # The unknowns are the values at those support positions.
    # Total unknowns = sum_r (|S_U[r]| + |S_V[r]| + |S_W[r]|)
    
    # For the linear system to be solvable, we need:
    # For each (a,b,c) where T[a,b,c] = 1: sum_r u_r[a]*v_r[b]*w_r[c] = 1
    # For each (a,b,c) where T[a,b,c] = 0: sum_r u_r[a]*v_r[b]*w_r[c] = 0
    
    # This is still a nonlinear system in general. But if we fix the supports and treat
    # the products as variables, it's not linear.
    
    # Alternative: fix the supports, and for each term r, the contribution to (a,b,c) is
    # u_r[a]*v_r[b]*w_r[c]. If we fix u and v, then w is determined linearly, and vice versa.
    
    # Better approach: fix supports for U and V, then solve for W linearly.
    # For each c, and for each (a,b): sum_r u_r[a]*v_r[b]*w_r[c] = T[a,b,c]
    # This is a linear system in w_r[c] for each (a,b,c).
    
    # Let's fix sparse supports for U and V, and solve for W.
    # For each c in range(dim_w), we have a linear system:
    #   For each (a,b) in range(dim_u) x range(dim_v):
    #     sum_{r: a in S_U[r] and b in S_V[r]} (u_r[a]*v_r[b]) * w_r[c] = T[a,b,c]
    
    # But we also need to solve for u_r[a] and v_r[b].
    
    # Simpler: fix ALL supports, and use iterative refinement or solve a bilinear system.
    # Actually, let's fix supports for U and V, and solve for W via least squares,
    # then check if it's exact.
    
    # Even simpler: fix supports for all three, and use a greedy/iterative approach.
    
    # Let's try: fix supports for U and V randomly, solve for W linearly.
    # For each c, the system is: A_c * w_c = b_c where
    # A_c[r, (a,b)] = u_r[a] * v_r[b] if a in S_U[r] and b in S_V[r], else 0
    # b_c[(a,b)] = T[a,b,c]
    # w_c[r] = w_r[c]
    
    # But we need to also determine u_r[a] and v_r[b].
    
    # Let's fix u_r[a] = 1 for all a in S_U[r], and v_r[b] = 1 for all b in S_V[r].
    # Then solve for w_r[c].
    
    # This gives a linear system for each c.
    
    # Number of unknowns per c: sum_r |S_W[r]| (but we only solve for w_r[c] for r where c in S_W[r])
    # Actually, for each c, the unknowns are w_r[c] for all r such that c in S_W[r].
    # The equations are for all (a,b) in dim_u x dim_v.
    
    # Let's build this.
    
    # First, generate random sparse supports.
    # Each term r has a support in U, V, W.
    # Let's say each term has k_u non-zeros in U, k_v in V, k_w in W.
    
    best_result = None
    best_rank = rank
    
    # Try different sparsity levels
    for k_u in [1, 2, 3]:
        for k_v in [1, 2, 3]:
            for k_w in [1, 2, 3]:
                if time.time() > deadline:
                    return best_result
                
                for trial in range(num_trials):
                    if time.time() > deadline:
                        return best_result
                    
                    # Generate random supports
                    U_support = []
                    V_support = []
                    W_support = []
                    for r in range(rank):
                        su = random.sample(range(dim_u), k_u)
                        sv = random.sample(range(dim_v), k_v)
                        sw = random.sample(range(dim_w), k_w)
                        U_support.append(su)
                        V_support.append(sv)
                        W_support.append(sw)
                    
                    # Set u_r[a] = 1 for a in U_support[r]
                    # Set v_r[b] = 1 for b in V_support[r]
                    # Solve for w_r[c]
                    
                    # For each c, build the linear system
                    # Unknowns: w_r[c] for r in 0..rank-1 (but only if c in W_support[r])
                    # Actually, let's just solve for all w_r[c] for all r, and set w_r[c] = 0 if c not in W_support[r]
                    
                    # Equations: for each (a, b) in range(dim_u) x range(dim_v):
                    #   sum_r (u_r[a] * v_r[b]) * w_r[c] = T[a, b, c]
                    # where u_r[a] = 1 if a in U_support[r] else 0
                    # and v_r[b] = 1 if b in V_support[r] else 0
                    
                    # So the coefficient for w_r[c] in equation (a,b) is:
                    #   1 if a in U_support[r] and b in V_support[r], else 0
                    
                    # Let's build the matrix A of shape (dim_u * dim_v, rank)
                    # A[(a,b), r] = 1 if a in U_support[r] and b in V_support[r] else 0
                    
                    # And b_vec of shape (dim_u * dim_v,)
                    # b_vec[(a,b)] = T[a, b, c]
                    
                    # T[a, b, c] = 1 if there exists (i,j) such that a = i*m+j, b = j*p+c, and
                    # the corresponding output index is consistent.
                    # Actually, T[a, b, c] = 1 iff a = i*m+j, b = j*p+c for some i,j, and
                    # c = k*n+i for some k. Since c is fixed, we need k = c//n and i = c%n,
                    # and then j = b//p, and we need a = i*m + j.
                    
                    # Let's precompute T as a 3D array.
                    T = np.zeros((dim_u, dim_v, dim_w))
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                a = i * m + j
                                b = j * p + k
                                c_idx = k * n + i
                                T[a, b, c_idx] = 1.0
                    
                    # Now solve for each c
                    W_vals = np.zeros((dim_w, rank))
                    for c in range(dim_w):
                        # Build A of shape (dim_u * dim_v, rank)
                        A = np.zeros((dim_u * dim_v, rank))
                        for r in range(rank):
                            su = set(U_support[r])
                            sv = set(V_support[r])
                            for a in su:
                                for b in sv:
                                    A[a * dim_v + b, r] = 1.0
                        
                        b_vec = T[:, :, c].flatten()
                        
                        # Solve A * w_c = b_vec using least squares
                        try:
                            w_c, residuals, rank_s, sv = np.linalg.lstsq(A, b_vec, rcond=None)
                        except:
                            continue
                        
                        W_vals[c, :] = w_c
                    
                    # Check if the solution is exact
                    # Reconstruct and verify
                    U_mat = np.zeros((dim_u, rank))
                    V_mat = np.zeros((dim_v, rank))
                    for r in range(rank):
                        for a in U_support[r]:
                            U_mat[a, r] = 1.0
                        for b in V_support[r]:
                            V_mat[b, r] = 1.0
                    
                    # Check
                    T_check = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_vals.T)
                    max_err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                a = i * m + j
                                b = j * p + k
                                c_idx = k * n + i
                                max_err = max(max_err, abs(T_check[a, b, c_idx] - 1.0))
                    # Check zeros
                    T_zero = T_check.copy()
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                a = i * m + j
                                b = j * p + k
                                c_idx = k * n + i
                                T_zero[a, b, c_idx] = 0.0
                    max_zero_err = np.max(np.abs(T_zero))
                    
                    if max_err < 1e-9 and max_zero_err < 1e-9:
                        # Found a valid decomposition!
                        # Convert to list format
                        U_list = [[int(round(U_mat[i, r] * 2)) / 2 for r in range(rank)] for i in range(dim_u)]
                        V_list = [[int(round(V_mat[i, r] * 2)) / 2 for r in range(rank)] for i in range(dim_v)]
                        W_list = [[int(round(W_vals[i, r] * 2)) / 2 for r in range(rank)] for i in range(dim_w)]
                        
                        # Verify once more
                        if _verify(U_list, V_list, W_list, n, m, p):
                            return {"U": U_list, "V": V_list, "W": W_list}
    
    return best_result

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130.0  # Safety margin
    
    # Start with the known rank-114 as fallback
    fallback = _build_rank114()
    
    # Try random sparse search at various ranks below 114
    # The theoretical lower bound for 4x4x9 is at least max(4*4, 4*9, 9*4) = 36
    # But in practice, it's likely higher. Let's try ranks from 40 to 113.
    
    # Try a few ranks with limited trials
    ranks_to_try = [40, 50, 60, 70, 80, 90, 100, 110]
    trials_per_rank = 50
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
        result = _random_sparse_search(n, m, p, rank, trials_per_rank, deadline)
        if result is not None:
            return result
    
    return fallback
