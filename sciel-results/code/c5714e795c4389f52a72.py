"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140  # safety margin before 150s kill

    # Build the rank-28 Strassen decomposition
    U28, V28, W28 = _strassen_28(n, m, p)
    
    # Try to reduce rank by merging columns with identical (U, W) patterns
    U_r = _try_reduce_rank(U28, V28, W28, n, m, p, deadline)
    
    if U_r is not None:
        return {"U": U_r[0], "V": U_r[1], "W": U_r[2]}
    
    # Fallback: rank 28
    return {"U": U28, "V": V28, "W": W28}


def _strassen_28(n, m, p):
    """Construct the rank-28 Strassen-based decomposition."""
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_terms = [
        {'U': [(0, 1), (3, 1)], 'V': [(0, 1), (3, 1)], 'W': [(0, 1), (3, 1)]},
        {'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(1, 1), (3, -1)]},
        {'U': [(0, 1)], 'V': [(1, 1), (3, -1)], 'W': [(2, 1), (3, 1)]},
        {'U': [(3, 1)], 'V': [(2, 1), (0, -1)], 'W': [(0, 1), (1, 1)]},
        {'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'W': [(0, -1), (2, 1)]},
        {'U': [(2, 1), (0, -1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)]},
        {'U': [(1, 1), (3, -1)], 'V': [(2, 1), (3, 1)], 'W': [(0, 1)]}
    ]
    
    blocks = [
        {'j_offset': 0, 'k_offset': 0},
        {'j_offset': 2, 'k_offset': 0},
        {'j_offset': 0, 'k_offset': 2},
        {'j_offset': 2, 'k_offset': 2}
    ]
    
    col = 0
    for block in blocks:
        j_off = block['j_offset']
        k_off = block['k_offset']
        for term in strassen_terms:
            for (idx, coeff) in term['U']:
                i = idx // 2
                j = idx % 2
                global_i = i
                global_j = j + j_off
                U[global_i * m + global_j][col] += coeff
            for (idx, coeff) in term['V']:
                j = idx // 2
                k = idx % 2
                global_j = j + j_off
                global_k = k + k_off
                V[global_j * p + global_k][col] += coeff
            for (idx, coeff) in term['W']:
                k = idx // 2
                i = idx % 2
                global_k = k + k_off
                global_i = i
                W[global_k * n + global_i][col] += coeff
            col += 1
    
    U = [[int(x) for x in row] for row in U]
    V = [[int(x) for x in row] for row in V]
    W = [[int(x) for x in row] for row in W]
    
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for t in range(rank):
                    s += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _try_reduce_rank(U28, V28, W28, n, m, p, deadline):
    """Try to reduce the rank by merging columns with identical (U,W) patterns."""
    t_start = time.time()
    
    # Convert to numpy for analysis
    U_arr = np.array(U28, dtype=float)  # shape (n*m, 28)
    V_arr = np.array(V28, dtype=float)  # shape (m*p, 28)
    W_arr = np.array(W28, dtype=float)  # shape (p*n, 28)
    
    n_m = n * m  # 8
    m_p = m * p  # 16
    p_n = p * n  # 8
    r_old = 28
    
    # Group columns by their (U, W) pattern
    # For each column t, the pattern is (U[:,t], W[:,t])
    col_patterns = {}
    for t in range(r_old):
        u_pat = tuple(U_arr[:, t])
        w_pat = tuple(W_arr[:, t])
        key = (u_pat, w_pat)
        if key not in col_patterns:
            col_patterns[key] = []
        col_patterns[key].append(t)
    
    # Sort groups by size (largest first)
    groups = sorted(col_patterns.items(), key=lambda x: -len(x[1]))
    
    # We want to keep at most one column per group
    # The kept column will have V = sum of V from all columns in the group
    # This works because: for any (i,j,k), the contribution from columns in a group
    # is sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
    # If U[i,j,t] and W[k,i,t] are the same for all t in the group,
    # then sum_t U[i,j,t] * V[j,k,t] * W[k,i,t] = U[i,j,t0] * W[k,i,t0] * sum_t V[j,k,t]
    
    # Build the new column set
    new_cols = []  # list of (rep_col, [all_cols_in_group])
    for key, cols in groups:
        rep = cols[0]
        new_cols.append((rep, cols))
    
    r_new = len(new_cols)
    
    if r_new >= r_old:
        return None  # No reduction possible
    
    # Build new U, V, W matrices
    U_new = np.zeros((n_m, r_new))
    V_new = np.zeros((m_p, r_new))
    W_new = np.zeros((p_n, r_new))
    
    for new_idx, (rep, cols) in enumerate(new_cols):
        U_new[:, new_idx] = U_arr[:, rep]
        W_new[:, new_idx] = W_arr[:, rep]
        # V_new is the sum of V from all columns in the group
        for c in cols:
            V_new[:, new_idx] += V_arr[:, c]
    
    # Verify the new decomposition
    U_list = U_new.tolist()
    V_list = V_new.tolist()
    W_list = W_new.tolist()
    
    # Round to integers (they should be integers already)
    U_list = [[int(round(x)) for x in row] for row in U_list]
    V_list = [[int(round(x)) for x in row] for row in V_list]
    W_list = [[int(round(x)) for x in row] for row in W_list]
    
    if _verify(U_list, V_list, W_list, n, m, p):
        return (U_list, V_list, W_list)
    
    # If verification failed, try a more careful approach
    # The issue might be that the (U,W) patterns aren't exactly identical
    # Let's try with a tolerance-based grouping
    
    # Try grouping with small tolerance
    def _group_with_tolerance(U_arr, W_arr, tol=1e-6):
        groups = []
        assigned = [False] * r_old
        for t in range(r_old):
            if assigned[t]:
                continue
            group = [t]
            assigned[t] = True
            for t2 in range(t+1, r_old):
                if assigned[t2]:
                    continue
                if np.allclose(U_arr[:, t], U_arr[:, t2], atol=tol) and \
                   np.allclose(W_arr[:, t], W_arr[:, t2], atol=tol):
                    group.append(t2)
                    assigned[t2] = True
            groups.append(group)
        return groups
    
    groups_tol = _group_with_tolerance(U_arr, W_arr)
    r_new_tol = len(groups_tol)
    
    if r_new_tol < r_old:
        U_new_tol = np.zeros((n_m, r_new_tol))
        V_new_tol = np.zeros((m_p, r_new_tol))
        W_new_tol = np.zeros((p_n, r_new_tol))
        
        for new_idx, cols in enumerate(groups_tol):
            rep = cols[0]
            U_new_tol[:, new_idx] = U_arr[:, rep]
            W_new_tol[:, new_idx] = W_arr[:, rep]
            for c in cols:
                V_new_tol[:, new_idx] += V_arr[:, c]
        
        U_list_tol = U_new_tol.tolist()
        V_list_tol = V_new_tol.tolist()
        W_list_tol = W_new_tol.tolist()
        
        U_list_tol = [[int(round(x)) for x in row] for row in U_list_tol]
        V_list_tol = [[int(round(x)) for x in row] for row in V_list_tol]
        W_list_tol = [[int(round(x)) for x in row] for row in W_list_tol]
        
        if _verify(U_list_tol, V_list_tol, W_list_tol, n, m, p):
            return (U_list_tol, V_list_tol, W_list_tol)
    
    return None
