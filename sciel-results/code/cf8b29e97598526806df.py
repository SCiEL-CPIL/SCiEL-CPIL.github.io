# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    return int(np.sum(a * b[np.ix_(p, p)]))


def _compute_deltas(a, b, p):
    """Compute delta for all pairs (i, j) with i < j.
    Returns a matrix D where D[i, j] is the delta for swapping i and j.
    """
    n = len(p)
    # Precompute b_perm = b[p, :] so b_perm[k, :] = b[p[k], :]
    # And b_perm_T = b[:, p] so b_perm_T[:, k] = b[:, p[k]]
    b_perm = b[np.ix_(p, p)] # b_perm[i, j] = b[p[i], p[j]]
    
    # We want Delta[i, j] for i != j.
    # Formula:
    # RowTerm = (a[i,:] - a[j,:]) @ (b[p[i], p] - b[p[j], p])
    # Note: b[p[i], p] is row p[i] of b, indexed by p.
    # This is equivalent to b_perm[i, :] if we consider b_perm[i, k] = b[p[i], p[k]].
    # Yes, b_perm[i, :] is exactly b[p[i], p].
    
    # So RowTerm[i, j] = (a[i,:] - a[j,:]) @ (b_perm[i, :] - b_perm[j, :])
    # ColTerm[i, j] = (a[:,i] - a[:,j]) @ (b[:, p[i]] - b[:, p[j]])
    # Note: b[:, p[i]] is column p[i] of b.
    # b_perm[:, i] is b[p, p[i]] which is b[:, p[i]].T? No.
    # b_perm[k, i] = b[p[k], p[i]].
    # So b_perm[:, i] is the vector [b[p[0], p[i]], ..., b[p[n-1], p[i]]].
    # This is exactly b[:, p[i]] indexed by p? No, b[:, p[i]] is a vector of length n (locations).
    # The term in ColTerm is sum_k (a[k,i]-a[k,j]) (b[p[k], v] - b[p[k], u]).
    # Here v=p[i], u=p[j].
    # b[p[k], v] is element (p[k], v) of b.
    # This is b_perm[k, i] because b_perm[k, i] = b[p[k], p[i]] = b[p[k], v].
    # So ColTerm[i, j] = (a[:,i] - a[:,j]) @ (b_perm[:, i] - b_perm[:, j])
    
    # Full Delta[i, j] = RowTerm[i, j] + ColTerm[i, j] - Correction[i, j] + Direct[i, j]
    
    # Let's compute RowTerm and ColTerm for all i, j.
    # This is O(n^3) if done naively, but we can use matrix multiplication tricks.
    
    # RowTerm[i, j] = a[i,:] @ b_perm[i,:] - a[i,:] @ b_perm[j,:] - a[j,:] @ b_perm[i,:] + a[j,:] @ b_perm[j,:]
    # Let M1 = a @ b_perm.T  -> M1[i, k] = sum_j a[i, j] b_perm[k, j] = sum_j a[i, j] b[p[k], p[j]]
    # Then a[i,:] @ b_perm[j,:] is M1[i, j].
    # So RowTerm[i, j] = M1[i, i] - M1[i, j] - M1[j, i] + M1[j, j]
    
    # ColTerm[i, j] = a[:,i] @ b_perm[:,i] - a[:,i] @ b_perm[:,j] - a[:,j] @ b_perm[:,i] + a[:,j] @ b_perm[:,j]
    # Let M2 = a.T @ b_perm  -> M2[k, i] = sum_j a[j, k] b_perm[j, i] = sum_j a[j, k] b[p[j], p[i]]
    # Wait, a[:, i] @ b_perm[:, i] is sum_k a[k, i] b_perm[k, i].
    # This is element (i, i) of M2?
    # M2[i, i] = sum_k a[k, i] b_perm[k, i]. Yes.
    # a[:, i] @ b_perm[:, j] is sum_k a[k, i] b_perm[k, j]. This is M2[i, j].
    # So ColTerm[i, j] = M2[i, i] - M2[i, j] - M2[j, i] + M2[j, j]
    
    M1 = a @ b_perm.T
    M2 = a.T @ b_perm
    
    # RowTerm matrix R where R[i, j] = M1[i, i] - M1[i, j] - M1[j, i] + M1[j, j]
    # ColTerm matrix C where C[i, j] = M2[i, i] - M2[i, j] - M2[j, i] + M2[j, j]
    
    diag_M1 = np.diag(M1)
    diag_M2 = np.diag(M2)
    
    R = diag_M1[:, None] + diag_M1[None, :] - M1 - M1.T
    C = diag_M2[:, None] + diag_M2[None, :] - M2 - M2.T
    
    Delta_full = R + C
    
    # Now subtract corrections for k=i, j and add direct terms.
    # Correction for pair (i, j):
    # R_i = (a[i,i]-a[j,i])(b[v,u]-b[u,u])
    # C_i = (a[i,i]-a[i,j])(b[u,v]-b[u,u])
    # R_j = (a[i,j]-a[j,j])(b[v,v]-b[u,v])
    # C_j = (a[j,i]-a[j,j])(b[v,v]-b[v,u])
    # v = p[i], u = p[j]
    
    # We need to construct matrices for these terms.
    # b_vu = b[p[i], p[j]] -> b_perm[i, j]
    # b_uu = b[p[j], p[j]] -> b_perm[j, j]
    # b_uv = b[p[i], p[j]]? No, b[u, v] = b[p[j], p[i]] -> b_perm[j, i]
    # b_vv = b[p[i], p[i]] -> b_perm[i, i]
    
    # Let's compute term by term matrices.
    # b_vu_mat[i, j] = b[p[i], p[j]] = b_perm[i, j]
    # b_uu_mat[i, j] = b[p[j], p[j]] = diag(b_perm)[j] -> broadcast to [i, j]
    # b_uv_mat[i, j] = b[p[j], p[i]] = b_perm[j, i]
    # b_vv_mat[i, j] = b[p[i], p[i]] = diag(b_perm)[i] -> broadcast to [i, j]
    
    b_vu = b_perm
    b_uu = np.diag(b_perm)[None, :]
    b_uv = b_perm.T
    b_vv = np.diag(b_perm)[:, None]
    
    # a_ii = a[i, i]
    # a_ji = a[j, i]
    # a_ij = a[i, j]
    # a_jj = a[j, j]
    
    a_ii = np.diag(a)[:, None]
    a_ji = a.T # a_ji[i, j] = a[j, i]
    a_ij = a
    a_jj = np.diag(a)[None, :]
    
    # R_i = (a_ii - a_ji) * (b_vu - b_uu)
    # C_i = (a_ii - a_ij) * (b_uv - b_uu)
    # R_j = (a_ij - a_jj) * (b_vv - b_uv)  # Wait, R_j used b[v,v]-b[u,v] which is b_vv - b_uv?
    # Let's re-verify R_j: (a[i,j]-a[j,j])(b[v,v]-b[u,v]).
    # b[v,v] is b_vv. b[u,v] is b[p[j], p[i]] = b_uv.
    # So yes, b_vv - b_uv.
    # C_j = (a[j,i]-a[j,j])(b[v,v]-b[v,u])
    # b[v,v] is b_vv. b[v,u] is b[p[i], p[j]] = b_vu.
    # So b_vv - b_vu.
    
    Corr_R_i = (a_ii - a_ji) * (b_vu - b_uu)
    Corr_C_i = (a_ii - a_ij) * (b_uv - b_uu)
    Corr_R_j = (a_ij - a_jj) * (b_vv - b_uv)
    Corr_C_j = (a_ji - a_jj) * (b_vv - b_vu)
    
    Correction = Corr_R_i + Corr_C_i + Corr_R_j + Corr_C_j
    
    # Direct Term:
    # a[i,j](b[v,u]-b[u,v]) + a[j,i](b[u,v]-b[v,u])
    # = a_ij * (b_vu - b_uv) + a_ji * (b_uv - b_vu)
    # = (a_ij - a_ji) * (b_vu - b_uv)
    
    Direct = (a_ij - a_ji) * (b_vu - b_uv)
    
    Delta = Delta_full - Correction + Direct
    
    # Set diagonal to 0 (no swap)
    np.fill_diagonal(Delta, 0)
    
    return Delta


def _local_search(a, b, p, deadline, best_cost, best_p):
    n = len(p)
    cur = _qap_cost(a, b, p)
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
    
    while time.time() < deadline:
        D = _compute_deltas(a, b, p)
        # Find best improvement
        i, j = np.unravel_index(np.argmax(D), D.shape)
        # Note: argmax finds max delta. We want min delta (most negative).
        # So we should use argmin.
        i, j = np.unravel_index(np.argmin(D), D.shape)
        
        if D[i, j] >= 0:
            break
        
        p[i], p[j] = p[j], p[i]
        cur += D[i, j]
        if cur < best_cost:
            best_cost = cur
            best_p = p.copy()
            
    return best_cost, best_p


def _perturb(p, rng, num_swaps=3):
    n = len(p)
    for _ in range(num_swaps):
        i = rng.integers(n)
        j = rng.integers(n)
        if i != j:
            p[i], p[j] = p[j], p[i]


def solve(a, b):
    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)
    n = len(a)
    t0 = time.time()
    deadline = t0 + 115.0

    rng = np.random.default_rng(12345)
    
    # Initial solution: Rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    best_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    
    # ILS Loop
    while time.time() < deadline - 5.0:
        # Local Search
        p_cur = best_p.copy()
        best_cost, best_p = _local_search(a, b, p_cur, deadline, best_cost, best_p)
        
        # Perturb
        p_perturbed = best_p.copy()
        _perturb(p_perturbed, rng, num_swaps=3)
        
        # Check if perturbed solution is better (unlikely but possible)
        cost_pert = _qap_cost(a, b, p_perturbed)
        if cost_pert < best_cost:
            best_cost = cost_pert
            best_p = p_perturbed.copy()
            
        # If time is tight, break
        if time.time() > deadline - 10.0:
            break
            
    # Final local search on best_p
    if time.time() < deadline - 1.0:
        best_cost, best_p = _local_search(a, b, best_p.copy(), deadline, best_cost, best_p)
        
    return best_p.tolist()
# EVOLVE-BLOCK-END
