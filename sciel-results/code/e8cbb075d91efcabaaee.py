"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _qap_cost(a, b, p):
    """Compute the QAP cost for assignment p."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _swap_delta(a, b, p, i, j):
    """Compute the change in cost if we swap facilities i and j in assignment p."""
    pi = p[i]
    pj = p[j]
    if pi == pj:
        return 0
    # The cost contribution involving i or j changes.
    # Current cost terms involving i: a[i,j]*b[pi,pj] + a[j,i]*b[pj,pi]
    # Current cost terms involving j with others k: sum_k a[i,k]*b[pi,p[k]] + a[k,i]*b[p[k],pi]
    # New cost terms after swap:
    # i goes to pj, j goes to pi
    # New terms involving i: a[i,j]*b[pj,pi] + a[j,i]*b[pi,pj]
    # New terms involving j with others k: sum_k a[j,k]*b[pi,p[k]] + a[k,j]*b[p[k],pi]
    
    # Let's compute the delta more carefully.
    # Cost = sum_{u,v} a[u,v] * b[p[u], p[v]]
    # Swap p[i] and p[j].
    # Terms that change are those where u or v is i or j.
    
    # Let's just recompute the affected part.
    # Affected rows/cols in b: pi and pj.
    # Affected rows/cols in a: i and j.
    
    # Current cost contributions for pairs (u,v) where u in {i,j} or v in {i,j}:
    # We can compute the total cost and subtract the unaffected part, but that's expensive.
    # Instead, let's compute the delta directly.
    
    # The terms are:
    # (i,i): a[i,i]*b[pi,pi] -> a[i,i]*b[pj,pj]
    # (i,j): a[i,j]*b[pi,pj] -> a[i,j]*b[pj,pi]
    # (j,i): a[j,i]*b[pj,pi] -> a[j,i]*b[pi,pj]
    # (j,j): a[j,j]*b[pj,pj] -> a[j,j]*b[pi,pi]
    # (i,k) for k != i,j: a[i,k]*b[pi,p[k]] -> a[i,k]*b[pj,p[k]]
    # (k,i) for k != i,j: a[k,i]*b[p[k],pi] -> a[k,i]*b[p[k],pj]
    # (j,k) for k != i,j: a[j,k]*b[pj,p[k]] -> a[j,k]*b[pi,p[k]]
    # (k,j) for k != i,j: a[k,j]*b[p[k],pj] -> a[k,j]*b[p[k],pi]
    
    # Delta = New - Old
    # Delta = a[i,i]*(b[pj,pj]-b[pi,pi]) + a[i,j]*(b[pj,pi]-b[pi,pj]) + a[j,i]*(b[pi,pj]-b[pj,pi]) + a[j,j]*(b[pi,pi]-b[pj,pj])
    #        + sum_{k!=i,j} a[i,k]*(b[pj,p[k]]-b[pi,p[k]]) + sum_{k!=i,j} a[k,i]*(b[p[k],pj]-b[p[k],pi])
    #        + sum_{k!=i,j} a[j,k]*(b[pi,p[k]]-b[pj,p[k]]) + sum_{k!=i,j} a[k,j]*(b[p[k],pi]-b[p[k],pj])
    
    # Let's vectorize this.
    # Create a mask for k != i, j
    mask = np.ones(len(p), dtype=bool)
    mask[i] = False
    mask[j] = False
    
    delta = 0
    delta += a[i,i] * (b[pj,pj] - b[pi,pi])
    delta += a[i,j] * (b[pj,pi] - b[pi,pj])
    delta += a[j,i] * (b[pi,pj] - b[pj,pi])
    delta += a[j,j] * (b[pi,pi] - b[pj,pj])
    
    # For k != i,j
    # sum_{k} a[i,k]*(b[pj,p[k]]-b[pi,p[k]]) where k != i,j
    # This is sum_k a[i,k]*(b[pj, p[k]] - b[pi, p[k]]) - a[i,i]*(b[pj,pi]-b[pi,pi]) - a[i,j]*(b[pj,pj]-b[pi,pj])
    # Wait, the mask excludes i and j.
    # Let's just compute the sums over k != i,j directly using numpy.
    
    # sum_{k!=i,j} a[i,k] * b[pj, p[k]] - sum_{k!=i,j} a[i,k] * b[pi, p[k]]
    # = (sum_k a[i,k]*b[pj,p[k]]) - a[i,i]*b[pj,pi] - a[i,j]*b[pj,pj] - (sum_k a[i,k]*b[pi,p[k]]) + a[i,i]*b[pi,pi] + a[i,j]*b[pi,pj]
    
    # This is getting complicated. Let's use a simpler approach for the delta.
    # Actually, the most efficient way is to precompute row/col sums of b weighted by a.
    # But for a single swap, the O(n) method is fine if we are careful.
    
    # Let's try a different formulation.
    # Let S_u = sum_v a[u,v] * b[p[u], p[v]]  (row sum of cost matrix C = A * B[p,p])
    # Let T_u = sum_v a[v,u] * b[p[v], p[u]]  (col sum of cost matrix C)
    # Total cost = sum_u S_u = sum_u T_u.
    # When we swap p[i] and p[j], the new positions are p'[i]=pj, p'[j]=pi, p'[k]=p[k] for k!=i,j.
    # The new cost C' = A * B[p',p'].
    # The change in cost is sum_{u,v} a[u,v] * (b[p'[u],p'[v]] - b[p[u],p[v]]).
    # This is non-zero only when u or v is i or j.
    
    # Let's just compute it directly with numpy operations that are O(n).
    # b_pi = b[pi, :]  (row pi of B)
    # b_pj = b[pj, :]  (row pj of B)
    # b_col_pi = b[:, pi] (col pi of B)
    # b_col_pj = b[:, pj] (col pj of B)
    
    # The terms are:
    # u=i, v=i: a[i,i]*(b[pj,pj]-b[pi,pi])
    # u=i, v=j: a[i,j]*(b[pj,pi]-b[pi,pj])
    # u=j, v=i: a[j,i]*(b[pi,pj]-b[pj,pi])
    # u=j, v=j: a[j,j]*(b[pi,pi]-b[pj,pj])
    # u=i, v=k (k!=i,j): a[i,k]*(b[pj,p[k]]-b[pi,p[k]])
    # u=k, v=i (k!=i,j): a[k,i]*(b[p[k],pj]-b[p[k],pi])
    # u=j, v=k (k!=i,j): a[j,k]*(b[pi,p[k]]-b[pj,p[k]])
    # u=k, v=j (k!=i,j): a[k,j]*(b[p[k],pi]-b[p[k],pj])
    
    # Sum of terms u=i, v=k for all k:
    # sum_k a[i,k]*(b[pj,p[k]]-b[pi,p[k]]) = sum_k a[i,k]*b[pj,p[k]] - sum_k a[i,k]*b[pi,p[k]]
    # Let row_i_a = a[i, :]
    # sum_k a[i,k]*b[pj,p[k]] = row_i_a @ b[pj, p]  (dot product of a[i,:] with b[pj, p[0], p[1], ...])
    # sum_k a[i,k]*b[pi,p[k]] = row_i_a @ b[pi, p]
    
    # So, sum_{k} a[i,k]*(b[pj,p[k]]-b[pi,p[k]]) = row_i_a @ (b[pj, p] - b[pi, p])
    # But we need to exclude k=i and k=j.
    # The term for k=i is a[i,i]*(b[pj,pi]-b[pi,pi])
    # The term for k=j is a[i,j]*(b[pj,pj]-b[pi,pj])
    
    # So, sum_{k!=i,j} a[i,k]*(b[pj,p[k]]-b[pi,p[k]]) = row_i_a @ (b[pj, p] - b[pi, p]) - a[i,i]*(b[pj,pi]-b[pi,pi]) - a[i,j]*(b[pj,pj]-b[pi,pj])
    
    # Similarly for the other sums.
    
    # Let's implement this.
    b_pi_p = b[pi, p]
    b_pj_p = b[pj, p]
    b_p_pi = b[p, pi]
    b_p_pj = b[p, pj]
    
    # sum_{k!=i,j} a[i,k]*(b[pj,p[k]]-b[pi,p[k]])
    sum_ik = np.dot(a[i, :], b_pj_p - b_pi_p) - a[i,i]*(b[pj,pi]-b[pi,pi]) - a[i,j]*(b[pj,pj]-b[pi,pj])
    
    # sum_{k!=i,j} a[k,i]*(b[p[k],pj]-b[p[k],pi])
    sum_ki = np.dot(a[:, i], b_p_pj - b_p_pi) - a[i,i]*(b[pi,pj]-b[pi,pi]) - a[j,i]*(b[pj,pj]-b[pj,pi])
    
    # sum_{k!=i,j} a[j,k]*(b[pi,p[k]]-b[pj,p[k]])
    sum_jk = np.dot(a[j, :], b_pi_p - b_pj_p) - a[j,i]*(b[pi,pi]-b[pj,pi]) - a[j,j]*(b[pi,pj]-b[pj,pj])
    
    # sum_{k!=i,j} a[k,j]*(b[p[k],pi]-b[p[k],pj])
    sum_kj = np.dot(a[:, j], b_p_pi - b_p_pj) - a[i,j]*(b[pi,pi]-b[pi,pj]) - a[j,j]*(b[pi,pj]-b[pj,pj])
    
    delta += sum_ik + sum_ki + sum_jk + sum_kj
    
    return delta


def _local_search(a, b, p, max_iters=1000, time_limit=10):
    """Perform 2-swap local search."""
    n = len(p)
    p = p.copy()
    current_cost = _qap_cost(a, b, p)
    start_time = time.time()
    
    # Precompute some things for speed? Maybe not necessary for n=100.
    
    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        best_delta = 0
        best_i, best_j = -1, -1
        for i in range(n):
            for j in range(i+1, n):
                delta = _swap_delta(a, b, p, i, j)
                if delta < best_delta:
                    best_delta = delta
                    best_i, best_j = i, j
        if best_delta < 0:
            p[best_i], p[best_j] = p[best_j], p[best_i]
            current_cost += best_delta
        else:
            break
    return p, current_cost


def _lap_guided_perturbation(a, b, p):
    """
    Use a linear surrogate to propose a globally consistent reassignment.
    Surrogate: minimize sum_{i,j} a[i,j] * b[p[i], p[j]] approximated by
    sum_i (sum_j a[i,j]) * b[p[i], c] for some center c?
    Or better: minimize sum_i row_sum_a[i] * b[p[i], p[i]]? No.
    
    A common linear surrogate for QAP is to use the diagonal of B or a fixed center.
    Let's use the approach: assign facility i to location l to minimize sum_j a[i,j] * b[l, j]?
    No, that's not right either.
    
    The linear assignment problem surrogate is often:
    minimize sum_{i,l} c_{i,l} x_{i,l}
    where c_{i,l} = sum_j a[i,j] * b[l, j]  (or similar).
    Let's try c_{i,l} = sum_j a[i,j] * b[l, j].
    This approximates the cost by assuming all facilities are assigned to locations in a way that
    the distance from l to j is weighted by the flow from i to j.
    """
    # c[i, l] = sum_j a[i, j] * b[l, j]
    # This is (A @ B.T)[i, l]
    c = a @ b.T
    # Solve LAP
    row_ind, col_ind = linear_sum_assignment(c)
    # p_new[i] = col_ind[row_ind] ? No.
    # linear_sum_assignment returns row_ind, col_ind such that x[row_ind[k], col_ind[k]] = 1.
    # So facility row_ind[k] is assigned to location col_ind[k].
    # We want p[i] = location.
    p_new = np.empty(len(a), dtype=np.int64)
    for i, l in zip(row_ind, col_ind):
        p_new[i] = l
    return p_new


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Initial solution: rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    best_p = p.copy()
    best_cost = _qap_cost(a, b, p)
    
    # Local search
    p, cost = _local_search(a, b, p, max_iters=500, time_limit=20)
    if cost < best_cost:
        best_p = p.copy()
        best_cost = cost
    
    # Iterative perturbation and local search
    while time.time() < deadline:
        # Perturb
        p_pert = _lap_guided_perturbation(a, b, p)
        
        # Local search
        p_ls, cost_ls = _local_search(a, b, p_pert, max_iters=300, time_limit=15)
        
        if cost_ls < best_cost:
            best_p = p_ls.copy()
            best_cost = cost_ls
            p = p_ls.copy()
        else:
            # Accept with some probability? Or just reject.
            # For now, reject.
            pass
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
