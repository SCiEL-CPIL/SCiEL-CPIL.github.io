"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def _qap_cost(a, b, p):
    """Compute exact QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))


def _delta_swap(a, b, p, i, j):
    """
    Compute exact delta for swapping positions i and j in permutation p.
    p is the current permutation BEFORE the swap.
    Returns the change in total cost if we swap p[i] and p[j].
    """
    li = p[i]
    lj = p[j]
    
    # For k not in {i,j}: p[k] unchanged
    # Contribution change:
    #   a[i,k]*(b[lj,p[k]] - b[li,p[k]]) + a[j,k]*(b[li,p[k]] - b[lj,p[k]])
    #   + a[k,i]*(b[p[k],lj] - b[p[k],li]) + a[k,j]*(b[p[k],li] - b[p[k],lj])
    # = (a[i,k]-a[j,k])*(b[lj,p[k]]-b[li,p[k]]) + (a[k,i]-a[k,j])*(b[p[k],lj]-b[p[k],li])
    
    # Vectorized sum over all k, then correct for k=i and k=j
    diff_row = a[i, :] - a[j, :]  # a[i,k] - a[j,k] for all k
    diff_col = a[:, i] - a[:, j]  # a[k,i] - a[k,j] for all k
    
    b_lj_p = b[lj, p]  # b[lj, p[k]] for all k
    b_li_p = b[li, p]  # b[li, p[k]] for all k
    b_p_lj = b[p, lj]  # b[p[k], lj] for all k
    b_p_li = b[p, li]  # b[p[k], li] for all k
    
    # Sum over all k of (a[i,k]-a[j,k])*(b[lj,p[k]]-b[li,p[k]])
    term1 = diff_row @ (b_lj_p - b_li_p)
    # Sum over all k of (a[k,i]-a[k,j])*(b[p[k],lj]-b[p[k],li])
    term2 = diff_col @ (b_p_lj - b_p_li)
    
    total = term1 + term2
    
    # Now correct: the above sums include k=i and k=j, but for those k,
    # p[k] also changes (p[i]: li->lj, p[j]: lj->li), so the formula above
    # is incorrect for k=i and k=j.
    
    # For k=i in term1: we computed a[i,i]-a[j,i] times (b[lj,p[i]]-b[li,p[i]])
    #   = (a[i,i]-a[j,i])*(b[lj,li]-b[li,li])
    # But the correct contribution for k=i involves the NEW p[i]=lj:
    #   The old cost had a[i,i]*b[li,li] and a[j,i]*b[lj,li] (from k=i terms in both i and j rows)
    #   Actually let me think more carefully.
    
    # The total cost is sum_{u,v} a[u,v]*b[p[u],p[v]]
    # When we swap p[i] and p[j]:
    #   p_new[i] = lj, p_new[j] = li, p_new[k] = p[k] for k!=i,j
    #
    # Delta = sum_{u,v} a[u,v]*(b[p_new[u],p_new[v]] - b[p[u],p[v]])
    #
    # Non-zero only when u or v is in {i,j} (since b indices change only there)
    #
    # Terms where u=i or v=i or u=j or v=j:
    #
    # u=i, v not in {i,j}: a[i,v]*(b[lj,p[v]] - b[li,p[v]])
    # u=j, v not in {i,j}: a[j,v]*(b[li,p[v]] - b[lj,p[v]])
    # v=i, u not in {i,j}: a[u,i]*(b[p[u],lj] - b[p[u],li])
    # v=j, u not in {i,j}: a[u,j]*(b[p[u],li] - b[p[u],lj])
    #
    # u=i, v=i: a[i,i]*(b[lj,lj] - b[li,li])
    # u=i, v=j: a[i,j]*(b[lj,li] - b[li,lj])
    # u=j, v=i: a[j,i]*(b[li,lj] - b[lj,li])
    # u=j, v=j: a[j,j]*(b[li,li] - b[lj,lj])
    
    # The vectorized terms above computed:
    # term1 = sum_k (a[i,k]-a[j,k])*(b[lj,p[k]]-b[li,p[k]])
    #   = sum_{k!=i,j} (a[i,k]-a[j,k])*(b[lj,p[k]]-b[li,p[k]])
    #     + (a[i,i]-a[j,i])*(b[lj,li]-b[li,li])  [k=i, p[i]=li]
    #     + (a[i,j]-a[j,j])*(b[lj,lj]-b[li,lj])  [k=j, p[j]=lj]
    #
    # But the correct terms for k!=i,j in the "u=i or u=j, v=k" part are:
    #   sum_{k!=i,j} a[i,k]*(b[lj,p[k]]-b[li,p[k]]) + a[j,k]*(b[li,p[k]]-b[lj,p[k]])
    #   = sum_{k!=i,j} (a[i,k]-a[j,k])*(b[lj,p[k]]-b[li,p[k]])
    # So term1's k!=i,j part is correct.
    #
    # The k=i term in term1: (a[i,i]-a[j,i])*(b[lj,li]-b[li,li])
    #   The correct contributions involving u=i,v=i and u=j,v=i are:
    #     a[i,i]*(b[lj,lj]-b[li,li]) + a[j,i]*(b[li,lj]-b[lj,li])
    #
    # The k=j term in term1: (a[i,j]-a[j,j])*(b[lj,lj]-b[li,lj])
    #   The correct contributions involving u=i,v=j and u=j,v=j are:
    #     a[i,j]*(b[lj,li]-b[li,lj]) + a[j,j]*(b[li,li]-b[lj,lj])
    
    # Similarly for term2:
    # term2 = sum_k (a[k,i]-a[k,j])*(b[p[k],lj]-b[p[k],li])
    #   k!=i,j part is correct.
    #   k=i: (a[i,i]-a[i,j])*(b[li,lj]-b[li,li])
    #   k=j: (a[j,i]-a[j,j])*(b[lj,lj]-b[lj,li])
    #   Correct: a[i,i]*(b[lj,lj]-b[li,li]) + a[i,j]*(b[lj,li]-b[li,lj])  [v=i, u=i and v=i, u=j]
    #           a[j,i]*(b[li,lj]-b[lj,li]) + a[j,j]*(b[li,li]-b[lj,lj])  [v=j, u=i and v=j, u=j]
    
    # So: remove the incorrect k=i,j terms from term1 and term2, add correct ones.
    
    # Remove from term1:
    total -= (a[i,i]-a[j,i])*(b[lj,li]-b[li,li])
    total -= (a[i,j]-a[j,j])*(b[lj,lj]-b[li,lj])
    
    # Remove from term2:
    total -= (a[i,i]-a[i,j])*(b[li,lj]-b[li,li])
    total -= (a[j,i]-a[j,j])*(b[lj,lj]-b[lj,li])
    
    # Add correct 4 corner terms:
    total += a[i,i]*(b[lj,lj]-b[li,li])
    total += a[i,j]*(b[lj,li]-b[li,lj])
    total += a[j,i]*(b[li,lj]-b[lj,li])
    total += a[j,j]*(b[li,li]-b[lj,lj])
    
    return int(total)


def _local_search_2opt(a, b, p, deadline, max_passes=50):
    """First-improvement 2-opt local search. Returns improved p and cost."""
    n = len(a)
    current_cost = _qap_cost(a, b, p)
    improved = True
    passes = 0
    
    while improved and passes < max_passes and time.time() < deadline:
        improved = False
        passes += 1
        for i in range(n):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n):
                delta = _delta_swap(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    break  # restart inner loop after improvement
    return p, current_cost


def _initial_permutation(a, b):
    """Generate initial permutation using spectral + Hungarian + random."""
    n = len(a)
    best_p = None
    best_cost = np.inf
    
    # Method 1: Rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p1 = np.empty(n, dtype=np.int64)
    p1[heavy] = central
    cost1 = _qap_cost(a, b, p1)
    
    # Method 2: Spectral embedding
    try:
        # Use the largest eigenvectors of a and b
        from scipy.sparse.linalg import eigsh
        # For dense matrices, use np.linalg.eigh
        w_a, v_a = np.linalg.eigh(a)
        w_b, v_b = np.linalg.eigh(b)
        # Take top k eigenvectors
        k = min(10, n)
        idx_a = np.argsort(w_a)[-k:]
        idx_b = np.argsort(w_b)[-k:]
        X = v_a[:, idx_a]  # n x k
        Y = v_b[:, idx_b]  # n x k
        # Solve assignment: minimize sum a[i,j]*b[p[i],p[j]]
        # Approximate by matching row norms and angles
        # Use Hungarian on a derived cost matrix
        # Cost matrix: c[i,j] = -X[i,:] @ Y[j,:].T  (maximize similarity)
        C = -X @ Y.T
        row_ind, col_ind = linear_sum_assignment(C)
        p2 = np.empty(n, dtype=np.int64)
        p2[row_ind] = col_ind
        cost2 = _qap_cost(a, b, p2)
    except Exception:
        p2 = p1.copy()
        cost2 = cost1
    
    # Method 3: Random
    p3 = np.random.permutation(n).astype(np.int64)
    cost3 = _qap_cost(a, b, p3)
    
    # Pick best
    candidates = [(cost1, p1), (cost2, p2), (cost3, p3)]
    best_cost = min(c[0] for c in candidates)
    best_p = [c[1] for c in candidates if c[0] == best_cost][0].copy()
    
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Generate initial solution
    p, best_cost = _initial_permutation(a, b)
    current_cost = best_cost
    best_p = p.copy()
    
    # Phase 1: Simulated Annealing
    T = 1000.0
    T_min = 1.0
    alpha = 0.9995
    
    while T > T_min and time.time() < deadline - 30.0:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        
        delta = _delta_swap(a, b, p, i, j)
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        T *= alpha
    
    # Phase 2: Local search polish on best solution found
    if time.time() < deadline - 10.0:
        p_ls = best_p.copy()
        p_ls, cost_ls = _local_search_2opt(a, b, p_ls, deadline - 5.0, max_passes=30)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls
    
    # Phase 3: Another SA restart from best if time remains
    if time.time() < deadline - 20.0:
        p2 = best_p.copy()
        T2 = 500.0
        T2_min = 1.0
        alpha2 = 0.9995
        current_cost2 = best_cost
        
        while T2 > T2_min and time.time() < deadline - 10.0:
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            
            delta = _delta_swap(a, b, p2, i, j)
            
            if delta < 0 or np.random.random() < np.exp(-delta / T2):
                p2[i], p2[j] = p2[j], p2[i]
                current_cost2 += delta
                
                if current_cost2 < best_cost:
                    best_cost = current_cost2
                    best_p = p2.copy()
            
            T2 *= alpha2
        
        # Final local search
        if time.time() < deadline - 5.0:
            p_ls2 = best_p.copy()
            p_ls2, cost_ls2 = _local_search_2opt(a, b, p_ls2, deadline - 2.0, max_passes=20)
            if cost_ls2 < best_cost:
                best_cost = cost_ls2
                best_p = p_ls2
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
