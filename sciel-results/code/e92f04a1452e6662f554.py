# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _qap_cost(a, b, p):
    return int(np.sum(a * b[np.ix_(p, p)]))


def _compute_C(a, b, p):
    """C[i] = sum_j a[i,j] * b[p[i], p[j]]"""
    B = b[np.ix_(p, p)]
    return (a @ B).astype(np.int64)


def _swap_delta_vec(a, b, p, i, j):
    """Compute exact delta for swapping facilities at positions i and j using vectorized ops."""
    n = len(p)
    pi = p[i]
    pj = p[j]
    
    # Vectorized computation of the four sums
    # S1 = sum_k a[i,k] * (b[pj,pk] - b[pi,pk])
    # S2 = sum_k a[j,k] * (b[pi,pk] - b[pj,pk])
    # S3 = sum_k a[k,i] * (b[pk,pj] - b[pk,pi])
    # S4 = sum_k a[k,j] * (b[pk,pi] - b[pk,pj])
    
    # diff_row[k] = b[pj,pk] - b[pi,pk]
    diff_row = b[pj, p] - b[pi, p]
    # diff_col[k] = b[pk,pj] - b[pk,pi]
    diff_col = b[p, pj] - b[p, pi]
    
    S1 = int(np.dot(a[i, :], diff_row))
    S2 = int(np.dot(a[j, :], -diff_row))
    S3 = int(np.dot(a[:, i], diff_col))
    S4 = int(np.dot(a[:, j], -diff_col))
    
    total = S1 + S2 + S3 + S4
    
    # Subtract the double-counted i-j interaction terms.
    b_pi_pi = b[pi, pi]
    b_pj_pj = b[pj, pj]
    b_pi_pj = b[pi, pj]
    b_pj_pi = b[pj, pi]
    
    Rem_ii = a[i, i] * (b_pj_pi - b_pi_pi + b_pi_pj - b_pi_pi)
    Rem_ij = a[i, j] * (b_pj_pj - b_pi_pj + b_pi_pi - b_pi_pj)
    Rem_ji = a[j, i] * (b_pi_pi - b_pj_pi + b_pj_pj - b_pj_pi)
    Rem_jj = a[j, j] * (b_pi_pj - b_pj_pj + b_pj_pi - b_pj_pj)
    
    Rem = Rem_ii + Rem_ij + Rem_ji + Rem_jj
    
    Direct = a[i, j] * (b_pj_pi - b_pi_pj) + a[j, i] * (b_pi_pj - b_pj_pi)
    
    return total - Rem + Direct

def _three_cycle_delta(a, b, p, i, j, k):
    """
    Compute delta for 3-cycle: p[i], p[j], p[k] -> p[j], p[k], p[i]
    i.e., i takes value of j, j takes value of k, k takes value of i.
    """
    pi = p[i]
    pj = p[j]
    pk = p[k]
    
    # The new permutation p' has:
    # p'[i] = pj
    # p'[j] = pk
    # p'[k] = pi
    # p'[m] = p[m] for m != i, j, k
    
    # Cost change = NewCost - OldCost
    # Terms involving i, j, k in the sum sum_{u,v} a[u,v] b[p[u], p[v]]
    
    # We can compute the delta by summing contributions of pairs (u,v) where at least one of u,v is in {i,j,k}.
    # Pairs:
    # (i,i), (i,j), (i,k)
    # (j,i), (j,j), (j,k)
    # (k,i), (k,j), (k,k)
    # (m,i), (m,j), (m,k) for m not in {i,j,k}
    # (i,m), (j,m), (k,m) for m not in {i,j,k}
    
    # Let's use the general formula:
    # Delta = sum_{u,v} a[u,v] (b[p'[u], p'[v]] - b[p[u], p[v]])
    
    # We can compute this by:
    # 1. Calculate sum over all u,v of a[u,v] b[p'[u], p'[v]]? No, that's O(N^2).
    # 2. Use the property that only rows/cols i,j,k change in p.
    
    # Let's define the "old" and "new" values for the indices i, j, k.
    # Old: p[i]=pi, p[j]=pj, p[k]=pk
    # New: p[i]=pj, p[j]=pk, p[k]=pi
    
    # The change in cost is:
    # Sum_{m not in {i,j,k}} a[i,m](b[pj, p[m]] - b[pi, p[m]]) + a[j,m](b[pk, p[m]] - b[pj, p[m]]) + a[k,m](b[pi, p[m]] - b[pk, p[m]])
    # + Sum_{m not in {i,j,k}} a[m,i](b[p[m], pj] - b[p[m], pi]) + a[m,j](b[p[m], pk] - b[p[m], pj]) + a[m,k](b[p[m], pi] - b[p[m], pk])
    # + Internal terms for (i,j,k)
    
    # Internal terms:
    # Old: a[i,i]b[pi,pi] + a[i,j]b[pi,pj] + a[i,k]b[pi,pk]
    #       a[j,i]b[pj,pi] + a[j,j]b[pj,pj] + a[j,k]b[pj,pk]
    #       a[k,i]b[pk,pi] + a[k,j]b[pk,pj] + a[k,k]b[pk,pk]
    # New: a[i,i]b[pj,pj] + a[i,j]b[pj,pk] + a[i,k]b[pj,pi]
    #       a[j,i]b[pk,pi] + a[j,j]b[pk,pk] + a[j,k]b[pk,pj]
    #       a[k,i]b[pi,pi] + a[k,j]b[pi,pj] + a[k,k]b[pi,pk]
    
    # This is complex to vectorize efficiently for a single call.
    # Given N=100, a simple loop over m for the external terms is O(N).
    # The internal terms are O(1).
    
    mask = np.ones(n, dtype=bool)
    mask[i] = False
    mask[j] = False
    mask[k] = False
    m_indices = np.where(mask)[0]
    
    # External terms
    # Row contributions (i, j, k as first index)
    # delta_row = sum_{m in M} a[i,m](b[pj, p[m]] - b[pi, p[m]]) + ...
    
    # Vectorized:
    # For i: diff_i = b[pj, p[m_indices]] - b[pi, p[m_indices]]
    # Sum a[i, m_indices] * diff_i
    
    diff_i = b[pj, p[m_indices]] - b[pi, p[m_indices]]
    diff_j = b[pk, p[m_indices]] - b[pj, p[m_indices]]
    diff_k = b[pi, p[m_indices]] - b[pk, p[m_indices]]
    
    ext_i = np.dot(a[i, m_indices], diff_i)
    ext_j = np.dot(a[j, m_indices], diff_j)
    ext_k = np.dot(a[k, m_indices], diff_k)
    
    # Col contributions (i, j, k as second index)
    # delta_col = sum_{m in M} a[m,i](b[p[m], pj] - b[p[m], pi]) + ...
    
    diff_i_col = b[p[m_indices], pj] - b[p[m_indices], pi]
    diff_j_col = b[p[m_indices], pk] - b[p[m_indices], pj]
    diff_k_col = b[p[m_indices], pi] - b[p[m_indices], pk]
    
    ext_i_col = np.dot(a[m_indices, i], diff_i_col)
    ext_j_col = np.dot(a[m_indices, j], diff_j_col)
    ext_k_col = np.dot(a[m_indices, k], diff_k_col)
    
    ext_total = ext_i + ext_j + ext_k + ext_i_col + ext_j_col + ext_k_col
    
    # Internal terms
    old_int = (a[i,i]*b[pi,pi] + a[i,j]*b[pi,pj] + a[i,k]*b[pi,pk] +
               a[j,i]*b[pj,pi] + a[j,j]*b[pj,pj] + a[j,k]*b[pj,pk] +
               a[k,i]*b[pk,pi] + a[k,j]*b[pk,pj] + a[k,k]*b[pk,pk])
    
    new_int = (a[i,i]*b[pj,pj] + a[i,j]*b[pj,pk] + a[i,k]*b[pj,pi] +
               a[j,i]*b[pk,pi] + a[j,j]*b[pk,pk] + a[j,k]*b[pk,pj] +
               a[k,i]*b[pi,pi] + a[k,j]*b[pi,pj] + a[k,k]*b[pi,pk])
    
    int_delta = new_int - old_int
    
    return int(ext_total + int_delta)


def _local_search_vnd(a, b, p, deadline, best_cost, best_p, rng):
    """Variable Neighborhood Descent (VND) with 2-opt and 3-opt neighborhoods."""
    n = len(p)
    cur = _qap_cost(a, b, p)
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 2-opt Best Improvement
        best_delta = 0
        best_i, best_j = -1, -1
        for i in range(n):
            pi = p[i]
            for j in range(i + 1, n):
                d = _swap_delta_vec(a, b, p, i, j)
                if d < best_delta:
                    best_delta = d
                    best_i, best_j = i, j
                    if time.time() > deadline:
                        return best_cost, best_p
                    
        if best_delta < 0:
            p[best_i], p[best_j] = p[best_j], p[best_i]
            cur += best_delta
            improved = True
            if cur < best_cost:
                best_cost = cur
                best_p = p.copy()
        else:
            # Try 3-opt if 2-opt is stuck
            # Sample a few random 3-cycles
            for _ in range(100):
                if time.time() > deadline:
                    return best_cost, best_p
                i, j, k = rng.choice(n, 3, replace=False)
                d = _three_cycle_delta(a, b, p, i, j, k)
                if d < 0:
                    # Apply 3-cycle: i->j, j->k, k->i
                    # p[i] = p[j], p[j] = p[k], p[k] = p[i]
                    pi, pj, pk = p[i], p[j], p[k]
                    p[i] = pj
                    p[j] = pk
                    p[k] = pi
                    cur += d
                    improved = True
                    if cur < best_cost:
                        best_cost = cur
                        best_p = p.copy()
                    break # Restart 2-opt search after a 3-opt move
    
    return best_cost, best_p


def _anneal(a, b, p, deadline, best_cost, best_p, rng, t_start=50.0, t_end=0.1, alpha=0.995):
    """Simulated annealing polish."""
    n = len(p)
    cur = _qap_cost(a, b, p)
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
    
    t = t_start
    while time.time() < deadline and t > t_end:
        i = rng.integers(n)
        j = rng.integers(n)
        if i == j:
            continue
        d = _swap_delta_vec(a, b, p, i, j)
        if d < 0 or rng.random() < np.exp(-d / max(t, 1e-10)):
            p[i], p[j] = p[j], p[i]
            cur += d
            if cur < best_cost:
                best_cost = cur
                best_p = p.copy()
        t *= alpha
    
    # Final local search
    cur = _qap_cost(a, b, p)
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
    best_cost, best_p = _local_search_vnd(a, b, p, deadline, best_cost, best_p, rng)
    return best_cost, best_p

def _hungarian_initial(a, b):
    """
    Create a surrogate cost matrix and solve with Hungarian algorithm.
    Surrogate: C[i, j] = rowSum(a)[i] * rowSum(b)[j]
    """
    a_sum = a.sum(axis=1)
    b_sum = b.sum(axis=1)
    # We want to assign facility i to location j such that sum a[i] * b[j] is minimized?
    # Actually, the QAP cost is sum a[i,k] b[p[i], p[k]].
    # A good approximation is to match high flow rows with low distance rows?
    # Or high flow rows with "central" locations?
    # Let's use the product of row sums as a weight.
    # We want to minimize sum_i (a_sum[i] * b_sum[p[i]])?
    # This is a linear assignment problem.
    
    # Cost matrix for Hungarian:
    # cost[i, j] = a_sum[i] * b_sum[j]
    cost = np.outer(a_sum, b_sum)
    
    # Hungarian algorithm minimizes the sum of cost[i, p[i]]
    row_ind, col_ind = linear_sum_assignment(cost)
    p = np.empty(len(a), dtype=np.int64)
    p[row_ind] = col_ind
    return p


def solve(a, b):
    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)
    n = len(a)
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin before 120s kill

    rng = np.random.default_rng(12345)
    
    # --- Constructive starts ---
    best_cost = np.iinfo(np.int64).max
    best_p = None
    
    # Strategy 1: Rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p1 = np.empty(n, dtype=np.int64)
    p1[heavy] = central
    c1 = _qap_cost(a, b, p1)
    if c1 < best_cost:
        best_cost = c1
        best_p = p1.copy()
        
    # Strategy 2: Greedy
    p2 = np.zeros(n, dtype=np.int64)
    used = np.zeros(n, dtype=bool)
    order = np.argsort(-a.sum(axis=1), kind="stable")
    for idx in order:
        if not np.any(used):
            p2[idx] = int(np.argmin(b.sum(axis=1)))
        else:
            placed = np.where(~used)[0]
            scores = np.zeros(n, dtype=np.float64)
            for j in placed:
                scores += a[idx, j] * b[:, p2[j]] + a[j, idx] * b[p2[j], :]
            candidates = np.where(~used)[0]
            p2[idx] = int(candidates[np.argmin(scores[candidates])])
        used[p2[idx]] = True
    c2 = _qap_cost(a, b, p2)
    if c2 < best_cost:
        best_cost = c2
        best_p = p2.copy()

    # Strategy 3: Hungarian
    try:
        p3 = _hungarian_initial(a, b)
        c3 = _qap_cost(a, b, p3)
        if c3 < best_cost:
            best_cost = c3
            best_p = p3.copy()
    except Exception:
        pass

    # --- Refinement ---
    p = best_p.copy()
    best_cost, best_p = _local_search_vnd(a, b, p, deadline, best_cost, best_p, rng)
    
    # --- Diversification with Annealing ---
    while time.time() < deadline - 5.0:
        # Perturb current best
        p = best_p.copy()
        # Random 2-swap perturbation
        for _ in range(3):
            i = rng.integers(n)
            j = rng.integers(n)
            if i != j:
                p[i], p[j] = p[j], p[i]
                
        # Anneal from perturbed state
        best_cost, best_p = _anneal(a, b, p, deadline, best_cost, best_p, rng)
        
        # If we didn't improve, try a fresh random start occasionally
        if rng.random() < 0.1:
            p = rng.permutation(n).astype(np.int64)
            best_cost, best_p = _local_search_vnd(a, b, p, deadline, best_cost, best_p, rng)

    return best_p.tolist()
# EVOLVE-BLOCK-END
