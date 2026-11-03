"""Bootstrap portfolio: SEARCH approach with time-bounded local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment


def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _swap_delta(a, b, p, i, j, a_sym=None):
    """
    Compute the change in cost when swapping facilities at positions i and j in permutation p.
    p is a numpy array where p[k] is the location assigned to facility k.
    We want to swap p[i] and p[j].
    
    Let x = p[i], y = p[j].
    
    Total cost = sum_{m,n} a[m,n] * b[p[m], p[n]]
    
    When we swap p[i] and p[j], let x = p[i], y = p[j].
    The terms that change are those where m=i or n=i or m=j or n=j.
    
    Let's define:
    S_i = sum_{k != i, j} a[i,k] * b[x, p[k]] + sum_{k != i, j} a[k,i] * b[p[k], x]
    S_j = sum_{k != i, j} a[j,k] * b[y, p[k]] + sum_{k != i, j} a[k,j] * b[p[k], y]
    Cross = a[i,j] * b[x,y] + a[j,i] * b[y,x]
    
    Old contribution = S_i + S_j + Cross
    Note: S_i includes terms with b[x, p[k]] and b[p[k], x]. Since b is symmetric, b[p[k],x] = b[x,p[k]].
    So S_i = 2 * sum_{k != i,j} a[i,k] * b[x, p[k]] if a is symmetric? No, a[i,k] != a[k,i] necessarily.
    Actually, S_i = sum_{k != i,j} a[i,k] * b[x, p[k]] + sum_{k != i,j} a[k,i] * b[p[k], x]
    = sum_{k != i,j} (a[i,k] + a[k,i]) * b[x, p[k]]  (since b is symmetric)
    
    After swap, p[i]=y, p[j]=x.
    New S_i' = sum_{k != i,j} a[i,k] * b[y, p[k]] + sum_{k != i,j} a[k,i] * b[p[k], y]
             = sum_{k != i,j} (a[i,k] + a[k,i]) * b[y, p[k]]
    New S_j' = sum_{k != i,j} a[j,k] * b[x, p[k]] + sum_{k != i,j} a[k,j] * b[p[k], x]
             = sum_{k != i,j} (a[j,k] + a[k,j]) * b[x, p[k]]
    New Cross' = a[i,j] * b[y,x] + a[j,i] * b[x,y] = 2 * a[i,j] * b[x,y] (same as Cross if a,b symmetric)
    
    So delta = (S_i' + S_j' + Cross') - (S_i + S_j + Cross)
             = S_i' - S_i + S_j' - S_j
             = sum_{k != i,j} (a[i,k] + a[k,i]) * (b[y, p[k]] - b[x, p[k]])
             + sum_{k != i,j} (a[j,k] + a[k,j]) * (b[x, p[k]] - b[y, p[k]])
             = sum_{k != i,j} [ (a[i,k] + a[k,i]) - (a[j,k] + a[k,j]) ] * (b[y, p[k]] - b[x, p[k]])
    
    This is O(n) per swap evaluation.
    """
    if a_sym is None:
        a_sym = a + a.T
    
    x = p[i]
    y = p[j]
    
    # diff_a[k] = (a[i,k] + a[k,i]) - (a[j,k] + a[k,j]) = a_sym[i,k] - a_sym[j,k]
    diff_a = a_sym[i, :] - a_sym[j, :]
    diff_b = b[y, :] - b[x, :]
    
    # Now compute sum_{k != i,j} diff_a[k] * diff_b[p[k]]
    temp = diff_a * diff_b[p]
    # Subtract the terms for k=i and k=j
    # k=i: diff_a[i] * diff_b[p[i]] = diff_a[i] * diff_b[x]
    # k=j: diff_a[j] * diff_b[p[j]] = diff_a[j] * diff_b[y]
    delta = np.sum(temp) - diff_a[i] * diff_b[x] - diff_a[j] * diff_b[y]
    
    return delta


def _hungarian_step(a, b, p, a_sym=None):
    """
    Perform a global reassignment using the Hungarian algorithm.
    
    The idea is to linearize the QAP cost around the current assignment p.
    For each facility i, the marginal cost of assigning it to location l, given the
    current assignments of other facilities, is:
      c[i, l] = sum_{k != i} (a[i,k] + a[k,i]) * b[l, p[k]]
    
    We can compute this efficiently:
      c[i, :] = sum_{k} (a[i,k] + a[k,i]) * b[:, p[k]] - (a[i,i] + a[i,i]) * b[:, p[i]]
               = (a_sym[i, :] @ b[:, p]) - 2*a[i,i] * b[:, p[i]]
    
    But wait, the term for k=i in the sum is (a[i,i] + a[i,i]) * b[:, p[i]] = 2*a[i,i]*b[:, p[i]].
    So c[i, :] = a_sym[i, :] @ b[:, p] - 2*a[i,i] * b[:, p[i]]
    
    However, this is the cost contribution from facility i to all locations, assuming other
    facilities stay fixed. The total cost is sum_i c[i, p[i]], but note that each pair (i,j) is
    counted twice in the sum over i (once for i and once for j). Actually, let's be careful.
    
    The QAP cost is sum_{i,j} a[i,j] * b[p[i], p[j]].
    We can rewrite this as sum_i sum_j a[i,j] * b[p[i], p[j]].
    For a fixed i, the term involving p[i] is sum_j a[i,j] * b[p[i], p[j]].
    This is not separable in p[i] alone because p[j] for j != i also appear.
    
    The linearization c[i, l] = sum_{k != i} (a[i,k] + a[k,i]) * b[l, p[k]] is an approximation
    where we fix p[k] for k != i. The total linearized cost is sum_i c[i, p[i]], which is
    an overcount because each pair (i,k) is counted in both c[i, p[i]] and c[k, p[k]].
    Specifically, the term a[i,k]*b[p[i], p[k]] appears in c[i, p[i]] (as part of sum_k for k) 
    and in c[k, p[k]] (as part of sum_i for i). So the linearized cost is 2 * QAP cost
    minus the diagonal terms 2 * sum_i a[i,i] * b[p[i], p[i]].
    
    Anyway, the Hungarian algorithm on c[i, l] will find a permutation that minimizes the
    linearized cost, which is a good heuristic for the original QAP.
    
    Returns: new permutation p_new, and the linearized cost (not the true QAP cost).
    """
    n = len(a)
    if a_sym is None:
        a_sym = a + a.T
    
    # Compute cost matrix c[i, l] = sum_{k != i} (a[i,k] + a[k,i]) * b[l, p[k]]
    # = a_sym[i, :] @ b[:, p] - 2*a[i,i] * b[:, p[i]]
    
    # b[:, p] is the matrix B where B[l, k] = b[l, p[k]]
    B = b[:, p]  # shape (n, n), B[l, k] = b[l, p[k]]
    
    # c[i, l] = sum_k a_sym[i, k] * B[l, k] - 2*a[i,i] * B[l, i]
    # = (a_sym @ B.T)[i, l] - 2*a[i,i] * B[l, i]
    
    # a_sym @ B.T has shape (n, n), (a_sym @ B.T)[i, l] = sum_k a_sym[i, k] * B.T[k, l] = sum_k a_sym[i, k] * B[l, k]
    c = a_sym @ B.T  # shape (n, n)
    
    # Subtract the diagonal terms: for each i, subtract 2*a[i,i] * B[:, i] from column i of c
    # c[i, l] -= 2*a[i,i] * B[l, i]
    diag_a = np.diag(a)
    for i in range(n):
        if diag_a[i] != 0:
            c[i, :] -= 2 * diag_a[i] * B[:, i]
    
    # Now solve the assignment problem: minimize sum_i c[i, p_new[i]]
    # linear_sum_assignment minimizes sum_i c[i, perm[i]]
    row_ind, col_ind = linear_sum_assignment(c)
    p_new = np.empty(n, dtype=np.int64)
    p_new[row_ind] = col_ind
    
    return p_new


def _local_search_2opt(a, b, p, time_limit, a_sym=None, max_passes=10):
    """
    Perform 2-opt local search on permutation p.
    Returns the improved permutation and its cost.
    """
    n = len(a)
    if a_sym is None:
        a_sym = a + a.T
    
    best_p = p.copy()
    best_cost = _qap_cost(a, b, best_p)
    
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n):
            if time.time() > time_limit:
                break
            for j in range(i + 1, n):
                if time.time() > time_limit:
                    break
                delta = _swap_delta(a, b, best_p, i, j, a_sym)
                if delta < 0:
                    best_p[i], best_p[j] = best_p[j], best_p[i]
                    best_cost += delta
                    improved = True
    
    return best_p, best_cost


def _solve_tai256c(a, b, time_limit=110.0):
    n = len(a)
    start_time = time.time()
    
    # Precompute
    a_sym = a + a.T
    
    # Generate initial candidate using rank matching
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.copy()
    
    best_p = p.copy()
    best_cost = _qap_cost(a, b, best_p)
    
    # Main loop: alternate between Hungarian global reassignment and 2-opt polishing
    while time.time() - start_time < time_limit - 5.0:
        # Step 1: Hungarian global reassignment
        p_hung = _hungarian_step(a, b, best_p, a_sym)
        
        # Step 2: 2-opt polishing
        p_polished, cost_polished = _local_search_2opt(
            a, b, p_hung, time_limit - 2.0, a_sym, max_passes=3
        )
        
        if cost_polished < best_cost:
            best_cost = cost_polished
            best_p = p_polished.copy()
        else:
            # If no improvement, try a small random perturbation and then polish
            p_pert = best_p.copy()
            for _ in range(5):
                i, j = random.sample(range(n), 2)
                p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
            
            p_polished2, cost_polished2 = _local_search_2opt(
                a, b, p_pert, time_limit - 2.0, a_sym, max_passes=2
            )
            if cost_polished2 < best_cost:
                best_cost = cost_polished2
                best_p = p_polished2.copy()
    
    # Final polish with more time
    remaining = time_limit - (time.time() - start_time)
    if remaining > 3.0:
        best_p, best_cost = _local_search_2opt(
            a, b, best_p, time_limit - 1.0, a_sym, max_passes=5
        )
    
    return best_p.tolist()


def solve(a, b):
    return _solve_tai256c(a, b)
# EVOLVE-BLOCK-END
