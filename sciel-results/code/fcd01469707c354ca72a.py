"""
VND + Simulated Annealing solver for QAPLIB wil100.
Strategy:
1. Hungarian initialization on linear surrogate.
2. Variable Neighborhood Descent (VND) local search: 2-opt, 3-opt, 4-opt.
3. Iterated Local Search with SA acceptance.
4. Final SA polish.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _swap_delta(a, b, p, i, j):
    """Compute cost change when swapping positions i and j in permutation p."""
    pi = p[i]
    pj = p[j]
    
    delta = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    delta += a[i, i] * (b[pj, pj] - b[pi, pi])
    delta += a[j, j] * (b[pi, pi] - b[pj, pj])
    
    diff_a_row = a[i, :] - a[j, :]
    diff_b_col = b[pj, p] - b[pi, p]
    sum_T1 = np.dot(diff_a_row, diff_b_col)
    sum_T1 -= (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    sum_T1 -= (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    
    diff_a_col = a[:, i] - a[:, j]
    diff_b_row = b[p, pj] - b[p, pi]
    sum_T2 = np.dot(diff_a_col, diff_b_row)
    sum_T2 -= (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    sum_T2 -= (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    
    delta += sum_T1 + sum_T2
    return int(delta)

def _cycle_delta(a, b, p, idxs):
    """
    Compute cost change for a cyclic shift of positions idxs.
    idxs = [i1, i2, ..., ik] means p[i1]->p[i2], p[i2]->p[i3], ..., p[ik]->p[i1].
    """
    k = len(idxs)
    if k < 2:
        return 0
    if k == 2:
        return _swap_delta(a, b, p, idxs[0], idxs[1])
    
    # For k-exchange, compute delta by direct cost difference on affected submatrix
    # This is O(k^2 + k*n) but k is small (3 or 4)
    positions = [p[idx] for idx in idxs]
    
    # Old cost contribution from pairs within idxs and between idxs and rest
    # New cost: cycle the positions
    
    # Build the new mapping
    new_pos = {}
    for t in range(k):
        new_pos[idxs[t]] = positions[(t + 1) % k]
    
    # Compute delta: for each pair (u,v) where at least one is in idxs
    # delta = sum over affected pairs of (new - old)
    
    # Affected pairs: (u,v) where u in idxs or v in idxs
    # = sum_{u in idxs, v in idxs} a[u,v]*(b[new[u],new[v]] - b[old[u],old[v]])
    #   + sum_{u in idxs, v not in idxs} a[u,v]*(b[new[u],old[v]] - b[old[u],old[v]])
    #   + sum_{u not in idxs, v in idxs} a[u,v]*(b[old[u],new[v]] - b[old[u],old[v]])
    
    delta = 0
    idxs_set = set(idxs)
    
    # Pairs within idxs
    for u in idxs:
        for v in idxs:
            old_uv = p[u]
            old_vv = p[v]
            new_uv = new_pos[u]
            new_vv = new_pos[v]
            delta += a[u, v] * (b[new_uv, new_vv] - b[old_uv, old_vv])
    
    # Pairs between idxs and rest
    for u in idxs:
        old_u = p[u]
        new_u = new_pos[u]
        # u as first index, v as second (v not in idxs)
        # a[u,v] * (b[new_u, p[v]] - b[old_u, p[v]])
        # = a[u,v] * (b[new_u, p[v]] - b[old_u, p[v]])
        # Sum over v not in idxs:
        # = sum_v a[u,v]*(b[new_u,p[v]] - b[old_u,p[v]]) - sum_{v in idxs} ...
        # We already handled v in idxs above, so subtract those
        
        # Vectorized: sum over all v, then subtract v in idxs
        diff_b_row_u = b[new_u, p] - b[old_u, p]
        sum_u_all = np.dot(a[u, :], diff_b_row_u)
        # Subtract v in idxs
        for v in idxs:
            sum_u_all -= a[u, v] * (b[new_u, p[v]] - b[old_u, p[v]])
        delta += sum_u_all
        
        # v as first index, u as second (v not in idxs)
        # a[v,u] * (b[p[v], new_u] - b[p[v], old_u])
        diff_b_col_u = b[p, new_u] - b[p, old_u]
        sum_v_all = np.dot(a[:, u], diff_b_col_u)
        for v in idxs:
            sum_v_all -= a[v, u] * (b[p[v], new_u] - b[p[v], old_u])
        delta += sum_v_all
    
    return int(delta)

def _vnd_2opt(a, b, p, deadline):
    """First-improvement 2-opt local search. Returns (p, cost, improved)."""
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    improved = True
    while improved and (time.time() < deadline):
        improved = False
        for i in range(n):
            if time.time() >= deadline:
                return p, current_cost, True
            for j in range(i + 1, n):
                delta = _swap_delta(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    break
            if improved:
                break
    return p, current_cost, improved

def _vnd_3opt(a, b, p, deadline):
    """First-improvement 3-opt local search. Returns (p, cost, improved)."""
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    improved = True
    # For 3-opt, we consider all triples (i,j,k) and try the two distinct 3-cycles
    # Cycle 1: i->j, j->k, k->i  (positions cycle)
    # Cycle 2: i->k, k->j, j->i
    while improved and (time.time() < deadline):
        improved = False
        for i in range(n):
            if time.time() >= deadline:
                return p, current_cost, True
            for j in range(i + 1, n):
                if time.time() >= deadline:
                    return p, current_cost, True
                for k in range(j + 1, n):
                    # Cycle: i->j, j->k, k->i
                    d1 = _cycle_delta(a, b, p, [i, j, k])
                    if d1 < 0:
                        # Apply cycle: p[i]=old p[j], p[j]=old p[k], p[k]=old p[i]
                        pi, pj, pk = p[i], p[j], p[k]
                        p[i], p[j], p[k] = pj, pk, pi
                        current_cost += d1
                        improved = True
                        break
                    # Cycle: i->k, k->j, j->i
                    d2 = _cycle_delta(a, b, p, [i, k, j])
                    if d2 < 0:
                        pi, pj, pk = p[i], p[j], p[k]
                        p[i], p[j], p[k] = pk, pi, pj
                        current_cost += d2
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return p, current_cost, improved

def _vnd_4opt(a, b, p, deadline):
    """First-improvement 4-opt local search (limited). Returns (p, cost, improved)."""
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    improved = True
    # For 4-opt, only try a limited set of 4-cycles to keep runtime manageable
    # We'll try: for each pair (i,j), and two other indices k,l, try the cycle i->j->k->l->i
    # This is expensive, so we limit the search
    max_checks = 5000
    checks = 0
    while improved and (time.time() < deadline):
        improved = False
        for i in range(n):
            if time.time() >= deadline or checks >= max_checks:
                return p, current_cost, True
            for j in range(i + 1, n):
                if time.time() >= deadline or checks >= max_checks:
                    return p, current_cost, True
                for k in range(j + 1, n):
                    if time.time() >= deadline or checks >= max_checks:
                        return p, current_cost, True
                    for l in range(k + 1, n):
                        if time.time() >= deadline or checks >= max_checks:
                            return p, current_cost, True
                        checks += 1
                        # Try cycle i->j->k->l->i
                        d = _cycle_delta(a, b, p, [i, j, k, l])
                        if d < 0:
                            pi, pj, pk, pl = p[i], p[j], p[k], p[l]
                            p[i], p[j], p[k], p[l] = pj, pk, pl, pi
                            current_cost += d
                            improved = True
                            break
                        # Try cycle i->l->k->j->i
                        d = _cycle_delta(a, b, p, [i, l, k, j])
                        if d < 0:
                            pi, pj, pk, pl = p[i], p[j], p[k], p[l]
                            p[i], p[j], p[k], p[l] = pl, pi, pj, pk
                            current_cost += d
                            improved = True
                            break
                        # Try cycle i->j->l->k->i
                        d = _cycle_delta(a, b, p, [i, j, l, k])
                        if d < 0:
                            pi, pj, pk, pl = p[i], p[j], p[k], p[l]
                            p[i], p[j], p[k], p[l] = pj, pl, pi, pk
                            current_cost += d
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return p, current_cost, improved

def _vnd_local_search(a, b, p, deadline, max_vnd_passes=3):
    """
    Variable Neighborhood Descent: cycle through 2-opt, 3-opt, 4-opt.
    Stop when no improvement in any neighborhood or deadline reached.
    """
    current_cost = _qap_cost(a, b, p)
    
    for _pass in range(max_vnd_passes):
        if time.time() >= deadline:
            break
        
        # 2-opt
        p, current_cost, imp2 = _vnd_2opt(a, b, p, deadline)
        if not imp2 and time.time() < deadline:
            # Try 3-opt
            p, current_cost, imp3 = _vnd_3opt(a, b, p, deadline)
            if not imp3 and time.time() < deadline:
                # Try 4-opt
                p, current_cost, imp4 = _vnd_4opt(a, b, p, deadline)
                if not imp4:
                    break
            elif imp3:
                # Found improvement in 3-opt, go back to 2-opt
                continue
            else:
                break
        elif imp2:
            # Found improvement in 2-opt, continue
            continue
        else:
            break
    
    return p, current_cost

def _hungarian_initial(a, b):
    """Hungarian initialization on linear surrogate."""
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    C = -np.outer(a_row_sums, b_row_sums)
    row_ind, col_ind = linear_sum_assignment(C)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p

def _perturb(a, b, p, k_swaps=5):
    """Apply k random swaps to perturb the permutation."""
    n = len(p)
    p_pert = p.copy()
    for _ in range(k_swaps):
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i != j:
            p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
    return p_pert

def solve(a, b):
    n = len(a)
    total_deadline = 140.0  # Safety margin before 150s limit
    start_time = time.time()
    
    np.random.seed(42)
    
    # Phase 1: Hungarian initialization
    p = _hungarian_initial(a, b)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Phase 2: VND local search on initial solution
    deadline_vnd = start_time + 60.0
    p, current_cost = _vnd_local_search(a, b, p, deadline_vnd, max_vnd_passes=5)
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
    
    # Phase 3: Iterated Local Search with SA acceptance
    T_start = 500.0
    T_end = 10.0
    p_ls = p.copy()
    cost_ls = current_cost
    
    ild_deadline = start_time + 110.0
    while time.time() < ild_deadline:
        # VND local search
        p_ls, cost_ls = _vnd_local_search(a, b, p_ls, ild_deadline, max_vnd_passes=3)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
        
        if time.time() > ild_deadline - 5:
            break
        
        # Perturbation
        k_swaps = np.random.randint(3, 8)
        p_pert = _perturb(a, b, p_ls, k_swaps)
        cost_pert = _qap_cost(a, b, p_pert)
        
        # SA acceptance
        frac = min(1.0, (time.time() - start_time) / ild_deadline)
        T = T_start * (T_end / T_start) ** frac
        
        delta = cost_pert - cost_ls
        if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-10)):
            p_ls = p_pert
            cost_ls = cost_pert
    
    # Phase 4: Final SA polish from best solution
    p_sa = best_p.copy()
    cost_sa = best_cost
    T_start2 = 200.0
    T_end2 = 1.0
    
    sa_deadline = total_deadline - 2.0
    sa_start = time.time()
    sa_duration = sa_deadline - sa_start
    if sa_duration > 1:
        while time.time() < sa_deadline:
            frac = min(1.0, (time.time() - sa_start) / sa_duration)
            T = T_start2 * (T_end2 / T_start2) ** frac
            
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            
            delta = _swap_delta(a, b, p_sa, i, j)
            
            if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-10)):
                p_sa[i], p_sa[j] = p_sa[j], p_sa[i]
                cost_sa += int(delta)
                
                if cost_sa < best_cost:
                    best_cost = cost_sa
                    best_p = p_sa.copy()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
