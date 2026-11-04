"""QAP solver for wil100: multi-start with vectorized local search and simulated annealing."""
import numpy as np
import time


def _qap_cost(a, b, p):
    """Exact QAP cost for permutation p (facility i -> location p[i])."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _swap_delta(a, b, p, i, j):
    """Compute change in QAP cost when swapping facilities i and j in assignment p.
    Uses vectorized operations for speed."""
    pi, pj = p[i], p[j]
    if pi == pj:
        return 0
    # diff_row[k] = b[pj, pk] - b[pi, pk]
    diff_row = b[pj, p] - b[pi, p]
    # diff_col[k] = b[pk, pj] - b[pk, pi]
    diff_col = b[p, pj] - b[p, pi]
    
    S1 = np.dot(a[i, :], diff_row)
    S2 = np.dot(a[j, :], -diff_row)
    S3 = np.dot(a[:, i], diff_col)
    S4 = np.dot(a[:, j], -diff_col)
    
    total = S1 + S2 + S3 + S4
    
    # Correct for double-counted terms (k=i and k=j)
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
    
    return int(total - Rem + Direct)


def _swap_delta_3opt(a, b, p, i, j, k):
    """Compute change in QAP cost for cyclic swap i->j, j->k, k->i."""
    pi, pj, pk = p[i], p[j], p[k]
    
    old_int = (a[i,j]*b[pi,pj] + a[j,i]*b[pj,pi] +
               a[j,k]*b[pj,pk] + a[k,j]*b[pk,pj] +
               a[k,i]*b[pk,pi] + a[i,k]*b[pi,pk])
    
    new_int = (a[i,j]*b[pj,pk] + a[j,i]*b[pk,pj] +
               a[j,k]*b[pk,pi] + a[k,j]*b[pi,pk] +
               a[k,i]*b[pi,pj] + a[i,k]*b[pj,pi])
    
    delta_int = new_int - old_int
    
    delta_ext = 0
    n = len(p)
    for l in range(n):
        if l == i or l == j or l == k:
            continue
        pl = p[l]
        old_ext = (a[i,l]*b[pi,pl] + a[l,i]*b[pl,pi] +
                   a[j,l]*b[pj,pl] + a[l,j]*b[pl,pj] +
                   a[k,l]*b[pk,pl] + a[l,k]*b[pl,pk])
        new_ext = (a[i,l]*b[pj,pl] + a[l,i]*b[pl,pj] +
                   a[j,l]*b[pk,pl] + a[l,j]*b[pl,pk] +
                   a[k,l]*b[pi,pl] + a[l,k]*b[pl,pi])
        delta_ext += new_ext - old_ext
        
    return int(delta_int + delta_ext)

def _local_search(a, b, p, deadline, best_cost, best_p, max_passes=10):
    """First-improvement 2-opt local search with 3-opt escape."""
    n = len(p)
    cur = _qap_cost(a, b, p)
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
    
    rng = np.random.default_rng()
    
    for _ in range(max_passes):
        if time.time() >= deadline:
            break
        improved = False
        for i in range(n):
            if time.time() >= deadline:
                break
            pi = p[i]
            for j in range(i + 1, n):
                d = _swap_delta(a, b, p, i, j)
                if d < 0:
                    p[i], p[j] = p[j], p[i]
                    cur += d
                    improved = True
                    if cur < best_cost:
                        best_cost = cur
                        best_p = p.copy()
                    break  # First improvement: restart inner loop for this i
        
        if not improved:
            # Try 3-opt escape
            found_3opt = False
            for _ in range(50):
                if time.time() >= deadline:
                    break
                i, j, k = rng.choice(n, 3, replace=False)
                d = _swap_delta_3opt(a, b, p, i, j, k)
                if d < 0:
                    pi, pj, pk = p[i], p[j], p[k]
                    p[i], p[j], p[k] = pj, pk, pi
                    cur += d
                    if cur < best_cost:
                        best_cost = cur
                        best_p = p.copy()
                    found_3opt = True
                    break
            
            if not found_3opt:
                break
    
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
    
    return best_cost, best_p


def _anneal(a, b, p, deadline, best_cost, best_p, rng, t_start=100.0, t_end=0.5, alpha=0.999):
    """Simulated annealing with 2-opt moves."""
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
        d = _swap_delta(a, b, p, i, j)
        if d < 0 or rng.random() < np.exp(-d / max(t, 1e-10)):
            p[i], p[j] = p[j], p[i]
            cur += d
            if cur < best_cost:
                best_cost = cur
                best_p = p.copy()
        t *= alpha
    
    # Final local search polish
    best_cost, best_p = _local_search(a, b, p, deadline, best_cost, best_p, max_passes=5)
    return best_cost, best_p


def _rank_matching_start(a, b):
    """Rank-matching heuristic: heavy flow rows to low distance rows."""
    n = len(a)
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p

def _spectral_init(a, b):
    """Spectral initialization using principal eigenvectors."""
    n = len(a)
    try:
        evals_a, evecs_a = np.linalg.eigh(a)
        evals_b, evecs_b = np.linalg.eigh(b)
        
        idx_a = np.argsort(-evals_a)
        idx_b = np.argsort(-evals_b)
        
        K = 5
        coords_f = evecs_a[:, idx_a[:K]]
        coords_l = evecs_b[:, idx_b[:K]]
        
        proj_f = coords_f[:, 0]
        proj_l = coords_l[:, 0]
        
        order_f = np.argsort(proj_f)
        order_l = np.argsort(proj_l)
        
        p = np.empty(n, dtype=np.int64)
        for k in range(n):
            p[order_f[k]] = order_l[k]
            
        return p
    except Exception:
        return _rank_matching_start(a, b)


def _greedy_start(a, b):
    """Greedy constructive heuristic."""
    n = len(a)
    p = np.zeros(n, dtype=np.int64)
    used = np.zeros(n, dtype=bool)
    order = np.argsort(-a.sum(axis=1), kind="stable")
    
    for idx in order:
        if not np.any(used):
            p[idx] = int(np.argmin(b.sum(axis=1)))
        else:
            placed = np.where(~used)[0]
            # Score each available location
            scores = np.full(n, np.inf)
            avail = np.where(~used)[0]
            for l in avail:
                # Cost of placing idx at l given current partial assignment
                s = 0.0
                for j in placed:
                    s += a[idx, j] * b[l, p[j]] + a[j, idx] * b[p[j], l]
                scores[l] = s
            p[idx] = int(np.argmin(scores))
        used[p[idx]] = True
    
    return p


def _random_start(n, rng):
    """Random permutation."""
    return rng.permutation(n).astype(np.int64)


def solve(a, b):
    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)
    n = len(a)
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin before 120s kill
    
    rng = np.random.default_rng(42)
    
    best_cost = np.iinfo(np.int64).max
    best_p = None
    
    # Phase 1: Multi-start with local search
    starts = []
    
    # Deterministic starts
    p_rank = _rank_matching_start(a, b)
    starts.append(p_rank)
    
    p_spectral = _spectral_init(a, b)
    starts.append(p_spectral)
    
    p_greedy = _greedy_start(a, b)
    starts.append(p_greedy)
    
    # Random starts
    n_random = 5
    for _ in range(n_random):
        p_rand = _random_start(n, rng)
        starts.append(p_rand)
    
    # Allocate time: 60% for local search on starts, 40% for annealing
    ls_deadline = t0 + (deadline - t0) * 0.6
    
    for p0 in starts:
        if time.time() >= ls_deadline:
            break
        p_work = p0.copy()
        best_cost, best_p = _local_search(a, b, p_work, ls_deadline, best_cost, best_p, max_passes=15)
    
    # Phase 2: Simulated annealing from best solution found
    if time.time() < deadline - 10.0 and best_p is not None:
        p_anneal = best_p.copy()
        best_cost, best_p = _anneal(a, b, p_anneal, deadline, best_cost, best_p, rng)
    
    # Phase 3: If time remains, try more random restarts with brief local search
    if time.time() < deadline - 5.0 and best_p is not None:
        while time.time() < deadline - 5.0:
            p_rand = _random_start(n, rng)
            p_work = p_rand.copy()
            # Brief local search
            brief_deadline = min(time.time() + 2.0, deadline - 5.0)
            best_cost, best_p = _local_search(a, b, p_work, brief_deadline, best_cost, best_p, max_passes=3)
    
    return best_p.tolist()
