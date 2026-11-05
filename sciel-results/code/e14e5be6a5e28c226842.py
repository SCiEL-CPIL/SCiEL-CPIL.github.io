import time
import random
import numpy as np
import scipy.optimize as opt
from scipy.sparse import csr_matrix

_N = 256
_DEADLINE = 85.0

def _build_conflict():
    """Build conflict adjacency: conflict[i][j] = True iff (i*j)%(i+j)==0 for i!=j."""
    conflict = [[False] * (_N + 1) for _ in range(_N + 1)]
    neighbors = [set() for _ in range(_N + 1)]
    for i in range(1, _N + 1):
        for j in range(i + 1, _N + 1):
            if (i * j) % (i + j) == 0:
                conflict[i][j] = True
                conflict[j][i] = True
                neighbors[i].add(j)
                neighbors[j].add(i)
    return conflict, neighbors

def _validate(S):
    """Check that S is a valid independent set."""
    S_set = set(S)
    for a in S:
        for b in S:
            if a != b and (a * b) % (a + b) == 0:
                return False
    return len(S) > 0

def _can_add(x, S_set, neighbors):
    return not (neighbors[x] & S_set)

def _greedy_init(neighbors, mode='degree'):
    if mode == 'degree':
        degrees = [len(neighbors[i]) for i in range(_N + 1)]
        order = list(range(1, _N + 1))
        order.sort(key=lambda x: (degrees[x], x))
    elif mode == 'random':
        order = list(range(1, _N + 1))
        random.shuffle(order)
    else:
        order = list(range(1, _N + 1))
    S = []
    S_set = set()
    for x in order:
        if _can_add(x, S_set, neighbors):
            S.append(x)
            S_set.add(x)
    return S, S_set

def _local_search(S, neighbors, time_limit):
    S = list(S)
    S_set = set(S)
    complement = set(range(1, _N + 1)) - S_set
    start = time.time()
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        # Add any available node, sorted by degree (prefer low-degree nodes)
        for x in sorted(complement, key=lambda v: len(neighbors[v])):
            if _can_add(x, S_set, neighbors):
                S.append(x)
                S_set.add(x)
                complement.discard(x)
                improved = True
        if time.time() >= time_limit:
            break
        # Try 1-for-k swaps with greedy refill
        if not improved:
            best_size = len(S)
            best_new_set = None
            for u in sorted(S, key=lambda v: -len(neighbors[v])):
                if time.time() >= time_limit:
                    break
                S_set.remove(u)
                new_S_set = set(S_set)
                # Try to add as many as possible from complement
                for x in sorted(complement, key=lambda v: len(neighbors[v])):
                    if _can_add(x, new_S_set, neighbors):
                        new_S_set.add(x)
                if len(new_S_set) > best_size:
                    best_size = len(new_S_set)
                    best_new_set = new_S_set
                S_set.add(u)
            if best_new_set is not None:
                S = list(best_new_set)
                S_set = best_new_set
                complement = set(range(1, _N + 1)) - S_set
                improved = True
    return S

def _milp_max_is(conflict, neighbors, time_limit):
    """
    Exact maximum independent set via MILP.
    Maximize sum(x_i) s.t. x_i + x_j <= 1 for each edge, x_i in {0,1}.
    """
    start = time.time()
    n = _N
    c = np.full(n, -1.0)
    
    rows = []
    cols = []
    vals = []
    rhs = []
    for i in range(1, _N + 1):
        for j in neighbors[i]:
            if i < j:
                idx = len(rhs)
                rows.append(idx)
                cols.append(i - 1)
                vals.append(1.0)
                rows.append(idx)
                cols.append(j - 1)
                vals.append(1.0)
                rhs.append(1.0)
    
    if len(rhs) == 0:
        return list(range(1, _N + 1))
    
    A = csr_matrix((vals, (rows, cols)), shape=(len(rhs), n))
    bounds = [(0, 1)] * n
    integrality = [1] * n
    
    remaining_time = time_limit - (time.time() - start)
    if remaining_time < 1:
        S, _ = _greedy_init(neighbors, 'degree')
        return S
    
    try:
        res = opt.milp(
            c=c,
            constraints=opt.LinearConstraint(A, lb=-np.inf, ub=rhs),
            integrality=integrality,
            bounds=bounds,
            options={'time_limit': remaining_time}
        )
        if res.success or res.status in (1, 2, 3, 4):
            x = res.x
            S = [i + 1 for i in range(n) if x[i] > 0.5]
            if _validate(S):
                return S
    except Exception:
        pass
    
    S, _ = _greedy_init(neighbors, 'degree')
    return S

def _two_swap_search(S, neighbors, time_limit):
    S = list(S)
    S_set = set(S)
    remaining = set(range(1, _N + 1)) - S_set
    start = time.time()
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        S_list = sorted(S, key=lambda v: -len(neighbors[v]))
        n = len(S_list)
        for i in range(n):
            if time.time() >= time_limit:
                break
            for j in range(i + 1, n):
                if time.time() >= time_limit:
                    break
                r1, r2 = S_list[i], S_list[j]
                S_set.discard(r1)
                S_set.discard(r2)
                new_S_set = set(S_set)
                added = 0
                for x in sorted(remaining, key=lambda v: len(neighbors[v])):
                    if _can_add(x, new_S_set, neighbors):
                        new_S_set.add(x)
                        added += 1
                if added > 2:
                    S = list(new_S_set)
                    S_set = new_S_set
                    remaining = set(range(1, _N + 1)) - S_set
                    improved = True
                    break
                else:
                    S_set.add(r1)
                    S_set.add(r2)
            if improved:
                break
    return S

def construct():
    t0 = time.time()
    conflict, neighbors = _build_conflict()
    
    # Phase 1: Greedy initialization (fast, gives a baseline)
    S_greedy, _ = _greedy_init(neighbors, 'degree')
    best_S = S_greedy
    
    # Phase 2: Local search polish on greedy
    if time.time() - t0 < _DEADLINE - 20:
        S_ls = _local_search(best_S, neighbors, _DEADLINE - 20)
        if len(S_ls) > len(best_S):
            best_S = S_ls
    
    # Phase 3: Try MILP for exact/near-exact maximum independent set
    if time.time() - t0 < _DEADLINE - 15:
        milp_time = _DEADLINE - 18
        S_milp = _milp_max_is(conflict, neighbors, milp_time)
        if _validate(S_milp) and len(S_milp) > len(best_S):
            best_S = S_milp
    
    # Phase 4: If MILP improved, do more local search
    if time.time() - t0 < _DEADLINE - 10:
        S_ls2 = _local_search(best_S, neighbors, _DEADLINE - 10)
        if len(S_ls2) > len(best_S):
            best_S = S_ls2
    
    # Phase 5: Two-swap search
    if time.time() - t0 < _DEADLINE - 5:
        S_2swap = _two_swap_search(best_S, neighbors, _DEADLINE - 5)
        if len(S_2swap) > len(best_S):
            best_S = S_2swap
    
    # Phase 6: Final local search
    if time.time() - t0 < _DEADLINE - 2:
        S_final = _local_search(best_S, neighbors, _DEADLINE - 2)
        if len(S_final) > len(best_S):
            best_S = S_final
    
    # Ensure validity
    if not _validate(best_S):
        best_S = list(range(1, _N + 1, 2))
    
    return best_S
