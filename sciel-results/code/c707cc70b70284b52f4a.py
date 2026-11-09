"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from collections import defaultdict
import numpy as np

# EVOLVE-BLOCK-START

# Precompute all K(2,2,2) constraint groups at module load time
_ALL_TRIPLES = list(itertools.combinations(range(16), 3))
_TRIPLE_TO_IDX = {t: i for i, t in enumerate(_ALL_TRIPLES)}
_N_TRIPLES = len(_ALL_TRIPLES)

# Build constraint groups: each group is a list of 8 triples that form a K(2,2,2)
_CONSTRAINT_GROUPS = []
_seen_configs = set()

for six_points in itertools.combinations(range(16), 6):
    a, b, c, d, e, f = six_points
    pairings = [
        [(a,b), (c,d), (e,f)],
        [(a,b), (c,e), (d,f)],
        [(a,b), (c,f), (d,e)],
        [(a,c), (b,d), (e,f)],
        [(a,c), (b,e), (d,f)],
        [(a,c), (b,f), (d,e)],
        [(a,d), (b,c), (e,f)],
        [(a,d), (b,e), (c,f)],
        [(a,d), (b,f), (c,e)],
        [(a,e), (b,c), (d,f)],
        [(a,e), (b,d), (c,f)],
        [(a,e), (b,f), (c,d)],
        [(a,f), (b,c), (d,e)],
        [(a,f), (b,d), (c,e)],
        [(a,f), (b,e), (c,d)],
    ]
    for pairing in pairings:
        norm_pairs = tuple(tuple(sorted(p)) for p in sorted(pairing))
        if norm_pairs in _seen_configs:
            continue
        _seen_configs.add(norm_pairs)
        (p1, p2, p3) = norm_pairs
        transversals = []
        for x in p1:
            for y in p2:
                for z in p3:
                    transversals.append(tuple(sorted((x, y, z))))
        _CONSTRAINT_GROUPS.append(transversals)

_N_CONSTRAINTS = len(_CONSTRAINT_GROUPS)

# Precompute: for each triple, which constraint groups it belongs to
_TRIPLE_TO_GROUPS = [[] for _ in range(_N_TRIPLES)]
for gi, group in enumerate(_CONSTRAINT_GROUPS):
    for t in group:
        _TRIPLE_TO_GROUPS[_TRIPLE_TO_IDX[t]].append(gi)

# Precompute: for each point, which triples contain it
_POINT_TO_TRIPLES = [[] for _ in range(16)]
for i, t in enumerate(_ALL_TRIPLES):
    for p in t:
        _POINT_TO_TRIPLES[p].append(i)


class _K222Checker:
    """Maintains constraint counts for O(1) validity checks."""
    
    def __init__(self, initial_set=None):
        self.counts = np.zeros(_N_CONSTRAINTS, dtype=np.int32)
        self.in_set = np.zeros(_N_TRIPLES, dtype=np.bool_)
        if initial_set:
            for t in initial_set:
                idx = _TRIPLE_TO_IDX[t]
                self.in_set[idx] = True
                for gi in _TRIPLE_TO_GROUPS[idx]:
                    self.counts[gi] += 1
    
    def is_valid(self, triple):
        """Check if adding triple keeps all constraint groups at most 7."""
        idx = _TRIPLE_TO_IDX[triple]
        if self.in_set[idx]:
            return True
        for gi in _TRIPLE_TO_GROUPS[idx]:
            if self.counts[gi] >= 7:
                return False
        return True
    
    def add(self, triple):
        idx = _TRIPLE_TO_IDX[triple]
        self.in_set[idx] = True
        for gi in _TRIPLE_TO_GROUPS[idx]:
            self.counts[gi] += 1
    
    def remove(self, triple):
        idx = _TRIPLE_TO_IDX[triple]
        self.in_set[idx] = False
        for gi in _TRIPLE_TO_GROUPS[idx]:
            self.counts[gi] -= 1


def _check_addition(triple, current_set):
    """
    Check if adding 'triple' to 'current_set' creates a K(2,2,2).
    Uses precomputed constraint groups for efficiency.
    """
    idx = _TRIPLE_TO_IDX[triple]
    for gi in _TRIPLE_TO_GROUPS[idx]:
        # Count how many of the 8 triples in this group are in current_set
        group = _CONSTRAINT_GROUPS[gi]
        cnt = 0
        for t in group:
            if t in current_set:
                cnt += 1
                if cnt >= 7:
                    return False
    return True


def _greedy_fallback(deadline):
    """Fast greedy construction using constraint-count checker."""
    all_triples = list(range(_N_TRIPLES))
    random.shuffle(all_triples)
    
    checker = _K222Checker()
    current_set = set()
    
    for idx in all_triples:
        if time.time() > deadline - 5:
            break
        t = _ALL_TRIPLES[idx]
        if checker.is_valid(t):
            checker.add(t)
            current_set.add(t)
    
    return current_set


def _solve_ilp(time_limit=80.0):
    """
    Solve the MILP to maximize the number of triples.
    Returns a set of triples or None if it fails.
    """
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import csr_matrix
    except ImportError:
        return None, 0.0
    
    start_time = time.time()
    
    n_vars = _N_TRIPLES
    
    # Build constraint matrix using precomputed groups
    row_idx = []
    col_idx = []
    data = []
    
    for i, group in enumerate(_CONSTRAINT_GROUPS):
        for t in group:
            row_idx.append(i)
            col_idx.append(_TRIPLE_TO_IDX[t])
            data.append(1)
    
    A = csr_matrix((data, (row_idx, col_idx)), shape=(_N_CONSTRAINTS, n_vars))
    
    c_obj = -np.ones(n_vars)
    
    bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
    
    lb = np.zeros(_N_CONSTRAINTS)
    ub = np.full(_N_CONSTRAINTS, 7)
    lin_cons = LinearConstraint(A, lb=lb, ub=ub)
    
    integrality = np.ones(n_vars)
    
    result = milp(
        c=c_obj,
        constraints=lin_cons,
        integrality=integrality,
        bounds=bounds,
        options={"time_limit": time_limit, "mip_rel_gap": 0.05}
    )
    
    elapsed = time.time() - start_time
    
    if result.x is not None:
        selected = set()
        for i, val in enumerate(result.x):
            if val > 0.5:
                selected.add(_ALL_TRIPLES[i])
        if len(selected) > 0:
            return selected, elapsed
    
    return None, elapsed


def _local_search(initial_set, time_limit=10.0):
    """Improve the triple set using local search with fast checker."""
    start_time = time.time()
    checker = _K222Checker(initial_set)
    current_set = set(initial_set)
    
    # Precompute list of triple indices
    all_indices = list(range(_N_TRIPLES))
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try to add a missing triple
        missing = [i for i in all_indices if not checker.in_set[i]]
        random.shuffle(missing)
        
        for idx in missing:
            t = _ALL_TRIPLES[idx]
            if checker.is_valid(t):
                checker.add(t)
                current_set.add(t)
                improved = True
                break
        
        if not improved:
            if time.time() - start_time > time_limit * 0.85:
                break
            
            # Try swap: remove one, try to add multiple
            current_list = list(current_set)
            random.shuffle(current_list)
            
            for t_rem in current_list[:15]:
                idx_rem = _TRIPLE_TO_IDX[t_rem]
                checker.remove(t_rem)
                current_set.remove(t_rem)
                
                added_count = 0
                added_triples = []
                
                missing = [i for i in all_indices if not checker.in_set[i] and i != idx_rem]
                random.shuffle(missing)
                
                for idx_add in missing:
                    t_add = _ALL_TRIPLES[idx_add]
                    if checker.is_valid(t_add):
                        checker.add(t_add)
                        current_set.add(t_add)
                        added_triples.append(t_add)
                        added_count += 1
                
                if added_count > 1:
                    improved = True
                    break
                else:
                    # Revert
                    for t_add in added_triples:
                        checker.remove(t_add)
                        current_set.remove(t_add)
                    checker.add(t_rem)
                    current_set.add(t_rem)
            
            if not improved:
                break

    return current_set


def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    g_set = _greedy_fallback(deadline)
    best_set = g_set
    best_size = len(g_set)
    
    remaining = deadline - time.time() - 15.0
    if remaining > 25.0:
        ilp_set, elapsed = _solve_ilp(time_limit=remaining)
        if ilp_set is not None and len(ilp_set) > best_size:
            best_set = ilp_set
            best_size = len(ilp_set)
    
    remaining = deadline - time.time() - 5.0
    if remaining > 2.0:
        ls_set = _local_search(best_set, time_limit=remaining)
        if len(ls_set) > best_size:
            best_set = ls_set
            best_size = len(ls_set)
    
    return [list(t) for t in best_set]
# EVOLVE-BLOCK-END
