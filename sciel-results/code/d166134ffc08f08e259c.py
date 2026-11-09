"""Improved search for minimum maximal no-isosceles set on N x N grid."""
N = 16
import time
import random
import itertools


def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _is_iso(a, b, c):
    d1 = _d2(a, b)
    d2_ = _d2(a, c)
    d3 = _d2(b, c)
    return d1 == d2_ or d1 == d3 or d2_ == d3


def _can_add(p, chosen):
    """Check if point p can be added to chosen without creating isosceles triangle."""
    for i in range(len(chosen)):
        for j in range(i + 1, len(chosen)):
            if _is_iso(p, chosen[i], chosen[j]):
                return False
    return True


def _is_maximal(chosen, all_points):
    """Check if no point can be added."""
    chosen_set = set(chosen)
    for p in all_points:
        if p in chosen_set:
            continue
        if _can_add(p, chosen):
            return False
    return True


def _is_valid_no_iso(chosen):
    """Verify no three points form isosceles triangle."""
    for i in range(len(chosen)):
        for j in range(i + 1, len(chosen)):
            for k in range(j + 1, len(chosen)):
                if _is_iso(chosen[i], chosen[j], chosen[k]):
                    return False
    return True


def _greedy(order, chosen=None):
    """Greedy construction following given point order."""
    if chosen is None:
        chosen = []
    else:
        chosen = list(chosen)
    chosen_set = set(chosen)
    for p in order:
        if p not in chosen_set and _can_add(p, chosen):
            chosen.append(p)
            chosen_set.add(p)
    return chosen


def _construct_maximal(chosen, all_points):
    """Add points greedily until maximal."""
    chosen = list(chosen)
    chosen_set = set(chosen)
    changed = True
    while changed:
        changed = False
        for p in all_points:
            if p in chosen_set:
                continue
            if _can_add(p, chosen):
                chosen.append(p)
                chosen_set.add(p)
                changed = True
    return chosen


def _remove_and_remaximize(current, all_points, idx):
    """Remove point at idx, then make maximal again. Return new set if valid."""
    new_set = list(current)
    new_set.pop(idx)
    
    # Check validity (no isosceles)
    valid = True
    for i in range(len(new_set)):
        for j in range(i + 1, len(new_set)):
            for k in range(j + 1, len(new_set)):
                if _is_iso(new_set[i], new_set[j], new_set[k]):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        return None
    
    # Make maximal
    extended = _construct_maximal(new_set, all_points)
    return extended


def _local_search(all_points, start_set, max_iters=5000, deadline=None):
    """Local search: try removing each point and re-maximizing, keep if better."""
    current = list(start_set)
    best = list(current)
    best_len = len(current)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for idx in range(len(current)):
            if deadline is not None and time.time() > deadline:
                break
            extended = _remove_and_remaximize(current, all_points, idx)
            if extended is not None and len(extended) < len(current):
                current = extended
                improved = True
                break
        
        if not improved:
            break
        
        if len(current) < best_len:
            best = list(current)
            best_len = len(best)
    
    return best


def _simulated_annealing(all_points, start_set, max_iters=20000, deadline=None):
    """
    Simulated annealing to minimize the size of a maximal no-isosceles set.
    Moves: remove a point, then try to re-maximize.
    """
    current = list(start_set)
    best = list(current)
    best_len = len(current)
    
    T = 2.0
    T_min = 0.01
    cooling = 0.999
    
    rng = random.Random(12345)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        if len(current) < best_len:
            best = list(current)
            best_len = len(current)
        
        if T < T_min:
            break
        
        # Random move: try to remove a random point and re-maximize
        if current:
            idx = rng.randint(0, len(current) - 1)
            extended = _remove_and_remaximize(current, all_points, idx)
            
            if extended is not None:
                delta = len(extended) - len(current)
                if delta <= 0 or rng.random() < (1.0 - delta) / T:
                    current = extended
        
        T *= cooling
    
    return best


def _backtracking_search(all_points, best_len, max_nodes=200000, deadline=None):
    """
    Backtracking search with strong pruning.
    Tries to find a solution smaller than best_len.
    """
    best = [None]
    best_len_ref = [best_len]
    nodes = [0]
    
    def _can_add_fast(p, chosen):
        for i in range(len(chosen)):
            for j in range(i + 1, len(chosen)):
                if _is_iso(p, chosen[i], chosen[j]):
                    return False
        return True
    
    def _is_maximal_fast(chosen):
        chosen_set = set(chosen)
        for p in all_points:
            if p in chosen_set:
                continue
            if _can_add_fast(p, chosen):
                return False
        return True
    
    def _search(chosen, remaining):
        if deadline is not None and time.time() > deadline:
            return
        if nodes[0] >= max_nodes:
            return
        nodes[0] += 1
        
        if len(chosen) >= best_len_ref[0]:
            return
        
        if not remaining:
            if _is_maximal_fast(chosen):
                if len(chosen) < best_len_ref[0]:
                    best_len_ref[0] = len(chosen)
                    best[0] = list(chosen)
            return
        
        p = remaining[0]
        rest = remaining[1:]
        
        # Option 1: include p if valid
        if _can_add_fast(p, chosen):
            _search(chosen + [p], rest)
        
        # Option 2: exclude p
        _search(chosen, rest)
    
    # Use a heuristic ordering
    pts = list(all_points)
    pts.sort(key=lambda p: (p[0] + p[1], p[0]))
    
    _search([], pts)
    
    return best[0] if best[0] is not None else None


def _diversified_greedy(all_points, rng, num_orderings=5):
    """Generate multiple diverse greedy orderings and return the best maximal set."""
    best = None
    best_len = float('inf')
    
    for _ in range(num_orderings):
        pts = all_points[:]
        rng.shuffle(pts)
        chosen = _greedy(pts)
        extended = _construct_maximal(chosen, all_points)
        if len(extended) < best_len:
            best = extended
            best_len = len(extended)
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 280  # safety margin before 300s limit
    
    all_points = list(itertools.product(range(N), repeat=2))
    
    best = None
    best_len = float('inf')
    
    # Phase 1: Quick greedy constructions with various deterministic orderings
    orderings = [
        ("row_major", list(itertools.product(range(N), repeat=2))),
        ("col_major", sorted(all_points, key=lambda p: (p[1], p[0]))),
        ("diag", sorted(all_points, key=lambda p: (p[0] + p[1], p[0]))),
        ("checker", [p for p in all_points if (p[0] + p[1]) % 2 == 0] +
                   [p for p in all_points if (p[0] + p[1]) % 2 == 1]),
        ("anti_diag", sorted(all_points, key=lambda p: (abs(p[0] - p[1]), p[0]))),
    ]
    
    for name, order in orderings:
        if time.time() > deadline:
            break
        chosen = _greedy(order)
        extended = _construct_maximal(chosen, all_points)
        if len(extended) < best_len:
            best = extended
            best_len = len(extended)
    
    # Phase 2: Backtracking search to find smaller solution
    if best is not None and time.time() < deadline - 30:
        bt_result = _backtracking_search(all_points, best_len, max_nodes=100000,
                                         deadline=deadline - 30)
        if bt_result is not None and len(bt_result) < best_len:
            best = bt_result
            best_len = len(best)
    
    # Phase 3: Local search from best
    if best is not None and time.time() < deadline - 20:
        local_best = _local_search(all_points, best, max_iters=20000,
                                   deadline=deadline - 20)
        if len(local_best) < best_len:
            best = local_best
            best_len = len(best)
    
    # Phase 4: Simulated annealing
    if best is not None and time.time() < deadline - 10:
        sa_best = _simulated_annealing(all_points, best, max_iters=30000,
                                       deadline=deadline - 10)
        if len(sa_best) < best_len:
            best = sa_best
            best_len = len(best)
    
    # Phase 5: Randomized restarts with local search
    rng = random.Random(42)
    attempts = 0
    while time.time() < deadline - 5:
        attempts += 1
        
        # Diversified greedy
        extended = _diversified_greedy(all_points, rng, num_orderings=3)
        if extended is not None and len(extended) < best_len:
            best = extended
            best_len = len(best)
        
        # Occasionally do local search
        if attempts % 5 == 0 and time.time() < deadline - 10:
            local = _local_search(all_points, extended, max_iters=3000,
                                  deadline=deadline - 10)
            if len(local) < best_len:
                best = local
                best_len = len(best)
    
    # Phase 6: Final polish - try more local search iterations
    if best is not None and time.time() < deadline - 2:
        final = _local_search(all_points, best, max_iters=10000,
                              deadline=deadline - 2)
        if len(final) < best_len:
            best = final
            best_len = len(best)
    
    # Final verification
    if best is not None:
        if not _is_valid_no_iso(best):
            best = _construct_maximal(_greedy(all_points), all_points)
        if not _is_maximal(best, all_points):
            best = _construct_maximal(best, all_points)
    
    return best
