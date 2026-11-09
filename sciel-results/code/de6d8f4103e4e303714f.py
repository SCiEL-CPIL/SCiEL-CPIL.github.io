"""Hypergraph degree-guided greedy + multi-move local search."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

N = 192
K = 3
MAX_ELEM = 48

def _can_add(selected_set, x):
    """Fast check: can x be added to selected_set (a set of ints)?"""
    sel = list(selected_set)
    n = len(sel)
    # Check x: x must not divide sum of any 3 from sel
    if n >= 3:
        for t in itertools.combinations(sel, 3):
            if (t[0] + t[1] + t[2]) % x == 0:
                return False
    # Check each a in sel: a must not divide sum of any 3 from (sel\{a} U {x})
    # New triples involving x: x + pair from sel\{a}
    for i, a in enumerate(sel):
        if n >= 3:
            # triples with x and 2 others from sel excluding a
            others = [v for j, v in enumerate(sel) if j != i]
            for pair in itertools.combinations(others, 2):
                if (x + pair[0] + pair[1]) % a == 0:
                    return False
    return True

def _degree(x, selected_set):
    """Count conflicts if x were in selected_set: how many existing elements would be violated."""
    sel = list(selected_set)
    n = len(sel)
    conflicts = 0
    # x divides sum of any 3 from sel
    if n >= 3:
        for t in itertools.combinations(sel, 3):
            if (t[0] + t[1] + t[2]) % x == 0:
                conflicts += 1
    # each a in sel: a divides x + pair from sel\{a}
    for i, a in enumerate(sel):
        if n >= 3:
            others = [v for j, v in enumerate(sel) if j != i]
            for pair in itertools.combinations(others, 2):
                if (x + pair[0] + pair[1]) % a == 0:
                    conflicts += 1
    return conflicts

def _greedy_degree_guided(seed, bias=0.0):
    """Greedy that picks next element by minimum degree (conflicts), with random tie-breaking."""
    rng = random.Random(seed)
    selected = set()
    candidates = list(range(1, N + 1))
    
    while candidates and len(selected) < MAX_ELEM:
        # Compute degree for each candidate
        best_deg = float('inf')
        best_cands = []
        for x in candidates:
            if not _can_add(selected, x):
                continue
            d = _degree(x, selected)
            # Add bias toward larger numbers
            score = d - bias * x / N
            if score < best_deg - 1e-9:
                best_deg = score
                best_cands = [x]
            elif abs(score - best_deg) < 1e-9:
                best_cands.append(x)
        
        if not best_cands:
            break
        
        # Pick randomly from best candidates
        chosen = rng.choice(best_cands)
        selected.add(chosen)
        candidates.remove(chosen)
    
    return list(selected)

def _greedy_fixed_order(order):
    """Greedy with fixed order."""
    selected = set()
    result = []
    for x in order:
        if _can_add(selected, x):
            selected.add(x)
            result.append(x)
    return result

def _local_search(selected, deadline, max_time_per_pass=2.0):
    """Local search with 1-swap, 2-swap, and 1-add moves."""
    selected = list(selected)
    sel_set = set(selected)
    best_size = len(selected)
    
    pass_start = time.time()
    improved = True
    passes = 0
    
    while improved and time.time() < deadline:
        improved = False
        passes += 1
        
        # Move 1: Try adding one element (if size < MAX_ELEM)
        if len(selected) < MAX_ELEM:
            for x in range(N, 0, -1):
                if x in sel_set:
                    continue
                if _can_add(sel_set, x):
                    selected.append(x)
                    sel_set.add(x)
                    improved = True
                    break
        
        # Move 2: Try 1-swap (remove one, add one)
        if not improved and len(selected) < MAX_ELEM:
            for i in range(len(selected)):
                if time.time() > deadline:
                    return selected
                removed = selected[i]
                remaining = [v for j, v in enumerate(selected) if j != i]
                rem_set = set(remaining)
                for x in range(N, 0, -1):
                    if x == removed or x in rem_set:
                        continue
                    if _can_add(rem_set, x):
                        selected = remaining + [x]
                        sel_set = set(selected)
                        improved = True
                        break
                if improved:
                    break
        
        # Move 3: Try 2-swap (remove two, add two) - can increase size by up to 2
        if not improved and len(selected) < MAX_ELEM and time.time() < deadline - 1:
            n = len(selected)
            for i in range(n):
                if time.time() > deadline:
                    return selected
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        return selected
                    removed = [selected[i], selected[j]]
                    remaining = [v for k, v in enumerate(selected) if k != i and k != j]
                    rem_set = set(remaining)
                    
                    # Try adding two elements
                    added = []
                    for x in range(N, 0, -1):
                        if x in rem_set or x in removed:
                            continue
                        if _can_add(rem_set, x):
                            added.append(x)
                            rem_set.add(x)
                            if len(added) == 2:
                                break
                    
                    if len(added) == 2 and len(remaining) + 2 > len(selected):
                        selected = remaining + added
                        sel_set = set(selected)
                        improved = True
                        break
                if improved:
                    break
        
        # Safety: limit passes to avoid infinite loop
        if passes > 500:
            break
    
    return selected

def construct():
    deadline = time.time() + 85
    
    best = []
    best_size = 0
    
    # Phase 1: Fixed-order greedy variants
    orders = [
        list(range(N, 0, -1)),
        list(range(1, N + 1)),
        list(range(N // 2 + 1, N + 1)) + list(range(N // 2, 0, -1)),
        list(range(2 * N // 3 + 1, N + 1)) + list(range(1, 2 * N // 3 + 1)),
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        result = _greedy_fixed_order(order)
        if len(result) > best_size:
            best = result
            best_size = len(result)
    
    # Phase 2: Degree-guided greedy with various seeds and biases
    for seed in range(30):
        if time.time() > deadline:
            break
        for bias in [0.0, 0.5, 1.0, 2.0]:
            if time.time() > deadline:
                break
            result = _greedy_degree_guided(seed * 100 + int(bias * 10), bias)
            if len(result) > best_size:
                best = result
                best_size = len(result)
    
    # Phase 3: Local search on best
    improved = _local_search(best, deadline)
    if len(improved) > best_size:
        best = improved
        best_size = len(best)
    
    # Phase 4: More random restarts with local search
    for seed in range(30, 150):
        if time.time() > deadline - 3:
            break
        result = _greedy_degree_guided(seed, random.Random(seed).uniform(0, 2))
        if len(result) > best_size:
            best = result
            best_size = len(result)
        # Local search if promising
        if len(result) >= best_size - 1 and time.time() < deadline - 5:
            improved = _local_search(result, min(deadline, time.time() + 5))
            if len(improved) > best_size:
                best = improved
                best_size = len(best)
    
    # Final local search
    if time.time() < deadline - 2:
        improved = _local_search(best, deadline)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    return best

# EVOLVE-BLOCK-END
