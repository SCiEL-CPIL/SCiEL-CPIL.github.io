"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def _build_conflict_bitmasks():
    """
    For each element x (0-indexed: 0..127), precompute a bitmask of elements
    that form a bad triple with x and some other already-selected element.
    
    conflict_mask[x] is a bitmask where bit y is set if there exists some z
    such that {x, y, z} is a bad triple. Actually, we need: for each x, a
    bitmask indicating which pairs (y,z) conflict. But for O(1) check, we
    precompute: for each x, a list of (y, z) pairs. Then for a given selected
    bitmask S, x can be added iff for all (y,z) in pairs[x], NOT (bit y AND bit z in S).
    
    Better approach: for each x, precompute conflict_mask[x] as a bitmask of
    all elements y such that there exists z with {x,y,z} bad. Then x can be
    added to S iff (conflict_mask[x] & S) == 0? No, that's not right either
    because we need BOTH y and z to be in S.
    
    Correct approach: For each x, store a list of pairs. For fast checking,
    we can precompute for each x a set of "forbidden pairs" and check if any
    pair is fully contained in S. With 128 elements, we can use a different
    strategy: for each x, precompute a 128-bit mask pair_mask[x] where for
    each y in S, we check if there's a z in S such that (y,z) conflicts with x.
    
    Most efficient: For each x, precompute conflict_pairs[x] = list of (y,z).
    Then can_add(x, S) = all(not (y in S and z in S) for y,z in conflict_pairs[x]).
    With bitmask: for each (y,z), check (S >> y) & 1 and (S >> z) & 1.
    """
    n = 128
    # Index: element value v -> index v-1 (0-based)
    conflict_pairs = [[] for _ in range(n)]
    nums = list(range(1, 129))
    for a, b, c in combinations(nums, 3):
        lab = _lcm(a, b)
        lac = _lcm(a, c)
        lbc = _lcm(b, c)
        if lab == lac == lbc:
            ia, ib, ic = a - 1, b - 1, c - 1
            conflict_pairs[ia].append((ib, ic))
            conflict_pairs[ib].append((ia, ic))
            conflict_pairs[ic].append((ia, ib))
    return conflict_pairs

def _can_add_mask(x_idx, selected_mask, conflict_pairs):
    """Check if element (x_idx+1) can be added to selected_mask without creating bad triple."""
    for y, z in conflict_pairs[x_idx]:
        if (selected_mask >> y) & 1 and (selected_mask >> z) & 1:
            return False
    return True

def _greedy_smart(conflict_pairs, order=None):
    """Greedy with conflict-degree-based ordering using bitmask."""
    n = 128
    if order is None:
        # Sort by conflict degree (fewer conflicts first)
        order = sorted(range(n), key=lambda x: len(conflict_pairs[x]))
    
    selected_mask = 0
    selected = []
    for x_idx in order:
        if _can_add_mask(x_idx, selected_mask, conflict_pairs):
            selected_mask |= (1 << x_idx)
            selected.append(x_idx + 1)
    return selected

def _local_search(selected, conflict_pairs, time_limit=10):
    """Local search: try removing one element and adding another."""
    n = 128
    selected_mask = 0
    for v in selected:
        selected_mask |= (1 << (v - 1))
    
    best_mask = selected_mask
    best_size = len(selected)
    start = time.time()
    
    while time.time() - start < time_limit:
        improved = False
        # Try each removal
        for rem in selected:
            rem_idx = rem - 1
            candidate_mask = selected_mask & ~(1 << rem_idx)
            # Try each addition
            for add_idx in range(n):
                if (candidate_mask >> add_idx) & 1:
                    continue
                if _can_add_mask(add_idx, candidate_mask, conflict_pairs):
                    new_mask = candidate_mask | (1 << add_idx)
                    new_size = bin(new_mask).count('1')
                    if new_size > best_size:
                        best_mask = new_mask
                        best_size = new_size
                        selected_mask = new_mask
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break
    
    # Convert back to list
    best = []
    for i in range(n):
        if (best_mask >> i) & 1:
            best.append(i + 1)
    return best

def _simulated_annealing(selected, conflict_pairs, time_limit=15):
    """Simulated annealing to escape local optima using bitmask."""
    n = 128
    current_mask = 0
    for v in selected:
        current_mask |= (1 << (v - 1))
    
    best_mask = current_mask
    best_size = len(selected)
    
    start = time.time()
    temperature = 5.0
    cooling = 0.95
    
    while time.time() - start < time_limit and temperature > 0.1:
        # Pick a random element to remove
        current_list = [i + 1 for i in range(n) if (current_mask >> i) & 1]
        if not current_list:
            break
        rem = random.choice(current_list)
        rem_idx = rem - 1
        candidate_mask = current_mask & ~(1 << rem_idx)
        
        # Try to add a random element
        available = [i for i in range(n) if not ((candidate_mask >> i) & 1)]
        if not available:
            temperature *= cooling
            continue
        add_idx = random.choice(available)
        
        if not _can_add_mask(add_idx, candidate_mask, conflict_pairs):
            temperature *= cooling
            continue
        
        new_mask = candidate_mask | (1 << add_idx)
        new_size = bin(new_mask).count('1')
        
        # Accept if better, or with probability based on temperature
        if new_size >= best_size or random.random() < (1.0 / (temperature + 0.1)):
            current_mask = new_mask
            if new_size > best_size:
                best_mask = new_mask
                best_size = new_size
        
        temperature *= cooling
    
    best = []
    for i in range(n):
        if (best_mask >> i) & 1:
            best.append(i + 1)
    return best

def _two_swap_search(selected, conflict_pairs, time_limit=10):
    """Try removing 2 and adding 2 elements simultaneously."""
    n = 128
    selected_mask = 0
    for v in selected:
        selected_mask |= (1 << (v - 1))
    
    best_mask = selected_mask
    best_size = len(selected)
    start = time.time()
    
    selected_list = [i + 1 for i in range(n) if (selected_mask >> i) & 1]
    not_selected = [i + 1 for i in range(n) if not ((selected_mask >> i) & 1)]
    
    while time.time() - start < time_limit:
        improved = False
        if len(selected_list) < 2 or len(not_selected) < 2:
            break
        for rem1 in selected_list:
            for rem2 in selected_list:
                if rem2 <= rem1:
                    continue
                r1, r2 = rem1 - 1, rem2 - 1
                candidate_mask = selected_mask & ~(1 << r1) & ~(1 << r2)
                for add1 in not_selected:
                    for add2 in not_selected:
                        if add2 <= add1:
                            continue
                        a1, a2 = add1 - 1, add2 - 1
                        if _can_add_mask(a1, candidate_mask, conflict_pairs) and \
                           _can_add_mask(a2, candidate_mask | (1 << a1), conflict_pairs):
                            new_mask = candidate_mask | (1 << a1) | (1 << a2)
                            new_size = bin(new_mask).count('1')
                            if new_size > best_size:
                                best_mask = new_mask
                                best_size = new_size
                                selected_mask = new_mask
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    best = []
    for i in range(n):
        if (best_mask >> i) & 1:
            best.append(i + 1)
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    conflict_pairs = _build_conflict_bitmasks()
    
    best = []
    best_size = 0
    
    # Initial greedy with smart ordering
    sel = _greedy_smart(conflict_pairs)
    best = sel
    best_size = len(sel)
    
    # Local search
    if time.time() < deadline - 10:
        improved = _local_search(best, conflict_pairs, time_limit=15)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    # Multiple restarts with different orderings
    random.seed(42)
    for restart in range(120):
        if time.time() > deadline - 5:
            break
        
        # Different ordering strategies
        if restart % 4 == 0:
            order = list(range(128))
            random.shuffle(order)
        elif restart % 4 == 1:
            order = sorted(range(128), key=lambda x: random.random())
        elif restart % 4 == 2:
            order = sorted(range(128), key=lambda x: (len(conflict_pairs[x]), random.random()))
        else:
            # Weighted by prime factors (heuristic: numbers with more distinct prime factors have more conflicts)
            def num_pf(x):
                v = x + 1
                cnt = 0
                d = 2
                while d * d <= v:
                    if v % d == 0:
                        cnt += 1
                        while v % d == 0:
                            v //= d
                    d += 1
                if v > 1:
                    cnt += 1
                return cnt
            order = sorted(range(128), key=lambda x: (num_pf(x), random.random()))
        
        sel = _greedy_smart(conflict_pairs, order)
        if len(sel) > best_size:
            best = sel
            best_size = len(sel)
        
        # Local search on promising solutions
        if len(sel) >= best_size - 1 and time.time() < deadline - 15:
            improved = _local_search(sel, conflict_pairs, time_limit=3)
            if len(improved) > best_size:
                best = improved
                best_size = len(best)
        
        # Two-swap search occasionally
        if restart % 8 == 4 and time.time() < deadline - 20:
            improved = _two_swap_search(best, conflict_pairs, time_limit=5)
            if len(improved) > best_size:
                best = improved
                best_size = len(best)
        
        # Simulated annealing on best solution
        if restart % 5 == 3 and time.time() < deadline - 10:
            improved = _simulated_annealing(best, conflict_pairs, time_limit=8)
            if len(improved) > best_size:
                best = improved
                best_size = len(best)
    
    # Final local search
    if time.time() < deadline - 3 and len(best) > 0:
        improved = _local_search(best, conflict_pairs, time_limit=15)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    # Final two-swap
    if time.time() < deadline - 1 and len(best) > 0:
        improved = _two_swap_search(best, conflict_pairs, time_limit=5)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    if not best:
        return [1]
    
    return best
# EVOLVE-BLOCK-END
