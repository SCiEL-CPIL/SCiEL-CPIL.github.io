"""Find maximum cardinality dissociated reciprocal subset of [1,22]."""
import time
import random
import math
from itertools import combinations

# EVOLVE-BLOCK-START

# LCM of 1..22
_LCM = 232792560

def _weights():
    """Return integer weights w_i = LCM / i for i in 1..22."""
    return [_LCM // i for i in range(1, 23)]

_W = _weights()  # _W[i-1] = LCM/i

# Precompute modular inverses for two large primes
_MOD1 = 10**9 + 7
_MOD2 = 10**9 + 9

def _mod_inv(e, mod):
    return pow(e, mod - 2, mod)

# Precompute modular reciprocals
_RECIP_MOD1 = [_mod_inv(i, _MOD1) for i in range(1, 23)]
_RECIP_MOD2 = [_mod_inv(i, _MOD2) for i in range(1, 23)]

def _mod_subset_sums_distinct(elements, mod, recip_mods):
    """Fast check: are all subset sums of reciprocals distinct mod p?
    Returns True if no collision found mod p (necessary condition for exact distinctness)."""
    n = len(elements)
    if n == 0:
        return True
    # Generate subset sums incrementally
    sums = [0]
    for e in elements:
        r = recip_mods[e - 1]
        new_sums = [(s + r) % mod for s in sums]
        all_sums = sums + new_sums
        if len(set(all_sums)) != len(all_sums):
            return False
        sums = all_sums
    return True

def _mod_filter(elements):
    """Two-prime modular filter: necessary condition for exact distinctness."""
    if not _mod_subset_sums_distinct(elements, _MOD1, _RECIP_MOD1):
        return False
    if not _mod_subset_sums_distinct(elements, _MOD2, _RECIP_MOD2):
        return False
    return True

def _exact_subset_sums_distinct(elements):
    """Exact check using integer weights. Returns True if all subset sums are distinct."""
    n = len(elements)
    if n == 0:
        return True
    ws = [_W[e - 1] for e in elements]
    sums = [0]
    for w in ws:
        new_sums = [s + w for s in sums]
        all_sums = sums + new_sums
        if len(set(all_sums)) != len(all_sums):
            return False
        sums = all_sums
    return True

def _is_dissociated(elements):
    """Two-tier check: fast modular filter then exact."""
    if not _mod_filter(elements):
        return False
    return _exact_subset_sums_distinct(elements)

def _is_dissociated_fast(elements):
    """Alias for _is_dissociated."""
    return _is_dissociated(elements)

def _min_gap(elements):
    """Compute the minimum difference between any two distinct subset sums.
    Returns 0 if there's a collision, otherwise the minimum positive gap."""
    n = len(elements)
    if n == 0:
        return float('inf')
    if n > 16:
        return 1  # Too expensive; return sentinel
    ws = [_W[e - 1] for e in elements]
    sums = [0]
    for w in ws:
        new_sums = [s + w for s in sums]
        all_sums = sums + new_sums
        if len(set(all_sums)) != len(all_sums):
            return 0
        sums = all_sums
    sums_sorted = sorted(sums)
    min_gap = float('inf')
    for i in range(1, len(sums_sorted)):
        gap = sums_sorted[i] - sums_sorted[i-1]
        if gap < min_gap:
            min_gap = gap
    return min_gap

def _greedy_construct(pool, deadline):
    """Greedy: add elements one by one if they maintain dissociation."""
    current = []
    for elem in pool:
        if time.time() > deadline:
            break
        if _is_dissociated_fast(current + [elem]):
            current.append(elem)
    return current

def _randomized_greedy(pool, deadline, n_restarts=50):
    """Multiple randomized greedy attempts."""
    best = []
    for _ in range(n_restarts):
        if time.time() > deadline:
            break
        shuffled = list(pool)
        random.shuffle(shuffled)
        result = _greedy_construct(shuffled, deadline)
        if len(result) > len(best):
            best = result
    return best

def _find_set_of_size(target_size, deadline):
    """Find a dissociated set of exactly target_size elements from [1,22] using backtracking."""
    
    def backtrack(remaining_pool, current):
        if time.time() > deadline:
            return None
        if len(current) == target_size:
            if _is_dissociated_fast(current):
                return list(current)
            return None
        
        if len(remaining_pool) < target_size - len(current):
            return None
        
        for i, elem in enumerate(remaining_pool):
            if time.time() > deadline:
                return None
            new_current = current + [elem]
            if _is_dissociated_fast(new_current):
                result = backtrack(remaining_pool[i+1:], new_current)
                if result is not None:
                    return result
        return None
    
    pool = list(range(22, 0, -1))
    return backtrack(pool, [])

def _local_search_refine(initial_set, all_elements, deadline, max_iter=20000):
    """Local search: try all single swaps and additions to improve."""
    current = list(initial_set)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_dissociated = _is_dissociated_fast(current)
        if not current_dissociated:
            break
        
        current_gap = _min_gap(current) if current else float('inf')
        
        # Try adding each element
        for elem in all_elements:
            if elem in current:
                continue
            new_set = current + [elem]
            if _is_dissociated_fast(new_set):
                new_gap = _min_gap(new_set)
                if len(new_set) > len(current) or (len(new_set) == len(current) and new_gap > current_gap):
                    current = new_set
                    current_gap = new_gap
                    improved = True
                    break
        
        # Try swaps
        if not improved:
            for i, elem_out in enumerate(current):
                for elem_in in all_elements:
                    if elem_in in current:
                        continue
                    new_set = current[:i] + current[i+1:] + [elem_in]
                    if _is_dissociated_fast(new_set):
                        new_gap = _min_gap(new_set)
                        if len(new_set) > len(current) or (len(new_set) == len(current) and new_gap > current_gap):
                            current = new_set
                            current_gap = new_gap
                            improved = True
                            break
                if improved:
                    break
    
    return current

def _simulated_annealing(initial_set, all_elements, deadline, n_iterations=50000):
    """Simulated annealing to find larger dissociated sets."""
    current = list(initial_set)
    current_gap = _min_gap(current) if current else float('inf')
    best = list(current)
    best_gap = current_gap
    best_size = len(current)
    
    T = 1.0
    cooling = 0.9999
    
    for it in range(n_iterations):
        if time.time() > deadline:
            break
        if T < 1e-10:
            break
        
        move_type = random.random()
        new_set = list(current)
        
        if move_type < 0.5:
            if len(current) == 0:
                if all_elements:
                    elem = random.choice(all_elements)
                    new_set.append(elem)
                else:
                    continue
            elif not all_elements:
                if current:
                    idx = random.randrange(len(current))
                    new_set.pop(idx)
                else:
                    continue
            else:
                remove_idx = random.randrange(len(current))
                remove_elem = current[remove_idx]
                outside = [e for e in all_elements if e not in current]
                if not outside:
                    new_set.pop(remove_idx)
                else:
                    add_elem = random.choice(outside)
                    new_set.pop(remove_idx)
                    new_set.append(add_elem)
        elif move_type < 0.75:
            outside = [e for e in all_elements if e not in current]
            if outside:
                elem = random.choice(outside)
                new_set.append(elem)
            else:
                continue
        else:
            if current:
                idx = random.randrange(len(current))
                new_set.pop(idx)
            else:
                continue
        
        new_dissociated = _is_dissociated_fast(new_set)
        if not new_dissociated:
            if random.random() > 0.05 * T:
                T *= cooling
                continue
        
        new_size = len(new_set)
        new_gap = _min_gap(new_set) if new_dissociated else 0
        
        if new_dissociated:
            old_score = len(current) * 10000 + (current_gap if current_gap != float('inf') else 0)
            new_score = new_size * 10000 + new_gap
        else:
            old_score = len(current) * 10000 + (current_gap if current_gap != float('inf') else 0)
            new_score = -1000000
        
        delta = new_score - old_score
        
        if delta >= 0 or random.random() < math.exp(delta / max(T, 1e-10)):
            current = new_set
            current_gap = new_gap
            if new_dissociated:
                if new_size > best_size or (new_size == best_size and new_gap > best_gap):
                    best = list(current)
                    best_gap = new_gap
                    best_size = new_size
        
        T *= cooling
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s limit
    
    all_elements = list(range(1, 23))
    
    # Known valid 9-element set
    seed_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best = list(seed_set)
    
    # Phase 1: Greedy from seed with different orderings
    remaining = [i for i in range(1, 23) if i not in seed_set]
    
    for order in [remaining, sorted(remaining, reverse=True), sorted(remaining)]:
        if time.time() > deadline:
            break
        current = list(seed_set)
        for elem in order:
            if time.time() > deadline:
                break
            if _is_dissociated_fast(current + [elem]):
                current.append(elem)
        if len(current) > len(best):
            best = list(current)
    
    # Phase 2: Exhaustive search for sizes 10, 11, 12
    for target in [10, 11, 12]:
        if time.time() > deadline:
            break
        if len(best) >= target:
            continue
        found = _find_set_of_size(target, deadline)
        if found is not None:
            best = found
    
    # Phase 3: Randomized greedy with multiple restarts
    if time.time() < deadline:
        rand_best = _randomized_greedy(all_elements, deadline, n_restarts=200)
        if len(rand_best) > len(best):
            best = rand_best
    
    # Phase 4: Simulated annealing to explore larger sets
    if time.time() < deadline:
        sa_deadline = min(deadline, time.time() + 25)
        sa_best = _simulated_annealing(best, all_elements, sa_deadline, n_iterations=100000)
        if len(sa_best) > len(best):
            best = sa_best
    
    # Phase 5: Local search refinement
    if time.time() < deadline:
        ls_best = _local_search_refine(best, all_elements, deadline, max_iter=50000)
        if len(ls_best) > len(best):
            best = ls_best
    
    # Phase 6: More randomized greedy if time remains
    if time.time() < deadline:
        more_best = _randomized_greedy(all_elements, deadline, n_restarts=300)
        if len(more_best) > len(best):
            best = more_best
    
    # Phase 7: Final local search on best
    if time.time() < deadline and len(best) < 14:
        final_best = _local_search_refine(best, all_elements, deadline, max_iter=50000)
        if len(final_best) > len(best):
            best = final_best
    
    return best

# EVOLVE-BLOCK-END
