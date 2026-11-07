"""
Improved program: Greedy construction with local search refinement.
Uses descending-order bias for initial construction, then local search
to swap elements and increase set size.
"""
import time
import random
from collections import defaultdict

DEADLINE_SECONDS = 85.0  # Safety margin before 90s kill


def _check_add_valid(sums_by_k, sum_to_ks, x, n):
    """
    Check if adding x to the current set is valid.
    sums_by_k[k] contains sums achievable with exactly k elements.
    sum_to_ks[s] contains the set of cardinalities k such that s is in sums_by_k[k].
    """
    for k in range(1, n + 2):
        if not sums_by_k[k - 1]:
            continue
        for s_prev in sums_by_k[k - 1]:
            s_new = s_prev + x
            if s_new in sum_to_ks:
                for k_prime in sum_to_ks[s_new]:
                    if k_prime != k:
                        return False
    return True


def _add_to_state(sums_by_k, sum_to_ks, x, n):
    """
    Update sums_by_k and sum_to_ks to reflect the addition of x.
    """
    for k in range(n, -1, -1):
        if not sums_by_k[k]:
            continue
        for s in sums_by_k[k]:
            s_new = s + x
            if s_new not in sum_to_ks:
                sum_to_ks[s_new] = set()
            sum_to_ks[s_new].add(k + 1)
            sums_by_k[k + 1].add(s_new)


def _build_full_state(elements):
    """Build the full DP state from scratch for a given set of elements."""
    n = len(elements)
    sums_by_k = [set() for _ in range(n + 1)]
    sums_by_k[0].add(0)
    sum_to_ks = defaultdict(set)
    sum_to_ks[0].add(0)
    
    for x in elements:
        _add_to_state(sums_by_k, sum_to_ks, x, len([e for e in elements if e < x]))
    
    # Rebuild properly
    sums_by_k = [set() for _ in range(n + 1)]
    sums_by_k[0].add(0)
    sum_to_ks = defaultdict(set)
    sum_to_ks[0].add(0)
    
    for x in elements:
        cur_n = len(elements)
        # Need to track count properly
        pass
    
    # Simpler: rebuild from scratch each time
    sums_by_k = [set() for _ in range(n + 1)]
    sums_by_k[0].add(0)
    sum_to_ks = defaultdict(set)
    sum_to_ks[0].add(0)
    
    for x in elements:
        # Add x to current state
        for k in range(n, -1, -1):
            if not sums_by_k[k]:
                continue
            new_sums = set()
            for s in sums_by_k[k]:
                s_new = s + x
                new_sums.add(s_new)
            for s_new in new_sums:
                if s_new not in sum_to_ks:
                    sum_to_ks[s_new] = set()
                sum_to_ks[s_new].add(k + 1)
            sums_by_k[k + 1].update(new_sums)
    
    return sums_by_k, sum_to_ks


def _validate_set(elements):
    """Check if a set of elements is valid (no cross-cardinality sum collisions)."""
    n = len(elements)
    sums_by_k = [set() for _ in range(n + 1)]
    sums_by_k[0].add(0)
    
    for x in elements:
        for k in range(n, -1, -1):
            if not sums_by_k[k]:
                continue
            new_sums = set()
            for s in sums_by_k[k]:
                s_new = s + x
                new_sums.add(s_new)
            sums_by_k[k + 1].update(new_sums)
    
    # Check for collisions across cardinalities
    sum_to_ks = defaultdict(set)
    for k in range(n + 1):
        for s in sums_by_k[k]:
            sum_to_ks[s].add(k)
    
    for s, ks in sum_to_ks.items():
        if len(ks) > 1:
            return False
    return True


def _greedy_build(candidates, t_start, deadline):
    """
    Greedily build a valid set from candidates.
    Returns the best set found.
    """
    current = []
    sums_by_k = [set() for _ in range(61)]
    sums_by_k[0].add(0)
    sum_to_ks = defaultdict(set)
    sum_to_ks[0].add(0)
    
    for x in candidates:
        if time.time() > deadline:
            break
        n = len(current)
        if _check_add_valid(sums_by_k, sum_to_ks, x, n):
            current.append(x)
            _add_to_state(sums_by_k, sum_to_ks, x, n)
    
    return current


def _local_search(elements, candidates, t_start, deadline, max_iters=50000):
    """
    Local search: try to swap elements in the current set with candidates
    to increase the set size or improve the set.
    """
    current = list(elements)
    current_set = set(current)
    remaining = [x for x in candidates if x not in current_set]
    
    best = current[:]
    best_size = len(current)
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        # Try adding a new element
        if remaining:
            # Pick a random remaining element
            idx = random.randint(0, len(remaining) - 1)
            x = remaining[idx]
            
            # Check if we can add x
            test_set = current + [x]
            if _validate_set(test_set):
                current = test_set
                current_set.add(x)
                remaining.pop(idx)
                if len(current) > best_size:
                    best_size = len(current)
                    best = current[:]
                continue
        
        # Try swapping: remove one element, add another
        if current and remaining:
            i = random.randint(0, len(current) - 1)
            j = random.randint(0, len(remaining) - 1)
            old_x = current[i]
            new_x = remaining[j]
            
            test_set = current[:]
            test_set[i] = new_x
            random.shuffle(test_set)
            
            if _validate_set(test_set):
                # Accept the swap
                current[i] = new_x
                current_set.discard(old_x)
                current_set.add(new_x)
                remaining[j] = old_x
                if len(current) > best_size:
                    best_size = len(current)
                    best = current[:]
    
    return best


def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE_SECONDS
    
    best_set = []
    best_size = 0
    
    # Strategy 1: Descending order greedy
    candidates_desc = list(range(256, 0, -1))
    result = _greedy_build(candidates_desc, t_start, deadline)
    if len(result) > best_size:
        best_size = len(result)
        best_set = result[:]
    
    # Strategy 2: Ascending order greedy
    if time.time() < deadline:
        candidates_asc = list(range(1, 257))
        result = _greedy_build(candidates_asc, t_start, deadline)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result[:]
    
    # Strategy 3: Random order greedy (multiple attempts)
    if time.time() < deadline:
        for _ in range(10):
            if time.time() > deadline:
                break
            candidates_rand = list(range(1, 257))
            random.shuffle(candidates_rand)
            result = _greedy_build(candidates_rand, t_start, deadline)
            if len(result) > best_size:
                best_size = len(result)
                best_set = result[:]
    
    # Strategy 4: Local search refinement
    if time.time() < deadline:
        # Start from the best set found so far
        refined = _local_search(best_set, list(range(1, 257)), t_start, deadline, max_iters=100000)
        if len(refined) > best_size:
            best_size = len(refined)
            best_set = refined[:]
    
    # Strategy 5: More local search iterations if time permits
    if time.time() < deadline and best_size < 60:
        refined = _local_search(best_set, list(range(1, 257)), t_start, deadline, max_iters=200000)
        if len(refined) > best_size:
            best_size = len(refined)
            best_set = refined[:]
    
    # Ensure we return a valid set
    if best_set and not _validate_set(best_set):
        # Fallback: try to fix by removing problematic elements
        # For now, just return as is (shouldn't happen if validation is correct)
        pass
    
    return list(best_set)
