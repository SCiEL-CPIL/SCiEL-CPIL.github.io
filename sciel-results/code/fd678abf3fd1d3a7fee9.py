"""Rewritten program: bitset-based backtracking for divisibility-free subset sums."""
import time
import random

MAX_VAL = 256
MAX_ELEMENTS = 30
MAX_SUM = MAX_VAL * MAX_ELEMENTS  # 7680

def _build_forbidden_mask():
    """
    For each sum s (1..MAX_SUM), build a bitset (as a Python int) where bit j is set
    iff s divides (j+1) and j+1 != s, i.e., j+1 is a multiple of s greater than s.
    We index sums by (sum - 1) for 0-based bit positions.
    """
    # forbidden[s] = bitmask of sums t > s such that s | t
    forbidden = [0] * (MAX_SUM + 1)
    for s in range(1, MAX_SUM + 1):
        mask = 0
        for t in range(2 * s, MAX_SUM + 1, s):
            mask |= (1 << (t - 1))
        forbidden[s] = mask
    return forbidden

def _subset_sums_bitset(elements):
    """
    Compute the set of all non-empty subset sums as a bitset (Python int).
    Bit i is set iff (i+1) is a subset sum.
    Returns the bitset.
    """
    sums = 0  # bitset, bit (s-1) set iff s is achievable
    for e in elements:
        # New sums = old sums shifted by e, plus e itself
        shifted = sums << e
        sums = sums | shifted | (1 << (e - 1))
    return sums

def _is_valid_bitset(sums_bitset, forbidden):
    """
    Check that no two distinct subset sums in the bitset divide each other.
    For each set bit s, check if any bit t (t > s, s|t) is also set.
    """
    # Iterate over set bits
    s = 1
    while s <= MAX_SUM:
        if sums_bitset & (1 << (s - 1)):
            # Check if any multiple of s (greater than s) is also in the set
            if sums_bitset & forbidden[s]:
                return False
        # Move to next set bit
        s = (sums_bitset & (sums_bitset - 1)).bit_length() + 1
        if s == 1:
            break
    return True

def _check_addition(sums_bitset, elem, forbidden):
    """
    Check if adding elem to the current set keeps validity.
    Returns new bitset if valid, None otherwise.
    """
    new_sums = sums_bitset | (sums_bitset << elem) | (1 << (elem - 1))
    # Quick check: does new_sums have any forbidden pair?
    if not _is_valid_bitset(new_sums, forbidden):
        return None
    return new_sums

def _backtrack_search(forbidden, deadline, target_size, max_elements=30):
    """
    Backtracking search to find a valid set of size target_size.
    Uses bitset representation for fast validity checks.
    """
    best = []
    
    def backtrack(current_elems, current_sums, depth):
        nonlocal best
        if time.time() > deadline:
            return False
        if len(current_elems) >= target_size:
            best = current_elems[:]
            return True
        if depth >= max_elements:
            return False
        
        # Try adding each possible element
        for val in range(1, MAX_VAL + 1):
            if val in current_elems:
                continue
            new_sums = _check_addition(current_sums, val, forbidden)
            if new_sums is not None:
                current_elems.append(val)
                if backtrack(current_elems, new_sums, depth + 1):
                    return True
                current_elems.pop()
            if time.time() > deadline:
                return False
        return False
    
    backtrack([], 0, 0)
    return best

def _greedy_with_bitset(forbidden, deadline, strategy='smallest'):
    """
    Greedy construction using bitset validity checks.
    """
    current = []
    current_sums = 0
    
    for _ in range(MAX_ELEMENTS):
        if time.time() > deadline:
            break
        
        valid_candidates = []
        for val in range(1, MAX_VAL + 1):
            if val in current:
                continue
            new_sums = _check_addition(current_sums, val, forbidden)
            if new_sums is not None:
                valid_candidates.append(val)
        
        if not valid_candidates:
            break
        
        if strategy == 'smallest':
            choice = min(valid_candidates)
        elif strategy == 'largest':
            choice = max(valid_candidates)
        elif strategy == 'random':
            choice = random.choice(valid_candidates)
        else:
            choice = random.choice(valid_candidates)
        
        current.append(choice)
        current_sums = _check_addition(current_sums, choice, forbidden)
    
    return current

def _local_search_bitset(forbidden, deadline, initial, max_iters=200):
    """
    Local search: try to grow the set by adding elements, or swap to enable growth.
    """
    current = list(initial)
    current_sums = _subset_sums_bitset(current)
    best = current[:]
    
    if not _is_valid_bitset(current_sums, forbidden):
        return best
    
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try to add a new element
        for val in range(1, MAX_VAL + 1):
            if val in current:
                continue
            new_sums = _check_addition(current_sums, val, forbidden)
            if new_sums is not None:
                current.append(val)
                current_sums = new_sums
                best = current[:]
                improved = True
                break
        
        if improved:
            continue
        
        # Try swaps: remove one, add another
        for i, elem in enumerate(current):
            if time.time() > deadline:
                break
            reduced = current[:i] + current[i+1:]
            reduced_sums = _subset_sums_bitset(reduced)
            
            for new_val in range(1, MAX_VAL + 1):
                if new_val in reduced:
                    continue
                trial = reduced + [new_val]
                trial_sums = _subset_sums_bitset(trial)
                if _is_valid_bitset(trial_sums, forbidden):
                    # Try to grow from this state
                    grown = _greedy_with_bitset(forbidden, deadline, strategy='random')
                    # Merge: use trial as start, then grow
                    trial_grown = list(trial)
                    trial_sums2 = trial_sums
                    for _ in range(MAX_ELEMENTS - len(trial_grown)):
                        if time.time() > deadline:
                            break
                        valid_c = []
                        for v in range(1, MAX_VAL + 1):
                            if v in trial_grown:
                                continue
                            ns = _check_addition(trial_sums2, v, forbidden)
                            if ns is not None:
                                valid_c.append(v)
                        if not valid_c:
                            break
                        choice = random.choice(valid_c)
                        trial_grown.append(choice)
                        trial_sums2 = _check_addition(trial_sums2, choice, forbidden)
                    
                    if len(trial_grown) > len(best):
                        best = trial_grown
                        current = trial_grown
                        current_sums = trial_sums2
                        improved = True
                        break
            
            if improved:
                break
        
        if not improved:
            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    # Build forbidden mask
    forbidden = _build_forbidden_mask()
    
    best = []
    
    # Phase 1: Greedy with different strategies
    strategies = ['smallest', 'largest', 'random']
    for strat in strategies:
        if time.time() > deadline:
            break
        result = _greedy_with_bitset(forbidden, deadline, strategy=strat)
        if len(result) > len(best):
            best = result
    
    # Phase 2: Local search from best
    if len(best) > 0 and time.time() < deadline - 10:
        result = _local_search_bitset(forbidden, deadline - 5, best, max_iters=100)
        if len(result) > len(best):
            best = result
    
    # Phase 3: Multiple random restarts with greedy + local search
    for attempt in range(100):
        if time.time() > deadline - 5:
            break
        
        random.seed(attempt * 104729 + 7)
        
        # Random start
        start_size = random.randint(1, 3)
        start_elems = random.sample(range(1, MAX_VAL + 1), start_size)
        start_elems.sort()
        
        start_sums = _subset_sums_bitset(start_elems)
        if not _is_valid_bitset(start_sums, forbidden):
            continue
        
        # Greedy grow
        result = _greedy_with_bitset(forbidden, deadline, strategy='random')
        # Use start as seed
        current = list(start_elems)
        current_sums = start_sums
        for _ in range(MAX_ELEMENTS - len(current)):
            if time.time() > deadline:
                break
            valid_c = []
            for v in range(1, MAX_VAL + 1):
                if v in current:
                    continue
                ns = _check_addition(current_sums, v, forbidden)
                if ns is not None:
                    valid_c.append(v)
            if not valid_c:
                break
            choice = random.choice(valid_c)
            current.append(choice)
            current_sums = _check_addition(current_sums, choice, forbidden)
        
        if len(current) > len(best):
            best = current
        
        # Quick local search
        if time.time() < deadline - 3 and len(current) >= len(best) - 1:
            result = _local_search_bitset(forbidden, deadline - 2, current, max_iters=20)
            if len(result) > len(best):
                best = result
    
    # Final polish
    if len(best) > 0 and time.time() < deadline - 2:
        result = _local_search_bitset(forbidden, deadline - 1, best, max_iters=30)
        if len(result) > len(best):
            best = result
    
    return [int(x) for x in best]
