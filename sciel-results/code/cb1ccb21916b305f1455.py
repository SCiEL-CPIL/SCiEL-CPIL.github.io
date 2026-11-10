"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 256, 'maximum_size': 30}

def _get_sums(S):
    """Compute all nonempty subset sums of S. Returns set or None if collision."""
    sums = set()
    n = len(S)
    for r in range(1, n + 1):
        for comb in combinations(S, r):
            s = sum(comb)
            if s in sums:
                return None
            sums.add(s)
    return sums

def _check_divisibility(sums_set):
    """Check if no two distinct values in sums_set divide one another."""
    sums_list = sorted(sums_set)
    sum_set = set(sums_list)
    max_sum = sums_list[-1] if sums_list else 0
    for s in sums_list:
        multiple = s * 2
        while multiple <= max_sum:
            if multiple in sum_set:
                return False
            multiple += s
    return True

def _is_valid_set(S):
    """Full validation of set S."""
    if len(set(S)) != len(S):
        return False
    if any(x < 1 or x > 256 for x in S):
        return False
    sums = _get_sums(S)
    if sums is None:
        return False
    return _check_divisibility(sums)

def _try_add(S, x, current_sums):
    """Try to add x to S. Returns (new_sums, success) where new_sums is the updated sum set or None."""
    if x in S:
        return None, False
    
    new_sums = set(current_sums)
    valid_distinct = True
    for s in current_sums:
        ns = s + x
        if ns in new_sums:
            valid_distinct = False
            break
        new_sums.add(ns)
    
    if not valid_distinct:
        return None, False
    
    if not _check_divisibility(new_sums):
        return None, False
    
    return new_sums, True

def _greedy_extend(S, max_val=256, time_limit=None):
    """Greedily extend set S by adding valid elements. Returns the extended set."""
    if time_limit is None:
        time_limit = 0.5
    
    current = list(S)
    current_sums = _get_sums(current)
    if current_sums is None:
        return current
    
    while time.time() < time_limit:
        extended = False
        # Try adding each candidate from smallest to largest
        for x in range(1, max_val + 1):
            if x in current:
                continue
            new_sums, success = _try_add(current, x, current_sums)
            if success:
                current.append(x)
                current_sums = new_sums
                extended = True
                break
        if not extended:
            break
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    best_set = [31, 32, 34]
    
    # Strategy 1: Greedy extend from known valid seed
    extended = _greedy_extend(best_set, time_limit=5)
    if len(extended) > len(best_set) and _is_valid_set(extended):
        best_set = extended
    
    # Strategy 2: Random restarts with greedy extension
    # Try many random starting sets and extend greedily
    rng = random.Random(42)
    
    def random_restart(start_size=2, max_start_val=100):
        """Create a random starting set and extend greedily."""
        # Pick a random starting set of size 2-3
        candidates = list(range(2, max_start_val + 1))
        rng.shuffle(candidates)
        
        # Try to build a valid starting set
        start_set = []
        for x in candidates:
            if len(start_set) >= start_size:
                break
            test = start_set + [x]
            if _is_valid_set(test):
                start_set.append(x)
        
        if len(start_set) < 2:
            return None
        
        # Extend greedily
        extended = _greedy_extend(start_set, time_limit=2.0)
        return extended
    
    # Multiple random restarts
    restart_count = 0
    while time.time() < deadline - 10:
        restart_count += 1
        result = random_restart()
        if result is not None and len(result) > len(best_set):
            if _is_valid_set(result):
                best_set = result
                # Try to extend the new best further
                further = _greedy_extend(best_set, time_limit=3.0)
                if len(further) > len(best_set) and _is_valid_set(further):
                    best_set = further
    
    # Strategy 3: Try superincreasing-like constructions
    # If a_i > sum of previous, subset sums are distinct
    # Try various bases
    for base in [3, 4, 5]:
        if time.time() > deadline - 15:
            break
        # Generate superincreasing sequence
        seq = []
        current_sum = 0
        for i in range(10):
            # Pick a value greater than current_sum
            min_val = current_sum + 1
            if min_val > 256:
                break
            # Try values from min_val to 256
            found = False
            for val in range(min_val, 257):
                test = seq + [val]
                if _is_valid_set(test):
                    seq.append(val)
                    current_sum += val
                    found = True
                    break
            if not found:
                break
        if len(seq) > len(best_set) and _is_valid_set(seq):
            best_set = seq
    
    # Final validation and return
    if not _is_valid_set(best_set):
        best_set = [31, 32, 34]
    
    return list(best_set)

# EVOLVE-BLOCK-END
