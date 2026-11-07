"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

# Precompute the conflict structure over the sum range [1, 7680]
# Max possible sum: 30 * 256 = 7680
MAX_SUM = 7680

# For each sum value s in [1, MAX_SUM], precompute a bitset of all t != s
# such that s divides t or t divides s (i.e., they are divisibility-related).
# We'll store this as a list of Python ints (bitsets).
# forbidden_mask[s] has bit t set if s and t are divisibility-incompatible.

def _build_conflict_masks():
    """Build forbidden-pair masks for all sums in [1, MAX_SUM]."""
    masks = [0] * (MAX_SUM + 1)
    for s in range(1, MAX_SUM + 1):
        mask = 0
        # All multiples of s (2s, 3s, ...)
        for t in range(2 * s, MAX_SUM + 1, s):
            mask |= (1 << t)
        # All divisors of s (proper divisors)
        # For each divisor d of s where d < s, d divides s
        # We can find them by checking all d from 1 to s-1, but that's slow.
        # Instead: for each d in 1..s//2, if s % d == 0, then d is a divisor.
        # But we need this for ALL s, so let's do it differently.
        # Actually, let's build it by: for each pair (a, b) where a < b and a|b,
        # set bit b in masks[a] and bit a in masks[b].
        # This is O(MAX_SUM * MAX_SUM / 2) which is too slow.
        # Better: for each a in 1..MAX_SUM, for each multiple b = 2a, 3a, ...,
        # set bit b in masks[a] and bit a in masks[b].
        # We'll do this in a separate pass.
    # Redo: build by iterating over pairs
    masks = [0] * (MAX_SUM + 1)
    for a in range(1, MAX_SUM + 1):
        for b in range(2 * a, MAX_SUM + 1, a):
            masks[a] |= (1 << b)
            masks[b] |= (1 << a)
    return masks

# Build the masks (this takes a few seconds)
_FORBIDDEN_MASKS = _build_conflict_masks()

def _subset_sums_bitset(elements):
    """Return a bitset (Python int) of all non-empty subset sums."""
    n = len(elements)
    sums = 0  # bitset
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += elements[i]
        sums |= (1 << s)
    return sums

def _is_valid_bitset(sums_bitset):
    """Check no two distinct sums in the bitset divide one another."""
    # For each set bit s, check if _FORBIDDEN_MASKS[s] & sums_bitset != 0
    # (excluding s itself, but _FORBIDDEN_MASKS[s] doesn't include bit s)
    temp = sums_bitset
    while temp:
        # Get lowest set bit
        lowbit = temp & (-temp)
        s = lowbit.bit_length() - 1
        if _FORBIDDEN_MASKS[s] & sums_bitset:
            return False
        temp ^= lowbit
    return True

def _is_valid(elements):
    """Full validation."""
    if not elements:
        return True
    if len(elements) != len(set(elements)):
        return False
    for x in elements:
        if not isinstance(x, int) or isinstance(x, bool):
            return False
        if not (1 <= x <= 256):
            return False
    sums = _subset_sums_bitset(elements)
    return _is_valid_bitset(sums)

def _check_add_bitset(elements, candidate, sums_bitset):
    """
    Check if adding candidate to elements keeps the set valid.
    Returns (new_sums_bitset, True) if valid, (None, False) otherwise.
    """
    if candidate in elements:
        return None, False
    # New sums = old sums | (old sums + candidate) | {candidate}
    shifted = sums_bitset << candidate
    new_sums = sums_bitset | shifted | (1 << candidate)
    # Check validity
    if not _is_valid_bitset(new_sums):
        return None, False
    return new_sums, True

def _greedy_grow(elements, candidates, sums_bitset, deadline):
    """Greedily add candidates in given order using bitset checks."""
    elements = list(elements)
    sums = sums_bitset
    for c in candidates:
        if time.time() > deadline:
            break
        if c in elements:
            continue
        new_sums, ok = _check_add_bitset(elements, c, sums)
        if ok:
            elements.append(c)
            sums = new_sums
    return elements, sums

def _backtrack(elements, sums_bitset, target_size, deadline, best_ref, candidate_pool, depth=0):
    """
    Backtracking search using bitset validity checks.
    best_ref is a list [best_elements] that gets updated.
    """
    if len(elements) > len(best_ref[0]):
        best_ref[0] = list(elements)
    if len(elements) >= target_size:
        return
    if time.time() > deadline:
        return
    
    for i, c in enumerate(candidate_pool):
        if time.time() > deadline:
            return
        if c in elements:
            continue
        new_sums, ok = _check_add_bitset(elements, c, sums_bitset)
        if ok:
            _backtrack(elements + [c], new_sums, target_size, deadline, best_ref, candidate_pool[i+1:], depth+1)
            if time.time() > deadline:
                return

def construct():
    start_time = time.time()
    deadline = start_time + 70  # Safety margin (90s limit, leave 20s buffer)
    
    random.seed(42)
    all_nums = list(range(1, 257))
    
    best_result = [31, 32, 34]
    
    # Strategy 1: Start with a few well-chosen numbers and greedily grow
    # Focus on larger numbers for denser sum clusters
    for base in range(200, 257):
        if time.time() > deadline:
            break
        start = [base]
        sums = _subset_sums_bitset(start)
        nearby = sorted(all_nums, key=lambda x: abs(x - base))
        nearby = [x for x in nearby if x != base]
        result, _ = _greedy_grow(start, nearby, sums, deadline)
        if len(result) > len(best_result):
            best_result = list(result)
    
    # Strategy 2: Start with pairs of nearby large numbers
    for a in range(180, 257, 3):
        if time.time() > deadline:
            break
        for b in range(a + 1, min(a + 30, 257)):
            if time.time() > deadline:
                break
            start = [a, b]
            sums = _subset_sums_bitset(start)
            nearby = sorted(all_nums, key=lambda x: min(abs(x - a), abs(x - b)))
            nearby = [x for x in nearby if x not in start]
            result, _ = _greedy_grow(start, nearby, sums, deadline)
            if len(result) > len(best_result):
                best_result = list(result)
    
    # Strategy 3: Randomized restarts with different orderings
    for trial in range(500):
        if time.time() > deadline:
            break
        
        k = random.randint(1, 4)
        start = random.sample(all_nums, k)
        sums = _subset_sums_bitset(start)
        
        remaining = [x for x in all_nums if x not in start]
        random.shuffle(remaining)
        
        result, _ = _greedy_grow(start, remaining, sums, deadline)
        if len(result) > len(best_result):
            best_result = list(result)
    
    # Strategy 4: Local search / hill climbing on the best found
    if len(best_result) >= 3 and time.time() < deadline - 10:
        current = list(best_result)
        sums = _subset_sums_bitset(current)
        
        for iteration in range(1000):
            if time.time() > deadline:
                break
            
            improved = False
            used = set(current)
            candidates = [x for x in all_nums if x not in used]
            random.shuffle(candidates)
            
            for c in candidates:
                new_sums, ok = _check_add_bitset(current, c, sums)
                if ok:
                    current.append(c)
                    sums = new_sums
                    improved = True
                    break
            
            if not improved:
                if len(current) >= 2:
                    remove_idx = random.randint(0, len(current) - 1)
                    removed = current.pop(remove_idx)
                    new_sums = _subset_sums_bitset(current)
                    
                    used = set(current)
                    candidates = [x for x in all_nums if x not in used]
                    random.shuffle(candidates)
                    
                    added_any = False
                    for c in candidates:
                        ns, ok = _check_add_bitset(current, c, new_sums)
                        if ok:
                            current.append(c)
                            new_sums = ns
                            added_any = True
                            break
                    
                    if not added_any:
                        current.append(removed)
                        new_sums = _subset_sums_bitset(current)
                    else:
                        sums = new_sums
                        if len(current) > len(best_result):
                            best_result = list(current)
    
    # Strategy 5: Targeted backtracking from promising starts
    if time.time() < deadline - 5:
        if len(best_result) < 15:
            target = len(best_result) + 1
            best_ref = [list(best_result)]
            used = set(best_result)
            pool = [x for x in all_nums if x not in used]
            pool.sort(key=lambda x: min(abs(x - e) for e in best_result))
            sums = _subset_sums_bitset(best_result)
            _backtrack(list(best_result), sums, target, deadline, best_ref, pool)
            if len(best_ref[0]) > len(best_result):
                best_result = list(best_ref[0])
    
    # Final validation
    if _is_valid(best_result):
        return list(best_result)
    else:
        # Fallback to known valid set
        return [31, 32, 34]
# EVOLVE-BLOCK-END
