"""Program to construct a large set with no subset-average property."""

import time
import random
import itertools

_DEADLINE = 85.0  # safety margin before 90s kill


def _check_violation(C, idx):
    """Check if C[idx] is the mean of any subset of >=2 OTHER elements.
    Uses bitset DP for subset sums of (y - C[idx])."""
    a = C[idx]
    n = len(C)
    if n < 3:
        return False
    
    # Collect differences
    diffs = []
    for j in range(n):
        if j != idx:
            diffs.append(C[j] - a)
    
    m = len(diffs)
    if m < 2:
        return False
    
    # Compute range
    max_pos = sum(d for d in diffs if d > 0)
    max_neg = -sum(d for d in diffs if d < 0)
    offset = max_neg
    total_span = max_pos + max_neg + 1
    
    if total_span > 100000:  # Too large for bitset
        # Fall back to checking small subset sizes only
        # For large sets, check subsets of size 2 and 3
        for i in range(m):
            for j in range(i+1, m):
                if diffs[i] + diffs[j] == 0:
                    return True
        for i in range(m):
            for j in range(i+1, m):
                for k in range(j+1, m):
                    if diffs[i] + diffs[j] + diffs[k] == 0:
                        return True
        return False
    
    # Bitset DP: dp[k] is a bitset, bit p set means sum = p - offset achievable with exactly k elements
    # Sum 0 corresponds to bit position = offset
    target_bit = 1 << offset
    mask = (1 << total_span) - 1
    
    dp = [0] * (m + 1)
    dp[0] = target_bit
    
    for d in diffs:
        # Update dp backwards to avoid reusing element
        for k in range(m, 0, -1):
            if dp[k-1] == 0:
                continue
            if d >= 0:
                dp[k] |= (dp[k-1] << d) & mask
            else:
                dp[k] |= (dp[k-1] >> (-d)) & mask
        
        # Early exit
        if dp[2] & target_bit:
            return True
        if m >= 3 and dp[3] & target_bit:
            return True
    
    # Check all subset sizes >= 2
    for k in range(2, m + 1):
        if dp[k] & target_bit:
            return True
    
    return False


def _is_valid(C):
    """Check if the entire set C is valid (no element is mean of subset of others)."""
    for i in range(len(C)):
        if _check_violation(C, i):
            return False
    return True


def _fast_incremental_check(C, new_elem):
    """Quick check when adding new_elem to set C.
    Check 1: is new_elem mean of any subset of C?
    Check 2: is any element in C mean of a subset involving new_elem?
    """
    a = new_elem
    n = len(C)
    
    # Check 1: is new_elem the mean of a subset S of C with |S| >= 2?
    # sum(S) = |S| * a
    # For |S|=2: sum of 2 elements = 2a
    for i in range(n):
        for j in range(i+1, n):
            if C[i] + C[j] == 2 * a:
                return False
    
    # For |S|=3: sum of 3 elements = 3a
    for i in range(n):
        for j in range(i+1, n):
            rem = 3 * a - C[i] - C[j]
            for k in range(j+1, n):
                if C[k] == rem:
                    return False
    
    # Check 2: is any C[i] the mean of a subset involving new_elem?
    # Subset = {new_elem} ∪ T where T ⊆ C, T doesn't contain C[i], |T| >= 1
    # (new_elem + sum(T)) / (1+|T|) = C[i]
    # new_elem + sum(T) = C[i] * (1+|T|)
    # sum(T) = C[i]*(1+|T|) - new_elem
    # For |T|=1: sum(T) = 2*C[i] - new_elem, T = {y} where y in C, y != C[i]
    C_set = set(C)
    for i in range(n):
        y = 2 * C[i] - a
        if y in C_set and y != C[i]:
            return False
    
    # For |T|=2: sum(T) = 3*C[i] - a, T = {y1, y2} in C, y1,y2 != C[i], y1 != y2
    for i in range(n):
        target_sum = 3 * C[i] - a
        for j in range(n):
            if C[j] == C[i]:
                continue
            for k in range(j+1, n):
                if C[k] == C[i]:
                    continue
                if C[j] + C[k] == target_sum:
                    return False
    
    return True


def _greedy_construct(order, time_limit):
    """Greedily build a valid set following the given order."""
    selected = []
    start = time.time()
    
    for x in order:
        if time.time() - start > time_limit:
            break
        if len(selected) >= 48:
            break
        
        # Fast incremental check
        if _fast_incremental_check(selected, x):
            # Full verification
            C = selected + [x]
            if _is_valid(C):
                selected.append(x)
    
    return selected


def _random_order(seed, n=512):
    """Generate a random ordering of [1..n]."""
    rng = random.Random(seed)
    order = list(range(1, n+1))
    rng.shuffle(order)
    return order


def _sidon_order():
    """Generate a Sidon-like order: greedy construction of B2 sequence."""
    S = []
    S_set = set()
    pair_sums = set()
    for x in range(1, 513):
        valid = True
        for y in S:
            if x + y in pair_sums:
                valid = False
                break
        if valid:
            S.append(x)
            S_set.add(x)
            for y in S:
                pair_sums.add(x + y)
    return S

def _structured_order_lo_hi():
    """Low-high alternating order."""
    order = []
    lo, hi = 1, 512
    while lo <= hi:
        order.append(lo)
        if lo != hi:
            order.append(hi)
        lo += 1
        hi -= 1
    return order


def _structured_order_hi_lo():
    """High-low alternating order."""
    order = []
    lo, hi = 1, 512
    while lo <= hi:
        order.append(hi)
        if lo != hi:
            order.append(lo)
        lo += 1
        hi -= 1
    return order


def _spread_order(step=10):
    """Pick elements spread across the range."""
    # Start with spread points
    base = list(range(1, 513, step))
    random.Random(123).shuffle(base)
    # Fill in gaps
    remaining = [x for x in range(1, 513) if x not in set(base)]
    random.Random(456).shuffle(remaining)
    return base + remaining


def _cluster_order():
    """Pick from clusters spread across range."""
    order = []
    rng = random.Random(789)
    # Create clusters
    cluster_centers = list(range(10, 520, 30))
    rng.shuffle(cluster_centers)
    for c in cluster_centers:
        # Add elements near center
        spread = rng.randint(5, 15)
        for d in range(-spread, spread+1):
            x = c + d
            if 1 <= x <= 512:
                order.append(x)
    # Remove duplicates while preserving order
    seen = set()
    unique_order = []
    for x in order:
        if x not in seen:
            seen.add(x)
            unique_order.append(x)
    # Add remaining
    remaining = [x for x in range(1, 513) if x not in seen]
    rng.shuffle(remaining)
    return unique_order + remaining


def construct():
    start_time = time.time()
    best_set = []
    best_size = 0
    
    # Generate diverse candidate orders
    orders = []
    
    # Sidon order is a strong start
    orders.append(_sidon_order())
    
    # Structured orders
    orders.append(_structured_order_lo_hi())
    orders.append(_structured_order_hi_lo())
    orders.append(list(range(1, 513)))       # Increasing
    orders.append(list(range(512, 0, -1)))   # Decreasing
    orders.append(_spread_order(10))
    orders.append(_spread_order(20))
    orders.append(_cluster_order())
    
    # Random orders with different seeds
    for seed in range(30):
        orders.append(_random_order(seed))
    
    # Try each order with greedy construction
    time_per_order = 2.0  # seconds per order attempt
    for i, order in enumerate(orders):
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE - 2:
            break
        
        remaining_time = _DEADLINE - elapsed - 1
        time_limit = min(time_per_order, remaining_time)
        
        result = _greedy_construct(order, time_limit)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result[:]
    
    # If we have time, try local improvement on best set
    elapsed = time.time() - start_time
    if elapsed < _DEADLINE - 3 and best_size < 48:
        # Try to add more elements
        used = set(best_set)
        rng = random.Random(999)
        candidates = [x for x in range(1, 513) if x not in used]
        rng.shuffle(candidates)
        
        for x in candidates:
            if time.time() - start_time > _DEADLINE - 1:
                break
            if len(best_set) >= 48:
                break
            
            if _fast_incremental_check(best_set, x):
                C = best_set + [x]
                if _is_valid(C):
                    best_set.append(x)
                    best_size = len(best_set)
    
    # Final validation
    if best_set and not _is_valid(best_set):
        # Fallback to powers of 2
        return [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    if not best_set:
        return [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    return best_set
