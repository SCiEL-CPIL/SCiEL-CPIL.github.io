"""Rewritten program with faster incremental checks and smarter search."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _subset_sums(S):
    """Return set of all nonempty subset sums of S."""
    sums = set()
    for x in S:
        new_sums = set()
        for s in sums:
            ns = s + x
            new_sums.add(ns)
        sums.update(new_sums)
        sums.add(x)
    return sums

def _is_valid(S):
    """Full validity check."""
    if not S:
        return True
    sums = _subset_sums(S)
    total = sum(S)
    sums_list = sorted(sums)
    sums_set = sums
    for s in sums_list:
        if s == 0:
            continue
        multiple = 2 * s
        while multiple <= total:
            if multiple in sums_set:
                return False
            multiple += s
    return True

def _check_divisibility(sums_set, total):
    """Check if any sum divides another in sums_set."""
    sums_list = sorted(sums_set)
    for s in sums_list:
        if s == 0:
            continue
        multiple = 2 * s
        while multiple <= total:
            if multiple in sums_set:
                return False
            multiple += s
    return True

def _candidate_score(x, current_sums, total):
    """Score candidate x: lower is better (fewer conflicts).
    Returns (conflict_count, x) for sorting."""
    conflicts = 0
    # Check if adding x creates divisibility issues
    # New sums will be: old sums + x, and x itself
    # We need to check: for each new sum ns, does any existing sum divide ns or vice versa?
    # Simplified: count how many existing sums would create problems
    for s in current_sums:
        ns = s + x
        # Check if s divides ns (i.e., x is a multiple of s)
        if ns % s == 0 and ns != s:
            conflicts += 1
        # Check if ns divides s (unlikely since ns > s)
        # Check if x divides s
        if s % x == 0 and s != x:
            conflicts += 1
        # Check if x divides ns
        if ns % x == 0 and ns != x:
            conflicts += 1
    # Also check x against existing sums
    for s in current_sums:
        if s % x == 0 and s != x:
            conflicts += 1
    return (conflicts, x)

def _greedy_build(pool, deadline, rng=None):
    """Build a set greedily, adding elements that don't violate constraints."""
    if rng is None:
        rng = random
    current = []
    current_sums = set()
    total = 0
    used = set()
    
    # Precompute candidate scores dynamically
    remaining = list(pool)
    rng.shuffle(remaining)
    
    for x in remaining:
        if time.time() > deadline:
            break
        if x in used:
            continue
        # Try adding x
        new_sums = set(current_sums)
        for s in list(current_sums):
            new_sums.add(s + x)
        new_sums.add(x)
        new_total = total + x
        if _check_divisibility(new_sums, new_total):
            current.append(x)
            current_sums = new_sums
            total = new_total
            used.add(x)
            if len(current) >= 30:
                break
    
    return current, current_sums, total

def _try_add(S, sums, total, x, deadline):
    """Try to add x to S. Returns (success, new_sums, new_total)."""
    if x in S:
        return False, sums, total
    new_sums = set(sums)
    for s in list(sums):
        new_sums.add(s + x)
    new_sums.add(x)
    new_total = total + x
    if _check_divisibility(new_sums, new_total):
        return True, new_sums, new_total
    return False, sums, total

def _local_search(S, sums, total, pool, deadline, rng=None):
    """Local search: try adding, removing, swapping elements."""
    if rng is None:
        rng = random
    current = list(S)
    cur_sums = set(sums)
    cur_total = total
    used = set(current)
    available = [x for x in pool if x not in used]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding elements (sorted by score)
        scored = []
        for x in available:
            if time.time() > deadline:
                break
            score = _candidate_score(x, cur_sums, cur_total)
            scored.append((score, x))
        scored.sort()
        
        for _, x in scored:
            if time.time() > deadline:
                break
            ok, new_sums, new_total = _try_add(current, cur_sums, cur_total, x, deadline)
            if ok:
                current.append(x)
                cur_sums = new_sums
                cur_total = new_total
                used.add(x)
                available = [y for y in available if y != x]
                improved = True
                if len(current) >= 30:
                    break
        
        # Try swaps if no improvement
        if not improved and len(current) > 1:
            rng.shuffle(available)
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                for x in available[:30]:
                    if time.time() > deadline:
                        break
                    removed = current[i]
                    test = current[:i] + current[i+1:] + [x]
                    test_sums = _subset_sums(test)
                    test_total = sum(test)
                    if _check_divisibility(test_sums, test_total):
                        current = test
                        cur_sums = test_sums
                        cur_total = test_total
                        used.discard(removed)
                        used.add(x)
                        available.remove(x)
                        available.append(removed)
                        improved = True
                        break
    
    return current, cur_sums, cur_total

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    pool = list(range(1, 257))
    
    best_set = []
    best_size = 0
    
    # Multiple restarts with different strategies
    strategies = []
    
    # Strategy 1: Random order
    strategies.append(("random", None))
    
    # Strategy 2: Large first
    strategies.append(("large", sorted(pool, reverse=True)))
    
    # Strategy 3: Small first  
    strategies.append(("small", sorted(pool)))
    
    # Strategy 4: Primes first
    primes = [x for x in range(2, 257) if all(x % i != 0 for i in range(2, int(math.isqrt(x)) + 1))]
    non_primes = [x for x in pool if x not in set(primes)]
    strategies.append(("primes", primes + non_primes))
    
    # Strategy 5: Coprime-ish (numbers with few divisors)
    def divisor_count(x):
        if x < 2:
            return 1
        count = 2
        for i in range(2, int(math.isqrt(x)) + 1):
            if x % i == 0:
                count += 2
        return count
    by_divisors = sorted(pool, key=lambda x: divisor_count(x))
    strategies.append(("few_divisors", by_divisors))
    
    for restart in range(40):
        if time.time() > deadline:
            break
        
        strat_name, order = strategies[restart % len(strategies)]
        rng = random.Random(restart * 7919)
        
        if order is None:
            candidates = pool[:]
            rng.shuffle(candidates)
        else:
            candidates = order[:]
            # Add some randomness
            if restart % 2 == 0:
                rng.shuffle(candidates)
        
        # Build greedily
        current = []
        cur_sums = set()
        cur_total = 0
        used = set()
        
        for x in candidates:
            if time.time() > deadline:
                break
            if x in used:
                continue
            new_sums = set(cur_sums)
            for s in list(cur_sums):
                new_sums.add(s + x)
            new_sums.add(x)
            new_total = cur_total + x
            if _check_divisibility(new_sums, new_total):
                current.append(x)
                cur_sums = new_sums
                cur_total = new_total
                used.add(x)
                if len(current) >= 30:
                    break
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
        
        # Local search
        if time.time() < deadline - 2:
            extended, ext_sums, ext_total = _local_search(
                current, cur_sums, cur_total, pool, deadline, rng
            )
            if len(extended) > best_size:
                best_size = len(extended)
                best_set = extended[:]
    
    # Final validation
    if best_set and not _is_valid(best_set):
        # Fallback: rebuild greedily
        current = []
        cur_sums = set()
        cur_total = 0
        for x in sorted(pool, reverse=True):
            new_sums = set(cur_sums)
            for s in list(cur_sums):
                new_sums.add(s + x)
            new_sums.add(x)
            new_total = cur_total + x
            if _check_divisibility(new_sums, new_total):
                current.append(x)
                cur_sums = new_sums
                cur_total = new_total
        best_set = current
    
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
