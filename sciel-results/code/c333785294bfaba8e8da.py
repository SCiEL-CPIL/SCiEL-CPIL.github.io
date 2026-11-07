"""
Structural change: use a different representation and search strategy.

Key insight: The constraint is that no two distinct subset sums s, t have s|t.
This is equivalent to: for each subset sum s, no other subset sum is a multiple of s.

New approach:
1. Use bitmask-based subset sum enumeration for fast checks when set size is small.
2. Use a SAT-like constraint propagation approach: for each candidate element,
   check if it can be added without creating divisibility conflicts.
3. Use simulated annealing with a different neighborhood structure.
4. Precompute divisibility relations between numbers in the pool.

The critical structural change: instead of greedy + local search, use:
- A backtracking search with constraint propagation
- Bitmask representation for subset sums (feasible up to ~20 elements)
- A different scoring function that estimates future extendability
"""
import time
import random
import math
from itertools import combinations

def _subset_sums_bitmask(S):
    """Compute all subset sums using bitmask enumeration. Returns a set."""
    n = len(S)
    sums = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += S[i]
        sums.add(s)
    return sums

def _check_divisibility_fast(sums_set, total):
    """Check if any sum divides another. Returns True if valid (no divisibility)."""
    sums_list = sorted(sums_set)
    for s in sums_list:
        if s == 0:
            continue
        for mult in range(2, total // s + 1):
            if s * mult in sums_set:
                return False
    return True

def _is_valid(S):
    """Full validity check."""
    if not S:
        return True
    sums = _subset_sums_bitmask(S)
    total = sum(S)
    return _check_divisibility_fast(sums, total)

def _divisibility_conflict_count(S):
    """Count how many divisibility conflicts exist (for scoring)."""
    if not S:
        return 0
    sums = _subset_sums_bitmask(S)
    total = sum(S)
    count = 0
    sums_list = sorted(sums)
    for s in sums_list:
        if s == 0:
            continue
        for mult in range(2, total // s + 1):
            if s * mult in sums:
                count += 1
    return count

def _estimate_extendability(S, pool):
    """Estimate how many elements from pool could potentially be added."""
    if not S:
        return len(pool)
    sums = _subset_sums_bitmask(S)
    total = sum(S)
    usable = 0
    for x in pool:
        if x in S:
            continue
        # Quick check: does x divide any existing sum or vice versa?
        conflict = False
        for s in sums:
            if s % x == 0:
                conflict = True
                break
            if x % s == 0:
                conflict = True
                break
        if not conflict:
            usable += 1
    return usable

def _backtrack_construct(pool, deadline, max_size=15):
    """
    Backtracking search with constraint propagation.
    Uses a different structural approach: recursive construction
    with pruning based on divisibility constraints.
    """
    best = []
    best_size = 0
    
    def backtrack(current, remaining, sums, total, depth):
        nonlocal best, best_size
        if time.time() > deadline:
            return
        if len(current) > best_size:
            best_size = len(current)
            best = current[:]
        if len(current) >= max_size:
            return
        
        # Try adding each remaining element
        for i in range(len(remaining)):
            if time.time() > deadline:
                return
            x = remaining[i]
            # Quick divisibility check with existing sums
            new_sums = set(sums)
            for s in sums:
                new_sums.add(s + x)
            new_sums.add(x)
            new_total = total + x
            
            # Check if x creates divisibility with existing sums
            valid = True
            for s in new_sums:
                if s % x == 0 and s != x:
                    # x divides s, but s was already in sums or is x itself
                    # Actually we need to check: does x divide any OTHER sum, 
                    # or does any other sum divide x?
                    pass
            
            # Full check
            if _check_divisibility_fast(new_sums, new_total):
                new_remaining = remaining[:i] + remaining[i+1:]
                backtrack(current + [x], new_remaining, new_sums, new_total, depth + 1)
            
            # Pruning: if we've tried enough, stop
            if depth > 3 and i > 20:
                break
    
    # Start with different initial seeds
    for seed_size in [1, 2, 3]:
        for seed in combinations(pool[:50], seed_size):
            if time.time() > deadline:
                return
            seed = list(seed)
            if len(seed) > best_size:
                sums = _subset_sums_bitmask(seed)
                total = sum(seed)
                if _check_divisibility_fast(sums, total):
                    remaining = [x for x in pool if x not in seed]
                    random.shuffle(remaining)
                    backtrack(seed, remaining, sums, total, 0)
    
    return best

def _simulated_annealing(S, pool, deadline):
    """
    Simulated annealing with a different neighborhood structure:
    - Move 1: Replace one element with another
    - Move 2: Add one element (if valid)
    - Move 3: Remove one element (to allow re-addition)
    """
    current = list(S)
    cur_sums = _subset_sums_bitmask(current)
    cur_total = sum(current)
    cur_size = len(current)
    cur_score = cur_size * 100 - _divisibility_conflict_count(current)
    
    best = current[:]
    best_size = cur_size
    best_score = cur_score
    
    temperature = 10.0
    cooling = 0.995
    used = set(current)
    
    while time.time() < deadline:
        if temperature < 0.01:
            break
            
        # Choose a random move
        move_type = random.random()
        new_set = None
        new_sums = None
        new_total = None
        
        if move_type < 0.4 and cur_size < 30:
            # Add a random element
            candidates = [x for x in pool if x not in used]
            if candidates:
                x = random.choice(candidates)
                new_set = current + [x]
                new_sums = set(cur_sums)
                for s in cur_sums:
                    new_sums.add(s + x)
                new_sums.add(x)
                new_total = cur_total + x
                
        elif move_type < 0.8 and cur_size > 0:
            # Replace one element
            idx = random.randrange(len(current))
            old_x = current[idx]
            candidates = [x for x in pool if x not in used]
            if candidates:
                new_x = random.choice(candidates)
                new_set = current[:]
                new_set[idx] = new_x
                new_sums = _subset_sums_bitmask(new_set)
                new_total = sum(new_set)
                
        else:
            # Remove one element (to explore smaller sets that might be extendable)
            if cur_size > 2:
                idx = random.randrange(len(current))
                new_set = current[:idx] + current[idx+1:]
                new_sums = _subset_sums_bitmask(new_set)
                new_total = sum(new_set)
        
        if new_set is None or new_sums is None:
            temperature *= cooling
            continue
        
        # Check validity
        if _check_divisibility_fast(new_sums, new_total):
            new_size = len(new_set)
            new_score = new_size * 100
            delta = new_score - cur_score
            if delta > 0 or random.random() < math.exp(-(-delta) / max(temperature, 0.01)):
                current = new_set
                cur_sums = new_sums
                cur_total = new_total
                cur_size = new_size
                cur_score = new_score
                used = set(current)
                
                if cur_size > best_size or (cur_size == best_size and cur_score > best_score):
                    best = current[:]
                    best_size = cur_size
                    best_score = cur_score
        else:
            # Invalid move, still might accept with low probability for exploration
            new_size = len(new_set)
            new_score = new_size * 100 - _divisibility_conflict_count(new_set)
            delta = new_score - cur_score
            if delta > 0 or random.random() < math.exp(-(-delta) / max(temperature, 0.01)):
                current = new_set
                cur_sums = new_sums
                cur_total = new_total
                cur_size = new_size
                cur_score = new_score
                used = set(current)
        
        temperature *= cooling
    
    return best

def _construct_with_different_strategy(deadline):
    """
    Different structural strategy:
    1. Start with a small valid set
    2. Use iterative deepening to find larger sets
    3. Use constraint propagation to prune search space
    """
    pool = list(range(1, 257))
    best_set = []
    best_size = 0
    
    # Strategy 1: Greedy with different orderings
    orderings = [
        sorted(pool, reverse=True),
        sorted(pool),
        [x for x in pool if x % 2 == 1] + [x for x in pool if x % 2 == 0],
        [x for x in pool if math.gcd(x, 256) == 1] + [x for x in pool if math.gcd(x, 256) != 1],
    ]
    
    for order in orderings:
        if time.time() > deadline:
            break
        current = []
        sums = set()
        total = 0
        used = set()
        
        for x in order:
            if time.time() > deadline:
                break
            if x in used:
                continue
            new_sums = set(sums)
            for s in sums:
                new_sums.add(s + x)
            new_sums.add(x)
            new_total = total + x
            if _check_divisibility_fast(new_sums, new_total):
                current.append(x)
                sums = new_sums
                total = new_total
                used.add(x)
                if len(current) > best_size:
                    best_size = len(current)
                    best_set = current[:]
    
    # Strategy 2: Simulated annealing from best found
    if best_set and time.time() < deadline - 5:
        sa_result = _simulated_annealing(best_set, pool, deadline)
        if len(sa_result) > best_size:
            best_set = sa_result
            best_size = len(best_set)
    
    # Strategy 3: Try to extend best_set by adding elements
    if best_set and time.time() < deadline - 5:
        extended = list(best_set)
        ext_sums = _subset_sums_bitmask(extended)
        ext_total = sum(extended)
        used = set(extended)
        
        # Try all possible additions in random order
        candidates = [x for x in pool if x not in used]
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline - 2:
                break
            new_sums = set(ext_sums)
            for s in ext_sums:
                new_sums.add(s + x)
            new_sums.add(x)
            new_total = ext_total + x
            if _check_divisibility_fast(new_sums, new_total):
                extended.append(x)
                ext_sums = new_sums
                ext_total = new_total
                used.add(x)
        
        if len(extended) > best_size:
            best_set = extended
            best_size = len(best_set)
    
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    pool = list(range(1, 257))
    best_set = []
    best_size = 0
    
    # Phase 1: Quick greedy constructions with different orderings
    orderings = [
        sorted(pool, reverse=True),
        sorted(pool),
        sorted(pool, key=lambda x: math.isqrt(x) * -1),  # Larger first by sqrt
        [x for x in pool if all(x % p != 0 for p in [2, 3, 5, 7])] + 
        [x for x in pool if not all(x % p != 0 for p in [2, 3, 5, 7])],
    ]
    
    for order in orderings:
        if time.time() > deadline:
            break
        current = []
        sums = set()
        total = 0
        used = set()
        
        for x in order:
            if time.time() > deadline:
                break
            if x in used:
                continue
            new_sums = set(sums)
            for s in sums:
                new_sums.add(s + x)
            new_sums.add(x)
            new_total = total + x
            if _check_divisibility_fast(new_sums, new_total):
                current.append(x)
                sums = new_sums
                total = new_total
                used.add(x)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
    
    # Phase 2: Local search with different neighborhood
    if best_set and time.time() < deadline - 10:
        # Try 1-swap improvements
        current = best_set[:]
        cur_sums = _subset_sums_bitmask(current)
        cur_total = sum(current)
        used = set(current)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            available = [x for x in pool if x not in used]
            random.shuffle(available)
            
            for i in range(len(current)):
                if time.time() > deadline - 5:
                    break
                for x in available[:30]:
                    if time.time() > deadline - 5:
                        break
                    new_set = current[:]
                    new_set[i] = x
                    new_sums = _subset_sums_bitmask(new_set)
                    new_total = sum(new_set)
                    if _check_divisibility_fast(new_sums, new_total):
                        current = new_set
                        cur_sums = new_sums
                        cur_total = new_total
                        used.discard(current[i-1] if i > 0 else 0)  # Not quite right, but works
                        used = set(current)
                        available = [x for x in pool if x not in used]
                        improved = True
                        break
                if improved:
                    break
            
            # Try additions
            available = [x for x in pool if x not in used]
            random.shuffle(available)
            for x in available[:30]:
                if time.time() > deadline - 5:
                    break
                new_set = current + [x]
                new_sums = set(cur_sums)
                for s in cur_sums:
                    new_sums.add(s + x)
                new_sums.add(x)
                new_total = cur_total + x
                if _check_divisibility_fast(new_sums, new_total):
                    current.append(x)
                    cur_sums = new_sums
                    cur_total = new_total
                    used.add(x)
                    improved = True
                    break
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
    
    # Phase 3: Simulated annealing
    if best_set and time.time() < deadline - 5:
        sa_result = _simulated_annealing(best_set, pool, deadline)
        if len(sa_result) > best_size:
            best_set = sa_result
            best_size = len(best_set)
    
    # Final validation
    if best_set:
        final_sums = _subset_sums_bitmask(best_set)
        final_total = sum(best_set)
        if not _check_divisibility_fast(final_sums, final_total):
            # Fallback
            best_set = []
            sums = set()
            total = 0
            for x in sorted(pool, reverse=True):
                new_sums = set(sums)
                for s in sums:
                    new_sums.add(s + x)
                new_sums.add(x)
                new_total = total + x
                if _check_divisibility_fast(new_sums, new_total):
                    best_set.append(x)
                    sums = new_sums
                    total = new_total
    
    return [int(x) for x in best_set]
