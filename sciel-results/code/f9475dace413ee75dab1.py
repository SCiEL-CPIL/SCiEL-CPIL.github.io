"""Backtracking with bitmask-accelerated feasibility for maximum divisibility-free subset-sum set."""
# EVOLVE-BLOCK-START
import time
import random
import math
from functools import lru_cache

def _compute_forbidden(sums_set, total):
    """Compute the set of values that would create a divisibility conflict if they appeared as a sum.
    A value v is forbidden if there exists s in sums_set such that v % s == 0 and v != s,
    or s % v == 0 and s != v (but s is already in sums_set, so we only care about new sums).
    Actually: a new sum t is invalid if there exists s in sums_set with s | t or t | s.
    """
    forbidden = set()
    for s in sums_set:
        # t must be a multiple of s (t = k*s, k>=2) OR s must be a multiple of t (t divides s)
        # For t being a multiple of s: t = 2s, 3s, ... up to total
        multiple = 2 * s
        while multiple <= total:
            forbidden.add(multiple)
            multiple += s
        # For t dividing s: t = s/k for k >= 2, i.e., t is a proper divisor of s
        # t must be >= 1 and t < s
        for d in range(1, int(math.isqrt(s)) + 1):
            if s % d == 0:
                if d != s:
                    forbidden.add(d)
                other = s // d
                if other != s and other != d:
                    forbidden.add(other)
    return forbidden

def _check_pairwise_conflict(a, b, existing_sums):
    """Check if adding both a and b (which are not in existing_sums individually causing issues)
    would create a conflict. This is complex, so we'll use the full check in the main search."""
    pass

def _subset_sums_bitmask(elements):
    """Compute all nonempty subset sums using a set (bitmask approach for tracking)."""
    sums = set()
    for x in elements:
        new_sums = set()
        for s in sums:
            new_sums.add(s + x)
        new_sums.add(x)
        sums.update(new_sums)
    return sums

def _is_valid_full(S):
    """Full validity check for a set S."""
    if not S:
        return True
    sums = _subset_sums_bitmask(S)
    total = sum(S)
    sums_list = sorted(sums)
    for s in sums_list:
        if s == 0:
            continue
        multiple = 2 * s
        while multiple <= total:
            if multiple in sums:
                return False
            multiple += s
        # Also check proper divisors
        for d in range(1, int(math.isqrt(s)) + 1):
            if s % d == 0:
                if d != s and d in sums:
                    return False
                other = s // d
                if other != s and other != d and other in sums:
                    return False
    return True

def _fast_validity_check(sums_set, total):
    """Fast check: does any pair in sums_set have a divisibility relation?"""
    sums_list = sorted(sums_set)
    n = len(sums_list)
    for i in range(n):
        s = sums_list[i]
        if s == 0:
            continue
        # Check multiples
        multiple = 2 * s
        while multiple <= total:
            if multiple in sums_set:
                return False
            multiple += s
        # Check divisors (proper)
        for d in range(1, int(math.isqrt(s)) + 1):
            if s % d == 0:
                if d != s and d in sums_set:
                    return False
                other = s // d
                if other != s and other != d and other in sums_set:
                    return False
    return True

def _backtracking_search(pool, current, sums_set, total, deadline, best_ref, node_count_ref):
    """Recursive backtracking with forward checking.
    
    pool: list of remaining candidate elements
    current: list of elements in current set
    sums_set: set of all nonempty subset sums of current
    total: sum of current
    deadline: time limit
    best_ref: list containing [best_size, best_set]
    node_count_ref: list containing node count
    """
    if time.time() > deadline:
        return
    
    node_count_ref[0] += 1
    if node_count_ref[0] % 10000 == 0 and time.time() > deadline:
        return
    
    if len(current) > best_ref[0]:
        best_ref[0] = len(current)
        best_ref[1] = current[:]
    
    if len(current) >= 30:
        return
    
    # Compute forbidden values: any new sum that would create a divisibility conflict
    forbidden = set()
    for s in sums_set:
        multiple = 2 * s
        while multiple <= total + max(pool) if pool else total:
            forbidden.add(multiple)
            multiple += s
        for d in range(1, int(math.isqrt(s)) + 1):
            if s % d == 0:
                if d != s:
                    forbidden.add(d)
                other = s // d
                if other != s and other != d:
                    forbidden.add(other)
    
    max_possible_total = total + sum(pool)
    
    # Try adding each candidate
    for idx in range(len(pool)):
        if time.time() > deadline:
            return
        
        x = pool[idx]
        remaining_pool = pool[:idx] + pool[idx+1:]
        
        # Check if x alone creates a conflict
        # x must not be in forbidden
        if x in forbidden:
            continue
        
        # Compute new sums
        new_sums = set(sums_set)
        for s in sums_set:
            new_sums.add(s + x)
        new_sums.add(x)
        new_total = total + x
        
        # Check if any new sum is in forbidden (excluding x itself which we already checked)
        valid = True
        for ns in new_sums:
            if ns == x:
                continue
            if ns in forbidden:
                valid = False
                break
        if not valid:
            continue
        
        # Also need to check: do any two NEW sums divide each other?
        # And do new sums conflict with old sums beyond what forbidden captured?
        # Forbidden was computed from old sums. New sums among themselves:
        new_sums_list = sorted([ns for ns in new_sums])
        conflict = False
        for i in range(len(new_sums_list)):
            s1 = new_sums_list[i]
            if s1 == 0:
                continue
            # Check if s1 divides any other new sum
            for j in range(i+1, len(new_sums_list)):
                s2 = new_sums_list[j]
                if s2 % s1 == 0:
                    conflict = True
                    break
            if conflict:
                break
        if conflict:
            continue
        
        # Recurse
        _backtracking_search(
            remaining_pool,
            current + [x],
            new_sums,
            new_total,
            deadline,
            best_ref,
            node_count_ref
        )
        
        if len(current) + len(remaining_pool) <= best_ref[0]:
            return  # Prune: can't beat best

def _greedy_with_pruning(pool, deadline, rng, order=None):
    """Greedy construction with smart ordering and pruning."""
    if order is None:
        candidates = pool[:]
        rng.shuffle(candidates)
    else:
        candidates = order[:]
    
    current = []
    sums_set = set()
    total = 0
    used = set()
    
    for x in candidates:
        if time.time() > deadline:
            break
        if x in used:
            continue
        
        # Check if x can be added
        new_sums = set(sums_set)
        for s in sums_set:
            new_sums.add(s + x)
        new_sums.add(x)
        new_total = total + x
        
        # Fast validity check
        if _fast_validity_check(new_sums, new_total):
            current.append(x)
            sums_set = new_sums
            total = new_total
            used.add(x)
            if len(current) >= 30:
                break
    
    return current, sums_set, total

def _local_search_improved(S, sums, total, pool, deadline, rng):
    """Local search with swap operations."""
    current = list(S)
    cur_sums = set(sums)
    cur_total = total
    used = set(current)
    available = [x for x in pool if x not in used]
    
    improved = True
    no_improve = 0
    max_no_improve = 8
    
    while improved and time.time() < deadline:
        improved = False
        no_improve += 1
        if no_improve > max_no_improve:
            break
        
        # Try additions
        rng.shuffle(available)
        for x in available:
            if time.time() > deadline:
                break
            new_sums = set(cur_sums)
            for s in cur_sums:
                new_sums.add(s + x)
            new_sums.add(x)
            new_total = cur_total + x
            if _fast_validity_check(new_sums, new_total):
                current.append(x)
                cur_sums = new_sums
                cur_total = new_total
                used.add(x)
                available.remove(x)
                improved = True
                no_improve = 0
                if len(current) >= 30:
                    return current, cur_sums, cur_total
        
        # Try 1-swap
        if not improved and current:
            rng.shuffle(available)
            found_swap = False
            for i in range(len(current)):
                if time.time() > deadline or found_swap:
                    break
                removed = current[i]
                temp_current = current[:i] + current[i+1:]
                temp_sums = _subset_sums_bitmask(temp_current)
                temp_total = sum(temp_current)
                
                for x in available[:40]:
                    if time.time() > deadline:
                        break
                    if x == removed:
                        continue
                    new_sums = set(temp_sums)
                    for s in temp_sums:
                        new_sums.add(s + x)
                    new_sums.add(x)
                    new_total = temp_total + x
                    if _fast_validity_check(new_sums, new_total):
                        current = temp_current + [x]
                        cur_sums = new_sums
                        cur_total = new_total
                        used.discard(removed)
                        used.add(x)
                        available.remove(x)
                        available.append(removed)
                        improved = True
                        no_improve = 0
                        found_swap = True
                        break
                if found_swap:
                    break
    
    return current, cur_sums, cur_total

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    pool = list(range(1, 257))
    best_size = 0
    best_set = []
    
    # Precompute useful orderings
    primes = [x for x in range(2, 257) if all(x % i != 0 for i in range(2, int(math.isqrt(x)) + 1))]
    non_primes = [x for x in pool if x not in set(primes)]
    
    def divisor_count(x):
        if x < 2:
            return 1
        count = 2
        for i in range(2, int(math.isqrt(x)) + 1):
            if x % i == 0:
                count += 2
        return count
    
    by_few_divisors = sorted(pool, key=lambda x: divisor_count(x))
    by_large = sorted(pool, reverse=True)
    by_small = sorted(pool)
    large_primes = [x for x in primes if x > 100]
    
    strategies = [
        ("large", by_large),
        ("few_div", by_few_divisors),
        ("primes_first", primes + non_primes),
        ("large_prime", large_primes + [x for x in pool if x not in set(large_primes)]),
        ("small", by_small),
    ]
    
    # Phase 1: Multiple greedy restarts with different orderings
    num_greedy = 30
    for i in range(num_greedy):
        if time.time() > deadline - 20:
            break
        
        strat_name, order = strategies[i % len(strategies)]
        rng = random.Random(i * 104729 + 7)
        
        # Add noise to ordering
        candidates = order[:]
        if i % 3 == 0:
            rng.shuffle(candidates)
        elif i % 3 == 1:
            mid = len(candidates) // 2
            tail = candidates[mid:]
            rng.shuffle(tail)
            candidates[mid:] = tail
        
        current, cur_sums, cur_total = _greedy_with_pruning(pool, deadline, rng, candidates)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
        
        # Local search to improve
        if time.time() < deadline - 15 and len(current) > 0:
            extended, ext_sums, ext_total = _local_search_improved(
                current, cur_sums, cur_total, pool, deadline, rng
            )
            if len(extended) > best_size:
                best_size = len(extended)
                best_set = extended[:]
    
    # Phase 2: Backtracking search from promising starting points
    # Use the best sets found so far as seeds for backtracking
    if time.time() < deadline - 10:
        # Try backtracking from a smaller seed to see if we can grow
        seed_sizes = [max(3, best_size - 2), max(2, best_size - 3), 4]
        for seed_size in seed_sizes:
            if time.time() > deadline - 5:
                break
            
            # Pick a seed from best_set (or random)
            if best_set and len(best_set) >= seed_size:
                seed = best_set[:seed_size]
            else:
                rng = random.Random(999)
                seed = rng.sample(pool, min(seed_size, len(pool)))
            
            seed_sums = _subset_sums_bitmask(seed)
            seed_total = sum(seed)
            
            # Pool: elements not in seed, sorted by a heuristic
            seed_set = set(seed)
            remaining_pool = [x for x in pool if x not in seed_set]
            # Sort by fewer conflicts (fewer divisors/multiples in existing sums)
            def conflict_score(x):
                score = 0
                for s in seed_sums:
                    if (s + x) % s == 0 or (s + x) % x == 0:
                        score += 1
                    if s % x == 0 or x % s == 0:
                        score += 2
                return score
            remaining_pool.sort(key=conflict_score)
            
            best_ref = [best_size, best_set[:]]
            node_count_ref = [0]
            
            # Limit backtracking depth/time
            bt_deadline = min(deadline - 3, time.time() + 8)
            _backtracking_search(
                remaining_pool, seed, seed_sums, seed_total,
                bt_deadline, best_ref, node_count_ref
            )
            
            if best_ref[0] > best_size:
                best_size = best_ref[0]
                best_set = best_ref[1][:]
    
    # Phase 3: Final local search polish
    if best_set and time.time() < deadline - 2:
        final_sums = _subset_sums_bitmask(best_set)
        final_total = sum(best_set)
        rng = random.Random(424242)
        polished, _, _ = _local_search_improved(best_set, final_sums, final_total, pool, deadline, rng)
        if len(polished) > best_size:
            best_size = len(polished)
            best_set = polished[:]
    
    # Final validation
    if best_set and not _is_valid_full(best_set):
        # Fallback: rebuild with large-first greedy
        rng = random.Random(0)
        current, _, _ = _greedy_with_pruning(pool, deadline, rng, by_large)
        best_set = current
    
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
