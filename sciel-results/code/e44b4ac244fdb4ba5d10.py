"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _check_divisibility_sorted(sorted_sums):
    """Check if no two distinct values in sorted_sums divide one another."""
    n = len(sorted_sums)
    for i in range(n):
        si = sorted_sums[i]
        for j in range(i + 1, n):
            sj = sorted_sums[j]
            if sj % si == 0:
                return False
    return True

def _valid_set(nums):
    """Check if the set satisfies the divisibility-incomparable subset sum condition."""
    n = len(nums)
    sums_set = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += nums[i]
        if s in sums_set:
            return False
        sums_set.add(s)
    sorted_sums = sorted(sums_set)
    return _check_divisibility_sorted(sorted_sums)

def _try_add(current, current_sums, cand):
    """Try to add cand to current set. Returns (new_sums) if valid, else None."""
    new_sums = set()
    for s in current_sums:
        new_sums.add(s + cand)
    new_sums.add(cand)
    
    all_sums = current_sums | new_sums
    if len(all_sums) != len(current_sums) + len(new_sums):
        return None
    
    sorted_sums = sorted(all_sums)
    if not _check_divisibility_sorted(sorted_sums):
        return None
    
    return all_sums

def _greedy_build(candidate_order, deadline, max_size=30):
    """Build a valid set by greedily adding elements in the given order."""
    current = []
    current_sums = set()
    
    for cand in candidate_order:
        if time.time() > deadline:
            break
        if len(current) >= max_size:
            break
        
        new_sums = _try_add(current, current_sums, cand)
        if new_sums is not None:
            current.append(cand)
            current_sums = new_sums
    
    return current, current_sums

def _local_search_with_removals(current, current_sums, candidate_pool, deadline, max_size=30):
    """Local search with add, swap, and removal (downhill) moves."""
    best = list(current)
    best_sums = set(current_sums)
    best_size = len(best)
    rng = random.Random(int(time.time() * 1000) % 100000)
    
    for step in range(20000):
        if time.time() > deadline:
            break
        
        action = rng.random()
        
        if action < 0.35 and len(current) < max_size:
            # ADD a new element
            cand = candidate_pool[rng.randint(0, len(candidate_pool) - 1)]
            if cand not in current:
                new_sums = _try_add(current, current_sums, cand)
                if new_sums is not None:
                    current.append(cand)
                    current_sums = new_sums
                    if len(current) > best_size:
                        best = list(current)
                        best_sums = set(current_sums)
                        best_size = len(current)
        
        elif action < 0.65 and len(current) >= 1:
            # SWAP: replace one element with another
            idx = rng.randint(0, len(current) - 1)
            old_val = current[idx]
            new_val = candidate_pool[rng.randint(0, len(candidate_pool) - 1)]
            
            if new_val != old_val and new_val not in current:
                # Try the swap
                trial = current[:idx] + current[idx+1:]
                # Build trial sums from scratch minus old_val
                # Easier: rebuild from trial
                trial_sums = set()
                n = len(trial)
                for mask in range(1, 1 << n):
                    s = 0
                    for i in range(n):
                        if mask & (1 << i):
                            s += trial[i]
                    if s in trial_sums:
                        break
                    trial_sums.add(s)
                else:
                    # No duplicate, now try adding new_val
                    new_sums = _try_add(trial, trial_sums, new_val)
                    if new_sums is not None:
                        current = trial + [new_val]
                        current_sums = new_sums
                        if len(current) > best_size:
                            best = list(current)
                            best_sums = set(current_sums)
                            best_size = len(current)
        
        else:
            # REMOVAL (downhill move): remove a random element to escape local optima
            if len(current) >= 2:
                idx = rng.randint(0, len(current) - 1)
                removed = current.pop(idx)
                # Rebuild sums
                n = len(current)
                current_sums = set()
                for mask in range(1, 1 << n):
                    s = 0
                    for i in range(n):
                        if mask & (1 << i):
                            s += current[i]
                    current_sums.add(s)
                # Put removed back into a pool for potential re-add
                # (implicitly handled by candidate_pool which includes all range)
    
    return best, best_sums

def construct():
    start = time.time()
    deadline = start + 82  # 8s safety margin
    
    best = [31, 32, 34]  # Fallback - known valid set of size 3
    rng = random.Random(42)
    
    # Strategy 1: Pure greedy from largest numbers
    candidates_desc = list(range(256, 127, -1))
    current, _ = _greedy_build(candidates_desc, deadline)
    
    if len(current) > len(best):
        best = list(current)
    
    # Strategy 2: Multiple randomized restarts with biased ordering
    for attempt in range(300):
        if time.time() > deadline:
            break
        
        pool = list(range(256, 127, -1))
        rng.shuffle(pool)
        
        current2, _ = _greedy_build(pool, deadline)
        
        if len(current2) > len(best):
            best = list(current2)
    
    # Strategy 3: Try adding mid-range numbers to best set
    if len(best) < 30 and time.time() < deadline:
        base = list(best)
        base_sums = set()
        for mask in range(1, 1 << len(base)):
            s = 0
            for i in range(len(base)):
                if mask & (1 << i):
                    s += base[i]
            base_sums.add(s)
        
        mid_candidates = list(range(128, 63, -1))
        rng.shuffle(mid_candidates)
        
        for cand in mid_candidates:
            if time.time() > deadline:
                break
            if len(base) >= 30:
                break
            if cand in base:
                continue
            
            new_sums = _try_add(base, base_sums, cand)
            if new_sums is not None:
                base.append(cand)
                base_sums = new_sums
        
        if len(base) > len(best):
            best = list(base)
    
    # Strategy 4: Local search with swaps, additions, AND REMOVALS (key improvement)
    if time.time() < deadline:
        current_best = list(best)
        current_best_sums = set()
        n = len(current_best)
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += current_best[i]
            current_best_sums.add(s)
        
        # Candidate pool for local search: mix of large and mid-range
        candidate_pool = list(range(65, 257))
        
        improved, improved_sums = _local_search_with_removals(
            current_best, current_best_sums, candidate_pool, deadline
        )
        
        if len(improved) > len(best):
            best = list(improved)
    
    # Strategy 5: Systematic search in narrow intervals
    if time.time() < deadline:
        for interval_start in [192, 160, 128, 96, 64]:
            if time.time() > deadline:
                break
            if len(best) >= 30:
                break
            
            interval = list(range(min(interval_start + 64, 257), max(interval_start, 1) - 1, -1))
            rng.shuffle(interval)
            
            current3, _ = _greedy_build(interval, deadline)
            
            if len(current3) < 30:
                smaller = list(range(interval_start, max(interval_start - 64, 1) - 1, -1))
                rng.shuffle(smaller)
                current3, _ = _greedy_build(smaller, deadline, max_size=30)
            
            if len(current3) > len(best):
                best = list(current3)
    
    # Strategy 6: Final local search on the best set found
    if time.time() < deadline and len(best) < 30:
        current_final = list(best)
        current_final_sums = set()
        n = len(current_final)
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += current_final[i]
                if mask & (1 << i):
                    s += current_final[i]
            current_final_sums.add(s)
        
        candidate_pool = list(range(65, 257))
        final_result, _ = _local_search_with_removals(
            current_final, current_final_sums, candidate_pool, deadline
        )
        
        if len(final_result) > len(best):
            best = list(final_result)
    
    # Final verification
    if _valid_set(best):
        return list(best)
    else:
        return [31, 32, 34]
# EVOLVE-BLOCK-END
