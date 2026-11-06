"""Optimized program: bitmask-accelerated Sidon checks + randomized greedy + local search."""
import random
import time
import sys

N = 256
MAX_SUM = 2 * N  # 512
GLOBAL_DEADLINE = time.time() + 25  # Safety margin: finish by 25s to be safe under 30s limit

def _is_sidon(sel):
    """Check if sel is a Sidon set (all unordered sums a+b with a<=b distinct)."""
    sums = set()
    for i in range(len(sel)):
        a = sel[i]
        for j in range(i, len(sel)):
            b = sel[j]
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    return True

def _build_sum_mask(sel):
    """Build a bitmask of all unordered sums a+b with a<=b from sel."""
    mask = 0
    for i in range(len(sel)):
        a = sel[i]
        for j in range(i, len(sel)):
            s = a + sel[j]
            mask |= (1 << s)
    return mask

def _can_add(sel, x, sum_mask):
    """Check if x can be added to sel given its sum_mask. Returns True if addable."""
    # Check 2x
    if (sum_mask >> (2 * x)) & 1:
        return False
    # Check x + a for all a in sel
    for a in sel:
        if (sum_mask >> (a + x)) & 1:
            return False
    return True

def _is_maximal_fast(sel):
    """Fast check if sel is inclusion-maximal using bitmask."""
    sel_set = set(sel)
    sum_mask = _build_sum_mask(sel)
    
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        if _can_add(sel, x, sum_mask):
            return False
    return True

def _is_maximal(sel):
    """Check if sel is inclusion-maximal: no missing element can be added."""
    return _is_maximal_fast(sel)

def _greedy_with_blocking(random_seed=42, time_budget=2.0):
    """Randomized greedy: at each step, add the element that blocks the most remaining."""
    rng = random.Random(random_seed)
    start = time.time()
    
    selected = []
    sum_mask = 0
    remaining = set(range(1, N + 1))
    
    while remaining:
        if time.time() > GLOBAL_DEADLINE or time.time() - start > time_budget:
            break
        
        best_elem = None
        best_block_count = -1
        best_sum_mask = 0
        
        # Sample candidates to keep it fast
        candidates = list(remaining)
        if len(candidates) > 30:
            rng.shuffle(candidates)
            candidates = candidates[:30]
        
        for x in candidates:
            # Check if x can be added
            if not _can_add(selected, x, sum_mask):
                continue
            
            # Compute new sum mask
            new_sum_mask = sum_mask
            for a in selected:
                new_sum_mask |= (1 << (a + x))
            new_sum_mask |= (1 << (2 * x))
            
            # Count how many remaining elements become blocked by adding x
            # Optimization: Only check a subset of remaining if it's large, or just count total conflicts
            # For speed, we approximate blocking by counting how many y in remaining conflict with new_sum_mask
            # Actually, let's just pick the first valid one or random valid one to be extremely fast?
            # No, let's keep the blocking logic but optimize the loop.
            
            blocked = 0
            # Iterate over a sample of remaining to estimate block count if remaining is large
            rem_list = list(remaining)
            if len(rem_list) > 50:
                rng.shuffle(rem_list)
                rem_list = rem_list[:50]
            
            for y in rem_list:
                if y == x:
                    continue
                if not _can_add(selected, y, new_sum_mask):
                    blocked += 1
            
            if blocked > best_block_count or \
               (blocked == best_block_count and (best_elem is None or x < best_elem)):
                best_block_count = blocked
                best_elem = x
                best_sum_mask = new_sum_mask
        
        if best_elem is None:
            break
        
        # Add best_elem
        selected.append(best_elem)
        sum_mask = best_sum_mask
        remaining.discard(best_elem)
        
        # Early termination: if already maximal, stop
        if len(selected) >= 5:
            if _is_maximal_fast(selected):
                break
    
    return selected

def _local_search(sel, rng, time_budget=5.0):
    """Try to improve (reduce size) by removing elements and checking maximality."""
    start = time.time()
    best = sorted(sel)
    best_len = len(best)
    
    # Limit iterations to avoid long loops
    max_iter = 500 
    for iteration in range(max_iter):
        if time.time() > GLOBAL_DEADLINE or time.time() - start > time_budget:
            break
        if best_len <= 6:
            break
        
        cur = sorted(best)
        improvement_found = False
        
        # Try removing each element and check if the result is still maximal
        for i in range(len(cur)):
            if time.time() > GLOBAL_DEADLINE or time.time() - start > time_budget:
                break
            x_remove = cur[i]
            new_sel = [x for x in cur if x != x_remove]
            
            # Check if still maximal (subset of Sidon is always Sidon)
            if _is_maximal_fast(new_sel):
                if len(new_sel) < best_len:
                    best = new_sel
                    best_len = len(new_sel)
                    improvement_found = True
                    break
        
        if not improvement_found:
            # Try swap: remove one, add another (same size, may enable future reductions)
            # This is expensive, so we only do it if we have time
            if time.time() - start > time_budget * 0.5:
                break
            
            cur = sorted(best)
            found_swap = False
            # Only try a few removals for swapping
            for i in range(min(3, len(cur))):
                if time.time() > GLOBAL_DEADLINE or time.time() - start > time_budget:
                    break
                x_remove = cur[i]
                new_sel = [x for x in cur if x != x_remove]
                new_sel_set = set(new_sel)
                new_sum_mask = _build_sum_mask(new_sel)
                
                # Try adding random candidates
                candidates = [x for x in range(1, N + 1) if x not in new_sel_set]
                rng.shuffle(candidates)
                
                for x_add in candidates[:20]:
                    if not _can_add(new_sel, x_add, new_sum_mask):
                        continue
                    
                    new_final = sorted(new_sel + [x_add])
                    # Check if this new set is maximal
                    if _is_maximal_fast(new_final):
                        # Try to see if we can reduce from here
                        for j in range(len(new_final)):
                            x_remove2 = new_final[j]
                            smaller = [x for x in new_final if x != x_remove2]
                            if _is_maximal_fast(smaller):
                                if len(smaller) < best_len:
                                    best = smaller
                                    best_len = len(smaller)
                                    improvement_found = True
                                    found_swap = True
                                    break
                        if found_swap:
                            break
                if found_swap:
                    break
    
    return best

def construct():
    # Use the global deadline
    deadline = GLOBAL_DEADLINE
    
    best_set = None
    best_len = N + 1
    
    # Phase 1: Multiple randomized greedy runs
    # Limit number of runs to ensure we don't run out of time
    # Each greedy run is capped at 2s, but we check global deadline
    num_greedy_runs = 50
    for seed in range(num_greedy_runs):
        if time.time() > deadline - 10:  # Leave 10s for local search and verification
            break
        sel = _greedy_with_blocking(random_seed=seed, time_budget=1.0)
        if sel and _is_sidon(sel):
            if _is_maximal_fast(sel):
                if len(sel) < best_len:
                    best_len = len(sel)
                    best_set = sel
                    if best_len <= 7:
                        break
    
    # Also try the original ascending greedy as baseline (very fast)
    selected, sums = [], set()
    for x in range(1, N + 1):
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    if _is_maximal_fast(selected):
        if len(selected) < best_len:
            best_len = len(selected)
            best_set = selected
    
    if best_set is None:
        # Fallback: return the ascending greedy (always valid)
        return selected
    
    # Phase 2: Local search on best
    if time.time() < deadline - 5:
        rng = random.Random(12345)
        # Allocate remaining time minus safety margin
        time_left = deadline - time.time() - 2
        if time_left > 1:
            improved = _local_search(best_set, rng, time_budget=time_left)
            if improved and _is_sidon(improved) and _is_maximal_fast(improved):
                if len(improved) < best_len:
                    best_set = improved
                    best_len = len(improved)
    
    # Final verification
    if best_set and _is_sidon(best_set) and _is_maximal_fast(best_set):
        return best_set
    
    # Absolute fallback
    return selected
