import time
import random
import numpy as np

N = 256
DEADLINE = time.time() + 55

# EVOLVE-BLOCK-START

def _build_sums(selected):
    """Build the set of all unordered pair sums a+b with a<=b."""
    sums = set()
    for i in range(len(selected)):
        for j in range(i, len(selected)):
            sums.add(selected[i] + selected[j])
    return sums

def _is_maximal_fast(selected, sums):
    """
    Fast vectorized check if the set is inclusion-maximal.
    A set S is maximal if for every x in [1, N] \ S, adding x violates Sidon property.
    x is blocked if 2x in sums, or x+a in sums for some a in S.
    """
    if not selected:
        return False
    sset = set(selected)
    sel_arr = np.array(selected, dtype=np.int32)
    
    # For each x in [1,N], check if it's blocked
    for x in range(1, N + 1):
        if x in sset:
            continue
        if 2 * x in sums:
            continue
        # Check if x+a is in sums for any a in selected
        # Vectorized: (x + sel_arr) in sums
        candidates = x + sel_arr
        # Check any in sums
        blocked = False
        for c in candidates:
            if c in sums:
                blocked = True
                break
        if not blocked:
            return False
    return True

def _is_maximal_fast2(selected, sums):
    """
    Even faster: precompute sums as a boolean array for O(1) lookup.
    """
    if not selected:
        return False
    sset = set(selected)
    # sums range: min is 2*1=2, max is 256+256=512
    sums_arr = np.zeros(513, dtype=np.bool_)
    for s in sums:
        if s <= 512:
            sums_arr[s] = True
    
    sel_arr = np.array(selected, dtype=np.int32)
    
    for x in range(1, N + 1):
        if x in sset:
            continue
        if sums_arr[2 * x]:
            continue
        # Check if any x+a is in sums
        candidates = x + sel_arr
        # Vectorized check
        if np.any(sums_arr[candidates]):
            continue
        # x is not blocked -> not maximal
        return False
    return True

def _get_addable_fast(selected, sums):
    """
    Return list of elements x in [1, N] \ selected that can be added (keep Sidon).
    """
    if not selected:
        return list(range(1, N + 1))
    
    sset = set(selected)
    sums_arr = np.zeros(513, dtype=np.bool_)
    for s in sums:
        if s <= 512:
            sums_arr[s] = True
    
    sel_arr = np.array(selected, dtype=np.int32)
    addable = []
    
    for x in range(1, N + 1):
        if x in sset:
            continue
        if sums_arr[2 * x]:
            continue
        candidates = x + sel_arr
        if np.any(sums_arr[candidates]):
            continue
        addable.append(x)
    
    return addable

def _compute_block_counts(selected, sums, candidates):
    """
    For each candidate in candidates, compute how many addable elements it would block.
    Returns dict: candidate -> block_count
    """
    if not selected:
        return {c: 0 for c in candidates}
    
    addable = _get_addable_fast(selected, sums)
    if not addable:
        return {c: 0 for c in candidates}
    
    sums_arr = np.zeros(513, dtype=np.bool_)
    for s in sums:
        if s <= 512:
            sums_arr[s] = True
    
    sel_arr = np.array(selected, dtype=np.int32)
    addable_set = set(addable)
    
    block_counts = {}
    
    for c in candidates:
        # New sums after adding c
        new_sums_arr = sums_arr.copy()
        new_sums_arr[2 * c] = True
        for a in selected:
            new_sums_arr[c + a] = True
        
        # Count how many addable elements are blocked
        blocked_count = 0
        for y in addable:
            if y == c:
                continue
            if new_sums_arr[2 * y]:
                blocked_count += 1
            else:
                # Check if y+a in new_sums for any a in selected+[c]
                candidates_y = y + np.append(sel_arr, c)
                if np.any(new_sums_arr[candidates_y]):
                    blocked_count += 1
        
        block_counts[c] = blocked_count
    
    return block_counts

def _greedy_core_and_cover(rng, core_size, deadline_offset=3):
    """
    1. Generate a random Sidon set of size core_size (the core).
    2. Iteratively add elements that block the most addable elements (greedy set cover).
    """
    # Step 1: Generate Core
    selected = []
    sums = set()
    perm = list(range(1, N + 1))
    rng.shuffle(perm)
    
    for x in perm:
        if len(selected) >= core_size:
            break
        if 2 * x in sums:
            continue
        conflict = False
        for a in selected:
            if x + a in sums:
                conflict = True
                break
        if conflict:
            continue
            
        selected.append(x)
        sums.add(2 * x)
        for a in selected[:-1]:
            sums.add(x + a)
            
    # Step 2: Cover Gaps - greedily add elements that block the most
    while time.time() < DEADLINE - deadline_offset:
        addable = _get_addable_fast(selected, sums)
        if not addable:
            break
        
        # Sample candidates if too many
        candidates = addable
        if len(candidates) > 60:
            candidates = rng.sample(candidates, 60)
        
        # Compute block counts
        block_counts = _compute_block_counts(selected, sums, candidates)
        
        if not block_counts:
            break
        
        # Pick the candidate that blocks the most
        best_candidate = max(block_counts, key=block_counts.get)
        best_count = block_counts[best_candidate]
        
        if best_count <= 0 and len(addable) > 1:
            # No progress possible with these candidates, try all addable
            break
        
        # Add best_candidate
        c = best_candidate
        selected.append(c)
        sums.add(2 * c)
        for a in selected[:-1]:
            sums.add(c + a)
            
    return selected, sums

def _greedy_pure_dominating(rng, deadline_offset=3):
    """
    Pure dominating greedy: start empty, repeatedly add element that blocks most.
    """
    selected = []
    sums = set()
    remaining = set(range(1, N + 1))
    
    while remaining and time.time() < DEADLINE - deadline_offset:
        # For each candidate in remaining, compute how many it blocks
        best_candidate = None
        best_block_count = -1
        
        candidates = list(remaining)
        if len(candidates) > 50:
            candidates = rng.sample(candidates, 50)
        
        for x in candidates:
            if 2 * x in sums:
                continue
            conflict = False
            for a in selected:
                if x + a in sums:
                    conflict = True
                    break
            if conflict:
                continue
            
            # Compute how many remaining elements this blocks
            new_sums = set(sums)
            new_sums.add(2 * x)
            for a in selected:
                new_sums.add(x + a)
            
            block_count = 0
            for y in remaining:
                if y == x:
                    continue
                if 2 * y in new_sums:
                    block_count += 1
                else:
                    blocked = False
                    for a in selected + [x]:
                        if y + a in new_sums:
                            blocked = True
                            break
                    if blocked:
                        block_count += 1
            
            if block_count > best_block_count:
                best_block_count = block_count
                best_candidate = x
        
        if best_candidate is None:
            break
        
        x = best_candidate
        selected.append(x)
        sums.add(2 * x)
        for a in selected[:-1]:
            sums.add(x + a)
        
        remaining.discard(x)
        
        # Remove all blocked elements
        still_remaining = set()
        for y in remaining:
            if 2 * y in sums:
                continue
            blocked = False
            for a in selected:
                if y + a in sums:
                    blocked = True
                    break
            if not blocked:
                still_remaining.add(y)
        remaining = still_remaining
    
    return selected, sums

def _local_search(selected, sums, rng, time_limit):
    """
    Local search to minimize size while maintaining maximality.
    Tries single removals and double removals.
    """
    best_size = len(selected)
    best_sel = list(selected)
    best_sums = set(sums)
    start = time.time()
    
    while time.time() - start < time_limit and time.time() < DEADLINE - 2:
        improved = False
        cur_sel = list(selected)
        cur_sums = set(sums)
        
        # --- Single Removal Pass ---
        for i in range(len(cur_sel)):
            if time.time() > DEADLINE - 2:
                return best_size, best_sel, best_sums
            
            e = cur_sel[i]
            new_sel = cur_sel[:i] + cur_sel[i+1:]
            
            # Recompute sums
            new_sums = set()
            for a in range(len(new_sel)):
                for b in range(a, len(new_sel)):
                    new_sums.add(new_sel[a] + new_sel[b])
            
            if _is_maximal_fast2(new_sel, new_sums):
                selected = new_sel
                sums = new_sums
                improved = True
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_sel = list(selected)
                    best_sums = set(sums)
                break
        
        if not improved:
            # --- Double Removal Pass ---
            n = len(cur_sel)
            for i in range(n):
                if time.time() > DEADLINE - 2:
                    return best_size, best_sel, best_sums
                for j in range(i + 1, n):
                    new_sel = [x for k, x in enumerate(cur_sel) if k != i and k != j]
                    
                    new_sums = set()
                    for a in range(len(new_sel)):
                        for b in range(a, len(new_sel)):
                            new_sums.add(new_sel[a] + new_sel[b])
                    
                    if _is_maximal_fast2(new_sel, new_sums):
                        selected = new_sel
                        sums = new_sums
                        improved = True
                        if len(selected) < best_size:
                            best_size = len(selected)
                            best_sel = list(selected)
                            best_sums = set(sums)
                        break
                if improved:
                    break
        
        if not improved:
            break

    return best_size, best_sel, best_sums

def _local_search_swap(selected, sums, rng, time_limit):
    """
    Local search using swaps: replace one element with another to enable further removals.
    """
    best_size = len(selected)
    best_sel = list(selected)
    best_sums = set(sums)
    start = time.time()
    
    while time.time() - start < time_limit and time.time() < DEADLINE - 2:
        improved = False
        cur_sel = list(selected)
        cur_sums = set(sums)
        sset = set(cur_sel)
        
        # Try swapping one element with a random missing element
        for i in range(len(cur_sel)):
            if time.time() > DEADLINE - 2:
                return best_size, best_sel, best_sums
            
            e = cur_sel[i]
            
            for _ in range(8):
                if time.time() > DEADLINE - 2:
                    return best_size, best_sel, best_sums
                
                candidates = [x for x in range(1, N + 1) if x not in sset]
                if not candidates:
                    break
                new_e = rng.choice(candidates)
                
                new_sel = [x for x in cur_sel if x != e] + [new_e]
                new_sel.sort()
                
                new_sums = set()
                for a in range(len(new_sel)):
                    for b in range(a, len(new_sel)):
                        new_sums.add(new_sel[a] + new_sel[b])
                
                # Check if valid Sidon
                expected = len(new_sel) * (len(new_sel) + 1) // 2
                if len(new_sums) != expected:
                    continue
                
                if _is_maximal_fast2(new_sel, new_sums):
                    selected = new_sel
                    sums = new_sums
                    improved = True
                    if len(selected) < best_size:
                        best_size = len(selected)
                        best_sel = list(selected)
                        best_sums = set(sums)
                    break
            
            if improved:
                break
        
        if not improved:
            break

    return best_size, best_sel, best_sums

def construct():
    rng = random.Random()
    best_size = N + 1
    best_result = None
    best_sums = None
    
    start = time.time()
    
    # Phase 1: Core + Cover with various core sizes
    core_sizes = [8, 10, 12, 15, 20, 25]
    
    for core_size in core_sizes:
        for _ in range(15):
            if time.time() > DEADLINE - 10:
                break
            selected, sums = _greedy_core_and_cover(rng, core_size)
            if len(selected) < best_size:
                best_size = len(selected)
                best_result = list(selected)
                best_sums = set(sums)
    
    # Phase 2: Pure dominating greedy as backup
    if time.time() < DEADLINE - 10:
        for _ in range(8):
            if time.time() > DEADLINE - 10:
                break
            selected, sums = _greedy_pure_dominating(rng)
            if len(selected) < best_size:
                best_size = len(selected)
                best_result = list(selected)
                best_sums = set(sums)
    
    # Phase 3: Local search on the best found
    if best_result is not None:
        remaining = DEADLINE - time.time()
        if remaining > 5:
            t_limit = remaining - 5
            new_size, new_sel, new_sums = _local_search(best_result, best_sums, rng, t_limit)
            if new_size < best_size:
                best_size = new_size
                best_result = new_sel
                best_sums = new_sums
        
        # Phase 4: Swap-based local search
        if best_result is not None:
            remaining = DEADLINE - time.time()
            if remaining > 3:
                t_limit = remaining - 3
                new_size, new_sel, new_sums = _local_search_swap(best_result, best_sums, rng, t_limit)
                if new_size < best_size:
                    best_size = new_size
                    best_result = new_sel
                    best_sums = new_sums
    
    # Phase 5: More restarts if time remains
    if time.time() < DEADLINE - 8 and best_size > 18:
        while time.time() < DEADLINE - 4:
            # Randomly choose core size
            core_size = rng.choice([8, 10, 12, 15, 20])
            selected, sums = _greedy_core_and_cover(rng, core_size)
            
            if len(selected) < best_size:
                best_size = len(selected)
                best_result = list(selected)
                best_sums = set(sums)
            
            # Quick local search
            t_rem = DEADLINE - time.time()
            if t_rem > 5 and len(selected) <= best_size + 2:
                new_size, new_sel, new_sums = _local_search(selected, sums, rng, min(2.0, t_rem - 5))
                if new_size < best_size:
                    best_size = new_size
                    best_result = new_sel
                    best_sums = new_sums

    if best_result is None:
        # Fallback: simple greedy
        selected, sums = [], set()
        for x in range(1, N + 1):
            if 2 * x in sums:
                continue
            conflict = False
            for a in selected:
                if x + a in sums:
                    conflict = True
                    break
            if conflict:
                continue
            selected.append(x)
            sums.add(2 * x)
            for a in selected[:-1]:
                sums.add(x + a)
        best_result = selected

    return best_result

# EVOLVE-BLOCK-END
