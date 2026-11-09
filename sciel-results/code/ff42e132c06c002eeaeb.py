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

def _is_maximal(selected, sums):
    """
    Check if the set is inclusion-maximal.
    """
    sset = set(selected)
    for x in range(1, N + 1):
        if x in sset:
            continue
        
        # Check if x can be added
        if 2 * x in sums:
            continue
        
        blocked = False
        for a in selected:
            if x + a in sums:
                blocked = True
                break
        
        if not blocked:
            return False
            
    return True

def _get_addable_elements(selected, sums):
    """
    Return a list of elements x in [1, N] \ selected that can be added to selected.
    """
    sset = set(selected)
    addable = []
    for x in range(1, N + 1):
        if x in sset:
            continue
        
        if 2 * x in sums:
            continue
        
        blocked = False
        for a in selected:
            if x + a in sums:
                blocked = True
                break
        
        if not blocked:
            addable.append(x)
            
    return addable

def _greedy_core_and_cover(rng, core_size=10):
    """
    1. Generate a random Sidon set of size core_size.
    2. Iteratively add elements to block all addable elements (Greedy Set Cover).
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
            
    # Step 2: Cover Gaps
    while time.time() < DEADLINE - 2:
        addable = _get_addable_elements(selected, sums)
        if not addable:
            break
            
        # Find the best candidate to add
        best_candidate = None
        max_blocked = -1
        
        S_list = selected
        
        for c in addable:
            # Pre-compute the set of new sums introduced by c
            new_sums_vals = {2 * c}
            for a in S_list:
                new_sums_vals.add(c + a)
                
            blocked_count = 0
            for y in addable:
                if y == c:
                    continue
                
                # Check if y is blocked by adding c
                # y is blocked if 2y in new_sums_vals OR (y+s in new_sums_vals for some s in S U {c})
                
                if 2 * y in new_sums_vals:
                    blocked_count += 1
                    continue
                
                blocked = False
                # Check against s in S
                for s in S_list:
                    if y + s in new_sums_vals:
                        blocked = True
                        break
                
                # Check against c
                if not blocked:
                    if y + c in new_sums_vals:
                        blocked = True
                        
                if blocked:
                    blocked_count += 1
                    
            if blocked_count > max_blocked:
                max_blocked = blocked_count
                best_candidate = c
                
        if best_candidate is None:
            break
            
        c = best_candidate
        selected.append(c)
        sums.add(2 * c)
        for a in selected[:-1]:
            sums.add(c + a)
            
    return selected, sums

def _local_search_prune(selected, sums, rng, time_limit):
    """
    Local search to minimize size while maintaining maximality.
    """
    best_size = len(selected)
    best_sel = list(selected)
    best_sums = set(sums)
    start = time.time()
    
    while time.time() - start < time_limit and time.time() < DEADLINE:
        improved = False
        cur_sel = list(selected)
        cur_sums = set(sums)
        
        # Try removing each element
        indices = list(range(len(cur_sel)))
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() > DEADLINE:
                return best_size, best_sel, best_sums
            
            e = cur_sel[i]
            new_sel = cur_sel[:i] + cur_sel[i+1:]
            
            # Rebuild sums
            new_sums = set()
            for a in range(len(new_sel)):
                for b in range(a, len(new_sel)):
                    new_sums.add(new_sel[a] + new_sel[b])
            
            # Check if maximal
            if _is_maximal(new_sel, new_sums):
                selected = new_sel
                sums = new_sums
                improved = True
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_sel = list(selected)
                    best_sums = set(sums)
                break # Restart
        
        if not improved:
            break
            
    return best_size, best_sel, best_sums

def construct():
    rng = random.Random()
    best_size = N + 1
    best_result = None
    
    start = time.time()
    
    # Try a range of core sizes
    # Smaller cores allow more "flexibility" but might need more cover steps.
    # Larger cores might be closer to maximal but harder to prune.
    core_sizes = [3, 4, 5, 6, 8, 10, 12]
    
    for core_size in core_sizes:
        # More restarts for smaller cores
        restarts = 30 if core_size <= 5 else 15
        
        for _ in range(restarts):
            if time.time() > DEADLINE - 5:
                break
            
            selected, sums = _greedy_core_and_cover(rng, core_size)
            
            if time.time() < DEADLINE - 2:
                _, selected, sums = _local_search_prune(selected, sums, rng, time_limit=1.0)
            
            if len(selected) < best_size:
                best_size = len(selected)
                best_result = list(selected)
                
    if best_result is None:
        # Fallback: Simple greedy
        selected, sums = [], set()
        for x in range(1, N + 1):
            if 2 * x in sums: continue
            conflict = False
            for a in selected:
                if x + a in sums:
                    conflict = True
                    break
            if conflict: continue
            selected.append(x)
            sums.add(2 * x)
            for a in selected[:-1]:
                sums.add(x + a)
        best_result = selected

    return best_result

# EVOLVE-BLOCK-END
