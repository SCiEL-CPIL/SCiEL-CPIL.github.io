import time
import random
import numpy as np
from itertools import combinations

N = 256
# Set a deadline slightly before the 60s limit to be safe
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
    A set S is maximal if for every x in [1, N] \ S, adding x violates Sidon property.
    """
    sset = set(selected)
    # Optimization: Pre-compute sums set is already provided
    for x in range(1, N + 1):
        if x in sset:
            continue
        
        # Check if x can be added
        # 1. Diagonal sum 2x must not be in sums
        if 2 * x in sums:
            continue
        
        # 2. Cross sums x+a must not be in sums for any a in S
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

def _greedy_core_and_cover(rng, core_size=15):
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
    # We use a greedy set cover approach.
    # At each step, pick the element that blocks the most currently 'addable' elements.
    while time.time() < DEADLINE - 2:
        addable = _get_addable_elements(selected, sums)
        if not addable:
            break
            
        # Find the best candidate to add
        best_candidate = None
        max_blocked = -1
        
        candidates = addable
        
        # To speed up, we can sample if there are too many candidates
        # But for N=256, the number of addable elements usually drops quickly.
        # Let's keep all candidates for accuracy in the cover phase, 
        # but limit the inner loop complexity if needed.
        
        for c in candidates:
            # c is addable, so it's a valid candidate for addition.
            # Calculate how many 'addable' elements would be blocked if we add c.
            
            # New sums if we add c
            new_sums = sums | {2 * c}
            for a in selected:
                new_sums.add(c + a)
                
            blocked_count = 0
            for y in addable:
                if y == c:
                    continue
                # Check if y is blocked in S U {c}
                # y is blocked if 2y in new_sums or y+a in new_sums for some a in S U {c}
                
                if 2 * y in new_sums:
                    blocked_count += 1
                    continue
                    
                # Check against existing a in S
                blocked = False
                for a in selected:
                    if y + a in new_sums:
                        blocked = True
                        break
                
                if blocked:
                    blocked_count += 1
                    continue
                    
                # Check against c
                if y + c in new_sums:
                    blocked_count += 1
                    
            if blocked_count > max_blocked:
                max_blocked = blocked_count
                best_candidate = c
                
        if best_candidate is None:
            # Should not happen if addable is not empty
            break
            
        # Add best_candidate
        c = best_candidate
        selected.append(c)
        sums.add(2 * c)
        for a in selected[:-1]:
            sums.add(c + a)
            
    return selected, sums

def _local_search_prune(selected, sums, rng, time_limit):
    """
    Local search to minimize size while maintaining maximality.
    Tries to remove elements from the set.
    If removing x keeps the set maximal, we remove it.
    """
    best_size = len(selected)
    best_sel = list(selected)
    best_sums = set(sums)
    start = time.time()
    
    # We iterate until no more removals are possible or time runs out
    while time.time() - start < time_limit and time.time() < DEADLINE:
        improved = False
        cur_sel = list(selected)
        cur_sums = set(sums)
        
        # Try removing each element
        for i in range(len(cur_sel)):
            if time.time() > DEADLINE:
                return best_size, best_sel, best_sums
            
            e = cur_sel[i]
            new_sel = cur_sel[:i] + cur_sel[i+1:]
            
            # Rebuild sums for the new set
            # Optimization: We could update sums incrementally, but rebuilding is safer and fast for small N
            new_sums = set()
            for a in range(len(new_sel)):
                for b in range(a, len(new_sel)):
                    new_sums.add(new_sel[a] + new_sel[b])
            
            # Check if the new set is still maximal
            if _is_maximal(new_sel, new_sums):
                # If it is maximal, we can safely remove e
                selected = new_sel
                sums = new_sums
                improved = True
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_sel = list(selected)
                    best_sums = set(sums)
                break # Restart the loop with the new set
        
        if not improved:
            # No single removal works. 
            # We could try double removals, but that is O(N^2) * O(N^2) which is expensive.
            # Given the time limit, single removal is the primary polish.
            # If single removal fails, we assume we are at a local minimum for this heuristic.
            break
            
    return best_size, best_sel, best_sums

def construct():
    rng = random.Random()
    best_size = N + 1
    best_result = None
    best_sums = None
    
    start = time.time()
    
    # Try different core sizes
    # Smaller cores might lead to smaller final sets if the cover is efficient,
    # but larger cores might provide a better "skeleton" for blocking.
    core_sizes = [5, 8, 10, 12, 15, 20]
    
    # Phase 1: Core + Cover + Prune
    for core_size in core_sizes:
        # Number of restarts per core size
        # We have ~55s. 6 core sizes.
        # If each restart takes ~1-2s, we can afford ~5-10 restarts per core size.
        # Let's do 10 restarts per core size, but break early if time is tight.
        for _ in range(10):
            if time.time() > DEADLINE - 5:
                break
            
            selected, sums = _greedy_core_and_cover(rng, core_size)
            
            # Apply local search pruning
            if time.time() < DEADLINE - 2:
                _, selected, sums = _local_search_prune(selected, sums, rng, time_limit=2.0)
            
            if len(selected) < best_size:
                best_size = len(selected)
                best_result = list(selected)
                best_sums = set(sums)
                
    # Phase 2: Pure Dominating Greedy (from original) as backup
    # This approach picks the best candidate from the start, without a fixed core.
    if time.time() < DEADLINE - 10:
        for _ in range(5):
            perm = list(range(1, N + 1))
            rng.shuffle(perm)
            selected = []
            sums = set()
            remaining = set(perm)
            
            while remaining and time.time() < DEADLINE - 2:
                best_candidate = None
                best_block_count = -1
                
                candidates = list(remaining)
                if len(candidates) > 30:
                    candidates = rng.sample(candidates, 30)
                
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
                    
                    block_count = 0
                    new_sums = sums | {2 * x}
                    for a in selected:
                        new_sums.add(x + a)
                    
                    for y in remaining:
                        if y == x:
                            continue
                        if 2 * y in new_sums:
                            block_count += 1
                        else:
                            for a in selected + [x]:
                                if y + a in new_sums:
                                    block_count += 1
                                    break
                                    
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
                
                # Update remaining to only include addable elements
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
                
            # Prune the result of Phase 2 as well
            if time.time() < DEADLINE - 2:
                _, selected, sums = _local_search_prune(selected, sums, rng, time_limit=2.0)
            
            if len(selected) < best_size:
                best_size = len(selected)
                best_result = list(selected)
                best_sums = set(sums)

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
