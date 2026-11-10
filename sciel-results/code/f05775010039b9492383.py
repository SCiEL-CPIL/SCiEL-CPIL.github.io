"""
Destruction-based strategy for maximizing 3-APs without 4-APs in [1, 256].

Instead of building up a set (greedy add), we start with the full set [1, 256]
and remove elements one by one. At each step, we remove the element that:
1. Resolves (breaks) the maximum number of existing 4-APs.
2. Among those, destroys the minimum number of 3-APs.

This "minimum damage" removal strategy tends to preserve density and 3-AP structure
while efficiently eliminating 4-APs, often outperforming constructive methods
because it starts from the theoretical maximum of 3-APs and only subtracts.
"""
import time
import numpy as np

N = 256
DEADLINE_FRACTION = 0.85

def _count_3ap(flags):
    """Count distinct 3-term APs (positive difference) using boolean array."""
    count = 0
    for a in range(1, N + 1):
        if not flags[a]:
            continue
        max_d = (N - a) // 2
        for d in range(1, max_d + 1):
            if flags[a + d] and flags[a + 2*d]:
                count += 1
    return count

def _find_all_4aps(flags):
    """
    Find all 4-term APs in the current set.
    Returns a list of sets, where each set contains the 4 elements of the AP.
    """
    ap_list = []
    for a in range(1, N + 1):
        if not flags[a]:
            continue
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            if flags[a + d] and flags[a + 2*d] and flags[a + 3*d]:
                ap_list.append({a, a + d, a + 2*d, a + 3*d})
    return ap_list

def _count_3aps_destroyed_by_removal(flags, x):
    """
    Count how many 3-APs are destroyed if we remove element x.
    A 3-AP (a, a+d, a+2d) is destroyed if x is one of its elements.
    """
    destroyed = 0
    
    # x is the first element: x, x+d, x+2d
    max_d = (N - x) // 2
    for d in range(1, max_d + 1):
        if flags[x + d] and flags[x + 2*d]:
            destroyed += 1
            
    # x is the second element: x-d, x, x+d
    max_d = min(x - 1, N - x)
    for d in range(1, max_d + 1):
        if flags[x - d] and flags[x + d]:
            destroyed += 1
            
    # x is the third element: x-2d, x-d, x
    max_d = (x - 1) // 2
    for d in range(1, max_d + 1):
        if flags[x - 2*d] and flags[x - d]:
            destroyed += 1
            
    return destroyed

def _count_4aps_resolved_by_removal(flags, x, active_4aps):
    """
    Count how many active 4-APs are broken if we remove element x.
    """
    resolved = 0
    for ap in active_4aps:
        if x in ap:
            resolved += 1
    return resolved

def _local_search(flags, current_score, deadline):
    """
    Improve the set by trying to add elements back if it increases 3-AP count
    without creating new 4-APs.
    """
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Check if adding any element back is beneficial
        # We only consider elements that were removed. To keep it simple and fast,
        # we just iterate through all non-present elements.
        for x in range(1, N + 1):
            if flags[x]:
                continue
            # Check if adding x creates a 4-AP
            creates_4ap = False
            # x is first
            max_d = (N - x) // 3
            for d in range(1, max_d + 1):
                if flags[x + d] and flags[x + 2*d] and flags[x + 3*d]:
                    creates_4ap = True
                    break
            if not creates_4ap:
                # x is second
                max_d = min(x - 1, (N - x) // 2)
                for d in range(1, max_d + 1):
                    if flags[x - d] and flags[x + d] and flags[x + 2*d]:
                        creates_4ap = True
                        break
            if not creates_4ap:
                # x is third
                max_d = min((x - 1) // 2, N - x)
                for d in range(1, max_d + 1):
                    if flags[x - 2*d] and flags[x - d] and flags[x + d]:
                        creates_4ap = True
                        break
            if not creates_4ap:
                # x is fourth
                max_d = (x - 1) // 3
                for d in range(1, max_d + 1):
                    if flags[x - 3*d] and flags[x - 2*d] and flags[x - d]:
                        creates_4ap = True
                        break
            
            if not creates_4ap:
                # Tentatively add x
                flags[x] = True
                new_score = _count_3ap(flags)
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                else:
                    flags[x] = False
    return current_score

def construct():
    start_time = time.time()
    deadline = start_time + 90 * DEADLINE_FRACTION
    
    # Start with the full set
    flags = [True] * (N + 1)
    flags[0] = False  # 0 is not in [1, N]
    
    current_score = _count_3ap(flags)
    best_score = current_score
    best_set = [i for i in range(1, N + 1) if flags[i]]
    
    # Iteratively remove elements to break 4-APs
    while time.time() < deadline:
        active_4aps = _find_all_4aps(flags)
        
        if not active_4aps:
            # No more 4-APs, we are done with removal
            break
            
        # For each element in the union of active 4-APs, calculate removal cost/benefit
        candidates = set()
        for ap in active_4aps:
            candidates.update(ap)
            
        best_x = None
        best_resolved = -1
        best_destroyed = float('inf')
        
        for x in candidates:
            if not flags[x]:
                continue
                
            resolved = _count_4aps_resolved_by_removal(flags, x, active_4aps)
            destroyed = _count_3aps_destroyed_by_removal(flags, x)
            
            # Priority: Maximize resolved, then minimize destroyed
            if resolved > best_resolved or (resolved == best_resolved and destroyed < best_destroyed):
                best_resolved = resolved
                best_destroyed = destroyed
                best_x = x
                
        if best_x is None:
            break
            
        # Remove best_x
        flags[best_x] = False
        current_score = _count_3ap(flags)
        
        if current_score > best_score:
            best_score = current_score
            best_set = [i for i in range(1, N + 1) if flags[i]]
            
    # If we ran out of time while still having 4-APs, we might have an invalid set.
    # However, the loop breaks when no 4-APs are found.
    # If we exit due to deadline, we must ensure the final set has NO 4-APs.
    # If we are still in a state with 4-APs, we must aggressively remove them
    # even if it destroys many 3-APs, to ensure validity.
    
    # Check validity of current flags
    if time.time() < deadline:
        while _find_all_4aps(flags):
            active_4aps = _find_all_4aps(flags)
            if not active_4aps:
                break
            candidates = set()
            for ap in active_4aps:
                candidates.update(ap)
            # Just pick the first one to break it quickly
            # Or pick the one that resolves the most, ignoring 3-AP loss
            best_x = None
            best_resolved = -1
            for x in candidates:
                if not flags[x]:
                    continue
                resolved = _count_4aps_resolved_by_removal(flags, x, active_4aps)
                if resolved > best_resolved:
                    best_resolved = resolved
                    best_x = x
            if best_x is None:
                break
            flags[best_x] = False
            
    # Final local search to try to add elements back
    if time.time() < deadline:
        final_score = _local_search(flags, _count_3ap(flags), deadline)
        if final_score > best_score:
            best_score = final_score
            best_set = [i for i in range(1, N + 1) if flags[i]]

    # Final safety check: ensure no 4-APs
    # If the set returned has 4-APs, it's invalid and scores 0.
    # The logic above should ensure this, but let's be safe.
    # If we are at the deadline and have 4-APs, we have to remove them.
    # The "aggressive" block above handles this if time permits.
    # If time does NOT permit, we might return an invalid set.
    # To be absolutely safe, we can do a final quick check and removal.
    
    # Actually, the loop `while _find_all_4aps(flags)` in the safety block might
    # take too long if called repeatedly. But it only runs if the main loop exited
    # due to deadline or if 4-APs remain.
    
    # Let's do one final verification and cleanup if possible
    if time.time() < deadline + 5: # small buffer for final cleanup
        # Re-check 4-APs
        active_4aps = _find_all_4aps(flags)
        while active_4aps:
            # Pick an element to remove
            # To be fast, just remove the first element of the first AP
            x_to_remove = list(active_4aps[0])[0]
            flags[x_to_remove] = False
            active_4aps = _find_all_4aps(flags)
            
    best_set = [i for i in range(1, N + 1) if flags[i]]
    
    # If for some reason best_set is empty (should not happen), return [1]
    if not best_set:
        return [1]
        
    return best_set
