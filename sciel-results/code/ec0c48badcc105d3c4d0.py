import time
import math
import random
from itertools import combinations

# EVOLVE-BLOCK-START

_LCM_22 = 1
for _i in range(1, 23):
    _LCM_22 = math.lcm(_LCM_22, _i)

_WEIGHTS = {i: _LCM_22 // i for i in range(1, 23)}

def _get_sums(subset):
    """Calculate all subset sums for a given subset of indices."""
    sums = [0]
    seen = {0}
    for i in subset:
        w = _WEIGHTS[i]
        new_sums = [s + w for s in sums]
        for ns in new_sums:
            if ns in seen:
                return None, None # Collision
        sums.extend(new_sums)
        seen.update(new_sums)
    return sums, seen

def _can_add(sums, seen, w):
    """Check if weight w can be added to the set of sums without collision."""
    new_sums = [s + w for s in sums]
    for ns in new_sums:
        if ns in seen:
            return False, None
    return True, new_sums

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # 58s safety margin for 60s limit
    
    best_set = []
    best_size = 0
    best_sums = []
    best_seen = set()
    
    # Initial construction: Greedy with random tie-breaking or specific ordering
    # Try a few different initial orderings
    
    orderings = [
        list(range(1, 23)),
        list(range(22, 0, -1)),
        sorted(range(1, 23), key=lambda x: _WEIGHTS[x], reverse=True),
        sorted(range(1, 23), key=lambda x: _WEIGHTS[x]),
    ]
    
    for _ in range(20):
        orderings.append(list(range(1, 23)))
        random.shuffle(orderings[-1])
    
    # Add some heuristic orderings based on weight patterns
    orderings.append(sorted(range(1, 23), key=lambda x: _WEIGHTS[x] % 7))
    orderings.append(sorted(range(1, 23), key=lambda x: _WEIGHTS[x] % 11))
    orderings.append(sorted(range(1, 23), key=lambda x: x * _WEIGHTS[x]))

    for order in orderings:
        if time.time() > deadline:
            break
            
        current_set = set()
        current_sums, current_seen = [0], {0}
        
        # Greedy Phase
        for e in order:
            w = _WEIGHTS[e]
            valid, new_sums = _can_add(current_sums, current_seen, w)
            if valid:
                current_set.add(e)
                current_sums.extend(new_sums)
                current_seen.update(new_sums)
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            best_sums = current_sums
            best_seen = current_seen
            
        # Local Search Phase
        # We will try to improve current_set
        # We maintain current_set, current_sums, current_seen
        # We need to be careful with state updates
        
        no_improve_count = 0
        max_no_improve = 500 # Limit iterations to prevent infinite loops if stuck
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            improved = False
            
            # 1. Try to add any element
            for e in range(1, 23):
                if e in current_set:
                    continue
                w = _WEIGHTS[e]
                valid, new_sums = _can_add(current_sums, current_seen, w)
                if valid:
                    current_set.add(e)
                    current_sums.extend(new_sums)
                    current_seen.update(new_sums)
                    improved = True
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = set(current_set)
                        best_sums = current_sums
                        best_seen = current_seen
                    break # Restart search to try adding more
            
            if improved:
                no_improve_count = 0
                continue
            
            # 2. Try single swaps (remove 1, add 1)
            # To make this efficient, we iterate through candidates to remove
            # and candidates to add.
            # We need to rebuild sums for the set without the removed element.
            # This is O(2^k) where k is current size.
            
            current_list = list(current_set)
            for remove_e in current_list:
                if time.time() > deadline:
                    break
                temp_set = current_set - {remove_e}
                temp_sums, temp_seen = _get_sums(list(temp_set))
                if temp_sums is None:
                    continue # Should not happen if current is valid
                
                for add_e in range(1, 23):
                    if add_e in temp_set:
                        continue
                    w = _WEIGHTS[add_e]
                    valid, new_sums = _can_add(temp_sums, temp_seen, w)
                    if valid:
                        # Perform swap
                        current_set = temp_set
                        current_set.add(add_e)
                        current_sums = temp_sums
                        current_sums.extend(new_sums)
                        current_seen = temp_seen
                        current_seen.update(new_sums)
                        improved = True
                        if len(current_set) > best_size:
                            best_size = len(current_set)
                            best_set = set(current_set)
                            best_sums = current_sums
                            best_seen = current_seen
                        break # Found a valid swap, restart search
            
            if improved:
                no_improve_count = 0
                continue
                
            # 3. Try double swaps (remove 2, add 2)
            # This is more expensive. We only do this if stuck.
            # We limit the number of double swap attempts per "no improve" cycle
            # to keep it fast.
            
            # We'll pick a random pair to remove and try to add a random pair
            # Or systematically try a limited number of pairs.
            
            # Let's do a random sampling of double swaps to avoid O(N^4)
            if no_improve_count > 10:
                for _ in range(100): # 100 random double swap attempts
                    if time.time() > deadline:
                        break
                    
                    if len(current_set) < 2:
                        break
                    
                    # Pick 2 distinct elements to remove
                    remove_pair = random.sample(list(current_set), 2)
                    remove_set = set(remove_pair)
                    temp_set = current_set - remove_set
                    temp_sums, temp_seen = _get_sums(list(temp_set))
                    if temp_sums is None:
                        continue
                        
                    # Pick 2 distinct elements to add
                    available = [e for e in range(1, 23) if e not in temp_set]
                    if len(available) < 2:
                        continue
                    add_pair = random.sample(available, 2)
                    
                    # Check if adding add_pair[0] and add_pair[1] works
                    w1 = _WEIGHTS[add_pair[0]]
                    valid1, sums1 = _can_add(temp_sums, temp_seen, w1)
                    if not valid1:
                        continue
                    seen1 = set(temp_seen)
                    seen1.update(sums1)
                    
                    w2 = _WEIGHTS[add_pair[1]]
                    valid2, sums2 = _can_add(sums1, seen1, w2)
                    if not valid2:
                        continue
                    
                    # Valid double swap
                    current_set = temp_set
                    current_set.update(add_pair)
                    current_sums = sums1
                    current_sums.extend(sums2)
                    current_seen = seen1
                    current_seen.update(sums2)
                    
                    improved = True
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = set(current_set)
                        best_sums = current_sums
                        best_seen = current_seen
                    break # Found a valid double swap
            
            if improved:
                no_improve_count = 0
                continue
                
            # No improvement found
            no_improve_count += 1
            
            # If we are stuck for a long time, we can try a random restart
            # or just break to try a new initial ordering
            if no_improve_count > 200:
                break

    if not best_set:
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
        
    # Final verification and return
    final_list = list(best_set)
    
    # Quick sanity check: verify all subset sums are distinct
    test_sums, test_seen = _get_sums(final_list)
    if test_sums is None:
        # Fallback to a known valid set
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    return final_list

# EVOLVE-BLOCK-END
