import time
import random

N = 256

def is_sidon(selected, sums, x):
    """Check if adding x to selected keeps it Sidon."""
    # Check 2x
    if 2 * x in sums:
        return False
    # Check x + a for a in selected
    for a in selected:
        if x + a in sums:
            return False
    return True

def is_maximal(selected, sums):
    """Check if selected is maximal in [1, N]."""
    for x in range(1, N + 1):
        if x in selected:
            continue
        if is_sidon(selected, sums, x):
            return False
    return True

def get_unblocked(selected, sums):
    """Return list of x in [1, N] \ selected that can be added (are not blocked)."""
    unblocked = []
    selected_set = set(selected)
    for x in range(1, N + 1):
        if x in selected_set:
            continue
        if is_sidon(selected, sums, x):
            unblocked.append(x)
    return unblocked

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin
    
    best_set = None
    best_size = N + 1
    
    # Run multiple restarts with random tie-breaking or randomization
    for restart in range(50):  # Max 50 restarts, break if time runs out
        if time.time() - start_time > timeout:
            break
            
        random.seed(restart)  # Deterministic randomization per restart
        
        selected = []
        sums = set()
        unblocked = list(range(1, N + 1))
        
        # Greedy construction
        while unblocked:
            if time.time() - start_time > timeout:
                break
                
            valid_candidates = []
            for x in unblocked:
                if is_sidon(selected, sums, x):
                    valid_candidates.append(x)
            
            if not valid_candidates:
                # Should not happen if unblocked is correct, but safety
                break
            
            # Score each candidate: how many unblocked elements will be blocked?
            best_candidate = None
            max_newly_blocked = -1
            
            # Optimization: If list is small, check all. If large, sample? 
            # N=256 is small enough to check all.
            for s in valid_candidates:
                # Simulate adding s
                new_sums = sums | {2 * s} | {s + a for a in selected}
                
                newly_blocked = 0
                for x in unblocked:
                    if x == s:
                        continue
                    # Check if x is blocked by selected U {s}
                    # x is blocked if NOT is_sidon(selected U {s}, new_sums, x)
                    # But is_sidon checks against sums. 
                    # We need to check if x can be added to selected U {s}.
                    # This is equivalent to:
                    # 1. 2x not in new_sums
                    # 2. x + y not in new_sums for all y in selected U {s}
                    
                    blocked = False
                    if 2 * x in new_sums:
                        blocked = True
                    else:
                        for y in selected:
                            if x + y in new_sums:
                                blocked = True
                                break
                        if not blocked and x + s in new_sums:
                            blocked = True
                        # Note: x+s is in new_sums by definition if s is in selected?
                        # No, new_sums contains s+a for a in selected.
                        # x+s is a new sum for the pair {x, s}.
                        # If x+s is in new_sums, it means x+s = 2s or x+s = s+a => x=s or x=a.
                        # x!=s (checked). x!=a (since x in unblocked, a in selected).
                        # So x+s cannot be in new_sums generated from s and selected.
                        # Wait. new_sums = {2s} U {s+a | a in selected}.
                        # x+s is not in this set unless x=s or x=a.
                        # So the check `x+s in new_sums` is redundant/always false for valid x.
                        # The collisions for x are:
                        # 1. 2x = 2s => x=s (impossible)
                        # 2. 2x = s+a => x = (s+a)/2
                        # 3. x+a = 2s => x = 2s-a
                        # 4. x+a = s+b => x = s+b-a
                        
                        # The loop `for y in selected: if x+y in new_sums` covers 3 and 4.
                        # And `2x in new_sums` covers 2.
                        # So the logic is correct.
                    
                    if blocked:
                        newly_blocked += 1
                
                if newly_blocked > max_newly_blocked:
                    max_newly_blocked = newly_blocked
                    best_candidate = s
            
            if best_candidate is None:
                break
            
            # Add best_candidate
            selected.append(best_candidate)
            sums.add(2 * best_candidate)
            for a in selected[:-1]:  # Exclude self
                sums.add(best_candidate + a)
            
            # Update unblocked
            unblocked = [x for x in unblocked if x != best_candidate and not is_sidon(selected, sums, x)]
            
            # If no unblocked left, we are maximal
            if not unblocked:
                break
        
        # After greedy, try to reduce size by removing elements
        # This is a local search for minimality
        improved = True
        while improved:
            improved = False
            # Shuffle to avoid bias
            random.shuffle(selected)
            for i in range(len(selected)):
                if time.time() - start_time > timeout:
                    break
                s = selected[i]
                # Try removing s
                new_selected = selected[:i] + selected[i+1:]
                new_sums = set()
                for a in new_selected:
                    new_sums.add(2 * a)
                    for b in new_selected:
                        if a < b:
                            new_sums.add(a + b)
                
                # Check if new_selected is maximal
                # If it is, we can remove s
                is_max = True
                for x in range(1, N + 1):
                    if x in new_selected:
                        continue
                    if is_sidon(new_selected, new_sums, x):
                        is_max = False
                        break
                
                if is_max:
                    selected = new_selected
                    # Recalculate sums for future removals? 
                    # We can just let the loop recompute or maintain.
                    # Recomputing sums is O(k^2). k is small.
                    sums = new_sums
                    improved = True
                    break # Restart shuffle/loop
            
            if time.time() - start_time > timeout:
                break
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected[:]
            
    if best_set is None:
        # Fallback to simple greedy if all else fails
        selected = []
        sums = set()
        for x in range(1, N + 1):
            if is_sidon(selected, sums, x):
                selected.append(x)
                sums.add(2 * x)
                for a in selected[:-1]:
                    sums.add(x + a)
        return selected
    
    return best_set
