import time
import random
import sys
from collections import defaultdict

def _triple_sums_involving_x(x, S):
    """
    Generate all unordered triple sums involving x and elements from S.
    S is a sorted list.
    Returns a set of sums.
    """
    sums = set()
    # Triples (x, x, x)
    sums.add(3 * x)
    
    n = len(S)
    for i in range(n):
        y = S[i]
        # Triples (x, x, y)
        sums.add(2 * x + y)
        # Triples (x, y, y)
        sums.add(x + 2 * y)
        # Triples (x, y, z) where y < z
        for j in range(i + 1, n):
            z = S[j]
            sums.add(x + y + z)
            
    return sums

def _all_triple_sums(S):
    """
    Compute all unordered triple sums for a set S.
    """
    sums = set()
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                sums.add(S[i] + S[j] + S[k])
    return sums

def _check_candidate(x, S, existing_sums):
    """
    Check if x can be added to S without collisions.
    Returns (valid, new_sums)
    """
    new_sums = _triple_sums_involving_x(x, S)
    
    # Check for internal collisions in new_sums? 
    # No, _triple_sums_involving_x generates distinct triples by construction 
    # (x,x,x), (x,x,y), (x,y,y), (x,y,z).
    # However, we must ensure no collision with existing_sums.
    
    if new_sums & existing_sums:
        return False, None
        
    # Also, we must ensure that the new triples themselves are unique.
    # The set `new_sums` handles uniqueness among the new triples.
    # But wait, does _triple_sums_involving_x guarantee unique sums for the new triples?
    # Yes, because the triples are distinct unordered tuples.
    # e.g., (x,x,x) sum is 3x. (x,x,y) sum is 2x+y.
    # Could 3x == 2x+y? => x=y. But x is new, y in S, x not in S. So distinct.
    # Could 2x+y == x+2y? => x=y. Distinct.
    # Could x+y+z == 2x+y? => z=x. But z in S, x not in S. Distinct.
    # So new_sums is a set of distinct values.
    
    return True, new_sums

def _greedy_build(deadline, seed=0, use_lookahead=False):
    """
    Build a B_3[1] set using greedy with optional lookahead.
    """
    rng = random.Random(seed)
    S = []
    existing_sums = set()
    
    # Candidates are 1..4096
    # To speed up, we can just iterate 1..4096.
    # If use_lookahead, we try a few candidates and pick the best one.
    
    candidates = list(range(1, 4097))
    
    while time.time() < deadline:
        if not S:
            S.append(1)
            existing_sums = {3}
            continue
            
        found = False
        if use_lookahead:
            # Try the first 5 valid candidates and pick the one that leaves the most room?
            # Simple heuristic: pick the smallest valid candidate, but if it fails later, we don't know.
            # Let's stick to smallest valid for speed, but maybe skip some small ones if they cause tight packing?
            # Actually, just picking smallest is standard.
            # Let's try a slight variation: pick smallest, but if we have time, we can do local search later.
            pass
            
        for x in candidates:
            if x in S:
                continue
            valid, new_sums = _check_candidate(x, S, existing_sums)
            if valid:
                S.append(x)
                existing_sums |= new_sums
                found = True
                break
                
        if not found:
            break
            
    return S

def _local_search(S, existing_sums, deadline, seed=0):
    """
    Improve the set by swapping elements.
    Goal: Increase cardinality or reduce max element to allow more elements.
    """
    rng = random.Random(seed)
    S = S[:]
    best_size = len(S)
    best_S = S[:]
    best_existing_sums = existing_sums
    
    # Precompute sums for fast removal?
    # Removing an element requires recomputing all sums involving it.
    
    for _ in range(50): # Max iterations
        if time.time() > deadline:
            break
            
        if not S:
            break
            
        # Pick a random element to remove
        idx = rng.randint(0, len(S) - 1)
        removed = S[idx]
        
        # Recompute existing_sums without removed
        # This is expensive O(n^3).
        # Optimization: Only recompute sums involving removed?
        # No, we need the full set to check new candidates.
        # Let's just recompute. If n < 30, n^3 is ~27000, which is fast.
        
        S_no_removed = S[:idx] + S[idx+1:]
        
        # Try to add a new element
        added = False
        new_S = S_no_removed[:]
        
        # We want to find an x not in new_S such that x can be added.
        # To increase cardinality, we need to add an element that was previously blocked.
        # Or, we can try to add ANY element to see if we can get a larger set later?
        # No, we removed one, so size is n-1. We need to add one to get back to n.
        # To improve, we need to add an element that allows *another* element to be added later?
        # That's complex.
        
        # Simpler: Try to find an x that can be added. If found, size is back to n.
        # Does this help? Maybe if the new x is smaller, we can add more later?
        # Or if the new x is larger, it doesn't help.
        
        # Better Local Search: 
        # 1. Remove x.
        # 2. Try to add y.
        # 3. If y is added, try to add z (the one that was blocked before)?
        
        # Let's try a "Swap" move: Remove x, add y.
        # If the new set allows an additional element z to be added, we gain +1.
        
        # For now, just try to swap x with a smaller y?
        # If we replace x with a smaller y, we might free up space for larger numbers?
        
        # Let's iterate through all possible y in 1..4096 not in new_S.
        # This is slow.
        
        # Alternative: Use the "greedy fill" after removal.
        # Remove x. Run greedy to fill the set again.
        # If the new set is larger, keep it.
        
        # Since greedy is deterministic and fast, let's use it.
        # But we need to start from the current state?
        # No, greedy starts from scratch.
        
        # Let's do a "Relaxation" search.
        # Remove x.
        # Try to add any y.
        # If we can add y, and then we can add MORE elements than we had before removing x, we win.
        
        # Implementation:
        # 1. Remove x from S.
        # 2. Compute sums for S\{x}.
        # 3. Run a short greedy phase from this state to see if we can add >1 element.
        
        S_current = S_no_removed
        sums_current = _all_triple_sums(S_current)
        
        # Try to add elements greedily from S_current
        added_count = 0
        new_S_trial = S_current[:]
        sums_trial = sums_current
        
        for y in range(1, 4097):
            if y in new_S_trial:
                continue
            valid, new_sums = _check_candidate(y, new_S_trial, sums_trial)
            if valid:
                new_S_trial.append(y)
                sums_trial |= new_sums
                added_count += 1
                # If we added more than 1, we definitely improved (since we removed 1).
                # If we added 1, size is same. We might want to keep the smaller element?
                # Let's only accept if added_count > 1 OR (added_count == 1 and new_S_trial[-1] < removed)
                # Actually, if added_count == 1, we are back to size n.
                # We should prefer the set with smaller max element to allow more room.
                
                # To save time, break after adding a few?
                # If we can add 2, we are strictly better.
                if added_count > 1:
                    break
        
        new_size = len(new_S_trial)
        
        if new_size > best_size:
            best_size = new_size
            best_S = new_S_trial[:]
            best_existing_sums = sums_trial
            S = new_S_trial[:]
            existing_sums = sums_trial
        elif new_size == best_size:
            # Check if max element is smaller
            if max(new_S_trial) < max(S):
                S = new_S_trial[:]
                existing_sums = sums_trial
                best_S = S[:]
                best_existing_sums = existing_sums
        else:
            # Revert
            pass
            
    return best_S

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    best_S = []
    best_size = 0
    
    # Strategy 1: Standard Greedy
    S1 = _greedy_build(deadline, seed=1)
    if len(S1) > best_size:
        best_S = S1[:]
        best_size = len(S1)
        
    # Strategy 2: Local Search from Greedy
    if time.time() < deadline:
        sums1 = _all_triple_sums(S1)
        S2 = _local_search(S1, sums1, deadline, seed=42)
        if len(S2) > best_size:
            best_S = S2[:]
            best_size = len(S2)
            
    # Strategy 3: Randomized Greedy (Multiple seeds)
    # Try different starting points or orders?
    # The standard greedy is "smallest first".
    # What if we skip small numbers?
    
    for seed in [10, 20, 30]:
        if time.time() > deadline - 5:
            break
            
        # Modified greedy: Skip first k numbers?
        # Or just run standard greedy with a different seed if we randomize the order?
        # But "smallest first" is a specific order.
        
        # Let's try a "biased" greedy:
        # Pick smallest valid, but with some probability, pick the second smallest?
        
        S_trial = []
        existing_sums = set()
        rng = random.Random(seed)
        
        for x in range(1, 4097):
            if time.time() > deadline:
                break
            valid, new_sums = _check_candidate(x, S_trial, existing_sums)
            if valid:
                # With prob 0.1, skip if there's a larger valid candidate soon?
                # Too complex. Just accept.
                S_trial.append(x)
                existing_sums |= new_sums
                
        if len(S_trial) > best_size:
            best_S = S_trial[:]
            best_size = len(S_trial)
            
        # Local search on this trial
        if time.time() < deadline - 2:
            sums_trial = _all_triple_sums(S_trial)
            S_improved = _local_search(S_trial, sums_trial, deadline, seed=seed+100)
            if len(S_improved) > best_size:
                best_S = S_improved[:]
                best_size = len(S_improved)
                
    return best_S
