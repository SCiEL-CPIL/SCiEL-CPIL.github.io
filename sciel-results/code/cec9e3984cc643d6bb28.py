import time
import random

DEADLINE_SECONDS = 85.0

def _check_and_add(S, S_sets, sum_to_k, x):
    """
    Try to add x to the current set S.
    S: list of current elements (not strictly needed for logic, but for size)
    S_sets: list of sets, S_sets[k] is set of sums of size k
    sum_to_k: dict mapping sum -> set of cardinalities
    x: candidate element
    
    Returns (success, new_S_sets, new_sum_to_k)
    """
    n = len(S)
    
    # Calculate new sums for each cardinality
    # New sums for cardinality k come from old sums of cardinality k-1 plus x
    new_sums = [set() for _ in range(n + 2)]
    
    for k in range(1, n + 2):
        if S_sets[k-1]:
            # Optimization: if S_sets[k-1] is large, just iterate
            for s in S_sets[k-1]:
                ns = s + x
                new_sums[k].add(ns)
                
    # Check for conflicts
    # A conflict exists if for some s in new_sums[k], s is already in sum_to_k 
    # with a cardinality j != k.
    # Note: If s is in new_sums[k] and also in S_sets[k] (old), that's fine. 
    # The conflict is only if s is associated with j != k in the *existing* sum_to_k.
    
    for k in range(1, n + 2):
        if not new_sums[k]:
            continue
        for s in new_sums[k]:
            if s in sum_to_k:
                existing_ks = sum_to_k[s]
                # If any existing cardinality is not k, we have a conflict
                # Note: k is the new cardinality for this sum s (formed by adding x to a subset of size k-1)
                if len(existing_ks) > 0:
                    # Check if k is the only one in existing_ks? No, existing_ks are from the OLD set.
                    # The new sum s is formed by cardinality k.
                    # So we need existing_ks to be a subset of {k}? No, existing_ks are disjoint from k by definition of the check?
                    # Actually, if s was in S_sets[j] for j != k, then sum_to_k[s] contains j.
                    # If j != k, conflict.
                    # If j == k, it's fine (we are just adding another way to get sum s with cardinality k).
                    for j in existing_ks:
                        if j != k:
                            return False, S_sets, sum_to_k
    
    # If no conflict, update
    new_S_sets = [set() for _ in range(n + 2)]
    for k in range(n + 2):
        if k <= n:
            new_S_sets[k] = set(S_sets[k])
        # else new_S_sets[k] is empty initially
        
    new_sum_to_k = {s: set(ks) for s, ks in sum_to_k.items()}
    
    for k in range(1, n + 2):
        for s in new_sums[k]:
            new_S_sets[k].add(s)
            if s in new_sum_to_k:
                new_sum_to_k[s].add(k)
            else:
                new_sum_to_k[s] = {k}
                
    return True, new_S_sets, new_sum_to_k

def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE_SECONDS
    
    def time_left():
        return deadline - time.time()
        
    best_size = 0
    best_set = []
    
    # Pre-define candidate lists
    # 1. Ascending
    # 2. Descending
    # 3. Random shuffles
    
    # Strategy: Greedy with efficient incremental check
    # We try multiple orderings.
    
    for attempt in range(100):
        if time_left() < 2.0:
            break
            
        if attempt == 0:
            candidates = list(range(1, 257))
        elif attempt == 1:
            candidates = list(range(256, 0, -1))
        else:
            candidates = list(range(1, 257))
            random.shuffle(candidates)
            
        current = []
        # S_sets[0] = {0}
        S_sets = [set() for _ in range(61)] # Max size 60, so indices 0..60
        S_sets[0].add(0)
        sum_to_k = {0: {0}}
        
        for x in candidates:
            if time_left() < 0.5:
                break
            if len(current) >= 60:
                break
                
            success, new_S_sets, new_sum_to_k = _check_and_add(current, S_sets, sum_to_k, x)
            if success:
                current.append(x)
                S_sets = new_S_sets
                sum_to_k = new_sum_to_k
                
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
            
    # If we have time, try local search on the best set found
    if time_left() > 5 and best_size > 0:
        # Try to swap elements in best_set to allow adding more
        # This is complex to implement efficiently with the state.
        # Instead, let's just run more random shuffles if time permits.
        # The greedy approach with random shuffles often finds good solutions.
        
        # Let's do a few more targeted attempts
        for _ in range(50):
            if time_left() < 2.0:
                break
            candidates = list(range(1, 257))
            random.shuffle(candidates)
            
            current = []
            S_sets = [set() for _ in range(61)]
            S_sets[0].add(0)
            sum_to_k = {0: {0}}
            
            for x in candidates:
                if time_left() < 0.5:
                    break
                if len(current) >= 60:
                    break
                success, new_S_sets, new_sum_to_k = _check_and_add(current, S_sets, sum_to_k, x)
                if success:
                    current.append(x)
                    S_sets = new_S_sets
                    sum_to_k = new_sum_to_k
                    
            if len(current) > best_size:
                best_size = len(current)
                best_set = current[:]

    # Final validation
    if best_set:
        # Verify validity
        S = best_set
        n = len(S)
        sums_by_k = [set() for _ in range(n + 1)]
        sums_by_k[0].add(0)
        valid = True
        for x in S:
            # We need to update sums_by_k in reverse
            for k in range(n, 0, -1):
                if sums_by_k[k-1]:
                    for s in sums_by_k[k-1]:
                        sums_by_k[k].add(s + x)
                        
        sum_to_ks = {}
        for k in range(n + 1):
            for s in sums_by_k[k]:
                if s in sum_to_ks:
                    sum_to_ks[s].add(k)
                else:
                    sum_to_ks[s] = {k}
                    
        for s, ks in sum_to_ks.items():
            if len(ks) > 1:
                valid = False
                break
                
        if not valid:
            # Fallback to a known small valid set
            best_set = [1]

    # Ensure return type is list of ints
    return [int(x) for x in best_set]
