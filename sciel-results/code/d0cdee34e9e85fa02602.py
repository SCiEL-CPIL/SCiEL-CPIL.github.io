import random
import time
from itertools import combinations

def can_add(x, S, pair_sums, triple_sums):
    """
    Check if x can be added to S.
    S: list of selected integers
    pair_sums: list of sums of all pairs in S
    triple_sums: list of sums of all triples in S
    """
    # 1. x must not divide any sum of 3 existing elements
    for t in triple_sums:
        if t % x == 0:
            return False
    
    # 2. No existing a in S can divide (x + sum of any 2 existing elements)
    for a in S:
        # Check if there exists p in pair_sums such that (p + x) % a == 0
        # This is equivalent to p % a == (-x) % a
        target = (-x) % a
        for p in pair_sums:
            if p % a == target:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # Greedy initialization: try to build a large set
    # Start with larger numbers as they're "safer" (fewer divisors in the relevant range for sums)
    selected = []
    pair_sums = []
    triple_sums = []
    
    for x in range(192, 0, -1):
        if time.time() > deadline:
            break
        
        if can_add(x, selected, pair_sums, triple_sums):
            # Add x
            selected.append(x)
            # Update pair_sums: new pairs are (x, s) for all s in selected[:-1]
            new_pairs = [x + s for s in selected[:-1]]
            pair_sums.extend(new_pairs)
            # Update triple_sums: new triples are (x, s1, s2) for all (s1, s2) in old pair_sums
            # Note: pair_sums currently contains old pairs + new pairs.
            # We need to add x + p for p in OLD pair_sums.
            old_pair_sums = pair_sums[:-len(new_pairs)] if new_pairs else []
            new_triples = [x + p for p in old_pair_sums]
            triple_sums.extend(new_triples)

    best_set = list(selected)
    best_size = len(best_set)
    
    # Local search to improve
    random.seed(42)
    
    # Pre-compute list of all numbers
    all_nums = list(range(1, 193))
    
    for iteration in range(50000):
        if time.time() > deadline:
            break
        
        current = list(best_set)
        current_size = len(current)
        
        # Re-compute pair and triple sums for current set if we modified it, 
        # but since we only modify best_set on success, we can keep them in sync 
        # or just re-compute for the candidate. 
        # For simplicity and correctness in local search, let's re-compute for the candidate 
        # only if we are about to check validity, or maintain them.
        # Given N is small, re-computing pair_sums and triple_sums for a set of size ~40 
        # takes O(N^2) and O(N^3) which is fast.
        
        # Strategy: Try to add a random number
        excluded = [x for x in all_nums if x not in set(current)]
        if excluded:
            x = random.choice(excluded)
            # Check validity
            # We need pair_sums and triple_sums of 'current'
            # Let's compute them on the fly for the check to avoid state management errors
            # But this is O(N^3). 50000 * 40^3 is too slow.
            # We must maintain pair_sums and triple_sums for best_set.
            
            # Let's maintain them properly.
            # We will store pair_sums_best and triple_sums_best corresponding to best_set.
            # If we change best_set, we update them.
            
            # To simplify this edit, I will refactor to maintain state variables.
            pass

    # Refactored Local Search with State Maintenance
    # Re-initializing state from best_set
    S = list(best_set)
    pair_sums = []
    triple_sums = []
    # Rebuild sums for best_set
    for i in range(len(S)):
        for j in range(i+1, len(S)):
            pair_sums.append(S[i] + S[j])
    for i in range(len(S)):
        for j in range(i+1, len(S)):
            for k in range(j+1, len(S)):
                triple_sums.append(S[i] + S[j] + S[k])

    for iteration in range(50000):
        if time.time() > deadline:
            break
        
        # Try to add a random number
        excluded = [x for x in all_nums if x not in set(S)]
        if excluded:
            x = random.choice(excluded)
            if can_add(x, S, pair_sums, triple_sums):
                # Accept
                S.append(x)
                new_pairs = [x + s for s in S[:-1]]
                pair_sums.extend(new_pairs)
                old_pair_sums = pair_sums[:-len(new_pairs)]
                new_triples = [x + p for p in old_pair_sums]
                triple_sums.extend(new_triples)
                if len(S) > best_size:
                    best_set = list(S)
                    best_size = len(S)
                continue

        # Try a swap: remove one, add one
        if S:
            remove_idx = random.randint(0, len(S) - 1)
            removed_val = S[remove_idx]
            
            # Tentatively remove
            S_tentative = [x for i, x in enumerate(S) if i != remove_idx]
            
            # Recompute sums for tentative? Too slow.
            # Instead, just try to add a random x to S_tentative using a slower check 
            # or just skip swaps if too complex. 
            # Given the greedy start is strong, simple additions might suffice.
            # But swaps are powerful.
            
            # Let's just try to add a random x to the *original* S (which is valid) 
            # but we already did that.
            
            # Alternative: Random restart / Perturbation
            # Remove a random element, then try to add several others.
            # This is complex to manage state for.
            
            # Let's stick to additions for now, as they are O(1) state update.
            # To get more diversity, we can occasionally remove a random element 
            # and then try to add others.
            
            if iteration % 10 == 0:
                # Remove a random element to allow different structures
                S.pop(remove_idx)
                # Rebuild sums (O(N^3))
                pair_sums = []
                triple_sums = []
                for i in range(len(S)):
                    for j in range(i+1, len(S)):
                        pair_sums.append(S[i] + S[j])
                for i in range(len(S)):
                    for j in range(i+1, len(S)):
                        for k in range(j+1, len(S)):
                            triple_sums.append(S[i] + S[j] + S[k])
                
                # Try to add 3 random numbers
                for _ in range(3):
                    if not S: break
                    excluded = [x for x in all_nums if x not in set(S)]
                    if not excluded: break
                    x = random.choice(excluded)
                    if can_add(x, S, pair_sums, triple_sums):
                        S.append(x)
                        new_pairs = [x + s for s in S[:-1]]
                        pair_sums.extend(new_pairs)
                        old_pair_sums = pair_sums[:-len(new_pairs)]
                        new_triples = [x + p for p in old_pair_sums]
                        triple_sums.extend(new_triples)
                        if len(S) > best_size:
                            best_set = list(S)
                            best_size = len(S)
                    else:
                        break

    return best_set
