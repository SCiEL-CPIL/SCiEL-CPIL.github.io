import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    fixed = [1, 2, 5, 11]
    best_set = fixed[:]
    best_size = len(fixed)
    
    # Initial sums
    def calc_initial_sums(S):
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums.add(S[i] + S[j])
        return sums

    initial_sums = calc_initial_sums(fixed)
    all_candidates = list(range(12, 285))
    
    def can_add(x, S, sums):
        # Check if x can be added to S without creating duplicate sums
        if 2 * x in sums:
            return False
        for y in S:
            if x + y in sums:
                return False
        return True

    def backtrack(S, sums, candidates):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
        
        if not candidates:
            return
            
        # Pruning: if even adding all remaining candidates doesn't beat best, stop
        if len(S) + len(candidates) <= best_size:
            return
            
        # Pick a candidate to branch on.
        # Strategy: Pick the smallest candidate first.
        # This is a heuristic. We could randomize this order in restarts.
        x = candidates[0]
        rest = candidates[1:]
        
        # Branch 1: Add x
        if can_add(x, S, sums):
            new_S = S + [x]
            new_sums = set(sums)
            new_sums.add(2 * x)
            for y in S:
                new_sums.add(x + y)
            
            # Forward Checking: Filter remaining candidates
            # A candidate c is viable if it can be added to new_S with new_sums
            # Note: We don't need to check c against other candidates in rest, 
            # only against the current set new_S.
            # However, if we filter here, we might remove candidates that 
            # could be added if we skipped some of new_S? No, new_S is fixed in this branch.
            # So if c conflicts with new_S, it can never be added in this branch.
            viable_rest = []
            for c in rest:
                if can_add(c, new_S, new_sums):
                    viable_rest.append(c)
            
            backtrack(new_S, new_sums, viable_rest)
            
        # Branch 2: Skip x
        # We skip x, so it's not in S, and not in sums.
        # The remaining candidates are just 'rest'.
        # We don't need to filter 'rest' based on x because x is not in the set.
        # However, we might prune if 'rest' is too small.
        backtrack(S, sums, rest)

    # Run multiple restarts with different candidate orderings to escape local optima
    # Ordering 1: Ascending
    backtrack(fixed[:], initial_sums, all_candidates)
    
    if time.time() > deadline - 10:
        return best_set

    # Ordering 2: Descending
    backtrack(fixed[:], initial_sums, sorted(all_candidates, reverse=True))
    
    if time.time() > deadline - 5:
        return best_set

    # Ordering 3: Randomized restarts
    # Try a few random orderings
    attempts = 0
    while time.time() < deadline - 2 and attempts < 20:
        random.shuffle(all_candidates)
        backtrack(fixed[:], initial_sums, all_candidates[:])
        attempts += 1
        
    return best_set
