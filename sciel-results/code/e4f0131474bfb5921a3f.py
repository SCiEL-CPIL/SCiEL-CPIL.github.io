import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    N = 1000
    max_rep = 2
    
    def is_valid_addition(x, S, counts):
        # Check if adding x to S keeps all sums <= max_rep
        # New sums are x+y for y in S
        # Note: if y == x, sum is 2x.
        # We need to check counts[x+y] < max_rep for all y in S
        # But wait, if y is in S, the pair (min(x,y), max(x,y)) is formed.
        # The sum is x+y.
        # We must ensure that before adding, counts[x+y] < max_rep.
        # Also, we must consider that if x is already in S, we can't add it again (set logic).
        # We assume x not in S.
        
        for y in S:
            s = x + y
            if counts.get(s, 0) >= max_rep:
                return False
        # Special case: 2*x
        # The loop above covers y=x if x was in S, but x is not in S.
        # So we need to check 2*x separately?
        # No, the pair (x,x) is a new representation.
        # The loop iterates y in S. x is NOT in S. So 2x is NOT checked in the loop.
        # We must check counts[2*x] < max_rep.
        if counts.get(2 * x, 0) >= max_rep:
            return False
        
        return True

    def update_counts_add(x, S, counts):
        for y in S:
            s = x + y
            counts[s] = counts.get(s, 0) + 1
        s2 = 2 * x
        counts[s2] = counts.get(s2, 0) + 1

    def update_counts_remove(x, S, counts):
        for y in S:
            s = x + y
            counts[s] -= 1
            if counts[s] == 0:
                del counts[s]
        s2 = 2 * x
        counts[s2] -= 1
        if counts[s2] == 0:
            del counts[s2]

    def greedy_build(order):
        S = []
        counts = {}
        S_set = set()
        for x in order:
            if is_valid_addition(x, S, counts):
                S.append(x)
                S_set.add(x)
                update_counts_add(x, S, counts)
        return S, counts, S_set

    # Strategy: Multiple random restarts with different orderings
    # 1. Sequential
    # 2. Random
    # 3. "Modular" heuristics?
    
    best_size = 0
    best_set = []
    best_counts = {}
    best_S_set = set()
    
    num_restarts = 50 # We will adjust this based on time
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        if r == 0:
            order = list(range(1, N + 1))
        elif r == 1:
            order = list(range(N, 0, -1))
        else:
            order = list(range(1, N + 1))
            random.shuffle(order)
            
        S, counts, S_set = greedy_build(order)
        
        # Local Search / Hill Climbing
        # Try to add elements from the complement
        # Or swap elements
        
        # Create a list of candidates not in S
        # We can just iterate through 1..N and check if not in S_set
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding any candidate
            # To be fast, we can just pick random candidates
            # Or iterate through all?
            # Let's try a few random swaps/adds per pass
            
            # 1. Try to add a random element
            candidates = [x for x in range(1, N + 1) if x not in S_set]
            if not candidates:
                break
                
            random.shuffle(candidates)
            
            # Try first few candidates
            added = False
            for x in candidates[:20]: # Check up to 20 random candidates
                if is_valid_addition(x, S, counts):
                    S.append(x)
                    S_set.add(x)
                    update_counts_add(x, S, counts)
                    improved = True
                    added = True
                    break
            
            if added:
                continue
                
            # 2. Try to swap
            # Pick a random element u from S and a random candidate v from complement
            # If removing u allows adding v (and maybe others), do it?
            # Simple swap: Remove u, add v. Size stays same.
            # We want to increase size.
            # So we need a move that increases size.
            # That means: Remove u, Add v1, Add v2.
            
            # Let's try: Remove u. Then try to add two elements.
            # This is expensive.
            
            # Alternative: Just try to add elements in a smarter way.
            # Maybe sort candidates by "potential" (fewest conflicts)?
            
            # Let's try a "repair" step:
            # If we can't add anything, try removing an element that is "blocking" many candidates.
            # This is complex. Let's stick to random restarts for now, they are robust.
            
            # If no improvement in this pass, break out of local search for this restart
            if not improved:
                break

        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            best_counts = counts
            best_S_set = S_set

    return best_set
