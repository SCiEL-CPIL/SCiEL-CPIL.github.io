import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin for 90s limit
    
    def get_reachable(S):
        """Returns the set of all subset sums of S that are <= 200."""
        reachable = {0}
        for x in S:
            new = {s + x for s in reachable if s + x <= 200}
            reachable |= new
        return reachable

    def is_valid_set(S):
        """Checks if no subset of S sums to 200."""
        reachable = {0}
        for x in S:
            new = {s + x for s in reachable if s + x <= 200}
            if 200 in new:
                return False
            reachable |= new
        return True

    def greedy_build(order):
        """Greedily adds elements from order to a set if valid."""
        S = set()
        reachable = {0}
        for x in order:
            # Check if adding x creates a sum of 200
            # This happens if 200 - x is already reachable
            if (200 - x) not in reachable:
                S.add(x)
                # Update reachable sums
                new_reach = {s + x for s in reachable if s + x <= 200}
                reachable |= new_reach
        return S

    best_S = set()
    best_size = 0
    
    # 1. Standard Greedy: 100 down to 1
    S = greedy_build(range(100, 0, -1))
    if len(S) > best_size:
        best_S = S
        best_size = len(S)

    # 2. Standard Greedy: 1 up to 100
    S = greedy_build(range(1, 101))
    if len(S) > best_size:
        best_S = S
        best_size = len(S)

    # 3. Randomized Greedy
    # Run as many as possible within a time budget
    time_budget_for_greedy = 20.0
    while time.time() < start_time + time_budget_for_greedy:
        order = list(range(1, 101))
        random.shuffle(order)
        S = greedy_build(order)
        if len(S) > best_size:
            best_S = S
            best_size = len(S)

    # 4. Local Search on the best candidate
    # We try to increase the size by adding elements, or maintaining size by swapping.
    S = set(best_S)
    
    # Helper to check validity and get reachable set
    # We maintain S and reachable
    
    def update_state(S):
        return get_reachable(S)
    
    reachable = update_state(S)
    
    # Local Search Loop
    while time.time() < deadline:
        improved = False
        
        # Try to add a random excluded element
        excluded = [x for x in range(1, 101) if x not in S]
        if excluded:
            random.shuffle(excluded)
            for x in excluded[:10]: # Try first 10 shuffled
                if (200 - x) not in reachable:
                    S.add(x)
                    new_reach = {s + x for s in reachable if s + x <= 200}
                    reachable |= new_reach
                    best_size = len(S)
                    improved = True
                    break
        
        if improved:
            continue

        # If no direct addition, try swaps
        # Pick a random x from excluded, y from S
        if excluded:
            for _ in range(50): # 50 random swap attempts
                if time.time() > deadline:
                    break
                x = random.choice(excluded)
                y = random.choice(list(S))
                
                # Construct candidate set
                S_candidate = S.copy()
                S_candidate.remove(y)
                S_candidate.add(x)
                
                # Check validity
                if is_valid_set(S_candidate):
                    S = S_candidate
                    reachable = update_state(S)
                    improved = True
                    break
                    
    return list(best_S)
