import time
import random

def construct():
    N = 256
    start_time = time.time()
    time_limit = 50.0  # Safety margin

    def compute_forbidden(S_bool):
        """
        Compute the set of numbers in [1, N] that cannot be added to the set S_bool
        without creating a 3-term arithmetic progression.
        """
        forbidden = set()
        # Get list of elements currently in the set
        S_list = [i for i in range(1, N + 1) if S_bool[i]]
        K = len(S_list)
        
        # Check all pairs
        for i in range(K):
            a = S_list[i]
            for j in range(i + 1, K):
                b = S_list[j]
                # a < b
                
                # Case 1: a, m, b is an AP. m is the middle term.
                # a + b = 2m => m = (a + b) / 2
                if (a + b) % 2 == 0:
                    m = (a + b) // 2
                    if 1 <= m <= N:
                        forbidden.add(m)
                        
                # Case 2: a, b, o is an AP. o is the larger term.
                # a + o = 2b => o = 2b - a
                o_large = 2 * b - a
                if 1 <= o_large <= N:
                    forbidden.add(o_large)
                    
                # Case 3: o, a, b is an AP. o is the smaller term.
                # o + b = 2a => o = 2a - b
                o_small = 2 * a - b
                if 1 <= o_small <= N:
                    forbidden.add(o_small)
                    
        return forbidden

    def greedy_fill(S_bool):
        """
        Repeatedly add the smallest possible valid element to S_bool.
        Returns the updated S_bool.
        """
        forbidden = compute_forbidden(S_bool)
        changed = True
        while changed:
            changed = False
            # Scan from 1 to N to find the smallest valid addition
            for x in range(1, N + 1):
                if not S_bool[x] and x not in forbidden:
                    S_bool[x] = True
                    changed = True
                    # Recompute forbidden set since S changed
                    forbidden = compute_forbidden(S_bool)
                    break # Restart scan from 1 to ensure we always pick the smallest valid one
        
        return S_bool

    def calculate_score(S_bool):
        """
        Calculate the sum of reciprocals of elements in S_bool.
        """
        total = 0.0
        for i in range(1, N + 1):
            if S_bool[i]:
                total += 1.0 / i
        return total

    # 1. Initialize with a greedy set
    S_bool = [False] * (N + 1)
    S_bool = greedy_fill(S_bool)
    best_score = calculate_score(S_bool)
    best_set = S_bool[:]

    # 2. Local Search with Random Restarts
    # We try to improve the set by removing some elements and greedily refilling.
    # This allows us to escape local optima where a large element blocks several smaller ones.
    
    while time.time() - start_time < time_limit:
        # Create a copy of the current best set
        S_current = best_set[:]
        
        # Identify elements in the current set
        indices = [i for i in range(1, N + 1) if S_current[i]]
        
        if not indices:
            break
            
        # Randomly remove a small number of elements
        # The number of removals is kept small to maintain structure
        num_removals = random.randint(1, 4)
        if len(indices) < num_removals:
            num_removals = len(indices)
            
        to_remove = random.sample(indices, num_removals)
        for idx in to_remove:
            S_current[idx] = False
            
        # Greedily fill the gaps created by removals
        S_current = greedy_fill(S_current)
        
        # Evaluate the new set
        current_score = calculate_score(S_current)
        
        if current_score > best_score:
            best_score = current_score
            best_set = S_current[:]

    # Convert boolean array to list of integers
    result = [i for i in range(1, N + 1) if best_set[i]]
    
    # Ensure non-empty
    if not result:
        result = [1]
        
    return result
