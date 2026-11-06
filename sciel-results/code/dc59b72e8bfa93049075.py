import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    N = 1000
    best_size = 0
    best_set = set()
    
    def evaluate_candidate(x, counts):
        """
        Returns (is_valid, cost) for adding x to S.
        cost = number of sums that will become tight (count goes 1 -> 2).
        Uses array-based counts for O(1) lookups.
        """
        cost = 0
        # Check 2*x
        c_2x = counts[2 * x]
        if c_2x >= 2:
            return False, 0
        if c_2x == 1:
            cost += 1
            
        # Check x + y for y in S (using precomputed set)
        # We need to iterate over existing elements, but we can use the counts array
        # Actually, we still need to know which y are in S to check x+y
        # But we can optimize by only checking sums that exist
        
        return True, cost

    def greedy_with_cost(seed=0):
        random.seed(seed)
        S = set()
        # Use an array for counts: counts[s] = number of representations of s as a+b with a<=b from S
        # s ranges from 2 to 2*N = 2000
        counts = np.zeros(2 * N + 1, dtype=np.int8)
        
        for _ in range(N):
            best_x = None
            best_cost = float('inf')
            valid_candidates = []
            
            # Evaluate all candidates not in S
            for x in range(1, N + 1):
                if x in S:
                    continue
                    
                # Check if adding x is valid
                valid = True
                cost = 0
                
                # Check 2*x
                c_2x = counts[2 * x]
                if c_2x >= 2:
                    valid = False
                elif c_2x == 1:
                    cost += 1
                    
                # Check x + y for y in S
                if valid:
                    for y in S:
                        s = x + y
                        c_xy = counts[s]
                        if c_xy >= 2:
                            valid = False
                            break
                        if c_xy == 1:
                            cost += 1
                
                if valid:
                    if cost < best_cost:
                        best_cost = cost
                        valid_candidates = [x]
                    elif cost == best_cost:
                        valid_candidates.append(x)
            
            if not valid_candidates:
                break
            
            # Pick randomly from the best cost candidates
            x = random.choice(valid_candidates)
            S.add(x)
            
            # Update counts
            counts[2 * x] += 1
            for y in S:
                if y == x:
                    continue
                s = x + y
                counts[s] += 1
        
        return S

    # Run multiple restarts with different seeds
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        S = greedy_with_cost(seed=i)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
        
    return list(best_set)
