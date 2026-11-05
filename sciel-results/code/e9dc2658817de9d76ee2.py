import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s safety margin
    
    n = 192
    best_set = []
    best_size = 0
    
    # Candidate pool
    candidates = list(range(1, n + 1))
    
    # Helper to check if x can be added to S
    def can_add(x, S):
        # 1. x does not divide sum of any 3 distinct others in S
        if len(S) >= 3:
            for a, b, c in combinations(S, 3):
                if (a + b + c) % x == 0:
                    return False
        
        # 2. No a in S divides x + b + c for distinct b, c in S \ {a}
        if len(S) >= 2:
            for a in S:
                others = [v for v in S if v != a]
                if len(others) < 2:
                    continue
                for b, c in combinations(others, 2):
                    if (x + b + c) % a == 0:
                        return False
        return True

    # Randomized Greedy Construction
    def greedy_random(order):
        S = []
        S_set = set()
        for x in order:
            if time.time() > deadline:
                break
            if x in S_set:
                continue
            if can_add(x, S):
                S.append(x)
                S_set.add(x)
        return S

    # Try different orders
    # 1. Descending
    # 2. Ascending
    # 3. Random
    # 4. Biased Random (more large numbers early)
    
    attempts = 0
    max_attempts = 200 # Limit attempts to allow time check
    
    while time.time() < deadline - 2:
        attempts += 1
        
        # Generate a random order with some bias
        # Strategy: Mix of random and sorted
        if attempts % 4 == 0:
            # Descending
            order = list(range(n, 0, -1))
        elif attempts % 4 == 1:
            # Ascending
            order = list(range(1, n + 1))
        elif attempts % 4 == 2:
            # Random
            order = candidates[:]
            random.shuffle(order)
        else:
            # Biased: Large numbers first, but shuffled within groups
            large = [x for x in candidates if x > n // 2]
            small = [x for x in candidates if x <= n // 2]
            random.shuffle(large)
            random.shuffle(small)
            order = large + small
            
        S = greedy_random(order)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
            # If we reached the theoretical max, we can stop
            if best_size == 48:
                break
                
    return best_set
