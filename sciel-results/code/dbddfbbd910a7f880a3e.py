import numpy as np
import time
import random

def construct():
    n = 512
    start_time = time.time()
    # Safety margin: evaluator kills at 90s. We use 85s as our hard deadline.
    deadline = start_time + 85.0

    def is_valid_addition(x, S, n):
        """Check if x can be added to set S (boolean array) without forming a 4-AP."""
        # Case 1: x is 4th term: x-3d, x-2d, x-d in S
        # d >= 1, x-3d >= 1 => d <= (x-1)//3
        for d in range(1, (x - 1) // 3 + 1):
            if S[x - d] and S[x - 2 * d] and S[x - 3 * d]:
                return False
        
        # Case 2: x is 1st term: x+d, x+2d, x+3d in S
        # x+3d <= n => d <= (n-x)//3
        for d in range(1, (n - x) // 3 + 1):
            if S[x + d] and S[x + 2 * d] and S[x + 3 * d]:
                return False
        
        # Case 3: x is 2nd term: x-d, x+d, x+2d in S
        # x-d >= 1 => d <= x-1
        # x+2d <= n => d <= (n-x)//2
        max_d = min(x - 1, (n - x) // 2)
        for d in range(1, max_d + 1):
            if S[x - d] and S[x + d] and S[x + 2 * d]:
                return False
        
        # Case 4: x is 3rd term: x-2d, x-d, x+d in S
        # x-2d >= 1 => d <= (x-1)//2
        # x+d <= n => d <= n-x
        max_d = min((x - 1) // 2, n - x)
        for d in range(1, max_d + 1):
            if S[x - 2 * d] and S[x - d] and S[x + d]:
                return False
        
        return True

    def has_no_two_in_base3(x):
        """Check if x has no digit '2' in base 3."""
        while x > 0:
            if x % 3 == 2:
                return False
            x //= 3
        return True

    def local_improve(S, count):
        """Try to add any valid element to S. Returns (S, new_count)."""
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for x in range(1, n + 1):
                if not S[x] and is_valid_addition(x, S, n):
                    S[x] = True
                    count += 1
                    improved = True
        return S, count

    # 1. Generate Initial Candidates
    # Candidate A: Cantor-like set (numbers with no '2' in base 3)
    S_cantor = np.zeros(n + 1, dtype=bool)
    for i in range(1, n + 1):
        if has_no_two_in_base3(i):
            S_cantor[i] = True
    
    # Candidate B: Standard Greedy (smallest to largest)
    S_greedy = np.zeros(n + 1, dtype=bool)
    for x in range(1, n + 1):
        if is_valid_addition(x, S_greedy, n):
            S_greedy[x] = True
    
    # Candidate C: Randomized Greedy (one sample to see if it's better than deterministic)
    S_rand_greedy = np.zeros(n + 1, dtype=bool)
    candidates = list(range(1, n + 1))
    random.shuffle(candidates)
    for x in candidates:
        if is_valid_addition(x, S_rand_greedy, n):
            S_rand_greedy[x] = True

    # Select the best initial set
    candidates_sets = [S_cantor, S_greedy, S_rand_greedy]
    best_set = None
    best_count = 0
    for S in candidates_sets:
        c = int(np.sum(S))
        if c > best_count:
            best_count = c
            best_set = S.copy()

    # 2. Polish the initial best set
    best_set, best_count = local_improve(best_set, best_count)

    # 3. Main Search Loop: Randomized Restart with Local Search
    # We will run as many iterations as possible until deadline.
    # Each iteration:
    #   a) Start from best_set
    #   b) Randomly remove k elements (k in 1..5)
    #   c) Perform a randomized greedy fill (shuffle remaining candidates, add if valid)
    #   d) Local improve
    #   e) If better, update best_set

    # Optimization: Pre-allocate arrays? No, copy is fast enough for n=512.
    
    while time.time() < deadline - 5.0:
        # Check time before starting a new restart to avoid long operations
        if time.time() > deadline - 2.0:
            break

        S_curr = best_set.copy()
        elements_in_set = [i for i in range(1, n + 1) if S_curr[i]]
        
        if len(elements_in_set) <= 5:
            continue

        # Randomly remove some elements to escape local optimum
        # Removing 1-3 is usually better than 5 for this size
        num_remove = random.randint(1, 4)
        if num_remove > len(elements_in_set):
            num_remove = len(elements_in_set)
            
        to_remove = random.sample(elements_in_set, num_remove)
        for elem in to_remove:
            S_curr[elem] = False
        
        # Randomized Greedy Fill
        # We want to try to add back elements.
        # Instead of shuffling ALL n elements, we can just shuffle the ones NOT in set?
        # Or shuffle all and check? Shuffling 512 is fast.
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if not S_curr[x] and is_valid_addition(x, S_curr, n):
                S_curr[x] = True
        
        curr_count = int(np.sum(S_curr))
        
        # Local improvement on the new candidate
        S_curr, curr_count = local_improve(S_curr, curr_count)
        
        if curr_count > best_count:
            best_count = curr_count
            best_set = S_curr.copy()
            
            # Optional: Immediate local improvement on the new best? 
            # local_improve was just called. But maybe we can do another pass?
            # It's already done inside local_improve.

    # 4. Final Polish
    best_set, best_count = local_improve(best_set, best_count)

    # Return the list of elements
    return [i for i in range(1, n + 1) if best_set[i]]

