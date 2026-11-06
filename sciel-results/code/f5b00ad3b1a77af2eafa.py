import time
import random
import numpy as np

def construct():
    deadline = time.time() + 85
    n_max = 4096
    max_sum = 3 * n_max  # 12288
    best_set = []
    
    def get_new_sums(x, S):
        """Get all new sums created by adding x to S."""
        sums = set()
        # (x,x,x)
        sums.add(3 * x)
        # (x,x,a) for a in S
        for a in S:
            sums.add(2 * x + a)
        # (x,a,b) for a,b in S, a<=b
        S_list = sorted(S)
        for i, a in enumerate(S_list):
            for b in S_list[i:]:
                sums.add(x + a + b)
        return sums
    
    def can_add(x, S, used_sum):
        """Check if x can be added to S without sum collisions."""
        new_sums = get_new_sums(x, S)
        for s in new_sums:
            if s > max_sum:
                continue
            if used_sum[s]:
                return False
        return True
    
    def add_element(x, S, used_sum):
        """Add x to S and update used_sum."""
        new_sums = get_new_sums(x, S)
        for s in new_sums:
            if s <= max_sum:
                used_sum[s] = True
        S.append(x)
    
    def build_greedy(candidates):
        """Build set by greedily adding candidates in given order."""
        S = []
        used_sum = np.zeros(max_sum + 1, dtype=bool)
        for x in candidates:
            if can_add(x, S, used_sum):
                add_element(x, S, used_sum)
        return S
    
    # Strategy 1: Greedy ascending
    S1 = build_greedy(range(1, n_max + 1))
    best_set = S1
    
    # Strategy 2: Multiple random restarts
    rng = random.Random(42)
    best_len = len(best_set)
    
    for restart in range(500):
        if time.time() > deadline - 10:
            break
        
        # Different candidate orderings
        r = restart % 5
        if r == 0:
            # Ascending
            candidates = list(range(1, n_max + 1))
        elif r == 1:
            # Descending
            candidates = list(range(n_max, 0, -1))
        elif r == 2:
            # Random
            candidates = list(range(1, n_max + 1))
            rng.shuffle(candidates)
        elif r == 3:
            # Small first, then random
            small = list(range(1, 200))
            rest = list(range(200, n_max + 1))
            rng.shuffle(rest)
            candidates = small + rest
        else:
            # Weighted: favor small
            candidates = list(range(1, n_max + 1))
            for i in range(len(candidates)):
                j = rng.randint(0, i)
                candidates[i], candidates[j] = candidates[j], candidates[i]
        
        S = build_greedy(candidates)
        
        if len(S) > best_len:
            best_len = len(S)
            best_set = S
    
    return best_set
