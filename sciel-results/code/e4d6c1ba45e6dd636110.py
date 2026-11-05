import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 4096
    best_set = []
    
    def get_new_sums(x, S):
        """Generate all triple sums involving x and elements of S."""
        new_sums = set()
        new_sums.add(3 * x)
        for y in S:
            new_sums.add(2 * x + y)
        for i in range(len(S)):
            for j in range(i, len(S)):
                new_sums.add(x + S[i] + S[j])
        return new_sums
    
    def can_add(x, S, existing_sums):
        """Check if x can be added to S without creating duplicate triple sums."""
        new_sums = get_new_sums(x, S)
        # Check internal collisions: the number of unique sums should match expected
        k = len(S)
        expected = 1 + k + k * (k + 1) // 2
        if len(new_sums) != expected:
            return False, set()
        # Check collision with existing sums
        if new_sums & existing_sums:
            return False, set()
        return True, new_sums
    
    def greedy_build(start_set, order):
        """Greedily build a set starting from start_set, trying elements in given order."""
        S = list(start_set)
        existing_sums = set()
        # Compute all triple sums of initial set
        for i in range(len(S)):
            for j in range(i, len(S)):
                for k in range(j, len(S)):
                    existing_sums.add(S[i] + S[j] + S[k])
        
        for x in order:
            if x in S:
                continue
            valid, new_sums = can_add(x, S, existing_sums)
            if valid:
                S.append(x)
                existing_sums.update(new_sums)
        return S
    
    # Strategy 1: Power-of-3 base, then greedy extend
    base = [1, 3, 9, 27, 81, 243, 729, 2187]
    remaining = [x for x in range(1, n + 1) if x not in base]
    S1 = greedy_build(base, remaining)
    if len(S1) > len(best_set):
        best_set = S1
    
    # Strategy 2: Pure greedy smallest-first
    S2 = []
    sums2 = set()
    for x in range(1, n + 1):
        valid, new_sums = can_add(x, S2, sums2)
        if valid:
            S2.append(x)
            sums2.update(new_sums)
    if len(S2) > len(best_set):
        best_set = S2
    
    # Strategy 3: Randomized greedy with multiple seeds
    for seed in range(200):
        if time.time() > deadline:
            break
        random.seed(seed)
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        S = []
        existing_sums = set()
        for x in candidates:
            if time.time() > deadline:
                break
            valid, new_sums = can_add(x, S, existing_sums)
            if valid:
                S.append(x)
                existing_sums.update(new_sums)
        
        if len(S) > len(best_set):
            best_set = S
    
    # Strategy 4: Start from power-of-3, try randomized extension
    for seed in range(200):
        if time.time() > deadline:
            break
        random.seed(seed + 10000)
        base = [1, 3, 9, 27, 81, 243, 729, 2187]
        remaining = [x for x in range(1, n + 1) if x not in base]
        random.shuffle(remaining)
        
        S = list(base)
        existing_sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                for k in range(j, len(S)):
                    existing_sums.add(S[i] + S[j] + S[k])
        
        for x in remaining:
            if time.time() > deadline:
                break
            valid, new_sums = can_add(x, S, existing_sums)
            if valid:
                S.append(x)
                existing_sums.update(new_sums)
        
        if len(S) > len(best_set):
            best_set = S
    
    return best_set
