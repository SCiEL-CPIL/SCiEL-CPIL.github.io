import time
import random
import numpy as np

def construct():
    n = 512
    start_time = time.time()
    time_limit = 57.0  # Leave 3s buffer before 60s kill
    
    # Precompute all 4-APs and build conflict lists
    # conflict[x] = list of (a,b,c) such that {x,a,b,c} is a 4-AP
    conflict = [[] for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            dd = a + 3 * d
            conflict[a].append((b, c, dd))
            conflict[b].append((a, c, dd))
            conflict[c].append((a, b, dd))
            conflict[dd].append((a, b, c))
    
    # Helper: Check if adding x to set S (as a set) creates a 4-AP
    def creates_4ap(S, x):
        for (a, b, c) in conflict[x]:
            if a in S and b in S and c in S:
                return True
        return False
    
    # Helper: Greedily fill the set given a starting set and order
    def greedy_fill(S, order):
        S = set(S)
        for x in order:
            if x not in S:
                if not creates_4ap(S, x):
                    S.add(x)
        return S
    
    # Helper: Generate numbers with only digits {1,2} in base 3
    def ternary_digits_12(max_val):
        result = []
        for i in range(1, max_val + 1):
            # Convert to base 3
            num = i
            digits = []
            while num > 0:
                digits.append(num % 3)
                num //= 3
            # Check if all digits are 1 or 2
            if all(d in (1, 2) for d in digits):
                result.append(i)
        return result
    
    # Initial construction using ternary digit-based heuristic
    # Start with numbers that have only digits {1,2} in base 3
    ternary_set = set(ternary_digits_12(n))
    
    # Verify this set is 4-AP-free (it should be by construction)
    # Then greedily add more elements
    current = greedy_fill(ternary_set, list(range(1, n + 1)))
    
    best_set = set(current)
    best_size = len(current)
    
    random.seed(42)
    
    # Also try standard greedy approaches for comparison
    current = greedy_fill(set(), list(range(1, n + 1)))
    if len(current) > best_size:
        best_set = set(current)
        best_size = len(current)
        
    current = greedy_fill(set(), list(range(n, 0, -1)))
    if len(current) > best_size:
        best_set = set(current)
        best_size = len(current)
        
    # Random order restarts
    while time.time() - start_time < time_limit * 0.3:
        order = list(range(1, n + 1))
        random.shuffle(order)
        current = greedy_fill(set(), order)
        if len(current) > best_size:
            best_set = set(current)
            best_size = len(current)
        
        # Also try starting from ternary set with random fill order
        if random.random() < 0.5:
            order = list(range(1, n + 1))
            random.shuffle(order)
            current = greedy_fill(set(ternary_set), order)
            if len(current) > best_size:
                best_set = set(current)
                best_size = len(current)

    # Local Search: Randomized Hill Climbing
    current_set = set(best_set)
    
    while time.time() - start_time < time_limit:
        # Try to add elements
        added = False
        candidates = [i for i in range(1, n + 1) if i not in current_set]
        
        if not candidates:
            break
        
        random.shuffle(candidates)
        for x in candidates[:300]:
            if not creates_4ap(current_set, x):
                current_set.add(x)
                added = True
                if len(current_set) > best_size:
                    best_set = set(current_set)
                    best_size = len(current_set)
                break
        
        if not added:
            # Stuck. Perform a kick: remove 1-4 random elements
            k = random.randint(1, 4)
            if len(current_set) > k:
                to_remove = random.sample(list(current_set), k)
                for x in to_remove:
                    current_set.remove(x)
                
                # Greedy fill from current state
                candidates = [i for i in range(1, n + 1) if i not in current_set]
                random.shuffle(candidates)
                for x in candidates:
                    if not creates_4ap(current_set, x):
                        current_set.add(x)
                
                if len(current_set) > best_size:
                    best_set = set(current_set)
                    best_size = len(current_set)

    return sorted(best_set)
