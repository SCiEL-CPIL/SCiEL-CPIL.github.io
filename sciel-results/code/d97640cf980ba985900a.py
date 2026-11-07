import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55
    N = 256
    all_nums = list(range(2, N + 1))
    
    def check_triple_valid(S):
        """O(n^3) full validity check"""
        n = len(S)
        for i in range(n):
            a = S[i]
            for j in range(i + 1, n):
                b = S[j]
                for k in range(j + 1, n):
                    c = S[k]
                    if (b * c) % a == 0:
                        return False
        return True
    
    def can_add(S, x):
        """Check if x can be added to valid set S"""
        for i in range(len(S)):
            a = S[i]
            for j in range(i + 1, len(S)):
                b = S[j]
                if (x * b) % a == 0:
                    return False
                if (a * b) % x == 0:
                    return False
                if (a * x) % b == 0:
                    return False
        return True
    
    best_S = []
    
    # Multiple greedy strategies
    strategies = []
    
    # Strategy 1: descending
    strategies.append(sorted(all_nums, reverse=True))
    
    # Strategy 2: ascending  
    strategies.append(sorted(all_nums))
    
    # Strategy 3: by number of divisors (fewer first)
    def num_divisors(x):
        count = 0
        for d in range(2, int(x**0.5) + 1):
            if x % d == 0:
                count += 2
                if d * d == x:
                    count -= 1
        return count if x > 1 else 0
    
    strategies.append(sorted(all_nums, key=num_divisors))
    
    # Strategy 4-7: random
    for _ in range(4):
        s = all_nums[:]
        random.shuffle(s)
        strategies.append(s)
    
    for order in strategies:
        if time.time() > deadline - 10:
            break
        S = []
        for x in order:
            if can_add(S, x):
                S.append(x)
        if len(S) > len(best_S):
            best_S = S[:]
    
    # Local search: try to swap one element out and one in
    if time.time() < deadline - 5:
        for iteration in range(50):
            if time.time() > deadline - 5:
                break
            improved = False
            S_set = set(best_S)
            for i in range(len(best_S)):
                if time.time() > deadline - 5:
                    break
                old = best_S[i]
                for x in all_nums:
                    if x in S_set:
                        continue
                    # Try: remove old, add x
                    new_S = best_S[:i] + [x] + best_S[i+1:]
                    if check_triple_valid(new_S):
                        best_S = new_S
                        S_set = set(best_S)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
    
    best_S.sort()
    return list(best_S)
