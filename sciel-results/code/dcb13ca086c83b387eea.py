import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    # 1. Define S_high
    S_high = set(range(91, 181))
    
    # 2. Compute A_safe
    # a is safe if there are NO distinct b, c in S_high such that 1/a = 1/b + 1/c
    # Equivalent to: no distinct divisors d1, d2 of a^2 such that a+d1, a+d2 in S_high
    A_safe = []
    for a in range(1, 91):
        a_sq = a * a
        divisors = []
        for i in range(1, int(a_sq**0.5) + 1):
            if a_sq % i == 0:
                divisors.append(i)
                if i != a_sq // i:
                    divisors.append(a_sq // i)
        
        is_bad = False
        # Check pairs of distinct divisors
        # We need b = a + d1, c = a + d2
        # b, c in S_high means 91 <= a+d <= 180
        # d1 != d2
        for i in range(len(divisors)):
            for j in range(i + 1, len(divisors)):
                d1 = divisors[i]
                d2 = divisors[j]
                b = a + d1
                c = a + d2
                if b in S_high and c in S_high:
                    is_bad = True
                    break
            if is_bad:
                break
        if not is_bad:
            A_safe.append(a)
            
    # 3. Greedy selection from A_safe
    # We want to pick a subset A such that:
    # 1. No pair x,y in A has xy/|x-y| in S_high
    # 2. No triple x,y,z in A forms 1/x = 1/y + 1/z
    
    def check_pair_conflict(x, y, S_high):
        if x == y:
            return False
        num = x * y
        den = abs(x - y)
        if num % den != 0:
            return False
        c = num // den
        return c in S_high

    def check_triple_conflict(x, y, z):
        # Check if any permutation forms 1/a = 1/b + 1/c
        # Sort to make it simple: smallest is a
        vals = sorted([x, y, z])
        a, b, c = vals
        # 1/a = 1/b + 1/c  <=> bc = a(b+c)
        return b * c == a * (b + c)

    best_set = set()
    
    # Try multiple random orderings
    for _ in range(1000):
        if time.time() > deadline:
            break
            
        candidates = A_safe[:]
        random.shuffle(candidates)
        
        current_set = []
        
        for a in candidates:
            # Check conflict with all existing elements (pairwise)
            conflict_pair = False
            for b in current_set:
                if check_pair_conflict(a, b, S_high):
                    conflict_pair = True
                    break
            if conflict_pair:
                continue
                
            # Check conflict with all pairs of existing elements (triple)
            conflict_triple = False
            for i in range(len(current_set)):
                for j in range(i + 1, len(current_set)):
                    b = current_set[i]
                    c = current_set[j]
                    if check_triple_conflict(a, b, c):
                        conflict_triple = True
                        break
                if conflict_triple:
                    break
            if conflict_triple:
                continue
            
            current_set.append(a)
            
        if len(current_set) > len(best_set):
            best_set = set(current_set)
            
    result = S_high.union(best_set)
    return list(result)
