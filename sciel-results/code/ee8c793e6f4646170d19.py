import time
import math
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s margin before 90s kill
    
    def get_all_sums(S):
        """Get all non-empty subset sums of S."""
        sums = set()
        n = len(S)
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += S[i]
            sums.add(s)
        return sums
    
    def check_divisibility_free(sums):
        """Check that no two distinct sums in the set divide each other."""
        sorted_sums = sorted(sums)
        for i in range(len(sorted_sums)):
            for j in range(i + 1, len(sorted_sums)):
                if sorted_sums[j] % sorted_sums[i] == 0:
                    return False
        return True
    
    def can_add(S, x, current_sums):
        """Check if we can add x to S without violating divisibility constraint."""
        # New sums are {x} ∪ {s + x for s in current_sums}
        new_sums = set()
        new_sums.add(x)
        for s in current_sums:
            new_sums.add(s + x)
        
        # Check all pairs in new_sums ∪ current_sums for divisibility
        all_sums = current_sums | new_sums
        sorted_sums = sorted(all_sums)
        for i in range(len(sorted_sums)):
            ai = sorted_sums[i]
            for j in range(i + 1, len(sorted_sums)):
                aj = sorted_sums[j]
                if aj % ai == 0:
                    return False
        return True
    
    def try_greedy(candidates, time_limit):
        """Try to build a set greedily from candidates."""
        S = []
        current_sums = set()
        
        for x in candidates:
            if time.time() > time_limit:
                break
            if x in S:
                continue
            if can_add(S, x, current_sums):
                S.append(x)
                # Update current_sums
                old_sums = set(current_sums)
                current_sums.add(x)
                for s in old_sums:
                    current_sums.add(s + x)
        
        return S
    
    # Start with a known valid small set
    best_set = [31, 32, 34]
    best_size = len(best_set)
    
    # Try greedy from different starting points and candidate orders
    candidates = list(range(1, 257))
    
    # Try several strategies
    strategies = []
    
    # Strategy 1: Greedy from high numbers
    strategies.append(sorted(candidates, reverse=True))
    
    # Strategy 2: Greedy from medium-high numbers
    strategies.append(sorted([x for x in candidates if x >= 50], reverse=True))
    
    # Strategy 3: Greedy from numbers that are primes or near-primes
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    primes = [x for x in candidates if is_prime(x)]
    strategies.append(primes)
    
    # Strategy 4: Numbers that are products of large primes
    def num_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    # Try numbers with few factors
    few_factors = sorted([x for x in candidates if num_factors(x) <= 2], reverse=True)
    strategies.append(few_factors)
    
    # Strategy 5: Random orders
    import random
    for _ in range(5):
        random.shuffle(candidates)
        strategies.append(list(candidates))
    
    for i, strat in enumerate(strategies):
        if time.time() > deadline - 5.0:
            break
        time_limit = deadline - 5.0
        S = try_greedy(strat, time_limit)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Try local search: start from best_set, try to swap or add
    if time.time() < deadline - 10.0:
        # Try to extend best_set
        current_sums = get_all_sums(best_set)
        for x in range(1, 257):
            if time.time() > deadline - 5.0:
                break
            if x in best_set:
                continue
            if can_add(best_set, x, current_sums):
                best_set.append(x)
                best_size = len(best_set)
                # Update current_sums
                old_sums = set(current_sums)
                current_sums.add(x)
                for s in old_sums:
                    current_sums.add(s + x)
    
    # Verify the result
    if best_size >= 2:
        all_sums = get_all_sums(best_set)
        if not check_divisibility_free(all_sums):
            # Fallback to known valid set
            best_set = [31, 32, 34]
            best_size = 3
    
    return list(best_set)
