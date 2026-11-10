import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85

    MAX_N = 256
    
    # 1. Sieve Primes
    is_prime = [True] * (MAX_N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(MAX_N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, MAX_N + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    
    # 2. Split Primes
    small_primes = [p for p in primes if p <= 13]
    large_primes = [p for p in primes if p > 13]
    
    # Large primes are always selected as themselves
    selected_large = large_primes
    sum_large = sum(selected_large)
    
    # 3. Generate Candidates for Small Primes
    # Candidates are numbers <= 256 with prime factors only in small_primes
    # small_primes: [2, 3, 5, 7, 11, 13]
    prime_to_idx = {p: i for i, p in enumerate(small_primes)}
    
    candidates = []
    for n in range(2, MAX_N + 1):
        # Check if all prime factors are in small_primes
        temp = n
        valid = True
        mask = 0
        for p in small_primes:
            if temp % p == 0:
                mask |= (1 << prime_to_idx[p])
                while temp % p == 0:
                    temp //= p
            if temp == 1:
                break
        if temp != 1:
            # Has a prime factor > 13
            continue
        # n is a candidate
        candidates.append((n, mask))
        
    # 4. Solve Max Weight Independent Set for Candidates
    # We want to select a subset of candidates such that their masks are disjoint
    # and sum of values is maximized.
    
    # Sort by value descending to help greedy/pruning, though exact search is fine
    candidates.sort(key=lambda x: -x[0])
    
    best_sum = 0
    best_set = []
    
    # Recursive search
    def search(start_idx, current_mask, current_sum, current_set):
        nonlocal best_sum, best_set
        
        # Pruning: if current_sum + sum of all remaining < best_sum, return
        # Simple bound: current_sum + sum(val for val, m in candidates[start_idx:] if (m & current_mask) == 0)
        # But calculating that bound every step is expensive.
        # Given small size, just go deep.
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_set = list(current_set)
            
        if start_idx == len(candidates):
            return
            
        # Option 1: Skip current candidate
        search(start_idx + 1, current_mask, current_sum, current_set)
        
        # Option 2: Include current candidate if compatible
        val, mask = candidates[start_idx]
        if (mask & current_mask) == 0:
            current_set.append(val)
            search(start_idx + 1, current_mask | mask, current_sum + val, current_set)
            current_set.pop()
            
    search(0, 0, 0, [])
    
    # 5. Construct Final List
    # 1 is always included
    final_set = [1] + selected_large + best_set
    
    # Verify uniqueness and range
    # The problem requires distinct integers.
    # Large primes are distinct.
    # Small candidates are distinct.
    # Large primes > 13, Small candidates <= 256 but composed of small primes.
    # Can a large prime be in best_set? No, best_set comes from candidates which have no large prime factors.
    # Can a small candidate be in selected_large? No, selected_large are primes > 13.
    # So they are disjoint.
    # 1 is distinct from all.
    
    # Check time
    if time.time() > deadline:
         # Fallback to just primes if search took too long (unlikely)
         return [1] + primes
         
    return [int(x) for x in final_set]
