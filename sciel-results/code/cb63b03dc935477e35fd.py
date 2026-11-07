import math
import time

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    
    # 1. Sieve of Eratosthenes to find primes
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i*i, N+1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N+1) if is_prime[i]]
    
    # 2. Identify forced numbers
    # Primes > 128 cannot be multiplied by any integer > 1 and stay <= 256
    forced = [1]
    for p in primes:
        if p > 128:
            forced.append(p)
            
    # Primes <= 128 are "flexible"
    flexible_primes = [p for p in primes if p <= 128]
    
    # 3. Generate candidates for the flexible part
    # Candidates are numbers <= 256 whose prime factors are all in flexible_primes
    # Actually, any number <= 256 is a candidate, but if it has a prime factor > 128,
    # it's already handled or impossible?
    # If a number has a prime factor > 128, that factor must be the number itself 
    # (since 2*129 > 256) or a multiple? 
    # 127 * 2 = 254. 127 is flexible.
    # 131 * 2 = 262 > 256.
    # So any number with a prime factor > 128 is just that prime itself (which is forced).
    # So we only need to consider numbers composed of primes <= 128.
    
    candidates = []
    for i in range(2, N + 1):
        # Check if all prime factors of i are <= 128
        temp = i
        valid = True
        for p in flexible_primes:
            while temp % p == 0:
                temp //= p
            if temp == 1:
                break
        if temp != 1:
            valid = False
        if valid:
            candidates.append(i)
            
    # Sort candidates by value descending
    candidates.sort(reverse=True)
    
    # Precompute prime factors for candidates for fast conflict checking
    # Map candidate -> set of prime factors
    cand_factors = {}
    for c in candidates:
        temp = c
        factors = set()
        for p in flexible_primes:
            while temp % p == 0:
                factors.add(p)
                temp //= p
            if temp == 1:
                break
        cand_factors[c] = factors
        
    best_sum = sum(forced)
    best_set = set(forced)
    
    # DFS
    # State: index in candidates, current_sum, used_primes (set)
    # We want to maximize sum.
    
    # Optimization: Prune if current_sum + sum of remaining candidates <= best_sum
    # Precompute suffix sums of candidates
    suffix_sums = [0] * (len(candidates) + 1)
    for i in range(len(candidates) - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i+1] + candidates[i]
        
    def dfs(idx, current_sum, used_primes):
        nonlocal best_sum, best_set
        if time.time() > deadline:
            return
            
        # Pruning
        if current_sum + suffix_sums[idx] <= best_sum:
            return
            
        if idx == len(candidates):
            if current_sum > best_sum:
                best_sum = current_sum
                # Reconstruct set? We just need the sum for the score, 
                # but the task asks for a list of ints.
                # We need to return the list.
                # We'll track the set in the recursion or reconstruct.
                # For simplicity in this draft, I'll just track sum.
                # But I need to return the list.
                pass
            return
        
        c = candidates[idx]
        factors = cand_factors[c]
        
        # Option 1: Skip c
        dfs(idx + 1, current_sum, used_primes)
        
        # Option 2: Take c
        # Check conflict
        if not used_primes.intersection(factors):
            new_used = used_primes.union(factors)
            dfs(idx + 1, current_sum + c, new_used)

    # To return the actual list, we need to track the choices.
    # Let's modify DFS to track the list.
    
    best_result = list(forced)
    
    def dfs_track(idx, current_sum, used_primes, current_list):
        nonlocal best_sum, best_result
        if time.time() > deadline:
            return
            
        if current_sum + suffix_sums[idx] <= best_sum:
            return
            
        if idx == len(candidates):
            if current_sum > best_sum:
                best_sum = current_sum
                best_result = current_list.copy()
            return
        
        c = candidates[idx]
        factors = cand_factors[c]
        
        # Option 1: Skip c
        dfs_track(idx + 1, current_sum, used_primes, current_list)
        
        # Option 2: Take c
        if not used_primes.intersection(factors):
            new_used = used_primes.union(factors)
            current_list.append(c)
            dfs_track(idx + 1, current_sum + c, new_used, current_list)
            current_list.pop()

    dfs_track(0, sum(forced), set(), list(forced))
    
    return best_result

# EVOLVE-BLOCK-END
