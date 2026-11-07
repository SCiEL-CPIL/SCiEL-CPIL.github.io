import time
from math import isqrt, gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85

    N = 256

    # 1. Sieve primes
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # 2. Calculate M_p
    M = {}
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        M[p] = power

    # 3. Generate candidates
    # A candidate is a number x > 1 that is NOT a prime power.
    # We check if using x instead of individual M_p's yields a gain.
    candidates = [] # List of (gain, x, set_of_primes)
    
    for x in range(2, N + 1):
        # Find prime factors of x
        pf = set()
        temp = x
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                pf.add(p)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            pf.add(temp)
            
        # If x is a prime power, pf has size 1. 
        # These are already covered by M[p] which is >= x. 
        # Actually M[p] is the max power, so M[p] >= x. 
        # Gain = x - M[p] <= 0. So we can skip prime powers.
        if len(pf) == 1:
            continue
            
        # Calculate baseline cost for these primes
        baseline_cost = sum(M[p] for p in pf)
        gain = x - baseline_cost
        
        if gain > 0:
            candidates.append((gain, x, pf))

    # Sort by gain descending
    candidates.sort(key=lambda item: item[0], reverse=True)
    
    # 4. Backtracking to find optimal set of non-conflicting candidates
    num_candidates = len(candidates)
    best_gain = 0
    best_selected = [] # List of indices into candidates
    
    # Precompute suffix sums of gains for pruning
    suffix_gains = [0] * (num_candidates + 1)
    for i in range(num_candidates - 1, -1, -1):
        suffix_gains[i] = suffix_gains[i + 1] + candidates[i][0]

    def backtrack(idx, used_primes, current_gain, selected):
        nonlocal best_gain, best_selected
        
        # Pruning
        if current_gain + suffix_gains[idx] <= best_gain:
            return
            
        if idx == num_candidates:
            if current_gain > best_gain:
                best_gain = current_gain
                best_selected = list(selected)
            return

        # Option 1: Skip candidate idx
        backtrack(idx + 1, used_primes, current_gain, selected)

        # Option 2: Take candidate idx
        gain, x, pf = candidates[idx]
        # Check conflict
        if used_primes.isdisjoint(pf):
            new_used = used_primes | pf
            selected.append(idx)
            backtrack(idx + 1, new_used, current_gain + gain, selected)
            selected.pop()

    backtrack(0, set(), 0, [])

    # 5. Construct Result
    selected_primes = set()
    result = [1]
    
    for idx in best_selected:
        gain, x, pf = candidates[idx]
        result.append(x)
        selected_primes.update(pf)

    for p in primes:
        if p not in selected_primes:
            result.append(M[p])

    # 6. Verification
    # Check distinct
    if len(set(result)) != len(result):
        # Fallback to baseline
        result = [1] + [M[p] for p in primes]
        
    # Check range
    for x in result:
        if x < 1 or x > N:
            result = [1] + [M[p] for p in primes]
            break
            
    # Check coprime
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        result = [1] + [M[p] for p in primes]

    return [int(x) for x in result]
