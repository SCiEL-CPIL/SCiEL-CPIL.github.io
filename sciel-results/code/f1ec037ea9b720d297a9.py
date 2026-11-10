import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    N = 479001600  # 12!
    
    # Prime factorization of 12!:
    # 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    primes = [2, 3, 5, 7, 11]
    target_exp = [10, 5, 2, 1, 1]
    
    # Generate all divisors of N that are > 12 and < N
    # For each, store (value, exponent_tuple)
    candidates = []
    
    # Enumerate all exponent combinations for divisors of N
    def gen_divisors(idx, current_exp, current_val):
        if idx == len(primes):
            if current_val > 12 and current_val < N:
                candidates.append((current_val, tuple(current_exp)))
            return
        for e in range(target_exp[idx] + 1):
            current_exp.append(e)
            gen_divisors(idx + 1, current_exp, current_val * (primes[idx] ** e))
            current_exp.pop()
    
    gen_divisors(0, [], 1)
    
    # Sort by value
    candidates.sort()
    n_cand = len(candidates)
    
    # We'll search for the minimum M such that there exists a subset of distinct candidates
    # with values <= M whose product is N and size <= 12
    
    # Use a different strategy: since the number of candidates is manageable,
    # we can use a DFS that builds up the factor list and checks if the remaining
    # product can be achieved with the remaining candidates
    
    # Precompute suffix products of exponent vectors for pruning
    # suffix_exp[i] = sum of exponent vectors of candidates[i:]
    suffix_exp = [[0]*5 for _ in range(n_cand + 1)]
    for i in range(n_cand - 1, -1, -1):
        for j in range(5):
            suffix_exp[i][j] = suffix_exp[i+1][j] + candidates[i][1][j]
    
    def dfs(idx, rem_exp, count, current_factors, max_val):
        if time.time() > deadline:
            return None
        
        if all(e == 0 for e in rem_exp):
            if 1 <= count <= 12:
                return list(current_factors)
            return None
        
        if idx >= n_cand:
            return None
        
        # Pruning: can the remaining candidates complete rem_exp?
        for j in range(5):
            if suffix_exp[idx][j] < rem_exp[j]:
                return None
        
        # Try including candidate idx
        val, exp = candidates[idx]
        if val <= max_val:
            new_rem = [rem_exp[j] - exp[j] for j in range(5)]
            if all(e >= 0 for e in new_rem):
                current_factors.append(val)
                result = dfs(idx + 1, new_rem, count + 1, current_factors, max_val)
                if result is not None:
                    return result
                current_factors.pop()
        
        # Try skipping candidate idx
        result = dfs(idx + 1, rem_exp, count, current_factors, max_val)
        if result is not None:
            return result
        
        return None
    
    # Try increasing max_val from the smallest candidate value upward
    unique_vals = sorted(set(c[0] for c in candidates))
    if N not in unique_vals:
        unique_vals.append(N)
    
    for M in unique_vals:
        if time.time() > deadline:
            break
        result = dfs(0, target_exp, 0, [], M)
        if result is not None:
            return result
    
    return [N]
