import math
import time
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 90s limit
    
    target = math.factorial(12)  # 479001600
    
    # Prime factorization of 12!: 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    primes = [2, 3, 5, 7, 11]
    exponents = [10, 5, 2, 1, 1]
    
    # Generate all divisors > 12 using prime factorization
    divisors_set = set()
    current_divs = [1]
    for i, (p, exp) in enumerate(zip(primes, exponents)):
        new_divs = []
        for d in current_divs:
            val = d
            for e in range(exp + 1):
                if val > 12:
                    divisors_set.add(val)
                new_divs.append(val)
                if e < exp:
                    val *= p
        current_divs = new_divs
    for d in current_divs:
        if d > 12:
            divisors_set.add(d)
    
    candidates = sorted(divisors_set)
    
    if not candidates:
        return [target]
    
    # Map each candidate to its exponent vector for fast subtraction
    def factorize(n):
        vec = [0] * len(primes)
        for i, p in enumerate(primes):
            while n % p == 0:
                vec[i] += 1
                n //= p
        return tuple(vec)
    
    cand_exp = {c: factorize(c) for c in candidates}
    target_exp = tuple(exponents)
    
    def rem_exp_to_val(rem_exp):
        val = 1
        for i, p in enumerate(primes):
            val *= p ** rem_exp[i]
        return val
    
    def can_decompose(M_idx):
        M_val = candidates[M_idx]
        fail_cache = set()
        
        def dfs(rem_exp, start_idx, depth):
            if time.time() > deadline:
                return None
            if depth > 12:
                return None
            if all(e == 0 for e in rem_exp):
                return []
            
            rem_val = rem_exp_to_val(rem_exp)
            
            # Quick check: if rem_val itself is a valid candidate at or after start_idx
            if rem_val > 12 and rem_val in cand_exp:
                idx = bisect.bisect_left(candidates, rem_val)
                if idx < len(candidates) and candidates[idx] == rem_val and idx >= start_idx and idx <= M_idx:
                    return [rem_val]
            
            if (rem_exp, start_idx) in fail_cache:
                return None
            
            # Pruning: if rem_val < smallest candidate at start_idx, fail
            if start_idx <= M_idx:
                min_cand = candidates[start_idx]
                if rem_val < min_cand:
                    fail_cache.add((rem_exp, start_idx))
                    return None
            
            for i in range(start_idx, M_idx + 1):
                f = candidates[i]
                if f > rem_val:
                    break
                
                f_exp = cand_exp[f]
                valid = True
                for k in range(len(primes)):
                    if f_exp[k] > rem_exp[k]:
                        valid = False
                        break
                if not valid:
                    continue
                
                new_rem_exp = tuple(rem_exp[k] - f_exp[k] for k in range(len(primes)))
                res = dfs(new_rem_exp, i + 1, depth + 1)
                if res is not None:
                    return [f] + res
            
            fail_cache.add((rem_exp, start_idx))
            return None
        
        return dfs(target_exp, 0, 0)
    
    lo = 0
    hi = len(candidates) - 1
    best_sol = None
    best_idx = len(candidates) - 1
    
    while lo <= hi:
        mid = (lo + hi) // 2
        sol = can_decompose(mid)
        if sol is not None:
            # Verify
            prod = 1
            for f in sol:
                prod *= f
            if prod == target and len(set(sol)) == len(sol) and all(f > 12 for f in sol) and 1 <= len(sol) <= 12:
                best_sol = sol
                best_idx = mid
                hi = mid - 1
            else:
                lo = mid + 1
        else:
            lo = mid + 1
            
    if best_sol is not None:
        return best_sol
    return [target]
