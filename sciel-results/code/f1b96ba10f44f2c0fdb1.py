import math
import time

def construct():
    start_time = time.time()
    DEADLINE = start_time + 80  # Safety margin before 90s kill
    
    N = 479001600  # 12!
    
    # Prime factorization of 12!
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    primes = [2, 3, 5, 7, 11]
    max_exponents = [10, 5, 2, 1, 1]
    
    # Generate all divisors of 12!
    divisors = [1]
    for i, p in enumerate(primes):
        new_divs = []
        for d in divisors:
            for e in range(max_exponents[i] + 1):
                new_divs.append(d * (p ** e))
        divisors = new_divs
    
    # Filter divisors > 12 and sort
    valid_divisors = sorted([d for d in divisors if d > 12])
    
    # Map divisor to its prime exponents tuple for fast lookup
    divisor_to_pf = {}
    for d in valid_divisors:
        n = d
        pf = [0, 0, 0, 0, 0]
        for i, p in enumerate(primes):
            while n % p == 0:
                pf[i] += 1
                n //= p
        divisor_to_pf[d] = tuple(pf)
        
    # Binary search helper for valid_divisors
    import bisect
    
    best_max_val = float('inf')
    best_factors = []

    # Target prime exponents
    target_pf = tuple(max_exponents)

    def dfs(last_factor, rem_pf, count_left, current_path, current_max):
        nonlocal best_max_val, best_factors
        
        # Time check
        if time.time() > DEADLINE:
            return

        if count_left == 0:
            if rem_pf == (0, 0, 0, 0, 0):
                if current_max < best_max_val:
                    best_max_val = current_max
                    best_factors = list(current_path)
            return

        # Pruning 1: If current_max is already >= best, no point continuing
        # Note: Since we pick factors in increasing order, current_max is the last factor added.
        # However, we are minimizing the MAX factor. Since factors are added in increasing order,
        # the final max will be the last factor added.
        # So if we are at count_left > 0, we will add more factors which are > last_factor.
        # Thus the final max will be > current_max.
        # We can prune if the smallest possible next factor is already >= best_max_val.
        # But more tightly, we can prune if the *smallest possible final max* is >= best_max_val.
        
        # Find the start index for the next factor
        # We need distinct integers > last_factor
        # valid_divisors is sorted.
        # Find first divisor > last_factor
        idx_start = bisect.bisect_right(valid_divisors, last_factor)
        
        if idx_start >= len(valid_divisors):
            return

        # Pruning: The next factor must be at least valid_divisors[idx_start].
        # If we have to pick count_left factors, the largest one will be at least the (idx_start + count_left - 1)-th divisor?
        # Not necessarily, because we can skip divisors. But the smallest possible largest factor
        # if we pick count_left distinct divisors starting from idx_start is valid_divisors[idx_start + count_left - 1].
        # If that is >= best_max_val, we can prune.
        if idx_start + count_left - 1 < len(valid_divisors):
            min_possible_max = valid_divisors[idx_start + count_left - 1]
            if min_possible_max >= best_max_val:
                return
        else:
            # Not enough divisors left to pick count_left distinct ones
            return

        # Iterate through possible next factors
        # We can limit the range of d.
        # d must be such that d * (smallest possible product of remaining count_left-1 factors) <= N / current_prod?
        # We don't track current_prod explicitly, but rem_pf encodes it.
        # Actually, current_prod = N / product of factors in path.
        # Let's just iterate d from valid_divisors[idx_start] upwards.
        
        # Upper bound for d:
        # If we pick d, the remaining product must be formed by count_left-1 distinct factors > d.
        # The smallest such product is valid_divisors[idx_next] * ... * valid_divisors[idx_next + count_left - 2].
        # Where idx_next is the index of the first divisor > d.
        
        for i in range(idx_start, len(valid_divisors)):
            d = valid_divisors[i]
            
            # Pruning: d is the current max candidate. If d >= best_max_val, break (since divisors are sorted)
            if d >= best_max_val:
                break
                
            # Check if d's prime factors are within rem_pf
            pf_d = divisor_to_pf[d]
            valid = True
            for j in range(5):
                if pf_d[j] > rem_pf[j]:
                    valid = False
                    break
            if not valid:
                continue
            
            new_rem_pf = tuple(rem_pf[j] - pf_d[j] for j in range(5))
            
            # If this is the last factor, check if new_rem_pf is all zeros
            if count_left == 1:
                if new_rem_pf == (0, 0, 0, 0, 0):
                    # Solution found
                    if d < best_max_val:
                        best_max_val = d
                        best_factors = list(current_path) + [d]
                # Since d is increasing, if we didn't find a solution with d,
                # and d >= best_max_val we would have broken.
                # But we can break early if d is large enough?
                # Actually, if count_left==1, we just need new_rem_pf to be 0.
                # There is only one such d (which is N / current_prod).
                # So we could break after checking the first valid d?
                # No, because d must be a divisor.
                # But wait, if new_rem_pf is not 0, no other d will work for count_left=1
                # because the product must be exactly 1.
                # So we can break here?
                # Yes, if count_left == 1, there is at most one valid d (the one that consumes all remaining primes).
                # So we can break after the first iteration of the loop for count_left==1?
                # Not quite, because we might skip some d's due to prime factor constraints.
                # But once we pass the unique d that works, no other d will work.
                # The unique d is determined by new_rem_pf being 0.
                # So we can just let the loop run, but it will be fast.
                pass
            else:
                # Recurse
                current_path.append(d)
                dfs(d, new_rem_pf, count_left - 1, current_path, d)
                current_path.pop()

    # Try all k from 12 down to 1?
    # Usually more factors allow smaller max.
    # Let's try k from 12 down to 1.
    # But we can prune k if the minimum possible max for k factors is >= best_max_val.
    # Min possible max for k factors is valid_divisors[k-1] (the k-th smallest divisor > 12).
    # Wait, distinct factors > 12.
    # The k smallest distinct factors > 12 are valid_divisors[0] ... valid_divisors[k-1].
    # Their max is valid_divisors[k-1].
    # If valid_divisors[k-1] >= best_max_val, then no solution with k factors can beat best_max_val.
    
    # Start with k=12, go down to 1.
    for k in range(12, 0, -1):
        if time.time() > DEADLINE:
            break
            
        # Pruning for k
        if k <= len(valid_divisors):
            min_possible_max_for_k = valid_divisors[k-1]
            if min_possible_max_for_k >= best_max_val:
                # Since we are iterating k downwards, and min_possible_max_for_k increases as k decreases?
                # No, valid_divisors is sorted ascending.
                # valid_divisors[11] > valid_divisors[10] > ...
                # So as k decreases, min_possible_max_for_k decreases.
                # So we cannot break the k-loop just because current k is pruned.
                # But we can skip this k.
                continue
        else:
            continue
            
        dfs(12, target_pf, k, [], 12)
        
        # Optional: Early termination if we found a very good solution
        # e.g., if best_max_val is close to theoretical minimum
        # Theoretical minimum max is at least (12!)^(1/12) ~ 10.8, so >12.
        # If best_max_val == 13, we can stop? No, distinct integers > 12.
        # Smallest distinct are 13, 14, ...
        # Product of 13*14*...*24 is much larger than 12! ?
        # 13*14*...*24 = 24! / 12!
        # 12! = 4.8e8
        # 24!/12! is huge.
        # So we can't just have 13, 14, ...
        
    if best_factors:
        return best_factors
    else:
        return [N]

# Quick test
if __name__ == "__main__":
    res = construct()
    print(res)
    prod = 1
    for x in res:
        prod *= x
    print(prod)
    print(max(res))
    print(len(res))
