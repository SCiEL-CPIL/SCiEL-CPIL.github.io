import math
import time

def construct():
    start_time = time.time()
    LIMIT = 10**18
    DEADLINE = 110.0  # Safety margin
    
    # First 20 primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    
    def get_score_from_exponents(exponents):
        """
        exponents: list of (prime, exp)
        Returns score.
        """
        n = 1
        for p, e in exponents:
            n *= (p ** e)
        if n > LIMIT or n < 2:
            return 0
            
        # Generate divisors
        divisors = [1]
        for p, e in exponents:
            new_divs = []
            for d in divisors:
                val = d
                for _ in range(e + 1):
                    new_divs.append(val)
                    val *= p
            divisors = new_divs
        divisors.sort()
        
        score = 0
        for i in range(len(divisors) - 1):
            if math.gcd(divisors[i], divisors[i+1]) == 1:
                score += 1
        return score

    best_score = 0
    best_n = 2

    def update_best(score, n):
        nonlocal best_score, best_n
        if score > best_score:
            best_score = score
            best_n = n

    # Helper to factorize a number into the given prime list
    def factorize_small(m, primes_list):
        factors = {}
        temp_m = m
        for p in primes_list:
            if p * p > temp_m:
                break
            if temp_m % p == 0:
                count = 0
                while temp_m % p == 0:
                    temp_m //= p
                    count += 1
                factors[p] = count
        if temp_m > 1:
            # If temp_m is a prime in our list, add it
            if temp_m in primes_list:
                factors[temp_m] = factors.get(temp_m, 0) + 1
            else:
                # It's a prime outside our base set, which means n would have a new prime factor
                # We handle this by including it in the exponents list
                factors[temp_m] = 1
        return factors

    # Search strategy:
    # Iterate over number of distinct base primes k
    # For each k, iterate over multipliers m
    
    # We focus on k=15, 14, 13, 12, 11
    # For larger k, the multiplier range is small.
    
    # Precompute primorials
    primorials = [1]
    for p in primes:
        primorials.append(primorials[-1] * p)
        
    # k=15 corresponds to primes[:15], product primorials[15]
    # We test k from 15 down to 10
    
    for k in range(15, 9, -1):
        if time.time() > DEADLINE:
            break
            
        base_n = primorials[k]
        if base_n > LIMIT:
            continue
            
        max_m = LIMIT // base_n
        
        # If max_m is very large, we might not iterate all. 
        # But for k>=10, max_m is manageable.
        # k=15: max_m ~ 1
        # k=14: max_m ~ 76
        # k=13: max_m ~ 3333
        # k=12: max_m ~ 150,000
        # k=11: max_m ~ 8,000,000 (Too large to iterate fully, need heuristic or sampling)
        
        primes_subset = primes[:k]
        
        if max_m <= 50000:
            # Iterate all m
            for m in range(1, max_m + 1):
                if time.time() > DEADLINE:
                    break
                
                # Factorize m
                m_factors = factorize_small(m, primes_subset)
                
                # Construct full exponents
                full_exp = []
                # Add base exponents (all 1)
                for p in primes_subset:
                    full_exp.append((p, 1))
                
                # Add/merge m_factors
                # We need to update the exponents in full_exp
                exp_dict = {p: e for p, e in full_exp}
                for p, e in m_factors.items():
                    if p in exp_dict:
                        exp_dict[p] += e
                    else:
                        # If p is not in base, it's a new prime
                        exp_dict[p] = e
                
                # Convert back to list for score function
                final_exp = [(p, e) for p, e in exp_dict.items() if e > 0]
                
                s = get_score_from_exponents(final_exp)
                n_val = base_n * m
                update_best(s, n_val)
        
        else:
            # For large max_m, we sample or use greedy?
            # Given k>=10, the structure is dense.
            # We test m=1 (the primorial itself)
            s = get_score_from_exponents([(p, 1) for p in primes_subset])
            update_best(s, base_n)
            
            # Also test some smooth multipliers
            # e.g. powers of 2, 3
            current_m = 1
            while current_m <= max_m:
                m_factors = factorize_small(current_m, primes_subset)
                exp_dict = {p: 1 for p in primes_subset}
                for p, e in m_factors.items():
                    if p in exp_dict:
                        exp_dict[p] += e
                    else:
                        exp_dict[p] = e
                final_exp = [(p, e) for p, e in exp_dict.items() if e > 0]
                s = get_score_from_exponents(final_exp)
                n_val = base_n * current_m
                update_best(s, n_val)
                
                current_m *= 2
                if current_m > max_m:
                    break
            
            current_m = 1
            while current_m <= max_m:
                m_factors = factorize_small(current_m, primes_subset)
                exp_dict = {p: 1 for p in primes_subset}
                for p, e in m_factors.items():
                    if p in exp_dict:
                        exp_dict[p] += e
                    else:
                        exp_dict[p] = e
                final_exp = [(p, e) for p, e in exp_dict.items() if e > 0]
                s = get_score_from_exponents(final_exp)
                n_val = base_n * current_m
                update_best(s, n_val)
                
                current_m *= 3
                if current_m > max_m:
                    break

    return best_n

# EVOLVE-BLOCK-START
# The construct function is defined above. 
# We just need to expose it.
# EVOLVE-BLOCK-END
