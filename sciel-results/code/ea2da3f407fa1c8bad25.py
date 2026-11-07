import math
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s buffer
    
    # Strategy: Generate candidate M values from small primes.
    # Check divisors k of M to see if k * sigma(k) == M.
    # k must divide M.
    
    # Primes to use
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    
    # We will generate M by taking products of these primes.
    # To find M with many solutions, we likely need M to be "smooth" or have many divisors.
    # Let's iterate over exponents for the first few primes.
    
    # To keep the number of candidates manageable, we limit exponents.
    # M <= 10^18.
    
    # We'll use a recursive or iterative approach to generate M.
    # Since we want to maximize the number of k's, and k divides M,
    # having many divisors in M increases the chance of finding valid k's.
    
    # Let's try a structured search over exponents of the first 5-6 primes.
    # If we use more primes, the number of combinations explodes.
    # Let's start with primes [2, 3, 5, 7, 11, 13].
    
    active_primes = primes[:6]
    
    # Max exponents based on 2^a * 3^b * ... <= 10^18
    # 2^60 is ~ 1e18. So a can be up to 60.
    # But we don't need to go that high for "many divisors" in a small search space.
    # Let's try exponents up to 10 for 2, 8 for 3, 6 for 5, 5 for 7, 4 for 11, 4 for 13.
    
    max_exps = [12, 10, 8, 6, 5, 4]
    
    # Precompute powers
    powers = []
    for p, max_e in zip(active_primes, max_exps):
        pows = [p**i for i in range(max_e + 1)]
        powers.append(pows)
        
    # Generate candidates
    # This might be too many: 13 * 11 * 9 * 7 * 6 * 5 = ~260k candidates.
    # For each candidate, we need to factorize (we already have it), generate divisors, and check.
    # Generating divisors for M with ~100 divisors is fast.
    # Checking sigma(k) for 100 divisors is fast.
    # 260k * (divisor_gen + sigma_check) might be slow in Python if not optimized.
    
    # Optimization:
    # Instead of checking ALL candidates, we can filter.
    # Or, we can just iterate and break early if time is up.
    
    best_M = 333312
    best_ks = [336, 372, 434]
    best_count = 3
    
    # Helper to calculate sigma from prime factorization
    def sigma_from_factors(factors):
        # factors: dict {p: exp}
        res = 1
        for p, e in factors.items():
            if e == 0:
                continue
            # Sum of geometric series: 1 + p + ... + p^e = (p^(e+1)-1)/(p-1)
            # Use integer arithmetic
            term = 1
            pe = 1
            for _ in range(e):
                pe *= p
                term += pe
            res *= term
        return res

    # Helper to get divisors from factors
    def get_divisors(factors):
        # factors: dict {p: exp}
        divs = [1]
        for p, e in factors.items():
            if e == 0:
                continue
            new_divs = []
            pow_p = 1
            for _ in range(e + 1):
                for d in divs:
                    new_divs.append(d * pow_p)
                pow_p *= p
            divs = new_divs
        return divs

    # Iterate through exponent combinations
    # Use product for cartesian product of exponents
    # To avoid generating all at once, we can use recursion or just product and break.
    
    # 260k iterations. In Python, a simple loop with some math might take 10-20s.
    # We need to be careful.
    
    # Let's optimize the inner loop.
    # We can pre-calculate the "base" factors for the first few primes and vary the last ones.
    
    # Actually, let's just run the product and check time.
    
    # To speed up sigma calculation for divisors, we can pre-compute sigma for the divisors
    # using the factorization of M.
    
    # Let's try a smaller set of primes first to ensure it runs, then expand if time permits.
    # Primes: [2, 3, 5, 7, 11]
    # Max exps: [15, 10, 8, 6, 5]
    
    # Revised plan:
    # 1. Generate M from primes [2,3,5,7,11,13] with limited exponents.
    # 2. For each M, get divisors.
    # 3. For each divisor k, check k*sigma(k) == M.
    
    # We will track the best found.
    
    # To make it faster, we can skip M if it's small (we already know small ones).
    # Let's start M from a certain size? No, let's just run.
    
    # Optimization: 
    # If M is a multiple of a known M with 3 solutions, it might have more?
    # Not necessarily.
    
    # Let's run the search.
    
    # To handle the time limit strictly:
    
    # We'll iterate through the cartesian product of exponents.
    # We'll break if time > deadline - 2.0
    
    # To reduce the number of candidates, we can restrict the search space.
    # Let's try exponents up to:
    # 2: 0-15
    # 3: 0-10
    # 5: 0-8
    # 7: 0-6
    # 11: 0-5
    # 13: 0-4
    
    # Total combos: 16 * 11 * 9 * 7 * 6 * 5 = 253,440.
    # This is feasible if the inner loop is fast.
    
    # Inner loop:
    # 1. Calculate M.
    # 2. Get divisors.
    # 3. For each divisor, check condition.
    
    # Getting divisors is the expensive part if M has many.
    # But average number of divisors for random M is small (~100).
    
    # Let's implement.
    
    p_list = [2, 3, 5, 7, 11, 13]
    e_limits = [15, 10, 8, 6, 5, 4]
    
    # Pre-compute powers for faster M calculation?
    # Or just use **.
    
    # We'll use a generator to yield exponent tuples
    # And process them one by one.
    
    # To speed up sigma(k) for divisors k of M:
    # Since we know the factorization of M, we can derive the factorization of k.
    # If k = prod p_i^a_i, where 0 <= a_i <= e_i (exponents in M).
    # sigma(k) = prod (p_i^(a_i+1)-1)/(p_i-1).
    
    # We can pre-compute the "sigma parts" for each prime power in M.
    # sigma_parts[i][a] = (p_i^(a+1)-1)/(p_i-1) for a in 0..e_i.
    
    # Then for a divisor k defined by exponents a_1..a_n,
    # sigma(k) = prod sigma_parts[i][a_i].
    # And k = prod p_i^a_i.
    # We check if k * sigma(k) == M.
    
    # This avoids re-factorizing k or computing sigma from scratch.
    
    # Pre-computation for each M:
    # 1. Calculate M.
    # 2. For each prime p_i in M with exponent e_i, compute array S_i where S_i[a] = sigma(p_i^a) for a=0..e_i.
    # 3. Enumerate all tuples (a_1, ..., a_n) with 0 <= a_i <= e_i.
    # 4. For each tuple, compute k = prod p_i^a_i and sigma_k = prod S_i[a_i].
    # 5. Check if k * sigma_k == M.
    
    # This is much faster than generating divisors and factorizing them.
    
    # Let's implement this.
    
    for exps in product(*[range(l+1) for l in e_limits]):
        if time.time() > deadline - 2.0:
            break
            
        # Calculate M and store exponents
        M = 1
        valid_exps = []
        for p, e in zip(p_list, exps):
            if e == 0:
                continue
            M *= p ** e
            valid_exps.append((p, e))
            
        if M > 10**18:
            continue
            
        if M < 1000: # Skip very small ones, we have the known 3
            continue
            
        # Pre-compute sigma parts for each prime power
        sigma_parts = []
        for p, e in valid_exps:
            parts = []
            s = 1
            pe = 1
            for a in range(e + 1):
                parts.append(s)
                pe *= p
                s += pe
            sigma_parts.append(parts)
            
        # Enumerate divisors via exponent tuples
        # We need to iterate over all combinations of exponents for the primes in M.
        # The number of divisors is prod(e_i + 1).
        
        # If M has too many divisors, this loop might be slow.
        # But max divisors for M <= 10^18 is ~100k.
        # 100k iterations * 6 mults = 600k ops.
        # If we do this for 250k M's, it's 150 billion ops. TOO SLOW.
        
        # We need to prune.
        # Most M will have few divisors.
        # But we are iterating over 250k M's.
        
        # Alternative:
        # Only check M if it has "potential".
        # Or, reduce the number of M candidates.
        
        # Let's reduce the exponent limits.
        # 2: 0-10
        # 3: 0-8
        # 5: 0-6
        # 7: 0-5
        # 11: 0-4
        # 13: 0-3
        # Combos: 11 * 9 * 7 * 6 * 5 * 4 = 83,160.
        # This is better.
        
        # Let's restart the loop logic with these limits in the code below.
        
    # The above loop structure is a bit messy with the break.
    # Let's write a clean version.
    
    # RESTART with optimized limits
    p_list = [2, 3, 5, 7, 11, 13]
    e_limits = [10, 8, 6, 5, 4, 3]
    
    # Pre-compute powers of primes for fast M calculation?
    # p_list is small.
    
    # We'll use a counter to track progress and break if needed.
    
    # To speed up the inner divisor enumeration, we can use recursion or product.
    # product is fine.
    
    # Let's run this.
    
    # One more optimization:
    # If M is not divisible by 2, sigma(k) must be even?
    # No.
    
    # Let's just run the search.
    
    # To ensure we don't miss the known best, we start with the known best.
    
    # We will store the best M and its k's.
    
    # To get the k's, we need to reconstruct them from the exponent tuples that worked.
    
    # We'll collect all valid k's for the current M.
    
    # If the count is > best_count, update best.
    
    # To avoid storing too many k's if M has many, we just count and store if better.
    
    # But we need to return the k's. So we must store them.
    
    # Let's implement.
    
    # Note: The previous loop block was just a sketch. The actual code below will be the implementation.
    
    # Implementation details:
    # 1. Iterate exponents.
    # 2. Compute M.
    # 3. If M > 10^18, skip.
    # 4. Compute sigma_parts.
    # 5. Iterate divisor exponents.
    # 6. Check condition.
    # 7. Update best.
    
    # To handle the time limit, we check time in the outer loop.
    
    # Let's refine the exponent limits to balance coverage and speed.
    # 10^18 is large.
    # 2^60 is 1e18.
    # If we use 2^10, 3^8, 5^6, 7^5, 11^4, 13^3.
    # Max M = 1024 * 6561 * 15625 * 16807 * 14641 * 2197.
    # This is huge. Many will be > 10^18.
    # So we will skip many.
    
    # This is fine.
    
    # Let's code it.

    # To speed up the inner loop, we can pre-compute the powers of primes for the exponents.
    
    # Let's try.
    
    best_M = 333312
    best_ks = [336, 372, 434]
    best_count = 3
    
    # Use a list of ranges for product
    ranges = [range(l+1) for l in e_limits]
    
    # To avoid creating a huge product object, we can use recursion or just product.
    # product is lazy, so it's fine.
    
    # We'll add a time check every 1000 iterations or so.
    
    count = 0
    for exps in product(*ranges):
        count += 1
        if count % 1000 == 0 and time.time() > deadline - 5.0:
            break
            
        M = 1
        valid_exps = []
        for p, e in zip(p_list, exps):
            if e > 0:
                M *= p ** e
                valid_exps.append((p, e))
                
        if M > 10**18:
            continue
            
        if M < 1000:
            continue
            
        # Pre-compute sigma parts
        sigma_parts = []
        for p, e in valid_exps:
            parts = [1]
            s = 1
            pe = 1
            for _ in range(e):
                pe *= p
                s += pe
                parts.append(s)
            sigma_parts.append(parts)
            
        # Enumerate divisors
        # Get ranges for divisor exponents
        div_ranges = [range(e+1) for _, e in valid_exps]
        
        # If no primes, M=1, skipped.
        
        valid_ks = []
        
        # Iterate over divisor exponents
        # This is the inner loop.
        # To speed up, we can calculate k and sigma_k incrementally?
        # Or just use product.
        
        # If the number of divisors is large, this might be slow.
        # But for most M, it's small.
        
        for d_exps in product(*div_ranges):
            k = 1
            sigma_k = 1
            for i, ae in enumerate(d_exps):
                if ae > 0:
                    p = valid_exps[i][0]
                    k *= p ** ae
                    sigma_k *= sigma_parts[i][ae]
                    
            if k * sigma_k == M:
                valid_ks.append(k)
                
        if len(valid_ks) > best_count:
            best_count = len(valid_ks)
            best_M = M
            best_ks = valid_ks
            # We could break here if we found a very high count?
            # Unlikely to find 100+ quickly.
            
    # Return the best found
    return {'M': int(best_M), 'k': [int(k) for k in best_ks]}

# EVOLVE-BLOCK-END
