"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from math import factorial, isqrt
    from itertools import combinations, product
    
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    # 12! = 479001600
    # Prime factorization: 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    target = factorial(12)
    
    # Split primes into two groups for meet-in-the-middle
    # Group A: {2, 3} with exponents (0..10, 0..5)
    # Group B: {5, 7, 11} with exponents (0..2, 0..1, 0..1)
    
    # Generate all possible values from group A (primes 2, 3)
    group_a_values = []
    for e2 in range(11):  # 0..10
        for e3 in range(6):  # 0..5
            val = (2 ** e2) * (3 ** e3)
            if val > 12:  # must be > 12
                group_a_values.append(val)
    
    # Generate all possible values from group B (primes 5, 7, 11)
    group_b_values = []
    for e5 in range(3):  # 0..2
        for e7 in range(2):  # 0..1
            for e11 in range(2):  # 0..1
                val = (5 ** e5) * (7 ** e7) * (11 ** e11)
                if val > 12:  # must be > 12
                    group_b_values.append(val)
    
    # Also include 1 as a "neutral" element for each group (exponent 0 for all)
    # Actually, we need to think differently.
    
    # Better approach: DFS with strong pruning
    # We need to partition the prime factors into 1-12 distinct factors > 12
    # minimizing the largest factor
    
    # Prime factorization of 12!:
    # 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    
    # Let's use a different MITM strategy:
    # Generate all possible "partial products" that can be formed from subsets of primes
    # with the constraint that each factor > 12 and we track which prime exponents are used
    
    # Actually, let me think about this more carefully.
    # The total product is 12! = 2^10 * 3^5 * 5^2 * 7 * 11
    
    # Each factor in our decomposition must be > 12 and distinct.
    # We want to minimize the maximum factor.
    
    # Strategy: Binary search on the maximum factor M.
    # For a given M, check if we can decompose 12! into distinct factors in (12, M].
    
    # But checking feasibility is hard. Let's try direct search.
    
    # Key insight: The factors must multiply to 12! and each must be > 12.
    # Since 12! = 479001600, and we need factors > 12, the number of factors is limited.
    # Minimum factor > 12, so if we have k factors, each >= 13 (actually > 12, so >= 13).
    # 13^k <= 12! => k <= log(12!)/log(13) ≈ log(479001600)/log(13) ≈ 19.98/2.56 ≈ 7.8
    # So k <= 7 at most (since factors must be distinct and > 12, minimum is 13*14*15*16*17*18*19 = quite large)
    
    # Let me compute: 13*14*15*16*17*18*19 = let me estimate
    # 13*19 = 247, 14*18 = 252, 15*17 = 255, so 247*252*255*16 ≈ 62244*255*16 ≈ 15872220*16 ≈ 253955520
    # That's less than 479001600, so 7 factors is possible.
    # 13*14*15*16*17*18*19*20 = 253955520*20 ≈ 5079110400, way too big.
    # So max 7 factors.
    
    # Actually wait, factors don't have to be consecutive. They just need to be distinct and > 12.
    # The point is that with factors > 12, we can have at most 7 factors (since even the smallest 7 distinct > 12 gives a product close to 12!).
    
    # Let me verify: smallest 7 distinct integers > 12 are 13,14,15,16,17,18,19
    # Product = 13*14*15*16*17*18*19
    # = (13*19)*(14*18)*(15*17)*16
    # = 247 * 252 * 255 * 16
    # 247*252 = 62244
    # 255*16 = 4080
    # 62244*4080 = 253,955,520
    # 12! = 479,001,600
    # So 7 factors is feasible in terms of product size.
    
    # With 8 factors: smallest would be 13*14*...*20 = 253955520*20 = 5,079,110,400 > 12!
    # So indeed max 7 factors.
    
    # So we search over k = 1 to 7 factors.
    
    # For each k, we want to find k distinct integers > 12 whose product is 12!
    # minimizing the maximum.
    
    # Better approach: Use the prime factor structure.
    # 12! = 2^10 * 3^5 * 5^2 * 7 * 11
    
    # Each factor is a product of some subset of these prime powers.
    # We need to assign each prime's exponent to one of the k factors.
    
    # This is still complex. Let me try a direct search with pruning.
    
    # Approach: Generate candidate factors > 12 that divide 12!, then search for subsets.
    
    # First, generate all divisors of 12! that are > 12.
    # 12! has (10+1)*(5+1)*(2+1)*(1+1)*(1+1) = 11*6*3*2*2 = 792 divisors.
    
    # Filter to those > 12.
    
    divisors = []
    for e2 in range(11):
        for e3 in range(6):
            for e5 in range(3):
                for e7 in range(2):
                    for e11 in range(2):
                        val = (2 ** e2) * (3 ** e3) * (5 ** e5) * (7 ** e7) * (11 ** e11)
                        if val > 12:
                            divisors.append(val)
    
    # Sort for potential pruning
    divisors.sort()
    
    # Now we need to find a subset of these divisors (distinct values) whose product is 12!
    # and the maximum is minimized.
    
    # But the divisors list may have duplicates? No, each combination of exponents gives a unique value.
    # But different exponent combinations could give the same value? No, by unique factorization.
    
    # So we have at most 792 - (number of divisors <= 12) candidates.
    # Divisors <= 12: 1,2,3,4,5,6,7,8,9,10,11,12 = 12 divisors.
    # So 792 - 12 = 780 candidates.
    
    # We need to find a subset of these 780 candidates whose product is exactly 12!
    # and the max is minimized.
    
    # This is a subset product problem, which is NP-hard in general, but here the structure helps.
    
    # Key observation: The product of all selected factors must be 12!.
    # Equivalently, in terms of prime exponents, the sum of exponents for each prime across all selected factors must equal the exponent in 12!.
    
    # So for prime 2: sum of e2 across selected factors = 10
    # For prime 3: sum of e3 across selected factors = 5
    # For prime 5: sum of e5 across selected factors = 2
    # For prime 7: sum of e7 across selected factors = 1
    # For prime 11: sum of e11 across selected factors = 1
    
    # This is a 5-dimensional constraint.
    
    # Let me use DFS over the sorted divisors, tracking the remaining exponents needed.
    
    best_max = float('inf')
    best_factors = None
    
    # Precompute exponent tuples for each divisor
    divisor_info = []
    for val in divisors:
        # Compute exponents
        e2 = 0
        temp = val
        while temp % 2 == 0:
            e2 += 1
            temp //= 2
        e3 = 0
        while temp % 3 == 0:
            e3 += 1
            temp //= 3
        e5 = 0
        while temp % 5 == 0:
            e5 += 1
            temp //= 5
        e7 = 0
        while temp % 7 == 0:
            e7 += 1
            temp //= 7
        e11 = 0
        while temp % 11 == 0:
            e11 += 1
            temp //= 11
        divisor_info.append((val, e2, e3, e5, e7, e11))
    
    # Target exponents
    target_e = (10, 5, 2, 1, 1)
    
    # DFS with pruning
    # State: index in divisor_info, current factors selected, current max, remaining exponents
    # To avoid duplicates, we enforce that factors are selected in increasing order (by value)
    
    def dfs(idx, selected, current_max, rem_e):
        nonlocal best_max, best_factors
        
        if time.time() > deadline:
            return
        
        # Check if we've used up all exponents
        if rem_e == (0, 0, 0, 0, 0):
            if selected:
                if current_max < best_max:
                    best_max = current_max
                    best_factors = list(selected)
            return
        
        # If no more divisors to consider
        if idx >= len(divisor_info):
            return
        
        # Pruning: if current_max >= best_max, no point continuing (since we're adding more factors, max can only stay same or increase)
        # Actually, the max of the final set is max(current_max, max of future additions). If current_max >= best_max, we can prune.
        if current_max >= best_max:
            return
        
        # Another pruning: the remaining product must be achievable
        # The remaining exponents must be achievable by some subset of divisors[idx:]
        # This is hard to check quickly, so let's rely on the max pruning.
        
        # Option 1: Skip divisor_info[idx]
        dfs(idx + 1, selected, current_max, rem_e)
        
        # Option 2: Take divisor_info[idx]
        val, e2, e3, e5, e7, e11 = divisor_info[idx]
        # Check if we can subtract these exponents
        if (e2 <= rem_e[0] and e3 <= rem_e[1] and e5 <= rem_e[2] and 
            e7 <= rem_e[3] and e11 <= rem_e[4]):
            new_rem = (rem_e[0] - e2, rem_e[1] - e3, rem_e[2] - e5, rem_e[3] - e7, rem_e[4] - e11)
            new_max = max(current_max, val)
            if new_max < best_max:  # Only proceed if it can potentially improve
                dfs(idx + 1, selected + [val], new_max, new_rem)
    
    # Start DFS
    dfs(0, [], 0, target_e)
    
    if best_factors is not None:
        return best_factors
    else:
        # Fallback: return the whole 12! as a single factor
        return [target]

# EVOLVE-BLOCK-END
