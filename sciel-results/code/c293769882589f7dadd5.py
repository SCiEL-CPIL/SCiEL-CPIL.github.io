import time
from math import log

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # Target: Minimize distinct prime factors of pairwise sums.
    # Hypothesis: Optimal solution uses only primes {2, 3} for the sums.
    # If we find 12 distinct integers <= 10000 such that all pairwise sums 
    # are of the form 2^a * 3^b, the answer is 2.
    
    # Step 1: Generate all 2,3-smooth numbers up to 20,000 (max possible sum)
    # A number is 2,3-smooth if it has no prime factors other than 2 and 3.
    smooth_23 = set()
    # 2^a * 3^b <= 20000
    # 2^14 = 16384, 2^15 = 32768
    for a in range(15):
        pow2 = 1 << a
        if pow2 > 20000:
            break
        b = 0
        val = pow2
        while val <= 20000:
            smooth_23.add(val)
            if val > 20000 // 3:
                break
            val *= 3
            b += 1
    
    # Step 2: Generate candidate integers a_i.
    # To maximize the chance that a_i + a_j is smooth, a_i and a_j 
    # should ideally be from a set where their sums are constrained.
    # If we restrict a_i to be 2,3-smooth numbers <= 10000, we have a manageable pool.
    # Let's check if there are at least 12 such numbers.
    candidates = [x for x in smooth_23 if x <= 10000]
    candidates.sort()
    
    if len(candidates) < 12:
        # Fallback if pool is too small (unlikely, 2,3-smooth numbers are common)
        # Expand to 2,3,5-smooth
        smooth_235 = set()
        for a in range(14):
            pow2 = 1 << a
            if pow2 > 10000:
                break
            for b in range(10):
                pow3 = 3 ** b
                if pow2 * pow3 > 10000:
                    break
                for c in range(8):
                    val = pow2 * pow3 * (5 ** c)
                    if val > 10000:
                        break
                    smooth_235.add(val)
        candidates = sorted(smooth_235)
        # If we use 2,3,5 for candidates, we should check sums against 2,3,5-smooth
        # But let's stick to the primary hypothesis: Sums are 2,3-smooth.
        # If we pick candidates from 2,3,5, their sums might not be 2,3-smooth.
        # So we stick to 2,3-smooth candidates for the first attempt.
        
    # Step 3: Backtracking search
    # We want to find a subset of size 12 from candidates
    # such that for all i, j in subset, candidates[i] + candidates[j] is in smooth_23
    
    n_needed = 12
    found = []
    
    # Helper to check if a list of chosen numbers is valid so far
    def is_valid_extension(current_choice, new_num):
        for num in current_choice:
            s = num + new_num
            if s not in smooth_23:
                return False
        return True

    def backtrack(current, start_idx):
        if time.time() > deadline:
            return False
        if len(current) == n_needed:
            found.extend(current)
            return True
            
        # Heuristic: Try smaller numbers first? Or larger?
        # Usually, smaller numbers have more "room" to sum with others to stay smooth?
        # Actually, 2^a + 2^b = 2^min(a,b)(1+2^|a-b|).
        # If we pick powers of 2: 1, 2, 4, 8...
        # 1+2=3 (ok), 1+4=5 (bad). So powers of 2 don't work well together.
        # We need a mix.
        
        for i in range(start_idx, len(candidates)):
            c = candidates[i]
            if is_valid_extension(current, c):
                current.append(c)
                if backtrack(current, i + 1):
                    return True
                current.pop()
                
        return False

    # Start backtracking
    # Optimization: We can prune the candidate list.
    # If a candidate c cannot form a valid sum with ANY other candidate in the pool 
    # (except itself, but we need distinct), it's useless.
    # However, checking this for all pairs is O(N^2). N is small (~100-200).
    
    # Let's just run the backtracker.
    # To speed it up, we can order candidates.
    # Sorting by value is good.
    
    if backtrack([], 0):
        # Verify solution
        if len(found) == 12:
            # The prime factors are just {2, 3}
            return found

    # If 2,3-smooth sums failed, try 2,3,5-smooth sums.
    # This requires a larger smooth set for sums and potentially different candidates.
    # But 2 primes is the target. If we can't find it, 3 is the next best.
    
    # Let's try a broader search for 3 primes: {2, 3, 5}
    # Generate 2,3,5-smooth numbers up to 20000 for sums
    smooth_235_sums = set()
    for a in range(15):
        pow2 = 1 << a
        if pow2 > 20000:
            break
        for b in range(10):
            pow3 = 3 ** b
            if pow2 * pow3 > 20000:
                break
            for c in range(8):
                val = pow2 * pow3 * (5 ** c)
                if val > 20000:
                    break
                smooth_235_sums.add(val)
    
    # Candidates: 2,3,5-smooth numbers <= 10000
    candidates_235 = [x for x in smooth_235_sums if x <= 10000]
    candidates_235.sort()
    
    found_235 = []
    
    def is_valid_extension_235(current_choice, new_num):
        for num in current_choice:
            s = num + new_num
            if s not in smooth_235_sums:
                return False
        return True

    def backtrack_235(current, start_idx):
        if time.time() > deadline:
            return False
        if len(current) == n_needed:
            found_235.extend(current)
            return True
            
        for i in range(start_idx, len(candidates_235)):
            c = candidates_235[i]
            if is_valid_extension_235(current, c):
                current.append(c)
                if backtrack_235(current, i + 1):
                    return True
                current.pop()
                
        return False

    if backtrack_235([], 0):
        if len(found_235) == 12:
            # Prime factors are {2, 3, 5}
            return found_235

    # Fallback: If specific structured search fails, use a local search on a broader pool
    # to ensure we return *something* better than the seed if possible, or the seed.
    # But the structured search above is very likely to succeed for N=12.
    
    # Generic fallback: Random local search on 2,3,5-smooth candidates
    import random
    random.seed(42)
    
    # Pool for local search: 2,3,5-smooth <= 10000
    pool = candidates_235
    if len(pool) < 12:
        # Should not happen
        return list(range(1, 13))
        
    # Initial random selection
    current_set = random.sample(pool, 12)
    
    def get_prime_factors(n):
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors
        
    def evaluate_set(numbers):
        prime_union = set()
        for i in range(12):
            for j in range(i + 1, 12):
                prime_union.update(get_prime_factors(numbers[i] + numbers[j]))
        return len(prime_union)
        
    best_score = float('inf')
    best_set = current_set[:]
    current_score = evaluate_set(current_set)
    if current_score < best_score:
        best_score = current_score
        best_set = current_set[:]
        
    for _ in range(5000):
        if time.time() > deadline:
            break
        idx = random.randint(0, 11)
        old_val = current_set[idx]
        new_val = random.choice(pool)
        if new_val in current_set and new_val != old_val:
            continue
        current_set[idx] = new_val
        new_score = evaluate_set(current_set)
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_set = current_set[:]
        else:
            current_set[idx] = old_val
            
    return best_set

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
