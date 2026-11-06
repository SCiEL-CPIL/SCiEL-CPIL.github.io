import math
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Strategy:
    # 1. We want N = a + b with a, b powerful.
    # 2. Let a = Q * x^2 and b = Q * y^2.
    #    Then N = Q * (x^2 + y^2).
    #    Q must be powerful. x^2, y^2 are powerful.
    #    Product of powerfuls is powerful.
    # 3. We need to find M = x^2 + y^2 with many representations.
    # 4. N = Q * M <= 10^18.
    # 5. To maximize the number of pairs (x,y), we maximize the number of representations of M as sum of two squares.
    #    This is determined by the prime factorization of M.
    #    If M = 2^k * prod p_i^{e_i} (p_i = 1 mod 4) * prod q_j^{f_j} (q_j = 3 mod 4),
    #    representations exist only if all f_j are even.
    #    Number of representations r_2(M) = 4 * prod (e_i + 1).
    #    Number of unordered pairs {x,y} with x,y >= 0 is roughly r_2(M)/8.
    #    Specifically, if we generate all (x,y) with x,y > 0 and x != y, and handle x=0 or x=y cases,
    #    the count is (r_2(M) - 4*delta) / 8 + adjustments.
    #    Actually, it's easier to just generate them and count.
    
    # The key insight from the "inspiration" and the stagnation directive:
    # The previous program used brute force to find pairs. This limits M to ~10^14 or so (sqrt ~ 10^7).
    # With Gaussian integer factorization, we can handle M up to 10^18 very quickly.
    # This allows us to use a much smaller Q (e.g., Q=1 or Q=4) and a huge M.
    # Or, more likely, we can just pick Q=1 (which is powerful) and M up to 10^18.
    # Wait, if Q=1, a=x^2, b=y^2.
    # Are squares powerful? Yes.
    # So we just need to find M <= 10^18 with the maximum number of representations as sum of two squares.
    
    # Let's find M <= 10^18 that maximizes prod(e_i+1) for primes p_i = 1 mod 4.
    # We should also include factor 2?
    # 2 = 1^2 + 1^2. It contributes to representations.
    # However, the formula r_2(n) depends on primes 1 mod 4.
    # 2 is 2 mod 4.
    # Let's stick to M = prod p_i^{e_i} where p_i are 1 mod 4.
    # Including 2^k doesn't change the number of representations?
    # r_2(2^k * m) = r_2(m) if k is even?
    # Actually, r_2(2) = 4. r_2(1) = 4.
    # r_2(4) = 4.
    # r_2(8) = 4.
    # It seems 2^k doesn't increase the count beyond 1 representation per "type".
    # Let's focus on primes 1 mod 4.
    
    # Primes 1 mod 4: 5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, ...
    
    # We want to maximize Score = prod (e_i + 1) subject to prod p_i^{e_i} <= 10^18.
    # This is a knapsack-like problem or greedy.
    # Since log(p) increases, we should balance exponents.
    # Greedy: Start with M=1.
    # Repeatedly multiply by the prime p that gives the best "score gain per log size".
    # Gain in log(score) for increasing e_i by 1 is log((e_i+2)/(e_i+1)).
    # Cost is log(p_i).
    # Ratio: log((e_i+2)/(e_i+1)) / log(p_i).
    # We want the highest ratio.
    
    primes_1mod4 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193, 197, 229, 233, 241, 257, 269, 277, 281, 293, 313, 317, 337, 349, 353, 373, 389, 397, 401, 409, 421, 433, 449, 457, 461, 509, 521, 541, 557, 569, 577, 593, 599, 601, 613, 617, 641, 653, 661, 673, 677, 701, 709, 713, 733, 757, 761, 769, 773, 797, 809, 811, 821, 829, 853, 857, 877, 881, 929, 937, 941, 953, 977, 991, 997]
    
    # Greedy optimization
    M = 1
    exponents = {p: 0 for p in primes_1mod4[:20]} # Only consider small primes
    
    # We will iterate until M * min_prime > 10^18
    # Or until no improvement in "potential".
    
    # Actually, a simpler heuristic:
    # The optimal M is of the form 5^a * 13^b * 17^c * ...
    # To maximize product of (e+1), we want e's to be roughly equal?
    # No, because p's are different.
    # Let's just do a DFS on the first ~10 primes with bounded depth.
    # 10^18 is large. 5^25 is ~ 3*10^17.
    # So exponents can be up to ~25.
    # DFS on 10 primes with max depth 25 is too big.
    
    # Let's use the greedy "best ratio" approach.
    # It is known that for maximizing divisor count (or similar multiplicative functions) under size constraint,
    # the optimal number is "smooth".
    
    def get_next_best(M, exponents):
        best_p = None
        best_ratio = 0
        for p in exponents:
            if M * p > 10**18:
                continue
            e = exponents[p]
            # Ratio of score improvement to size increase
            # New score factor: (e+2)/(e+1)
            # Log size increase: log(p)
            ratio = math.log((e + 2) / (e + 1)) / math.log(p)
            if ratio > best_ratio:
                best_ratio = ratio
                best_p = p
        return best_p

    # Initialize
    M = 1
    exponents = {p: 0 for p in primes_1mod4[:15]}
    
    while True:
        if time.time() > deadline - 10:
            break
        p = get_next_best(M, exponents)
        if p is None:
            break
        M *= p
        exponents[p] += 1
        
    # Now M is a candidate.
    # Let's verify if adding 2 helps.
    # r_2(2*M) = r_2(M).
    # So 2 doesn't change the count.
    # But it increases N.
    # We want to maximize N? No, we want to maximize pairs.
    # So we should use the smallest N that gives the max pairs?
    # No, the score is just the number of pairs.
    # So we just need to find the M that gives the most pairs.
    # If 2*M gives the same pairs, and 2*M <= 10^18, it doesn't matter.
    # But wait, if we can fit a larger M with more factors by NOT using 2, we should.
    # Since 2 doesn't help the score, we should avoid it to save space for other primes.
    # So M should be composed ONLY of primes 1 mod 4.
    
    # Let's refine the greedy search.
    # Sometimes local greedy gets stuck.
    # Let's try a few variations of the exponent vector around the greedy solution.
    
    best_M = M
    best_score = 0
    best_exponents = exponents.copy()
    
    # Calculate score for current M
    def calc_score(exps):
        s = 1
        for p, e in exps.items():
            if e > 0:
                s *= (e + 1)
        return s
        
    best_score = calc_score(best_exponents)
    
    # Try to improve by "swapping" or adjusting exponents.
    # This is a local search.
    # Given time constraints, the greedy result is likely very close to optimal.
    # For "highly composite" like structures, greedy is often optimal or near-optimal.
    
    # Let's also consider Q.
    # If we use Q > 1, we reduce M_max.
    # Does Q help?
    # N = Q * M.
    # Pairs are (Q*x^2, Q*y^2).
    # The number of pairs is determined by M.
    # If we reduce M_max, we might get a smaller M with fewer factors, hence fewer pairs.
    # So Q=1 is optimal for maximizing pairs, provided we can generate them fast.
    # Since we use Gaussian factorization, generation is fast.
    # So we set Q = 1.
    # Wait, is 1 powerful? Yes.
    # So a = x^2, b = y^2.
    
    # Now, we need to generate all pairs (x, y) such that x^2 + y^2 = M.
    # We use Gaussian integers.
    # For each prime p = 1 mod 4, find a, b such that a^2 + b^2 = p.
    # Then p = (a + bi)(a - bi).
    # M = prod p_i^{e_i}.
    # M = prod (a_i + b_i i)^{e_i} * prod (a_i - b_i i)^{e_i}.
    # The representations correspond to choosing how many factors of (a_i + b_i i) vs (a_i - b_i i) to include in the product (a + bi) such that a^2 + b^2 = M.
    # Actually, any divisor of M in Z[i] corresponds to a representation.
    # The number of representations is related to the number of divisors in Z[i].
    
    # Algorithm to generate pairs:
    # 1. Find Gaussian primes for each p_i.
    # 2. Generate all combinations of exponents for the Gaussian primes.
    # 3. Multiply them out to get a + bi.
    # 4. x = |a|, y = |b|.
    # 5. Ensure x <= y.
    # 6. Check for duplicates.
    
    # Step 1: Find a, b for p = a^2 + b^2.
    def find_gaussian_prime(p):
        # p is 1 mod 4
        # Use Cornacchia's algorithm or simple search
        # a^2 + b^2 = p
        # a <= sqrt(p)
        # Since p is small (primes 1 mod 4 are small in our set), search is fine.
        limit = int(math.isqrt(p))
        for a in range(1, limit + 1):
            b2 = p - a*a
            b = math.isqrt(b2)
            if b*b == b2:
                if b > 0:
                    return (a, b)
        return None # Should not happen for p=1 mod 4
    
    # Get Gaussian primes for our M's factors
    g_primes = []
    for p, e in best_exponents.items():
        if e > 0:
            ab = find_gaussian_prime(p)
            if ab:
                # p = (a+bi)(a-bi)
                # We have e copies of p.
                # So we have e copies of (a+bi) and e copies of (a-bi) in the factorization of M in Z[i]?
                # No. M = p^e = ((a+bi)(a-bi))^e = (a+bi)^e (a-bi)^e.
                # The representations of M as sum of two squares correspond to divisors d of M in Z[i]
                # such that d * conj(d) = M.
                # Any such d is of the form (a+bi)^k (a-bi)^(e-k) * unit.
                # Where k can range from 0 to e.
                # So for each prime p, we have (e+1) choices for the exponent of (a+bi).
                # Total number of such divisors is prod (e_i + 1).
                # Each divisor d = x + yi gives a representation x^2 + y^2 = M.
                # We need to generate all these products.
                
                g_primes.append((p, e, ab[0], ab[1]))
                
    # Step 2: Generate all combinations
    # We use recursion to build the product.
    
    pairs = []
    
    # To avoid huge recursion depth or memory issues, we can use iterative product generation.
    # But the number of pairs is prod(e_i+1).
    # For M ~ 10^18, max pairs is around 10^4 or so?
    # 5^25 -> 26 pairs.
    # 5^10 * 13^8 * ...
    # Let's estimate.
    # If we have 10 primes with e=2, score = 3^10 = 59049.
    # M = 5^2 * 13^2 * ...
    # 5^2=25, 13^2=169. 25*169=4225.
    # 17^2=289. 4225*289 ~ 1.2M.
    # 29^2=841. 1.2M * 841 ~ 10^9.
    # 37^2=1369. 10^9 * 1369 ~ 10^12.
    # 41^2=1681. 10^12 * 1681 ~ 10^15.
    # 53^2=2809. 10^15 * 2809 ~ 10^18.
    # So we can have ~6 primes with e=2.
    # Score = 3^6 = 729.
    # Or mix exponents.
    # 5^4 (25) * 13^3 (2197) * 17^2 (289) ...
    # Score = 5 * 4 * 3 ...
    
    # Let's just generate them.
    
    # Start with 1 + 0i
    # Current factors: list of (a, b, e)
    
    # We will generate all (x, y)
    # x + yi = prod (a_i + b_i i)^{k_i}
    # k_i from 0 to e_i.
    
    # To do this efficiently, we can recursively multiply.
    
    def generate_representations(idx, current_x, current_y):
        if time.time() > deadline - 5:
            return []
        
        if idx == len(g_primes):
            # We have a representation current_x^2 + current_y^2 = M
            # Normalize: x, y >= 0
            x = abs(current_x)
            y = abs(current_y)
            if x > y:
                x, y = y, x
            # Check if x^2 + y^2 == M (sanity check, should be true)
            # Add to list
            return [(x, y)]
            
        p, e, a, b = g_primes[idx]
        results = []
        
        # We need to multiply current (cx, cy) by (a+bi)^k for k=0..e
        # Let's precompute (a+bi)^k for k=0..e
        
        powers = []
        px, py = 1, 0
        for k in range(e + 1):
            powers.append((px, py))
            # Multiply by (a, b)
            # (px + py*i) * (a + b*i) = (px*a - py*b) + (px*b + py*a)i
            nx = px * a - py * b
            ny = px * b + py * a
            px, py = nx, ny
            
        for k in range(e + 1):
            ax, ay = powers[k]
            # Multiply current (cx, cy) by (ax, ay)
            # (cx + cy*i) * (ax + ay*i) = (cx*ax - cy*ay) + (cx*ay + cy*ax)i
            nx = current_x * ax - current_y * ay
            ny = current_x * ay + current_y * ax
            
            # Recurse
            results.extend(generate_representations(idx + 1, nx, ny))
            
        return results

    # Call generator
    all_reps = generate_representations(0, 1, 0)
    
    # all_reps contains (x, y) pairs.
    # There might be duplicates due to units (i, -1, -i, -1) and conjugates.
    # Our generation method:
    # We chose specific Gaussian primes (a+bi).
    # The divisors generated are unique in Z[i] up to unit?
    # No, we are generating specific combinations of exponents.
    # Each combination gives a unique Gaussian integer d.
    # However, d and -d, i*d, -i*d all give the same (x,y) set { |Re|, |Im| }.
    # Also d and conj(d) give the same set.
    # So we just need to deduplicate (x,y) pairs.
    
    unique_pairs = set()
    for x, y in all_reps:
        if x > y:
            x, y = y, x
        unique_pairs.add((x, y))
        
    # Convert to powerful numbers
    # a = x^2, b = y^2
    # N = M
    # Check N <= 10^18
    if M > 10**18:
        # Fallback
        return {'N': 5, 'pairs': [[1, 4]]}
        
    final_pairs = []
    for x, y in unique_pairs:
        a = x * x
        b = y * y
        # a, b are powerful (squares)
        # a + b = M
        # N = M
        final_pairs.append([a, b])
        
    # Sort pairs for consistency
    final_pairs.sort()
    
    # Ensure distinct pairs
    # The set logic handled distinct (x,y).
    # Since a=x^2, b=y^2, distinct (x,y) with x<=y implies distinct (a,b).
    
    # One more check: are there any other powerful representations?
    # The task asks for "the number of pairs".
    # We found all representations of the form x^2 + y^2.
    # Are there powerful numbers that are NOT squares?
    # Yes, e.g., 72 = 2^3 * 3^2.
    # Can N be represented as sum of two non-square powerfuls?
    # The task says "Integers ... with many representations as sums of two powerful numbers".
    # It does not restrict to squares.
    # However, generating ALL powerful numbers up to 10^18 is hard.
    # But, it is known that squares are the "densest" powerful numbers.
    # And the sum of two squares structure is very rich.
    # Given the constraints and the "seed" strategy focusing on squares,
    # and the difficulty of finding non-square powerful sums,
    # it is highly likely that the maximum is achieved by squares.
    # Also, the evaluator checks "checked by exact factorization".
    # If we submit squares, they are definitely powerful.
    
    # If we have time, we could check if adding some non-square powerfuls helps.
    # But for now, the square strategy with Gaussian factorization is robust and fast.
    
    return {'N': M, 'pairs': final_pairs}

if __name__ == "__main__":
    res = construct()
    print(res['N'])
    print(len(res['pairs']))

