import math
import time

# Global cache to ensure instant return on subsequent calls or if the evaluator
# restarts the process (though typically it's one process, this ensures safety
# against re-entrant calls or if the module is imported multiple times in a test harness)
_CACHE = None

def construct():
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    start_time = time.time()
    deadline = start_time + 145.0

    def time_left():
        return deadline - time.time()

    # Strategy:
    # 1. Choose N = Q * M where Q is a powerful number (square of a product of small primes)
    #    and M is a powerful number composed of primes p_i.
    # 2. The number of representations of N as a sum of two powerful numbers is related
    #    to the number of representations of M as a sum of two squares, scaled by Q.
    #    Specifically, if N = Q * M, and Q is a square, then a+b=N with a,b powerful
    #    implies a = Q*x^2, b = Q*y^2 where x^2+y^2=M.
    #    Wait, a and b must be powerful.
    #    If a = Q * x^2, for a to be powerful, Q * x^2 must be powerful.
    #    If Q is a square of a product of distinct primes (primorial squared), 
    #    Q = (p1...pk)^2. Then a = (p1...pk)^2 * x^2 = (x * p1...pk)^2.
    #    This is a perfect square. Every perfect square is a powerful number.
    #    So, if we set N = Q * M where Q is a square, any representation M = x^2 + y^2
    #    yields a representation N = (Q*x^2) + (Q*y^2).
    #    Since Q is a square, Q*x^2 is a square, hence powerful.
    #    Thus, the number of pairs (a,b) of powerful numbers summing to N is at least
    #    the number of pairs (x,y) of integers such that x^2 + y^2 = M.
    
    # To maximize this, we want M to have many representations as a sum of two squares.
    # The number of representations r_2(M) is 4 * (d_1(M) - d_3(M)), where d_1(M) is the 
    # number of divisors of M congruent to 1 mod 4, and d_3(M) is the number of divisors 
    # of M congruent to 3 mod 4.
    # If M is a product of primes p_i all congruent to 1 mod 4, then all divisors are 
    # 1 mod 4, so r_2(M) = 4 * d(M).
    # The number of distinct pairs (x,y) with x,y >= 1 and x <= y is roughly r_2(M)/8 
    # (accounting for symmetries and zeros).
    # To maximize d(M), we should pick M to be a product of distinct primes, or powers?
    # d(M) is maximized for a given size by using small primes.
    # However, M must be composed of primes 1 mod 4.
    # Small primes 1 mod 4: 5, 13, 17, 29, 37, 41, ...
    
    # Let's choose M to be the product of the first few primes 1 mod 4.
    # Let M = 5 * 13 * 17 * 29 * 37 * 41 * 53 * 61 * 73 * 89 * 97 * 101
    # This M will have 2^12 = 4096 divisors.
    # The number of pairs (x,y) with x^2+y^2=M and x,y > 0, x <= y is approx 4096 / 8 = 512.
    # This is a good number.
    
    # We also need N <= 10^18.
    # N = Q * M.
    # Q must be a square. Let Q = 1 for simplicity? 
    # If Q=1, N=M. M is product of 12 primes ~ 5*13*...*101.
    # Log10(M) approx sum(log10(p)) for p in [5, 101].
    # 5(0.7) 13(1.1) 17(1.2) 29(1.5) 37(1.6) 41(1.6) 53(1.7) 61(1.8) 73(1.9) 89(1.9) 97(2.0) 101(2.0)
    # Sum ~ 19. So M is around 10^19, which is too big.
    # We need M < 10^18.
    # Let's use fewer primes.
    # First 10 primes 1 mod 4: 5, 13, 17, 29, 37, 41, 53, 61, 73, 89.
    # Product approx 10^17.
    # Let's calculate exactly.
    
    primes_1mod4 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113]
    
    M = 1
    count_primes = 0
    for p in primes_1mod4:
        if M * p <= 10**17: # Leave room for Q if needed, but Q=1 is fine
            M *= p
            count_primes += 1
        else:
            break
            
    # M is product of first count_primes primes 1 mod 4.
    # d(M) = 2^count_primes.
    # Expected pairs ~ d(M) / 8.
    
    # We can also multiply M by a square of a prime 1 mod 4 to increase d(M) further?
    # If M = p^k * ..., d(M) increases.
    # But we are constrained by N <= 10^18.
    # Since Q=1, N=M.
    # We can make M larger by using powers.
    # Example: 5^2 * 13 * 17 ...
    # d(5^2 * 13 * 17) = 3 * 2 * 2 = 12. vs 2*2*2=8 for distinct.
    # So powers are better for d(M).
    
    # Let's construct M to maximize d(M) subject to M <= 10^18.
    # This is a knapsack-like problem.
    # We want to maximize product of (e_i + 1) subject to sum(e_i * log(p_i)) <= log(10^18).
    # Greedy approach: assign exponents to smallest primes first.
    
    max_log_M = 18 * math.log(10)
    current_log_M = 0
    exponents = {}
    
    # Sort primes 1 mod 4
    # We should probably include 5 with high power, then 13, etc.
    
    # Let's try a specific strong candidate known in literature or compute it.
    # A common highly composite-like number for sum of squares.
    
    # Let's use a greedy algorithm to build M.
    # Candidates for primes: 5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101
    # We want to maximize score.
    
    # Let's just pick a fixed, pre-computed good M.
    # M = 5^4 * 13^2 * 17^2 * 29 * 37
    # Log M = 4*1.6 + 2*2.56 + 2*2.83 + 3.36 + 3.61 = 6.4 + 5.12 + 5.66 + 3.36 + 3.61 = 24.15
    # 10^24.15 is way too big.
    
    # Target Log10(M) <= 18.
    # Let's try: 5^3 * 13^2 * 17 * 29 * 37
    # Log10: 3*0.7 + 2*1.11 + 1.23 + 1.46 + 1.57 = 2.1 + 2.22 + 1.23 + 1.46 + 1.57 = 8.58. Too small.
    
    # We can use more primes.
    # 5^2 * 13^2 * 17^2 * 29^2 * 37^2 * 41^2 * 53
    # Log10: 2*(0.7+1.11+1.23+1.46+1.57+1.61) + 1.72
    # = 2 * 7.68 + 1.72 = 15.36 + 1.72 = 17.08.
    # This fits!
    # d(M) = (2+1)*(2+1)*(2+1)*(2+1)*(2+1)*(2+1)*(1+1) = 3^6 * 2 = 729 * 2 = 1458.
    # Pairs ~ 1458 / 8 = 182.
    
    # Can we do better?
    # Try 5^3 * 13^2 * 17^2 * 29^2 * 37 * 41 * 53
    # Log10: 3*0.7 + 2*(1.11+1.23+1.46) + 1.57 + 1.61 + 1.72
    # = 2.1 + 2*3.8 + 4.9 = 2.1 + 7.6 + 4.9 = 14.6. Still room.
    
    # Try 5^4 * 13^2 * 17^2 * 29^2 * 37 * 41 * 53
    # Log10: 4*0.7 + 7.6 + 4.9 = 2.8 + 12.5 = 15.3.
    
    # Try 5^5 * 13^2 * 17^2 * 29^2 * 37 * 41 * 53
    # Log10: 5*0.7 + 12.5 = 3.5 + 12.5 = 16.0.
    
    # Try 5^6 * 13^2 * 17^2 * 29^2 * 37 * 41 * 53
    # Log10: 6*0.7 + 12.5 = 4.2 + 12.5 = 16.7.
    
    # Try 5^7 * 13^2 * 17^2 * 29^2 * 37 * 41 * 53
    # Log10: 7*0.7 + 12.5 = 4.9 + 12.5 = 17.4.
    # d(M) = (7+1)*3*3*3*2*2*2 = 8 * 27 * 8 = 1728.
    # Pairs ~ 1728 / 8 = 216.
    
    # Try 5^7 * 13^3 * 17^2 * 29^2 * 37 * 41
    # Log10: 4.9 + 3*1.11 + 2*1.23 + 2*1.46 + 1.57 + 1.61
    # = 4.9 + 3.33 + 2.46 + 2.92 + 3.18 = 16.79.
    # d(M) = 8 * 4 * 3 * 3 * 2 * 2 = 8 * 4 * 3 * 3 * 4 = 1152. Worse.
    
    # Try 5^7 * 13^2 * 17^3 * 29^2 * 37 * 41
    # d(M) = 8 * 3 * 4 * 3 * 2 * 2 = 1152. Worse.
    
    # It seems 5^7 * 13^2 * 17^2 * 29^2 * 37 * 41 * 53 is a strong candidate.
    # Let's check if we can add 61.
    # 5^7 * 13^2 * 17^2 * 29^2 * 37 * 41 * 53 * 61
    # Log10: 17.4 + 1.78 = 19.18. Too big.
    
    # Can we reduce exponent of 5 to add 61?
    # 5^6 * 13^2 * 17^2 * 29^2 * 37 * 41 * 53 * 61
    # Log10: 16.7 + 1.78 = 18.48. Too big.
    # 5^5 * 13^2 * 17^2 * 29^2 * 37 * 41 * 53 * 61
    # Log10: 16.0 + 1.78 = 17.78. Fits!
    # d(M) = (5+1)*3*3*3*2*2*2*2 = 6 * 27 * 16 = 2592.
    # Pairs ~ 2592 / 8 = 324.
    
    # Can we go higher?
    # 5^5 * 13^3 * 17^2 * 29^2 * 37 * 41 * 53
    # Log10: 3.5 + 3.33 + 2.46 + 2.92 + 1.57 + 1.61 + 1.72 = 17.11.
    # d(M) = 6 * 4 * 3 * 3 * 2 * 2 * 2 = 6 * 4 * 3 * 3 * 8 = 1728. Worse.
    
    # 5^5 * 13^2 * 17^3 * 29^2 * 37 * 41 * 53
    # d(M) = 6 * 3 * 4 * 3 * 2 * 2 * 2 = 1728. Worse.
    
    # 5^5 * 13^2 * 17^2 * 29^3 * 37 * 41 * 53
    # d(M) = 6 * 3 * 3 * 4 * 2 * 2 * 2 = 1728. Worse.
    
    # So 5^5 * 13^2 * 17^2 * 29^2 * 37 * 41 * 53 * 61 looks best so far.
    # Let's try adding 73 instead of 61? No, 61 is smaller.
    # What if we drop 53 and add 61, 73?
    # 5^5 * 13^2 * 17^2 * 29^2 * 37 * 41 * 61 * 73
    # Log10: 3.5 + 2.22 + 2.46 + 2.92 + 1.57 + 1.61 + 1.78 + 1.86 = 17.92.
    # d(M) = 6 * 3 * 3 * 3 * 2 * 2 * 2 * 2 = 2592. Same d(M), larger M.
    # So keeping 53 is better (smaller M, same d).
    
    # What if we increase 5 to 6?
    # 5^6 * 13^2 * 17^2 * 29^2 * 37 * 41 * 53 * 61
    # Log10: 4.2 + 12.5 (rest) = 16.7 + 1.78 (61) = 18.48. Too big.
    
    # So M = 5^5 * 13^2 * 17^2 * 29^2 * 37 * 41 * 53 * 61
    # N = M (since Q=1)
    
    M_candidate = 1
    M_candidate *= 5**5
    M_candidate *= 13**2
    M_candidate *= 17**2
    M_candidate *= 29**2
    M_candidate *= 37
    M_candidate *= 41
    M_candidate *= 53
    M_candidate *= 61
    
    if M_candidate > 10**18:
        # Fallback to a smaller one
        M_candidate = 5**4 * 13**2 * 17**2 * 29**2 * 37 * 41 * 53 * 61
    
    N = M_candidate
    
    # Now we need to find all pairs (a,b) such that a+b=N, a,b powerful.
    # We know that if x^2 + y^2 = N, then a=N*x^2, b=N*y^2 is NOT what we want.
    # We want a+b=N.
    # If N = M, and M = x^2 + y^2, then a=x^2, b=y^2 is a pair of powerful numbers.
    # So we just need to find all (x,y) such that x^2 + y^2 = N.
    # Then a=x^2, b=y^2.
    # Since x^2 and y^2 are squares, they are powerful.
    # And x^2 + y^2 = N.
    # So the number of pairs is exactly the number of representations of N as sum of two squares.
    
    pairs = []
    # Iterate x from 1 to sqrt(N/2)
    # x^2 <= N/2 => x <= sqrt(N/2)
    limit_x = int(math.isqrt(N // 2))
    
    # Optimization: N is fixed.
    # We can iterate x and check if N - x^2 is a square.
    
    # To speed up, we can use the fact that N is composed of primes 1 mod 4.
    # But simple iteration is O(sqrt(N)) which is 10^9. Too slow.
    
    # We must use the structure of N to generate representations.
    # Since N = prod p_i^{e_i}, we can use the formula for sum of two squares.
    # However, generating the actual (x,y) pairs from the prime factorization
    # is complex (Gaussian integers).
    
    # Alternative: Since we have a fixed N, we can pre-compute the pairs
    # and hard-code them? No, we must generate them.
    
    # But wait, the evaluation is 150s.
    # Iterating 10^9 steps in Python is impossible.
    
    # We need a faster way to find (x,y).
    # We can use the property that the number of representations is d(N).
    # We can generate the representations using the Gaussian integer factorization.
    # Or, we can simply iterate x up to a smaller limit if we can? No.
    
    # Let's try to generate the pairs using the recursive formula for sum of two squares.
    # If N = p1^e1 ... pk^ek, and p_i = a_i^2 + b_i^2, then the representations of N
    # can be generated by multiplying the Gaussian integers.
    
    # This is a known algorithm.
    # 1. Find a^2+b^2 for each prime p_i.
    # 2. For each p_i, we have (e_i+1) choices for the "power" in the Gaussian sense?
    #    Actually, if p = a+bi, then p^k has k+1 divisors in Z[i].
    #    The number of representations is the product of (e_i+1) for p_i 1 mod 4,
    #    times 1 for p_i 3 mod 4 (if e_i even), else 0.
    #    Here all p_i are 1 mod 4.
    #    So total representations r_2(N) = 4 * prod(e_i+1).
    #    The number of pairs (x,y) with x,y>0, x<=y is r_2(N)/8.
    
    # We need to generate these pairs.
    # We can use the identity:
    # (a^2+b^2)(c^2+d^2) = (ac-bd)^2 + (ad+bc)^2 = (ac+bd)^2 + (ad-bc)^2
    
    # Let's implement a generator for sum of two squares representations.
    
    def sum_of_two_squares_pairs(n):
        """Generate all (x, y) with x^2 + y^2 = n, x <= y, x, y > 0."""
        # Factorize n into primes 1 mod 4
        # For each prime p, find a, b such that p = a^2 + b^2
        # Then combine them.
        
        # Factorization of n
        # n is fixed, so we can hardcode the factorization or compute it once.
        # n = 5^5 * 13^2 * 17^2 * 29^2 * 37 * 41 * 53 * 61
        
        # Primes and their reps:
        # 5 = 1^2 + 2^2
        # 13 = 2^2 + 3^2
        # 17 = 1^2 + 4^2
        # 29 = 2^2 + 5^2
        # 37 = 1^2 + 6^2
        # 41 = 4^2 + 5^2
        # 53 = 2^2 + 7^2
        # 61 = 5^2 + 6^2
        
        # We need to handle powers.
        # (a+bi)^k.
        # We can just do iterative multiplication of Gaussian integers.
        
        # Start with 1 + 0i
        # For each prime power p^e, multiply by (a+bi)^e.
        # But we need all combinations?
        # No, the representations are generated by the product of the divisors in Z[i].
        
        # A simpler way:
        # The set of representations is the set of all products of the form:
        # prod (a_i + b_i i)^{k_i} where 0 <= k_i <= e_i.
        # And their conjugates and rotations.
        
        # Actually, the number of representations is 4 * prod(e_i+1).
        # We can generate the "base" representations and then apply symmetries.
        
        # Let's generate all Gaussian integers z such that |z|^2 = n.
        # z = prod (a_i + b_i i)^{k_i}
        
        # We'll use a DFS to generate all products.
        
        primes_info = [
            (5, 1, 2, 5),   # p, a, b, e
            (13, 2, 3, 2),
            (17, 1, 4, 2),
            (29, 2, 5, 2),
            (37, 1, 6, 1),
            (41, 4, 5, 1),
            (53, 2, 7, 1),
            (61, 5, 6, 1)
        ]
        
        # Precompute powers of (a+bi)
        powers = []
        for p, a, b, e in primes_info:
            pw = [(1, 0)]
            for _ in range(e):
                last_re, last_im = pw[-1]
                # (last_re + last_im*i) * (a + b*i)
                new_re = last_re * a - last_im * b
                new_im = last_re * b + last_im * a
                pw.append((new_re, new_im))
            powers.append(pw)
            
        # Now combine
        # We want all products of one element from each pw list.
        # Number of products = prod(e_i+1) = 6*3*3*3*2*2*2*2 = 2592.
        
        # We can do this iteratively.
        current_products = [(1, 0)]
        for pw in powers:
            new_products = []
            for re1, im1 in current_products:
                for re2, im2 in pw:
                    # (re1 + im1*i) * (re2 + im2*i)
                    new_re = re1 * re2 - im1 * im2
                    new_im = re1 * im2 + im1 * re2
                    new_products.append((new_re, new_im))
            current_products = new_products
            
        # Now current_products contains all z such that |z|^2 = n.
        # There are 2592 of them.
        # Each z = x + yi gives a representation x^2 + y^2 = n.
        # We need x, y > 0 and x <= y.
        
        seen = set()
        result = []
        for x, y in current_products:
            if x == 0 or y == 0:
                continue
            # Normalize to x > 0, y > 0
            # The product can be in any quadrant.
            # We take absolute values?
            # No, (x,y) and (-x,y) are different representations in Z[i] but same in R^2.
            # The set of z generated includes all rotations and reflections?
            # No, it generates specific z.
            # However, if z is a solution, so is -z, iz, -iz, conjugate(z).
            # Our generation via (a+bi)^k might not cover all symmetries directly?
            # Actually, the set of divisors of n in Z[i] has size 4 * prod(e_i+1).
            # My product generation gives prod(e_i+1) elements.
            # These are the "fundamental" ones?
            # Let's just take absolute values and check.
            
            ax, ay = abs(x), abs(y)
            if ax == 0 or ay == 0:
                continue
            if ax > ay:
                ax, ay = ay, ax
            if (ax, ay) not in seen:
                seen.add((ax, ay))
                result.append((ax, ay))
                
        return result

    # Get pairs
    reps = sum_of_two_squares_pairs(N)
    
    # Convert to [a, b] where a=x^2, b=y^2
    # Wait, a+b=N. If x^2+y^2=N, then a=x^2, b=y^2.
    # a and b are powerful.
    
    final_pairs = []
    for x, y in reps:
        a = x * x
        b = y * y
        if a <= b:
            final_pairs.append([a, b])
            
    # Sort and deduplicate just in case
    final_pairs = sorted(list(set(tuple(p) for p in final_pairs)))
    final_pairs = [list(p) for p in final_pairs]
    
    # Ensure N is valid
    if N < 2 or N > 10**18:
        N = 5
        final_pairs = [[1, 4]]
        
    _CACHE = {'N': N, 'pairs': final_pairs}
    return _CACHE

if __name__ == "__main__":
    res = construct()
    print(res['N'])
    print(len(res['pairs']))
