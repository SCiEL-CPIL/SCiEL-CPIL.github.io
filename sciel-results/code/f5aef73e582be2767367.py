import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    def time_left():
        return deadline - time.time()
    
    # Primes 1 mod 4 and their representations as sum of two squares
    # p = a^2 + b^2
    # We need a, b such that a^2+b^2=p
    def find_sum_sq_rep(p):
        # Simple search for small p
        limit = int(math.isqrt(p)) + 1
        for a in range(1, limit):
            b2 = p - a*a
            b = math.isqrt(b2)
            if b*b == b2:
                return (a, b)
        return (1, 1) # Fallback, shouldn't happen for 1 mod 4 primes

    # List of primes 1 mod 4
    primes_1mod4 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193, 197, 229, 233, 241, 257, 269, 277, 281, 293, 313, 317, 337, 349, 353, 373, 389, 397, 401, 409, 421, 433, 449, 457, 461, 509]
    
    # Precompute representations
    reps = {}
    for p in primes_1mod4:
        reps[p] = find_sum_sq_rep(p)
        
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    best_N = 0
    best_pairs = []
    
    # We want to maximize the number of pairs.
    # Pairs come from representations of M as x^2+y^2.
    # If M = prod p_i^e_i, num_reps = prod (e_i+1).
    # To maximize this under M <= 10^18/Q, we should use small primes with exponent 1.
    
    # Strategy: Pick a set of primes P such that their product <= M_max.
    # Then M = product(P).
    # Number of pairs = 2^(|P|) / 4 = 2^(|P|-2) roughly (considering symmetries and x,y order).
    # Actually, the number of distinct pairs (a,b) with a<=b is roughly 2^(|P|-2).
    
    # Let's try to fit as many small primes 1 mod 4 as possible.
    # Product of first k primes 1 mod 4:
    # 5*13*17*29*37*41*53*61*73*89*97*101*109*113*137*149*157*173*181*193
    # Let's calculate cumulative products to see how many fit in 10^18.
    
    # Q = 1 (S=1) is best to maximize M.
    # So N = M.
    # We can use all primes 1 mod 4.
    
    # Let's select the first K primes such that product <= 10^18.
    prod = 1
    selected_primes = []
    for p in primes_1mod4:
        if prod * p <= 10**18:
            prod *= p
            selected_primes.append(p)
        else:
            break
            
    # If we can't fit more, we might try to use exponents > 1 for the largest primes?
    # No, exponent 1 gives factor (1+1)=2 to the count.
    # Exponent 2 gives factor (2+1)=3. Ratio 3/2 = 1.5 vs cost p^2.
    # Exponent 1 is much more efficient for count per log(M).
    
    # So, M = product of selected_primes.
    # N = M.
    
    # Now we must generate all pairs (x, y) such that x^2 + y^2 = M.
    # We use the Gaussian integer multiplication.
    
    # Each prime p_i = a_i^2 + b_i^2 corresponds to Gaussian integer g_i = a_i + b_i i.
    # The representations of M are given by the real and imaginary parts of products of g_i^e_i and their conjugates.
    # Since e_i = 1 for all, we just distribute each g_i or its conjugate (a_i - b_i i) to the product.
    # There are 2^k combinations.
    
    # However, we also need to account for the fact that (x,y) and (y,x) and (-x,y) etc are the same pair {x^2, y^2}.
    # We just need to generate all unique (x^2, y^2) pairs.
    
    # Let's generate all Gaussian integer products.
    # Start with 1 + 0i.
    # For each prime, multiply by (a+bi) or (a-bi).
    
    # Optimization: We can prune symmetric duplicates.
    
    if not selected_primes:
        # Fallback
        return {'N': 5, 'pairs': [[1, 4]]}
        
    M = prod
    N = M
    
    # Generate pairs
    # state: list of (real, imag)
    # Start with [(1, 0)]
    states = [(1, 0)]
    
    for p in selected_primes:
        a, b = reps[p]
        new_states = []
        for (r, i) in states:
            # Multiply by (a + bi)
            r1 = r * a - i * b
            i1 = r * b + i * a
            new_states.append((r1, i1))
            
            # Multiply by (a - bi)
            r2 = r * a + i * b
            i2 = r * b - i * a
            new_states.append((r2, i2))
            
        # Check time
        if time_left() < 5:
            # If we run out of time, we might not finish. 
            # But 2^k growth is fast.
            # k=20 -> 1 million states. 
            # k=25 -> 33 million states. This might be too slow in Python.
            # We need to limit k.
            pass
        
        states = new_states
        if time_left() < 10:
            break

    # If we broke early, M is smaller than the full product.
    # We should recalculate M based on the primes actually processed?
    # No, the code above just breaks the loop. The `M` variable still holds the full product.
    # This would be invalid.
    
    # Better approach: Determine K first based on time budget or fixed limit.
    # 2^20 = 1M operations. 2^25 = 33M.
    # Let's limit to 20 primes to be safe.
    
    # Restart logic for safety
    selected_primes = selected_primes[:20]
    M = 1
    for p in selected_primes:
        M *= p
    N = M
    
    states = [(1, 0)]
    for p in selected_primes:
        a, b = reps[p]
        new_states = []
        for (r, i) in states:
            r1 = r * a - i * b
            i1 = r * b + i * a
            new_states.append((r1, i1))
            
            r2 = r * a + i * b
            i2 = r * b - i * a
            new_states.append((r2, i2))
        states = new_states
        
    # Extract pairs
    # x = |r|, y = |i|
    # a = x^2, b = y^2
    # Ensure a <= b
    
    seen = set()
    pairs = []
    for (r, i) in states:
        x = abs(r)
        y = abs(i)
        # x^2 + y^2 should be M
        if x*x + y*y != M:
            continue
        a = x*x
        b = y*y
        if a > b:
            a, b = b, a
        if (a, b) not in seen:
            seen.add((a, b))
            pairs.append([a, b])
            
    if len(pairs) > len(best_pairs):
        best_pairs = pairs
        best_N = N
        
    # Fallback if something went wrong
    if not best_pairs:
        best_N = 5
        best_pairs = [[1, 4]]
        
    return {'N': best_N, 'pairs': best_pairs}

if __name__ == "__main__":
    res = construct()
    print(res['N'])
    print(len(res['pairs']))
