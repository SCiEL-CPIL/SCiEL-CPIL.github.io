import math
import time
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Step 1: Find optimal M composed of primes 1 mod 4
    # Primes 1 mod 4
    primes_1_4 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193, 197, 229, 233, 241, 257, 269, 277, 281, 293, 313, 317, 337, 349, 353, 373, 389, 397, 401, 409, 421, 433, 449, 457, 461, 509, 521, 541, 557, 569, 577, 593, 601, 613, 617, 641, 653, 661, 673, 677, 701, 709, 733, 757, 761, 769, 773, 797, 809, 821, 829, 853, 857, 877, 881, 929, 937, 941, 953, 977, 997]
    
    # Log budget
    max_log = math.log(10**18)
    
    # DP to maximize product of (e+1)
    # We use a dictionary for DP to handle sparse states or just array if budget is small enough
    # Budget in units of 0.001 log?
    # Let's use exact log comparison in a recursive search with memoization, 
    # but the state space of "remaining log" is continuous.
    # Better: Since we only have ~30 primes and max exponent ~25, 
    # we can do a DFS with pruning.
    
    best_product = 1
    best_exponents = [0] * len(primes_1_4)
    
    def dfs(idx, current_log, current_product):
        nonlocal best_product, best_exponents
        if time.time() > deadline - 20:
            return
        
        # Pruning: even if we use max possible product from remaining primes, can we beat best?
        # Max possible product from remaining primes: 
        # Assume we can use all remaining budget on the most efficient prime (which is 5, but we are past it).
        # Actually, just check if current_product * (max_possible_future) < best_product.
        # A simple upper bound: assume we can multiply by 2 for each remaining prime (e=1).
        # This is weak.
        
        if idx == len(primes_1_4):
            if current_product > best_product:
                best_product = current_product
                # Reconstruct exponents? We need to store the path.
                # This is tricky with nonlocal. Let's store the path in the recursion or use a different approach.
                pass
            return
            
        p = primes_1_4[idx]
        log_p = math.log(p)
        
        # Max exponent we can use
        max_e = int((max_log - current_log) // log_p)
        
        # Try exponents from max_e down to 0
        for e in range(max_e, -1, -1):
            new_log = current_log + e * log_p
            if new_log > max_log:
                continue
            new_product = current_product * (e + 1)
            
            # Store best
            if new_product > best_product:
                # We need to keep track of exponents.
                # Let's use a global list and backtrack.
                pass
            
            dfs(idx + 1, new_log, new_product)
            
    # The DFS above doesn't track exponents properly for the best solution.
    # Let's use a DP table where we store the best product and the exponents.
    # Since max_log is ~41, and we have ~30 primes, 
    # we can discretize log? No, log is irrational.
    
    # Alternative: Since the number of primes is small, we can just iterate over all combinations?
    # No, $25^{30}$ is too big.
    
    # Let's use the property that we should use small primes with higher exponents.
    # We can use a greedy approach with local search?
    # Or, we can just use the first K primes with exponent 1, and then try to increase exponents of the smallest primes.
    
    # Let's try a simpler heuristic:
    # Start with all exponents 0.
    # While we have budget, add 1 to the exponent of the prime that gives the best "gain per log cost".
    # Gain: $(e+1)/(e)$ is not quite right.
    # We want to maximize $\prod (e_i+1)$.
    # $\ln(\text{Product}) = \sum \ln(e_i+1)$.
    # We want to maximize $\sum \ln(e_i+1)$ subject to $\sum e_i \ln p_i \le B$.
    # This is a fractional knapsack if we relax integrality, but we need integers.
    # The marginal gain of increasing $e_i$ by 1 is $\ln(e_i+2) - \ln(e_i+1) = \ln(1 + 1/(e_i+1))$.
    # The cost is $\ln p_i$.
    # Ratio: $\frac{\ln(1 + 1/(e_i+1))}{\ln p_i}$.
    # We should always pick the prime with the highest ratio.
    
    exponents = [0] * len(primes_1_4)
    current_log = 0.0
    
    while True:
        best_idx = -1
        best_ratio = 0
        for i in range(len(primes_1_4)):
            e = exponents[i]
            # If we increase e by 1, new log cost is log_p
            # Gain in log product is log(e+2) - log(e+1)
            gain = math.log(e + 2) - math.log(e + 1)
            cost = math.log(primes_1_4[i])
            if current_log + cost <= max_log:
                ratio = gain / cost
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_idx = i
        
        if best_idx == -1:
            break
        
        exponents[best_idx] += 1
        current_log += math.log(primes_1_4[best_idx])
        
    # Construct M
    M = 1
    for i, p in enumerate(primes_1_4):
        if exponents[i] > 0:
            M *= (p ** exponents[i])
            
    if M > 10**18:
        M = 10**18 # Fallback, though greedy should prevent this
        
    # Step 2: Generate representations of M as sum of two squares
    # We need to factor M into primes 1 mod 4.
    # We already have the exponents.
    
    # For each prime p^e, find representations x^2+y^2=p^e.
    # p = a^2+b^2.
    # We can generate them recursively.
    
    def get_prime_reps(p, e):
        # Find a,b such that a^2+b^2=p
        a = 0
        b = 0
        for i in range(1, int(math.isqrt(p)) + 1):
            j2 = p - i*i
            j = int(math.isqrt(j2))
            if j*j == j2:
                a = i
                b = j
                break
        # Now generate reps for p^e
        # Base case: p^1 -> (a,b)
        # p^e = (a+bi)^e * (a-bi)^e
        # We can just multiply complex numbers.
        reps = [(1, 0)]
        for _ in range(e):
            new_reps = []
            for x, y in reps:
                # Multiply (x+yi) by (a+bi) and (a-bi)
                # (x+yi)(a+bi) = (xa-yb) + (xb+ya)i
                # (x+yi)(a-bi) = (xa+yb) + (ya-xb)i
                r1 = (x*a - y*b, x*b + y*a)
                r2 = (x*a + y*b, y*a - x*b)
                new_reps.append(r1)
                new_reps.append(r2)
            reps = new_reps
        # Filter unique and normalize
        unique_reps = set()
        for x, y in reps:
            # We want x^2+y^2=p^e
            # Normalize: x,y can be negative. We just need the set of (x,y) up to sign and swap.
            # Actually, we will combine all primes later.
            unique_reps.add((x, y))
        return list(unique_reps)

    # Combine representations from all prime powers
    all_reps = [(1, 0)]
    for i, p in enumerate(primes_1_4):
        e = exponents[i]
        if e == 0:
            continue
        p_reps = get_prime_reps(p, e)
        new_all_reps = []
        for x1, y1 in all_reps:
            for x2, y2 in p_reps:
                # Multiply (x1+y1i) and (x2+y2i)
                x_new = x1*x2 - y1*y2
                y_new = x1*y2 + y1*x2
                new_all_reps.append((x_new, y_new))
        all_reps = new_all_reps
        
    # Extract pairs
    pairs = set()
    for x, y in all_reps:
        if x == 0 or y == 0:
            continue
        a = x*x
        b = y*y
        if a > b:
            a, b = b, a
        if a + b == M:
            pairs.add((a, b))
            
    if not pairs:
        # Fallback
        return {'N': 5, 'pairs': [[1, 4]]}
        
    N = M
    pair_list = [[a, b] for a, b in pairs]
    
    return {'N': N, 'pairs': pair_list}

if __name__ == "__main__":
    res = construct()
    print(res['N'])
    print(len(res['pairs']))
