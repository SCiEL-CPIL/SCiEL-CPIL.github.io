import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Primes 1 mod 4
    primes = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193, 197, 229, 233, 241, 257, 269, 277, 281, 293, 313, 317, 337, 349, 353, 373, 389, 397, 401, 409, 421, 433, 449, 457, 461, 481, 491, 493, 497, 509, 521, 541, 557, 569, 577, 593, 599, 601, 613, 617, 641, 653, 661, 673, 677, 701, 709, 713, 733, 757, 761, 769, 773, 797, 809, 811, 821, 829, 853, 857, 863, 881, 883, 887, 901, 911, 929, 937, 941, 953, 977, 997]
    
    best_M = 1
    best_score = 1
    best_exponents = []
    
    def dfs(idx, current_M, current_score, max_exp, exponents):
        nonlocal best_M, best_score, best_exponents
        
        if time.time() > deadline:
            return
            
        if current_score > best_score:
            best_score = current_score
            best_M = current_M
            best_exponents = exponents[:]
            
        if idx == len(primes):
            return
            
        p = primes[idx]
        # Max exponent for this prime
        # current_M * p^e <= 10^18
        max_e = int(math.log(10**18 / current_M, p)) if current_M > 0 else 60
        max_e = min(max_e, max_exp)
        
        for e in range(max_e, -1, -1):
            new_M = current_M * (p ** e)
            if new_M > 10**18:
                continue
            new_score = current_score * (e + 1)
            exponents.append(e)
            dfs(idx + 1, new_M, new_score, e, exponents)
            exponents.pop()
            
    dfs(0, 1, 1, 60, [])
    
    # Now find pairs for best_M
    M = best_M
    if M == 1:
        return {'N': 1, 'pairs': []}
        
    # Factorize M using the exponents
    # We need to find a^2 + b^2 = p for each prime p
    # Then combine
    reps = [(1, 0)]
    
    for i, e in enumerate(best_exponents):
        if e == 0:
            continue
        p = primes[i]
        # Find a, b such that a^2 + b^2 = p
        a = None
        b = None
        for x in range(1, int(math.isqrt(p)) + 1):
            y2 = p - x * x
            y = int(math.isqrt(y2))
            if y * y == y2:
                a = x
                b = y
                break
        if a is None:
            continue # Should not happen for primes 1 mod 4
            
        # Update reps
        new_reps = []
        # We need to add e copies of (a, b)
        # This is equivalent to multiplying by (a+ib)^e
        # We can do this iteratively
        current_reps = reps
        for _ in range(e):
            next_reps = []
            for (u, v) in current_reps:
                # (u+iv)(a+ib) = (ua-vb) + i(ub+va)
                new_u = u * a - v * b
                new_v = u * b + v * a
                next_reps.append((new_u, new_v))
            current_reps = next_reps
        reps = current_reps
        
    # Convert to pairs [u^2, v^2]
    pairs = set()
    for u, v in reps:
        if u < 0:
            u = -u
        if v < 0:
            v = -v
        if u == 0 or v == 0:
            continue
        a = u * u
        b = v * v
        if a > b:
            a, b = b, a
        pairs.add((a, b))
        
    pairs_list = sorted(list(pairs))
    
    return {'N': M, 'pairs': pairs_list}
