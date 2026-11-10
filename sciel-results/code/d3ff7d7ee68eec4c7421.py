import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    limit = 10**18
    
    best_n = 2
    best_score = 0
    best_tau = 0
    
    # To speed up, we can prune branches that cannot exceed current best_tau
    # But we need to calculate score, not just tau.
    # So we will collect candidates with high tau, then evaluate score.
    
    candidates = []
    
    def search(idx, current_n, max_exp, current_tau):
        nonlocal best_n, best_score, best_tau, deadline
        
        if time.time() > deadline:
            return
            
        if idx >= len(primes):
            return
            
        p = primes[idx]
        
        # Try exponents from max_exp down to 0
        # We can limit the number of candidates we store to avoid memory issues
        # But the number of valid (n, tau) pairs is small.
        
        # Optimization: If current_tau * (something) < best_tau, maybe skip?
        # Not easy to bound score directly.
        
        # We will just collect all valid n with high tau.
        # How many such n are there? Very few.
        
        # Let's add the current state if idx > 0 or current_n > 1
        # Actually, we should add leaves or significant nodes.
        
        # To ensure we find the max tau, we just traverse.
        # The number of nodes is small.
        
        for e in range(max_exp, -1, -1):
            new_n = current_n * (p ** e)
            if new_n > limit:
                continue
                
            new_tau = current_tau * (e + 1)
            
            # If we have reached a "deep" enough level or just track all
            # We want to evaluate score for the top candidates.
            
            # Store candidate
            # We can't store millions, but the count is likely < 1000
            candidates.append((new_tau, new_n))
            
            # Recurse
            if idx + 1 < len(primes):
                # Next max exponent is e (non-increasing)
                # Also, we can stop if p^(e+1) * current_n > limit
                search(idx + 1, new_n, e, new_tau)
                
            # Check time
            if time.time() > deadline:
                return

    # Start search
    # Max exponent for 2 is 60
    search(0, 1, 60, 1)
    
    # Sort candidates by tau descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Evaluate score for top K candidates
    # K should be enough to cover the "best" score.
    # Let's take top 100.
    top_candidates = candidates[:100]
    
    for tau, n in top_candidates:
        if time.time() > deadline:
            break
            
        score = calc_score(n)
        if score > best_score:
            best_score = score
            best_n = n
            best_tau = tau
            
    return best_n

def calc_score(n):
    # Generate divisors
    divs = [1]
    # Simple factorization for speed (n is product of small primes)
    # Or just trial division up to sqrt(n)
    # Since n is highly composite, it has many small factors.
    
    temp_n = n
    factors = []
    d = 2
    while d * d <= temp_n:
        if temp_n % d == 0:
            cnt = 0
            while temp_n % d == 0:
                temp_n //= d
                cnt += 1
            factors.append((d, cnt))
        d += 1 if d == 2 else 2 # Skip evens after 2
    
    if temp_n > 1:
        factors.append((temp_n, 1))
        
    # Generate divisors
    for p, e in factors:
        new_divs = []
        for d in divs:
            val = 1
            for i in range(e + 1):
                new_divs.append(d * val)
                val *= p
        divs = new_divs
        
    divs.sort()
    
    score = 0
    for i in range(len(divs) - 1):
        if math.gcd(divs[i], divs[i+1]) == 1:
            score += 1
            
    return score
