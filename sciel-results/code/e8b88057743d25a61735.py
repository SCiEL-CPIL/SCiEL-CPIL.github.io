import math
import time
import sys
from itertools import product

def construct():
    start_time = time.time()
    DEADLINE = 148.0
    LIMIT = 10**18

    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes_pool = sieve(1000)
    
    # Strategy:
    # 1. The number of distinct prime factors (omega(n)) is crucial.
    #    For n <= 10^18, the maximum omega(n) is 15 (product of first 15 primes ~ 6.1e17).
    #    Product of first 16 primes is ~ 3.2e19, which is too large.
    #    So max distinct primes is 15.
    
    # 2. For a fixed set of primes P, we want to choose exponents e_i.
    #    Generally, e_i = 1 is optimal for large primes.
    #    For small primes (2, 3, 5...), higher exponents might be allowed if the product stays <= LIMIT.
    #    However, increasing exponents usually decreases tau_perp because it creates more non-coprime adjacencies.
    #    Let's verify if square-free is always optimal.
    #    If n = p*q, score 2.
    #    If n = p^2*q, divisors: 1, p, p^2, q, pq, p^2q.
    #    Pairs: (1,p)=1, (p,p^2)=0, (p^2,q)=1, (q,pq)=0, (pq,p^2q)=0. Score 2.
    #    Same score, but n is larger. So square-free is at least as good.
    #    If n = p^3*q, divisors: 1, p, p^2, p^3, q, pq, p^2q, p^3q.
    #    Sorted: 1, p, p^2, p^3, q, pq, p^2q, p^3q (assuming p < q and p^3 < q? No, usually p^3 might be > q or < q).
    #    If p=2, q=3. 1, 2, 4, 8, 3, 6, 12, 24.
    #    Sorted: 1, 2, 3, 4, 6, 8, 12, 24.
    #    (1,2)=1, (2,3)=1, (3,4)=1, (4,6)=0, (6,8)=0, (8,12)=0, (12,24)=0. Score 3.
    #    Square-free 2*3=6. Divisors 1,2,3,6. Score 2.
    #    Wait, 2^3*3 = 24. Score 3.
    #    So exponents CAN help!
    
    # Let's re-evaluate the "square-free is optimal" hypothesis.
    # For n=2^3*3, score 3.
    # For n=2*3*5=30. Divisors: 1,2,3,5,6,10,15,30.
    # (1,2)=1, (2,3)=1, (3,5)=1, (5,6)=1, (6,10)=0, (10,15)=0, (15,30)=0. Score 4.
    # So 30 is better than 24.
    
    # What about n=2^2*3*5 = 60?
    # Divisors: 1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60.
    # (1,2)=1, (2,3)=1, (3,4)=1, (4,5)=1, (5,6)=1, (6,10)=0, (10,12)=0, (12,15)=0, (15,20)=0, (20,30)=0, (30,60)=0.
    # Score 5.
    # So 60 > 30.
    
    # It seems using exponents on small primes allows "filling the gaps" with coprime adjacencies.
    # Specifically, the sequence 1, 2, 3, 4, 5, 6... has many coprime adjacencies.
    # The "density" of coprime adjacencies in the divisor list is maximized when the divisors are "dense" at the low end.
    # This suggests we should maximize the number of small divisors.
    # This implies using small primes with exponents.
    
    # New Strategy:
    # Search over combinations of exponents for the first K primes.
    # K=15 is max distinct primes.
    # But if we use exponents, we might need fewer distinct primes to stay within LIMIT?
    # No, small primes with exponents grow slower than adding new large primes?
    # 2^10 = 1024. 2*3*5*7*11*13 = 30030.
    # Actually, the product of the first 15 primes is 6.1e17.
    # We have ~1.6e17 budget left.
    # We can multiply by 2^3 (8) and stay under limit? 6.1e17 * 8 = 4.8e18 > 1e18.
    # So we can't just add exponents to the primorial.
    
    # We need to search the space of (p_1^{e_1} * ... * p_k^{e_k}) <= 10^18.
    # This is a knapsack-like problem.
    
    # Efficient Evaluation:
    # Instead of generating all divisors for every candidate, we can pre-compute?
    # No, the set of divisors changes with exponents.
    # But the number of candidates is small if we restrict to first 15 primes.
    # Max exponent for 2: 10^18 / (3*5*...*47) ~ 1.6e17 / 6.1e17 < 1.
    # So if we use all 15 primes, all exponents must be 1.
    # If we drop the largest prime (47), we have budget for 47 * 2^e?
    # 6.1e17 / 47 = 1.3e16. 2^e <= 1.3e16 -> e <= 53.
    # But do we want 53 exponents of 2?
    # Divisors of 2^53 * (product of 14 primes).
    # The divisors will be heavily clustered at powers of 2.
    # Likely score drops.
    
    # Let's test a few specific structures.
    # 1. Primorial 15 primes: Score ~ 21.9 (from current program).
    # 2. Primorial 14 primes * 2^k.
    # 3. Primorial 13 primes * 2^a * 3^b.
    
    # Let's write a fast evaluator that generates divisors and sorts them.
    # For k=15, 2^15 = 32768 divisors. Sort is fast.
    # We can afford to check ~1000-5000 candidates in 140s.
    
    def eval_score(n):
        # Get divisors
        divs = [1]
        # Factor n? No, we know the factors from the search.
        # Pass factors to make it faster?
        pass
        
    def eval_score_from_factors(factors):
        """
        factors: list of (prime, exponent)
        """
        divs = [1]
        for p, e in factors:
            new_divs = []
            for d in divs:
                val = d
                for _ in range(e):
                    val *= p
                    new_divs.append(val)
            divs.extend(new_divs)
        
        divs.sort()
        
        score = 0
        for i in range(1, len(divs)):
            if math.gcd(divs[i-1], divs[i]) == 1:
                score += 1
        return score

    # Search Space:
    # We will iterate over the number of distinct primes k from 1 to 15.
    # For each k, we use the first k primes.
    # We need to distribute the "budget" (log limit) among exponents.
    # This is a multi-dimensional knapsack.
    
    # Optimization:
    # The function tau_perp is likely maximized when the divisors are as "dense" as possible at the start.
    # This favors small primes.
    
    # Let's use a recursive search with pruning.
    # State: index of prime, current product, current factors list.
    
    best_score = 0
    best_factors = []
    
    primes_to_use = primes_pool[:15]
    
    def dfs(idx, current_prod, current_factors):
        nonlocal best_score, best_factors
        
        if time.time() > DEADLINE:
            return
            
        # Update best if valid
        if current_prod > 1:
            # Only evaluate if we have a "complete" candidate?
            # No, we can evaluate at any step, but deeper is usually better?
            # Actually, adding more prime factors (even with exp 1) usually increases score.
            # So we should go as deep as possible.
            pass
        
        if idx == len(primes_to_use):
            if current_prod > 1:
                s = eval_score_from_factors(current_factors)
                if s > best_score:
                    best_score = s
                    best_factors = current_factors[:]
            return
        
        p = primes_to_use[idx]
        
        # Try exponents 0 to max possible
        max_exp = 0
        val = current_prod
        while val <= LIMIT:
            if val * p > LIMIT:
                break
            if val == current_prod:
                max_exp += 1
            else:
                break
            val *= p
            
        # Actually, simpler loop:
        val = current_prod
        for e in range(0, 100): # Max exponent for 2 is ~60
            if val > LIMIT:
                break
            if e > 0:
                # Only consider if we include this prime
                # We must include prime idx if we are going to use primes idx+1...
                # So if we pick e=0, we MUST pick 0 for all subsequent primes?
                # No, we can skip a prime and pick a later one?
                # "First k primes" assumption: if we skip p_i, we can't use p_j for j>i?
                # Actually, using non-consecutive primes is suboptimal because larger primes are worse.
                # So we assume we use a prefix of the prime list.
                # If e=0, we stop searching deeper for this branch?
                # Yes, if we skip p_i, we should skip all larger primes to maintain "small primes" property.
                # So if e=0, we evaluate current state and return (don't recurse).
                # Wait, what if we skip 2? No, 2 is always best.
                # So we assume we use p_1, p_2, ..., p_k.
                
                # Recurse to next prime
                current_factors.append((p, e))
                dfs(idx + 1, val, current_factors)
                current_factors.pop()
            else:
                # e=0: Stop here. This is a complete candidate using primes 0..idx-1.
                if current_prod > 1:
                    s = eval_score_from_factors(current_factors)
                    if s > best_score:
                        best_score = s
                        best_factors = current_factors[:]
                # Do not recurse further, as skipping p_i means we don't use p_{i+1} either.
                return
            
            val *= p
            
    # Start DFS
    # We must include at least one prime.
    # The DFS above handles e=0 by stopping.
    # But we need to start with idx=0.
    # If we pick e=0 for p=2, we stop. Score 0 (n=1 invalid).
    # So we must pick e>=1 for p=2.
    
    # Let's refine the DFS to force e>=1 for the first prime, and e>=0 for others,
    # but if e=0, it terminates the chain.
    
    def dfs_refined(idx, current_prod, current_factors):
        nonlocal best_score, best_factors
        
        if time.time() > DEADLINE:
            return
        
        if idx == len(primes_to_use):
            if current_prod > 1:
                s = eval_score_from_factors(current_factors)
                if s > best_score:
                    best_score = s
                    best_factors = current_factors[:]
            return
        
        p = primes_to_use[idx]
        
        # Determine max exponent
        # We can pick e=0 only if we are terminating the sequence.
        # If we pick e=0, we do NOT recurse.
        
        # Case 1: Terminate here (if idx > 0, i.e., we have at least one prime)
        if idx > 0:
            s = eval_score_from_factors(current_factors)
            if s > best_score:
                best_score = s
                best_factors = current_factors[:]
        
        # Case 2: Continue with prime p
        val = current_prod
        for e in range(1, 100):
            if val * p > LIMIT:
                break
            val *= p
            current_factors.append((p, e))
            dfs_refined(idx + 1, val, current_factors)
            current_factors.pop()
            
    dfs_refined(0, 1, [])
    
    if not best_factors:
        # Fallback to primorial 15
        best_factors = [(p, 1) for p in primes_pool[:15]]
        
    # Construct n
    n = 1
    for p, e in best_factors:
        n *= (p ** e)
        
    return n

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
