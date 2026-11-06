import time
import math
import random

_START = time.time()
_HARD_DEADLINE = _START + 80.0  # Keep safety margin

def _init_spf(limit):
    spf = list(range(limit + 1))
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

# Max sum of two k_i. If B=60, max k=166, max sum=332.
# If B is small, max k is larger.
# If B=1, max k=10000, max sum=20000.
# We will precompute up to 20000 to be safe for any base.
SPF_LIMIT = 20000
_spf = _init_spf(SPF_LIMIT)

def _prime_factors(num):
    if num <= 1:
        return frozenset()
    out = set()
    x = num
    while x > 1:
        p = _spf[x]
        out.add(p)
        while x % p == 0:
            x //= p
    return frozenset(out)

def construct():
    deadline = time.time() + 80.0
    n = 12
    max_val = 10000
    
    best_overall = None
    best_score_overall = 10**9
    
    # Candidate bases composed of small primes
    # B=60 (2,3,5), B=210 (2,3,5,7), B=30 (2,3,5), B=120 (2,3,5), B=180 (2,3,5)
    bases = [60, 120, 180, 210, 300, 420, 30, 12, 24, 48, 96, 192, 384, 768, 1536, 3072, 6144]
    
    # Add some random bases? No, structured is better.
    
    for base in bases:
        base_factors = _prime_factors(base)
        max_k = max_val // base
        
        if max_k < n:
            continue
            
        # Local Search for k_i
        # Initial guess: 1, 2, ..., n
        cur_k = list(range(1, n + 1))
        
        def get_score(k_list):
            union = set(base_factors)
            # Optimization: precompute factors for all possible sums?
            # Sum range: 1+1=2 to max_k+max_k
            # We just factor on the fly, it's fast with SPF.
            for i in range(n - 1):
                for j in range(i + 1, n):
                    s = k_list[i] + k_list[j]
                    union |= _prime_factors(s)
            return len(union)
        
        cur_score = get_score(cur_k)
        
        # Hill Climbing
        improved = True
        iterations = 0
        max_iterations = 500
        
        while improved and iterations < max_iterations and time.time() < deadline - 5.0:
            improved = False
            iterations += 1
            
            # Try replacing each k_i
            for i in range(n):
                if time.time() > deadline - 5.0:
                    break
                old_k = cur_k[i]
                best_new_k = old_k
                best_new_score = cur_score
                
                # Search space for new_k: 1..max_k
                # To speed up, we can sample or use a restricted pool.
                # But max_k is at most 10000 (if base=1).
                # If base=60, max_k=166. Looping 166 times is fast.
                # If base=1, max_k=10000. Looping 10000 * 12 * 500 is too slow.
                # So, if max_k is large, we should restrict search.
                
                if max_k > 200:
                    # Use a pool of "good" candidates
                    # Candidates: smooth numbers? Or just random?
                    # Let's use a mix of small numbers and random.
                    candidates = list(range(1, 100))
                    # Add some randoms
                    candidates += [random.randint(1, max_k) for _ in range(100)]
                else:
                    candidates = range(1, max_k + 1)
                    
                for new_k in candidates:
                    if new_k == old_k:
                        continue
                    if new_k in cur_k:
                        continue
                    
                    cur_k[i] = new_k
                    new_score = get_score(cur_k)
                    
                    if new_score < best_new_score:
                        best_new_score = new_score
                        best_new_k = new_k
                    elif new_score == best_new_score and new_k != old_k:
                        # Tie-breaking: prefer smaller k? Or random?
                        pass
                    cur_k[i] = old_k
                
                if best_new_k != old_k:
                    cur_k[i] = best_new_k
                    cur_score = best_new_score
                    improved = True
                    # Restart i-loop? Or continue?
                    # Continuing allows finding multiple improvements in one pass.
                    # But restarting is safer for convergence.
                    # Let's restart i-loop to ensure we don't miss better global moves.
                    if cur_score < best_score_overall:
                        best_score_overall = cur_score
                        best_overall = [base * k for k in cur_k]
                    break # Break i-loop, restart while loop
            
            if not improved:
                # Stuck. Try a random restart or perturbation
                if iterations < max_iterations // 2:
                    # Random perturbation
                    for _ in range(5):
                        idx = random.randint(0, n - 1)
                        old = cur_k[idx]
                        new = random.randint(1, max_k)
                        if new not in cur_k:
                            cur_k[idx] = new
                            if get_score(cur_k) < cur_score:
                                cur_score = get_score(cur_k)
                                improved = True
                                break
                            else:
                                cur_k[idx] = old
                else:
                    break
        
        # Update global best
        if cur_score < best_score_overall:
            best_score_overall = cur_score
            best_overall = [base * k for k in cur_k]
            
        if best_score_overall <= 3:
            break # 3 is very good (primes 2,3,5). 2 is likely impossible for n=12.

    if best_overall is None:
        # Fallback: just use small numbers
        best_overall = list(range(1, 13))

    return best_overall
