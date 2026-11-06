import numpy as np
import time
import random

def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i:n+1:i] = [False] * len(sieve[i*i:n+1:i])
    return [i for i, is_prime in enumerate(sieve) if is_prime]

PRIMES = _primes_up_to(1000)
N_PRIMES = len(PRIMES)
MAX_L_CAP = 5000 # Start with 5000, can increase if needed
X_VEC = np.arange(1, MAX_L_CAP + 1, dtype=np.int64)

def _score(res):
    counts = np.zeros(MAX_L_CAP, dtype=np.int8)
    for i, p in enumerate(PRIMES):
        r = res[i]
        counts += (X_VEC % p == r).astype(np.int8)
    
    fail_indices = np.where(counts < 2)[0]
    if len(fail_indices) > 0:
        return fail_indices[0]
    else:
        return MAX_L_CAP

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initial random
    res = [random.randint(0, p-1) for p in PRIMES]
    best_res = res.copy()
    best_score = _score(res)
    current_score = best_score
    
    no_improve = 0
    T = 10.0 # Temperature for SA
    
    while time.time() < deadline:
        # Calculate counts and find first failure
        counts = np.zeros(MAX_L_CAP, dtype=np.int8)
        for i, p in enumerate(PRIMES):
            r = res[i]
            counts += (X_VEC % p == r).astype(np.int8)
        
        fail_indices = np.where(counts < 2)[0]
        
        if len(fail_indices) > 0:
            fail_idx = fail_indices[0]
            x_fail = fail_idx + 1
            
            # Directed Move
            eligible = [i for i, p in enumerate(PRIMES) if res[i] != x_fail % p]
            
            if eligible:
                i = random.choice(eligible)
                old_r = res[i]
                res[i] = x_fail % PRIMES[i]
                
                new_score = _score(res)
                
                if new_score > current_score:
                    current_score = new_score
                    best_res = res.copy()
                    best_score = current_score
                    no_improve = 0
                else:
                    res[i] = old_r
                    no_improve += 1
                    
                    # If stuck, try SA step
                    if no_improve > 10:
                        i_rand = random.randint(0, N_PRIMES - 1)
                        old_r_rand = res[i_rand]
                        res[i_rand] = random.randint(0, PRIMES[i_rand] - 1)
                        
                        new_score_rand = _score(res)
                        
                        if new_score_rand > current_score:
                            current_score = new_score_rand
                            best_res = res.copy()
                            best_score = current_score
                            no_improve = 0
                        else:
                            # Accept worse move with prob
                            delta = new_score_rand - current_score
                            if random.random() < np.exp(delta / T):
                                current_score = new_score_rand
                                # Don't update best_res if worse
                            else:
                                res[i_rand] = old_r_rand
                            
                        T *= 0.95
                        no_improve = 0
            else:
                # Should not happen
                pass
        else:
            # Covered MAX_L_CAP
            # Try to increase MAX_L_CAP?
            # For now, break
            break
            
    return best_res
