import time
import random
import math
import numpy as np

N = 5000
DEADLINE = 320.0

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

PRIMES = _primes(N + 1)

def _is_admissible(S, primes):
    n = len(S)
    for p in primes:
        if p > n:
            break
        residues = set(x % p for x in S)
        if len(residues) == p:
            return False
    return True

def construct():
    start_time = time.time()
    best_S = []
    best_size = 0
    
    # Precompute residue element lists as numpy arrays
    residue_elements = {}
    for p in PRIMES:
        residue_elements[p] = []
        for r in range(p):
            if r <= N:
                residue_elements[p].append(np.arange(r, N + 1, p))
            else:
                residue_elements[p].append(np.array([], dtype=int))
    
    while time.time() - start_time < DEADLINE - 5:
        rng = random.Random()
        
        # Prime ordering: largest first with light randomization
        prime_order = sorted(PRIMES, reverse=True)
        if rng.random() < 0.4:
            for _ in range(max(1, len(prime_order) // 5)):
                i = rng.randint(0, len(prime_order) - 2)
                prime_order[i], prime_order[i + 1] = prime_order[i + 1], prime_order[i]
        
        removed = np.zeros(N + 1, dtype=bool)
        residue_choices = {}
        
        for p in prime_order:
            best_r = -1
            best_marginal = N + 1
            for r in range(p):
                elems = residue_elements[p][r]
                if len(elems) == 0:
                    marginal = 0
                else:
                    marginal = int(np.count_nonzero(~removed[elems]))
                if marginal < best_marginal:
                    best_marginal = marginal
                    best_r = r
                if marginal == 0:
                    break
            
            residue_choices[p] = best_r
            if best_r is not None:
                elems = residue_elements[p][best_r]
                if len(elems) > 0:
                    removed[elems] = True
        
        S = [x for x in range(N + 1) if not removed[x]]
        size = len(S)
        
        if size > best_size:
            best_size = size
            best_S = S
        
        # Local search
        if time.time() - start_time < DEADLINE - 10:
            ls_time = min(10.0, DEADLINE - 10 - (time.time() - start_time))
            if ls_time > 1.0:
                current_removed = removed.copy()
                current_size = size
                ls_start = time.time()
                improved = True
                
                while improved and (time.time() - ls_start) < ls_time:
                    improved = False
                    trial_primes = [p for p in PRIMES if p <= 500]
                    rng.shuffle(trial_primes)
                    
                    for p in trial_primes:
                        if (time.time() - ls_start) > ls_time:
                            break
                        old_r = residue_choices[p]
                        for _ in range(5):
                            new_r = rng.randint(0, p - 1)
                            if new_r == old_r:
                                continue
                            old_elems = residue_elements[p][old_r]
                            new_elems = residue_elements[p][new_r]
                            if len(old_elems) > 0:
                                current_removed[old_elems] = False
                            if len(new_elems) > 0:
                                current_removed[new_elems] = True
                            new_size = N + 1 - int(np.count_nonzero(current_removed))
                            if new_size > current_size:
                                residue_choices[p] = new_r
                                current_size = new_size
                                improved = True
                                break
                            else:
                                if len(new_elems) > 0:
                                    current_removed[new_elems] = False
                                if len(old_elems) > 0:
                                    current_removed[old_elems] = True
                        if improved:
                            continue
                    if not improved:
                        break
                
                if current_size > best_size:
                    best_size = current_size
                    best_S = [x for x in range(N + 1) if not current_removed[x]]
        
        if time.time() - start_time > DEADLINE - 5:
            break
    
    return best_S
