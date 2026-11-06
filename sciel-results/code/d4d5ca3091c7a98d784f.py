import random
import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin
    
    # Sieve for prime factors up to 20000
    MAX_SUM = 20000
    prime_factors = [[] for _ in range(MAX_SUM + 1)]
    for i in range(2, MAX_SUM + 1):
        if not prime_factors[i]:
            for j in range(i, MAX_SUM + 1, i):
                prime_factors[j].append(i)
    
    def get_prime_count(arr):
        """Returns the number of distinct prime factors for all pairwise sums."""
        primes = set()
        n = len(arr)
        for i in range(n):
            for j in range(i+1, n):
                s = arr[i] + arr[j]
                primes.update(prime_factors[s])
        return len(primes)
    
    # Initialization
    candidates = []
    
    # Candidate 1: Random
    candidates.append(random.sample(range(1, 10001), 12))
    
    # Candidate 2: Small evens
    candidates.append(list(range(2, 26, 2)))
    
    # Candidate 3: Multiples of 6
    candidates.append(list(range(6, 79, 6)))
    
    # Candidate 4: Numbers 1 mod 210
    cand4 = [210*k + 1 for k in range(1, 13) if 210*k + 1 <= 10000]
    if len(cand4) >= 12:
        candidates.append(cand4[:12])
        
    # Candidate 5: Random small numbers
    candidates.append(random.sample(range(1, 100), 12))
    
    # Candidate 6: Multiples of 210
    cand6 = [210 * k for k in range(1, 13)]
    if len(cand6) == 12 and max(cand6) <= 10000:
        candidates.append(cand6)
        
    # Candidate 7: Multiples of 30 (2*3*5)
    cand7 = [30 * k for k in range(1, 13)]
    if len(cand7) == 12 and max(cand7) <= 10000:
        candidates.append(cand7)

    best_arr = None
    best_score = float('inf')
    
    for c in candidates:
        if len(set(c)) != 12: continue
        if any(x < 1 or x > 10000 for x in c): continue
        s = get_prime_count(c)
        if s < best_score:
            best_score = s
            best_arr = list(c)
            
    if best_arr is None:
        # Fallback
        best_arr = list(range(1, 13))
        best_score = get_prime_count(best_arr)

    current_arr = best_arr
    current_score = best_score
    
    # Local Search: Simulated Annealing
    T = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Generate a neighbor
        new_arr = list(current_arr)
        idx = random.randint(0, 11)
        new_val = random.randint(1, 10000)
        
        # Avoid duplicates
        if new_val in new_arr:
            for _ in range(10):
                new_val = random.randint(1, 10000)
                if new_val not in new_arr:
                    break
            if new_val in new_arr:
                continue
                
        new_arr[idx] = new_val
        
        new_score = get_prime_count(new_arr)
        
        if new_score < current_score:
            current_arr = new_arr
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_arr = list(current_arr)
        else:
            delta = new_score - current_score
            if delta == 0:
                if random.random() < 0.5:
                    current_arr = new_arr
            else:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_arr = new_arr
        
        T *= cooling
        if T < 0.1:
            T = 0.1
            
    return best_arr
# EVOLVE-BLOCK-END
