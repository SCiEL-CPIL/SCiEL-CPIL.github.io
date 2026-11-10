import time
import random
import math

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin before 60s kill

    def score(A):
        sums = set()
        prods = set()
        n = len(A)
        for i in range(n):
            ai = A[i]
            for j in range(i, n):
                aj = A[j]
                sums.add(ai + aj)
                prods.add(ai * aj)
        return max(len(sums), len(prods))

    # Generate pool of smooth numbers (2,3,5,7,11-smooth) <= 10000
    primes = [2, 3, 5, 7, 11]
    pool = []
    for i in range(1, 10001):
        x = i
        for p in primes:
            while x % p == 0:
                x //= p
        if x == 1:
            pool.append(i)
            
    # Ensure pool has enough elements
    if len(pool) < 100:
        pool = list(range(1, 10001))

    best_set = None
    best_score = float('inf')
    
    def sa_search(initial_set):
        nonlocal best_set, best_score
        current = initial_set[:]
        current_score = score(current)
        
        if current_score < best_score:
            best_score = current_score
            best_set = current[:]
            
        T = 8.0
        cooling = 0.997
        min_T = 0.3
        
        while time.time() - start_time < timeout:
            idx = random.randint(0, 19)
            old_val = current[idx]
            
            # Pick new value from pool
            new_val = random.choice(pool)
            
            if new_val == old_val:
                continue
            if new_val in current:
                continue
                
            new_set = current[:]
            new_set[idx] = new_val
            new_score = score(new_set)
            
            delta = new_score - current_score
            
            if delta < 0:
                current = new_set
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_set = current[:]
            else:
                if T > min_T:
                    prob = math.exp(-delta / T)
                else:
                    prob = 0.0
                if random.random() < prob:
                    current = new_set
                    current_score = new_score
            
            T *= cooling
            if T < min_T:
                T = min_T

    def local_search_polish(start_set):
        """Best-improvement local search to polish a candidate."""
        nonlocal best_set, best_score
        current = start_set[:]
        current_score = score(current)
        
        if current_score < best_score:
            best_score = current_score
            best_set = current[:]
            
        improved = True
        while improved and time.time() - start_time < timeout:
            improved = False
            best_delta = 0
            best_idx = -1
            best_new_val = -1
            
            for idx in range(20):
                old_val = current[idx]
                for new_val in pool:
                    if new_val == old_val:
                        continue
                    if new_val in current:
                        continue
                    
                    new_set = current[:]
                    new_set[idx] = new_val
                    new_score = score(new_set)
                    delta = new_score - current_score
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_idx = idx
                        best_new_val = new_val
            
            if best_delta < 0:
                current[best_idx] = best_new_val
                current_score += best_delta
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_set = current[:]
        
        return current

    # Initial Candidates
    candidates = []
    
    # 1. Arithmetic Progression
    candidates.append(list(range(1, 21)))
    
    # 2. Odd Arithmetic Progression
    candidates.append(list(range(1, 41, 2)))
    
    # 3. First 20 Smooth Numbers
    candidates.append(pool[:20])
    
    # 4. Small geometric-like: 1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192 + extras
    geo_base = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    # Add more small numbers to get to 20
    extra = [3, 5, 6, 7, 9, 10, 11, 12, 13]
    geo_candidate = sorted(set(geo_base + extra))[:20]
    if len(geo_candidate) < 20:
        # Fill from pool
        for x in pool:
            if x not in geo_candidate:
                geo_candidate.append(x)
            if len(geo_candidate) == 20:
                break
    geo_candidate = sorted(geo_candidate[:20])
    candidates.append(geo_candidate)
    
    # 5. Random Smooth Numbers
    for _ in range(5):
        random_set = random.sample(pool, 20)
        candidates.append(sorted(random_set))
        
    # 6. Random Uniform
    for _ in range(3):
        random_set = random.sample(range(1, 10001), 20)
        candidates.append(sorted(random_set))
    
    # 7. Mixed: some small, some geometric
    mixed = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 16, 18, 20, 24, 32, 36, 40, 48, 64, 80]
    candidates.append(mixed)

    # Run SA on each candidate
    for cand in candidates:
        if time.time() - start_time > timeout - 5:
            break
        sa_search(cand)
        
    # Final polish on best set found
    if best_set is not None and time.time() - start_time < timeout - 2:
        local_search_polish(best_set)
        
    return best_set
