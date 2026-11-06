import time
import random
import math

def construct():
    n = 20
    max_val = 10000
    
    def score(A):
        sums = set()
        prods = set()
        for a in A:
            for b in A:
                sums.add(a + b)
                prods.add(a * b)
        return max(len(sums), len(prods))
    
    def score_detailed(A):
        sums = set()
        prods = set()
        for a in A:
            for b in A:
                sums.add(a + b)
                prods.add(a * b)
        return max(len(sums), len(prods)), len(sums), len(prods)
    
    # Generate candidate pool: focus on highly composite/smooth numbers
    # and small integers to maximize collision opportunities
    candidates = set()
    
    # Small integers
    for i in range(1, 200):
        candidates.add(i)
    
    # Smooth numbers: products of small primes with limited exponents
    primes = [2, 3, 5, 7, 11, 13]
    
    # Generate 7-smooth numbers up to max_val
    def gen_smooth(idx, current):
        if current > max_val:
            return
        if current > 0:
            candidates.add(current)
        if idx < len(primes):
            p = primes[idx]
            val = current
            while val <= max_val:
                gen_smooth(idx + 1, val)
                val *= p
    
    gen_smooth(0, 1)
    
    # Multiples of small numbers for additional coverage
    for i in range(1, 150):
        for mult in [2, 3, 4, 5, 6, 8, 9, 10, 12]:
            v = i * mult
            if v <= max_val:
                candidates.add(v)
    
    # Powers of 2 and 3
    for e in range(1, 15):
        if 2**e <= max_val:
            candidates.add(2**e)
        if 3**e <= max_val:
            candidates.add(3**e)
    
    candidate_list = sorted([x for x in candidates if 1 <= x <= max_val])
    
    # Limit candidate list to keep search tractable
    # Focus on smaller numbers which have more collision potential
    if len(candidate_list) > 300:
        # Keep all small numbers and a subset of larger ones
        candidate_list = [x for x in candidate_list if x <= 500]
    
    # Greedy construction with collision-maximizing selection
    best_A = None
    best_score = float('inf')
    
    start_time = time.time()
    deadline = start_time + 75  # Leave 15s safety margin
    
    # Try several different starting strategies
    start_strategies = []
    for s in [1, 2, 3, 4, 6, 8, 12, 24, 48, 72]:
        start_strategies.append([s])
    for _ in range(5):
        start_strategies.append([random.choice(candidate_list[:50])])
    
    for start in start_strategies:
        if time.time() > deadline - 15:
            break
        
        current = start[:]
        current_set = set(current)
        
        # Precompute current sums and prods for efficiency
        while len(current) < n and time.time() < deadline - 15:
            best_cand = None
            best_collisions = -1
            
            # Compute current sums and prods once
            current_sums = set()
            current_prods = set()
            for a in current:
                for b in current:
                    current_sums.add(a + b)
                    current_prods.add(a * b)
            
            for cand in candidate_list:
                if cand in current_set:
                    continue
                
                # Count collisions: how many new sums/prods are already in the sets
                collisions = 0
                for a in current:
                    if (a + cand) in current_sums:
                        collisions += 1
                    if (a * cand) in current_prods:
                        collisions += 1
                # Self-interactions
                if (cand + cand) in current_sums:
                    collisions += 1
                if (cand * cand) in current_prods:
                    collisions += 1
                
                if collisions > best_collisions:
                    best_collisions = collisions
                    best_cand = cand
            
            if best_cand is None:
                break
            
            current.append(best_cand)
            current_set.add(best_cand)
        
        if len(current) == n:
            sc = score(current)
            if sc < best_score:
                best_score = sc
                best_A = current[:]
    
    # Local search: try to improve by swapping elements with incremental scoring
    if best_A is not None:
        current_A = best_A[:]
        current_score = best_score
        
        # Precompute sums and prods for current_A
        cur_sums = set()
        cur_prods = set()
        for a in current_A:
            for b in current_A:
                cur_sums.add(a + b)
                cur_prods.add(a * b)
        
        improved_any = True
        while improved_any and time.time() < deadline:
            improved_any = False
            random.shuffle(current_A)
            
            for i in range(n):
                if time.time() > deadline:
                    break
                old_val = current_A[i]
                best_new = None
                best_new_score = current_score
                
                for cand in candidate_list:
                    if cand == old_val or cand in current_A:
                        continue
                    
                    # Try swapping: remove old_val, add cand
                    # Compute new sums and prods incrementally
                    # Remove contributions involving old_val
                    new_sums = set()
                    new_prods = set()
                    for a in current_A:
                        for b in current_A:
                            if a == old_val and b == old_val:
                                continue
                            new_sums.add(a + b)
                            new_prods.add(a * b)
                    # Add contributions involving cand
                    for a in current_A:
                        if a == old_val:
                            continue
                        new_sums.add(a + cand)
                        new_prods.add(a * cand)
                    new_sums.add(cand + cand)
                    new_prods.add(cand * cand)
                    
                    temp_score = max(len(new_sums), len(new_prods))
                    
                    if temp_score < best_new_score:
                        best_new_score = temp_score
                        best_new = cand
                
                if best_new is not None and best_new != old_val:
                    current_A[i] = best_new
                    current_score = best_new_score
                    improved_any = True
    
    return best_A
