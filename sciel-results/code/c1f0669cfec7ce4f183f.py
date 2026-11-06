import random
import time

def construct():
    start_time = time.time()
    limit = 85.0  # Safety margin
    
    def get_score(A):
        # A is a list of 20 distinct integers
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
        
    def local_search(A, deadline):
        A = list(A)
        current_score = get_score(A)
        best_A = list(A)
        best_score = current_score
        T = 5.0
        cooling = 0.9995
        no_improve_count = 0
        max_no_improve = 5000
        
        while time.time() < deadline:
            if no_improve_count > max_no_improve:
                break
                
            i = random.randint(0, 19)
            old_val = A[i]
            
            # Generate candidate
            r = random.random()
            if r < 0.2:
                # Local small change
                new_val = old_val + random.randint(-5, 5)
            elif r < 0.4:
                # Multiplicative change
                if random.random() < 0.5:
                    new_val = old_val * 2
                else:
                    if old_val % 2 == 0:
                        new_val = old_val // 2
                    else:
                        new_val = old_val * 3
            elif r < 0.6:
                # Additive combination with another element
                j = random.randint(0, 19)
                if i == j:
                    continue
                new_val = old_val + A[j]
            elif r < 0.8:
                # Multiplicative combination
                j = random.randint(0, 19)
                if i == j:
                    continue
                new_val = old_val * A[j]
            else:
                # Random global
                new_val = random.randint(1, 10000)
                
            if new_val < 1 or new_val > 10000:
                continue
                
            # Check distinctness
            # We are replacing A[i] with new_val.
            # If new_val is in A[k] for k != i, it's invalid.
            if new_val in A:
                # If new_val is the old_val, it's a no-op (skip)
                # If new_val is at another index, it's a collision
                if new_val != old_val:
                    continue
            
            A[i] = new_val
            new_score = get_score(A)
            
            if new_score < current_score:
                current_score = new_score
                no_improve_count = 0
                if current_score < best_score:
                    best_score = current_score
                    best_A = list(A)
            else:
                no_improve_count += 1
                # Accept worse solution with probability
                # Use log-sum-exp style or simple exp
                delta = new_score - current_score
                if delta < 0: # Should not happen if not better
                    pass
                prob = pow(2.71828, -delta / max(T, 1e-9))
                if random.random() < prob:
                    current_score = new_score
                else:
                    A[i] = old_val # Revert
            
            T *= cooling
            if T < 0.1:
                T = 0.1
                
        return best_A, best_score

    # Initial candidates
    candidates = []
    candidates.append(list(range(1, 21)))
    
    # AP with step 2
    candidates.append([1 + 2*i for i in range(20)])
    
    # Random smooth-ish
    # Generate numbers with many small prime factors
    # Simple heuristic: multiples of 6, 12, etc
    smooth_candidates = []
    for _ in range(10):
        c = []
        while len(c) < 20:
            # Pick random small base
            base = random.randint(1, 50)
            val = base
            # Multiply by small primes
            for _ in range(random.randint(0, 3)):
                val *= random.choice([2, 3, 5])
            if 1 <= val <= 10000 and val not in c:
                c.append(val)
        candidates.append(c)
        
    # Random sets
    for _ in range(5):
        candidates.append(random.sample(range(1, 10001), 20))
        
    best_global_A = candidates[0]
    best_global_score = get_score(best_global_A)
    
    # Run local search on each candidate
    # Allocate time
    num_candidates = len(candidates)
    time_per = (limit - (time.time() - start_time)) / num_candidates
    
    for c in candidates:
        if time.time() - start_time > limit:
            break
        current_deadline = time.time() + time_per
        A_res, score_res = local_search(c, current_deadline)
        if score_res < best_global_score:
            best_global_score = score_res
            best_global_A = A_res
            
    # Final polish
    remaining_time = limit - (time.time() - start_time)
    if remaining_time > 5:
        A_res, score_res = local_search(best_global_A, time.time() + remaining_time)
        if score_res < best_global_score:
            best_global_score = score_res
            best_global_A = A_res
            
    return best_global_A
