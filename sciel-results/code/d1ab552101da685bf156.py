import time
import random
import math

def compute_metrics(A):
    """Compute max(|A+A|, |A*A|) for a set A."""
    n = len(A)
    sum_set = set()
    prod_set = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sum_set.add(ai + aj)
            prod_set.add(ai * aj)
    return max(len(sum_set), len(prod_set))

def compute_metrics_fast(A):
    """Faster metric computation using sorted list and early termination."""
    n = len(A)
    sums = []
    prods = []
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            sums.append(ai + A[j])
            prods.append(ai * A[j])
    # Use set for dedup but with pre-allocation hint
    sum_set = set(sums)
    prod_set = set(prods)
    return max(len(sum_set), len(prod_set))

def generate_pool(limit=10000):
    """Generate a pool of smooth numbers and powers of 2."""
    pool = set()
    # Powers of 2
    p = 1
    while p <= limit:
        pool.add(p)
        p *= 2
    # 2,3-smooth
    p2 = 1
    while p2 <= limit:
        p3 = p2
        while p3 <= limit:
            pool.add(p3)
            p3 *= 3
        p2 *= 2
    # 2,3,5-smooth
    p2 = 1
    while p2 <= limit:
        p3 = p2
        while p3 <= limit:
            p5 = p3
            while p5 <= limit:
                pool.add(p5)
                p5 *= 5
            p3 *= 3
        p2 *= 2
    # 2,3,5,7-smooth (Hamming numbers)
    p2 = 1
    while p2 <= limit:
        p3 = p2
        while p3 <= limit:
            p5 = p3
            while p5 <= limit:
                p7 = p5
                while p7 <= limit:
                    pool.add(p7)
                    p7 *= 7
                p5 *= 5
            p3 *= 3
        p2 *= 2
    return sorted(list(pool))

def generate_diverse_pool(limit=10000):
    """Generate a more diverse pool including some non-smooth numbers."""
    pool = generate_pool(limit)
    # Add some random numbers to break symmetry
    random.seed(12345)
    for _ in range(200):
        x = random.randint(1, limit)
        pool.append(x)
    return sorted(set(pool))

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    pool = generate_pool(10000)
    pool_size = len(pool)
    pool_set = set(pool)
    
    random.seed(42)
    
    best_A = None
    best_score = float('inf')
    
    # Multiple initializations
    num_starts = 5
    start_time_init = time.time()
    
    for start_idx in range(num_starts):
        if time.time() > deadline - 10:
            break
        
        # Different initialization strategies
        if start_idx == 0:
            # Original: powers of 2 + smooths
            init_A = []
            p = 1
            while len(init_A) < 14 and p <= 10000:
                init_A.append(p)
                p *= 2
            for x in pool:
                if len(init_A) >= 20:
                    break
                if x not in init_A:
                    init_A.append(x)
        elif start_idx == 1:
            # Geometric progression with ratio ~1.5
            init_A = []
            x = 1
            while len(init_A) < 20 and x <= 10000:
                init_A.append(x)
                x = int(x * 1.5) + 1
                if x in init_A:
                    x += 1
        elif start_idx == 2:
            # Mix of small and large smooth numbers
            init_A = []
            # First 10 smallest smooths
            for x in pool[:10]:
                init_A.append(x)
            # Next 10 from middle of pool
            mid = len(pool) // 2
            for x in pool[mid:mid+10]:
                if x not in init_A:
                    init_A.append(x)
        elif start_idx == 3:
            # Random from pool
            init_A = random.sample(pool, 20)
        else:
            # Powers of 2 and 3 mixed
            init_A = []
            p2 = 1
            p3 = 1
            while len(init_A) < 20:
                if p2 <= 10000 and p2 not in init_A:
                    init_A.append(p2)
                    p2 *= 2
                if len(init_A) >= 20:
                    break
                if p3 <= 10000 and p3 not in init_A:
                    init_A.append(p3)
                    p3 *= 3
                if p2 > 10000 and p3 > 10000:
                    break
            # Fill remaining
            for x in pool:
                if len(init_A) >= 20:
                    break
                if x not in init_A:
                    init_A.append(x)
        
        init_A = list(set(init_A))
        if len(init_A) < 20:
            for x in range(1, 10001):
                if len(init_A) >= 20:
                    break
                if x not in init_A:
                    init_A.append(x)
        
        current_A = sorted(init_A)
        current_score = compute_metrics(current_A)
        
        if current_score < best_score:
            best_score = current_score
            best_A = current_A[:]
        
        # Local search
        no_improve_count = 0
        max_no_improve = 30
        
        while time.time() < deadline:
            improved = False
            # Shuffle pool to avoid bias
            pool_shuffled = pool[:]
            random.shuffle(pool_shuffled)
            
            for i in range(20):
                if time.time() > deadline:
                    break
                old_val = current_A[i]
                
                # Generate candidates: focus on numbers that might be compatible
                # Include: smooth numbers, numbers close to old_val, random from pool
                candidates = set()
                
                # Add some from pool (random sample)
                sample_size = min(30, pool_size)
                candidates.update(random.sample(pool_shuffled, sample_size))
                
                # Add numbers near old_val
                for delta in range(-20, 21):
                    x = old_val + delta
                    if 1 <= x <= 10000:
                        candidates.add(x)
                
                # Add multiples and divisors of old_val
                for d in range(1, min(old_val + 1, 50)):
                    if old_val % d == 0:
                        candidates.add(d)
                        if old_val // d <= 10000:
                            candidates.add(old_val // d)
                    if old_val * d <= 10000:
                        candidates.add(old_val * d)
                
                # Filter out values already in current_A
                candidates = [c for c in candidates if c not in current_A and 1 <= c <= 10000]
                
                if not candidates:
                    continue
                
                # Try each candidate
                for new_val in candidates:
                    if new_val in current_A:
                        continue
                    
                    trial_A = current_A[:]
                    trial_A[i] = new_val
                    trial_A.sort()
                    
                    # Quick validation
                    if len(set(trial_A)) != 20:
                        continue
                    
                    trial_score = compute_metrics(trial_A)
                    
                    if trial_score < current_score:
                        current_A = trial_A
                        current_score = trial_score
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_A = current_A[:]
                        break # Restart search for this i with new set
            
            if not improved:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    # Perturbation: swap 2 random elements with random pool values
                    i, j = random.sample(range(20), 2)
                    new_val1 = random.choice(pool)
                    new_val2 = random.choice(pool)
                    
                    if new_val1 not in current_A and new_val2 not in current_A and new_val1 != new_val2:
                        trial_A = current_A[:]
                        trial_A[i] = new_val1
                        trial_A[j] = new_val2
                        trial_A.sort()
                        
                        if len(set(trial_A)) == 20 and all(1 <= x <= 10000 for x in trial_A):
                            trial_score = compute_metrics(trial_A)
                            # Allow slight worsening to escape local optima
                            if trial_score <= best_score * 1.05:
                                current_A = trial_A
                                current_score = trial_score
                    else:
                        # Reset to best
                        current_A = best_A[:]
                        current_score = best_score
                    
                    no_improve_count = 0
            else:
                no_improve_count = 0
            
            # Check if we've hit the theoretical minimum
            # For n=20, |A+A| >= 2n-1 = 39, |A*A| >= n = 20
            # So max >= 39. If we're at 39, we can't do better.
            if best_score <= 40:
                break
        
        # If we found a very good solution, we can stop early
        if best_score <= 42:
            break
    
    # Final polish: try all single-element swaps from best_A
    if best_A is not None and time.time() < deadline - 5:
        final_A = best_A[:]
        final_score = best_score
        
        # Try each position with all pool candidates
        for i in range(20):
            if time.time() > deadline - 2:
                break
            old_val = final_A[i]
            
            # Try all pool values (or a large sample)
            candidates = pool if pool_size <= 200 else random.sample(pool, 200)
            
            for new_val in candidates:
                if new_val in final_A:
                    continue
                
                trial_A = final_A[:]
                trial_A[i] = new_val
                trial_A.sort()
                
                if len(set(trial_A)) != 20:
                    continue
                
                trial_score = compute_metrics(trial_A)
                
                if trial_score < final_score:
                    final_A = trial_A
                    final_score = trial_score
                    break
        
        if final_score < best_score:
            best_score = final_score
            best_A = final_A
    
    return best_A
