import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    n = 20
    max_val = 10000
    
    def evaluate(A):
        """
        Vectorized evaluation using NumPy.
        Returns (max_score, sumset_size, prodset_size).
        """
        A = np.array(A, dtype=np.int64)
        # Create upper triangle indices for i <= j
        i_idx, j_idx = np.triu_indices(n)
        
        sums = A[i_idx] + A[j_idx]
        prods = A[i_idx] * A[j_idx]
        
        # unique is faster than converting to set in pure Python for these sizes
        # np.unique sorts, but we just need count
        sumset_size = len(np.unique(sums))
        prodset_size = len(np.unique(prods))
        
        return max(sumset_size, prodset_size), sumset_size, prodset_size

    # 1. Generate diverse initial candidates
    # Strategy: Mix of smooth numbers and geometric-like progressions
    # Smooth numbers tend to have small product sets.
    # Arithmetic progressions tend to have small sum sets (2n-1).
    
    def get_smooth_numbers(limit, max_prime=7, count=100):
        smooths = []
        for i in range(1, limit + 1):
            x = i
            p = 2
            while p * p <= x and p <= max_prime:
                while x % p == 0:
                    x //= p
                p += 1
            if x == 1 or (x > 1 and x <= max_prime):
                smooths.append(i)
            if len(smooths) >= count:
                break
        return smooths

    # Candidate 1: 1..20 (Base)
    init1 = list(range(1, 21))
    
    # Candidate 2: Small smooth numbers
    smooths = get_smooth_numbers(10000, 7, 100)
    init2 = random.sample(smooths, 20)
    
    # Candidate 3: Geometric progression with integer ratio where possible, mixed
    # e.g., 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 (14)
    # Need 6 more. Maybe mix in some mid-range numbers.
    gp = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    # Fill with random numbers from 1..100 not in GP
    remaining = [x for x in range(1, 100) if x not in gp]
    init3 = gp + random.sample(remaining, 6)
    init3 = sorted(init3)
    
    # Candidate 4: Random integers 1..50
    init4 = random.sample(range(1, 51), 20)
    
    candidates = [init1, init2, init3, init4]
    
    best_set = None
    best_score = float('inf')
    
    for c in candidates:
        # Ensure distinctness and validity
        if len(set(c)) == 20 and all(1 <= x <= max_val for x in c):
            score, _, _ = evaluate(c)
            if score < best_score:
                best_score = score
                best_set = c[:]
    
    current_set = best_set[:]
    current_score = best_score
    current_sum, current_prod = evaluate(current_set)[1:]
    
    temperature = 50.0
    cooling_rate = 0.999
    min_temp = 0.5
    
    steps = 0
    # With vectorized eval, we can do many more steps.
    # Let's see how fast it is. 1000 steps should be < 1s.
    # 60s budget -> potentially 60,000+ steps.
    
    while time.time() < deadline:
        steps += 1
        
        # Perturbation strategies:
        # 1. Single element replacement (Random)
        # 2. Single element replacement (Local: +/- delta)
        # 3. Block shift (Shift a contiguous block of elements by a small delta)
        
        r = random.random()
        
        if r < 0.5:
            # Single element replacement
            idx = random.randint(0, n - 1)
            old_val = current_set[idx]
            
            # Bias towards smaller numbers to keep prodset small?
            # Or random 1..max_val
            if random.random() < 0.5:
                new_val = random.randint(1, 100)
            else:
                new_val = random.randint(1, max_val)
            
            if new_val == old_val or new_val in current_set:
                continue
                
            current_set[idx] = new_val
            new_score, new_sum, new_prod = evaluate(current_set)
            
            delta = new_score - current_score
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_score = new_score
                current_sum = new_sum
                current_prod = new_prod
                if current_score < best_score:
                    best_score = current_score
                    best_set = current_set[:]
            else:
                current_set[idx] = old_val
                
        elif r < 0.8:
            # Local move
            idx = random.randint(0, n - 1)
            old_val = current_set[idx]
            delta_val = random.randint(-10, 10)
            new_val = old_val + delta_val
            
            if new_val < 1 or new_val > max_val or new_val == old_val:
                continue
            if new_val in current_set:
                continue
                
            current_set[idx] = new_val
            new_score, new_sum, new_prod = evaluate(current_set)
            
            delta = new_score - current_score
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_score = new_score
                current_sum = new_sum
                current_prod = new_prod
                if current_score < best_score:
                    best_score = current_score
                    best_set = current_set[:]
            else:
                current_set[idx] = old_val
                
        else:
            # Block shift: Select a random contiguous block of size k (2..5) and shift all by delta
            k = random.randint(2, 5)
            start_idx = random.randint(0, n - k)
            end_idx = start_idx + k - 1
            
            delta_val = random.randint(-5, 5)
            if delta_val == 0:
                continue
                
            # Check validity for all in block
            valid = True
            new_vals = []
            for i in range(start_idx, end_idx + 1):
                nv = current_set[i] + delta_val
                if nv < 1 or nv > max_val:
                    valid = False
                    break
                new_vals.append(nv)
            
            if not valid:
                continue
                
            # Check distinctness with rest of set
            rest_set = set(current_set[:start_idx] + current_set[end_idx+1:])
            if any(nv in rest_set for nv in new_vals):
                continue
            if len(set(new_vals)) != len(new_vals):
                continue
                
            # Apply
            old_block = current_set[start_idx:end_idx+1]
            for i in range(start_idx, end_idx + 1):
                current_set[i] = new_vals[i - start_idx]
                
            new_score, new_sum, new_prod = evaluate(current_set)
            
            delta = new_score - current_score
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_score = new_score
                current_sum = new_sum
                current_prod = new_prod
                if current_score < best_score:
                    best_score = current_score
                    best_set = current_set[:]
            else:
                for i in range(start_idx, end_idx + 1):
                    current_set[i] = old_block[i - start_idx]
        
        temperature *= cooling_rate
        
        # Reheat
        if steps % 2000 == 0 and temperature < 5.0:
            temperature = 20.0
            
    return sorted(best_set)
