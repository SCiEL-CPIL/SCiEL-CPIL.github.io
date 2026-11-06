import time
import random
import math

def compute_metrics(A):
    """Compute max(|A+A|, |A*A|) for a set A. Uses symmetry to halve work."""
    n = len(A)
    sum_set = set()
    prod_set = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sum_set.add(ai + aj)
            prod_set.add(ai * aj)
    return max(len(sum_set), len(prod_set)), len(sum_set), len(prod_set)

def generate_smooth_numbers(limit, max_prime=13):
    """Generate smooth numbers up to limit using primes up to max_prime."""
    primes = [2, 3, 5, 7, 11, 13]
    smooth = set([1])
    for p in primes:
        new_smooth = set()
        for s in smooth:
            val = s
            while val <= limit:
                new_smooth.add(val)
                val *= p
        smooth.update(new_smooth)
    return sorted([x for x in smooth if x <= limit])

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin (90s limit)
    
    # Generate candidate pool: smooth numbers (rich factor structure)
    smooth_pool = generate_smooth_numbers(10000, max_prime=13)
    
    # Also add some geometric progression elements
    geo_elements = set()
    for base in [2, 3, 4, 5]:
        val = base
        while val <= 10000:
            geo_elements.add(val)
            val *= base
    smooth_pool = list(set(smooth_pool) | geo_elements)
    smooth_pool.sort()
    
    # Initial candidate sets to try
    initial_sets = []
    
    # Set 1: Powers of 2
    powers2 = [2**i for i in range(20)]
    if all(x <= 10000 for x in powers2):
        initial_sets.append(powers2)
    
    # Set 2: Mix of powers of 2 and 3
    mix23 = sorted(set([2**i for i in range(12)] + [3**i for i in range(10)]))
    if len(mix23) >= 20:
        initial_sets.append(mix23[:20])
    
    # Set 3: First 20 smooth numbers
    if len(smooth_pool) >= 20:
        initial_sets.append(smooth_pool[:20])
    
    # Set 4: Geometric progression with ratio ~1.5 (approximated by smooth)
    # Try to build a near-geometric progression from smooth numbers
    target_ratio = 1.5
    geo_smooth = []
    current = 1
    while len(geo_smooth) < 20 and current <= 10000:
        # Find closest smooth number to current
        best = min(smooth_pool, key=lambda x: abs(x - current))
        if best not in geo_smooth:
            geo_smooth.append(best)
        current *= target_ratio
    if len(geo_smooth) == 20:
        initial_sets.append(geo_smooth)
    
    # Set 5: Consecutive (baseline)
    initial_sets.append(list(range(1, 21)))
    
    # Filter valid initial sets
    valid_initials = []
    for s in initial_sets:
        s = list(set(s))
        s = [x for x in s if 1 <= x <= 10000]
        if len(s) == 20:
            valid_initials.append(s)
    
    if not valid_initials:
        valid_initials = [list(range(1, 21))]
    
    # Pick best initial
    best_score = float('inf')
    best_A = valid_initials[0][:]
    for s in valid_initials:
        score, _, _ = compute_metrics(s)
        if score < best_score:
            best_score = score
            best_A = s[:]
    
    # Simulated Annealing
    current_A = best_A[:]
    current_score = best_score
    best_score_sa = best_score
    best_A_sa = best_A[:]
    
    T = 10.0  # Initial temperature
    cooling_rate = 0.995
    min_T = 0.1
    
    random.seed(42)
    
    while time.time() < deadline and T > min_T:
        # Pick random element to replace
        idx = random.randint(0, 19)
        old_val = current_A[idx]
        
        # Pick random candidate from smooth pool
        new_val = random.choice(smooth_pool)
        
        if new_val in current_A:
            T *= cooling_rate
            continue
        
        trial_A = current_A[:]
        trial_A[idx] = new_val
        trial_A.sort()
        
        if len(set(trial_A)) != 20 or any(x < 1 or x > 10000 for x in trial_A):
            T *= cooling_rate
            continue
        
        trial_score, _, _ = compute_metrics(trial_A)
        delta = trial_score - current_score
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_A = trial_A
            current_score = trial_score
            
            if current_score < best_score_sa:
                best_score_sa = current_score
                best_A_sa = current_A[:]
        
        T *= cooling_rate
    
    # Final local search: hill climbing
    current_A = best_A_sa[:]
    current_score = best_score_sa
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(20):
            if time.time() >= deadline:
                break
            old_val = current_A[i]
            # Try replacing with each candidate (limited for speed)
            for new_val in random.sample(smooth_pool, min(100, len(smooth_pool))):
                if new_val in current_A:
                    continue
                trial_A = current_A[:]
                trial_A[i] = new_val
                trial_A.sort()
                if len(set(trial_A)) != 20 or any(x < 1 or x > 10000 for x in trial_A):
                    continue
                trial_score, _, _ = compute_metrics(trial_A)
                if trial_score < current_score:
                    current_A = trial_A
                    current_score = trial_score
                    improved = True
                    break
    
    if current_score < best_score:
        best_score = current_score
        best_A = current_A[:]
    
    return best_A
