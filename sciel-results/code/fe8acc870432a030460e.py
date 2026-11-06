import time
import random
from math import log

def compute_metrics(A):
    """Compute max(|A+A|, |A*A|) for set A."""
    sum_set = set()
    prod_set = set()
    for a in A:
        for b in A:
            sum_set.add(a + b)
            prod_set.add(a * b)
    return max(len(sum_set), len(prod_set))

def generate_smooth_numbers(max_val, primes=[2,3,5,7]):
    """Generate smooth numbers up to max_val."""
    smooth = set()
    def gen(idx, current):
        if current > max_val:
            return
        if current > 0:
            smooth.add(current)
        if idx < len(primes):
            for _ in range(30):  # Limit exponent depth
                if current * primes[idx] > max_val:
                    break
                current *= primes[idx]
                gen(idx, current)
            gen(idx + 1, current // (primes[idx] ** 30) if primes[idx] ** 30 <= current else current)
    gen(0, 1)
    return sorted([x for x in smooth if x >= 1])

def simulated_annealing(candidates, n=20, max_val=10000, time_limit=45.0):
    """Optimize set using simulated annealing."""
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Initialize with a random subset of smooth numbers
    if len(candidates) < n:
        # Fallback: use arithmetic progression
        base = 1
        step = (max_val - base) // (n - 1) if n > 1 else 1
        current = [base + i * step for i in range(n)]
    else:
        # Weight towards smaller smooth numbers for better product collisions
        weights = [1.0 / log(x + 1) for x in candidates]
        total_w = sum(weights)
        weights = [w / total_w for w in weights]
        current = random.choices(candidates, weights=weights, k=n)
        # Ensure distinctness
        while len(set(current)) < n:
            idx = random.randint(0, n-1)
            current[idx] = random.choice(candidates)
    
    current = sorted(set(current))
    if len(current) < n:
        # Pad with next available numbers
        used = set(current)
        for x in range(1, max_val + 1):
            if x not in used and len(current) < n:
                current.append(x)
        current = sorted(current)
    
    best = current[:]
    best_score = compute_metrics(best)
    current_score = best_score
    
    temperature = 100.0
    cooling = 0.995
    
    while time.time() < deadline:
        # Generate neighbor: swap one element
        idx = random.randint(0, n-1)
        new_val = random.choice(candidates)
        if new_val in current and current.index(new_val) != idx:
            continue
        
        old_val = current[idx]
        current[idx] = new_val
        current = sorted(set(current))
        
        if len(current) != n:
            current[idx] = old_val
            continue
        
        new_score = compute_metrics(current)
        delta = new_score - current_score
        
        if delta < 0 or random.random() < (1.0 / (1.0 + temperature)) * (-delta / (1 + abs(delta))):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best = current[:]
        else:
            current[idx] = old_val
            current = sorted(set(current))
        
        temperature *= cooling
        if temperature < 0.01:
            temperature = 0.01
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 90s kill
    
    # Generate smooth number candidates
    smooth_nums = generate_smooth_numbers(10000, primes=[2, 3, 5, 7, 11])
    
    if not smooth_nums or len(smooth_nums) < 20:
        # Fallback
        return list(range(1, 21))
    
    # Run simulated annealing
    best_set, best_score = simulated_annealing(smooth_nums, n=20, max_val=10000, time_limit=45.0)
    
    # Validate result
    if len(set(best_set)) != 20 or any(x < 1 or x > 10000 for x in best_set):
        return list(range(1, 21))
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
