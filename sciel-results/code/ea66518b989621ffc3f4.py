"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import numpy as np

# EVOLVE-BLOCK-START
def _count_progressions(arr):
    """Count distinct d > 0 such that a, a+d, a+2d are all in arr."""
    s = set(arr)
    distinct_d = set()
    n = len(arr)
    for i in range(n):
        a = arr[i]
        for j in range(i + 1, n):
            b = arr[j]
            d = b - a
            if a + 2 * d in s:
                distinct_d.add(d)
    return len(distinct_d)

def _local_search(arr, max_iters=5000, timeout=140.0):
    """Local search by swapping elements to improve progression count."""
    start_time = time.time()
    arr = list(arr)
    best_score = _count_progressions(arr)
    best_arr = list(arr)
    n = len(arr)
    s = set(arr)
    
    for it in range(max_iters):
        if time.time() - start_time > timeout:
            break
        if random.random() < 0.1:
            # Random restart with slight perturbation
            arr = list(best_arr)
            for _ in range(3):
                idx = random.randint(0, n - 1)
                new_val = random.randint(1, 512)
                if new_val not in s:
                    arr[idx] = new_val
            s = set(arr)
            continue
        
        # Try swapping two elements
        i, j = random.sample(range(n), 2)
        old_i, old_j = arr[i], arr[j]
        
        # Try replacing arr[i] with a new random value
        new_val = random.randint(1, 512)
        if new_val in s and new_val != old_i:
            continue
        
        arr[i] = new_val
        s.discard(old_i)
        s.add(new_val)
        score = _count_progressions(arr)
        
        if score > best_score:
            best_score = score
            best_arr = list(arr)
        else:
            # Revert
            arr[i] = old_i
            s.discard(new_val)
            s.add(old_i)
    
    return best_arr, best_score

def _genetic_algorithm(timeout=140.0):
    """Genetic algorithm to find 32-element set with many 3-term progressions."""
    start_time = time.time()
    pop_size = 20
    n = 32
    
    # Initial population
    population = []
    for _ in range(pop_size):
        if random.random() < 0.3:
            # Structured: arithmetic progression-like
            base = random.randint(1, 50)
            d = random.randint(1, 15)
            arr = [base + i * d for i in range(n)]
            # Filter out of range and duplicates
            arr = [x for x in arr if 1 <= x <= 512]
            while len(arr) < n:
                x = random.randint(1, 512)
                if x not in arr:
                    arr.append(x)
            arr = arr[:n]
        else:
            # Random
            arr = random.sample(range(1, 513), n)
        population.append((arr, _count_progressions(arr)))
    
    best_arr, best_score = max(population, key=lambda x: x[1])
    
    while time.time() - start_time < timeout:
        # Selection and crossover
        new_population = []
        for _ in range(pop_size):
            if random.random() < 0.1:
                # Random new individual
                if random.random() < 0.3:
                    base = random.randint(1, 50)
                    d = random.randint(1, 15)
                    arr = [base + i * d for i in range(n)]
                    arr = [x for x in arr if 1 <= x <= 512]
                    while len(arr) < n:
                        x = random.randint(1, 512)
                        if x not in arr:
                            arr.append(x)
                    arr = arr[:n]
                else:
                    arr = random.sample(range(1, 513), n)
            else:
                # Crossover from top individuals
                top = sorted(population, key=lambda x: -x[1])[:5]
                p1, p2 = random.sample(top, 2)
                arr1, arr2 = p1[0], p2[0]
                # Uniform crossover
                mask = [random.random() < 0.5 for _ in range(n)]
                arr = [arr1[i] if mask[i] else arr2[i] for i in range(n)]
                # Ensure distinct and in range
                seen = set()
                new_arr = []
                for x in arr:
                    if x not in seen and 1 <= x <= 512:
                        seen.add(x)
                        new_arr.append(x)
                while len(new_arr) < n:
                    x = random.randint(1, 512)
                    if x not in seen:
                        seen.add(x)
                        new_arr.append(x)
                arr = new_arr[:n]
            
            score = _count_progressions(arr)
            new_population.append((arr, score))
        
        population = new_population
        best_in_pop = max(population, key=lambda x: x[1])
        if best_in_pop[1] > best_score:
            best_score = best_in_pop[1]
            best_arr = best_in_pop[0]
    
    return best_arr, best_score

def construct():
    deadline = time.time() + 145.0
    random.seed(42)
    
    # Start with a good initial guess
    # Try structured sets first
    candidates = []
    
    # Arithmetic progression
    for d in range(1, 17):
        arr = [1 + i * d for i in range(32)]
        if max(arr) <= 512:
            candidates.append(arr)
    
    # Random structured
    for _ in range(50):
        base = random.randint(1, 30)
        d = random.randint(1, 15)
        arr = [base + i * d for i in range(32)]
        arr = [x for x in arr if 1 <= x <= 512]
        while len(arr) < 32:
            x = random.randint(1, 512)
            if x not in arr:
                arr.append(x)
        candidates.append(arr[:32])
    
    # Random
    for _ in range(50):
        candidates.append(random.sample(range(1, 513), 32))
    
    # Evaluate candidates
    best_arr = max(candidates, key=lambda x: _count_progressions(x))
    best_score = _count_progressions(best_arr)
    
    # Local search
    if time.time() < deadline - 10:
        best_arr, best_score = _local_search(best_arr, max_iters=3000, timeout=min(30, deadline - time.time() - 5))
    
    # Genetic algorithm if time permits
    if time.time() < deadline - 30:
        ga_timeout = min(60, deadline - time.time() - 5)
        ga_arr, ga_score = _genetic_algorithm(timeout=ga_timeout)
        if ga_score > best_score:
            best_arr = ga_arr
            best_score = ga_score
    
    # Final local search polish
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 2
        best_arr, best_score = _local_search(best_arr, max_iters=2000, timeout=remaining)
    
    return list(best_arr)
# EVOLVE-BLOCK-END
