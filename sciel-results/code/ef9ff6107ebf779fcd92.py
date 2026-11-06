import time
import random
import math

MAX_N = 20
MAX_VAL = 10000
TIME_LIMIT = 85  # safety margin before 90s kill

def _objective(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers."""
    A_sorted = sorted(A)
    n = len(A_sorted)
    
    sums = set()
    for i in range(n):
        ai = A_sorted[i]
        for j in range(i, n):
            sums.add(ai + A_sorted[j])
    
    products = set()
    for i in range(n):
        ai = A_sorted[i]
        for j in range(i, n):
            products.add(ai * A_sorted[j])
    
    return max(len(sums), len(products))

def _is_valid(A):
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    for x in A:
        if not isinstance(x, int) or x < 1 or x > 10000:
            return False
    return True

def _smooth_numbers(limit, count):
    """Generate the 'count' smallest 5-smooth numbers <= limit."""
    smooths = set()
    i_max = int(math.log2(limit)) + 1
    j_max = int(math.log(limit, 3)) + 1
    k_max = int(math.log(limit, 5)) + 1
    
    for i in range(i_max + 1):
        pow2 = 1 << i
        if pow2 > limit:
            break
        for j in range(j_max + 1):
            pow3 = pow2 * (3 ** j)
            if pow3 > limit:
                break
            for k in range(k_max + 1):
                val = pow3 * (5 ** k)
                if val > limit:
                    break
                smooths.add(val)
    
    sorted_smooths = sorted(smooths)
    return sorted_smooths[:count]

def _construct_candidates():
    """Generate a list of candidate starting sets, prioritizing smooth numbers."""
    candidates = []
    
    # 1. Primary: 5-smooth numbers (Hamming numbers) - best for product collisions
    try:
        smooths = _smooth_numbers(10000, 20)
        if len(smooths) == 20:
            candidates.append(smooths)
    except:
        pass
    
    # 2. Variants of smooth numbers with slight perturbations
    try:
        smooths_full = _smooth_numbers(10000, 40)
        if len(smooths_full) >= 20:
            # Take first 20
            candidates.append(smooths_full[:20])
            # Take a mix: first 15 + last 5
            if len(smooths_full) >= 20:
                mixed = smooths_full[:15] + smooths_full[-5:]
                if len(set(mixed)) == 20 and max(mixed) <= 10000:
                    candidates.append(sorted(mixed))
    except:
        pass
    
    # 3. Geometric Progressions with ratios close to smooth number ratios
    for ratio in [1.2, 1.25, 1.3, 1.5, 1.6, 1.8, 2.0]:
        for start in [1, 2, 3]:
            A = []
            current = float(start)
            for _ in range(20):
                val = int(round(current))
                if val in A:
                    val += 1
                    while val in A:
                        val += 1
                if val > 10000:
                    break
                A.append(val)
                current *= ratio
            if len(A) == 20 and _is_valid(A):
                candidates.append(A)
                
    # 4. Arithmetic Progressions (less optimal but good for sum structure)
    for start in [1, 2, 5]:
        for step in [1, 2, 3]:
            A = [start + i * step for i in range(20)]
            if max(A) <= 10000 and _is_valid(A):
                candidates.append(A)
                
    # 5. Random samples from small range
    rng_seed = random.Random(123)
    candidates.append(rng_seed.sample(range(1, 1001), 20))
    candidates.append(rng_seed.sample(range(1, 5001), 20))
    
    return candidates

def _local_search(A, time_limit, rng):
    """Local search with simulated annealing, double moves, and polish phase."""
    A = sorted(A)
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    temperature = 80.0
    cooling_rate = 0.998
    min_temp = 0.5
    
    # SA Phase with single and double moves
    while temperature > min_temp:
        if time.time() - start_time > time_limit:
            break
            
        r = rng.random()
        if r < 0.6:
            move_type = 'single'
        elif r < 0.85:
            move_type = 'double'
        else:
            move_type = 'swap'

        if move_type == 'single':
            idx = rng.randint(0, 19)
            old_val = A[idx]
            
            # Generate candidate
            if rng.random() < 0.6:
                # Local perturbation
                step = rng.randint(1, 30)
                new_val = old_val + rng.choice([-1, 1]) * step
            else:
                # Global perturbation
                new_val = rng.randint(1, 10000)
                
            if 1 <= new_val <= 10000 and new_val not in A:
                A[idx] = new_val
                new_obj = _objective(A)
                
                delta = new_obj - current_obj
                if delta <= 0 or rng.random() < min(1.0, max(0.0, -delta / (temperature + 1e-10))):
                    current_obj = new_obj
                    if new_obj < best_obj:
                        best_obj = new_obj
                        best_A = list(A)
                else:
                    A[idx] = old_val
                    
        elif move_type == 'double':
            idx1 = rng.randint(0, 19)
            idx2 = rng.randint(0, 19)
            if idx1 == idx2:
                idx2 = (idx2 + 1) % 20
                
            old_val1 = A[idx1]
            old_val2 = A[idx2]
            
            # Generate candidates with local perturbations
            step1 = rng.randint(1, 30)
            step2 = rng.randint(1, 30)
            new_val1 = old_val1 + rng.choice([-1, 1]) * step1
            new_val2 = old_val2 + rng.choice([-1, 1]) * step2
            
            if 1 <= new_val1 <= 10000 and 1 <= new_val2 <= 10000:
                if new_val1 not in A and new_val2 not in A and new_val1 != new_val2:
                    A[idx1] = new_val1
                    A[idx2] = new_val2
                    new_obj = _objective(A)
                    
                    delta = new_obj - current_obj
                    if delta <= 0 or rng.random() < min(1.0, max(0.0, -delta / (temperature + 1e-10))):
                        current_obj = new_obj
                        if new_obj < best_obj:
                            best_obj = new_obj
                            best_A = list(A)
                    else:
                        A[idx1] = old_val1
                        A[idx2] = old_val2
                        
        else:  # swap
            idx1 = rng.randint(0, 19)
            idx2 = rng.randint(0, 19)
            if idx1 == idx2:
                continue
                
            old_val1 = A[idx1]
            old_val2 = A[idx2]
            
            # Try swapping values with small perturbations
            new_val1 = old_val2 + rng.randint(-5, 5)
            new_val2 = old_val1 + rng.randint(-5, 5)
            
            if 1 <= new_val1 <= 10000 and 1 <= new_val2 <= 10000:
                if new_val1 not in A and new_val2 not in A and new_val1 != new_val2:
                    A[idx1] = new_val1
                    A[idx2] = new_val2
                    new_obj = _objective(A)
                    
                    delta = new_obj - current_obj
                    if delta <= 0 or rng.random() < min(1.0, max(0.0, -delta / (temperature + 1e-10))):
                        current_obj = new_obj
                        if new_obj < best_obj:
                            best_obj = new_obj
                            best_A = list(A)
                    else:
                        A[idx1] = old_val1
                        A[idx2] = old_val2

        temperature *= cooling_rate
        
    # Polish Phase: Strict Hill Climbing with systematic local search
    polish_time_limit = min(8.0, time_limit - (time.time() - start_time))
    if polish_time_limit > 1.0:
        polish_start = time.time()
        improved = True
        while improved and (time.time() - polish_start) < polish_time_limit:
            improved = False
            for i in range(20):
                if (time.time() - polish_start) > polish_time_limit:
                    break
                old_val = best_A[i]
                # Try systematic local perturbations
                found_better = False
                for step in range(1, 61):
                    for direction in [-1, 1]:
                        new_val = old_val + direction * step
                        if new_val < 1 or new_val > 10000:
                            continue
                        if new_val in best_A:
                            continue
                        
                        temp_A = list(best_A)
                        temp_A[i] = new_val
                        new_obj = _objective(temp_A)
                        
                        if new_obj < best_obj:
                            best_obj = new_obj
                            best_A = temp_A
                            improved = True
                            found_better = True
                            break
                    if found_better:
                        break
    return best_A, best_obj

def construct():
    start_time = time.time()
    time_budget = TIME_LIMIT - 5
    
    candidates = _construct_candidates()
    
    best_A = None
    best_obj = float('inf')
    
    rng = random.Random(42)
    
    # Sort candidates by initial objective to prioritize good starting points
    def _initial_obj(A):
        try:
            return _objective(A)
        except:
            return float('inf')
    
    valid_candidates = []
    for A in candidates:
        if not _is_valid(A):
            continue
        obj = _initial_obj(A)
        valid_candidates.append((obj, A))
    
    valid_candidates.sort(key=lambda x: x[0])
    
    for obj, A in valid_candidates:
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        if remaining < 2.0:
            break
        if obj < best_obj:
            # This candidate alone is already better
            best_A = list(A)
            best_obj = obj
        refined_A, refined_obj = _local_search(A, remaining, rng)
        if refined_obj < best_obj:
            best_obj = refined_obj
            best_A = refined_A
            
    if best_A is None:
        best_A = list(range(1, 21))
        best_obj = _objective(best_A)
        
    # Final refinement with remaining time
    remaining = TIME_LIMIT - 5 - (time.time() - start_time)
    if remaining > 2.0:
        try:
            refined_A, refined_obj = _local_search(best_A, remaining, rng)
            if refined_obj < best_obj:
                best_A = refined_A
                best_obj = refined_obj
        except Exception:
            pass

    if not _is_valid(best_A):
        best_A = list(range(1, 21))
        
    return sorted(best_A)
