import numpy as np
import random
import time
def _objective(A):
    """
    Compute the objective function: max(|A+A|, |A*A|).
    A is a list of distinct integers.
    """
    A = np.array(A)
    sums = A[:, None] + A[None, :]
    prods = A[:, None] * A[None, :]
    n_sums = len(np.unique(sums))
    n_prods = len(np.unique(prods))
    return max(n_sums, n_prods)

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
    """
    Construct a set of 20 distinct positive integers <= 10000
    minimizing max(|A+A|, |A*A|).
    """
    start_time = time.time()
    deadline = 85.0
    
    rng = random.Random(42)
    A = list(range(1, 21))
    
    time_elapsed = time.time() - start_time
    time_left = deadline - time_elapsed
    
    if time_left > 1.0:
        best_A, best_obj = _local_search(A, time_left, rng)
    else:
        best_A = A
        best_obj = _objective(A)
        
    best_A = sorted(set(best_A))
    if len(best_A) != 20:
        best_A = list(range(1, 21))
        
    return best_A
