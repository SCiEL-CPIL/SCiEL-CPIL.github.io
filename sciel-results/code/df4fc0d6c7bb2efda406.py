"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def _set_sizes(A):
    """
    Compute |A+A| and |A*A| for a list A of distinct integers.
    Returns (sum_count, prod_count).
    """
    n = len(A)
    sums = set()
    prods = set()
    # Exploit symmetry i <= j
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sums.add(ai + aj)
            prods.add(ai * aj)
    return len(sums), len(prods)

def _objective(A):
    s, p = _set_sizes(A)
    return max(s, p)

def _validate(A, n, max_val):
    if len(A) != n:
        return False
    if len(set(A)) != n:
        return False
    if any(x < 1 or x > max_val for x in A):
        return False
    return True

def _generate_ap(start, step, n, max_val):
    A = [start + i * step for i in range(n)]
    if max(A) <= max_val:
        return sorted(A)
    return None

def _generate_gp(n, max_val):
    # Powers of 2
    A = [2**i for i in range(n)]
    if max(A) <= max_val:
        return sorted(A)
    return None

def _generate_hybrid(n, max_val):
    # Small AP + Geometric tail
    A = set()
    # 1..10
    for i in range(1, 11):
        A.add(i)
    # Powers of 2 starting from 16
    v = 16
    while len(A) < n:
        if v <= max_val:
            A.add(v)
        v *= 2
    if len(A) >= n:
        return sorted(list(A)[:n])
    # Fallback fill
    c = 1
    while len(A) < n:
        if c not in A:
            A.add(c)
        c += 1
    return sorted(A)

def _generate_random(n, max_val):
    return sorted(random.sample(range(1, max_val + 1), n))

def _generate_log_spaced(n, max_val):
    # Logarithmically spaced integers
    # 10^0 to 10^4
    # Generate points on log scale
    import math
    points = []
    # Use a finer grid to ensure distinctness
    # Range 1 to 10000
    # Let's try generating 100 log-spaced points and sampling 20
    # Or just generate specific sets
    # Set 1: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 (14 items)
    # Fill with small ints?
    # Let's stick to structured generators in main loop
    
    # Another idea: Multiplicative AP?
    # a, a*r, a*r^2...
    # This is GP.
    
    # Let's generate a set based on divisors of a highly composite number?
    # Or just return None to rely on others.
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    n = 20
    max_val = 10000
    
    # 1. Generate Initial Candidates
    candidates = []
    
    # APs
    candidates.append(_generate_ap(1, 1, n, max_val))
    candidates.append(_generate_ap(1, 2, n, max_val))
    candidates.append(_generate_ap(1, 3, n, max_val))
    candidates.append(_generate_ap(10, 1, n, max_val))
    candidates.append(_generate_ap(100, 10, n, max_val))
    
    # GP
    candidates.append(_generate_gp(n, max_val))
    
    # Hybrid
    candidates.append(_generate_hybrid(n, max_val))
    
    # Randoms
    for _ in range(20):
        candidates.append(_generate_random(n, max_val))
        
    # Filter valid candidates
    valid_candidates = []
    for c in candidates:
        if c is not None and _validate(c, n, max_val):
            valid_candidates.append(c)
    
    if not valid_candidates:
        # Fallback
        return list(range(1, 21))
        
    best_A = valid_candidates[0]
    best_cost = _objective(best_A)
    
    # 2. Local Search (Simulated Annealing-like Hill Climbing)
    # We process each candidate, trying to improve it.
    
    for init_A in valid_candidates:
        if time.time() > deadline:
            break
            
        current_A = init_A[:]
        current_cost = _objective(current_A)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_A = current_A[:]
        
        no_improve_count = 0
        max_steps_per_candidate = 2000 # Adjusted for time
        
        while time.time() < deadline and no_improve_count < max_steps_per_candidate:
            # Strategy: 
            # 1. Random replacement (exploration)
            # 2. Small perturbation (exploitation)
            # 3. Swap (sometimes)
            
            move_type = random.random()
            
            if move_type < 0.5:
                # Random replacement
                idx = random.randint(0, n - 1)
                old_val = current_A[idx]
                new_val = random.randint(1, max_val)
                
                if new_val == old_val:
                    no_improve_count += 1
                    continue
                
                # Check distinctness
                is_distinct = True
                for i, v in enumerate(current_A):
                    if i != idx and v == new_val:
                        is_distinct = False
                        break
                
                if not is_distinct:
                    no_improve_count += 1
                    continue
                
                current_A[idx] = new_val
                new_cost = _objective(current_A)
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    no_improve_count = 0
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = current_A[:]
                else:
                    # Revert
                    current_A[idx] = old_val
                    no_improve_count += 1
                    
            elif move_type < 0.8:
                # Perturb current value
                idx = random.randint(0, n - 1)
                old_val = current_A[idx]
                delta = random.randint(-100, 100)
                new_val = old_val + delta
                
                if new_val < 1:
                    new_val = 1
                if new_val > max_val:
                    new_val = max_val
                
                if new_val == old_val:
                    no_improve_count += 1
                    continue
                
                is_distinct = True
                for i, v in enumerate(current_A):
                    if i != idx and v == new_val:
                        is_distinct = False
                        break
                
                if not is_distinct:
                    no_improve_count += 1
                    continue
                
                current_A[idx] = new_val
                new_cost = _objective(current_A)
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    no_improve_count = 0
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = current_A[:]
                else:
                    current_A[idx] = old_val
                    no_improve_count += 1
                    
            else:
                # Swap two elements
                i, j = random.sample(range(n), 2)
                old_i, old_j = current_A[i], current_A[j]
                current_A[i], current_A[j] = old_j, old_i
                
                new_cost = _objective(current_A)
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    no_improve_count = 0
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = current_A[:]
                else:
                    current_A[i], current_A[j] = old_i, old_j
                    no_improve_count += 1
    
    return best_A

# EVOLVE-BLOCK-END
