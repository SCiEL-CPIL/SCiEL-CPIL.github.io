"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

def _score_set(U):
    """Compute the lower bound score for a set U using numpy for speed."""
    if not U or 0 not in U:
        return 0.0
    U_arr = np.array(sorted(set(U)), dtype=np.int64)
    n = len(U_arr)
    if n < 2:
        return 0.0
    max_U = int(U_arr[-1])
    if max_U == 0:
        return 0.0
    
    # Compute sumset U+U using numpy broadcasting
    sumset = np.unique(U_arr[:, None] + U_arr[None, :])
    
    # Compute difference set U-U using numpy broadcasting
    diffset = np.unique(U_arr[:, None] - U_arr[None, :])
    
    sumset_size = len(sumset)
    diffset_size = len(diffset)
    
    if sumset_size == 0:
        return 0.0
    
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    
    log_base = math.log(2 * max_U + 1)
    if log_base <= 0:
        return 0.0
    
    score = 1 + math.log(ratio) / log_base
    return score

def _generate_candidates(U, max_val_limit):
    """
    Generate a pool of candidate elements based on the arithmetic structure
    (sums and differences) of the current set U. This helps escape local minima
    by exploring the 'arithmetic neighborhood'.
    """
    U_set = set(U)
    candidates = set()
    
    # Sums and differences of existing elements
    # We limit the loops to avoid O(N^2) explosion if N is large, 
    # but for N~50 it's fine.
    for a in U:
        for b in U:
            s = a + b
            d = abs(a - b)
            if 0 < s <= max_val_limit and s not in U_set:
                candidates.add(s)
            if 0 < d <= max_val_limit and d not in U_set:
                candidates.add(d)
    
    # Add some random small integers to ensure diversity and avoid getting stuck
    # in purely structural patterns
    for _ in range(50):
        r = random.randint(1, min(200, max_val_limit))
        if r not in U_set:
            candidates.add(r)
            
    # Add some random larger jumps
    for _ in range(20):
        r = random.randint(1, max_val_limit)
        if r not in U_set:
            candidates.add(r)

    return list(candidates)

def _local_search_dynamic(U, deadline, max_val_limit, max_n_limit=60):
    """
    Local search with dynamic candidate generation.
    Uses a mix of Add, Remove, and Swap moves.
    """
    U = sorted(set(U))
    if 0 not in U:
        U.append(0)
        U.sort()
    
    current_score = _score_set(U)
    best_local_score = current_score
    best_local_set = list(U)
    
    # Initial candidate pool
    candidates = _generate_candidates(U, max_val_limit)
    
    no_improvement_count = 0
    max_no_improvement = 40
    
    while time.time() < deadline - 2:
        if no_improvement_count > max_no_improvement:
            break
            
        improved = False
        move_type = random.random()
        
        # Update candidate pool periodically or if empty
        if not candidates or random.random() < 0.15:
            current_max = max(U)
            new_limit = max(max_val_limit, current_max * 2)
            candidates = _generate_candidates(U, new_limit)
        
        if move_type < 0.5:
            # Try to ADD an element
            if len(U) < max_n_limit and candidates:
                random.shuffle(candidates)
                # Try a few random candidates
                for c in candidates[:15]:
                    if c in U:
                        continue
                    new_U = U + [c]
                    new_U.sort()
                    new_score = _score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        # Regenerate candidates immediately as structure changed significantly
                        candidates = _generate_candidates(U, max(max_val_limit, max(U)*2))
                        break
                        
        elif move_type < 0.7:
            # Try to REMOVE an element
            if len(U) > 2:
                removable = [x for x in U if x != 0]
                if removable:
                    c = random.choice(removable)
                    new_U = [x for x in U if x != c]
                    new_score = _score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = _generate_candidates(U, max_val_limit)
                        
        else:
            # Try to REPLACE an element (Swap)
            if len(U) > 1 and candidates:
                # Pick a random element to remove (not 0)
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                
                random.shuffle(candidates)
                for new_val in candidates[:15]:
                    if new_val == old_val or new_val in U:
                        continue
                    new_U = U.copy()
                    new_U[idx] = new_val
                    new_U.sort()
                    new_score = _score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = _generate_candidates(U, max(max_val_limit, max(U)*2))
                        break
            
        if improved:
            no_improvement_count = 0
            if current_score > best_local_score:
                best_local_score = current_score
                best_local_set = list(U)
        else:
            no_improvement_count += 1
                
    return best_local_score, best_local_set

def _construct_continuous_approximation(deadline):
    """
    Strategy: Model the set construction as a continuous optimization problem.
    Represent U as the image of a strictly convex, differentiable function f(t) 
    sampled at integer points.
    
    We approximate f(t) = a * t^b + c * t.
    Since U must be integers, we round f(t) to the nearest integer and ensure uniqueness.
    We optimize parameters a, b, c to maximize the score.
    
    This is a structural change: instead of discrete search, we search the 
    continuous parameter space of a generating function.
    """
    best_score = 0.0
    best_set = [0, 1, 2]
    
    # Parameters to optimize
    # a: scale factor
    # b: exponent (convexity)
    # c: linear component
    # n: number of points
    
    # We will use a simple grid search + local refinement on parameters
    # because gradient descent on discrete cardinalities is tricky.
    
    # Define ranges
    a_range = np.linspace(0.1, 10.0, 20)
    b_range = np.linspace(1.1, 3.0, 20)
    c_range = np.linspace(0.0, 5.0, 10)
    n_range = [10, 20, 30, 40, 50, 60, 80]
    
    # Pre-compute t values for max n
    max_n = max(n_range)
    t_vals = np.arange(max_n)
    
    for n in n_range:
        if time.time() > deadline - 5:
            break
            
        for a in a_range:
            for b in b_range:
                for c in c_range:
                    # Generate continuous values
                    f_vals = a * (t_vals[:n] ** b) + c * t_vals[:n]
                    
                    # Round to integers
                    U_int = np.round(f_vals).astype(np.int64)
                    
                    # Ensure 0 is in the set and unique
                    # f(0) = 0, so U_int[0] should be 0.
                    # Check for duplicates
                    if len(np.unique(U_int)) < n:
                        continue
                    
                    U_list = sorted(set(U_int.tolist()))
                    if 0 not in U_list:
                        continue
                        
                    # Check if max value is reasonable (not too large to kill score via log denom)
                    if max(U_list) > 500000:
                        continue
                        
                    score = _score_set(U_list)
                    if score > best_score:
                        best_score = score
                        best_set = U_list
                        
    return best_score, best_set

def construct():
    """Main construction function."""
    start_time = time.time()
    deadline = start_time + 145  # Use more of the 150s budget
    
    random.seed(42)
    
    # Start with a reasonable initial set
    best_set = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
    best_score = _score_set(best_set)
    
    # Phase 1: Continuous Approximation (Structural Change)
    # This explores a completely different family of sets (smooth convex curves)
    if time.time() < deadline - 20:
        cont_score, cont_set = _construct_continuous_approximation(deadline)
        if cont_score > best_score:
            best_score = cont_score
            best_set = cont_set
    
    # Phase 2: Local Search with Dynamic Candidate Generation
    # This refines the best set found so far (either continuous or initial)
    # using the structural insight from the inspiration.
    if time.time() < deadline - 10:
        # We might want to restart from a few different points
        # But let's first refine the current best
        
        # If the continuous set was good, refine it.
        # If not, maybe refine the initial set or a random one.
        
        # Let's run local search on the best set found so far
        ls_score, ls_set = _local_search_dynamic(best_set, deadline, max_val_limit=100000, max_n_limit=80)
        if ls_score > best_score:
            best_score = ls_score
            best_set = ls_set
        
        # If we still have time, try a few restarts from random/structured sets
        restarts = 0
        while time.time() < deadline - 5 and restarts < 5:
            restarts += 1
            
            # Choose a starting set
            if restarts % 2 == 0:
                # Start from a structured set (e.g., powers of 2 or 3)
                base = random.choice([2, 3, 4])
                U = [0]
                val = 1
                for _ in range(10):
                    U.append(val)
                    val *= base
                if max(U) > 50000:
                    U = U[:8]
            else:
                # Start from random
                n = random.randint(10, 30)
                U = [0] + random.sample(range(1, 5000), n - 1)
                U.sort()
            
            # Run local search
            ls_score, ls_set = _local_search_dynamic(U, deadline, max_val_limit=100000, max_n_limit=80)
            
            if ls_score > best_score:
                best_score = ls_score
                best_set = ls_set

    return best_set
# EVOLVE-BLOCK-END
