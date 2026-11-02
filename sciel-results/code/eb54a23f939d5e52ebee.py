"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
import random

# EVOLVE-BLOCK-START

def _compute_score(U):
    """Compute the score 1 + log(|U-U|/|U+U|)/log(2 max U + 1) for a sorted list U."""
    if not U:
        return 0.0
    max_u = U[-1]
    n = len(U)
    if n < 2:
        return 1.0
    if max_u == 0:
        return 1.0
    
    # Use FFT for efficiency
    size = 2 * max_u + 1
    v = np.zeros(size, dtype=np.float64)
    v[U] = 1.0
    
    # Sumset: v * v
    conv_sum = np.convolve(v, v, mode='full')
    # The sumset U+U has elements in range [0, 2*max_u]
    # conv_sum[i] is the number of pairs (a,b) with a+b=i
    # |U+U| is the number of i where conv_sum[i] > 0
    sumset_card = np.sum(conv_sum > 0.5)
    
    # Difference set: U - U has elements in range [-max_u, max_u]
    w = v[::-1]
    conv_diff = np.convolve(w, v, mode='full')
    # conv_diff has length 2*size - 1 = 4*max_u + 1
    # We need indices from max_u to 3*max_u (corresponding to d from -max_u to max_u)
    diff_part = conv_diff[max_u : 3*max_u + 1]
    diffset_card = np.sum(diff_part > 0.5)
    
    if sumset_card == 0 or diffset_card == 0:
        return 1.0
    
    ratio = diffset_card / sumset_card
    if ratio <= 0:
        return 1.0
    
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom == 0:
        return 1.0
    
    score = 1.0 + log_ratio / log_denom
    return score

def _local_search(U, time_limit, random_state):
    """Local search to improve the set U."""
    U = sorted(U)
    best_score = _compute_score(U)
    best_U = U.copy()
    rng = np.random.RandomState(random_state)
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        # Generate a candidate move
        move_type = rng.randint(0, 3)
        U_curr = best_U.copy()
        
        if move_type == 0:
            # Add a new element
            if len(U_curr) < 300:
                max_u = U_curr[-1] if U_curr else 10
                # Add element in a reasonable range
                if rng.rand() < 0.5:
                    # Add a small element
                    new_elem = rng.randint(0, min(max_u // 2 + 1, 100))
                else:
                    # Add a larger element
                    new_elem = rng.randint(max_u // 2 + 1, max_u + 50)
                
                if new_elem not in U_curr:
                    U_curr.append(new_elem)
                    U_curr.sort()
        elif move_type == 1:
            # Remove an element (except 0)
            if len(U_curr) > 2:
                # Pick a random element to remove (not 0)
                non_zero = [i for i in range(len(U_curr)) if U_curr[i] != 0]
                if non_zero:
                    idx = non_zero[rng.randint(0, len(non_zero))]
                    U_curr.pop(idx)
        else:
            # Swap: remove one and add another
            if len(U_curr) > 2:
                non_zero = [i for i in range(len(U_curr)) if U_curr[i] != 0]
                if non_zero:
                    idx = non_zero[rng.randint(0, len(non_zero))]
                    removed = U_curr.pop(idx)
                    max_u = U_curr[-1] if U_curr else 10
                    if rng.rand() < 0.5:
                        new_elem = rng.randint(0, min(max_u // 2 + 1, 100))
                    else:
                        new_elem = rng.randint(max_u // 2 + 1, max_u + 50)
                    
                    if new_elem not in U_curr:
                        U_curr.append(new_elem)
                        U_curr.sort()
        
        # Evaluate
        if not U_curr or 0 not in U_curr:
            continue
        
        score = _compute_score(U_curr)
        if score > best_score:
            best_score = score
            best_U = U_curr
    
    return best_U, best_score

def _build_initial_set(n, max_u):
    """Build an initial set with n elements including 0, max element <= max_u."""
    U = [0]
    remaining = n - 1
    if remaining <= 0:
        return sorted(U)
    
    # Use a mix of small and larger elements
    U_set = {0}
    val = 1
    while len(U_set) < n and val <= max_u:
        U_set.add(val)
        val *= 2
    
    # If we need more elements, add random ones
    while len(U_set) < n:
        elem = np.random.randint(0, max_u + 1)
        U_set.add(elem)
        if len(U_set) >= n:
            break
    
    U = sorted(list(U_set))
    # Ensure 0 is included
    if 0 not in U:
        U = [0] + U[:n-1]
    return U[:n]

def _construct():
    """Construct a finite set U of nonnegative integers containing 0."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try different initial configurations and local search
    best_U = [0, 1, 3]
    best_score = _compute_score(best_U)
    
    # Try several random starts with local search
    rng = np.random.RandomState(42)
    
    attempts = [
        (10, 50),
        (20, 100),
        (30, 200),
        (50, 500),
        (100, 2000),
        (20, 50),
        (30, 100),
    ]
    
    for trial, (n, max_u) in enumerate(attempts):
        if time.time() > deadline - 5:
            break
        
        # Build initial set
        U_init = _build_initial_set(n, max_u)
        
        # Local search
        time_per_attempt = (deadline - time.time()) / (len(attempts) - trial)
        U_local, score_local = _local_search(U_init, time_per_attempt, 42 + trial)
        
        if score_local > best_score:
            best_score = score_local
            best_U = U_local
    
    # Final check
    if time.time() < deadline - 2:
        # One more round of local search on the best found
        U_final, score_final = _local_search(best_U, deadline - time.time() - 1, 123)
        if score_final > best_score:
            best_U = U_final
            best_score = score_final
    
    return sorted(best_U)

def construct():
    """Construct a finite set U of nonnegative integers containing 0."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try different initial configurations and local search
    best_U = [0, 1, 3]
    best_score = _compute_score(best_U)
    
    # Try several random starts with local search
    rng = np.random.RandomState(42)
    
    attempts = [
        (10, 50),
        (20, 100),
        (30, 200),
        (50, 500),
        (100, 2000),
        (20, 50),
        (30, 100),
    ]
    
    for trial, (n, max_u) in enumerate(attempts):
        if time.time() > deadline - 5:
            break
        
        # Build initial set
        U_init = _build_initial_set(n, max_u)
        
        # Local search
        time_per_attempt = (deadline - time.time()) / (len(attempts) - trial)
        U_local, score_local = _local_search(U_init, time_per_attempt, 42 + trial)
        
        if score_local > best_score:
            best_score = score_local
            best_U = U_local
    
    # Final check
    if time.time() < deadline - 2:
        # One more round of local search on the best found
        U_final, score_final = _local_search(best_U, deadline - time.time() - 1, 123)
        if score_final > best_score:
            best_U = U_final
            best_score = score_final
    
    return sorted(best_U)

# EVOLVE-BLOCK-END
