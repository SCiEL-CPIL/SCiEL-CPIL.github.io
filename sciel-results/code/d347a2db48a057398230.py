import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START
def _fast_score(U):
    """
    Compute the score for a set U of nonnegative integers containing 0.
    Score = 1 + log(|U-U| / |U+U|) / log(2 * max(U) + 1)
    Uses numpy for vectorized computation of sums and differences.
    """
    if not U or 0 not in U:
        return 0.0
    
    U_arr = np.array(U, dtype=np.int64)
    n = len(U_arr)
    if n < 2:
        return 0.0
        
    mx = U_arr[-1] # Assuming sorted
    if mx <= 0:
        return 0.0
        
    # Compute U+U (sums of pairs i<=j)
    # Using broadcasting might be memory intensive for large n, but n is likely < 100-200 in SA.
    # To be safe and fast, we can use a loop if n is large, but let's try vectorized first.
    # If n > 1000, we might need to be careful with memory (1000x1000 int64 is 8MB, which is fine).
    
    # Sums
    # U[i] + U[j] for i <= j
    # We can compute full outer sum and take upper triangle, or just compute and unique.
    # unique is O(N log N).
    
    # For speed, let's compute sums and diffs.
    # Sums:
    sums = U_arr[:, None] + U_arr[None, :]
    # We only need i <= j for the set of sums? No, U+U is the set of all sums.
    # Since addition is commutative, the set of values is the same as {a+b | a,b in U}.
    # We can just take the unique values of the full matrix.
    # However, memory for n=200 is 200*200*8 = 320KB. n=1000 is 8MB. n=5000 is 200MB.
    # Let's assume n stays under 2000 to be safe within 120s.
    
    # Differences:
    diffs = U_arr[:, None] - U_arr[None, :]
    
    # Get unique counts
    # np.unique is fast for 1D arrays. Flatten first.
    n_plus = len(np.unique(sums))
    n_minus = len(np.unique(diffs))
    
    if n_plus == 0:
        return 0.0
        
    ratio = n_minus / n_plus
    if ratio <= 0:
        return 0.0
        
    log_den = math.log(2 * mx + 1)
    if log_den <= 0:
        return 0.0
        
    val = 1.0 + math.log(ratio) / log_den
    return val

def construct():
    t0 = time.time()
    deadline = t0 + 115.0 # Safety margin
    rng = random.Random(42)
    np_rng = np.random.default_rng(42)
    
    # Initial population: A mix of small sets to find a good starting point
    # Start with a simple lacunary set
    best_U = [0, 1, 2, 4]
    best_score = _fast_score(best_U)
    
    # Try a few initial candidates
    candidates = [
        [0, 1, 3, 7],
        [0, 1, 4, 9],
        [0, 1, 2, 5, 10],
        [0, 1, 3, 6, 10],
        [0, 1, 2, 4, 8, 16],
        [0, 1, 3, 9, 27],
    ]
    
    for c in candidates:
        s = _fast_score(c)
        if s > best_score:
            best_score = s
            best_U = c
            
    # Simulated Annealing
    # State: List of integers (sorted)
    # Moves: 
    # 1. Add a random integer
    # 2. Remove a random integer (not 0)
    # 3. Modify a random integer
    
    current_U = list(best_U)
    current_score = best_score
    
    T = 100.0 # Initial temperature
    cooling = 0.995
    
    step = 0
    while time.time() < deadline and T > 0.1:
        step += 1
        
        # Generate neighbor
        neighbor = list(current_U)
        move_type = rng.randint(0, 2)
        
        if move_type == 0 and len(neighbor) < 300:
            # Add
            # Choose a value. Bias towards values that don't create too many collisions?
            # Random value in range [1, 2*max]
            mx = neighbor[-1] if neighbor else 0
            # Sometimes add a value that is "lacunary-like" to maintain structure
            if rng.random() < 0.5:
                val = rng.randint(1, max(2, mx * 2))
            else:
                val = mx * rng.randint(2, 4) + rng.randint(0, mx)
            
            if val not in neighbor:
                neighbor.append(val)
                neighbor.sort()
        
        elif move_type == 1 and len(neighbor) > 2:
            # Remove (not 0)
            if len(neighbor) > 1:
                idx = rng.randint(1, len(neighbor) - 1)
                neighbor.pop(idx)
        
        elif move_type == 2 and len(neighbor) > 1:
            # Modify
            if len(neighbor) > 1:
                idx = rng.randint(1, len(neighbor) - 1)
                old_val = neighbor[idx]
                # Perturb
                delta = rng.randint(-max(1, old_val//10), max(1, old_val//10))
                new_val = old_val + delta
                if new_val > 0 and new_val not in neighbor:
                    neighbor[idx] = new_val
                    neighbor.sort()
        
        if neighbor == current_U:
            continue
            
        # Evaluate
        try:
            new_score = _fast_score(neighbor)
        except Exception:
            new_score = 0.0
            
        if new_score > current_score:
            current_U = neighbor
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_U = list(neighbor)
        else:
            # Accept with probability exp(-delta / T)
            delta = current_score - new_score
            if delta < 0:
                accept = True
            else:
                # delta is positive (bad move)
                # prob = exp(-delta / T)
                # To avoid underflow/overflow issues with very small T or large delta
                if delta > 10 * T:
                    accept = False
                else:
                    accept = (rng.random() < math.exp(-delta / T))
            
            if accept:
                current_U = neighbor
                current_score = new_score
        
        T *= cooling
        
        # Occasionally, if stuck, do a "restart" from best with noise
        if step % 500 == 0 and current_score < best_score - 0.01:
             # Mutate best
             current_U = list(best_U)
             # Add some noise
             for _ in range(3):
                 if len(current_U) > 1:
                     idx = rng.randint(1, len(current_U)-1)
                     current_U[idx] = max(1, current_U[idx] + rng.randint(-10, 10))
                     current_U.sort()
             current_score = _fast_score(current_U)
    
    return best_U
# EVOLVE-BLOCK-END
