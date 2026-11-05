import random
import time
import math

def calc_max_disc_and_witnesses(signs, n=256, limit=None):
    """
    Calculates max discrepancy and returns the witnesses (progressions) achieving it.
    If limit is provided and exceeded, returns early with the current best witness.
    """
    max_disc = 1
    best_witnesses = []
    for d in range(1, n):
        for r in range(d):
            curr = 0
            min_p = 0
            max_p = 0
            idx = r
            while idx < n:
                curr += signs[idx]
                if curr < min_p: min_p = curr
                if curr > max_p: max_p = curr
                idx += d
            disc = max_p - min_p
            if disc > max_disc:
                max_disc = disc
                if limit is not None and max_disc > limit:
                    return max_disc, [(d, r, min_p, max_p)]
                best_witnesses = [(d, r, min_p, max_p)]
            elif disc == max_disc:
                best_witnesses.append((d, r, min_p, max_p))
    return max_disc, best_witnesses

def get_indices_from_witnesses(witnesses, n=256):
    """Extract unique indices from a list of witnesses."""
    indices = set()
    for d, r, min_p, max_p in witnesses:
        idx = r
        while idx < n:
            indices.add(idx)
            idx += d
    return list(indices)

def greedy_polish(signs, n=256, deadline=None, max_rounds=50):
    """
    Greedy polish: repeatedly find the flip that most reduces max discrepancy.
    More thorough than simple greedy by trying multiple passes.
    """
    current_disc, _ = calc_max_disc_and_witnesses(signs, n)
    improved = True
    rounds = 0
    while improved and rounds < max_rounds and (deadline is None or time.time() < deadline):
        improved = False
        best_gain = 0
        best_idx = -1
        # Try flipping each position and see if it reduces the discrepancy
        for i in range(n):
            signs[i] *= -1
            new_disc, _ = calc_max_disc_and_witnesses(signs, n, limit=current_disc)
            signs[i] *= -1
            gain = current_disc - new_disc
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx >= 0 and best_gain > 0:
            signs[best_idx] *= -1
            current_disc -= best_gain
            improved = True
            rounds += 1
    return current_disc

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55  # 55s margin
    
    # Best solution found across all runs
    global_best_disc = float('inf')
    global_best_signs = None
    
    # Multiple restarts to explore different basins
    num_restarts = 3
    
    for restart in range(num_restarts):
        if time.time() >= deadline:
            break
            
        # Initial solution: start with random, then do a quick greedy improvement
        signs = [random.choice((-1, 1)) for _ in range(n)]
        
        # Quick greedy initialization to get a decent starting point
        greedy_polish(signs, n, deadline=start_time + 15)
        
        # Initial full calculation
        current_disc, current_witnesses = calc_max_disc_and_witnesses(signs, n)
        
        # Update global best
        if current_disc < global_best_disc:
            global_best_disc = current_disc
            global_best_signs = signs[:]
        
        # Targeted Simulated Annealing with adaptive parameters
        T = 3.0
        T_min = 0.005
        cooling = 0.9995
        no_improve_counter = 0
        max_no_improve = 2000
        
        while time.time() < deadline and no_improve_counter < max_no_improve:
            # 1. Select move based on current witnesses with some randomness
            if not current_witnesses:
                # Fallback to random if no witnesses found (should not happen)
                i = random.randint(0, n - 1)
            else:
                hot_indices = get_indices_from_witnesses(current_witnesses, n)
                if not hot_indices:
                    i = random.randint(0, n - 1)
                else:
                    # Mix hot indices with some random exploration
                    if random.random() < 0.8:
                        i = random.choice(hot_indices)
                    else:
                        i = random.randint(0, n - 1)
                        
            # 2. Flip sign
            signs[i] *= -1
            
            # 3. Evaluate new state
            # Use limit to speed up rejection
            limit = current_disc + int(T * 3) + 1
            new_disc, new_witnesses = calc_max_disc_and_witnesses(signs, n, limit=limit)
            
            delta = new_disc - current_disc
            
            # 4. Accept or Reject
            if delta <= 0 or random.random() < math.exp(-delta / T):
                # Accept
                current_disc = new_disc
                current_witnesses = new_witnesses
                
                if current_disc < global_best_disc:
                    global_best_disc = current_disc
                    global_best_signs = signs[:]
                    no_improve_counter = 0
                else:
                    no_improve_counter += 1
            else:
                # Reject: Revert sign
                signs[i] *= -1
                no_improve_counter += 1
            
            # 5. Cool with adaptive rate
            T *= cooling
            if T < T_min:
                T = T_min
                # If we've cooled completely, try a small reheating to escape local minima
                if no_improve_counter > 1000:
                    T = 1.0
                    no_improve_counter = 0
    
    # Final polish on the best solution found
    if global_best_signs is not None:
        final_signs = global_best_signs[:]
        final_disc = greedy_polish(final_signs, n, deadline=deadline)
        if final_disc < global_best_disc:
            global_best_disc = final_disc
            global_best_signs = final_signs
    
    return global_best_signs
