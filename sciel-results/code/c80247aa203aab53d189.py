"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _count_progression_diffs(S):
    """Count distinct d > 0 such that there exists a with a, a+d, a+2d in S."""
    S_set = set(S)
    diffs = set()
    for a in S:
        for b in S:
            d = b - a
            if d > 0 and (a + 2 * d) in S_set:
                diffs.add(d)
    return len(diffs)

def _simulated_annealing(initial_S, time_limit=10.0, start_temp=10.0, cooling_rate=0.999):
    """
    Perform Simulated Annealing to maximize the score.
    Returns the best set found and its score.
    """
    current_S = list(initial_S)
    current_score = _count_progression_diffs(current_S)
    best_S = list(current_S)
    best_score = current_score
    
    start_time = time.time()
    temp = start_temp
    
    # Pre-compute range for random selection
    range_1_512 = range(1, 513)
    
    while time.time() - start_time < time_limit:
        # Generate a neighbor: swap one element
        i = random.randint(0, len(current_S) - 1)
        old_val = current_S[i]
        
        # Pick a new value not in the set
        # To avoid infinite loops if set is dense (it's not, 32/512), 
        # we can just pick random and check.
        new_val = random.choice(range_1_512)
        
        if new_val == old_val:
            continue
            
        S_set = set(current_S)
        if new_val in S_set:
            continue
            
        # Apply move
        current_S[i] = new_val
        new_score = _count_progression_diffs(current_S)
        
        # Check if better
        if new_score > best_score:
            best_score = new_score
            best_S = list(current_S)
            
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0:
            # Always accept improvements
            current_score = new_score
        else:
            # Accept worse solutions with probability exp(delta / temp)
            if temp > 0:
                prob = math.exp(delta / temp)
                if random.random() < prob:
                    current_score = new_score
                else:
                    # Reject move, revert
                    current_S[i] = old_val
        
        # Cool down
        temp *= cooling_rate
        
    return best_S, best_score

def _local_search_polish(S, time_limit=2.0):
    """
    Fast hill climbing to polish the result.
    Only accepts improvements.
    """
    current_S = list(S)
    current_score = _count_progression_diffs(current_S)
    best_S = list(current_S)
    best_score = current_score
    
    start_time = time.time()
    range_1_512 = range(1, 513)
    
    while time.time() - start_time < time_limit:
        i = random.randint(0, len(current_S) - 1)
        old_val = current_S[i]
        new_val = random.choice(range_1_512)
        
        if new_val == old_val:
            continue
            
        S_set = set(current_S)
        if new_val in S_set:
            continue
            
        current_S[i] = new_val
        new_score = _count_progression_diffs(current_S)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_S = list(current_S)
        else:
            current_S[i] = old_val
            
    return best_S, best_score

def construct():
    """
    Construct a 32-element subset of 1..512 maximizing the number of distinct
    3-term arithmetic progression differences.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    best_S = None
    best_score = 0
    
    # Initial candidate: Consecutive integers
    S_init = list(range(1, 33))
    score_init = _count_progression_diffs(S_init)
    best_S = S_init
    best_score = score_init
    
    # Strategy: Multiple Simulated Annealing runs
    # We have ~140s. Let's do runs of ~10s each.
    # That allows about 10-12 runs.
    
    # Use a fixed seed for reproducibility of the search process? 
    # The task doesn't require determinism, but it helps debugging.
    # However, random restarts are better with different seeds.
    
    num_sa_runs = 12
    sa_time_limit = 10.0
    
    for run_idx in range(num_sa_runs):
        elapsed = time.time() - start_time
        if elapsed > deadline - sa_time_limit - 5: # Leave 5s for polish/final checks
            break
            
        # Generate a random initial set for diversity
        # Mix of random and structured?
        # Let's try random sets, and occasionally the baseline
        if run_idx == 0:
            init_S = list(range(1, 33))
        else:
            init_S = random.sample(range(1, 513), 32)
            
        # Run SA
        sa_S, sa_score = _simulated_annealing(init_S, time_limit=sa_time_limit)
        
        if sa_score > best_score:
            best_score = sa_score
            best_S = sa_S
            
    # Final Polish: Local search on the best result found
    if time.time() - start_time < deadline - 2:
        polish_time = min(5.0, deadline - (time.time() - start_time) - 1)
        if polish_time > 0:
            final_S, final_score = _local_search_polish(best_S, time_limit=polish_time)
            if final_score > best_score:
                best_score = final_score
                best_S = final_S
                
    # Ensure validity
    result = sorted(set(best_S))
    if len(result) != 32:
        # Fallback
        result = list(range(1, 33))
        
    return result

# EVOLVE-BLOCK-END
