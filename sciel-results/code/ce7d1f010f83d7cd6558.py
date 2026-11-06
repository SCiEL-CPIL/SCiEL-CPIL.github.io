"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _count_progression_diffs(S):
    """Count distinct d > 0 such that there exists a with a, a+d, a+2d in S."""
    S_set = set(S)
    diffs = set()
    # Optimization: iterate over pairs (a, b) with a < b
    # d = b - a
    # Check if a + 2d is in S
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            d = S[j] - S[i]
            if d > 0 and (S[i] + 2 * d) in S_set:
                diffs.add(d)
    return len(diffs)

def _sa_search(S_init, time_limit=60.0):
    """
    Simulated Annealing search.
    """
    S = list(S_init)
    current_score = _count_progression_diffs(S)
    best_S = list(S)
    best_score = current_score
    
    T = 10.0
    T_min = 0.1
    cooling = 0.9995
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        # Generate a move: replace one element
        i = random.randint(0, len(S) - 1)
        old_val = S[i]
        new_val = random.randint(1, 512)
        
        if new_val == old_val or new_val in S:
            continue
            
        S[i] = new_val
        new_score = _count_progression_diffs(S)
        
        delta = new_score - current_score
        
        if delta > 0 or random.random() < math.exp(delta / T):
            # Accept move
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_S = list(S)
        else:
            # Reject move
            S[i] = old_val
            
        T *= cooling
        if T < T_min:
            T = T_min
            # Optional: restart from best if stuck too long? 
            # For now, just keep cooling floor.
            
    return best_S, best_score

def construct():
    """
    Construct a 32-element subset of 1..512 maximizing the number of distinct
    3-term arithmetic progression differences.
    """
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    best_S = None
    best_score = 0
    
    # Candidate 1: Consecutive integers 1..32
    S1 = list(range(1, 33))
    score1 = _count_progression_diffs(S1)
    best_S = S1
    best_score = score1
    
    # Candidate 2: Two blocks to capture more d's?
    # Block 1: 1..16, Block 2: 257..272
    S2 = list(range(1, 17)) + list(range(257, 273))
    score2 = _count_progression_diffs(S2)
    if score2 > best_score:
        best_score = score2
        best_S = S2
        
    # Candidate 3: Random
    random.seed(42)
    for _ in range(20):
        if time.time() - start_time > deadline - 30:
            break
        S_rand = random.sample(range(1, 513), 32)
        score_rand = _count_progression_diffs(S_rand)
        if score_rand > best_score:
            best_score = score_rand
            best_S = S_rand
            
    # Run SA from best candidate
    remaining = deadline - (time.time() - start_time) - 5
    if remaining > 10:
        best_S, best_score = _sa_search(best_S, time_limit=remaining)
        
    # Run SA from a few other good candidates if time permits
    candidates = [S1, S2]
    for cand in candidates:
        if time.time() - start_time > deadline - 10:
            break
        remaining = deadline - (time.time() - start_time) - 2
        if remaining < 5:
            break
        # Run a short SA from this candidate
        S_new, score_new = _sa_search(cand, time_limit=remaining)
        if score_new > best_score:
            best_score = score_new
            best_S = S_new

    # Final check
    result = sorted(set(best_S))
    if len(result) != 32:
        result = list(range(1, 33))
        
    return result

# EVOLVE-BLOCK-END
