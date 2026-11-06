"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
import math

_N = 256

# Precompute all 4-term arithmetic progressions as flat index arrays
_AP_A = []
_AP_B = []
_AP_C = []
_AP_D = []
for d in range(1, _N // 3 + 1):
    for a in range(_N - 3 * d):
        _AP_A.append(a)
        _AP_B.append(a + d)
        _AP_C.append(a + 2 * d)
        _AP_D.append(a + 3 * d)

_AP_A = np.array(_AP_A, dtype=np.int32)
_AP_B = np.array(_AP_B, dtype=np.int32)
_AP_C = np.array(_AP_C, dtype=np.int32)
_AP_D = np.array(_AP_D, dtype=np.int32)
_TOTAL_APS = len(_AP_A)

def _score_fast(colors):
    """
    colors: numpy array of 0/1 of length 256.
    Returns the count of monochromatic 4-term APs.
    """
    # Map {0,1} to {-1,1}
    # Using 2*colors - 1 is fast
    x = 2 * colors - 1
    
    # Vectorized product and sum
    # This creates a temporary array of size _TOTAL_APS
    p = x[_AP_A] * x[_AP_B] * x[_AP_C] * x[_AP_D]
    sum_p = np.sum(p)
    
    # Mono count = (Total + Sum) / 2
    # Since Total and Sum have the same parity (both even or both odd? 
    # Sum is sum of +/-1. Total is integer.
    # (Total + Sum) is always even because if all are 1, Sum=Total. If mixed, 
    # actually, let's just use integer division.
    return (_TOTAL_APS + sum_p) // 2

def _evaluate(colours_list):
    """
    colours_list: list of 256 values in {0,1}
    Returns the count of monochromatic 4-term APs.
    """
    arr = np.array(colours_list, dtype=np.int8)
    return _score_fast(arr)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    def get_time_left():
        return deadline - time.time()

    best_colours = None
    best_score = float('inf')
    
    def update_best(c, s):
        nonlocal best_colours, best_score
        if s < best_score:
            best_score = s
            best_colours = c.copy()

    # 1. Initialization: Try various structured and random colorings
    candidates = []
    
    # Random seeds
    for s in range(20):
        random.seed(s)
        candidates.append(np.array([random.randint(0, 1) for _ in range(_N)], dtype=np.int8))
        
    # Structured
    for bs in [2, 4, 8, 16, 32]:
        candidates.append(np.array([(i // bs) % 2 for i in range(_N)], dtype=np.int8))
        
    # Thue-Morse
    candidates.append(np.array([bin(i).count('1') % 2 for i in range(_N)], dtype=np.int8))
    
    # Parity
    candidates.append(np.array([i % 2 for i in range(_N)], dtype=np.int8))
    
    # Evaluate all candidates
    for cand in candidates:
        score = _score_fast(cand)
        update_best(cand, score)
        
    # 2. Search: Simulated Annealing + Greedy Local Search
    # We will run multiple restarts.
    
    restarts = 0
    while get_time_left() > 10.0:
        restarts += 1
        
        # Start from best found so far for early restarts, then random
        if restarts <= 5:
            current = best_colours.copy()
            current_score = best_score
        else:
            random.seed(1000 + restarts)
            current = np.array([random.randint(0, 1) for _ in range(_N)], dtype=np.int8)
            current_score = _score_fast(current)
            update_best(current, current_score)
            
        # Simulated Annealing
        T = 2.0
        cooling = 0.99995
        steps = 0
        max_steps = 100000  # Cap steps per restart to allow time checks
        
        while steps < max_steps and get_time_left() > 5.0:
            steps += 1
            i = random.randint(0, _N - 1)
            
            # Flip bit
            old_val = current[i]
            current[i] = 1 - old_val
            new_score = _score_fast(current)
            
            delta = new_score - current_score
            
            # Accept if better or with probability exp(-delta/T)
            if delta < 0 or (T > 0.01 and random.random() < math.exp(-delta / T)):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_colours = current.copy()
            else:
                current[i] = old_val
                
            T *= cooling
            
            # If temperature is low, switch to pure greedy descent
            if T < 0.01:
                improved = True
                while improved and get_time_left() > 5.0:
                    improved = False
                    # Shuffle indices to avoid bias
                    indices = list(range(_N))
                    random.shuffle(indices)
                    for j in indices:
                        old_val = current[j]
                        current[j] = 1 - old_val
                        new_score = _score_fast(current)
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                            if new_score < best_score:
                                best_score = new_score
                                best_colours = current.copy()
                        else:
                            current[j] = old_val
                # Break out of SA loop
                break

    return best_colours.tolist()
# EVOLVE-BLOCK-END
