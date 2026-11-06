"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import product

# Precompute all 4-term arithmetic progressions
_N = 256
_AP_INDICES = []
for d in range(1, _N // 3 + 1):
    for a in range(0, _N - 3 * d):
        _AP_INDICES.append((a, a + d, a + 2 * d, a + 3 * d))

# Convert to numpy arrays for fast vectorized access
_AP_A = np.array([t[0] for t in _AP_INDICES], dtype=np.int32)
_AP_B = np.array([t[1] for t in _AP_INDICES], dtype=np.int32)
_AP_C = np.array([t[2] for t in _AP_INDICES], dtype=np.int32)
_AP_D = np.array([t[3] for t in _AP_INDICES], dtype=np.int32)

def _evaluate_fast(arr):
    """
    arr: numpy array of 0/1 of length 256.
    Returns the count of monochromatic 4-term APs.
    """
    # Get colors at all AP positions
    ca = arr[_AP_A]
    cb = arr[_AP_B]
    cc = arr[_AP_C]
    cd = arr[_AP_D]
    
    # Check if all four are equal
    # (ca == cb) & (cb == cc) & (cc == cd)
    mask = (ca == cb) & (cb == cc) & (cc == cd)
    return np.sum(mask)

def _evaluate(colours_list):
    """
    colours_list: list of 256 values in {0,1}
    Returns the count of monochromatic 4-term APs.
    """
    arr = np.array(colours_list, dtype=np.int8)
    return _evaluate_fast(arr)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    def get_time_left():
        return deadline - time.time()

    # Helper to evaluate numpy array directly for speed in loops
    def eval_np(arr):
        return _evaluate_fast(arr)

    # 1. Initialization: Try various structured and random colorings
    best_colours = None
    best_score = float('inf')
    
    # Candidate generators
    def gen_random(seed):
        random.seed(seed)
        return np.array([random.randint(0, 1) for _ in range(256)], dtype=np.int8)

    def gen_block(block_size):
        return np.array([(i // block_size) % 2 for i in range(256)], dtype=np.int8)

    def gen_thue_morse():
        # Thue-Morse sequence: number of 1s in binary representation mod 2
        return np.array([bin(i).count('1') % 2 for i in range(256)], dtype=np.int8)

    def gen_parity():
        return np.array([i % 2 for i in range(256)], dtype=np.int8)

    candidates = []
    
    # Structured
    for bs in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 32, 40, 64, 128]:
        candidates.append(gen_block(bs))
    
    candidates.append(gen_thue_morse())
    candidates.append(gen_parity())
    
    # Random seeds
    for s in range(20):
        candidates.append(gen_random(s))
        
    # Also try some "low density" or "high density" biases
    for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        random.seed(100 + int(p*100))
        candidates.append(np.array([1 if random.random() < p else 0 for _ in range(256)], dtype=np.int8))

    # Evaluate all candidates
    for cand in candidates:
        score = eval_np(cand)
        if score < best_score:
            best_score = score
            best_colours = cand.copy()
            
    # 2. Local Search (First Descent)
    # We can do many passes because evaluation is fast
    current = best_colours.copy()
    current_score = best_score
    
    # Precompute indices for fast access
    # We will flip one bit at a time and check improvement
    
    # Optimization: Pre-calculate the "potential" change for each bit flip?
    # The score is sum over APs of [all 4 same].
    # Flipping bit k affects all APs containing k.
    # Let's just use the fast evaluator for now, it's ~10-20us per eval.
    # 256 flips * 20us = 5ms per pass.
    # We can do ~1000 passes in 5 seconds.
    
    improved = True
    pass_count = 0
    while improved and get_time_left() > 10.0:
        improved = False
        # Randomize order to avoid cycles
        indices = list(range(256))
        random.shuffle(indices)
        
        for i in indices:
            # Flip
            current[i] = 1 - current[i]
            new_score = eval_np(current)
            
            if new_score < current_score:
                current_score = new_score
                improved = True
                if new_score < best_score:
                    best_score = new_score
                    best_colours = current.copy()
            else:
                # Revert
                current[i] = 1 - current[i]
        
        pass_count += 1
        
        # If stuck, try a small random perturbation (kick)
        if not improved:
            # Try flipping 2-3 random bits to escape local minimum
            kicks = 5
            for _ in range(kicks):
                if get_time_left() < 10.0: break
                trial = current.copy()
                # Flip 2 or 3 random bits
                num_flips = random.randint(2, 3)
                flip_idx = random.sample(range(256), num_flips)
                for idx in flip_idx:
                    trial[idx] = 1 - trial[idx]
                
                trial_score = eval_np(trial)
                if trial_score < current_score:
                    current = trial
                    current_score = trial_score
                    improved = True
                    break
                # If not improving, we might accept a slightly worse one if we're stuck for too long?
                # For now, stick to strict descent with kicks.

    # 3. Simulated Annealing (if time permits)
    if get_time_left() > 20.0:
        current = best_colours.copy()
        current_score = best_score
        temp = 5.0
        cooling = 0.995
        
        while get_time_left() > 10.0 and temp > 0.01:
            i = random.randint(0, 255)
            current[i] = 1 - current[i]
            new_score = eval_np(current)
            delta = new_score - current_score
            
            if delta < 0 or random.random() < np.exp(-delta / max(temp, 1e-6)):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_colours = current.copy()
            else:
                current[i] = 1 - current[i]
            
            temp *= cooling
            
            # Final polish after SA
            if get_time_left() < 15.0:
                 # Run a few more local search passes
                 improved = True
                 while improved and get_time_left() > 2.0:
                     improved = False
                     indices = list(range(256))
                     random.shuffle(indices)
                     for i in indices:
                         current[i] = 1 - current[i]
                         new_score = eval_np(current)
                         if new_score < current_score:
                             current_score = new_score
                             improved = True
                             if new_score < best_score:
                                 best_score = new_score
                                 best_colours = current.copy()
                         else:
                             current[i] = 1 - current[i]

    return best_colours.tolist()
# EVOLVE-BLOCK-END
