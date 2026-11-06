import numpy as np
import time
import random

_N = 256

# Precompute all 4-term arithmetic progressions
# Indices are 0-indexed internally, but the problem is 1-indexed.
# We will store 0-indexed positions in the array.
_AP_LIST = []
_AP_A = []
_AP_B = []
_AP_C = []
_AP_D = []

# Build inverse mapping: for each index i, which APs contain it?
# inv_ap[i] = list of AP indices
inv_ap = [[] for _ in range(_N)]

ap_idx = 0
for d in range(1, _N // 3 + 1):
    for a in range(_N - 3 * d):
        b = a + d
        c = a + 2 * d
        e = a + 3 * d
        # Store 0-indexed positions
        _AP_A.append(a)
        _AP_B.append(b)
        _AP_C.append(c)
        _AP_D.append(e)
        _AP_LIST.append((a, b, c, e))
        
        inv_ap[a].append(ap_idx)
        inv_ap[b].append(ap_idx)
        inv_ap[c].append(ap_idx)
        inv_ap[e].append(ap_idx)
        
        ap_idx += 1

_AP_A = np.array(_AP_A, dtype=np.int32)
_AP_B = np.array(_AP_B, dtype=np.int32)
_AP_C = np.array(_AP_C, dtype=np.int32)
_AP_D = np.array(_AP_D, dtype=np.int32)
_NUM_APS = len(_AP_LIST)

def _score(arr):
    """
    arr: numpy array of 0/1 of length 256.
    Returns the count of monochromatic 4-term APs.
    """
    ca = arr[_AP_A]
    cb = arr[_AP_B]
    cc = arr[_AP_C]
    cd = arr[_AP_D]
    mask = (ca == cb) & (cb == cc) & (cc == cd)
    return np.sum(mask)

def _is_mono(arr, a, b, c, d):
    return (arr[a] == arr[b]) and (arr[b] == arr[c]) and (arr[c] == arr[d])

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    def get_time_left():
        return deadline - time.time()

    best_colours = None
    best_score = float('inf')
    
    # Helper to update best
    def update_best(cand, score):
        nonlocal best_colours, best_score
        if score < best_score:
            best_score = score
            best_colours = cand.copy()

    # Generate initial candidates
    def gen_random(seed):
        random.seed(seed)
        return np.array([random.randint(0, 1) for _ in range(_N)], dtype=np.int8)

    def gen_thue_morse():
        # Thue-Morse sequence
        t = np.zeros(_N, dtype=np.int8)
        for i in range(_N):
            t[i] = bin(i).count('1') % 2
        return t

    def gen_block(block_size):
        return np.array([(i // block_size) % 2 for i in range(_N)], dtype=np.int8)

    candidates = []
    # Thue-Morse is often good for these problems
    candidates.append(gen_thue_morse())
    
    # Various block sizes
    for bs in [2, 4, 8, 16, 32, 64]:
        candidates.append(gen_block(bs))
        
    # Random starts
    for s in range(10):
        candidates.append(gen_random(s))

    # Evaluate initial candidates
    for cand in candidates:
        score = _score(cand)
        update_best(cand, score)

    # Local Search with Delta Evaluation
    # We will run local search on the best candidate found so far
    # And also on a few other promising starts to avoid local minima
    
    def local_search(initial_arr, time_limit):
        arr = initial_arr.copy()
        current_score = _score(arr)
        
        improved = True
        # Use a list of indices to shuffle for fairness
        indices = list(range(_N))
        
        while improved and (deadline - time.time()) > time_limit:
            improved = False
            random.shuffle(indices)
            
            for i in indices:
                # Calculate delta for flipping i
                delta = 0
                # Check all APs containing i
                for ap_k in inv_ap[i]:
                    a, b, c, d = _AP_LIST[ap_k]
                    # Current state
                    is_mono_before = _is_mono(arr, a, b, c, d)
                    
                    # Flip i
                    arr[i] = 1 - arr[i]
                    
                    # New state
                    is_mono_after = _is_mono(arr, a, b, c, d)
                    
                    if is_mono_after and not is_mono_before:
                        delta += 1
                    elif is_mono_before and not is_mono_after:
                        delta -= 1
                    
                    # Unflip i to restore state for next AP check
                    arr[i] = 1 - arr[i]
                
                # If delta < 0, the flip improves the score
                if delta < 0:
                    arr[i] = 1 - arr[i]
                    current_score += delta
                    improved = True
                    if current_score < best_score:
                        update_best(arr, current_score)
                    # Restart the pass to catch new opportunities? 
                    # Or continue? Continuing is faster, but might miss immediate cascades.
                    # Let's continue for now, and do another pass.
        
        return arr, current_score

    # Run local search on the global best
    if best_colours is not None:
        # Reserve 10 seconds for final annealing
        if get_time_left() > 15.0:
            arr, score = local_search(best_colours, 10.0)
            update_best(arr, score)
            
            # Try Simulated Annealing from the best local minimum
            if get_time_left() > 5.0:
                sa_arr = best_colours.copy()
                sa_score = _score(sa_arr)
                T = 2.0
                cooling = 0.999
                # Number of steps per temperature?
                # Just run until time limit
                while get_time_left() > 1.0:
                    i = random.randint(0, _N - 1)
                    # Calculate delta
                    delta = 0
                    for ap_k in inv_ap[i]:
                        a, b, c, d = _AP_LIST[ap_k]
                        is_mono_before = _is_mono(sa_arr, a, b, c, d)
                        sa_arr[i] = 1 - sa_arr[i]
                        is_mono_after = _is_mono(sa_arr, a, b, c, d)
                        if is_mono_after and not is_mono_before:
                            delta += 1
                        elif is_mono_before and not is_mono_after:
                            delta -= 1
                        sa_arr[i] = 1 - sa_arr[i]
                    
                    if delta < 0 or random.random() < np.exp(-delta / max(T, 0.01)):
                        sa_arr[i] = 1 - sa_arr[i]
                        sa_score += delta
                        if sa_score < best_score:
                            update_best(sa_arr, sa_score)
                    T *= cooling
                    if T < 0.01:
                        T = 0.01
                        
            # Final polish with local search
            if get_time_left() > 0.5:
                arr, score = local_search(best_colours, 0.1)
                update_best(arr, score)

    return best_colours.tolist()
# EVOLVE-BLOCK-END
