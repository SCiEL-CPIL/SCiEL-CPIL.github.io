"""Minimize monochromatic 4-term arithmetic progressions in binary coloring of [1..256]."""
import numpy as np
import time
import random

_N = 256

# Precompute all 4-term APs as index arrays
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
_NUM_APS = len(_AP_A)

def _score(arr):
    """Count monochromatic 4-term APs. arr: numpy int8 array of 0/1, length 256."""
    c0 = arr[_AP_A]
    c1 = arr[_AP_B]
    c2 = arr[_AP_C]
    c3 = arr[_AP_D]
    mask = (c0 == c1) & (c1 == c2) & (c2 == c3)
    return int(np.sum(mask))

def _delta_flip(arr, i):
    """
    Compute the change in monochromatic 4-AP count if we flip position i.
    Returns (delta, new_arr) where new_arr is a copy with the flip applied.
    """
    # Find all APs that include position i
    # Position i can be at index a, a+d, a+2d, or a+3d in an AP
    # For each role, find all APs where the position equals i
    
    # We need to check each AP that contains position i
    # Role 0: a = i, so AP is (i, i+d, i+2d, i+3d) for valid d
    # Role 1: a+d = i, so a = i-d, AP is (i-d, i, i+d, i+2d)
    # Role 2: a+2d = i, so a = i-2d, AP is (i-2d, i-d, i, i+d)
    # Role 3: a+3d = i, so a = i-3d, AP is (i-3d, i-2d, i-d, i)
    
    delta = 0
    c_i = arr[i]
    
    # Role 0: i is the first element
    for d in range(1, (_N - 1 - i) // 3 + 1):
        a = i
        if a + 3*d >= _N:
            break
        c1 = arr[i + d]
        c2 = arr[i + 2*d]
        c3 = arr[i + 3*d]
        # Before flip: all equal?
        before_mono = (c_i == c1) and (c1 == c2) and (c2 == c3)
        # After flip: c_i becomes 1-c_i
        after_mono = ((1 - c_i) == c1) and (c1 == c2) and (c2 == c3)
        if before_mono and not after_mono:
            delta -= 1
        elif not before_mono and after_mono:
            delta += 1
    
    # Role 1: i is the second element (a+d = i)
    for d in range(1, i + 1):
        a = i - d
        if a < 0:
            break
        if a + 3*d >= _N:
            continue
        c0 = arr[a]
        c2 = arr[i + d]
        c3 = arr[i + 2*d]
        before_mono = (c0 == c_i) and (c_i == c2) and (c2 == c3)
        after_mono = (c0 == (1 - c_i)) and ((1 - c_i) == c2) and (c2 == c3)
        if before_mono and not after_mono:
            delta -= 1
        elif not before_mono and after_mono:
            delta += 1
    
    # Role 2: i is the third element (a+2d = i)
    for d in range(1, (i + 1) // 2 + 1):
        a = i - 2*d
        if a < 0:
            break
        if a + 3*d >= _N:
            continue
        c0 = arr[a]
        c1 = arr[i - d]
        c3 = arr[i + d]
        before_mono = (c0 == c1) and (c1 == c_i) and (c_i == c3)
        after_mono = (c0 == c1) and (c1 == (1 - c_i)) and ((1 - c_i) == c3)
        if before_mono and not after_mono:
            delta -= 1
        elif not before_mono and after_mono:
            delta += 1
    
    # Role 3: i is the fourth element (a+3d = i)
    for d in range(1, (i + 1) // 3 + 1):
        a = i - 3*d
        if a < 0:
            break
        c0 = arr[a]
        c1 = arr[i - 2*d]
        c2 = arr[i - d]
        before_mono = (c0 == c1) and (c1 == c2) and (c2 == c_i)
        after_mono = (c0 == c1) and (c1 == c2) and (c2 == (1 - c_i))
        if before_mono and not after_mono:
            delta -= 1
        elif not before_mono and after_mono:
            delta += 1
    
    return delta

def _evaluate(colours_list):
    """
    colours_list: list of 256 values in {0,1}
    Returns the count of monochromatic 4-term APs.
    """
    arr = np.array(colours_list, dtype=np.int8)
    return _score(arr)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    random.seed(42)
    
    # Generate several initial colorings
    candidates = []
    
    # 1. Thue-Morse
    def thue_morse(n):
        return bin(n).count('1') % 2
    tm = [thue_morse(i) for i in range(_N)]
    candidates.append(tm)
    
    # 2. Alternating
    alt = [i % 2 for i in range(_N)]
    candidates.append(alt)
    
    # 3. Block patterns
    for block_size in [2, 3, 4, 5, 6, 7, 8, 12, 16]:
        block = [(i // block_size) % 2 for i in range(_N)]
        candidates.append(block)
    
    # 4. Random initializations
    for seed in range(20):
        random.seed(seed)
        rc = [random.randint(0, 1) for _ in range(_N)]
        candidates.append(rc)
    
    # 5. Greedy construction: at each step, choose color that minimizes new mono-4-APs
    # This is expensive but let's try a simpler version
    # Actually, let's use a different approach: start random, then hill-climb
    
    # Evaluate all candidates
    best_colours = None
    best_score = float('inf')
    
    for cand in candidates:
        arr = np.array(cand, dtype=np.int8)
        s = _score(arr)
        if s < best_score:
            best_score = s
            best_colours = cand[:]
    
    # Local search with delta evaluation (much faster)
    def local_search(colours, max_time):
        """Hill-climbing with delta evaluation."""
        arr = np.array(colours, dtype=np.int8)
        current_score = _score(arr)
        improved = True
        while improved and time.time() < max_time:
            improved = False
            for i in range(_N):
                if time.time() > max_time:
                    break
                delta = _delta_flip(arr, i)
                if delta < 0:
                    arr[i] = 1 - arr[i]
                    current_score += delta
                    improved = True
        return arr.tolist(), current_score
    
    # Run local search from best candidate
    if time.time() < deadline - 20.0:
        best_colours, best_score = local_search(best_colours, deadline - 20.0)
    
    # Simulated annealing
    if time.time() < deadline - 10.0:
        current = np.array(best_colours, dtype=np.int8)
        current_score = best_score
        temp = 5.0
        cooling = 0.995
        
        while time.time() < deadline - 5.0 and temp > 0.05:
            i = random.randint(0, _N - 1)
            delta = _delta_flip(current, i)
            current[i] = 1 - current[i]
            new_score = current_score + delta
            
            if delta < 0 or random.random() < np.exp(-delta / max(temp, 0.01)):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_colours = current.tolist()
            else:
                current[i] = 1 - current[i]  # revert
            
            temp *= cooling
    
    # Final local search pass
    if time.time() < deadline - 2.0:
        best_colours, best_score = local_search(best_colours, deadline - 2.0)
    
    # Verify
    final_arr = np.array(best_colours, dtype=np.int8)
    final_score = _score(final_arr)
    if final_score != best_score:
        best_score = final_score
    
    return best_colours
# EVOLVE-BLOCK-END
