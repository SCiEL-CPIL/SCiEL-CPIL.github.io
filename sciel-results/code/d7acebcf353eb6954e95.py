import time
import math
import random
import numpy as np

_START = time.time()
_DEADLINE = _START + 140.0  # Safety margin for 150s limit

def _score_from_set(U):
    """
    Calculate the score for a given set U.
    Score = 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
    Uses FFT for efficient computation.
    Returns 0.0 if invalid.
    """
    if not U or len(U) < 2:
        return 0.0
    
    max_val = max(U)
    if max_val < 1:
        return 0.0
        
    # Determine FFT size
    L = max_val + 1
    fft_len = 1
    while fft_len < 2 * L - 1:
        fft_len *= 2
        
    vec = np.zeros(fft_len, dtype=np.float64)
    for u in U:
        vec[u] = 1.0
        
    f = np.fft.fft(vec)
    
    # Sum set: Convolution of vec with vec
    conv_sum = np.fft.ifft(f * f).real
    # Sums are in indices 0 to 2*max_val
    sums_count = np.sum(conv_sum[:2*max_val+1] > 0.5)
    
    # Difference set: Autocorrelation of vec with vec
    conv_diff = np.fft.ifft(np.conj(f) * f).real
    # Count non-zero autocorrelations
    diffs_count = np.sum(conv_diff > 0.5)
    
    if sums_count == 0:
        return 0.0
        
    ratio = diffs_count / float(sums_count)
    if ratio <= 1:
        return 0.0
        
    return 1.0 + math.log(ratio) / math.log(2 * max_val + 1)

def _score_from_vector(vec):
    """
    Calculate score directly from a binary numpy vector.
    This avoids converting to a list every time.
    """
    # Find max index where vec is 1
    # vec is 1-indexed? No, let's assume vec[i] corresponds to integer i.
    # We need max(U).
    indices = np.nonzero(vec > 0.5)[0]
    if len(indices) < 2:
        return 0.0
        
    max_val = indices[-1]
    if max_val < 1:
        return 0.0
        
    L = max_val + 1
    fft_len = 1
    while fft_len < 2 * L - 1:
        fft_len *= 2
        
    # Pad vec to fft_len
    if len(vec) < fft_len:
        vec_padded = np.zeros(fft_len, dtype=np.float64)
        vec_padded[:len(vec)] = vec
    else:
        vec_padded = vec[:fft_len]
        
    f = np.fft.fft(vec_padded)
    
    # Sum set
    conv_sum = np.fft.ifft(f * f).real
    sums_count = np.sum(conv_sum[:2*max_val+1] > 0.5)
    
    # Difference set
    conv_diff = np.fft.ifft(np.conj(f) * f).real
    diffs_count = np.sum(conv_diff > 0.5)
    
    if sums_count == 0:
        return 0.0
        
    ratio = diffs_count / float(sums_count)
    if ratio <= 1:
        return 0.0
        
    return 1.0 + math.log(ratio) / math.log(2 * max_val + 1)

def _generate_initial_set(n, max_range):
    """
    Generate a random set of size n from [0, max_range].
    Always includes 0.
    """
    U = {0}
    while len(U) < n:
        U.add(random.randint(1, max_range))
    return sorted(U)

def _simulated_annealing(initial_U, max_range, time_limit):
    """
    Perform Simulated Annealing on the set U.
    State is represented as a set of integers.
    Moves:
    1. Remove a random element (not 0) and add a random new element.
    2. Move a random element to a new random position.
    """
    U = sorted(initial_U)
    current_score = _score_from_set(U)
    best_score = current_score
    best_set = U[:]
    
    start_time = time.time()
    T = 0.01  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.99
    
    # Pre-compute a set for fast lookup
    U_set = set(U)
    
    while time.time() < start_time + time_limit:
        if time.time() > _DEADLINE - 2.0:
            break
            
        # Choose a move
        if random.random() < 0.5:
            # Move an element
            if len(U) > 1:
                idx = random.randint(1, len(U) - 1) # Don't move 0
                old_val = U[idx]
                new_val = random.randint(1, max_range)
                if new_val not in U_set:
                    U_set.remove(old_val)
                    U_set.add(new_val)
                    U[idx] = new_val
                    U.sort()
        else:
            # Swap: remove one, add another
            if len(U) > 1:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                new_val = random.randint(1, max_range)
                if new_val not in U_set:
                    U_set.remove(old_val)
                    U_set.add(new_val)
                    U[idx] = new_val
                    U.sort()
                    
        new_score = _score_from_set(U)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_set = U[:]
        else:
            # Accept worse solution with probability exp(-delta/T)
            delta = current_score - new_score
            if delta < 0:
                current_score = new_score
            else:
                if random.random() < math.exp(-delta / T):
                    current_score = new_score
                else:
                    # Reject: revert
                    # We need to revert U and U_set
                    # Since we only changed one element, we can find it
                    # Actually, it's safer to keep a copy of the previous state for rejection
                    # But copying is expensive.
                    # Let's just revert the specific change.
                    # We know old_val and new_val.
                    # Wait, in the 'move' case, we replaced U[idx].
                    # In the 'swap' case, we replaced U[idx].
                    # So we just put old_val back.
                    U[idx] = old_val
                    U.sort()
                    U_set.remove(new_val)
                    U_set.add(old_val)
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    return best_set, best_score

def construct():
    global _DEADLINE
    best_score = 0.0
    best_set = [0, 1]
    
    random.seed(42)
    
    # Try different set sizes and ranges
    # Based on the problem, smaller sets might have better ratio |U-U|/|U+U|
    # but max(U) also matters in the denominator.
    # Let's explore n=5 to n=20.
    
    configs = [
        (5, 100),
        (10, 200),
        (15, 500),
        (20, 1000),
        (10, 100),
        (15, 200),
    ]
    
    time_per_config = 10.0
    
    for n, max_range in configs:
        if time.time() > _DEADLINE - 5.0:
            break
            
        # Generate a few random initializations for this config
        for _ in range(3):
            if time.time() > _DEADLINE - 5.0:
                break
                
            U_init = _generate_initial_set(n, max_range)
            U_opt, score_opt = _simulated_annealing(U_init, max_range, time_per_config / 3.0)
            
            if score_opt > best_score:
                best_score = score_opt
                best_set = U_opt[:]
                
    # Final local search polish on the best set found
    if best_score > 0:
        # Run a short local search to refine
        # Simple hill climbing
        U = best_set[:]
        current_score = best_score
        start_time = time.time()
        improved = True
        while improved and (time.time() - start_time) < 5.0:
            improved = False
            for i in range(1, len(U)):
                if time.time() > _DEADLINE - 1.0:
                    break
                old_val = U[i]
                for delta in range(-5, 6):
                    new_val = old_val + delta
                    if new_val < 1 or new_val in U:
                        continue
                    U[i] = new_val
                    U.sort()
                    new_score = _score_from_set(U)
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                        break
                    else:
                        U[i] = old_val
                        U.sort()
                
        if current_score > best_score:
            best_score = current_score
            best_set = U[:]
            
    if best_score <= 0:
        best_set = [0, 1]
        
    return best_set

if __name__ == "__main__":
    # For local testing
    pass
