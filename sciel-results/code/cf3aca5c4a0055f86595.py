import time
import math
import random
import numpy as np

def _score_slow(U):
    """Fallback scorer using sets, for small sets or verification."""
    U = list(set(int(x) for x in U))
    if 0 not in U:
        U.append(0)
    if not U:
        return 0.0
    mx = max(U)
    if mx <= 0:
        return 0.0
    n = len(U)
    plus = set()
    minus = set()
    for a in U:
        for b in U:
            plus.add(a + b)
            minus.add(a - b)
    nplus = len(plus)
    nminus = len(minus)
    if nplus == 0:
        return 0.0
    log_den = math.log(2 * mx + 1)
    if log_den <= 0:
        return 0.0
    ratio = nminus / nplus
    if ratio <= 0:
        return 0.0
    val = 1.0 + math.log(ratio) / log_den
    return val

def _score_fft(U, max_val):
    """
    Fast scorer using FFT convolution.
    U: list of integers
    max_val: current max element (or upper bound)
    Returns score.
    """
    n = len(U)
    if n == 0:
        return 0.0
    mx = max(U)
    if mx <= 0:
        return 0.0
    
    # For very small sets, direct calculation is faster than FFT overhead
    if n < 8 or mx < 20:
        return _score_slow(U)
        
    # We need to compute |U+U| and |U-U|.
    # U+U is the support of the convolution of indicator(U) with itself.
    # U-U is the support of the cross-correlation, or convolution with reflected U.
    
    # Determine FFT size.
    # Max sum is 2*mx. Max diff is mx (in magnitude, so range -mx to mx).
    # We can use a single FFT size that covers the sum.
    # Let N_fft be a power of 2 >= 2*mx + 1.
    
    size = 1
    while size < 2 * mx + 1:
        size *= 2
        
    # Create indicator vector
    vec = np.zeros(size, dtype=np.float64)
    for u in U:
        vec[u] = 1.0
        
    # FFT
    v_fft = np.fft.rfft(vec)
    
    # Sumset: U+U
    # Convolution of vec with vec
    sum_fft = v_fft * v_fft
    sum_conv = np.fft.irfft(sum_fft, n=size)
    # The support is where sum_conv > 0.5 (since coefficients are integers >= 1)
    nplus = np.sum(sum_conv > 0.5)
    
    # Difference set: U-U
    # This is the convolution of vec with reflected vec.
    # reflected_vec[i] = vec[-i mod size]? 
    # Actually, U-U = {a-b}.
    # Let's create a reflected vector.
    # If we define rev_vec such that rev_vec[k] = 1 if k is in -U.
    # Since U contains non-negative integers, -U contains non-positive.
    # In the circular convolution of size 'size', index k corresponds to value k if k <= size/2, 
    # and k - size if k > size/2.
    # We want the support of {a-b}.
    # Note that a-b can be negative.
    # The convolution (vec * rev_vec)[k] counts pairs (a,b) such that a - b = k (mod size)?
    # Let's be precise.
    # Let f(x) = 1 if x in U.
    # Let g(x) = 1 if x in -U.
    # Then (f * g)(z) = sum_x f(x)g(z-x) = sum_x f(x)f(x-z) = sum_x f(x)f(x-z).
    # This is the number of pairs (x, y) such that x - y = z.
    # So the support of the convolution of f and g is U-U.
    
    # Construct g (reflected vec)
    # g[k] = 1 if k in -U.
    # Since 0 in U, -0 = 0 is in -U.
    # For u in U, -u is in -U.
    # In the array of size 'size', index for value v is v if v >= 0, else size + v.
    # Since u >= 0, -u <= 0.
    # Index for -u is size - u (if u > 0) or 0 (if u == 0).
    
    rev_vec = np.zeros(size, dtype=np.float64)
    for u in U:
        if u == 0:
            rev_vec[0] = 1.0
        else:
            rev_vec[size - u] = 1.0
            
    g_fft = np.fft.rfft(rev_vec)
    diff_fft = v_fft * g_fft
    diff_conv = np.fft.irfft(diff_fft, n=size)
    
    # The support of U-U is where diff_conv > 0.5
    # Note: The convolution is circular.
    # Since max |a-b| <= mx < size/2, there is no wrap-around ambiguity for the support.
    # The values a-b are in [-mx, mx].
    # In the array, these map to [size-mx, size-1] U [0, mx].
    # np.sum(diff_conv > 0.5) will correctly count these indices.
    
    nminus = np.sum(diff_conv > 0.5)
    
    if nplus == 0:
        return 0.0
        
    log_den = math.log(2 * mx + 1)
    if log_den <= 0:
        return 0.0
    ratio = nminus / nplus
    if ratio <= 0:
        return 0.0
    val = 1.0 + math.log(ratio) / log_den
    return val

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin
    
    # Initial set: Try a few good seeds
    # The paper mentions size 2003 and 54265.
    # Small sets like {0, 1, 3} score ~1.028.
    # Let's start with a lacunary-like set which tends to have large differences.
    # e.g., powers of 3: 0, 1, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683, 59049
    # Max ~ 59049.
    
    # Let's generate a few candidates to pick the best start.
    candidates = [
        [0, 1, 3],
        [0, 1, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683, 59049],
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256],
    ]
    
    best_val = -1e9
    best_set = [0, 1, 3]
    
    for c in candidates:
        v = _score_fft(c, max(c))
        if v > best_val:
            best_val = v
            best_set = list(c)
            
    # Simulated Annealing
    current_set = list(best_set)
    current_val = best_val
    global_best_set = list(best_set)
    global_best_val = best_val
    
    # Parameters
    # We need to define the state space.
    # The set size and max value can change.
    # Let's define a maximum possible max value, M_max.
    # If M is too large, FFT is slow.
    # If M is too small, we might miss good structures.
    # Let's cap M at 100,000. FFT size 2^17 = 131,072.
    # rfft of 131k is fast (~ms).
    # We can do many iterations.
    
    M_max = 100000
    
    # Temperature schedule
    # Start high, decay exponentially.
    T_start = 1.0
    T_end = 0.01
    # We will run for a fixed number of steps or time.
    # Let's estimate speed.
    # One eval ~ 1-2 ms (FFT + Python loop overhead).
    # 140 seconds / 0.002s = 70,000 steps.
    # Let's target 50,000 steps.
    
    steps = 50000
    # But we must check time.
    
    rng = random.Random(42)
    
    # To make perturbations meaningful, we need to handle the variable size of the set.
    # Representation: sorted list of integers.
    # Perturbations:
    # 1. Add a random element in [1, M_max]
    # 2. Remove a random element (not 0)
    # 3. Shift a random element by +/- delta
    
    # We must ensure 0 is always present.
    
    for i in range(steps):
        if time.time() > deadline:
            break
            
        # Temperature decay
        # Linear decay in i/steps is simple.
        # Or exponential.
        # Let's use exponential cooling: T = T_start * (T_end/T_start)^(i/steps)
        T = T_start * (T_end / T_start) ** (i / steps)
        
        # Generate perturbation
        U = list(current_set)
        # Ensure 0 is in U
        if 0 not in U:
            U.append(0)
            
        r = rng.random()
        
        if r < 0.4:
            # Add element
            # Bias towards smaller elements? Or uniform?
            # Uniform in [1, M_max] might add huge gaps.
            # Let's try uniform in [1, max(U)*2] capped at M_max.
            mx = max(U) if U else 1
            upper = min(M_max, mx * 2 + 100)
            new_el = rng.randint(1, upper)
            if new_el not in U:
                U.append(new_el)
        elif r < 0.7:
            # Remove element
            if len(U) > 1:
                # Don't remove 0
                removable = [x for x in U if x != 0]
                if removable:
                    rem_el = rng.choice(removable)
                    U.remove(rem_el)
        else:
            # Shift element
            if len(U) > 1:
                shiftable = [x for x in U if x != 0]
                if shiftable:
                    sh_el = rng.choice(shiftable)
                    # Small shift
                    delta = rng.randint(-10, 10)
                    new_val = sh_el + delta
                    if new_val >= 0 and new_val != 0:
                        if new_val in U:
                            # Collision, skip or resolve
                            pass
                        else:
                            U.remove(sh_el)
                            U.append(new_val)
            
        # Sort for consistent representation (though set logic handles it)
        U = list(set(U)) # Ensure uniqueness
        if 0 not in U:
            U.append(0)
            
        if not U:
            continue
            
        mx = max(U)
        
        # Evaluate
        try:
            v = _score_fft(U, mx)
        except Exception:
            continue
            
        if v > global_best_val:
            global_best_val = v
            global_best_set = list(U)
            
        # Metropolis criterion
        delta_val = v - current_val
        if delta_val > 0:
            current_set = U
            current_val = v
        else:
            # Accept with probability exp(delta_val / T)
            # If T is very small, this is 0.
            # If delta_val is very negative, exp is 0.
            if T > 0:
                # Avoid overflow/underflow
                # delta_val is likely small relative to T initially, but T decays.
                # If delta_val / T < -50, exp is ~0.
                ratio = delta_val / T
                if ratio > -50:
                    if rng.random() < math.exp(ratio):
                        current_set = U
                        current_val = v
                        
    # Final verification with slow method for the best set found
    # This ensures we return a valid, correctly scored set.
    # The FFT method might have floating point issues or edge cases.
    # Also, the score function in the evaluator uses exact sets.
    
    final_set = list(set(int(x) for x in global_best_set))
    if 0 not in final_set:
        final_set.append(0)
    final_set.sort()
    
    # One last check: maybe a simple hill climb from the best found?
    # If we have time left.
    if time.time() < deadline - 5.0:
        cur = list(final_set)
        cur_val = _score_slow(cur) # Use slow to be sure
        best_local_val = cur_val
        best_local_set = cur
        
        improved = True
        it = 0
        while improved and time.time() < deadline - 2.0 and it < 50:
            improved = False
            it += 1
            mx = max(cur)
            # Try adding neighbors
            candidates = []
            for d in range(1, 20):
                candidates.append(mx + d)
            for d in range(1, 20):
                candidates.append(mx - d)
            # Random
            for _ in range(20):
                candidates.append(rng.randint(1, mx * 2))
                
            for c in candidates:
                if c < 0 or c == 0:
                    continue
                U_new = cur + [c]
                U_new = list(set(U_new))
                if 0 not in U_new:
                    U_new.append(0)
                v = _score_slow(U_new)
                if v > best_local_val + 1e-12:
                    best_local_val = v
                    best_local_set = U_new
                    cur = U_new
                    improved = True
                    break
        
        if best_local_val > global_best_val:
            final_set = best_local_set
            
    return final_set

# EVOLVE-BLOCK-END
