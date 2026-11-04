import numpy as np
import time

N = 400

def _autoconv(x):
    # Linear autoconvolution
    # Using FFT for speed
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    X = np.fft.fft(x, size)
    C = np.fft.ifft(X * X).real
    return C[:2*n-1]

def _metric(x):
    S = np.sum(x)
    if S == 0:
        return float('inf')
    c = _autoconv(x)
    # Exclude k=0
    max_c = np.max(np.abs(c[1:]))
    return max_c / (S * S)

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # Start with all ones
    x = np.ones(N, dtype=np.int8)
    best_x = x.copy()
    best_score = _metric(best_x)
    
    # Local Search
    # We want to minimize score
    # Try flipping bits
    # Heuristic: Focus on reducing the max peak
    
    current_score = best_score
    x = best_x.copy()
    
    # Precompute indices
    indices = np.arange(N)
    
    while time.time() < deadline:
        c = _autoconv(x)
        # Find max peak (excluding 0)
        max_val = np.max(np.abs(c[1:]))
        if max_val == 0:
            break
        
        # Find index of max peak
        # c is length 2N-1
        # k ranges from 1 to 2N-2
        # c[k] corresponds to lag k
        # We want to reduce c[k_max]
        k_max = np.argmax(np.abs(c[1:])) + 1
        target_c = c[k_max]
        
        # We want to flip x[i] such that it reduces |c[k_max]|
        # Delta c[k] = -4 * x[i] * x[i+k] (if i+k < N)
        # We need to check valid i
        valid_i = indices[indices + k_max < N]
        
        if len(valid_i) == 0:
             # Try other k? Or break
             break
             
        # Calculate potential delta for all valid i
        # x[i] * x[i+k_max]
        prod = x[valid_i] * x[valid_i + k_max]
        # Delta = -4 * prod
        # New c = c + Delta
        # We want to minimize |c + Delta|
        # If c > 0, we want Delta < 0 => prod > 0
        # If c < 0, we want Delta > 0 => prod < 0
        
        # Score of each flip
        # New max might change elsewhere, so this is a heuristic
        # Let's just try the flip that gives the largest reduction in |c[k_max]|
        
        # prod is 1 or -1
        # If target_c > 0, we want prod = 1 (Delta = -4)
        # If target_c < 0, we want prod = -1 (Delta = 4)
        
        if target_c > 0:
            candidates = valid_i[prod == 1]
        else:
            candidates = valid_i[prod == -1]
            
        if len(candidates) == 0:
            # No good flip for this k
            # Try random flip
            if np.random.rand() < 0.1:
                 i = np.random.choice(N)
                 x[i] *= -1
            else:
                 # Stuck? Try next max k
                 c_sorted = np.argsort(np.abs(c[1:]))[::-1]
                 # Skip k_max
                 k_max = c_sorted[0] + 1
                 if k_max == 0: k_max = 1
                 target_c = c[k_max]
                 valid_i = indices[indices + k_max < N]
                 if len(valid_i) == 0: continue
                 prod = x[valid_i] * x[valid_i + k_max]
                 if target_c > 0:
                     candidates = valid_i[prod == 1]
                 else:
                     candidates = valid_i[prod == -1]
                 if len(candidates) == 0:
                     continue

        # Pick a candidate
        i = np.random.choice(candidates)
        
        # Apply flip
        x[i] *= -1
        
        # Check score
        new_score = _metric(x)
        if new_score < current_score:
            current_score = new_score
            best_x = x.copy()
            best_score = current_score
        else:
            # Revert
            x[i] *= -1
            
    return best_x

# EVOLVE-BLOCK-START
import numpy as np
import time

N = 400

def _autoconv(x):
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    X = np.fft.fft(x, size)
    C = np.fft.ifft(X * X).real
    return C[:2*n-1]

def _metric(x):
    S = np.sum(x)
    if S == 0:
        return float('inf')
    c = _autoconv(x)
    # Exclude k=0
    max_c = np.max(np.abs(c[1:]))
    return max_c / (S * S)

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # Start with all ones
    x = np.ones(N, dtype=np.int8)
    best_x = x.copy()
    best_score = _metric(best_x)
    
    current_score = best_score
    x = best_x.copy()
    
    indices = np.arange(N)
    
    while time.time() < deadline:
        c = _autoconv(x)
        max_val = np.max(np.abs(c[1:]))
        if max_val == 0:
            break
        
        k_max = np.argmax(np.abs(c[1:])) + 1
        target_c = c[k_max]
        
        valid_i = indices[indices + k_max < N]
        
        if len(valid_i) == 0:
            break
        prod = x[valid_i] * x[valid_i + k_max]
        
        if target_c > 0:
            candidates = valid_i[prod == 1]
        else:
            candidates = valid_i[prod == -1]
            
        if len(candidates) == 0:
            continue

        i = np.random.choice(candidates)
        x[i] *= -1
        
        new_score = _metric(x)
        if new_score < current_score:
            current_score = new_score
            best_x = x.copy()
            best_score = current_score
        else:
            x[i] *= -1
            
    return best_x
# EVOLVE-BLOCK-END
