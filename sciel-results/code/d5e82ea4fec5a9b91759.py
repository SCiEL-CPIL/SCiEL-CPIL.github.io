import numpy as np
import time

N = 400
DEADLINE = 140.0

def _calc_acf(h):
    # h is +/- 1
    # Pad to 2*N to get linear autocorrelation
    h_padded = np.zeros(2*N)
    h_padded[:N] = h
    H = np.fft.fft(h_padded)
    acf = np.fft.ifft(H * np.conj(H)).real
    # The linear autocorrelation for lags 0 to N-1 is in acf[:N]
    # Note: acf[N:] contains the negative lags (symmetric)
    return acf[:N]

def _score(h):
    acf = _calc_acf(h)
    # Peak sidelobe: max absolute value for k != 0
    if len(acf) > 1:
        return np.max(np.abs(acf[1:]))
    return 0.0

def _make_sine_pattern(frequency):
    """Generate a signed step function based on a sine wave."""
    t = np.arange(N)
    signal = np.sin(2 * np.pi * frequency * t / N)
    # Map to +/- 1
    # Use sign, but handle 0s (rare) by keeping previous or random
    h = np.sign(signal)
    h[h == 0] = 1.0
    return h

def construct():
    start_time = time.time()
    
    # 1. Structured Initialization: Try a few sine frequencies and pick the best
    np.random.seed(42)
    candidates = []
    for freq in [1, 2, 3, 5, 7, 13, 21, 34]:
        h = _make_sine_pattern(freq)
        candidates.append(h)
    
    # Also add a random one just in case
    candidates.append(np.random.choice([-1, 1], N))
    
    best_h = None
    best_score = float('inf')
    for h in candidates:
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
            
    current_h = best_h.copy()
    current_score = best_score
    
    # Simulated Annealing with Block Flips
    T = 5.0 # Higher start temp to allow larger jumps
    T_min = 0.1
    cooling = 0.9995
    
    def elapsed_ok():
        return (time.time() - start_time) < DEADLINE

    while elapsed_ok():
        # Move: Flip a random block of bits
        # Block size between 1 and N//10
        max_block = max(1, N // 10)
        block_size = np.random.randint(1, max_block + 1)
        start_idx = np.random.randint(0, N - block_size + 1)
        
        new_h = current_h.copy()
        new_h[start_idx:start_idx + block_size] *= -1
        
        new_score = _score(new_h)
        
        if new_score < current_score:
            current_h = new_h
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = new_h.copy()
        else:
            delta = new_score - current_score
            if delta == 0:
                current_h = new_h
                current_score = new_score
            elif np.random.rand() < np.exp(-delta / T):
                current_h = new_h
                current_score = new_score
        
        T *= cooling
        if T < T_min:
            # Reheat
            T = T_min * 10
            # Optional: Random restart if stuck for too long, but let's just reheat for now
            # If we reheat, we might want to nudge the state
            if np.random.rand() < 0.1:
                 # Small random perturbation
                 idx = np.random.randint(0, N)
                 current_h[idx] *= -1
                 current_score = _score(current_h)

    # Greedy Polish
    # Try flipping each bit to see if it improves
    improved = True
    while improved and elapsed_ok():
        improved = False
        for i in range(N):
            if not elapsed_ok():
                break
            # Flip i
            best_h[i] *= -1
            new_score = _score(best_h)
            if new_score < best_score:
                best_score = new_score
                improved = True
            else:
                # Revert
                best_h[i] *= -1

    return best_h.astype(np.float64)
