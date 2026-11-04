"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_max_abs(f):
    """
    Compute the upper bound on C3, defined as the maximum absolute value
    of the aperiodic autoconvolution (excluding the zero-lag self-term).
    """
    n = len(f)
    # Use FFT for efficient linear convolution
    size = 2 * n
    F = np.fft.fft(f, n=size)
    conv = np.fft.ifft(F * F).real[:n]
    
    # conv[0] is the zero-lag term (sum of squares).
    # We want the maximum over the non-zero lags.
    if n > 1:
        return np.max(np.abs(conv[1:]))
    return 0.0

def _fast_flip_score(f, i, current_score):
    """
    Estimate the new score after flipping index i.
    Note: This is an approximation or a fast check. For exactness in local search,
    we might need to recompute, but for speed in many iterations, 
    we can use the fact that flipping one element changes the convolution in a structured way.
    However, recomputing the max is O(N log N). 
    Given N=400, recomputing is fast enough.
    """
    f_new = f.copy()
    f_new[i] *= -1.0
    return _autoconv_max_abs(f_new)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill

    rng = np.random.RandomState(42)
    
    # Strategy: 
    # 1. Generate a structured initial candidate using a quasi-periodic 
    #    complex exponential sum (inspired by the directive).
    # 2. Refine using fast local search (inspired by current program's robustness 
    #    and inspiration 2/3's metric).
    
    # --- Phase 1: Structured Initialization ---
    # Represent h as sign of real part of sum of complex exponentials
    # h_k = sign( Re( sum_{m=1}^{M} A_m exp(i * 2*pi * f_m * k) ) )
    # We fix irrational frequencies to avoid periodicity.
    
    # Try a few different harmonic structures to find a good starting point
    best_f = np.ones(N)
    best_score = _autoconv_max_abs(best_f)
    
    # Candidate 1: Simple alternating (baseline)
    alt = np.ones(N)
    alt[1::2] = -1.0
    score_alt = _autoconv_max_abs(alt)
    if score_alt < best_score:
        best_score = score_alt
        best_f = alt.copy()
        
    # Candidate 2: Quasi-periodic structure
    # Use small number of harmonics with irrational frequencies
    num_harmonics = 5
    # Frequencies chosen to be incommensurate (e.g., based on square roots or golden ratio)
    # k is in [0, N-1]
    k = np.arange(N)
    
    # Fixed random but deterministic frequencies
    # f_m = sqrt(m) / N * something? 
    # Let's use f_m = (m * phi) mod 1, where phi is golden ratio
    phi = (1 + np.sqrt(5)) / 2.0
    freqs = np.array([(m * phi) % 1.0 for m in range(1, num_harmonics + 1)])
    
    # Amplitudes can be optimized or fixed. Let's start with fixed equal amplitudes.
    # We can also try optimizing amplitudes quickly.
    
    signal = np.zeros(N)
    for m in range(num_harmonics):
        signal += np.cos(2 * np.pi * freqs[m] * k)
        
    h_structured = np.sign(signal)
    # Handle zeros (if any) by assigning +1
    h_structured[h_structured == 0] = 1.0
    
    score_structured = _autoconv_max_abs(h_structured)
    if score_structured < best_score:
        best_score = score_structured
        best_f = h_structured.copy()
        
    # Candidate 3: Random initialization
    rand_f = rng.choice([-1.0, 1.0], size=N)
    score_rand = _autoconv_max_abs(rand_f)
    if score_rand < best_score:
        best_score = score_rand
        best_f = rand_f.copy()
        
    # --- Phase 2: Local Search Refinement ---
    # Use simulated annealing or simple hill climbing with random restarts.
    # Since N=400, computing the score is O(N log N) which is very fast.
    # We can afford many iterations.
    
    current_f = best_f.copy()
    current_score = best_score
    
    T = 10.0
    T_min = 0.1
    cooling = 0.999
    
    # Single flip moves
    while time.time() < deadline - 5.0:
        # Pick a random index to flip
        i = rng.randint(N)
        
        old_val = current_f[i]
        current_f[i] = -old_val
        new_score = _autoconv_max_abs(current_f)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_f = current_f.copy()
        else:
            # Accept worse move with probability exp(-delta/T)
            delta = new_score - current_score
            if T > T_min and rng.random_sample() < np.exp(-delta / T):
                current_score = new_score
            else:
                current_f[i] = old_val
                
        T *= cooling
        if T < T_min:
            T = T_min
            
    # --- Phase 3: Pairwise Flip Moves (if time permits) ---
    # Flipping two elements at once can escape local minima that single flips cannot.
    if time.time() < deadline - 5.0:
        # Use a fixed number of iterations or time-based
        iterations = 5000
        for _ in range(iterations):
            if time.time() > deadline - 5.0:
                break
            
            i, j = rng.choice(N, 2, replace=False)
            
            vi = current_f[i]
            vj = current_f[j]
            
            current_f[i] = -vi
            current_f[j] = -vj
            
            new_score = _autoconv_max_abs(current_f)
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_f = current_f.copy()
            else:
                current_f[i] = vi
                current_f[j] = vj
                
    return best_f
# EVOLVE-BLOCK-END
