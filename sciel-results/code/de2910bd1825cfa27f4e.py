import numpy as np
import time

N = 400

def compute_score(x, exclude_lag0=True):
    """
    Computes the score: max |autoconvolution| / (sum x)^2
    """
    integral = np.sum(x)
    if integral == 0:
        return float('inf')
    
    # Linear autoconvolution
    # Use FFT for speed
    n_fft = 2 * N
    X = np.fft.fft(x, n_fft)
    C = np.fft.ifft(X * X).real
    
    if exclude_lag0:
        # Exclude lag 0 (index 0)
        # Also exclude negative lags? Autoconvolution is symmetric.
        # Max absolute value over k != 0
        # C[0] is the peak.
        # We look at C[1:N] and C[N+1:2N-1] (wrapped? No, linear is 0 to 2N-1)
        # Linear conv length is 2N-1.
        # C[0] corresponds to lag 0.
        # C[1:N-1] are positive lags.
        # C[N:2N-1] are negative lags (symmetric).
        # So max over C[1:N] is sufficient due to symmetry.
        max_abs = np.max(np.abs(C[1:N]))
    else:
        max_abs = np.max(np.abs(C[0:N]))
        
    return max_abs / (integral * integral)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_score = float('inf')
    best_x = np.ones(N)
    
    # Candidate 1: All Ones
    s1 = compute_score(best_x)
    if s1 < best_score:
        best_score = s1
        best_x = best_x.copy()
        
    # Candidate 2: Gaussian
    # Try different sigmas
    for sigma in [5, 10, 20, 50, 100]:
        x_gauss = np.exp(-0.5 * (np.arange(N) - N/2)**2 / (sigma**2))
        # Normalize to have sum 1? No, score is scale invariant?
        # max(C) scales with x^2. I^2 scales with x^2. Ratio is invariant.
        s_gauss = compute_score(x_gauss)
        if s_gauss < best_score:
            best_score = s_gauss
            best_x = x_gauss.copy()
            
    # Candidate 3: Local Search starting from All Ones
    # We want to minimize score.
    # Flip bits to -1 to reduce sidelobes.
    x_curr = np.ones(N)
    curr_score = compute_score(x_curr)
    
    # If exclude_lag0 is True, All Ones is a local minimum?
    # Flipping a bit reduces I, and changes C.
    
    # Simulated Annealing / Random Hill Climbing
    x_opt = np.ones(N)
    opt_score = compute_score(x_opt)
    
    # Try random flips
    rng = np.random.default_rng(42)
    
    # Time budget for search
    max_iters = 5000
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
            
        # Randomly flip 1-5 bits
        num_flips = rng.integers(1, 6)
        indices = rng.choice(N, num_flips, replace=False)
        
        x_new = x_opt.copy()
        x_new[indices] *= -1
        
        new_score = compute_score(x_new)
        
        if new_score < opt_score:
            x_opt = x_new
            opt_score = new_score
            
            if opt_score < best_score:
                best_score = opt_score
                best_x = x_opt.copy()
                
        # Optional: Accept worse solutions with some probability (SA)
        # But simple hill climbing might get stuck.
        # Given the time, let's stick to greedy or simple SA.
        # Let's try a bit of SA:
        # T = 0.1 * (1 - i/max_iters)
        # if np.log(np.random.rand()) < (opt_score - new_score) / T:
        #    x_opt = x_new
        #    opt_score = new_score
            
    # Candidate 4: LFSR based
    # Generate m-sequence of length 511
    # Map to {-1, 1}
    # Take first 400
    # This has low periodic autocorrelation.
    # Linear autocorrelation of a truncated m-sequence is not ideal, but might be better than random.
    
    # Primitive polynomial for degree 9: x^9 + x^4 + 1
    # Or use a simple LFSR implementation
    
    def lfsr_sequence(length, degree=9, poly=0b100010001, seed=1):
        # poly bits: 1 (x^9) + 0 + 0 + 0 + 1 (x^4) + 0 + 0 + 0 + 1 (x^0)
        # Actually standard representation:
        # Let's use a known primitive polynomial.
        # For degree 9, x^9 + x^4 + 1 is primitive.
        # Coefficients: 1 0 0 0 1 0 0 0 1
        # Feedback taps at 4 and 0 (0-indexed from LSB? No, standard is x^9 + x^4 + 1)
        # Let's implement a generic LFSR.
        
        # State is degree bits.
        state = [1] + [0]*(degree-1)
        seq = []
        for _ in range(length):
            # Output last bit
            out = state[-1]
            seq.append(out)
            # Feedback
            # x^9 = x^4 + 1
            # new_bit = state[4] XOR state[0] ?
            # Depends on convention.
            # Let's just use a known good sequence if possible, or random if LFSR is buggy.
            # To be safe, I'll use a random sequence if LFSR is complex.
            # But the prompt suggested LFSR.
            # Let's try a simple LFSR.
            # Tap positions for degree 9: 4, 0.
            # new_bit = state[4] ^ state[0]
            new_bit = state[4] ^ state[0]
            state.pop()
            state.insert(0, new_bit)
            
        # Map 0->1, 1->-1
        return np.array([1 if b==0 else -1 for b in seq])
        
    try:
        x_lfsr = lfsr_sequence(N)
        # The integral of a truncated m-sequence is small.
        # To improve score, we might want to flip the sign of the whole sequence?
        # Or flip individual bits to increase I?
        # Let's just evaluate it.
        s_lfsr = compute_score(x_lfsr)
        if s_lfsr < best_score:
            best_score = s_lfsr
            best_x = x_lfsr.copy()
            
        # Try to optimize LFSR by flipping bits to increase I
        # Start with x_lfsr, flip bits to 1 if they are -1?
        # This moves towards all-ones.
        # We can do a local search starting from x_lfsr.
        x_opt2 = x_lfsr.copy()
        opt_score2 = compute_score(x_opt2)
        
        for i in range(1000):
            if time.time() > deadline:
                break
            idx = rng.integers(0, N)
            x_new = x_opt2.copy()
            x_new[idx] *= -1
            new_score = compute_score(x_new)
            if new_score < opt_score2:
                x_opt2 = x_new
                opt_score2 = new_score
                if opt_score2 < best_score:
                    best_score = opt_score2
                    best_x = x_opt2.copy()
    except Exception:
        pass
        
    return best_x

# EVOLVE-BLOCK-START
# (The code above is the content of the evolve block)
# EVOLVE-BLOCK-END
