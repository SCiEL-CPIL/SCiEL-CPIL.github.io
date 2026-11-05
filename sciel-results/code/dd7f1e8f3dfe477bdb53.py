import numpy as np
import time

N = 400
FFT_SIZE = 1024  # Power of 2, larger than 2*N-1

def _compute_peak(x):
    """
    Computes the maximum absolute value of the linear autocorrelation
    for lags k != 0.
    """
    x_padded = np.zeros(FFT_SIZE)
    x_padded[:N] = x
    
    X = np.fft.rfft(x_padded)
    P = X * np.conjugate(X)
    R = np.fft.irfft(P, FFT_SIZE)
    
    # R[0] is the DC term (sum of squares = N)
    # R[1:N] are the positive lags
    lags = R[1:N]
    
    return np.max(np.abs(lags))

def _thue_morse(n):
    """
    Generate the Thue-Morse sequence of length n.
    The Thue-Morse sequence is 0,1,1,0,1,0,0,1,...
    We map 0 -> -1, 1 -> 1 for our signed step function.
    """
    t = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        t[i] = t[i >> 1] ^ (i & 1)
    return 1 - 2 * t  # Map {0,1} to {1,-1}

def _compute_grad(x):
    """
    Compute the gradient of the peak autocorrelation with respect to x.
    This is an approximation. The true gradient is complex, but we can
    use the autocorrelation itself to guide flips.
    
    Actually, for a better approach, we can use the fact that flipping x[i]
    changes the autocorrelation R[k] by -2*x[i]*x[i+k] for k>0 and -2*x[i]*x[i-k] for k<0.
    So the change in R[k] is -2*x[i]*(x[i+k] + x[i-k]) (with boundary handling).
    
    However, the objective is max |R[k]|, which is non-differentiable.
    A common heuristic is to minimize the sum of squares of the autocorrelation,
    which is differentiable. But the task asks for the upper bound on C3, which
    is related to the peak.
    
    Let's stick with a more direct approach: use the autocorrelation values to
    determine which flips are most promising.
    """
    x_padded = np.zeros(FFT_SIZE)
    x_padded[:N] = x
    X = np.fft.rfft(x_padded)
    P = X * np.conjugate(X)
    R = np.fft.irfft(P, FFT_SIZE)
    return R

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for 120s limit
    
    best_x = None
    best_score = float('inf')
    rng = np.random.default_rng(42)
    
    # Start with Thue-Morse sequence, which is known to have good autocorrelation properties
    x_tm = _thue_morse(N).astype(np.float64)
    current_score = _compute_peak(x_tm)
    best_x = x_tm.copy()
    best_score = current_score
    
    # Try a few random initializations as well
    for _ in range(5):
        if time.time() > deadline:
            break
        x = rng.choice([-1, 1], N)
        score = _compute_peak(x)
        if score < best_score:
            best_score = score
            best_x = x.copy()
    
    # Local search: try flipping each bit and keep the best improvement
    x = best_x.copy()
    current_score = best_score
    
    # We'll do multiple passes of local search
    for iteration in range(50):  # Max 50 passes
        if time.time() > deadline:
            break
        
        improved = False
        # Try flipping each bit
        for i in range(N):
            if time.time() > deadline:
                break
            
            old_val = x[i]
            x[i] = -old_val
            new_score = _compute_peak(x)
            x[i] = old_val
            
            if new_score < current_score - 1e-12:
                # Accept the flip
                x[i] = -old_val
                current_score = new_score
                improved = True
                
                if current_score < best_score:
                    best_score = current_score
                    best_x = x.copy()
        
        if not improved:
            # No improvement in this pass, try a random restart from current state
            # or break if we've been stuck for a while
            if iteration > 5:
                break
            # Perturb: flip a few random bits to escape local minimum
            for _ in range(10):
                idx = rng.integers(N)
                x[idx] = -x[idx]
            current_score = _compute_peak(x)
    
    return best_x

# EVOLVE-BLOCK-START
# The code above is the complete program.
# The construct function is the entry point.
# EVOLVE-BLOCK-END
