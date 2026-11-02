"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv_max_abs(x):
    """Compute the maximum absolute value of the autoconvolution of x."""
    # Use FFT for efficient convolution
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    X = np.fft.rfft(x, size)
    conv = np.fft.irfft(X * X, size)[:2 * n - 1]
    return np.max(np.abs(conv))


def _score(x):
    """Score = max_abs_autoconv / (sum(x))^2. Lower is better."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    return _autoconv_max_abs(x) / (s * s)


def construct():
    """
    Construct signed step heights minimizing the maximum absolute autoconvolution
    divided by squared integral.
    
    Strategy: Discrete local search on the hypercube {+1, -1}^N.
    Use stochastic coordinate descent: repeatedly flip the sign of a single element
    if it reduces the current maximum absolute autoconvolution.
    
    Key insight: The objective is max_lag |C(lag)| / (sum)^2.
    Since sum is fixed in magnitude (|sum| <= N), we focus on minimizing max |C(lag)|.
    The autoconvolution C(lag) = sum_i x_i * x_{i+lag}.
    Flipping x_j changes C(lag) by -2*x_j*(x_{j+lag} + x_{j-lag}) for each lag.
    We track the current max and try to reduce it.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 140s safety margin before 150s kill
    
    # Initialize with random signs
    rng = np.random.default_rng(42)
    x = rng.choice([-1.0, 1.0], size=N)
    best_x = x.copy()
    best_score = _score(x)
    
    # Precompute: for each position j, the contribution to autoconvolution
    # C(lag) = sum_i x_i * x_{i+lag}
    # Flipping x_j: new_C(lag) = C(lag) - 2*x_j*(x_{j+lag} + x_{j-lag})
    # where x_{j+lag} means x[j+lag] if in range, else 0
    
    # We'll use a greedy approach: at each step, find the flip that most reduces
    # the current peak autoconvolution value.
    
    # For efficiency, we'll do batches of coordinate flips.
    # Track the current autoconvolution to avoid recomputing from scratch.
    # But recomputing via FFT each time is O(N log N), which for N=400 is fast.
    
    # Let's use a simpler but effective approach:
    # 1. Start with random signs
    # 2. Repeatedly: find the single flip that most reduces max|C|, apply it
    # 3. Use simulated annealing to escape local minima
    
    # First, get initial autoconvolution
    def compute_conv(x):
        n = len(x)
        size = 1
        while size < 2 * n - 1:
            size *= 2
        X = np.fft.rfft(x, size)
        conv = np.fft.irfft(X * X, size)[:2 * n - 1]
        return conv
    
    conv = compute_conv(x)
    current_max = np.max(np.abs(conv))
    best_max = current_max
    best_x = x.copy()
    
    # Simulated annealing on the hypercube
    T = 50.0  # initial temperature
    cooling = 0.995
    min_T = 0.1
    
    # Precompute neighbor indices for efficiency
    # For flipping position j, the change in C(lag) is -2*x_j*(x_{j+lag} + x_{j-lag})
    # We want to find the j that most reduces the peak.
    
    # Strategy: At each step, evaluate a random subset of positions for flipping
    # and pick the best one (greedy) or accept with probability (annealing).
    
    iters = 0
    while time.time() < deadline and T > min_T:
        iters += 1
        
        # Check deadline more frequently
        if iters % 10 == 0 and time.time() > deadline - 5.0:
            break
        
        # Try a random set of candidate positions
        n_candidates = min(N, 50)
        candidates = rng.choice(N, size=n_candidates, replace=False)
        
        best_delta = 0.0  # reduction in max|C|
        best_j = -1
        best_new_conv = None
        
        for j in candidates:
            # Compute what happens if we flip x[j]
            old_val = x[j]
            # Change in C(lag): delta_C(lag) = -2 * old_val * (x[j+lag] + x[j-lag])
            # We need to compute this for all lags
            # For lag k: delta_C(k) = -2 * old_val * (x[j+k] if j+k < N else 0 + x[j-k] if j-k >= 0 else 0)
            
            # Efficient computation:
            # The contribution of x[j] to C is:
            # C_k includes x[j]*x[j+k] (if j+k < N) and x[j-k]*x[j] (if j-k >= 0)
            # So flipping x[j] changes C_k by -2*x[j]*(x[j+k] + x[j-k]) where out-of-range terms are 0
            
            new_conv = conv.copy()
            # For each lag k from -(N-1) to (N-1):
            # In our conv array, index i corresponds to lag i - (N-1)
            # C(lag) = sum_i x_i * x_{i+lag}
            # The terms involving x[j] are:
            #   x[j] * x[j+lag] for lag such that j+lag < N
            #   x[j-lag] * x[j] for lag such that j-lag >= 0
            # Both contribute to C(lag)
            
            # For lag k (where k can be negative):
            # term1: if j+k >= 0 and j+k < N: x[j+k]
            # term2: if j-k >= 0 and j-k < N: x[j-k]
            # delta_C(k) = -2 * x[j] * (term1 + term2)
            
            # Let's compute this vectorized:
            # lags range from -(N-1) to (N-1)
            # In conv array: index i = lag + (N-1), so lag = i - (N-1)
            
            lags = np.arange(-(N-1), N)  # shape (2N-1,)
            
            # term1: x[j+k] valid if 0 <= j+k < N
            idx1 = j + lags
            term1 = np.where((idx1 >= 0) & (idx1 < N), x[np.clip(idx1, 0, N-1)], 0.0)
            
            # term2: x[j-k] valid if 0 <= j-k < N
            idx2 = j - lags
            term2 = np.where((idx2 >= 0) & (idx2 < N), x[np.clip(idx2, 0, N-1)], 0.0)
            
            delta = -2.0 * old_val * (term1 + term2)
            new_conv = conv + delta
            
            new_max = np.max(np.abs(new_conv))
            delta_max = current_max - new_max  # positive means improvement
            
            if delta_max > best_delta:
                best_delta = delta_max
                best_j = j
                best_new_conv = new_conv
        
        # Apply the best flip or use annealing
        if best_j >= 0:
            # Accept if it improves, or with probability exp(-delta/T) if it worsens
            if best_delta > 0 or (T > min_T and rng.random() < np.exp(-max(0, -best_delta) / T)):
                x[best_j] = -x[best_j]
                conv = best_new_conv
                current_max = np.max(np.abs(conv))
                
                if current_max < best_max:
                    best_max = current_max
                    best_x = x.copy()
        
        T *= cooling
    
    # Final polish: greedy descent until no improvement
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for j in range(N):
            old_val = x[j]
            lags = np.arange(-(N-1), N)
            idx1 = j + lags
            term1 = np.where((idx1 >= 0) & (idx1 < N), x[np.clip(idx1, 0, N-1)], 0.0)
            idx2 = j - lags
            term2 = np.where((idx2 >= 0) & (idx2 < N), x[np.clip(idx2, 0, N-1)], 0.0)
            delta = -2.0 * old_val * (term1 + term2)
            new_conv = conv + delta
            new_max = np.max(np.abs(new_conv))
            
            if new_max < current_max - 1e-12:
                x[j] = -x[j]
                conv = new_conv
                current_max = new_max
                improved = True
                if current_max < best_max:
                    best_max = current_max
                    best_x = x.copy()
    
    # Return the best solution found
    return best_x
# EVOLVE-BLOCK-END
