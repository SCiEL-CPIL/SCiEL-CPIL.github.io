import numpy as np
import time

N = 400

def _compute_acf(h):
    """Compute the autocorrelation (autoconvolution) of h."""
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    return acf[:N]

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as +/-1 array."""
    # Thue-Morse: t(0)=0, t(2n)=t(n), t(2n+1)=1-t(n)
    # Convert to +/-1: 0->+1, 1->-1
    tm = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        # Count trailing ones or use bit manipulation
        # Simpler: t(i) = popcount(i) % 2
        tm[i] = bin(i).count('1') % 2
    return 1 - 2 * tm  # 0->1, 1->-1

def _vectorized_greedy(h, acf, deadline):
    """
    Vectorized greedy local search.
    For each position i, compute the effect of flipping h[i] on all acf[k].
    Use vectorized operations to evaluate all flips simultaneously.
    """
    current_score = np.max(np.abs(acf[1:N]))
    
    # Precompute neighbor sums for fast evaluation
    # For flip at i: delta_acf[k] = -2 * h[i] * (h[i+k] + h[i-k]) for valid k
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_gain = 1e-9
        best_idx = -1
        
        # For each i, compute the resulting max |acf| if we flip i
        # Vectorize: create matrix where row i, col k = |acf[k] - 2*h[i]*(h[i+k]+h[i-k])|
        
        # Efficient approach: for each i, compute the change vector
        # change[i, k] = -2 * h[i] * (h[i+k] + h[i-k]) for valid (i,k)
        
        # Build the change matrix using broadcasting
        # indices: i from 0 to N-1, k from 1 to N-1
        i_idx = np.arange(N)
        k_idx = np.arange(1, N)
        
        # For each (i, k), valid if i+k < N (for h[i+k]) or i-k >= 0 (for h[i-k])
        # We need to handle boundary conditions
        
        # Create mask for valid pairs
        # i+k < N means k < N-i
        # i-k >= 0 means k <= i
        
        # Use advanced indexing
        I, K = np.meshgrid(i_idx, k_idx, indexing='ij')
        
        # Valid conditions
        valid_plus = (I + K) < N  # h[i+k] exists
        valid_minus = (I - K) >= 0  # h[i-k] exists
        
        # Compute neighbor sum: h[i+k] + h[i-k] where valid
        neighbor_sum = np.zeros((N, N-1), dtype=np.float64)
        
        # For valid_plus: add h[i+k]
        # We need to safely index h[I+K] where valid_plus
        safe_I_plus = np.where(valid_plus, I + K, 0)
        safe_I_minus = np.where(valid_minus, I - K, 0)
        
        neighbor_sum = np.where(valid_plus, h[safe_I_plus], 0.0) + \
                       np.where(valid_minus, h[safe_I_minus], 0.0)
        
        # Change in acf if we flip position i:
        # delta_acf[k] = -2 * h[i] * neighbor_sum[i, k]
        # new_acf[k] = acf[k] + delta_acf[k]
        
        h_col = h[:, np.newaxis]  # shape (N, 1)
        delta = -2.0 * h_col * neighbor_sum  # shape (N, N-1)
        
        new_acf_matrix = acf[1:N][np.newaxis, :] + delta  # shape (N, N-1)
        
        # Compute max |new_acf| for each flip
        new_scores = np.max(np.abs(new_acf_matrix), axis=1)  # shape (N,)
        
        # Find best flip
        gains = current_score - new_scores
        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]
        
        if best_gain > 1e-9:
            # Apply the flip
            h[best_idx] *= -1
            acf = _compute_acf(h)
            current_score = np.max(np.abs(acf[1:N]))
            improved = True
    
    return h, acf, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_h = None
    best_score = float('inf')
    
    # Phase 1: Thue-Morse initialization with vectorized greedy
    h = _thue_morse(N).astype(np.float64)
    acf = _compute_acf(h)
    h, acf, current_score = _vectorized_greedy(h, acf, deadline - 10)
    
    if current_score < best_score:
        best_score = current_score
        best_h = h.copy()
    
    # Phase 2: Try a few random restarts if time permits
    rng = np.random.RandomState(42)
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        h = rng.choice([-1.0, 1.0], size=N)
        acf = _compute_acf(h)
        h, acf, current_score = _vectorized_greedy(h, acf, deadline - 5)
        
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
        
        if best_score < 22:
            break
    
    # Phase 3: Try shifted Thue-Morse if time permits
    if time.time() < deadline - 15 and best_score > 23:
        for shift in [1, 2, 3]:
            if time.time() > deadline - 10:
                break
            h = _thue_morse(N).astype(np.float64)
            h = np.roll(h, shift)
            acf = _compute_acf(h)
            h, acf, current_score = _vectorized_greedy(h, acf, deadline - 5)
            
            if current_score < best_score:
                best_score = current_score
                best_h = h.copy()
    
    if best_h is None:
        return np.ones(N)
        
    return best_h
