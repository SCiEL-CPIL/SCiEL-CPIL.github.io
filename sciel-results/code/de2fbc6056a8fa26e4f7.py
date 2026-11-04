"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

TIME_LIMIT = 140  # safety margin before 150s kill


def _compute_c3_bound(heights):
    """
    Compute the upper bound on C3 from the autoconvolution of the signed step function.
    """
    h = np.asarray(heights, dtype=np.float64)
    n = len(h)
    dx = 0.5 / n  # total length 0.5, n intervals
    
    # Linear autoconvolution via FFT
    f_fft = np.fft.fft(h, n=2 * n)
    conv = np.fft.ifft(f_fft * np.conj(f_fft)).real
    
    max_abs = np.max(np.abs(conv))
    c3_bound = dx * max_abs
    
    return c3_bound


def _compute_conv(h):
    """Compute the full autoconvolution of h."""
    n = len(h)
    f_fft = np.fft.fft(h, n=2 * n)
    conv = np.fft.ifft(f_fft * np.conj(f_fft)).real
    return conv


def _max_abs_conv(conv):
    """Get max absolute value of convolution."""
    return np.max(np.abs(conv))


def _try_flip_fast(h, conv, idx):
    """
    Compute the change in max|conv| if we flip h[idx].
    Flipping h[idx] from v to -v changes conv by:
    delta_conv[k] = -2 * v * h[idx + k] for k where idx+k in [0, n-1]
    and delta_conv[k] = -2 * v * h[idx - k] for k where idx-k in [0, n-1]
    More precisely, conv[k] = sum_i h[i]*h[i+k], so flipping h[idx] changes:
    conv[k] -= 2*v*h[idx+k] for k such that 0 <= idx+k < n
    conv[k] += 2*v*h[idx-k] for k such that 0 <= idx-k < n (this is conv[-k])
    
    Actually, let's think more carefully. The autoconvolution is:
    conv[m] = sum_{i} h[i] * h[i+m] for m in [-n+1, n-1]
    
    If we flip h[idx] (change v to -v), then:
    conv[m] changes by -2*v*h[idx+m] for each m where 0 <= idx+m < n
    conv[m] changes by -2*v*h[idx-m] for each m where 0 <= idx-m < n
    
    Wait, let me be more careful. conv[m] = sum_i h[i]*h[i+m].
    The terms involving h[idx] are:
    - When i = idx: h[idx]*h[idx+m]
    - When i+m = idx, i.e., i = idx-m: h[idx-m]*h[idx]
    
    So flipping h[idx] from v to -v:
    conv[m] changes by: (-v - v)*h[idx+m] + h[idx-m]*(-v - v) = -2v*h[idx+m] - 2v*h[idx-m]
    for valid indices.
    
    This is O(n) per flip evaluation, but we can precompute the affected conv values.
    """
    n = len(h)
    v = h[idx]
    
    # The change to conv at lag m is:
    # delta[m] = -2*v*(h[idx+m] if 0<=idx+m<n else 0) - 2*v*(h[idx-m] if 0<=idx-m<n else 0)
    # But conv is indexed from -(n-1) to (n-1), stored as 0 to 2n-2
    # conv_stored[k] = conv[k - (n-1)]
    
    # For efficiency, just compute the new max by updating only affected lags
    # Affected lags: m where idx+m in [0,n-1] or idx-m in [0,n-1]
    # That's m in [-idx, n-1-idx] union [idx-n+1, idx]
    
    # Simple approach: compute delta for all affected positions
    new_conv = conv.copy()
    
    # m such that 0 <= idx+m < n => m in [-idx, n-1-idx]
    m_start = max(-(n-1), -idx)
    m_end = min(n-1, n-1-idx)
    if m_start <= m_end:
        m_vals = np.arange(m_start, m_end + 1)
        idx_plus = idx + m_vals
        delta1 = -2 * v * h[idx_plus]
        # Convert m to stored index
        stored_idx = m_vals + (n-1)
        new_conv[stored_idx] += delta1
    
    # m such that 0 <= idx-m < n => m in [idx-n+1, idx]
    m_start2 = max(-(n-1), idx-n+1)
    m_end2 = min(n-1, idx)
    if m_start2 <= m_end2:
        m_vals2 = np.arange(m_start2, m_end2 + 1)
        idx_minus = idx - m_vals2
        delta2 = -2 * v * h[idx_minus]
        stored_idx2 = m_vals2 + (n-1)
        new_conv[stored_idx2] += delta2
    
    return _max_abs_conv(new_conv)


def _local_search_fast(h, conv, max_iter=50000, time_budget=10):
    """
    Fast local search using incremental convolution updates.
    """
    h = np.array(h, dtype=np.float64)
    conv = _compute_conv(h)
    current_max = _max_abs_conv(conv)
    start_time = time.time()
    
    improved = True
    iters = 0
    no_improve_count = 0
    
    while improved and iters < max_iter:
        if time.time() - start_time > time_budget:
            break
        
        improved = False
        iters += 1
        best_gain = 0
        best_idx = -1
        
        # Try flipping each position
        for i in range(N):
            if time.time() - start_time > time_budget:
                break
            new_max = _try_flip_fast(h, conv, i)
            gain = current_max - new_max
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        
        if best_idx >= 0:
            # Apply the flip
            h[best_idx] = -h[best_idx]
            conv = _compute_conv(h)  # Recompute to avoid accumulated errors
            current_max = _max_abs_conv(conv)
            improved = True
            no_improve_count = 0
        else:
            no_improve_count += 1
            # If stuck for a while, try a random perturbation
            if no_improve_count > 50:
                for _ in range(5):
                    rand_idx = np.random.randint(0, N)
                    h[rand_idx] = -h[rand_idx]
                conv = _compute_conv(h)
                current_max = _max_abs_conv(conv)
                no_improve_count = 0
    
    return h, conv, current_max


def _simulated_annealing(h, conv, time_budget=30):
    """
    Simulated annealing to escape local minima.
    """
    h = np.array(h, dtype=np.float64)
    conv = _compute_conv(h)
    current_max = _max_abs_conv(conv)
    best_h = h.copy()
    best_conv = conv.copy()
    best_max = current_max
    
    start_time = time.time()
    T = 0.5  # Initial temperature
    cooling = 0.9995
    
    while time.time() - start_time < time_budget:
        if T < 0.01:
            break
        
        # Random flip
        idx = np.random.randint(0, N)
        h[idx] = -h[idx]
        new_conv = _compute_conv(h)
        new_max = _max_abs_conv(new_conv)
        
        delta = new_max - current_max
        
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            current_max = new_max
            conv = new_conv
            if new_max < best_max:
                best_max = new_max
                best_h = h.copy()
                best_conv = conv.copy()
        
        T *= cooling
    
    return best_h, best_conv, best_max


def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as +/-1 values."""
    seq = [1]
    while len(seq) < n:
        seq += [-x for x in seq]
    return np.array(seq[:n], dtype=np.float64)


def _chaotic_sequence(n, r=3.99, x0=0.12345, seed=42):
    """
    Generate a signed sequence from the logistic map near the edge of chaos.
    """
    x = x0
    seq = np.empty(n, dtype=np.float64)
    for i in range(n):
        seq[i] = 1.0 if x > 0.5 else -1.0
        x = r * x * (1.0 - x)
        if x < 0:
            x = 0.0
        elif x > 1:
            x = 1.0
    return seq


def _golay_like(n):
    """
    Generate a Golay-like sequence using the recursive construction.
    Golay sequences have optimal autocorrelation properties.
    """
    # Start with length 2 Golay sequence
    seq = np.array([1, 1], dtype=np.float64)
    
    # Build up to length >= n using Golay construction
    # A+B, A-B where A and B are complementary Golay sequences
    # For simplicity, use the known Golay pair construction
    
    # Actually, let's just use a good pseudo-random initialization
    # with some structure
    np.random.seed(123)
    seq = np.random.choice([-1.0, 1.0], size=n)
    return seq


def construct():
    start_time = time.time()
    
    # Multiple initializations
    candidates = []
    
    # Thue-Morse
    tm = _thue_morse(N)
    candidates.append(tm)
    
    # Chaotic sequences
    chaotic_params = [
        (3.99, 0.12345),
        (3.999, 0.12345),
        (3.99, 0.45678),
        (3.999, 0.78901),
        (3.95, 0.12345),
    ]
    for r, x0 in chaotic_params:
        if time.time() - start_time > TIME_LIMIT - 40:
            break
        candidates.append(_chaotic_sequence(N, r=r, x0=x0))
    
    # Random initializations
    np.random.seed(42)
    for trial in range(10):
        if time.time() - start_time > TIME_LIMIT - 40:
            break
        candidates.append(np.random.choice([-1.0, 1.0], size=N))
    
    # Find best initialization
    best_h = None
    best_max = np.inf
    for h in candidates:
        conv = _compute_conv(h)
        m = _max_abs_conv(conv)
        if m < best_max:
            best_max = m
            best_h = h.copy()
    
    if best_h is None:
        best_h = np.ones(N)
        best_max = _max_abs_conv(_compute_conv(best_h))
    
    # Local search with fast updates
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 20:
        local_time = min(remaining - 15, 40)
        best_h, best_conv, best_max = _local_search_fast(
            best_h, _compute_conv(best_h), 
            max_iter=100000, time_budget=local_time
        )
    
    # Simulated annealing to escape local minima
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 25:
        sa_time = min(remaining - 15, 30)
        sa_h, sa_conv, sa_max = _simulated_annealing(
            best_h, _compute_conv(best_h), time_budget=sa_time
        )
        if sa_max < best_max:
            best_max = sa_max
            best_h = sa_h.copy()
    
    # Another local search polish
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 15:
        polish_time = min(remaining - 10, 20)
        best_h, best_conv, best_max = _local_search_fast(
            best_h, _compute_conv(best_h),
            max_iter=100000, time_budget=polish_time
        )
    
    # Restart with different initializations if time permits
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 30:
        np.random.seed(999)
        for trial in range(3):
            if time.time() - start_time > TIME_LIMIT - 15:
                break
            
            h = np.random.choice([-1.0, 1.0], size=N)
            conv = _compute_conv(h)
            m = _max_abs_conv(conv)
            
            if m < best_max:
                local_time = min(TIME_LIMIT - (time.time() - start_time) - 10, 15)
                h, conv, m = _local_search_fast(
                    h, conv, max_iter=50000, time_budget=local_time
                )
                if m < best_max:
                    best_max = m
                    best_h = h.copy()
    
    # Final polish
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 10:
        polish_time = min(remaining - 5, 10)
        best_h, best_conv, best_max = _local_search_fast(
            best_h, _compute_conv(best_h),
            max_iter=50000, time_budget=polish_time
        )
    
    # Final verification
    final_conv = _compute_conv(best_h)
    final_max = _max_abs_conv(final_conv)
    
    return best_h


# Test
if __name__ == "__main__":
    h = construct()
    print(f"Length: {len(h)}")
    print(f"First 20: {h[:20]}")
    print(f"C3 bound: {_compute_c3_bound(h)}")
    print(f"Score: {1.0 / _compute_c3_bound(h)}")
# EVOLVE-BLOCK-END
