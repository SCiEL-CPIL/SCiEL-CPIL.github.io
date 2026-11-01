"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

DEADLINE_MARGIN = 10.0  # seconds before the 150s kill


def _autocorr_full(h):
    """
    Compute full linear autocorrelation using FFT.
    Returns array of length 2*N-1 where R[k] = sum_i h[i]*h[i+k] for k=0..N-1
    and R[N+k] = sum_i h[i+k]*h[i] for k=0..N-2 (symmetric).
    """
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    R = np.fft.irfft(power, n=n_fft).real[:2 * N - 1]
    return R


def _compute_c3(h):
    """
    Compute the upper bound C3 = max |R[k]| / R[0] for k != 0.
    R[0] = sum h[i]^2 = N for ±1 signals.
    """
    R = _autocorr_full(h)
    R0 = R[0]
    if R0 == 0:
        return 1.0
    # Sidelobes: R[1:N] (positive lags) and R[N:2*N-1] (negative lags)
    # But R[1:N] and R[N:2*N-1] are the same by symmetry for real signals
    # Actually R[k] = R[2*N-1-k], so we just need R[1:N]
    max_sidelobe = np.max(np.abs(R[1:N]))
    return max_sidelobe / R0


def _thue_morse(n):
    """Generate Thue-Morse sequence: 1 if popcount(i) is even, -1 otherwise."""
    seq = np.zeros(n)
    for i in range(n):
        c = bin(i).count('1')
        seq[i] = 1.0 if c % 2 == 0 else -1.0
    return seq


def _m_sequence(n):
    """
    Generate an m-sequence (maximal length LFSR sequence) of length n.
    Uses a primitive polynomial for a good balance of ±1 values.
    """
    # Use a simple LFSR with a known primitive polynomial
    # For n=400, we need a sequence of length >= 400
    # Use period 2^8 - 1 = 255, or 2^9-1 = 511
    # Let's use 2^9-1 = 511 and take first n
    # Primitive polynomial x^9 + x^4 + 1 (one example)
    # Tap positions: 9, 4, 1 (1-indexed, LSB is position 1)
    # Actually let's use a well-known one: x^9 + x^5 + x^4 + x^3 + 1
    # Taps: 9, 5, 4, 3
    
    # Simpler: use a known good LFSR
    # x^8 + x^6 + x^5 + x^4 + 1 has period 255
    # Let's just use multiple m-sequences combined
    
    # Actually, for a cleaner approach, use a de Bruijn-like or just 
    # generate from a known primitive polynomial
    
    # Use x^12 + x^6 + x^4 + x + 1 (period 4095)
    # Taps at positions 12, 6, 4, 1
    seq = np.zeros(n)
    state = 1  # 12-bit register
    for i in range(n):
        # Output bit
        seq[i] = 1.0 if (state & 1) == 0 else -1.0
        # Feedback
        fb = ((state >> 11) ^ (state >> 5) ^ (state >> 3) ^ (state >> 0)) & 1
        state = ((state >> 1) | (fb << 11)) & 0xFFF
    
    return seq


def _chaotic_sequence(n, r=3.99, x0=0.5):
    """Generate chaotic logistic map sequence, binarized."""
    seq = np.zeros(n)
    val = x0
    for i in range(n):
        val = r * val * (1.0 - val)
        seq[i] = 1.0 if val > 0.5 else -1.0
    return seq


def _van_der_corput(n, base=2):
    """Van der Corput low-discrepancy sequence, binarized."""
    seq = np.zeros(n)
    for i in range(1, n + 1):
        v = 0.0
        f = 1.0 / base
        m = i
        while m > 0:
            m, r = divmod(m, base)
            v += r * f
            f /= base
        seq[i - 1] = 1.0 if v > 0.5 else -1.0
    return seq


def _block_pattern(n, block_size):
    """Alternating blocks of +1 and -1."""
    seq = np.zeros(n)
    for i in range(n):
        block_idx = i // block_size
        seq[i] = 1.0 if block_idx % 2 == 0 else -1.0
    return seq


def _generate_initializations():
    """Generate diverse initializations."""
    inits = []
    
    # Thue-Morse
    inits.append(_thue_morse(N))
    
    # M-sequence
    inits.append(_m_sequence(N))
    
    # Chaotic sequences with different parameters
    for r in [3.9, 3.99, 3.999]:
        for x0 in [0.1, 0.3, 0.5, 0.7, 0.9]:
            inits.append(_chaotic_sequence(N, r=r, x0=x0))
    
    # Van der Corput
    for base in [2, 3, 5, 7]:
        inits.append(_van_der_corput(N, base=base))
    
    # Block patterns
    for bs in [1, 2, 4, 8, 16, 32]:
        inits.append(_block_pattern(N, bs))
    
    # Random
    rng = np.random.RandomState(42)
    inits.append(rng.choice([-1, 1], size=N).astype(np.float64))
    inits.append(rng.choice([-1, 1], size=N).astype(np.float64))
    
    return inits


def _local_search(x, max_iter=20000, time_limit=None):
    """
    Fast local search: flip single positions, accept if C3 improves.
    Uses incremental autocorrelation update for speed.
    """
    start_time = time.time()
    
    # Precompute autocorrelation
    R = _autocorr_full(x)
    R0 = R[0]
    current_c3 = np.max(np.abs(R[1:N])) / R0
    best_x = x.copy()
    best_c3 = current_c3
    no_improve = 0
    max_no_improve = 500
    
    for iteration in range(max_iter):
        # Time check
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        if time.time() > time.time() + DEADLINE_MARGIN:  # This won't work, use absolute
            pass
        
        # Pick random index to flip
        idx = np.random.randint(N)
        old_val = x[idx]
        x[idx] = -old_val
        
        # Update autocorrelation incrementally
        # R[k] = sum_i x[i]*x[i+k]
        # When we flip x[idx], the change in R[k] is:
        # For lag k: x[idx] appears in R[k] as x[idx]*x[idx+k] (if idx+k < N)
        #            and as x[idx-k]*x[idx] (if idx-k >= 0)
        # So delta R[k] = -2*old_val*x[idx+k] - 2*old_val*x[idx-k]
        # (since new_val = -old_val, change = new_val*x[idx+k] - old_val*x[idx+k] = -2*old_val*x[idx+k])
        
        # Actually let's just recompute for simplicity, but use FFT
        # For N=400, FFT is fast enough
        R_new = _autocorr_full(x)
        new_c3 = np.max(np.abs(R_new[1:N])) / R0
        
        if new_c3 < current_c3 - 1e-12:
            current_c3 = new_c3
            R = R_new
            no_improve = 0
            if current_c3 < best_c3:
                best_c3 = current_c3
                best_x = x.copy()
        else:
            # Revert
            x[idx] = old_val
            no_improve += 1
            if no_improve > max_no_improve:
                break
    
    return best_x, best_c3


def _simulated_annealing(x, max_iter=50000, T_start=0.1, T_end=0.001, time_limit=None):
    """
    Simulated annealing with single-flip moves.
    """
    start_time = time.time()
    R = _autocorr_full(x)
    R0 = R[0]
    current_c3 = np.max(np.abs(R[1:N])) / R0
    best_x = x.copy()
    best_c3 = current_c3
    T = T_start
    
    for iteration in range(max_iter):
        # Time check
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        # Annealing schedule
        T = T_start * (T_end / T_start) ** (iteration / max_iter)
        
        # Pick random index
        idx = np.random.randint(N)
        old_val = x[idx]
        x[idx] = -old_val
        
        R_new = _autocorr_full(x)
        new_c3 = np.max(np.abs(R_new[1:N])) / R0
        
        delta = new_c3 - current_c3
        if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-10)):
            current_c3 = new_c3
            R = R_new
            if current_c3 < best_c3:
                best_c3 = current_c3
                best_x = x.copy()
        else:
            x[idx] = old_val
    
    return best_x, best_c3


def _pairwise_exchange(x, max_iter=10000, time_limit=None):
    """
    Local search with pairwise exchanges (flip two positions).
    """
    start_time = time.time()
    R = _autocorr_full(x)
    R0 = R[0]
    current_c3 = np.max(np.abs(R[1:N])) / R0
    best_x = x.copy()
    best_c3 = current_c3
    no_improve = 0
    max_no_improve = 300
    
    for iteration in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        # Pick two random indices
        i1, i2 = np.random.choice(N, 2, replace=False)
        x[i1] = -x[i1]
        x[i2] = -x[i2]
        
        R_new = _autocorr_full(x)
        new_c3 = np.max(np.abs(R_new[1:N])) / R0
        
        if new_c3 < current_c3 - 1e-12:
            current_c3 = new_c3
            R = R_new
            no_improve = 0
            if current_c3 < best_c3:
                best_c3 = current_c3
                best_x = x.copy()
        else:
            x[i1] = -x[i1]
            x[i2] = -x[i2]
            no_improve += 1
            if no_improve > max_no_improve:
                break
    
    return best_x, best_c3


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    np.random.seed(42)
    
    best_x = None
    best_c3 = np.inf
    
    # Generate initializations
    inits = _generate_initializations()
    
    # Phase 1: Local search from diverse initializations
    for i, x_init in enumerate(inits):
        if time.time() > deadline:
            break
        
        # Allocate time based on remaining budget
        remaining = deadline - time.time()
        num_remaining = len(inits) - i
        time_per_init = max(5.0, remaining / max(num_remaining, 1) * 0.8)
        
        x_trial, c3_trial = _local_search(x_init.copy(), max_iter=5000, time_limit=time_per_init)
        
        if c3_trial < best_c3:
            best_c3 = c3_trial
            best_x = x_trial.copy()
    
    # Phase 2: Simulated annealing from best
    if best_x is not None and time.time() < deadline - 20:
        time_remaining = deadline - time.time()
        x_sa, c3_sa = _simulated_annealing(best_x.copy(), max_iter=30000, 
                                           T_start=0.1, T_end=0.001,
                                           time_limit=time_remaining * 0.5)
        if c3_sa < best_c3:
            best_c3 = c3_sa
            best_x = x_sa.copy()
    
    # Phase 3: Pairwise exchange refinement
    if best_x is not None and time.time() < deadline - 10:
        time_remaining = deadline - time.time()
        x_pe, c3_pe = _pairwise_exchange(best_x.copy(), max_iter=20000,
                                          time_limit=time_remaining * 0.6)
        if c3_pe < best_c3:
            best_c3 = c3_pe
            best_x = x_pe.copy()
    
    # Phase 4: Final local search polish
    if best_x is not None and time.time() < deadline - 5:
        time_remaining = deadline - time.time()
        x_final, c3_final = _local_search(best_x.copy(), max_iter=10000,
                                           time_limit=time_remaining * 0.8)
        if c3_final < best_c3:
            best_c3 = c3_final
            best_x = x_final.copy()
    
    # Fallback
    if best_x is None:
        best_x = np.ones(N)
    
    return best_x
# EVOLVE-BLOCK-END
