"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

DEADLINE = 150.0
TIME_MARGIN = 15.0  # safety margin in seconds


def _compute_c3(x):
    """
    Compute the upper bound C3 for a signed step function x of length N.
    C3 = max_{k=1,...,N-1} |R[k]| / R[0]
    where R[k] is the autocorrelation at lag k.
    """
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(x, n=n_fft)
    power = F * np.conj(F)
    R = np.fft.irfft(power, n=n_fft)[:2 * N - 1]
    R0 = R[0]
    if R0 == 0:
        return 1.0
    sidelobes = np.concatenate([R[1:N], R[N:2*N-1]])
    max_sidelobe = np.max(np.abs(sidelobes))
    C3 = max_sidelobe / R0
    return C3


def _init_autocorr(x):
    """Initialize autocorrelation array from x using FFT."""
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(x, n=n_fft)
    power = F * np.conj(F)
    R = np.fft.irfft(power, n=n_fft)[:2 * N - 1]
    return R


def _update_autocorr_flip_fast(R, x, idx):
    """
    Vectorized update of autocorrelation R after flipping x[idx].
    O(N) but with numpy vectorization for speed.
    """
    old_val = x[idx]
    x[idx] = -old_val
    
    # For positive lags k = 1..N-1:
    # delta[k] = -2*old_val*x[idx+k] (if idx+k < N) - 2*x[idx-k]*old_val (if idx-k >= 0)
    k_pos = np.arange(1, N)
    delta_pos = np.zeros(N - 1)
    
    # Term 1: i = idx, requires idx + k < N -> k < N - idx
    mask1 = k_pos < (N - idx)
    if np.any(mask1):
        delta_pos[mask1] -= 2.0 * old_val * x[idx + k_pos[mask1]]
    
    # Term 2: i = idx-k, requires idx-k >= 0 -> k <= idx
    mask2 = k_pos <= idx
    if np.any(mask2):
        delta_pos[mask2] -= 2.0 * x[idx - k_pos[mask2]] * old_val
    
    R[1:N] += delta_pos
    
    # For negative lags: R[N+k] corresponds to correlation at lag -k
    # R[N+k] = sum_{i=k}^{N-1} x[i]*x[i-k]
    # After flipping x[idx]:
    #   If idx >= k: term i=idx: delta = -2*old_val*x[idx-k]
    #   If idx+k < N: term i=idx+k: delta = -2*x[idx+k]*old_val
    k_neg = np.arange(1, N)
    delta_neg = np.zeros(N - 1)
    
    # Term 1: i = idx, requires idx >= k
    mask1 = k_neg <= idx
    if np.any(mask1):
        delta_neg[mask1] -= 2.0 * old_val * x[idx - k_neg[mask1]]
    
    # Term 2: i = idx+k, requires idx+k < N
    mask2 = k_neg < (N - idx)
    if np.any(mask2):
        delta_neg[mask2] -= 2.0 * x[idx + k_neg[mask2]] * old_val
    
    R[N:2*N-1] += delta_neg
    
    return R, x


def _smooth_c3(x, temperature=0.5):
    """
    Smooth surrogate for C3 using soft-max.
    x is a continuous vector; we compute autocorrelation and use log-sum-exp
    for the max of |R[k]|.
    """
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(x, n=n_fft)
    power = F * np.conj(F)
    R = np.fft.irfft(power, n=n_fft)[:2 * N - 1]
    R0 = R[0]
    if abs(R0) < 1e-10:
        return 1.0
    
    sidelobes = np.concatenate([R[1:N], R[N:2*N-1]])
    abs_sidelobes = np.abs(sidelobes)
    
    # Soft max using log-sum-exp
    beta = 1.0 / temperature
    m = np.max(abs_sidelobes)
    if m < 1e-10:
        return 0.0
    log_sum_exp = m + np.log(np.sum(np.exp(beta * (abs_sidelobes - m))))
    soft_max = (1.0 / beta) * log_sum_exp
    
    return soft_max / R0


def _tanh_relaxation(x_init, max_iter=200, time_limit=None):
    """
    Smooth optimization using tanh relaxation.
    We optimize a continuous vector z, where x = tanh(z) gives values in (-1, 1).
    Then we binarize at the end.
    """
    start_time = time.time()
    
    # Initialize z from x_init: z = atanh(x) approximately
    x_clipped = np.clip(x_init, -0.999, 0.999)
    z0 = np.arctanh(x_clipped)
    
    def objective(z):
        if time.time() - start_time > time_limit:
            return 1e6
        x = np.tanh(z)
        return _smooth_c3(x, temperature=0.3)
    
    try:
        result = minimize(objective, z0, method='L-BFGS-B',
                         options={'maxiter': max_iter, 'maxfun': 5000,
                                  'maxls': 20, 'ftol': 1e-8, 'gtol': 1e-5})
        
        # Binarize
        x_opt = np.sign(np.tanh(result.x))
        x_opt[x_opt == 0] = 1.0
        
        return x_opt, _compute_c3(x_opt)
    except Exception:
        return x_init, _compute_c3(x_init)


def _logistic_sequence(n, r=3.99, x0=0.5):
    """Generate a chaotic logistic map sequence and binarize it."""
    x = np.zeros(n)
    val = x0
    for i in range(n):
        val = r * val * (1.0 - val)
        x[i] = 1.0 if val > 0.5 else -1.0
    return x


def _van_der_corput(n, base=2):
    """Generate a van der Corput low-discrepancy sequence and binarize it."""
    seq = np.zeros(n)
    for i in range(1, n + 1):
        v = 0.0
        f = 1.0 / base
        m = i
        while m > 0:
            m, r = divmod(m, base)
            v += r * f
            f /= base
        # Map [0,1) to {-1, +1}
        seq[i-1] = 1.0 if v > 0.5 else -1.0
    return seq


def _chaos_initializations():
    """Generate multiple chaotic/quasi-random initializations."""
    inits = []
    
    # Logistic map with different parameters
    for r in [3.9, 3.99, 3.999]:
        for x0 in [0.1, 0.3, 0.5, 0.7, 0.9]:
            inits.append(_logistic_sequence(N, r=r, x0=x0))
    
    # Van der Corput sequences
    inits.append(_van_der_corput(N, base=2))
    inits.append(_van_der_corput(N, base=3))
    inits.append(_van_der_corput(N, base=5))
    
    # Thue-Morse-like (alternating pattern)
    tm = np.zeros(N)
    for i in range(N):
        # Count set bits
        c = 0
        m = i
        while m:
            c += m & 1
            m >>= 1
        tm[i] = 1.0 if c % 2 == 0 else -1.0
    inits.append(tm)
    
    # Mersenne Twister based (deterministic)
    rng = np.random.RandomState(12345)
    inits.append(rng.choice([-1, 1], size=N))
    
    # Block patterns
    for block_size in [2, 4, 8, 16, 32]:
        pattern = np.tile([1, -1], N // block_size + 1)[:N]
        inits.append(pattern)
    
    # Alternating with period 3
    pattern3 = np.tile([1, 1, -1], N // 3 + 1)[:N]
    inits.append(pattern3)
    
    return inits


def _local_search(x, max_iter=50000, time_limit=None):
    """
    Local search with vectorized incremental autocorrelation update.
    Includes multi-flip moves for escaping local minima.
    """
    start_time = time.time()
    R = _init_autocorr(x)
    current_c3 = _compute_c3(x)
    best_x = x.copy()
    best_c3 = current_c3
    no_improve_count = 0
    max_no_improve = 300
    
    for iteration in range(max_iter):
        # Check time
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        # Check global deadline
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        # Choose move type: single flip (80%) or multi-flip (20%)
        if random.random() < 0.8:
            num_flips = 1
        else:
            num_flips = random.choice([2, 3, 4, 5])
        
        # Select indices to flip
        indices = random.sample(range(N), num_flips)
        
        # Apply flips and update R
        x_new = x.copy()
        R_new = R.copy()
        for idx in indices:
            R_new, x_new = _update_autocorr_flip_fast(R_new, x_new, idx)
        
        # Compute new C3 from R_new
        if abs(R_new[0]) < 1e-10:
            new_c3 = 1.0
        else:
            sidelobes = np.concatenate([R_new[1:N], R_new[N:2*N-1]])
            max_sidelobe = np.max(np.abs(sidelobes))
            new_c3 = max_sidelobe / R_new[0]
        
        if new_c3 < current_c3 - 1e-12:
            x = x_new
            R = R_new
            current_c3 = new_c3
            no_improve_count = 0
            if current_c3 < best_c3:
                best_x = x.copy()
                best_c3 = current_c3
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
    
    return best_x, best_c3


def _simulated_annealing(x, max_iter=50000, T_start=1.0, T_end=0.01, time_limit=None):
    """
    Simulated annealing with vectorized incremental autocorrelation update.
    """
    start_time = time.time()
    R = _init_autocorr(x)
    current_c3 = _compute_c3(x)
    best_x = x.copy()
    best_c3 = current_c3
    T = T_start
    
    for iteration in range(max_iter):
        # Check time
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        # Annealing schedule
        T = T_start * (T_end / T_start) ** (iteration / max_iter)
        
        # Choose move type
        if random.random() < 0.7:
            num_flips = 1
        else:
            num_flips = random.choice([2, 3, 4])
        
        indices = random.sample(range(N), num_flips)
        
        x_new = x.copy()
        R_new = R.copy()
        for idx in indices:
            R_new, x_new = _update_autocorr_flip_fast(R_new, x_new, idx)
        
        # Compute new C3 from R_new
        if abs(R_new[0]) < 1e-10:
            new_c3 = 1.0
        else:
            sidelobes = np.concatenate([R_new[1:N], R_new[N:2*N-1]])
            max_sidelobe = np.max(np.abs(sidelobes))
            new_c3 = max_sidelobe / R_new[0]
        
        delta = new_c3 - current_c3
        if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
            x = x_new
            R = R_new
            current_c3 = new_c3
            if current_c3 < best_c3:
                best_x = x.copy()
                best_c3 = current_c3
    
    return best_x, best_c3


def _construct():
    """
    Main construction function that searches for a good signed step function.
    Combines smooth tanh-relaxation optimization with discrete hill-climbing.
    """
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)
    
    best_x = np.ones(N)
    best_c3 = _compute_c3(best_x)
    
    # Strategy 1: Chaotic/quasi-random initializations + local search
    chaos_inits = _chaos_initializations()
    for i, x_init in enumerate(chaos_inits):
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        # Give more time to earlier (likely better) initializations
        time_limit = max(8.0, 20.0 - i * 1.5)
        x_trial, c3_trial = _local_search(x_init.copy(), max_iter=30000, time_limit=time_limit)
        if c3_trial < best_c3:
            best_x = x_trial
            best_c3 = c3_trial
    
    # Strategy 2: Standard random initialization + local search
    for trial in range(5):
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        x_init = np.random.choice([-1, 1], size=N)
        x2, c3_2 = _local_search(x_init, max_iter=20000, time_limit=10)
        if c3_2 < best_c3:
            best_x = x2
            best_c3 = c3_2
    
    # Strategy 3: Smooth tanh-relaxation from best solution
    if time.time() < DEADLINE - TIME_MARGIN:
        remaining = DEADLINE - TIME_MARGIN - time.time()
        x3, c3_3 = _tanh_relaxation(best_x, max_iter=100, time_limit=remaining * 0.3)
        if c3_3 < best_c3:
            best_x = x3
            best_c3 = c3_3
    
    # Strategy 4: Simulated annealing from best solution
    if time.time() < DEADLINE - TIME_MARGIN:
        remaining = DEADLINE - TIME_MARGIN - time.time()
        x4, c3_4 = _simulated_annealing(best_x, max_iter=50000, T_start=0.5, T_end=0.01,
                                         time_limit=remaining * 0.4)
        if c3_4 < best_c3:
            best_x = x4
            best_c3 = c3_4
    
    # Final local search polish
    if time.time() < DEADLINE - TIME_MARGIN:
        remaining = DEADLINE - TIME_MARGIN - time.time()
        x5, c3_5 = _local_search(best_x, max_iter=50000, time_limit=remaining)
        if c3_5 < best_c3:
            best_x = x5
            best_c3 = c3_5
    
    return best_x

def construct():
    """
    Entry point: return a sequence of signed step heights (±1) of length N=400.
    """
    x = _construct()
    # Ensure output is a numpy array of ±1 values
    return np.sign(x).astype(int)
# EVOLVE-BLOCK-END
