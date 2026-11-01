"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

DEADLINE = 150.0
TIME_MARGIN = 5.0  # safety margin in seconds


def _compute_c3(x):
    """
    Compute the upper bound C3 for a signed step function x of length N.
    
    The autoconvolution upper bound C3 is the maximum absolute value of the
    autocorrelation sidelobes relative to the main lobe.
    
    For a signed step function (values in {-1, +1}), the autocorrelation is:
      R[k] = sum_{i=0}^{N-1-k} x[i] * x[i+k]  for k = 0, ..., N-1
    
    C3 = max_{k != 0} |R[k]| / R[0]
    """
    # Compute autocorrelation using FFT for efficiency
    n_fft = 1 << (2 * N - 1).bit_length()  # next power of 2 for linear correlation
    F = np.fft.rfft(x, n=n_fft)
    power = F * np.conj(F)
    R = np.fft.irfft(power, n=n_fft)[:2 * N - 1]
    
    R0 = R[0]
    if R0 == 0:
        return 1.0  # degenerate case
    
    # Maximum absolute sidelobe
    sidelobes = np.concatenate([R[1:N], R[N:2*N-1]])
    max_sidelobe = np.max(np.abs(sidelobes))
    
    C3 = max_sidelobe / R0
    return C3


def _compute_c3_with_fft(x):
    """
    Fast computation of C3 using pre-allocated FFT buffer.
    Returns (C3, R_array) where R_array is the full autocorrelation.
    """
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(x, n=n_fft)
    power = F * np.conj(F)
    R = np.fft.irfft(power, n=n_fft)[:2 * N - 1]
    
    R0 = R[0]
    if R0 == 0:
        return 1.0, R
    
    sidelobes = np.concatenate([R[1:N], R[N:2*N-1]])
    max_sidelobe = np.max(np.abs(sidelobes))
    
    C3 = max_sidelobe / R0
    return C3, R


def _local_search_fast(x, max_iter=10000, time_limit=None):
    """
    Fast local search: try flipping individual positions and accept if it improves C3.
    Uses incremental updates where possible.
    """
    start_time = time.time()
    current_c3, R = _compute_c3_with_fft(x)
    best_x = x.copy()
    best_c3 = current_c3
    no_improve_count = 0
    max_no_improve = 200
    
    for iteration in range(max_iter):
        # Check time
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        # Check global deadline
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        # Try a random flip
        idx = random.randint(0, N - 1)
        x_new = x.copy()
        x_new[idx] = -x_new[idx]
        new_c3, R_new = _compute_c3_with_fft(x_new)
        
        if new_c3 < current_c3 - 1e-12:
            x = x_new
            current_c3 = new_c3
            R = R_new
            no_improve_count = 0
            if current_c3 < best_c3:
                best_x = x.copy()
                best_c3 = current_c3
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
    
    return best_x, best_c3


def _simulated_annealing_fast(x, max_iter=20000, T_start=1.0, T_end=0.001, time_limit=None):
    """
    Simulated annealing to escape local minima.
    """
    start_time = time.time()
    current_c3, _ = _compute_c3_with_fft(x)
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
        
        # Try a random flip
        idx = random.randint(0, N - 1)
        x_new = x.copy()
        x_new[idx] = -x_new[idx]
        new_c3, _ = _compute_c3_with_fft(x_new)
        
        delta = new_c3 - current_c3
        if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
            x = x_new
            current_c3 = new_c3
            if current_c3 < best_c3:
                best_x = x.copy()
                best_c3 = current_c3
    
    return best_x, best_c3


def _soft_max_c3(x, temperature=5.0):
    """
    Smooth approximation of max sidelobe using log-sum-exp.
    This provides a differentiable surrogate for local optimization.
    """
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(x, n=n_fft)
    power = F * np.conj(F)
    R = np.fft.irfft(power, n=n_fft)[:2 * N - 1]
    
    R0 = R[0]
    if R0 == 0:
        return 1.0
    
    sidelobes = np.concatenate([R[1:N], R[N:2*N-1]])
    abs_sidelobes = np.abs(sidelobes)
    
    # Soft max using log-sum-exp
    max_val = np.max(abs_sidelobes)
    log_sum_exp = np.log(np.sum(np.exp((abs_sidelobes - max_val) / temperature))) + max_val / temperature
    soft_max = temperature * log_sum_exp
    
    return soft_max / R0


def _gradient_descent_continuous(x0, max_iter=5000, lr=0.01, time_limit=None):
    """
    Gradient descent on the continuous relaxation using soft-max surrogate.
    Returns the best binary solution found during the process.
    """
    start_time = time.time()
    x = x0.astype(float).copy()
    best_binary = np.sign(x).astype(int)
    best_c3 = _compute_c3(best_binary)
    
    for iteration in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        # Compute gradient of soft-max C3
        n_fft = 1 << (2 * N - 1).bit_length()
        F = np.fft.rfft(x, n=n_fft)
        power = F * np.conj(F)
        R = np.fft.irfft(power, n=n_fft)[:2 * N - 1]
        
        R0 = R[0]
        sidelobes = np.concatenate([R[1:N], R[N:2*N-1]])
        abs_sidelobes = np.abs(sidelobes)
        
        # Soft max gradient
        max_val = np.max(abs_sidelobes)
        exp_terms = np.exp((abs_sidelobes - max_val) / 5.0)
        weights = exp_terms / np.sum(exp_terms)
        
        # Gradient of |R[k]| w.r.t. x[i]
        # For autocorrelation R[k] = sum x[i]*x[i+k], dR[k]/dx[i] = x[i+k] + x[i-k]
        # d|u|/du = sign(u)
        grad = np.zeros(N)
        
        # This is complex; use numerical gradient for simplicity
        eps = 1e-4
        for i in range(0, N, 10):  # Sample some coordinates for speed
            x_plus = x.copy()
            x_plus[i] += eps
            c3_plus = _soft_max_c3(x_plus)
            
            x_minus = x.copy()
            x_minus[i] -= eps
            c3_minus = _soft_max_c3(x_minus)
            
            grad[i] = (c3_plus - c3_minus) / (2 * eps)
        
        # Update
        x -= lr * grad
        x = np.tanh(x)  # Keep in [-1, 1]
        
        # Check binary version
        binary = np.sign(x).astype(int)
        if np.all(binary == 0):
            binary = np.ones(N, dtype=int)
        c3_bin = _compute_c3(binary)
        if c3_bin < best_c3:
            best_c3 = c3_bin
            best_binary = binary.copy()
    
    return best_binary, best_c3


def _construct():
    """
    Main construction function that searches for a good signed step function.
    """
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)
    
    best_x = np.ones(N)
    best_c3 = _compute_c3(best_x)
    
    # Strategy 1: Start with all ones, do local search
    x1, c3_1 = _local_search_fast(np.ones(N), max_iter=5000, time_limit=20)
    if c3_1 < best_c3:
        best_x = x1
        best_c3 = c3_1
    
    # Check time
    if time.time() > DEADLINE - TIME_MARGIN:
        return best_x
    
    # Strategy 2: Random initialization + local search
    for trial in range(8):
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        x_init = np.random.choice([-1, 1], size=N)
        x2, c3_2 = _local_search_fast(x_init, max_iter=3000, time_limit=10)
        if c3_2 < best_c3:
            best_x = x2
            best_c3 = c3_2
    
    # Check time
    if time.time() > DEADLINE - TIME_MARGIN:
        return best_x
    
    # Strategy 3: Simulated annealing from best solution
    if time.time() < DEADLINE - TIME_MARGIN:
        remaining = DEADLINE - TIME_MARGIN - time.time()
        x3, c3_3 = _simulated_annealing_fast(best_x, max_iter=15000, T_start=0.5, T_end=0.001,
                                             time_limit=remaining * 0.5)
        if c3_3 < best_c3:
            best_x = x3
            best_c3 = c3_3
    
    # Check time
    if time.time() > DEADLINE - TIME_MARGIN:
        return best_x
    
    # Strategy 4: Gradient descent on continuous relaxation
    if time.time() < DEADLINE - TIME_MARGIN:
        remaining = DEADLINE - TIME_MARGIN - time.time()
        x4, c3_4 = _gradient_descent_continuous(best_x, max_iter=2000, lr=0.05, time_limit=remaining * 0.3)
        if c3_4 < best_c3:
            best_x = x4
            best_c3 = c3_4
    
    # Final local search polish
    if time.time() < DEADLINE - TIME_MARGIN:
        remaining = DEADLINE - TIME_MARGIN - time.time()
        x5, c3_5 = _local_search_fast(best_x, max_iter=5000, time_limit=remaining * 0.5)
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
