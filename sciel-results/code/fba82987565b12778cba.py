"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

DEADLINE = 120.0
TIME_MARGIN = 10.0  # safety margin in seconds

# Precompute FFT size for efficiency
N_FFT = 1 << (2 * N - 1).bit_length()


def _compute_c3(x):
    """
    Compute the upper bound C3 for a signed step function x of length N.
    C3 = max_{k=1,...,N-1} |R[k]| / R[0]
    """
    F = np.fft.rfft(x, n=N_FFT)
    power = F * np.conj(F)
    R = np.fft.irfft(power, n=N_FFT)[:2 * N - 1]
    
    R0 = R[0]
    if R0 == 0:
        return 1.0
    
    sidelobes = np.concatenate([R[1:N], R[N:2*N-1]])
    max_sidelobe = np.max(np.abs(sidelobes))
    
    C3 = max_sidelobe / R0
    return C3


def _compute_soft_c3(x, beta=10.0):
    """
    Compute a smooth surrogate for C3 using log-sum-exp.
    soft_c3 = (1/beta) * log(sum_{k!=0} exp(beta * |R[k]| / R0))
    This approximates max |R[k]|/R0 for large beta, but is differentiable.
    """
    F = np.fft.rfft(x, n=N_FFT)
    power = F * np.conj(F)
    R = np.fft.irfft(power, n=N_FFT)[:2 * N - 1]
    
    R0 = R[0]
    if R0 == 0:
        return 1.0
    
    sidelobes = np.concatenate([R[1:N], R[N:2*N-1]])
    abs_sidelobes = np.abs(sidelobes) / R0
    
    # Log-sum-exp for numerical stability
    max_val = np.max(abs_sidelobes)
    if max_val == 0:
        return 0.0
    
    log_sum = max_val + np.log(np.sum(np.exp(beta * (abs_sidelobes - max_val))))
    soft_c3 = log_sum / beta
    
    return soft_c3


def _local_search(x, max_iter=5000, time_limit=None):
    """
    Local search: try flipping individual positions and accept if it improves C3.
    """
    start_time = time.time()
    current_c3 = _compute_c3(x)
    best_x = x.copy()
    best_c3 = current_c3
    no_improve_count = 0
    max_no_improve = 100
    
    for iteration in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        idx = random.randint(0, N - 1)
        x_new = x.copy()
        x_new[idx] = -x_new[idx]
        new_c3 = _compute_c3(x_new)
        
        if new_c3 < current_c3 - 1e-12:
            x = x_new
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


def _soft_guided_local_search(x, max_iter=8000, time_limit=None, beta=5.0):
    """
    Local search guided by the soft surrogate objective.
    Use soft_c3 to guide move acceptance, but track exact C3 for best solution.
    """
    start_time = time.time()
    current_c3 = _compute_c3(x)
    current_soft = _compute_soft_c3(x, beta=beta)
    best_x = x.copy()
    best_c3 = current_c3
    no_improve_count = 0
    max_no_improve = 150
    
    for iteration in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        idx = random.randint(0, N - 1)
        x_new = x.copy()
        x_new[idx] = -x_new[idx]
        new_soft = _compute_soft_c3(x_new, beta=beta)
        
        # Accept if soft objective improves (smooth guidance)
        if new_soft < current_soft - 1e-10:
            x = x_new
            current_soft = new_soft
            new_c3 = _compute_c3(x)
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


def _simulated_annealing(x, max_iter=20000, T_start=1.0, T_end=0.01, time_limit=None):
    """
    Simulated annealing to escape local minima.
    """
    start_time = time.time()
    current_c3 = _compute_c3(x)
    best_x = x.copy()
    best_c3 = current_c3
    T = T_start
    
    for iteration in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        T = T_start * (T_end / T_start) ** (iteration / max_iter)
        
        idx = random.randint(0, N - 1)
        x_new = x.copy()
        x_new[idx] = -x_new[idx]
        new_c3 = _compute_c3(x_new)
        
        delta = new_c3 - current_c3
        if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
            x = x_new
            current_c3 = new_c3
            if current_c3 < best_c3:
                best_x = x.copy()
                best_c3 = current_c3
    
    return best_x, best_c3


def _polish(x, time_limit=None):
    """
    High-precision polish: multiple passes of local search with decreasing tolerance.
    """
    start_time = time.time()
    current_c3 = _compute_c3(x)
    best_x = x.copy()
    best_c3 = current_c3
    
    for pass_num in range(3):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        remaining = (time_limit - (time.time() - start_time)) if time_limit else (DEADLINE - TIME_MARGIN - time.time())
        if remaining < 2:
            break
        
        x, c3 = _local_search(x, max_iter=3000, time_limit=remaining)
        if c3 < best_c3:
            best_x = x.copy()
            best_c3 = c3
            x = best_x.copy()
    
    return best_x, best_c3


def _construct():
    """
    Main construction function that searches for a good signed step function.
    """
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)
    
    best_x = np.ones(N)
    best_c3 = _compute_c3(best_x)
    
    # Strategy 1: Start with all ones, do soft-guided local search
    x1, c3_1 = _soft_guided_local_search(np.ones(N), max_iter=5000, time_limit=25, beta=5.0)
    if c3_1 < best_c3:
        best_x = x1
        best_c3 = c3_1
    
    # Check time
    if time.time() > DEADLINE - TIME_MARGIN:
        return best_x
    
    # Strategy 2: Random initialization + soft-guided local search
    for trial in range(4):
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        x_init = np.random.choice([-1, 1], size=N)
        x2, c3_2 = _soft_guided_local_search(x_init, max_iter=3000, time_limit=12, beta=3.0)
        if c3_2 < best_c3:
            best_x = x2
            best_c3 = c3_2
    
    # Check time
    if time.time() > DEADLINE - TIME_MARGIN:
        return best_x
    
    # Strategy 3: Simulated annealing from best solution
    if time.time() < DEADLINE - TIME_MARGIN:
        x3, c3_3 = _simulated_annealing(best_x, max_iter=8000, T_start=0.5, T_end=0.01,
                                         time_limit=DEADLINE - TIME_MARGIN - time.time())
        if c3_3 < best_c3:
            best_x = x3
            best_c3 = c3_3
    
    # Final high-precision polish
    if time.time() < DEADLINE - TIME_MARGIN:
        x4, c3_4 = _polish(best_x, time_limit=DEADLINE - TIME_MARGIN - time.time())
        if c3_4 < best_c3:
            best_x = x4
            best_c3 = c3_4
    
    return best_x

def construct():
    """
    Entry point: return a sequence of signed step heights (±1) of length N=400.
    """
    x = _construct()
    return np.sign(x).astype(int)
# EVOLVE-BLOCK-END
