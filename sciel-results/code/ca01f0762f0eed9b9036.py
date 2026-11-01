"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

DEADLINE = 145.0
TIME_MARGIN = 10.0


def _compute_R(x):
    """Compute full autocorrelation R[0..2N-2] via FFT."""
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(x, n=n_fft)
    power = F * np.conj(F)
    return np.fft.irfft(power, n=n_fft)[:2 * N - 1]


def _c3_from_R(R):
    """Compute C3 from precomputed R."""
    R0 = R[0]
    if R0 == 0:
        return 1.0
    sidelobes = np.concatenate([R[1:N], R[N:2*N-1]])
    return np.max(np.abs(sidelobes)) / R0


def _flip_effect(x, i, R):
    """Compute new R after flipping x[i]."""
    R_new = R.copy()
    xi = x[i]
    for k in range(N):
        delta = 0.0
        if i + k < N:
            delta -= 2.0 * xi * x[i + k]
        if i - k >= 0:
            delta -= 2.0 * x[i - k] * xi
        R_new[k] += delta
    for k in range(N):
        R_new[2*N-2-k] = R_new[k]
    return R_new


def _spectral_objective(params, N):
    """
    Minimize peak sidelobe of inverse DFT of log-amplitude spectrum.
    params: real vector of length N_fft//2+1 (log amplitudes for rfft)
    We optimize the power spectrum shape to minimize max |R[k]| for k != 0.
    """
    n_fft = 1 << (2 * N - 1).bit_length()
    # params are log-amplitudes for the rfft bins
    n_bins = n_fft // 2 + 1
    if len(params) < n_bins:
        params = np.concatenate([params, np.zeros(n_bins - len(params))])
    else:
        params = params[:n_bins]
    
    # Exponentiate to get amplitudes, with some regularization
    log_amp = params
    # Normalize so that the zero-frequency bin gives R[0] ~ 1
    amp = np.exp(log_amp - np.mean(log_amp))
    
    # Build complex spectrum with zero phase (real signal)
    F = np.zeros(n_fft // 2 + 1, dtype=complex)
    F.real = amp
    F.imag = 0.0
    # F[0] and F[n_fft//2] must be real
    # Reconstruct full spectrum
    F_full = np.zeros(n_fft, dtype=complex)
    F_full[:n_fft//2+1] = F
    if n_fft > 1:
        F_full[n_fft//2+1:] = np.conj(F[1:n_fft//2][::-1])
    
    # Inverse FFT to get R
    R_full = np.fft.ifft(F_full * np.conj(F_full)).real
    R = R_full[:2*N-1]
    
    R0 = R[0]
    if R0 < 1e-10:
        return 1e10
    sidelobes = np.concatenate([R[1:N], R[N:2*N-1]])
    max_sidelobe = np.max(np.abs(sidelobes))
    return max_sidelobe / R0


def _spectral_optimize(N, time_limit, seed=42):
    """
    Optimize the power spectrum to minimize peak sidelobe.
    Returns the corresponding real signal x of length N.
    """
    rng = np.random.RandomState(seed)
    n_fft = 1 << (2 * N - 1).bit_length()
    n_bins = n_fft // 2 + 1
    
    # Initialize with random phase but we're optimizing real amplitudes
    # Start with flat spectrum (white noise)
    log_amp_init = np.zeros(n_bins)
    
    # Try a few random initializations
    best_x = np.ones(N)
    best_R = _compute_R(best_x)
    best_c3 = _c3_from_R(best_R)
    
    for trial in range(3):
        if time.time() > time_limit:
            break
        
        # Random initialization of log-amplitudes
        log_amp0 = rng.normal(0, 0.5, size=n_bins)
        # Keep DC bin at 0
        log_amp0[0] = 0.0
        if n_fft > 1:
            log_amp0[n_fft//2] = 0.0
        
        result = minimize(
            _spectral_objective,
            log_amp0,
            args=(N,),
            method='L-BFGS-B',
            options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        
        if result.success or result.fun < _spectral_objective(log_amp0, N):
            log_amp_opt = result.x
            amp = np.exp(log_amp_opt - np.mean(log_amp_opt))
            
            F = np.zeros(n_fft // 2 + 1, dtype=complex)
            F.real = amp
            F_full = np.zeros(n_fft, dtype=complex)
            F_full[:n_fft//2+1] = F
            if n_fft > 1:
                F_full[n_fft//2+1:] = np.conj(F[1:n_fft//2][::-1])
            
            # The signal is the inverse FFT of sqrt of power spectrum
            # Actually, we want x such that |FFT(x)|^2 ~ our optimized spectrum
            # x = IFFT(sqrt(P)) where P is the power spectrum
            P = F_full * np.conj(F_full)
            sqrt_P = np.sqrt(P)
            x_cont = np.fft.ifft(sqrt_P).real[:N]
            
            # Threshold to ±1
            x_bin = np.sign(x_cont)
            x_bin[x_bin == 0] = 1.0
            
            R = _compute_R(x_bin)
            c3 = _c3_from_R(R)
            
            if c3 < best_c3:
                best_x = x_bin
                best_R = R
                best_c3 = c3
    
    return best_x, best_R, best_c3


def _local_search_fast(x, R, max_iter=100000, time_limit=None):
    """Local search with O(N) flip evaluation."""
    start_time = time.time()
    current_c3 = _c3_from_R(R)
    best_x = x.copy()
    best_R = R.copy()
    best_c3 = current_c3
    no_improve = 0
    max_no_improve = 300
    
    for it in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        idx = random.randint(0, N - 1)
        R_new = _flip_effect(x, idx, R)
        new_c3 = _c3_from_R(R_new)
        
        if new_c3 < current_c3 - 1e-12:
            x[idx] = -x[idx]
            R = R_new
            current_c3 = new_c3
            no_improve = 0
            if current_c3 < best_c3:
                best_x = x.copy()
                best_R = R.copy()
                best_c3 = current_c3
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                break
    
    return best_x, best_R, best_c3


def _simulated_annealing_fast(x, R, max_iter=50000, T_start=0.05, T_end=0.0001, time_limit=None):
    """Simulated annealing with O(N) flip evaluation."""
    start_time = time.time()
    current_c3 = _c3_from_R(R)
    best_x = x.copy()
    best_R = R.copy()
    best_c3 = current_c3
    T = T_start
    
    for it in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        T = T_start * (T_end / T_start) ** (it / max_iter)
        
        idx = random.randint(0, N - 1)
        R_new = _flip_effect(x, idx, R)
        new_c3 = _c3_from_R(R_new)
        
        delta = new_c3 - current_c3
        if delta < 0 or (T > 1e-10 and random.random() < np.exp(-delta / T)):
            x[idx] = -x[idx]
            R = R_new
            current_c3 = new_c3
            if current_c3 < best_c3:
                best_x = x.copy()
                best_R = R.copy()
                best_c3 = current_c3
    
    return best_x, best_R, best_c3


def _thue_morse(n):
    """Generate Thue-Morse sequence."""
    seq = np.zeros(n)
    for i in range(n):
        c = bin(i).count('1')
        seq[i] = 1.0 if c % 2 == 0 else -1.0
    return seq


def construct():
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)
    
    best_x = np.ones(N)
    best_R = _compute_R(best_x)
    best_c3 = _c3_from_R(best_R)
    
    # Spectral optimization to get a good initial signal
    time_left = DEADLINE - TIME_MARGIN - time.time()
    if time_left > 20.0:
        time_limit_spectral = min(30.0, time_left * 0.4)
        x_spec, R_spec, c3_spec = _spectral_optimize(N, time_limit=time_limit_spectral, seed=42)
        if c3_spec < best_c3:
            best_x = x_spec
            best_R = R_spec
            best_c3 = c3_spec
    
    # Also try standard initializations
    inits = []
    inits.append(_thue_morse(N))
    rng = np.random.RandomState(12345)
    inits.append(rng.choice([-1, 1], size=N))
    rng2 = np.random.RandomState(67890)
    inits.append(rng2.choice([-1, 1], size=N))
    # Logistic map
    val = 0.5
    log_seq = np.zeros(N)
    for i in range(N):
        val = 3.99 * val * (1.0 - val)
        log_seq[i] = 1.0 if val > 0.5 else -1.0
    inits.append(log_seq)
    
    # Local search from each initialization
    for i, x_init in enumerate(inits):
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        R_init = _compute_R(x_init)
        c3_init = _c3_from_R(R_init)
        if c3_init < best_c3:
            best_x = x_init.copy()
            best_R = R_init
            best_c3 = c3_init
        
        time_left = DEADLINE - TIME_MARGIN - time.time()
        time_limit = min(20.0, time_left * 0.25)
        if time_limit < 2.0:
            break
        x_trial, R_trial, c3_trial = _local_search_fast(x_init.copy(), R_init, max_iter=50000, time_limit=time_limit)
        if c3_trial < best_c3:
            best_x = x_trial
            best_R = R_trial
            best_c3 = c3_trial
    
    # Simulated annealing from best
    if time.time() < DEADLINE - TIME_MARGIN - 5.0:
        time_left = DEADLINE - TIME_MARGIN - time.time()
        time_limit = min(30.0, time_left * 0.5)
        if time_limit > 2.0:
            x_sa, R_sa, c3_sa = _simulated_annealing_fast(best_x.copy(), best_R.copy(), max_iter=50000, time_limit=time_limit)
            if c3_sa < best_c3:
                best_x = x_sa
                best_R = R_sa
                best_c3 = c3_sa
    
    # Final local search polish
    if time.time() < DEADLINE - TIME_MARGIN - 2.0:
        time_left = DEADLINE - TIME_MARGIN - time.time()
        time_limit = max(1.0, time_left * 0.8)
        x_final, R_final, c3_final = _local_search_fast(best_x.copy(), best_R.copy(), max_iter=100000, time_limit=time_limit)
        if c3_final < best_c3:
            best_x = x_final
            best_R = R_final
            best_c3 = c3_final
    
    return best_x
# EVOLVE-BLOCK-END
