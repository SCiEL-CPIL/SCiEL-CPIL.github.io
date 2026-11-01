"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

DEADLINE = 150.0
TIME_MARGIN = 15.0


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
    """Compute new R after flipping x[i] using O(N) update."""
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


def _local_search_fast(x, R, max_iter=100000, time_limit=None):
    """Local search with O(N) flip evaluation."""
    start_time = time.time()
    current_c3 = _c3_from_R(R)
    best_x = x.copy()
    best_R = R.copy()
    best_c3 = current_c3
    no_improve = 0
    max_no_improve = 200
    
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


def _simulated_annealing_fast(x, R, max_iter=50000, T_start=0.1, T_end=0.001, time_limit=None):
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


def _spectral_optimize(x0, time_limit=30.0, start_time=None):
    """
    Optimize the continuous power spectrum to minimize peak autocorrelation sidelobe.
    Work in frequency domain: P[k] = |F[k]|^2, minimize max |IFFT(P)[k]| for k!=0.
    Then project back to time domain and threshold to ±1.
    """
    if start_time is None:
        start_time = time.time()
    
    n_fft = 1 << (2 * N - 1).bit_length()
    
    # Initialize power spectrum from x0
    F0 = np.fft.rfft(x0, n=n_fft)
    P0 = np.abs(F0)**2
    # Work with log-power for stability
    logP0 = np.log(P0 + 1e-10)
    
    def objective(logP):
        """Minimize soft-max of |IFFT(P)[k]| for k != 0, normalized by IFFT(P)[0]."""
        if time.time() - start_time > DEADLINE - TIME_MARGIN - 2.0:
            return 1e10
        
        P = np.exp(logP)
        R = np.fft.irfft(P, n=n_fft)[:2*N-1]
        R0 = R[0]
        if R0 < 1e-10:
            return 1e10
        
        # Only consider sidelobes (k=1..N-1 and k=N..2N-2)
        sidelobes = np.concatenate([R[1:N], R[N:2*N-1]])
        
        # Soft-max of |sidelobes| / R0
        abs_s = np.abs(sidelobes) / R0
        eps = 1e-10
        abs_s_smooth = np.sqrt(abs_s**2 + eps)
        
        # Use log-sum-exp for soft-max
        T = 0.5
        max_val = np.max(abs_s_smooth)
        if max_val < 1e-15:
            return 0.0
        shifted = (abs_s_smooth - max_val) / T
        shifted = np.clip(shifted, -100, 100)
        lse = np.log(np.sum(np.exp(shifted))) + max_val / T
        soft_max = T * lse
        
        return soft_max
    
    def gradient(logP):
        """Compute gradient of objective w.r.t. logP via FFT chain rule."""
        if time.time() - start_time > DEADLINE - TIME_MARGIN - 2.0:
            return np.zeros_like(logP)
        
        P = np.exp(logP)
        R = np.fft.irfft(P, n=n_fft)[:2*N-1]
        R0 = R[0]
        if R0 < 1e-10:
            return np.zeros_like(logP)
        
        sidelobes = np.concatenate([R[1:N], R[N:2*N-1]])
        abs_s = np.abs(sidelobes) / R0
        eps = 1e-10
        abs_s_smooth = np.sqrt(abs_s**2 + eps)
        
        T = 0.5
        max_val = np.max(abs_s_smooth)
        if max_val < 1e-15:
            return np.zeros_like(logP)
        shifted = (abs_s_smooth - max_val) / T
        shifted = np.clip(shifted, -100, 100)
        weights = np.exp(shifted) / np.sum(np.exp(shifted))
        
        # d(soft_max)/d(sidelobes[k]) = weights[k] * sign(sidelobes[k]) / R0
        # Need to map back to full R indices
        dR = np.zeros(2*N-1)
        # First half: indices 1..N-1
        dR[1:N] = weights[:N-1] * np.sign(sidelobes[:N-1]) / R0
        # Second half: indices N..2N-2
        dR[N:2*N-1] = weights[N-1:] * np.sign(sidelobes[N-1:]) / R0
        
        # Also account for R0 in denominator: d/dR0 = -max_abs / R0^2 * ... (approximate)
        # For simplicity, ignore R0 gradient (it's slowly varying)
        
        # dR/dP: R = IFFT(P), so dR[k]/dP[m] = exp(2*pi*i*k*m/n_fft) / n_fft (for real IFFT)
        # dP/dlogP = P
        # So d(objective)/dlogP[m] = sum_k dR[k] * exp(2*pi*i*k*m/n_fft) / n_fft * P[m]
        
        # Use FFT for efficiency: grad_logP = P * IFFT(dR) (conjugate for real)
        dR_full = np.zeros(n_fft)
        dR_full[:2*N-1] = dR
        # For real IFFT, we need to handle symmetry
        grad_logP = np.real(np.fft.ifft(dR_full) * P) / n_fft
        
        return grad_logP
    
    # Run L-BFGS-B in log-power domain
    result = minimize(
        objective, logP0, method='L-BFGS-B', jac=gradient,
        options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 600}
    )
    
    if time.time() - start_time > DEADLINE - TIME_MARGIN - 2.0:
        return None
    
    # Recover continuous signal from optimized power spectrum
    P_opt = np.exp(result.x)
    # Take square root magnitude and reconstruct phase from original
    F_mag = np.sqrt(P_opt + 1e-15)
    F_phase = F0 / (np.abs(F0) + 1e-15)
    F_recon = F_mag * F_phase
    h_cont = np.real(np.fft.irfft(F_recon, n=n_fft))[:N]
    
    # Project to ±1
    h_disc = np.sign(h_cont)
    h_disc[h_disc == 0] = 1.0
    return h_disc


def construct():
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)
    
    best_x = np.ones(N)
    best_R = _compute_R(best_x)
    best_c3 = _c3_from_R(best_R)
    
    # Generate diverse initial candidates for spectral optimization
    candidates = []
    
    # All ones
    candidates.append(np.ones(N))
    
    # Alternating
    alt = np.ones(N)
    alt[1::2] = -1.0
    candidates.append(alt)
    
    # Thue-Morse
    candidates.append(_thue_morse(N))
    
    # Random
    rng = np.random.RandomState(123)
    candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # Block patterns
    for block_size in [5, 10, 20]:
        n_blocks = N // block_size
        block_vals = np.random.RandomState(100 + block_size).choice([-1, 1], size=n_blocks)
        candidates.append(np.repeat(block_vals, block_size))
    
    # Spectral optimization from each candidate
    spectral_results = []
    for i, x_init in enumerate(candidates):
        if time.time() > DEADLINE - TIME_MARGIN - 5.0:
            break
        time_left = DEADLINE - TIME_MARGIN - time.time()
        if time_left < 5.0:
            break
        time_limit = min(20.0, time_left * 0.35)
        if time_limit < 3.0:
            break
        
        h_disc = _spectral_optimize(x_init.copy(), time_limit=time_limit, start_time=start_time)
        if h_disc is not None:
            spectral_results.append(h_disc)
    
    # Combine all candidates for local search
    all_candidates = candidates + spectral_results
    
    # Local search from each candidate
    for i, x_init in enumerate(all_candidates):
        if time.time() > DEADLINE - TIME_MARGIN - 3.0:
            break
        R_init = _compute_R(x_init)
        c3_init = _c3_from_R(R_init)
        
        time_left = DEADLINE - TIME_MARGIN - time.time()
        if time_left < 3.0:
            break
        time_limit = min(15.0, time_left * 0.25)
        if time_limit < 2.0:
            break
        
        x_trial, R_trial, c3_trial = _local_search_fast(
            x_init.copy(), R_init, max_iter=50000, time_limit=time_limit
        )
        if c3_trial < best_c3:
            best_x = x_trial
            best_R = R_trial
            best_c3 = c3_trial
    
    # Simulated annealing from best
    if time.time() < DEADLINE - TIME_MARGIN - 5.0:
        time_left = DEADLINE - TIME_MARGIN - time.time()
        time_limit = min(30.0, time_left * 0.5)
        if time_limit > 2.0:
            x_sa, R_sa, c3_sa = _simulated_annealing_fast(
                best_x.copy(), best_R.copy(), max_iter=50000, time_limit=time_limit
            )
            if c3_sa < best_c3:
                best_x = x_sa
                best_R = R_sa
                best_c3 = c3_sa
    
    # Final local search polish
    if time.time() < DEADLINE - TIME_MARGIN - 2.0:
        time_left = DEADLINE - TIME_MARGIN - time.time()
        time_limit = max(1.0, time_left * 0.8)
        x_final, R_final, c3_final = _local_search_fast(
            best_x.copy(), best_R.copy(), max_iter=100000, time_limit=time_limit
        )
        if c3_final < best_c3:
            best_x = x_final
            best_R = R_final
            best_c3 = c3_final
    
    return best_x
# EVOLVE-BLOCK-END
