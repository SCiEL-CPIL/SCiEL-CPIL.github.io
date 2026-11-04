"""Spectral Phase-Only Optimization with Simulated Annealing for Autoconvolution Peak Minimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

# Precompute indices for real FFT
_M = _FFT_SIZE // 2 + 1


def _autoconv_max_abs(x):
    """Compute the maximum absolute value of the linear autoconvolution."""
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return float(np.max(np.abs(conv[:2 * N - 1])))


def _integral(x):
    """Compute the sum of the signal (integral for step functions)."""
    return float(np.sum(x))


def _metric(x):
    """Compute the objective metric: max_abs_autoconv / (sum(x))^2."""
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)


def _phase_to_signal(mags, phases):
    """Reconstruct signal from magnitudes and phases.
    We enforce Hermitian symmetry for real-valued signals.
    mags: array of length M
    phases: array of length M
    """
    # Build full FFT spectrum
    # For real signals, X[k] = X[N-k]*
    # We work with rfft size M = FFT_SIZE//2 + 1
    
    # Create complex spectrum
    spec = np.zeros(_FFT_SIZE, dtype=np.complex128)
    
    # Positive frequencies (0 to M-1)
    # Note: rfft gives bins 0, 1, ..., M-1
    # We need to map these back to full fft
    # Bin 0 is DC. Bin M-1 is Nyquist if FFT_SIZE is even.
    
    # For rfft, the output is [0, 1, ..., M-1]
    # The full spectrum is [0, 1, ..., M-1, N-M, ..., 1]
    
    # Let's just use ifft on the reconstructed full spectrum
    # Construct full spectrum
    full_spec = np.zeros(_FFT_SIZE, dtype=np.complex128)
    
    # Fill positive part
    # mags and phases are for the rfft bins
    # We assume mags[0] and mags[-1] are real (phase 0 or pi) for real signal?
    # Actually, to ensure real signal, we must enforce symmetry.
    # It's easier to optimize phases for the full complex spectrum and then take real part?
    # No, that breaks the metric.
    
    # Better approach:
    # Optimize phases for the first N/2+1 bins.
    # Enforce symmetry: X[k] = mags[k] * exp(i * phases[k])
    # X[FFT_SIZE - k] = conj(X[k])
    
    for k in range(_M):
        val = mags[k] * np.exp(1j * phases[k])
        full_spec[k] = val
        if k != 0 and k != _FFT_SIZE // 2:
            full_spec[_FFT_SIZE - k] = np.conj(val)
        elif k == _FFT_SIZE // 2:
            # Nyquist frequency must be real
            full_spec[k] = mags[k] # phase should be 0 or pi, we force 0 for simplicity or handle in phase
            # Actually, if we optimize phase for Nyquist, we should allow it to be 0 or pi.
            # But exp(i*pi) = -1. So we can just use the complex value.
            # However, for real signal, X[N/2] must be real.
            # So phase must be 0 or pi.
            # We will just use the complex value and take real part at the end?
            # No, ifft of a non-symmetric spectrum is complex.
            # We must ensure symmetry.
            # If we set full_spec[N/2] = mags[N/2] * exp(i*phases[N/2]), it might not be real.
            # To be safe, we force phase of Nyquist to 0 or pi.
            # Or simply, we don't optimize the phase of Nyquist, we fix it to 0.
            # Given N=400, FFT_SIZE=1024. Nyquist is 512.
            # The impact of one bin is small.
            pass

    # Reconstruct signal
    x = np.fft.ifft(full_spec).real
    return x


def _sa_optimize_phases(mags, init_phases, deadline, rng):
    """Simulated Annealing on the phase vector."""
    phases = init_phases.copy()
    # Ensure DC and Nyquist phases are fixed (0 or pi)
    # DC (k=0) phase should be 0 to keep integral positive?
    # If we flip DC phase, integral becomes negative, but metric uses (sum)^2, so sign doesn't matter for metric.
    # However, changing DC phase changes the mean of the signal.
    # Let's fix DC phase to 0.
    phases[0] = 0.0
    if _FFT_SIZE % 2 == 0:
        phases[_FFT_SIZE // 2] = 0.0 # Fix Nyquist too for simplicity

    x = _phase_to_signal(mags, phases)
    current_metric = _metric(x)
    
    best_phases = phases.copy()
    best_metric = current_metric
    best_x = x.copy()
    
    T = 1.0
    cooling = 0.999
    
    # Precompute indices to optimize
    # Exclude DC and Nyquist
    opt_indices = list(range(1, _M))
    if _FFT_SIZE % 2 == 0:
        opt_indices.remove(_FFT_SIZE // 2)
        
    while time.time() < deadline:
        if T < 1e-6:
            break
            
        # Pick a random index to perturb
        idx = opt_indices[rng.integers(len(opt_indices))]
        
        # Perturb phase
        old_phase = phases[idx]
        new_phase = old_phase + rng.normal(0, 0.5)
        new_phase = new_phase % (2 * np.pi)
        phases[idx] = new_phase
        
        # Evaluate
        x_new = _phase_to_signal(mags, phases)
        new_metric = _metric(x_new)
        
        # Acceptance criterion
        if new_metric < current_metric or rng.random() < np.exp(-(new_metric - current_metric) / T):
            current_metric = new_metric
            if new_metric < best_metric:
                best_metric = new_metric
                best_phases = phases.copy()
                best_x = x_new.copy()
        else:
            # Revert
            phases[idx] = old_phase
            
        T *= cooling
        
    return best_x, best_metric, best_phases


def _try_flat_spectrum(deadline, rng):
    """Strategy: Flat magnitude spectrum, optimize phases."""
    # Flat magnitude spectrum
    mags = np.ones(_M)
    # Scale mags so that sum(x) is reasonable?
    # The metric is scale invariant: (c * x) -> (c^2 max_conv) / (c^2 sum^2) = max_conv/sum^2
    # So we can normalize x after reconstruction.
    
    # Initial phases: Random
    init_phases = rng.uniform(0, 2 * np.pi, size=_M)
    init_phases[0] = 0.0
    if _FFT_SIZE % 2 == 0:
        init_phases[_FFT_SIZE // 2] = 0.0
        
    x, val, _ = _sa_optimize_phases(mags, init_phases, deadline, rng)
    
    # Normalize x
    s = _integral(x)
    if abs(s) > 1e-12:
        x = x / s
        # Recalculate metric after normalization (should be same, but ensures consistency)
        val = _metric(x)
        
    return x, val


def _try_chirp_phases(deadline, rng):
    """Strategy: Initial phases as a chirp, then optimize."""
    mags = np.ones(_M)
    k = np.arange(_M)
    # Chirp: phase = 2*pi * k^2 / N
    init_phases = 2 * np.pi * (k ** 2) / N
    init_phases[0] = 0.0
    if _FFT_SIZE % 2 == 0:
        init_phases[_FFT_SIZE // 2] = 0.0
        
    x, val, _ = _sa_optimize_phases(mags, init_phases, deadline, rng)
    
    s = _integral(x)
    if abs(s) > 1e-12:
        x = x / s
        val = _metric(x)
        
    return x, val


def _try_random_mags_phases(deadline, rng):
    """Strategy: Random magnitudes and phases, optimize both? 
    No, just random mags, optimize phases.
    """
    # Random magnitudes, but keep DC high to ensure non-zero integral
    mags = rng.uniform(0.5, 1.5, size=_M)
    mags[0] = 1.0
    
    init_phases = rng.uniform(0, 2 * np.pi, size=_M)
    init_phases[0] = 0.0
    if _FFT_SIZE % 2 == 0:
        init_phases[_FFT_SIZE // 2] = 0.0
        
    x, val, _ = _sa_optimize_phases(mags, init_phases, deadline, rng)
    
    s = _integral(x)
    if abs(s) > 1e-12:
        x = x / s
        val = _metric(x)
        
    return x, val


def construct():
    deadline = time.time() + 145.0
    rng = np.random.default_rng(2024)
    
    best_x = None
    best_val = float('inf')
    
    # 1. Try flat spectrum with random phases
    t_start = time.time()
    time_left = deadline - t_start - 5.0
    if time_left > 10:
        x, val = _try_flat_spectrum(deadline - 5.0, rng)
        if val < best_val:
            best_val = val
            best_x = x.copy()
            
    # 2. Try chirp phases
    t_start = time.time()
    time_left = deadline - t_start - 5.0
    if time_left > 10:
        x, val = _try_chirp_phases(deadline - 5.0, rng)
        if val < best_val:
            best_val = val
            best_x = x.copy()
            
    # 3. Try random magnitudes
    t_start = time.time()
    time_left = deadline - t_start - 5.0
    if time_left > 10:
        x, val = _try_random_mags_phases(deadline - 5.0, rng)
        if val < best_val:
            best_val = val
            best_x = x.copy()
            
    # 4. If we have time, run a final polish on the best solution
    # We can do a local search on the phases of the best solution
    if best_x is not None and time.time() < deadline - 10:
        # Reconstruct phases from best_x
        fx = np.fft.rfft(best_x, _FFT_SIZE)
        mags_best = np.abs(fx)
        phases_best = np.angle(fx)
        
        # Ensure DC and Nyquist are handled
        phases_best[0] = 0.0
        if _FFT_SIZE % 2 == 0:
            phases_best[_FFT_SIZE // 2] = 0.0
            
        # Run SA again on these phases
        x, val, _ = _sa_optimize_phases(mags_best, phases_best, deadline - 2.0, rng)
        
        s = _integral(x)
        if abs(s) > 1e-12:
            x = x / s
            val = _metric(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()

    if best_x is None:
        # Fallback: all ones
        best_x = np.ones(N)
        
    return best_x
# EVOLVE-BLOCK-END
