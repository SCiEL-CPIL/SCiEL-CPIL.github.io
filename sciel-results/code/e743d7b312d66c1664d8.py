"""Combined approach: FFT-based overlap + Fourier-mode parameterization + multi-start optimization."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = 135.0  # seconds, with 15s safety margin before 150s limit

def _compute_max_overlap(h):
    """
    Compute the maximum overlap of h with shifted 1-h over all circular shifts.
    
    h is an array of N values on [0, 2], so dx = 2/N.
    sum(h) * dx = 1, so sum(h) = N/2.
    
    overlap(k) = sum_j h[j] * (1 - h[(j+k)%N]) * dx
               = (sum_j h[j] - sum_j h[j]*h[(j+k)%N]) * dx
               = (N/2 - C[k]) * dx
    where C[k] = sum_j h[j]*h[(j+k)%N]
    """
    n = len(h)
    dx = 2.0 / n
    sum_h = np.sum(h)
    
    # Compute circular cross-correlation C[k] = sum_j h[j]*h[(j+k)%n]
    # Using FFT: autocorr = ifft(fft(h) * conj(fft(h)))
    # autocorr[m] = sum_j h[j] * h[(j-m)%n]
    # So C[k] = sum_j h[j]*h[(j+k)%n] = autocorr[(-k)%n]
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    # autocorr[m] = sum_j h[j]*h[(j-m)%n] for m=0,...,n-1
    # C[0] = autocorr[0]
    # C[k] = autocorr[n-k] for k=1,...,n-1
    C = np.zeros(n)
    C[0] = autocorr[0]
    if n > 1:
        C[1:] = autocorr[n-1:0:-1]
    
    overlaps = (sum_h - C) * dx
    return np.max(overlaps)

def _build_h_from_fourier(coeffs):
    """
    Build h from Fourier coefficients.
    coeffs = [a_1, b_1, a_2, b_2, ..., a_m, b_m]
    h(x) = 0.5 + sum_{k=1}^{m} (a_k * cos(pi*k*x) + b_k * sin(pi*k*x))
    where x = 2*j/N for j=0,...,N-1
    """
    x = np.linspace(0, 2, N, endpoint=False)
    h = np.full(N, 0.5)
    n_modes = len(coeffs) // 2
    for k in range(1, n_modes + 1):
        a_k = coeffs[2*(k-1)]
        b_k = coeffs[2*(k-1)+1]
        h += a_k * np.cos(np.pi * k * x) + b_k * np.sin(np.pi * k * x)
    
    # Clip to [0, 1]
    h = np.clip(h, 0.0, 1.0)
    
    # Normalize to sum = N/2
    target_sum = N / 2.0
    current_sum = h.sum()
    if current_sum > 1e-10:
        h = h * (target_sum / current_sum)
        h = np.clip(h, 0.0, 1.0)
        # Iterate a few times to converge
        for _ in range(5):
            s = h.sum()
            if s > 1e-10:
                h = h * (target_sum / s)
                h = np.clip(h, 0.0, 1.0)
            else:
                break
    
    return h

def _objective_fourier(coeffs):
    """Objective function: max overlap from Fourier coefficients."""
    h = _build_h_from_fourier(coeffs)
    return _compute_max_overlap(h)

def _try_candidate(h):
    """Try a candidate h, return (h, score) if valid and better, else (None, inf)."""
    h = np.clip(h, 0.0, 1.0)
    target_sum = N / 2.0
    current_sum = h.sum()
    if current_sum > 1e-10:
        h = h * (target_sum / current_sum)
        h = np.clip(h, 0.0, 1.0)
        for _ in range(5):
            s = h.sum()
            if s > 1e-10:
                h = h * (target_sum / s)
                h = np.clip(h, 0.0, 1.0)
            else:
                break
    score = _compute_max_overlap(h)
    return h, score

def construct():
    start_time = time.time()
    
    # Baseline: uniform 0.5
    best_h = np.full(N, 0.5)
    best_score = _compute_max_overlap(best_h)  # should be 0.25
    
    x = np.linspace(0, 2, N, endpoint=False)
    
    # Try analytic candidates first
    candidates = []
    
    # h = 0.5 + 0.5*sin(pi*x)
    h1 = 0.5 + 0.5 * np.sin(np.pi * x)
    candidates.append(h1)
    
    # h = 0.5 + 0.5*cos(pi*x)
    h2 = 0.5 + 0.5 * np.cos(np.pi * x)
    candidates.append(h2)
    
    # Square wave: 1 on [0,1), 0 on [1,2)
    h3 = (x < 1.0).astype(float)
    candidates.append(h3)
    
    # h = 0.5 + 0.3*sin(2*pi*x)
    h4 = 0.5 + 0.3 * np.sin(2 * np.pi * x)
    candidates.append(h4)
    
    # h = 0.5 + 0.4*cos(2*pi*x)
    h5 = 0.5 + 0.4 * np.cos(2 * np.pi * x)
    candidates.append(h5)
    
    # h = 0.5 + 0.3*sin(3*pi*x)
    h6 = 0.5 + 0.3 * np.sin(3 * np.pi * x)
    candidates.append(h6)
    
    for h_cand in candidates:
        h_new, score = _try_candidate(h_cand)
        if score < best_score:
            best_score = score
            best_h = h_new
    
    # Multi-start optimization with Fourier coefficients
    n_modes = 6  # Use 6 modes for flexibility
    n_coeffs = 2 * n_modes
    bounds = [(-0.5, 0.5)] * n_coeffs
    
    rng = np.random.default_rng(42)
    
    n_starts = 20
    for start_idx in range(n_starts):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Random initial point
        if start_idx == 0:
            x0 = np.zeros(n_coeffs)
        elif start_idx == 1:
            # Start from best known candidate's approximate Fourier coefficients
            x0 = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            x0 = rng.uniform(-0.4, 0.4, n_coeffs)
        
        try:
            result = minimize(
                _objective_fourier, x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            h_new = _build_h_from_fourier(result.x)
            score = _compute_max_overlap(h_new)
            
            if score < best_score:
                best_score = score
                best_h = h_new.copy()
        except Exception:
            pass
    
    # Additional refinement with Nelder-Mead on the best Fourier coefficients if time permits
    if time.time() - start_time < DEADLINE - 10:
        try:
            # Reconstruct Fourier coefficients from best_h
            h_best = best_h
            x = np.linspace(0, 2, N, endpoint=False)
            # Fit Fourier coefficients via least squares
            n_modes_refine = 4
            A = np.zeros((N, 2 * n_modes_refine))
            for k in range(1, n_modes_refine + 1):
                A[:, 2*(k-1)] = np.cos(np.pi * k * x)
                A[:, 2*(k-1)+1] = np.sin(np.pi * k * x)
            target = h_best - 0.5
            coeffs, _, _, _ = np.linalg.lstsq(A, target, rcond=None)
            
            # Refine with Nelder-Mead
            result = minimize(
                _objective_fourier, coeffs,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            h_new = _build_h_from_fourier(result.x)
            score = _compute_max_overlap(h_new)
            
            if score < best_score:
                best_score = score
                best_h = h_new.copy()
        except Exception:
            pass
    
    # Final validation
    best_h = np.clip(best_h, 0.0, 1.0)
    if abs(np.sum(best_h) - N/2) > 1e-6:
        best_h = np.full(N, 0.5)
    
    return best_h
