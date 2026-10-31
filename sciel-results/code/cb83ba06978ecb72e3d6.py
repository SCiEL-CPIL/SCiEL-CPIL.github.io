"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_max_overlap(h):
    """Compute the maximum cyclic overlap sum_i h[i]*(1-h[i+k]) over all shifts k."""
    n = len(h)
    # autocorrelation: corr[k] = sum_i h[i]*h[i+k] (cyclic)
    # overlap[k] = sum_i h[i]*(1-h[i+k]) = sum(h) - corr[k]
    # We want to minimize max_k overlap[k] = sum(h) - min_k corr[k]
    # So we want to maximize min_k corr[k]
    corr = np.correlate(h, h, mode='full')
    # For cyclic correlation, we need to handle wrap-around
    # corr from np.correlate for mode='full' gives non-cyclic
    # Let's compute cyclic correlation properly
    corr_cyclic = np.zeros(n)
    for k in range(n):
        corr_cyclic[k] = np.sum(h * np.roll(h, -k))
    total = np.sum(h)
    overlaps = total - corr_cyclic
    return np.max(overlaps)

def _build_h_from_fourier(coeffs, N):
    """Build h from Fourier coefficients, ensuring h in [0,1] and sum=N/2."""
    # coeffs: array of (amplitude, phase) pairs for frequencies 1..max_freq
    h = np.full(N, 0.5)
    for idx, (amp, phase) in enumerate(coeffs):
        freq = idx + 1
        h += amp * np.sin(2 * np.pi * freq * np.arange(N) / N + phase)
    # Clip to [0,1] and renormalize
    h = np.clip(h, 0, 1)
    # Adjust to make sum exactly N/2
    current_sum = np.sum(h)
    if current_sum > 0:
        # Scale toward 0.5 to fix sum while staying in [0,1]
        # Add a constant offset
        offset = (N / 2.0 - current_sum) / N
        h = h + offset
        h = np.clip(h, 0, 1)
        # Final correction: distribute remaining error
        remaining = N / 2.0 - np.sum(h)
        if abs(remaining) > 1e-10:
            # Add to elements with room
            h = h + remaining / N
            h = np.clip(h, 0, 1)
    return h

def _objective(params, N, max_freq):
    """Objective: maximize min autocorrelation (equivalent to minimize max overlap)."""
    coeffs = [(params[2*i], params[2*i+1]) for i in range(max_freq)]
    h = _build_h_from_fourier(coeffs, N)
    # Return negative of min correlation (we want to maximize it)
    corr_cyclic = np.zeros(N)
    for k in range(N):
        corr_cyclic[k] = np.sum(h * np.roll(h, -k))
    return -np.min(corr_cyclic)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = float('inf')
    best_h = np.full(N, 0.5)
    
    # Initial score for uniform
    best_score = _compute_max_overlap(best_h)
    
    # Try different numbers of Fourier modes
    for max_freq in [1, 2, 3, 4, 5]:
        if time.time() > deadline:
            break
        
        # Grid search over amplitudes and phases
        # For each mode, amplitude in [0, 0.4], phase in [0, 2*pi)
        # Use a coarse grid first
        n_amp_steps = 8
        n_phase_steps = 8
        
        amp_range = np.linspace(0, 0.45, n_amp_steps)
        phase_range = np.linspace(0, 2*np.pi, n_phase_steps, endpoint=False)
        
        # For max_freq=1, do full grid
        if max_freq == 1:
            for a1 in amp_range:
                if time.time() > deadline:
                    break
                for p1 in phase_range:
                    if time.time() > deadline:
                        break
                    h = _build_h_from_fourier([(a1, p1)], N)
                    score = _compute_max_overlap(h)
                    if score < best_score:
                        best_score = score
                        best_h = h.copy()
        
        elif max_freq == 2:
            # Coarser grid for 2 modes
            n_coarse = 5
            amp_coarse = np.linspace(0, 0.4, n_coarse)
            phase_coarse = np.linspace(0, 2*np.pi, n_coarse, endpoint=False)
            for a1 in amp_coarse:
                if time.time() > deadline:
                    break
                for p1 in phase_coarse:
                    if time.time() > deadline:
                        break
                    for a2 in amp_coarse:
                        if time.time() > deadline:
                            break
                        for p2 in phase_coarse:
                            if time.time() > deadline:
                                break
                            h = _build_h_from_fourier([(a1, p1), (a2, p2)], N)
                            score = _compute_max_overlap(h)
                            if score < best_score:
                                best_score = score
                                best_h = h.copy()
        
        # Local refinement using scipy
        if time.time() < deadline - 20:
            # Start from best found so far, try to refine
            # Convert best_h to Fourier coefficients via DFT
            H = np.fft.fft(best_h)
            # Use first max_freq positive frequencies
            params = []
            for f in range(1, max_freq + 1):
                amp = 2 * np.abs(H[f]) / N
                phase = np.angle(H[f])
                params.extend([amp, phase])
            
            result = minimize(
                _objective, 
                np.array(params), 
                args=(N, max_freq),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            if result.success or result.fun < _objective(np.array(params), N, max_freq):
                coeffs = [(result.x[2*i], result.x[2*i+1]) for i in range(max_freq)]
                h_refined = _build_h_from_fourier(coeffs, N)
                score = _compute_max_overlap(h_refined)
                if score < best_score:
                    best_score = score
                    best_h = h_refined.copy()
    
    # Try a different structural approach: alternating high/low blocks
    if time.time() < deadline - 10:
        # Try block patterns: alternate between high and low values
        # h = [a, b, a, b, ...] or [a, a, b, b, a, a, b, b, ...]
        for block_size in [1, 2, 5, 10]:
            if time.time() > deadline:
                break
            n_blocks = N // block_size
            if n_blocks < 2:
                continue
            # Try different high/low value pairs
            for a in np.linspace(0.3, 0.7, 7):
                if time.time() > deadline:
                    break
                b = N / 2.0 / n_blocks - a  # to make average work out
                # Actually, sum = n_blocks * block_size * (a+b)/2 if alternating
                # Let's just try: first half of blocks = a, second half = b
                # No, let's do proper alternating
                for b in np.linspace(0.05, 0.95, 7):
                    if time.time() > deadline:
                        break
                    # Build pattern
                    h_pattern = np.zeros(N)
                    for i in range(N):
                        block_idx = i // block_size
                        if block_idx % 2 == 0:
                            h_pattern[i] = a
                        else:
                            h_pattern[i] = b
                    # Check sum
                    s = np.sum(h_pattern)
                    if abs(s - N/2.0) < 1e-6:
                        score = _compute_max_overlap(h_pattern)
                        if score < best_score:
                            best_score = score
                            best_h = h_pattern.copy()
    
    # Final: ensure sum is exactly N/2
    s = np.sum(best_h)
    if abs(s - N/2.0) > 1e-10:
        # Distribute correction
        correction = (N/2.0 - s) / N
        best_h = best_h + correction
        best_h = np.clip(best_h, 0, 1)
        # If clipping changed sum, do iterative correction
        for _ in range(10):
            s = np.sum(best_h)
            if abs(s - N/2.0) < 1e-10:
                break
            correction = (N/2.0 - s) / N
            best_h = best_h + correction
            best_h = np.clip(best_h, 0, 1)
    
    return best_h
# EVOLVE-BLOCK-END
