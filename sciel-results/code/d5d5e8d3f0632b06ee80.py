"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Spectral design: represent h as 0.5 + x where x is zero-mean.
    Use Gaussian power spectra to ensure positive autocorrelation sidelobes.
    Optimize phases to minimize time-domain peak for better scaling.
    """
    np.random.seed(42)
    
    best_h = np.full(N, 0.5)
    best_score = np.inf
    
    # Search over Gaussian spectrum widths
    # sigma controls the width of the Gaussian in frequency domain
    # Larger sigma -> broader spectrum -> flatter autocorrelation (closer to 0)
    # Smaller sigma -> narrower spectrum -> oscillatory autocorrelation
    # We want autocorrelation to be positive and as large as possible relative to peak.
    # Actually, we want R_x(k) to be close to R_x(0).
    # A Gaussian spectrum gives a Gaussian autocorrelation.
    # R_x(k) = R_x(0) * exp(-k^2 / 2 sigma_t^2) where sigma_t is related to 1/sigma_f.
    # We want R_x(k) to be high. So we want a narrow time-domain pulse?
    # No, we want R_x(k) to decay slowly. So we want a broad time-domain signal?
    # Broad time-domain signal corresponds to narrow frequency spectrum.
    # So we want small sigma_f (narrow frequency spectrum).
    
    sigma_values = np.linspace(0.5, 5.0, 20)
    
    for sigma in sigma_values:
        # Define amplitudes
        j = np.arange(N)
        # Gaussian centered at 0
        amps = np.exp(-(j**2) / (2 * sigma**2))
        # Also center at N (due to periodicity of FFT)
        amps += np.exp(-((j - N)**2) / (2 * sigma**2))
        amps += np.exp(-((j + N)**2) / (2 * sigma**2))
        
        amps[0] = 0  # Zero mean
        
        # Try a few random phase configurations
        for trial in range(5):
            phases = np.random.uniform(0, 2 * np.pi, N)
            
            # Optimize phases to minimize max |x| (PAPR)
            # This allows us to scale x up as much as possible
            for it in range(100):
                X = amps * np.exp(1j * phases)
                x = np.fft.ifft(X).real
                
                # Find peak
                n_peak = np.argmax(np.abs(x))
                x_peak = x[n_peak]
                
                # Gradient step on phases to reduce x_peak
                # x_n = sum A_k cos(phi_k + 2 pi k n / N)
                # dx_n / d phi_k = -A_k sin(phi_k + 2 pi k n / N)
                # We want to reduce |x_n_peak|.
                # If x_peak > 0, we want to decrease x_peak.
                # So we want dx/dphi * dphi < 0.
                # dphi_k = -eta * sign(x_peak) * dx/dphi_k
                
                angles = phases + 2 * np.pi * np.outer(np.arange(N), np.arange(N)) / N
                # This is O(N^2), might be slow for N=100? 100x100=10k, fine.
                # But we only need the row for n_peak
                angles_peak = phases + 2 * np.pi * np.arange(N) * n_peak / N
                
                grad = -amps * np.sin(angles_peak)
                
                # Update phases
                step_size = 0.1
                if x_peak > 0:
                    phases -= step_size * grad
                else:
                    phases += step_size * grad
                
                phases = phases % (2 * np.pi)
            
            # Final reconstruction
            X = amps * np.exp(1j * phases)
            x = np.fft.ifft(X).real
            
            # Scale to fit [-0.5, 0.5]
            max_abs = np.max(np.abs(x))
            if max_abs > 1e-10:
                scale = 0.5 / max_abs
                x = x * scale
            
            # Ensure zero mean
            x = x - np.mean(x)
            
            # Clip
            x = np.clip(x, -0.5, 0.5)
            
            # Correct sum
            x = x - np.mean(x)
            x = np.clip(x, -0.5, 0.5)
            
            h = 0.5 + x
            h = np.clip(h, 0, 1)
            
            # Compute score
            H = np.fft.fft(h)
            autocorr = np.fft.ifft(np.abs(H)**2).real
            C_vals = N/2 - autocorr
            score = np.max(C_vals)
            
            if score < best_score:
                best_score = score
                best_h = h.copy()
    
    # Final correction
    sum_h = np.sum(best_h)
    if abs(sum_h - N/2) > 1e-10:
        best_h = best_h - (sum_h - N/2) / N
        best_h = np.clip(best_h, 0, 1)
    
    return best_h
# EVOLVE-BLOCK-END
