"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    # The problem: minimize max(autoconvolution) / ||h||_2^2 (or similar normalization)
    # where h is a nonnegative step function on [-1/4, 1/4] with 600 intervals.
    # 
    # Key insight: The autoconvolution of h is (h * h)(x) = ∫ h(y)h(x-y) dy.
    # By Plancherel's theorem, the L2 norm of the autoconvolution relates to |H|^2.
    # But we want to minimize the MAX of the autoconvolution, not the L2 norm.
    #
    # Strategy: Use a smooth, concentrated function that minimizes peak autoconvolution.
    # A Gaussian-like function has autoconvolution that is also Gaussian, with peak 
    # proportional to 1/sigma. But we need to balance concentration vs spread.
    #
    # Better approach: Directly optimize the step heights using gradient-based methods.
    # Parameterize h as exp(-x^2 / sigma^2) * (1 + eps * perturbation) to enforce nonnegativity.
    
    # Time budget for optimization
    time_budget = 130.0
    
    # Start with a Gaussian-shaped function
    # The interval is [-1/4, 1/4], width = 1/2
    # Each interval has width = (1/2) / 600 = 1/1200
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 1.0/(2*1200)  # midpoints
    
    def compute_score(h):
        """Compute the normalized maximum of the autoconvolution."""
        # Autoconvolution via FFT
        # h is defined on [-1/4, 1/4] with 600 intervals
        # The autoconvolution will be on [-1/2, 1/2] with 1199 intervals
        # But we need to be careful about the normalization
        
        # Use FFT for efficient convolution
        # Pad to avoid circular convolution
        n_fft = 2 * N
        H = np.fft.rfft(h, n=n_fft)
        conv = np.fft.irfft(H * H, n=n_fft)
        
        # The autoconvolution of a function on [-1/4, 1/4] is on [-1/2, 1/2]
        # conv[0] corresponds to x = 0 (peak), conv[1] to x = dx, etc.
        # dx = (1/2) / N = 1/1200
        # But the convolution computed by FFT gives us values at the right points
        
        # Actually, let's think more carefully.
        # h[i] represents the value on interval [x_i - dx/2, x_i + dx/2] where dx = 1/1200
        # The autoconvolution (h*h)(x) = ∫ h(y)h(x-y) dy
        # For discrete steps, this is a sum: sum_i h[i] * h[j] * overlap
        # 
        # Using FFT: if we treat h as samples, the linear convolution gives us
        # the autoconvolution sampled at the right points.
        
        # The peak of the autoconvolution is at x=0 (center)
        # conv[0] is the value at x=0
        # We need the maximum of the autoconvolution
        
        # The autoconvolution is symmetric and peaked at 0
        # So the max is at conv[0]
        
        # Normalization: ||h||_2^2 = sum(h^2) * dx
        norm_sq = np.sum(h**2) * (1.0/1200.0)
        
        if norm_sq < 1e-12:
            return 1e10
        
        # The normalized max is max(conv) / norm_sq? Or max(conv) * something?
        # The task says "normalized maximum of its autoconvolution"
        # This likely means max(autoconv) / ||h||_2^2 or similar
        
        # Let's use max(conv) / norm_sq as the score
        max_conv = np.max(conv)
        score = max_conv / norm_sq
        return score
    
    def objective(params):
        """Objective function for optimization."""
        # params = [log_sigma, log_amplitude, ...perturbation params]
        log_sigma = params[0]
        log_amp = params[1]
        
        sigma = np.exp(log_sigma)
        amp = np.exp(log_amp)
        
        # Gaussian centered at 0
        h = amp * np.exp(-0.5 * (x / sigma)**2)
        
        # Ensure nonnegativity (already guaranteed by exp)
        # Clip to avoid numerical issues
        h = np.maximum(h, 0.0)
        
        return compute_score(h)
    
    # Try multiple initial conditions with different sigma values
    best_score = 1e10
    best_h = None
    
    # Initial guesses for sigma (standard deviation)
    # The domain is [-0.25, 0.25], so sigma should be in [0.01, 0.25]
    sigma_guesses = [0.02, 0.05, 0.1, 0.15, 0.2]
    
    for sigma_guess in sigma_guesses:
        if time.time() - start_time > deadline:
            break
        
        # Initial parameters: log(sigma), log(amplitude)
        # Choose amplitude to normalize ||h||_2^2 = 1
        h0 = np.exp(-0.5 * (x / sigma_guess)**2)
        norm0 = np.sqrt(np.sum(h0**2) * (1.0/1200.0))
        if norm0 > 1e-12:
            h0 = h0 / norm0
            amp0 = h0[0] / np.exp(-0.5 * (x[0] / sigma_guess)**2)
            x0 = [np.log(sigma_guess), np.log(max(amp0, 1e-10))]
        else:
            x0 = [np.log(sigma_guess), 0.0]
        
        # Use Nelder-Mead for robustness (no gradient needed)
        try:
            result = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 200,
                    'xatol': 1e-6,
                    'fatol': 1e-8
                }
            )
            
            if result.success or result.fun < best_score:
                log_sigma_opt = result.x[0]
                log_amp_opt = result.x[1]
                sigma_opt = np.exp(log_sigma_opt)
                amp_opt = np.exp(log_amp_opt)
                
                h_opt = amp_opt * np.exp(-0.5 * (x / sigma_opt)**2)
                h_opt = np.maximum(h_opt, 0.0)
                
                score_opt = result.fun
                if score_opt < best_score:
                    best_score = score_opt
                    best_h = h_opt.copy()
                    
        except Exception:
            pass
    
    # If we found a good solution, return it
    if best_h is not None and time.time() - start_time < deadline:
        # Additional refinement: try to further optimize with more iterations
        # But first, let's see if we can do better with a different parametrization
        
        # Try a "top-hat" plus Gaussian mixture or other shapes
        # For now, return the best Gaussian found
        
        # Normalize to ensure the function is reasonable
        # The score is scale-invariant in a specific way, so normalization doesn't matter
        # but let's keep the values in a reasonable range
        max_val = np.max(best_h)
        if max_val > 0:
            # Scale so that max is around 1 (for numerical stability)
            pass  # Actually, the score is invariant to scaling, so no need
        
        return list(best_h)
    
    # Fallback: return a simple Gaussian
    sigma = 0.1
    h = np.exp(-0.5 * (x / sigma)**2)
    return list(h)
# EVOLVE-BLOCK-END
