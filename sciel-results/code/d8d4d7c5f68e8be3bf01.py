"""Spectral amplitude shaping for the first autocorrelation inequality."""
import time
import numpy as np
from scipy.optimize import minimize

N = 600
DEADLINE = 115.0  # safety margin before 120s kill

def _compute_score(heights):
    """Compute the normalized maximum of the circular autoconvolution."""
    h = np.asarray(heights, dtype=np.float64)
    if np.any(h < -1e-12):
        return 1e10
    h = np.maximum(h, 0.0)
    
    # Circular autocorrelation via FFT
    nfft = 1 << (2*N - 1).bit_length()  # next power of 2 >= 2N-1
    H = np.fft.rfft(h, n=nfft)
    autocorr = np.fft.irfft(H * np.conj(H), n=nfft)[:N]
    
    total = np.sum(h)
    if total < 1e-12:
        return 1e10
    
    score = N * np.max(autocorr) / (total * total)
    return float(score)

def _smooth_score(h, beta=100.0):
    """Smooth approximation of the score using log-sum-exp for the max."""
    h = np.maximum(h, 0.0)
    nfft = 1 << (2*N - 1).bit_length()
    H = np.fft.rfft(h, n=nfft)
    autocorr = np.fft.irfft(H * np.conj(H), n=nfft)[:N]
    
    total = np.sum(h)
    if total < 1e-12:
        return 1e10
    base = N / (total * total)
    
    # Log-sum-exp for max(autocorr)
    a = autocorr * base
    max_a = np.max(a)
    if max_a > 0:
        lse = max_a + np.log(np.sum(np.exp(a - max_a)))
    else:
        lse = 0.0
    return lse

def _build_fine_grid():
    """Build a fine grid for interpolation if needed."""
    return np.linspace(-0.25, 0.25, N, endpoint=False)

def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    def time_left():
        return DEADLINE - elapsed()
    
    best_h = np.ones(N)
    best_score = _compute_score(best_h)
    
    # Strategy: Use a low-dimensional parameterization with smooth basis functions
    # Parameterize h as a sum of Gaussian bumps with optimized centers, widths, and amplitudes
    # This allows complex shapes while keeping optimization tractable
    
    def _gaussian_basis_param(x):
        """
        x = [amp_1, center_1, width_1, amp_2, center_2, width_2, ...]
        Returns h vector of length N
        """
        n_bumps = len(x) // 3
        h = np.zeros(N)
        idx = np.arange(N)
        # Map index to position in [-0.25, 0.25]
        positions = -0.25 + idx * (0.5 / N)
        
        for i in range(n_bumps):
            amp = x[3*i]
            center = x[3*i+1]
            width = x[3*i+2]
            # Ensure positive width
            width = max(abs(width), 1e-4)
            h += amp * np.exp(-0.5 * ((positions - center) / width) ** 2)
        
        return np.maximum(h, 0.0)
    
    def _obj_gaussian(x):
        h = _gaussian_basis_param(x)
        return _smooth_score(h)
    
    # Try different numbers of Gaussian bumps
    n_bumps_list = [3, 5, 8, 12]
    
    np.random.seed(42)
    
    for n_bumps in n_bumps_list:
        if time_left() < 20.0:
            break
        
        n_params = 3 * n_bumps
        
        # Multiple random initializations
        for trial in range(3):
            if time_left() < 10.0:
                break
            
            # Initialize parameters
            x0 = np.zeros(n_params)
            for i in range(n_bumps):
                x0[3*i] = 1.0  # amplitude
                x0[3*i+1] = np.random.uniform(-0.2, 0.2)  # center
                x0[3*i+2] = np.random.uniform(0.02, 0.1)  # width
            
            # Bounds
            bounds = []
            for i in range(n_bumps):
                bounds.append((0.0, 10.0))  # amplitude
                bounds.append((-0.25, 0.25))  # center
                bounds.append((0.005, 0.5))  # width
            
            try:
                result = minimize(
                    _obj_gaussian,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
                
                h_opt = _gaussian_basis_param(result.x)
                sc = _compute_score(h_opt)
                
                if sc < best_score:
                    best_score = sc
                    best_h = h_opt.copy()
            except Exception:
                pass
    
    # Additional strategy: Optimize a symmetric cosine-tapered shape
    # h[i] = A * (0.5 + 0.5*cos(pi * (2*i/N - 1))) for i in [0, N)
    # with optional gaps at the edges
    
    def _cosine_taper_param(x):
        """
        x = [amp, gap_left, gap_right, taper_power]
        """
        amp = x[0]
        gap_left = x[1]
        gap_right = x[2]
        taper_power = x[3]
        
        h = np.zeros(N)
        n_active = N - int(gap_left * N) - int(gap_right * N)
        if n_active <= 0:
            return h
        
        start = int(gap_left * N)
        end = N - int(gap_right * N)
        
        if end <= start:
            return h
        
        t = np.linspace(0, 1, end - start)
        # Cosine taper with power
        taper = (0.5 + 0.5 * np.cos(np.pi * t)) ** taper_power
        h[start:end] = amp * taper
        
        return np.maximum(h, 0.0)
    
    def _obj_cosine(x):
        h = _cosine_taper_param(x)
        return _smooth_score(h)
    
    if time_left() > 15.0:
        # Try cosine taper optimization
        x0 = np.array([1.0, 0.0, 0.0, 1.0])
        bounds = [(0.1, 10.0), (0.0, 0.4), (0.0, 0.4), (0.1, 5.0)]
        
        try:
            result = minimize(
                _obj_cosine,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            h_opt = _cosine_taper_param(result.x)
            sc = _compute_score(h_opt)
            if sc < best_score:
                best_score = sc
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final polishing: direct optimization on h with L-BFGS-B
    if time_left() > 10.0:
        def obj_direct(x):
            return _smooth_score(x)
        
        try:
            result = minimize(
                obj_direct,
                best_h,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 30}
            )
            h_polish = np.maximum(result.x, 0.0)
            sc2 = _compute_score(h_polish)
            if sc2 < best_score:
                best_score = sc2
                best_h = h_polish.copy()
        except Exception:
            pass
    
    return [float(x) for x in best_h]

if __name__ == '__main__':
    result = construct()
    print(f"Score: {_compute_score(result)}")
