import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 600
DT = (1/4 - (-1/4)) / N  # interval width = 1/600
X_GRID = -1/4 + (np.arange(N) + 0.5) * DT  # midpoints of intervals

def _gaussian_mixture_heights(params, n_components=8):
    """Convert parameter vector to step heights using Gaussian mixture.
    
    params layout: [amp_1, center_1, width_1, amp_2, center_2, width_2, ...]
    """
    h = np.zeros(N)
    for k in range(n_components):
        amp = params[3*k]
        center = params[3*k+1]
        width = params[3*k+2]
        if width < 1e-6:
            width = 1e-6
        h += amp * np.exp(-0.5 * ((X_GRID - center) / width) ** 2)
    h = np.maximum(h, 0)
    s = np.sum(h)
    if s < 1e-15:
        return np.ones(N) / N
    return h / s

def _compute_score(h):
    """Compute normalized maximum autoconvolution."""
    s = np.sum(h)
    if s < 1e-15:
        return 1e10
    hn = h / s
    F = np.fft.fft(hn, n=2*N)
    conv = np.fft.ifft(F * F).real[:2*N-1]
    return np.max(conv)

def _smooth_max(conv, beta):
    """Log-sum-exp approximation to max. Higher beta = sharper."""
    m = np.max(conv)
    if m < -1e15:
        return m
    shifted = beta * (conv - m)
    # Clip to avoid overflow
    shifted = np.clip(shifted, -700, 700)
    return m + np.log(np.sum(np.exp(shifted))) / beta

def construct():
    start_time = time.time()
    time_limit = 145.0
    
    def elapsed():
        return time.time() - start_time
    
    # Start with uniform baseline
    h_uniform = np.ones(N)
    best_h = h_uniform.copy()
    best_val = _compute_score(best_h)
    
    # Try some simple structural candidates first
    candidates = [
        np.ones(N),
        np.clip(1.0 - 4.0 * np.abs(X_GRID), 0, None),  # triangular
        np.exp(-10.0 * X_GRID**2),  # narrow Gaussian
        np.exp(-2.0 * X_GRID**2),   # wide Gaussian
        np.clip(1.0 - 2.0 * np.abs(X_GRID), 0, None),  # wide triangle
    ]
    
    for h_cand in candidates:
        if elapsed() > time_limit:
            break
        h_cand = np.maximum(h_cand, 0)
        s = np.sum(h_cand)
        if s < 1e-15:
            continue
        h_cand = h_cand / s
        val = _compute_score(h_cand)
        if val < best_val:
            best_val = val
            best_h = h_cand.copy()
    
    # Phase 2: Differential evolution on Gaussian mixture parameters
    n_components = 8
    n_params = 3 * n_components
    
    # Bounds: amplitude in [0, 2], center in [-0.25, 0.25], width in [0.01, 0.5]
    bounds = []
    for _ in range(n_components):
        bounds.append((0.0, 2.0))   # amplitude
        bounds.append((-0.25, 0.25)) # center
        bounds.append((0.01, 0.5))   # width
    
    def objective(params):
        if elapsed() > time_limit:
            return 1e10
        h = _gaussian_mixture_heights(params, n_components)
        return _compute_score(h)
    
    # Run differential evolution
    try:
        # Use moderate beta for DE to get smoother landscape
        beta_de = 30.0
        
        def de_objective(params):
            if elapsed() > time_limit:
                return 1e10
            h = _gaussian_mixture_heights(params, n_components)
            s = np.sum(h)
            if s < 1e-15:
                return 1e10
            hn = h / s
            F = np.fft.fft(hn, n=2*N)
            conv = np.fft.ifft(F * F).real[:2*N-1]
            return _smooth_max(conv, beta_de)
        
        result = differential_evolution(
            de_objective,
            bounds,
            maxiter=200,
            popsize=15,
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            polish=False
        )
        
        h_result = _gaussian_mixture_heights(result.x, n_components)
        val_result = _compute_score(h_result)
        
        if val_result < best_val:
            best_val = val_result
            best_h = h_result.copy()
            
    except Exception:
        pass
    
    # Phase 3: Fine-tune with L-BFGS-B using smooth max with annealed temperature
    if elapsed() < time_limit - 20:
        if 'result' in locals() and result.success:
            x0 = result.x.copy()
        else:
            x0 = np.array([
                np.random.uniform(0, 1) for _ in range(n_components)
            ] + [
                np.random.uniform(-0.2, 0.2) for _ in range(n_components)
            ] + [
                np.random.uniform(0.05, 0.3) for _ in range(n_components)
            ])
        
        try:
            # Anneal beta from low (smooth) to high (sharp)
            betas = [10.0, 50.0, 200.0, 1000.0]
            current_x = x0.copy()
            for beta in betas:
                if elapsed() > time_limit - 15:
                    break
                
                def smooth_objective(params, b=beta):
                    if elapsed() > time_limit:
                        return 1e10
                    h = _gaussian_mixture_heights(params, n_components)
                    s = np.sum(h)
                    if s < 1e-15:
                        return 1e10
                    hn = h / s
                    F = np.fft.fft(hn, n=2*N)
                    conv = np.fft.ifft(F * F).real[:2*N-1]
                    return _smooth_max(conv, b)
                
                res = minimize(
                    smooth_objective,
                    current_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 3000}
                )
                
                h_res = _gaussian_mixture_heights(res.x, n_components)
                val_res = _compute_score(h_res)
                
                if val_res < best_val:
                    best_val = val_res
                    best_h = h_res.copy()
                
                current_x = res.x
                
        except Exception:
            pass
    
    # Phase 4: Final high-precision polish with very high beta (near-exact max)
    if elapsed() < time_limit - 5:
        # Use the best h found to reconstruct parameters, or use last x
        if 'current_x' in locals():
            x0 = current_x.copy()
        elif 'result' in locals() and result.success:
            x0 = result.x.copy()
        else:
            x0 = np.array([
                np.random.uniform(0, 1) for _ in range(n_components)
            ] + [
                np.random.uniform(-0.2, 0.2) for _ in range(n_components)
            ] + [
                np.random.uniform(0.05, 0.3) for _ in range(n_components)
            ])
        
        try:
            # Use very high beta for near-exact max
            beta_final = 5000.0
            
            def final_objective(params):
                if elapsed() > time_limit:
                    return 1e10
                h = _gaussian_mixture_heights(params, n_components)
                s = np.sum(h)
                if s < 1e-15:
                    return 1e10
                hn = h / s
                F = np.fft.fft(hn, n=2*N)
                conv = np.fft.ifft(F * F).real[:2*N-1]
                return _smooth_max(conv, beta_final)
            
            result3 = minimize(
                final_objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 5000}
            )
            
            h_result3 = _gaussian_mixture_heights(result3.x, n_components)
            val_result3 = _compute_score(h_result3)
            
            if val_result3 < best_val:
                best_val = val_result3
                best_h = h_result3.copy()
                
        except Exception:
            pass
    
    return list(best_h)
