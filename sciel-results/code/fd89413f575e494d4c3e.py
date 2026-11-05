import numpy as np
import time
import sys

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def compute_value(h):
        """Compute the objective value (lower is better)."""
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        conv = np.convolve(h, h)
        return abs(2.0 * N * np.max(conv) / (s * s))

    def smooth_max(x, tau=0.01):
        """Soft approximation of max using log-sum-exp."""
        x_max = np.max(x)
        if tau < 1e-8:
            return x_max
        return x_max + tau * np.log(np.sum(np.exp((x - x_max) / tau)))

    def generate_h_from_bandwidth(bw, window_type='hann'):
        """
        Generate h as a windowed sinc (low-pass FIR filter coefficients).
        The ideal low-pass impulse response is sinc(n * bw).
        We center it at N//2.
        """
        n = np.arange(N)
        center = N // 2
        # Frequency scaling: bw is the cutoff frequency relative to Nyquist?
        # Let's define bw in [0, 1] where 1 is Nyquist.
        # The discrete sinc is sin(pi * bw * k) / (pi * bw * k)
        k = n - center
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            sinc_vals = np.where(k == 0, bw, np.sin(np.pi * bw * k) / (np.pi * bw * k))
        
        # Apply window to reduce Gibbs phenomenon (sidelobes)
        if window_type == 'hann':
            window = 0.5 * (1 - np.cos(2 * np.pi * n / (N - 1)))
        elif window_type == 'hamming':
            window = 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))
        elif window_type == 'blackman':
            window = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 0.08 * np.cos(4 * np.pi * n / (N - 1))
        else:
            window = np.ones(N)
            
        h = sinc_vals * window
        return h

    def objective_bw(bw, window_type='hann', tau=0.05):
        """Objective function for bandwidth optimization."""
        if bw <= 0.01 or bw >= 1.0:
            return 1e10
        h = generate_h_from_bandwidth(bw, window_type)
        s = np.sum(h)
        if abs(s) < 1e-4:
            return 1e10
        conv = np.convolve(h, h)
        sm = smooth_max(conv, tau=tau)
        return 2.0 * N * sm / (s * s)

    best_h = np.ones(N)
    best_val = compute_value(best_h)

    # Phase 1: Global search over bandwidth and window type
    # The structure of sinc-based sequences is known to have low sidelobes.
    # We search for the optimal cutoff frequency.
    
    window_types = ['hann', 'hamming', 'blackman']
    
    # Coarse grid search for bandwidth
    bws = np.linspace(0.05, 0.95, 50)
    
    for wt in window_types:
        if time.time() > deadline - 30.0:
            break
        for bw in bws:
            val = objective_bw(bw, wt, tau=0.1)
            if val < best_val:
                best_val = val
                best_h = generate_h_from_bandwidth(bw, wt)
                print(f"Grid Search: bw={bw:.4f}, window={wt}, val={best_val:.6f}", file=sys.stderr)

    # Phase 2: Local refinement of bandwidth using scipy.optimize.minimize_scalar
    if time.time() < deadline - 20.0:
        try:
            from scipy.optimize import minimize_scalar
            
            best_bw = None
            best_wt = 'hann'
            min_val = np.inf
            
            # Find best window from coarse search
            for wt in window_types:
                res = minimize_scalar(
                    lambda x: objective_bw(x, wt, tau=0.05),
                    bounds=(0.05, 0.95),
                    method='bounded',
                    options={'xatol': 1e-6}
                )
                if res.fun < min_val:
                    min_val = res.fun
                    best_bw = res.x
                    best_wt = wt
            
            if best_bw is not None and min_val < best_val:
                best_val = min_val
                best_h = generate_h_from_bandwidth(best_bw, best_wt)
                print(f"Minimize Scalar: bw={best_bw:.6f}, window={best_wt}, val={best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"Minimize Scalar error: {e}", file=sys.stderr)

    # Phase 3: Fine-tuning via gradient descent on the direct h-vector
    # The sinc structure provides a great starting point.
    # We can now perform a few steps of coordinate descent or L-BFGS-B
    # to break the strict sinc symmetry if it helps further reduce the peak.
    
    if time.time() < deadline - 10.0:
        try:
            from scipy.optimize import minimize
            
            # Use a tight smooth max for the final polish
            def objective_tight(params, tau=0.005):
                h = params
                s = np.sum(h)
                if abs(s) < 1e-4:
                    return 1e10 + (1e-4 - abs(s)) * 1e6
                conv = np.convolve(h, h)
                sm = smooth_max(conv, tau=tau)
                return 2.0 * N * sm / (s * s)
                
            # Limit iterations to ensure we finish in time
            result = minimize(
                objective_tight,
                best_h,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 20}
            )
            h_opt = result.x
            val_opt = compute_value(h_opt)
            if val_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-6:
                best_h = h_opt.copy()
                best_val = val_opt
                print(f"Final L-BFGS-B: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"Final L-BFGS-B error: {e}", file=sys.stderr)

    # Final safety check
    if time.time() > deadline - 2.0:
        return np.ones(N)

    if best_val < 2.0 and np.all(np.isfinite(best_h)) and abs(np.sum(best_h)) > 1e-6:
        return best_h
    else:
        return np.ones(N)
