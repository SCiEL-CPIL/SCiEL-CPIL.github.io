"""
Optimized solution for the third autocorrelation inequality task.

Structural approach: parameterize h as DC offset + few sinusoidal modes
with controlled symmetry, then do focused search over low-dimensional space.
This is far more efficient than generic gradient descent on 400 variables.
"""
import numpy as np
import time

N = 400

def _evaluate(h):
    """Evaluate the objective: abs(2*N*max(autoconv) / sum(h)^2)"""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _build_h(dc, modes):
    """
    Build h from DC offset and sinusoidal modes.
    modes: list of (freq, amp, phase, symmetry)
    symmetry: 'antisym' means a[i] = -a[N-1-i], 'sym' means a[i] = a[N-1-i]
    """
    t = np.arange(N)
    h = dc * np.ones(N)
    for (freq, amp, phase, sym) in modes:
        if sym == 'antisym':
            # Antisymmetric about center: sin with appropriate phase
            # For antisym: a[i] = -a[N-1-i]
            # Use sin(2*pi*freq*(t - (N-1)/2) / N) which is antisymmetric
            a = amp * np.sin(2 * np.pi * freq * (t - (N - 1) / 2.0) / N + phase)
            # Enforce antisymmetry explicitly
            a = (a - a[::-1]) / 2.0
            h += a
        elif sym == 'sym':
            # Symmetric about center
            a = amp * np.cos(2 * np.pi * freq * (t - (N - 1) / 2.0) / N + phase)
            # Enforce symmetry
            a = (a + a[::-1]) / 2.0
            h += a
        else:
            # General
            a = amp * np.sin(2 * np.pi * freq * t / N + phase)
            h += a
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    rng = np.random.default_rng(123)
    
    best_h = None
    best_val = 1e10
    
    def try_h(h):
        nonlocal best_h, best_val
        val = _evaluate(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
        return val
    
    # Phase 1: Systematic grid search over DC and 1-2 mode parameters
    # Key insight: we want sum(h) large (big DC) but max autoconv small.
    # Antisymmetric modes have sum=0, so they don't affect sum(h).
    # The autoconvolution of DC is N*dc^2 at lag 0.
    # Cross terms between DC and antisymmetric mode can be controlled.
    
    # Try various DC levels
    dc_values = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    
    # Try various frequencies and amplitudes for antisymmetric modes
    freqs = [1, 2, 3, 4, 5, 7, 10]
    amp_fracs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Single antisymmetric mode
    for dc in dc_values:
        for freq in freqs:
            for amp_frac in amp_fracs:
                if time.time() > deadline - 30.0:
                    break
                amp = dc * amp_frac
                h = _build_h(dc, [(freq, amp, 0.0, 'antisym')])
                try_h(h)
    
    # Two antisymmetric modes
    if time.time() < deadline - 20.0:
        for dc in [5.0, 10.0, 20.0, 50.0]:
            for freq1 in [1, 2, 3]:
                for freq2 in [2, 3, 5, 7]:
                    if freq1 == freq2:
                        continue
                    for amp_frac1 in [0.3, 0.5, 0.7]:
                        for amp_frac2 in [0.2, 0.3, 0.5]:
                            if time.time() > deadline - 15.0:
                                break
                            modes = [
                                (freq1, dc * amp_frac1, 0.0, 'antisym'),
                                (freq2, dc * amp_frac2, 0.0, 'antisym'),
                            ]
                            h = _build_h(dc, modes)
                            try_h(h)
    
    # Phase 2: Local refinement around best found
    if best_h is None:
        best_h = np.ones(N)
        best_val = _evaluate(best_h)
    
    # Use scipy for local optimization on the full 400-dim space
    try:
        from scipy.optimize import minimize
        use_scipy = True
    except ImportError:
        use_scipy = False
    
    if use_scipy and time.time() < deadline - 25.0:
        # Soft-max approximation for differentiability
        def soft_max(x, beta=200.0):
            m = np.max(x)
            return m + np.log(np.sum(np.exp(x - m)) / len(x)) / beta
        
        def objective_scipy(h_flat):
            h = h_flat.reshape(N)
            s = np.sum(h)
            if abs(s) < 1e-10:
                return 1e10
            conv = np.convolve(h, h)
            sm = soft_max(conv, beta=100.0)
            return abs(2.0 * N * sm / (s * s))
        
        # Multiple restarts from perturbed best
        for restart in range(8):
            if time.time() > deadline - 10.0:
                break
            
            if restart == 0:
                h0 = best_h.copy()
            else:
                scale = 0.05 * np.std(best_h) * (1 + 0.5 * restart)
                h0 = best_h + rng.standard_normal(N) * scale
            
            try:
                result = minimize(
                    objective_scipy,
                    h0.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                h_opt = result.x.reshape(N)
                val_opt = _evaluate(h_opt)
                if val_opt < best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
            except Exception:
                pass
    
    # Phase 3: Coordinate descent refinement
    if time.time() < deadline - 15.0:
        h_curr = best_h.copy()
        lr = 0.01
        for coord_iter in range(500):
            if time.time() > deadline - 8.0:
                break
            
            improved = False
            for _ in range(N):  # One pass over coordinates
                if time.time() > deadline - 8.0:
                    break
                j = rng.integers(0, N)
                old_val = h_curr[j]
                
                # Try a few steps
                for step in [-lr, -lr/2, lr/2, lr]:
                    h_curr[j] = old_val + step
                    val_new = _evaluate(h_curr)
                    if val_new < best_val:
                        best_val = val_new
                        best_h = h_curr.copy()
                        improved = True
                        break
                    else:
                        h_curr[j] = old_val
                
                if not improved:
                    # Decay learning rate
                    lr *= 0.99
                    if lr < 1e-6:
                        break
    
    # Phase 4: Final random perturbations
    if time.time() < deadline - 3.0:
        for _ in range(300):
            if time.time() > deadline - 2.0:
                break
            h_new = best_h + rng.standard_normal(N) * 0.005 * np.std(best_h)
            val_new = _evaluate(h_new)
            if val_new < best_val:
                best_val = val_new
                best_h = h_new.copy()
    
    # Safety check
    if best_h is None or abs(np.sum(best_h)) < 1e-10:
        best_h = np.ones(N)
        best_val = _evaluate(best_h)
    
    return best_h
