"""ADMM-based optimization: minimize autoconvolution peak via alternating direction method of multipliers."""
import time
import numpy as np
from scipy.fft import rfft, irfft

N = 400

# EVOLVE-BLOCK-START

def _autoconv_fft(h):
    """Compute full autoconvolution using FFT for speed."""
    n = 2 * N - 1
    H = rfft(h, n=n)
    c = irfft(H * H, n=n)
    return c

def _evaluate_exact(h):
    """Compute the exact objective value."""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    c = _autoconv_fft(h)
    return abs(2.0 * N * np.max(c) / (s * s))

def _admm_solve(rho, max_iter, tol=1e-8, target_peak=1.0):
    """
    ADMM solver for: min (1/2)||h||^2 subject to sum(h)=N, c_k <= target_peak for all k
    where c = autoconv(h).
    
    Variables: h (heights), c (autoconv entries), y (dual variables for c constraint)
    """
    n_conv = 2 * N - 1
    
    # Initialize
    h = np.ones(N)
    c = _autoconv_fft(h)
    y = np.zeros(n_conv)  # dual for c <= target_peak
    y_sum = 0.0  # dual for sum constraint
    
    best_h = h.copy()
    best_val = _evaluate_exact(h)
    
    for it in range(max_iter):
        # --- h-update: min (1/2)||h||^2 + (rho/2)||autoconv(h) - c + y/rho||^2 + y_sum*(sum(h)-N)
        # This is a quadratic in h. We solve via gradient descent on the h-subproblem.
        # Actually, let's reformulate: the h-update minimizes:
        # (1/2)||h||^2 + (rho/2)||autoconv(h) - c + y/rho||^2 + y_sum*(sum(h) - N)
        # = (1/2)||h||^2 + (rho/2)||autoconv(h) - (c - y/rho)||^2 + y_sum*sum(h) - y_sum*N
        # 
        # Gradient w.r.t. h: h + rho * d(autoconv)/dh * (autoconv(h) - c + y/rho) + y_sum
        # 
        # d(autoconv)/dh[i] = sum_j h[j] where the convolution index matches
        # This is complex. Let's use a simpler approach:
        # 
        # Alternative: Use the fact that autoconv in frequency domain is |H|^2
        # So the h-update becomes a quadratic form in H.
        
        # Let's use a simpler ADMM formulation:
        # min (1/2)||h||^2 + (rho/2)||autoconv(h) - z + u||^2 + y_sum*(sum(h)-N)
        # where z is the auxiliary variable for the autoconv, u is the dual
        
        # Actually, let me use a cleaner formulation:
        # Primal: min_{h,z} (1/2)||h||^2 + I_{z<=t}(z) s.t. z = autoconv(h), sum(h) = N
        # 
        # ADMM:
        # h-update: min (1/2)||h||^2 + (rho/2)||autoconv(h) - z + u||^2 + y_sum*(sum(h)-N)
        # z-update: min I_{z<=t}(z) + (rho/2)||autoconv(h) - z + u||^2  =>  z = min(autoconv(h)+u, t)
        # u-update: u = u + autoconv(h_new) - z_new
        # y_sum-update: y_sum = y_sum + sum(h_new) - N
        
        z = c  # auxiliary for autoconv
        u = y  # dual for autoconv constraint
        y_s = y_sum
        
        for sub_it in range(50):
            # h-update via gradient descent
            auto_h = _autoconv_fft(h)
            resid = auto_h - z + u
            # Gradient: h + rho * (d autoconv/dh)^T resid + y_s
            # d autoconv / dh: if autoconv[k] = sum_i h[i]*h[i+k] (for k>=0)
            # d autoconv[k] / dh[j] = h[j+k] + h[j-k] (with appropriate bounds)
            # This is equivalent to: gradient = h + rho * (something involving resid and h)
            
            # Simpler: use the fact that in frequency domain, autoconv = |H|^2
            # So d/dh (rho/2 ||autoconv - z + u||^2) 
            # = rho * (d autoconv/dh)^T (autoconv - z + u)
            # 
            # In frequency: autoconv = ifft(|fft(h)|^2)
            # d autoconv / dh in freq domain: 2 * Re(H * conj(dH/dh))
            # This gets complex. Let's just do numerical gradient or a simpler approach.
            
            # Let's use a simpler approach: proximal gradient
            # Actually, for the h-update, since it's quadratic, we can solve it exactly
            # if we write it in frequency domain.
            
            # Let me try a different approach: use the structure more directly.
            # The objective for h-update is:
            # (1/2)||h||^2 + (rho/2)||ifft(|fft(h)|^2) - z + u||^2 + y_s * sum(h)
            
            # This is hard to solve exactly. Let's use gradient descent with FFT-based gradients.
            
            # Compute gradient numerically via FFT
            # d/dh_j (1/2)||h||^2 = h_j
            # d/dh_j (y_s * sum(h)) = y_s
            # d/dh_j (rho/2)||autoconv(h) - z + u||^2 
            #   = rho * sum_k (autoconv[k] - z[k] + u[k]) * d autoconv[k]/dh_j
            # d autoconv[k]/dh_j = h[j+k] + h[j-k] (circular, with bounds)
            
            # This is O(N^2) which is too slow. Let's use a different strategy.
            
            # Alternative: Use the spectral representation
            # autoconv in freq: C_freq = |H_freq|^2
            # So the term is (rho/2)||ifft(|H|^2) - z + u||^2
            # = (rho/2) || |H|^2 - fft(z - u) ||^2 (by Parseval)
            
            # Let Z = fft(z - u)
            # Then we minimize: (1/2)||h||^2 + (rho/2)|| |H|^2 - Z ||^2 + y_s * sum(h)
            
            # In frequency: (1/2)||H||^2 + (rho/2)|| |H|^2 - Z ||^2 + y_s * N * Re(H[0])
            
            # This is still complex. Let me just do coordinate descent on h.
            
            # For now, let's use a simpler approach: just do the z and u updates,
            # and for h, use a fixed-point iteration or simple gradient step.
            
            # Actually, let me simplify the whole approach.
            
            break  # Skip the inner loop for now, use a simpler method
        
        # Simplified: Just do z and u updates, and for h, use a projection
        # h-update: project onto sum(h)=N while minimizing distance to something
        # For now, let's just scale h to have sum N
        h = h / np.sum(h) * N
        
        # z-update: z = min(autoconv(h) + u, target_peak)
        auto_h = _autoconv_fft(h)
        z = np.minimum(auto_h + u, target_peak)
        
        # u-update
        u = u + auto_h - z
        
        # y_sum update
        y_s = y_s + np.sum(h) - N
        
        # Check convergence
        primal_resid = np.max(np.abs(auto_h - z))
        if primal_resid < tol:
            break
        
        # Track best
        val = _evaluate_exact(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    return best_h, best_val

def _admm_v2(rho, max_iter, tol=1e-8):
    """
    ADMM v2: More careful formulation.
    
    Problem: min_h (1/2)||h||^2 s.t. sum(h) = N, max(autoconv(h)) <= T
    
    We'll use a penalty approach within ADMM:
    min (1/2)||h||^2 + (rho/2)*[max(0, max(autoconv(h)) - T)]^2 + (rho_s/2)*(sum(h)-N)^2
    
    And solve via gradient descent with FFT-based gradient computation.
    """
    h = np.ones(N)
    rho_s = 1.0
    T = 1.0  # target peak
    
    best_h = h.copy()
    best_val = _evaluate_exact(h)
    
    lr = 0.01
    
    for it in range(max_iter):
        auto_h = _autoconv_fft(h)
        s = np.sum(h)
        
        # Gradient of penalty terms
        # d/dh (rho/2)*[max(0, max(autoconv) - T)]^2
        max_idx = np.argmax(auto_h)
        max_val = auto_h[max_idx]
        if max_val > T:
            penalty_grad = rho * (max_val - T) * _autoconv_grad(h, max_idx)
        else:
            penalty_grad = np.zeros(N)
        
        # d/dh (rho_s/2)*(sum(h)-N)^2 = rho_s*(sum(h)-N)*1
        sum_grad = rho_s * (s - N) * np.ones(N)
        
        # d/dh (1/2)||h||^2 = h
        grad = h + penalty_grad + sum_grad
        
        # Gradient descent step
        h = h - lr * grad
        
        # Ensure sum is close to N
        h = h + (N - np.sum(h)) / N
        
        # Track best
        val = _evaluate_exact(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
        
        # Adaptive learning rate
        if it % 10 == 0:
            lr *= 0.95
    
    return best_h, best_val

def _autoconv_grad(h, k):
    """
    Compute gradient of autoconv[k] with respect to h.
    autoconv[k] = sum_i h[i]*h[i+k] for k >= 0 (with bounds)
    d autoconv[k]/dh[j] = h[j+k] + h[j-k] (where defined)
    """
    N_local = len(h)
    grad = np.zeros(N_local)
    
    if k < N_local:
        # k >= 0 case: autoconv[k] = sum_{i=0}^{N-1-k} h[i]*h[i+k]
        for i in range(N_local - k):
            grad[i] += h[i + k]
            grad[i + k] += h[i]
    else:
        # k >= N case: let lag = k - (2*N - 2), autoconv[k] = sum_{i=0}^{N-1-lag} h[i+lag]*h[i]
        lag = k - (2 * N_local - 2)
        for i in range(N_local - lag):
            grad[i + lag] += h[i]
            grad[i] += h[i + lag]
    
    return grad

def _spectral_flatten(h, iterations=100, lr=0.01):
    """
    Flatten the spectrum of h to reduce autoconv peaks.
    Minimize ||autoconv(h)||_inf by adjusting h in frequency domain.
    """
    best_h = h.copy()
    best_val = _evaluate_exact(h)
    
    for it in range(iterations):
        H = rfft(h, n=2*N-1)
        auto_freq = H * H
        
        # We want to reduce the peak of ifft(auto_freq)
        # Gradient of ||ifft(auto_freq)||_inf w.r.t. H
        auto_real = irfft(auto_freq, n=2*N-1)
        max_idx = np.argmax(auto_real)
        
        # d auto_real[max_idx] / dH[j] for real FFT
        # This is complex, so let's use a simpler approach:
        # Reduce the magnitude of H at frequencies that contribute to the peak
        
        # Simple heuristic: reduce H where |H| is large
        H_mag = np.abs(H)
        H_phase = np.angle(H)
        
        # Compress dynamic range
        H_new_mag = H_mag / (1 + 0.1 * H_mag)
        H_new = H_new_mag * np.exp(1j * H_phase)
        
        h_new = irfft(H_new, n=2*N-1)[:N]
        
        # Ensure sum constraint
        h_new = h_new + (N - np.sum(h_new)) / N
        
        val = _evaluate_exact(h_new)
        if val < best_val:
            best_val = val
            best_h = h_new.copy()
            h = h_new.copy()
        else:
            # Try smaller step
            H_blend = 0.5 * H + 0.5 * H_new
            h_blend = irfft(H_blend, n=2*N-1)[:N]
            h_blend = h_blend + (N - np.sum(h_blend)) / N
            val_blend = _evaluate_exact(h_blend)
            if val_blend < best_val:
                best_val = val_blend
                best_h = h_blend.copy()
                h = h_blend.copy()
    
    return best_h, best_val

def _randomized_admm(rho, max_iter, n_restarts=3, seed=42):
    """
    Randomized ADMM: multiple restarts with different initializations.
    """
    rng = np.random.default_rng(seed)
    best_h = np.ones(N)
    best_val = _evaluate_exact(best_h)
    
    for restart in range(n_restarts):
        if restart == 0:
            h = np.ones(N)
        elif restart == 1:
            h = np.ones(N) + rng.standard_normal(N) * 0.1
            h = h + (N - np.sum(h)) / N
        else:
            h = np.ones(N) + rng.standard_normal(N) * 0.5
            h = h + (N - np.sum(h)) / N
        
        h_opt, val_opt = _admm_v2(rho, max_iter)
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt.copy()
    
    return best_h, best_val

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin from 150s limit
    
    best_h = np.ones(N)
    best_val = _evaluate_exact(best_h)  # Should be 2.0
    
    def try_h(h):
        nonlocal best_h, best_val
        if not np.all(np.isfinite(h)):
            return
        s = np.sum(h)
        if abs(s) < 1e-10:
            return
        val = _evaluate_exact(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # === Phase 1: ADMM with different rho values ===
    for rho in [0.1, 0.5, 1.0, 2.0, 5.0]:
        if time.time() > deadline - 30.0:
            break
        
        h_admm, val_admm = _admm_v2(rho, max_iter=500)
        try_h(h_admm)
        
        # Spectral flattening on the result
        if time.time() < deadline - 25.0:
            h_flat, val_flat = _spectral_flatten(h_admm, iterations=50)
            try_h(h_flat)
    
    # === Phase 2: Randomized ADMM ===
    if time.time() < deadline - 20.0:
        h_rand, val_rand = _randomized_admm(rho=1.0, max_iter=300, n_restarts=5)
        try_h(h_rand)
    
    # === Phase 3: Spectral flattening from multiple starting points ===
    if time.time() < deadline - 15.0:
        rng = np.random.default_rng(123)
        for _ in range(5):
            if time.time() > deadline - 15.0:
                break
            h_init = np.ones(N) + rng.standard_normal(N) * 0.2
            h_init = h_init + (N - np.sum(h_init)) / N
            h_flat, val_flat = _spectral_flatten(h_init, iterations=100)
            try_h(h_flat)
    
    # === Phase 4: Coordinate descent polish ===
    if time.time() < deadline - 5.0:
        step_sizes = [0.1, 0.05, 0.01, 0.005]
        improved = True
        while improved and (time.time() < deadline - 5.0):
            improved = False
            for _ in range(100):
                if time.time() > deadline - 5.0:
                    break
                idx = rng.integers(0, N)
                for delta in step_sizes:
                    for sign in [1, -1]:
                        h_new = best_h.copy()
                        h_new[idx] += sign * delta
                        val_new = _evaluate_exact(h_new)
                        if val_new < best_val - 1e-14:
                            best_val = val_new
                            best_h = h_new
                            improved = True
                            break
                    if improved:
                        break
            if not improved:
                step_sizes = [s * 0.5 for s in step_sizes]
                if step_sizes[0] < 1e-5:
                    break
    
    # Safety check
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-10:
        return np.ones(N)
    
    return best_h

# EVOLVE-BLOCK-END
