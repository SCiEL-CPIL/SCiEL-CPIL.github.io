"""Model-and-optimize: spectral phase retrieval + L-BFGS + coordinate descent + spectral amplitude polish + game-theoretic saddle point."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 145

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - 3)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective_only(x):
    """Compute objective value only (fast, no gradient)."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    return m / (s * s)

def _soft_max_and_grad(x, beta):
    """
    Compute soft-max surrogate for max|ac| and its gradient.
    """
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    m = np.max(abs_ac)
    exps = np.exp(beta * (abs_ac - m))
    sum_exps = np.sum(exps)
    soft_val = m + np.log(sum_exps) / beta
    
    weights = exps / sum_exps
    signs = np.sign(ac)
    signs[ac == 0] = 1.0
    grad_ac = weights * signs
    
    L_corr = 1
    while L_corr < 3 * N:
        L_corr *= 2
    
    g_padded = np.zeros(L_corr)
    g_padded[:2*N-1] = grad_ac
    
    X_f = np.fft.rfft(x, n=L_corr)
    G_f = np.fft.rfft(g_padded, n=L_corr)
    
    corr = np.fft.irfft(G_f * np.conj(X_f), n=L_corr)
    grad_x = 2.0 * corr[:N]
    
    return soft_val, grad_x, weights

def _objective_and_grad(x, beta):
    """Compute objective and gradient."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N)
    soft_val, grad_soft, _ = _soft_max_and_grad(x, beta)
    val = soft_val / (s * s)
    grad = grad_soft / (s * s) - 2.0 * soft_val * np.ones(N) / (s * s * s)
    return val, grad

def _spectral_init(rng, mode=0):
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    
    if mode == 0:
        freqs = np.fft.rfftfreq(L_fft)
        mag = np.ones(L_fft // 2 + 1)
        taper = np.minimum(1.0, np.abs(freqs) * L_fft / 5)
        mag *= taper
    elif mode == 1:
        freqs = np.fft.rfftfreq(L_fft)
        mag = 0.5 + 0.5 * np.cos(2 * np.pi * freqs * L_fft * 0.5)
    elif mode == 2:
        mag = 1.0 + 0.3 * rng.normal(0, 1, size=L_fft // 2 + 1)
        mag = np.maximum(0.1, mag)
    elif mode == 3:
        freqs = np.fft.rfftfreq(L_fft)
        mag = np.exp(-20 * (freqs * L_fft / 2) ** 2)
    elif mode == 4:
        freqs = np.fft.rfftfreq(L_fft)
        cutoff = 0.4
        transition = 0.05
        mag = np.where(np.abs(freqs) < cutoff - transition, 1.0,
                       np.where(np.abs(freqs) < cutoff + transition,
                                0.5 * (1 + np.cos(np.pi * (np.abs(freqs) - cutoff) / transition)),
                                0.1))
    else:
        mag = np.ones(L_fft // 2 + 1)
    
    phase = rng.uniform(0, 2 * np.pi, size=L_fft // 2 + 1)
    phase[0] = 0
    
    spectrum = mag * np.exp(1j * phase)
    spectrum[0] = mag[0]
    
    x_full = np.fft.irfft(spectrum, n=L_fft)
    x = x_full[:N]
    
    s = np.sum(x)
    if s < 0:
        x = -x
    
    return x

def _game_theoretic_optimize(x0, rng, max_iters=800, lr_x=0.02, lr_p=0.05):
    """
    Zero-sum game formulation with adaptive step size and better convergence.
    """
    best_x = x0.copy()
    best_val = _objective_only(x0)
    L = 2 * N - 1
    
    # Initialize p concentrated on peaks
    ac = _autoconv_fft(best_x)
    abs_ac = np.abs(ac)
    m_ac = np.max(abs_ac)
    p = np.exp(2.0 * (abs_ac - m_ac))
    p = p / np.sum(p)
    
    L_corr = 1
    while L_corr < 3 * N:
        L_corr *= 2
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        ac = _autoconv_fft(best_x)
        abs_ac = np.abs(ac)
        s = np.sum(best_x)
        s2 = s * s
        if abs(s2) < 1e-10:
            break
        
        # Update p (adversary): move toward maximizing expected |ac|
        m_ac = np.max(abs_ac)
        p_unnorm = p * np.exp(lr_p * (abs_ac - m_ac))
        p = p_unnorm / np.sum(p_unnorm)
        
        # Gradient via FFT
        signs = np.sign(ac)
        signs[ac == 0] = 1.0
        g_ac = p * signs
        
        g_padded = np.zeros(L_corr)
        g_padded[:L] = g_ac
        
        X_f = np.fft.rfft(best_x, n=L_corr)
        G_f = np.fft.rfft(g_padded, n=L_corr)
        
        corr = np.fft.irfft(G_f * np.conj(X_f), n=L_corr)
        grad_x = 2.0 * corr[:N] / s2
        grad_x -= 2.0 * np.sum(p * abs_ac) * np.ones(N) / (s2 * s)
        
        # Adaptive step size with backtracking
        step = lr_x
        improved = False
        for _ in range(5):
            x_trial = best_x - step * grad_x
            s_trial = np.sum(x_trial)
            if s_trial < 0:
                x_trial = -x_trial
            val_trial = _objective_only(x_trial)
            if val_trial < best_val:
                best_val = val_trial
                best_x = x_trial.copy()
                improved = True
                break
            step *= 0.5
        
        if not improved:
            lr_x = max(0.001, lr_x * 0.9)
        else:
            lr_x = min(0.05, lr_x * 1.05)
        
        if it % 100 == 99:
            lr_p = max(0.01, lr_p * 0.97)
    
    return best_x, best_val

def _peak_targeted_polish(x0, rng, max_iters=1500, step_size=0.01):
    """Vectorized peak-targeted polish: computes gradient from top peaks simultaneously."""
    best_x = x0.copy()
    best_val = _objective_only(x0)
    ac = _autoconv_fft(best_x)
    abs_ac = np.abs(ac)
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        if it % 5 == 0:
            ac = _autoconv_fft(best_x)
            abs_ac = np.abs(ac)
        
        top_k = 20
        sorted_idx = np.argsort(-abs_ac)[:top_k]
        
        # Build combined gradient from all top peaks at once
        signs = np.sign(ac)
        signs[ac == 0] = 1.0
        g_ac = np.zeros(2 * N - 1)
        for k in sorted_idx:
            g_ac[k] = signs[k]
        
        # Compute gradient via FFT correlation
        L_corr = 1
        while L_corr < 3 * N:
            L_corr *= 2
        g_padded = np.zeros(L_corr)
        g_padded[:2*N-1] = g_ac
        X_f = np.fft.rfft(best_x, n=L_corr)
        G_f = np.fft.rfft(g_padded, n=L_corr)
        corr = np.fft.irfft(G_f * np.conj(X_f), n=L_corr)
        grad_x = 2.0 * corr[:N]
        
        norm_grad = np.linalg.norm(grad_x)
        if norm_grad > 1e-10:
            direction = -grad_x / norm_grad
            
            for scale_mult in [1.0, 0.5, 1.5, 0.25]:
                scale = step_size * scale_mult
                x_trial = best_x + scale * direction
                val = _objective_only(x_trial)
                if val < best_val:
                    best_val = val
                    best_x = x_trial.copy()
                    break
            else:
                step_size = min(0.1, step_size * 1.04)
                continue
        
        step_size = max(0.0003, step_size * 0.94)
        
        if it % 300 == 299:
            x_perturbed = best_x + rng.normal(0, 0.05, size=N)
            val_pert = _objective_only(x_perturbed)
            if val_pert < best_val:
                best_val = val_pert
                best_x = x_perturbed.copy()
                step_size = 0.015
    
    return best_x, best_val

def _spectral_alt_optimize(x0, rng, max_iters=200, lr_phase=0.1, lr_amp=0.05):
    """
    Alternating spectral amplitude/phase optimization.
    Fixed flat-top magnitude target; optimize phase, then project amplitude.
    """
    best_x = x0.copy()
    best_val = _objective_only(x0)
    
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    
    # Target: flat-top window magnitude
    freqs = np.fft.rfftfreq(L_fft)
    target_mag = np.ones(L_fft // 2 + 1)
    taper = np.minimum(1.0, np.abs(freqs) * L_fft / 6)
    target_mag *= taper
    target_mag[0] = 1.0
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        # Get current spectrum
        X = np.fft.rfft(best_x, n=L_fft)
        mag = np.abs(X)
        phase = np.angle(X)
        
        # Phase gradient: minimize max|ac| via soft-max
        ac = _autoconv_fft(best_x)
        abs_ac = np.abs(ac)
        m_ac = np.max(abs_ac)
        beta = 200.0
        exps = np.exp(beta * (abs_ac - m_ac))
        weights = exps / np.sum(exps)
        signs = np.sign(ac)
        signs[ac == 0] = 1.0
        g_ac = weights * signs
        
        # Gradient of phase update
        L_corr = 1
        while L_corr < 3 * N:
            L_corr *= 2
        g_padded = np.zeros(L_corr)
        g_padded[:2*N-1] = g_ac
        X_f = np.fft.rfft(best_x, n=L_corr)
        G_f = np.fft.rfft(g_padded, n=L_corr)
        corr = np.fft.irfft(G_f * np.conj(X_f), n=L_corr)
        grad_x = 2.0 * corr[:N]
        
        # Update time domain
        x_new = best_x - lr_phase * grad_x
        s_new = np.sum(x_new)
        if s_new < 0:
            x_new = -x_new
        
        val_new = _objective_only(x_new)
        
        # Spectral projection: blend toward target amplitude
        if val_new < best_val:
            best_x = x_new.copy()
            best_val = val_new
            
            # Project amplitude toward target
            X_new = np.fft.rfft(best_x, n=L_fft)
            mag_new = np.abs(X_new)
            phase_new = np.angle(X_new)
            
            # Soft projection: blend magnitudes
            alpha = 0.3
            mag_proj = (1 - alpha) * mag_new + alpha * target_mag * (np.sum(mag_new) / np.sum(target_mag))
            
            X_proj = mag_proj * np.exp(1j * phase_new)
            x_proj = np.fft.irfft(X_proj, n=L_fft)[:N]
            
            s_proj = np.sum(x_proj)
            if s_proj < 0:
                x_proj = -x_proj
            
            val_proj = _objective_only(x_proj)
            if val_proj < best_val:
                best_val = val_proj
                best_x = x_proj.copy()
        
        # Adaptive learning rates
        if it % 30 == 29:
            lr_phase = max(0.01, lr_phase * 0.95)
    
    return best_x, best_val

def construct():
    best_x = np.ones(N)
    best_val = _objective_only(best_x)
    
    if _check_deadline():
        return best_x
    
    rng = np.random.default_rng(12345)
    
    t = np.linspace(0, 1, N)
    initial_points = [
        np.ones(N),
        1.0 + 0.3 * np.sin(2 * np.pi * t),
        1.0 + 0.5 * np.cos(2 * np.pi * t),
        1.0 + 0.3 * np.sin(4 * np.pi * t),
        1.0 + 0.5 * np.cos(4 * np.pi * t),
        np.exp(1.5 * t),
        np.exp(-1.5 * t),
        np.exp(0.5 * t),
        np.exp(-0.5 * t),
        1.0 + 0.5 * (t - 0.5),
        1.0 + 0.8 * (t - 0.5)**2,
        1.0 + 0.5 * np.sin(6 * np.pi * t) * np.cos(2 * np.pi * t),
        1.0 + 0.3 * (t - 0.5)**3,
        np.maximum(0.5, 1.0 - 2.0 * np.abs(t - 0.5)),
        np.maximum(0.5, 1.0 - 3.0 * np.abs(t - 0.5)),
        1.0 + 0.4 * np.sin(8 * np.pi * t),
        1.0 + 0.4 * np.cos(10 * np.pi * t),
        np.maximum(0.3, 1.0 + 0.6 * np.sin(2 * np.pi * 3 * t)),
        1.0 + 0.3 * np.sin(2 * np.pi * 5 * t + 0.5),
        1.0 + 0.4 * np.cos(2 * np.pi * 7 * t),
        1.0 + 0.2 * np.sin(2 * np.pi * 11 * t) + 0.2 * np.cos(2 * np.pi * 13 * t),
        np.maximum(0.2, 1.0 - 4.0 * np.abs(t - 0.5)),
        1.0 + 0.5 * np.sin(2 * np.pi * t) * np.sin(4 * np.pi * t),
        0.8 + 0.4 * np.cos(2 * np.pi * 2 * t + 1.0),
        np.exp(-3.0 * (t - 0.5)**2),
        1.0 + 0.3 * np.tanh(3.0 * (t - 0.5)),
    ]
    
    for mode in range(5):
        for _ in range(3):
            initial_points.append(_spectral_init(rng, mode))
    
    for _ in range(8):
        scale = rng.uniform(0.4, 1.6)
        initial_points.append(scale * rng.normal(1.0, 0.25, size=N))
    
    for _ in range(4):
        x_sparse = np.ones(N)
        n_sparse = rng.integers(10, 50)
        idx = rng.choice(N, size=n_sparse, replace=False)
        x_sparse[idx] = rng.uniform(0.3, 1.5, size=n_sparse)
        initial_points.append(x_sparse)
    
    # Phase 1: L-BFGS on best initial points
    betas = [50.0, 150.0, 400.0]
    
    # Sort initial points by objective and take top 15
    scored_points = [(x, _objective_only(x)) for x in initial_points]
    scored_points.sort(key=lambda t: t[1])
    top_points = [sp[0] for sp in scored_points[:15]]
    
    for x0 in top_points:
        if _check_deadline():
            break
        
        x_start = x0 + rng.normal(0, 0.01, size=N)
        
        for beta in betas:
            if _check_deadline():
                break
            
            try:
                def obj_and_grad_func(x, _beta=beta):
                    return _objective_and_grad(x, _beta)
                
                result = minimize(obj_and_grad_func, x_start, method='L-BFGS-B',
                                  jac=True, options={'maxiter': 250, 'gtol': 1e-10, 'ftol': 1e-13, 'maxls': 40})
                
                if result.fun < best_val:
                    best_val = result.fun
                    best_x = result.x.copy()
                    
            except Exception:
                pass
    
    if _check_deadline():
        return best_x
    
    # Phase 2: Spectral alternating optimization
    x_spec, val_spec = _spectral_alt_optimize(best_x, rng, max_iters=250, lr_phase=0.08)
    if val_spec < best_val:
        best_val = val_spec
        best_x = x_spec.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 3: Game-theoretic optimization
    x_game, val_game = _game_theoretic_optimize(best_x, rng, max_iters=800, lr_x=0.015, lr_p=0.04)
    if val_game < best_val:
        best_val = val_game
        best_x = x_game.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Vectorized peak-targeted polish
    x_peak, val_peak = _peak_targeted_polish(best_x, rng, max_iters=2000, step_size=0.012)
    if val_peak < best_val:
        best_val = val_peak
        best_x = x_peak.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 5: High-precision L-BFGS polish
    for beta in [500.0, 2000.0, 8000.0, 20000.0, 50000.0]:
        if _check_deadline():
            break
        try:
            def obj_and_grad_func(x, _beta=beta):
                return _objective_and_grad(x, _beta)
            
            result = minimize(obj_and_grad_func, best_x, method='L-BFGS-B',
                              jac=True, options={'maxiter': 200, 'gtol': 1e-12, 'ftol': 1e-15, 'maxls': 35})
            
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
                
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 6: Coordinate descent with structured perturbations
    step_size = 0.02
    no_improve = 0
    max_no_improve = 15000
    t_arr = np.arange(N)
    
    while no_improve < max_no_improve and not _check_deadline():
        r = rng.random()
        if r < 0.15:
            idx = rng.integers(0, N)
            delta = step_size * rng.choice([-1, 1])
            x_trial = best_x.copy()
            x_trial[idx] += delta
        elif r < 0.35:
            n_perturb = rng.integers(2, 12)
            indices = rng.choice(N, size=n_perturb, replace=False)
            perturbation = np.zeros(N)
            perturbation[indices] = rng.normal(0, step_size, size=n_perturb)
            x_trial = best_x + perturbation
        elif r < 0.55:
            freq = rng.integers(1, 10)
            phase = rng.uniform(0, 2 * np.pi)
            perturbation = step_size * np.sin(2 * np.pi * freq * t_arr / N + phase)
            x_trial = best_x + perturbation
        elif r < 0.75:
            freq1 = rng.integers(1, 6)
            freq2 = rng.integers(5, 18)
            phase1 = rng.uniform(0, 2 * np.pi)
            phase2 = rng.uniform(0, 2 * np.pi)
            perturbation = step_size * (0.7 * np.sin(2 * np.pi * freq1 * t_arr / N + phase1) +
                                        0.3 * np.sin(2 * np.pi * freq2 * t_arr / N + phase2))
            x_trial = best_x + perturbation
        elif r < 0.9:
            n_perturb = rng.integers(5, 30)
            indices = rng.choice(N, size=n_perturb, replace=False)
            perturbation = np.zeros(N)
            perturbation[indices] = rng.normal(0, step_size * 2.5, size=n_perturb)
            x_trial = best_x + perturbation
        else:
            ac = _autoconv_fft(best_x)
            abs_ac = np.abs(ac)
            top_lag = np.argmax(abs_ac)
            n_lo = max(0, top_lag - N + 1)
            n_hi = min(top_lag, N - 1)
            if n_hi >= n_lo:
                n_perturb = rng.integers(2, min(15, n_hi - n_lo + 1))
                indices = rng.choice(np.arange(n_lo, n_hi + 1), size=n_perturb, replace=False)
                perturbation = np.zeros(N)
                perturbation[indices] = rng.normal(0, step_size * 1.5, size=n_perturb)
                x_trial = best_x + perturbation
            else:
                x_trial = best_x + rng.normal(0, step_size, size=N)
        
        val = _objective_only(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            step_size = max(0.0003, step_size * 0.95)
            no_improve = 0
        else:
            no_improve += 1
            step_size = min(0.12, step_size * 1.03)
        
        if no_improve > 3000 and rng.random() < 0.03:
            x_restart = best_x + rng.normal(0, 0.08, size=N)
            val_restart = _objective_only(x_restart)
            if val_restart < best_val:
                best_val = val_restart
                best_x = x_restart.copy()
                no_improve = 0
                step_size = 0.02
    
    # Final polish
    if not _check_deadline():
        try:
            def final_obj_grad(x):
                return _objective_and_grad(x, 80000.0)
            
            result = minimize(final_obj_grad, best_x, method='L-BFGS-B',
                              jac=True, options={'maxiter': 200, 'gtol': 1e-13, 'ftol': 1e-16, 'maxls': 30})
            
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    return best_x
# EVOLVE-BLOCK-END
