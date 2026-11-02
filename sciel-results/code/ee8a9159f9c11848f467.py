"""Riemannian optimization on the unit sphere: minimize max|ac|/S^2 via sphere-constrained L-BFGS."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START
START_TIME = time.time()
TIME_LIMIT = 135  # safety margin before 150s kill

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - 5)

def _fft_len(n):
    L = 1
    while L < n:
        L *= 2
    return L

def _autoconv_fft(x):
    L = 2 * N - 1
    L_fft = _fft_len(L)
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective_only(x, beta=1e6):
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if beta < 1e6:
        exps = np.exp(beta * (abs_ac - m))
        log_sum = np.log(np.sum(exps))
        soft_val = m + log_sum / beta
    else:
        soft_val = m
    return soft_val / (s * s)

def _sphere_objective_and_grad(x, beta=50.0):
    """Objective and gradient on the unit sphere (x already normalized)."""
    # Ensure unit norm
    nrm = np.linalg.norm(x)
    if nrm < 1e-10:
        return 1e10, np.zeros(N)
    x = x / nrm
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N)
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    exps = np.exp(beta * (abs_ac - m))
    log_sum = np.log(np.sum(exps))
    soft_val = m + log_sum / beta
    val = soft_val / (s * s)
    
    # Gradient of soft_val / s^2 with respect to x
    # soft_val = sum_k w_k |ac_k| where w_k = exp(beta(|ac_k|-m))/sum
    # d(soft_val)/dx = sum_k w_k * sign(ac_k) * d(ac_k)/dx
    # d(ac_k)/dx_j = x_{j+k} + x_{j-k} (with bounds)
    weights = exps / np.sum(exps)
    signs = np.sign(ac)
    signs[ac == 0] = 1.0
    g_ac = weights * signs  # length 2N-1
    
    # grad of soft_val: for each j, sum over k of g_ac[k] * (x_{j+k} + x_{j-k})
    # ac[k] = sum_i x[i] x[i+k] for k>=0, ac[k] = sum_i x[i] x[i-k] for k<0
    # d(ac[k])/dx[j] = x[j+k] (if 0<=j+k<N) + x[j-k] (if 0<=j-k<N) for k>=0
    # Actually ac[k] = sum_{i} x[i]*x[i+k], so d/dx[j] = x[j+k] + x[j-k] (with x out of range = 0)
    # Let's compute via correlation: grad_soft[j] = sum_k g_ac[k] * (x[j+k] + x[j-k])
    # This is a convolution of g_ac with x
    L_corr = _fft_len(2 * N - 1)
    X_f = np.fft.rfft(x, n=L_corr)
    G_f = np.fft.rfft(g_ac, n=L_corr)
    corr = np.fft.irfft(X_f * G_f, n=L_corr)
    # corr[i] = sum_j x[j] * g_ac[i-j] ... need to be careful
    # Let's just do it directly with the known formula
    # d(ac[k])/dx[j] for k in [0, 2N-2]:
    #   ac[k] = sum_{i=max(0,k-N+1)}^{min(N-1, N-1+k)} x[i]*x[i+k] (for k>=0)
    #   Actually simpler: ac[k] = <x, shift_k(x)> where shift_k shifts by k
    #   d/dx[j] ac[k] = x[j+k] (if valid) + x[j-k] (if valid, for the symmetric term)
    # For k >= 0: ac[k] = sum_i x[i] x[i+k], d/dx[j] = x[j+k] + (x[j-k] if k>0 else 0)
    # For k < 0 (i.e., ac[2N-1+k] for k<0): ac[-k] = sum_i x[i] x[i-k], d/dx[j] = x[j-k] + x[j+k]
    # So in all cases: d/dx[j] ac[|k|] = x[j+|k|] + x[j-|k|] (with out-of-range = 0)
    # Wait, ac is indexed 0 to 2N-2. ac[k] for k=0..N-1 is the "positive lag" and 
    # ac[N-1+m] for m=1..N-1 corresponds to lag -(N-1-m)? No.
    # Standard linear convolution: ac[k] = sum_i x[i] x[k-i] for k=0..2N-2
    # So ac[k] = sum_i x[i] x[k-i]. d/dx[j] = x[k-j] + x[j-k] (both terms from the two x's)
    # = x[k-j] (if 0<=k-j<N) + x[j-k] (if 0<=j-k<N)
    
    # Let me compute grad_soft via FFT cross-correlation properly
    # grad_soft[j] = sum_{k=0}^{2N-2} g_ac[k] * (x[k-j] * [0<=k-j<N] + x[j-k] * [0<=j-k<N])
    # = sum_k g_ac[k] x[k-j] (k-j in range) + sum_k g_ac[k] x[j-k] (j-k in range)
    # First term: sum_k g_ac[k] x[k-j] = (g_ac * x)[j] where * is cross-correlation
    # Second term: sum_k g_ac[k] x[j-k] = (g_ac * x)[j] as well (same thing by symmetry of the sum)
    # Actually both terms give the same sum, so grad_soft = 2 * (g_ac cross-correlate x)
    # Let me verify: cross-correlation c[j] = sum_k g_ac[k] x[k-j]
    # Using FFT: c = ifft(fft(g_ac) * conj(fft(x)))
    L_c = _fft_len(2 * N - 1)
    G_f2 = np.fft.rfft(g_ac, n=L_c)
    X_f2 = np.fft.rfft(x, n=L_c)
    c = np.fft.irfft(G_f2 * np.conj(X_f2), n=L_c)
    grad_soft = 2.0 * c[:N]
    
    # grad of val = soft_val/s^2:
    # d(val)/dx = grad_soft/s^2 - 2*soft_val * ones/s^3
    grad = grad_soft / (s * s) - 2.0 * soft_val * np.ones(N) / (s * s * s)
    
    # Project gradient onto tangent space of sphere: grad - <grad, x> x
    proj = grad - np.dot(grad, x) * x
    return val, proj

def _riemannian_lbgfs(x0, beta=50.0, max_iter=200):
    """Riemannian L-BFGS on the unit sphere."""
    x = x0 / np.linalg.norm(x0)
    
    def func_and_grad(x_flat):
        return _sphere_objective_and_grad(x_flat, beta)
    
    result = minimize(func_and_grad, x, method='L-BFGS-B', jac=True,
                      options={'maxiter': max_iter, 'gtol': 1e-8, 'ftol': 1e-14, 'maxls': 30})
    
    x_out = result.x
    nrm = np.linalg.norm(x_out)
    if nrm > 1e-10:
        x_out = x_out / nrm
    return x_out, result.fun

def _peak_targeting(x, n_rounds=50, beta=100.0):
    """Directly target the dominant autocorrelation lags."""
    x = x.copy()
    nrm = np.linalg.norm(x)
    if nrm < 1e-10:
        return x, _objective_only(x)
    x = x / nrm
    best_val = _objective_only(x)
    rng = np.random.default_rng(77)
    
    for r in range(n_rounds):
        if _check_deadline():
            break
        ac = _autoconv_fft(x)
        abs_ac = np.abs(ac)
        s = np.sum(x)
        if abs(s) < 1e-10:
            break
        cur_val = np.max(abs_ac) / (s * s)
        
        # Find top lag
        peak_idx = np.argmax(abs_ac)
        
        improved = False
        for attempt in range(30):
            if _check_deadline():
                break
            # Random perturbation
            n_pert = rng.integers(5, 20)
            idx = rng.choice(N, size=n_pert, replace=False)
            deltas = rng.normal(0, 0.03, size=n_pert)
            x_trial = x.copy()
            x_trial[idx] += deltas
            nrm_t = np.linalg.norm(x_trial)
            if nrm_t < 1e-10:
                continue
            x_trial = x_trial / nrm_t
            val = _objective_only(x_trial)
            if val < best_val * (1 - 1e-7):
                best_val = val
                x = x_trial
                improved = True
                break
        
        if not improved:
            # Larger perturbation
            for attempt in range(15):
                if _check_deadline():
                    break
                n_pert = rng.integers(10, 40)
                idx = rng.choice(N, size=n_pert, replace=False)
                deltas = rng.normal(0, 0.1, size=n_pert)
                x_trial = x.copy()
                x_trial[idx] += deltas
                nrm_t = np.linalg.norm(x_trial)
                if nrm_t < 1e-10:
                    continue
                x_trial = x_trial / nrm_t
                val = _objective_only(x_trial)
                if val < best_val * (1 - 1e-7):
                    best_val = val
                    x = x_trial
                    improved = True
                    break
        if not improved:
            break
    
    return x, best_val

def _coordinate_descent_sphere(x, max_rounds=100):
    """Coordinate descent on the sphere (renormalize after each step)."""
    x = x.copy()
    nrm = np.linalg.norm(x)
    if nrm < 1e-10:
        return x, _objective_only(x)
    x = x / nrm
    best_val = _objective_only(x)
    rng = np.random.default_rng(55)
    
    for r in range(max_rounds):
        if _check_deadline():
            break
        improved = False
        for ss in [0.02, 0.01, 0.005]:
            if _check_deadline():
                break
            for j in range(N):
                if _check_deadline():
                    break
                for sign in [1, -1]:
                    x_trial = x.copy()
                    x_trial[j] += sign * ss
                    nrm_t = np.linalg.norm(x_trial)
                    if nrm_t < 1e-10:
                        continue
                    x_trial = x_trial / nrm_t
                    val = _objective_only(x_trial)
                    if val < best_val * (1 - 1e-8):
                        best_val = val
                        x = x_trial
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return x, best_val

def _make_initializations(rng, n_structured=12, n_random=15):
    """Generate diverse initializations on the unit sphere."""
    t = np.linspace(0, 1, N)
    inits = []
    
    structured = [
        np.ones(N),
        1.0 + 0.5 * np.sin(2 * np.pi * t),
        1.0 + 0.5 * np.cos(2 * np.pi * t),
        1.0 + 0.5 * np.sin(4 * np.pi * t),
        1.0 + 0.5 * np.cos(4 * np.pi * t),
        1.0 + 0.3 * np.sin(8 * np.pi * t),
        np.exp(1.5 * t),
        np.exp(-1.5 * t),
        1.0 + 0.8 * (t - 0.5),
        np.maximum(0.3, 1.0 - 2.0 * np.abs(t - 0.5)),
        1.0 + 0.5 * np.sin(2 * np.pi * 3 * t),
        1.0 + 0.5 * np.cos(2 * np.pi * 5 * t),
    ]
    for s in structured[:n_structured]:
        nrm = np.linalg.norm(s)
        if nrm > 1e-10:
            inits.append(s / nrm)
    
    for _ in range(n_random):
        # Random with varying sparsity and structure
        scale = rng.uniform(0.5, 2.0)
        x_rand = scale * rng.normal(1.0, 0.3, size=N)
        # Occasionally make it sparse
        if rng.random() < 0.3:
            mask = rng.random(N) < 0.5
            x_rand = np.where(mask, x_rand, 0.0)
        nrm = np.linalg.norm(x_rand)
        if nrm > 1e-10:
            inits.append(x_rand / nrm)
    
    return inits

def construct():
    best_x = np.ones(N)
    best_x = best_x / np.linalg.norm(best_x)
    best_val = _objective_only(best_x)
    
    if _check_deadline():
        return best_x
    
    rng = np.random.default_rng(12345)
    inits = _make_initializations(rng)
    
    # Phase 1: Riemannian L-BFGS from each initialization
    for x0 in inits:
        if _check_deadline():
            break
        x_opt, val_opt = _riemannian_lbgfs(x0, beta=50.0, max_iter=150)
        if val_opt < best_val:
            best_val = val_opt
            best_x = x_opt.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 2: Peak targeting
    best_x, best_val = _peak_targeting(best_x, n_rounds=60, beta=100.0)
    
    if _check_deadline():
        return best_x
    
    # Phase 3: Coordinate descent on sphere
    best_x, best_val = _coordinate_descent_sphere(best_x, max_rounds=80)
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Another round of Riemannian L-BFGS from the polished point with higher beta
    x_opt, val_opt = _riemannian_lbgfs(best_x, beta=100.0, max_iter=150)
    if val_opt < best_val:
        best_val = val_opt
        best_x = x_opt.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 5: Final peak targeting
    best_x, best_val = _peak_targeting(best_x, n_rounds=30, beta=150.0)
    
    if _check_deadline():
        return best_x
    
    # Phase 6: Final high-precision polish with very high beta
    x_opt, val_opt = _riemannian_lbgfs(best_x, beta=200.0, max_iter=100)
    if val_opt < best_val:
        best_val = val_opt
        best_x = x_opt.copy()
    
    return best_x
# EVOLVE-BLOCK-END
