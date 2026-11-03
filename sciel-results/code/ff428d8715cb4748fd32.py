"""Program for signed step heights minimizing max|autoconv|/(sum h)^2.

Combined approach:
- Compact basis parameterization + global optimization (DE, L-BFGS-B, basin hopping)
- O(N) fast incremental autoconvolution update for local search
- Smooth log-sum-exp surrogate with adaptive beta for gradient-based refinement
- Analytical gradient computation for the smooth surrogate
- Peak-suppression phase targeting the argmax of the autoconvolution
- Sign-flip local search with efficient incremental updates
- Block-flip local search with precomputed FFT basis for speed
- Direct gradient descent on smooth surrogate (no basis projection)
"""

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution, basinhopping

_FSIZE = 1
while _FSIZE < 2 * N - 1:
    _FSIZE *= 2

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    H = np.fft.rfft(h, _FSIZE)
    ac = np.fft.irfft(H * H, _FSIZE)
    return np.max(np.abs(ac[:2*N-1]))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _autoconv(h):
    """Compute full autoconvolution using FFT."""
    H = np.fft.rfft(h, _FSIZE)
    ac = np.fft.irfft(H * H, _FSIZE)
    return ac[:2*N-1]

def _soft_max_abs(ac, beta=50.0):
    """
    Soft approximation of max|ac| using log-sum-exp for differentiability.
    """
    abs_ac = np.abs(ac)
    max_val = np.max(abs_ac)
    if max_val < 1e-12:
        return 0.0
    shifted = abs_ac - max_val
    log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
    return max_val + log_sum

def _score_smooth(h, beta=50.0):
    """
    Smooth surrogate for the objective: soft_max|autoconv| / (sum h)^2.
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    
    H = np.fft.rfft(h, _FSIZE)
    ac = np.fft.irfft(H * H, _FSIZE)[:2*N-1]
    
    soft_max = _soft_max_abs(ac, beta)
    return soft_max / (s * s)

def _soft_max_abs_grad(h, beta=50.0):
    """
    Compute the soft max |autoconv| value and its gradient w.r.t. h.
    Returns (value, gradient) where value is S/s^2 and gradient is d(value)/d(h).
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10, np.zeros_like(h)
    
    ac = _autoconv(h)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    
    if m < 1e-15:
        S = 0.0
        grad_S = np.zeros_like(h)
    else:
        scaled = beta * (abs_ac - m)
        scaled = np.clip(scaled, -50, 50)
        exp_scaled = np.exp(scaled)
        sum_exp = np.sum(exp_scaled)
        log_sum = np.log(sum_exp)
        S = m + log_sum / beta
        
        # Weights for the soft-max
        w = exp_scaled / sum_exp
        
        # v_k = w_k * sign(ac_k)
        v = w * np.sign(ac)
        
        # Gradient of S w.r.t h:
        # dS/dh_i = 2 * sum_k v_k h_{k-i}
        # This is 2 * cross_correlation(v, h)
        
        fsize_corr = 1
        while fsize_corr < 3 * N:
            fsize_corr *= 2
            
        V = np.fft.rfft(v, fsize_corr)
        H = np.fft.rfft(h, fsize_corr)
        C = np.fft.irfft(V * np.conj(H), fsize_corr)
        
        grad_S = 2 * C[:N]
    
    # Objective f = S / s^2
    # df/dh_i = (dS/dh_i * s^2 - S * 2 * s) / s^4
    # = (dS/dh_i * s - 2 * S) / s^3
    
    grad = (grad_S * s - 2 * S) / (s ** 3)
    
    return S / (s * s), grad

def _make_basis_functions(n, n_basis=25):
    """
    Create a set of basis functions for parameterizing h.
    """
    x = np.arange(n, dtype=float)
    basis = []
    
    # Constant function
    basis.append(np.ones(n))
    
    # Low-frequency cosines
    for k in range(1, 8):
        basis.append(np.cos(2 * np.pi * k * x / n))
    
    # Low-frequency sines
    for k in range(1, 8):
        basis.append(np.sin(2 * np.pi * k * x / n))
    
    # Gaussian bumps at different positions
    for center_frac in [0.2, 0.4, 0.6, 0.8]:
        mu = center_frac * n
        sigma = n / 8
        basis.append(np.exp(-(x - mu)**2 / (2 * sigma**2)))
    
    # Cosine window
    basis.append(np.cos(np.pi * x / n))
    
    # Quadratic polynomial
    basis.append((x - n/2)**2 / (n**2))
    
    # Linear ramp
    basis.append(x / n - 0.5)
    
    # Higher frequency cosine
    for k in [10, 15, 20]:
        basis.append(np.cos(2 * np.pi * k * x / n))
    
    return np.array(basis[:n_basis])

def _try_flip(h, idx, ac, cur_sum):
    """
    Efficiently compute the new max|autoconv|/sum^2 after flipping h[idx] in O(N) time.
    """
    v = h[idx]
    new_sum = cur_sum - 2.0 * v
    nsq = new_sum * new_sum
    if nsq < 1e-10:
        return None, new_sum, np.inf
    
    lo = idx
    hi = idx + N
    if hi > 2 * N - 1:
        hi = 2 * N - 1
    
    affected = ac[lo:hi] - 4.0 * v * h[:hi - lo]
    two_idx = 2 * idx
    if lo <= two_idx < hi:
        affected[two_idx - lo] = ac[two_idx]
    
    max_affected = float(np.max(np.abs(affected)))
    
    if lo > 0:
        max_before = float(np.max(np.abs(ac[:lo])))
    else:
        max_before = 0.0
    if hi < 2 * N - 1:
        max_after = float(np.max(np.abs(ac[hi:])))
    else:
        max_after = 0.0
    max_unaffected = max(max_before, max_after)
    
    new_max = max(max_affected, max_unaffected)
    new_score = new_max / nsq
    
    return None, new_sum, new_score

def _local_search_flip(h, max_passes=200, deadline=None):
    """
    Local optimization: try flipping each element's sign, keep if improves score.
    Uses O(N) incremental update for efficiency.
    """
    h = h.copy()
    ac = _autoconv(h)
    cur_sum = float(np.sum(h))
    best_val = float(np.max(np.abs(ac)) / (cur_sum * cur_sum))
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        best_j = -1
        best_js = best_val
        best_new_sum = cur_sum
        
        for j in range(N):
            if deadline is not None and time.time() > deadline:
                break
            _, new_sum, new_score = _try_flip(h, j, ac, cur_sum)
            if new_score < best_js:
                best_js = new_score
                best_j = j
                best_new_sum = new_sum
        
        if best_j == -1 or best_js >= best_val - 1e-12:
            break
            
        v = h[best_j]
        h[best_j] = -v
        lo = best_j
        hi = min(best_j + N, 2 * N - 1)
        ac[lo:hi] -= 4.0 * v * h[:hi - lo]
        two_idx = 2 * best_j
        if lo <= two_idx < hi:
            ac[two_idx] += 4.0 * v * v
        cur_sum = best_new_sum
        best_val = best_js
    
    return h, best_val

def _local_search_block(h, deadline=None, block_sizes=None):
    """
    Block flip local search: flip contiguous blocks of signs.
    Uses FFT incrementally with precomputed basis for speed.
    """
    if block_sizes is None:
        block_sizes = [2, 3, 5, 7, 11, 15]
    
    h = h.copy()
    best_val = _score(h)
    H = np.fft.rfft(h, _FSIZE)
    cur_sum = float(np.sum(h))
    
    # Precompute frequency basis vectors
    freqs = np.arange(_FSIZE//2+1)
    basis_vecs = {}
    for i in range(N):
        basis_vecs[i] = np.exp(-2j * np.pi * freqs * i / _FSIZE)
    
    improved = True
    pass_count = 0
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        pass_count += 1
        
        for bs in block_sizes:
            if deadline is not None and time.time() > deadline:
                break
            step = max(1, (N - bs) // 50)
            for start in range(0, N - bs + 1, step):
                if deadline is not None and time.time() > deadline:
                    break
                
                # Compute delta for flipping the block
                delta = np.zeros(_FSIZE//2+1, dtype=complex)
                for j in range(bs):
                    idx = start + j
                    delta += -2.0 * h[idx] * basis_vecs[idx]
                
                H_new = H + delta
                conv = np.fft.irfft(H_new * H_new, _FSIZE)[:2*N-1]
                new_sum = cur_sum - 2.0 * np.sum(h[start:start+bs])
                if abs(new_sum) < 1e-12:
                    continue
                new_val = np.max(np.abs(conv)) / (new_sum * new_sum)
                
                if new_val < best_val - 1e-12:
                    # Apply the flip
                    h[start:start+bs] = -h[start:start+bs]
                    H = H_new
                    cur_sum = new_sum
                    best_val = new_val
                    improved = True
    
    return h, best_val

def _local_search_perturb(h, max_passes=8, deadline=None, step_size=0.03):
    """
    Local optimization: try small perturbations to each element, keep if improves score.
    """
    h = h.copy()
    best_val = _score(h)
    rng = np.random.default_rng(777)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        idx_order = rng.permutation(N)
        for i in idx_order:
            if deadline is not None and time.time() > deadline:
                break
            old_val = h[i]
            best_local = old_val
            best_local_val = best_val
            for delta in [step_size, -step_size, 2*step_size, -2*step_size, 
                         0.5*step_size, -0.5*step_size]:
                h[i] = old_val + delta
                new_val = _score(h)
                if new_val < best_local_val:
                    best_local_val = new_val
                    best_local = h[i]
            if best_local != old_val:
                h[i] = best_local
                best_val = best_local_val
                improved = True
            else:
                h[i] = old_val
        if not improved:
            break
    return h, best_val

def _gaussian_perturbation(h, scale=0.1, deadline=None, max_iters=20000):
    """
    Simulated annealing style search with Gaussian perturbations.
    """
    h_curr = h.copy()
    h_best = h.copy()
    curr_val = _score(h_curr)
    best_val = curr_val
    rng = np.random.default_rng(999)
    
    T_start = 0.5
    T_end = 0.01
    n_iter = max_iters
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        T = T_start * (T_end / T_start) ** (it / max(1, n_iter - 1))
        
        perturb = rng.standard_normal(N) * scale
        h_trial = h_curr + perturb
        trial_val = _score(h_trial)
        
        if trial_val < curr_val or rng.random() < np.exp(-(trial_val - curr_val) / max(T, 1e-10)):
            h_curr = h_trial
            curr_val = trial_val
            if curr_val < best_val:
                best_val = curr_val
                h_best = h_curr.copy()
        
        scale *= 0.995
    
    return h_best, best_val

def _peak_suppression(h, basis, max_iters=50, deadline=None):
    """
    Peak suppression phase: directly target the argmax of the autoconvolution.
    """
    h = h.copy()
    s = np.sum(h)
    if abs(s) < 1e-6:
        return h, _score(h)
    h = h / s
    
    for _ in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        ac = _autoconv(h)
        abs_ac = np.abs(ac)
        peak_idx = np.argmax(abs_ac)
        peak_val = abs_ac[peak_idx]
        
        params, _, _, _ = np.linalg.lstsq(basis.T, h, rcond=None)
        
        best_params = params.copy()
        best_val = _score(h)
        rng = np.random.default_rng(12345)
        
        for _ in range(20):
            if deadline is not None and time.time() > deadline:
                break
            pert = rng.standard_normal(len(params)) * 0.1
            new_params = params + pert
            new_h = np.dot(new_params, basis)
            new_s = np.sum(new_h)
            if abs(new_s) < 1e-6:
                continue
            new_h = new_h / new_s
            new_val = _score(new_h)
            if new_val < best_val:
                best_val = new_val
                best_params = new_params
        
        h = np.dot(best_params, basis)
        s = np.sum(h)
        if abs(s) < 1e-6:
            break
        h = h / s
    
    return h, _score(h)

def _gradient_descent_smooth_direct(h, beta_schedule=None, max_iters=150, deadline=None):
    """
    Direct gradient descent on the smooth surrogate in full N-dimensional space.
    Uses analytical gradients without basis projection.
    """
    if beta_schedule is None:
        beta_schedule = [30.0, 50.0, 80.0, 120.0, 200.0]
    
    h_curr = h.copy()
    s_curr = np.sum(h_curr)
    if abs(s_curr) < 1e-6:
        return h, _score(h)
    h_curr = h_curr / s_curr
    
    best_val = _score(h_curr)
    best_h = h_curr.copy()
    
    for beta in beta_schedule:
        lr = 0.02
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline:
                break
            
            val, grad = _soft_max_abs_grad(h_curr, beta)
            
            # Gradient step
            h_new = h_curr - lr * grad
            
            # Normalize
            s_new = np.sum(h_new)
            if abs(s_new) < 1e-6:
                break
            h_new = h_new / s_new
            
            new_val = _score(h_new)
            if new_val < best_val:
                best_val = new_val
                best_h = h_new.copy()
            
            # Adaptive learning rate
            if new_val > val + 1e-6:
                lr *= 0.7
            else:
                lr = min(lr * 1.05, 0.05)
            
            h_curr = h_new
    
    return best_h, best_val

def _gradient_descent_smooth(h, basis, beta_schedule=None, max_iters=100, deadline=None):
    """
    Gradient descent on the smooth surrogate using analytical gradients.
    Projects back to basis space and normalizes.
    """
    if beta_schedule is None:
        beta_schedule = [30.0, 50.0, 80.0, 120.0]
    
    h_curr = h.copy()
    s_curr = np.sum(h_curr)
    if abs(s_curr) < 1e-6:
        return h, _score(h)
    h_curr = h_curr / s_curr
    
    best_val = _score(h_curr)
    best_h = h_curr.copy()
    
    for beta in beta_schedule:
        lr = 0.05
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline:
                break
            
            val, grad = _soft_max_abs_grad(h_curr, beta)
            
            # Project gradient to basis space
            grad_basis = basis @ grad
            
            # Gradient step in basis space
            params, _, _, _ = np.linalg.lstsq(basis.T, h_curr, rcond=None)
            params -= lr * grad_basis
            h_new = basis @ params
            
            # Normalize
            s_new = np.sum(h_new)
            if abs(s_new) < 1e-6:
                break
            h_new = h_new / s_new
            
            new_val = _score(h_new)
            if new_val < best_val:
                best_val = new_val
                best_h = h_new.copy()
            
            h_curr = h_new
    
    return best_h, best_val

def construct():
    t_start = time.time()
    deadline = t_start + 140.0

    best_score = np.inf
    best_h = np.ones(N) / N
    
    base_score = _score(best_h)
    if base_score < best_score:
        best_score = base_score
        best_h = best_h.copy()
    
    n_basis = 25
    basis = _make_basis_functions(N, n_basis)
    
    # Phase 1: Global optimization with differential evolution
    if time.time() < deadline - 45:
        bounds = [(-2.0, 2.0)] * n_basis
        
        def de_objective(params):
            if time.time() > deadline - 40:
                return 1e10
            h = np.dot(params, basis)
            s = np.sum(h)
            if abs(s) < 1e-9:
                return 1e10
            return _autoconv_max_abs(h) / (s * s)
        
        try:
            res_de = differential_evolution(
                de_objective, bounds, 
                seed=42, maxiter=35, tol=1e-5,
                popsize=12, mutation=(0.3, 1.2), recombination=0.7,
                polish=False
            )
            if res_de.fun < best_score:
                h_de = np.dot(res_de.x, basis)
                s_de = np.sum(h_de)
                if abs(s_de) > 1e-6:
                    h_de = h_de / s_de
                    val_de = _score(h_de)
                    if val_de < best_score:
                        best_score = val_de
                        best_h = h_de.copy()
        except Exception:
            pass
    
    # Phase 2: L-BFGS-B refinement with smooth surrogate and adaptive beta
    if time.time() < deadline - 28:
        inits = []
        
        h_curr = best_h.copy()
        s_curr = np.sum(h_curr)
        if abs(s_curr) > 1e-6:
            h_curr = h_curr / s_curr
        
        if len(basis.shape) == 2:
            params_init, _, _, _ = np.linalg.lstsq(basis.T, h_curr, rcond=None)
            inits.append(params_init)
        
        rng = np.random.default_rng(123)
        for i in range(12):
            params_rand = rng.standard_normal(n_basis) * 0.5
            inits.append(params_rand)
        
        for amp in [0.1, 0.3, 0.5, 0.7, 1.0]:
            params_struct = np.zeros(n_basis)
            params_struct[0] = 1.0
            params_struct[1] = amp * 0.1
            inits.append(params_struct)
        
        beta_schedule = [60.0, 120.0, 250.0]
        
        for params_init in inits[:10]:
            if time.time() > deadline - 22:
                break
            
            try:
                current_params = params_init.copy()
                
                for beta in beta_schedule:
                    if time.time() > deadline - 20:
                        break
                    
                    def smooth_obj(params, b=beta):
                        if time.time() > deadline - 18:
                            return 1e10
                        h = np.dot(params, basis)
                        s = np.sum(h)
                        if abs(s) < 1e-9:
                            return 1e10
                        return _score_smooth(h, b)
                    
                    res_smooth = minimize(
                        smooth_obj, current_params, method='BFGS',
                        options={'maxiter': 150, 'disp': False, 'gtol': 1e-7}
                    )
                    current_params = res_smooth.x
                
                def lbfgs_obj(params):
                    if time.time() > deadline - 16:
                        return 1e10
                    h = np.dot(params, basis)
                    s = np.sum(h)
                    if abs(s) < 1e-9:
                        return 1e10
                    return _autoconv_max_abs(h) / (s * s)
                
                res_lbfgs = minimize(
                    lbfgs_obj, current_params, method='L-BFGS-B',
                    options={'maxiter': 250, 'disp': False, 'gtol': 1e-8}
                )
                h_opt = np.dot(res_lbfgs.x, basis)
                s_opt = np.sum(h_opt)
                if abs(s_opt) > 1e-6:
                    h_opt = h_opt / s_opt
                    val_opt = _score(h_opt)
                    if val_opt < best_score:
                        best_score = val_opt
                        best_h = h_opt.copy()
            except Exception:
                pass
    
    # Phase 3: Direct gradient descent with analytical gradients (full space)
    if time.time() < deadline - 22:
        try:
            h_gd, val_gd = _gradient_descent_smooth_direct(
                best_h, 
                beta_schedule=[30.0, 50.0, 80.0, 120.0, 200.0],
                max_iters=100,
                deadline=deadline - 20
            )
            if val_gd < best_score:
                best_score = val_gd
                best_h = h_gd.copy()
        except Exception:
            pass
    
    # Phase 4: Basis-projected gradient descent
    if time.time() < deadline - 18:
        try:
            h_gd2, val_gd2 = _gradient_descent_smooth(
                best_h, basis, 
                beta_schedule=[30.0, 50.0, 80.0, 120.0],
                max_iters=80,
                deadline=deadline - 16
            )
            if val_gd2 < best_score:
                best_score = val_gd2
                best_h = h_gd2.copy()
        except Exception:
            pass
    
    # Phase 5: Peak suppression
    if time.time() < deadline - 14:
        try:
            h_ps, val_ps = _peak_suppression(best_h, basis, max_iters=25, deadline=deadline-12)
            if val_ps < best_score:
                best_score = val_ps
                best_h = h_ps.copy()
        except Exception:
            pass
    
    # Phase 6: Basin hopping
    if time.time() < deadline - 12:
        try:
            h_curr = best_h.copy()
            s_curr = np.sum(h_curr)
            if abs(s_curr) > 1e-6:
                h_curr = h_curr / s_curr
            params_curr, _, _, _ = np.linalg.lstsq(basis.T, h_curr, rcond=None)
            
            def basin_obj(params):
                if time.time() > deadline - 8:
                    return 1e10
                h = np.dot(params, basis)
                s = np.sum(h)
                if abs(s) < 1e-9:
                    return 1e10
                return _autoconv_max_abs(h) / (s * s)
            
            res_bh = basinhopping(
                basin_obj, params_curr, 
                niter=15, T=0.1, stepsize=0.3,
                minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 100, 'gtol': 1e-7}},
                seed=42
            )
            if res_bh.fun < best_score:
                h_bh = np.dot(res_bh.x, basis)
                s_bh = np.sum(h_bh)
                if abs(s_bh) > 1e-6:
                    h_bh = h_bh / s_bh
                    val_bh = _score(h_bh)
                    if val_bh < best_score:
                        best_score = val_bh
                        best_h = h_bh.copy()
        except Exception:
            pass
    
    # Phase 7: Local search refinement - sign flips
    if time.time() < deadline - 8:
        h_flip, val_flip = _local_search_flip(best_h, max_passes=1200, deadline=deadline-6)
        if val_flip < best_score:
            best_score = val_flip
            best_h = h_flip.copy()
    
    # Phase 8: Block flip local search
    if time.time() < deadline - 6:
        try:
            h_block, val_block = _local_search_block(best_h, deadline=deadline-4)
            if val_block < best_score:
                best_score = val_block
                best_h = h_block.copy()
        except Exception:
            pass
    
    # Phase 9: Perturbation local search
    if time.time() < deadline - 4:
        h_pert, val_pert = _local_search_perturb(best_h, max_passes=6, deadline=deadline-3, step_size=0.03)
        if val_pert < best_score:
            best_score = val_pert
            best_h = h_pert.copy()
    
    # Phase 10: Simulated annealing
    if time.time() < deadline - 3:
        h_sa, val_sa = _gaussian_perturbation(best_h, scale=0.06, deadline=deadline-1.5, max_iters=15000)
        if val_sa < best_score:
            best_score = val_sa
            best_h = h_sa.copy()
    
    # Final high-precision polish
    if time.time() < deadline - 1.5:
        try:
            h_curr = best_h.copy()
            s_curr = np.sum(h_curr)
            if abs(s_curr) > 1e-6:
                h_curr = h_curr / s_curr
            params_curr, _, _, _ = np.linalg.lstsq(basis.T, h_curr, rcond=None)
            
            def final_obj(params):
                if time.time() > deadline - 1:
                    return 1e10
                h = np.dot(params, basis)
                s = np.sum(h)
                if abs(s) < 1e-9:
                    return 1e10
                return _autoconv_max_abs(h) / (s * s)
            
            res_final = minimize(
                final_obj, params_curr, method='L-BFGS-B',
                options={'maxiter': 500, 'disp': False, 'gtol': 1e-12}
            )
            h_final = np.dot(res_final.x, basis)
            s_final = np.sum(h_final)
            if abs(s_final) > 1e-6:
                h_final = h_final / s_final
                val_final = _score(h_final)
                if val_final < best_score:
                    best_score = val_final
                    best_h = h_final.copy()
        except Exception:
            pass
    
    # Second polish pass
    if time.time() < deadline - 0.5:
        try:
            h_curr = best_h.copy()
            s_curr = np.sum(h_curr)
            if abs(s_curr) > 1e-6:
                h_curr = h_curr / s_curr
            params_curr, _, _, _ = np.linalg.lstsq(basis.T, h_curr, rcond=None)
            
            def final_obj2(params):
                if time.time() > deadline - 0.3:
                    return 1e10
                h = np.dot(params, basis)
                s = np.sum(h)
                if abs(s) < 1e-9:
                    return 1e10
                return _autoconv_max_abs(h) / (s * s)
            
            res_final2 = minimize(
                final_obj2, params_curr, method='L-BFGS-B',
                options={'maxiter': 200, 'disp': False, 'gtol': 1e-14}
            )
            h_final2 = np.dot(res_final2.x, basis)
            s_final2 = np.sum(h_final2)
            if abs(s_final2) > 1e-6:
                h_final2 = h_final2 / s_final2
                val_final2 = _score(h_final2)
                if val_final2 < best_score:
                    best_score = val_final2
                    best_h = h_final2.copy()
        except Exception:
            pass
    
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N) / N
    
    return best_h

# EVOLVE-BLOCK-END
