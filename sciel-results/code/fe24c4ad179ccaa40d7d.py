"""Improved program: spectral subspace + epigraph polish for C3'."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _get_fft_size(n):
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    return fsize

def _autoconv_fft(h):
    n = len(h)
    fsize = _get_fft_size(n)
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:2*n-1]

def _raw_value(h):
    """Exact score: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    max_ac = np.max(np.abs(ac))
    return max_ac / (s * s)

def _soft_max_abs(ac, beta=100.0):
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -50, 50)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return m + log_sum / beta

def _smooth_objective(h, beta=100.0):
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    sm = _soft_max_abs(ac, beta)
    return sm / (s * s)

def _smooth_gradient(h, beta=100.0):
    """
    Analytical gradient of the soft-max surrogate objective.
    
    Objective: f(h) = softmax_abs(ac(h)) / (sum h)^2
    where ac = h * h (autoconvolution)
    
    Using the identity:
    softmax_abs(ac) = m + (1/beta) * log(sum(exp(beta*(|ac_i| - m))))
    
    d(softmax_abs)/d(ac_j) = exp(beta*(|ac_j| - m)) / sum(exp(beta*(|ac_k| - m))) * sign(ac_j)
    
    d(ac_j)/d(h_i) = h_{j-i} + h_{i-j}  (for valid indices, else 0)
    
    Then use chain rule with the 1/s^2 factor.
    """
    n = len(h)
    s = np.sum(h)
    if abs(s) < 1e-9:
        return np.zeros(n)
    
    ac = _autoconv_fft(h)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    
    if m < 1e-15:
        return np.zeros(n)
    
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -50, 50)
    exp_scaled = np.exp(scaled)
    sum_exp = np.sum(exp_scaled)
    
    if sum_exp < 1e-30:
        return np.zeros(n)
    
    # d(softmax_abs)/d(ac_j)
    weights = exp_scaled / sum_exp
    signs = np.sign(ac)
    dac_dac = weights * signs
    
    # Now compute gradient w.r.t. h
    # ac[k] = sum_j h[j] * h[k-j]
    # d(ac[k])/d(h[i]) = h[k-i] + h[i-k]  (with appropriate bounds)
    
    # For efficiency, use FFT-based approach:
    # d(ac)/d(h) is like a convolution structure
    # grad_h[i] = sum_k dac_dac[k] * (h[k-i] + h[i-k])
    
    # This is a double sum. Let's use FFT to accelerate.
    # Note: d(ac[k])/d(h[i]) = h[(k-i) mod n] + h[(i-k) mod n] for circular,
    # but we have linear autoconvolution.
    
    # For linear autoconvolution of length n:
    # ac[k] = sum_{j=0}^{n-1} h[j] * h[k-j]  where h[m] = 0 for m outside [0, n-1]
    # d(ac[k])/d(h[i]) = h[k-i] + h[i-k]
    
    # So grad_h[i] = sum_{k=0}^{2n-2} dac_dac[k] * (h[k-i] + h[i-k])
    
    # Let's define g[i] = sum_k dac_dac[k] * h[k-i]
    # This is a convolution: g = dac_dac * h (with appropriate indexing)
    
    # Actually, let's use a direct approach with FFT for the key part.
    
    # grad_h[i] = sum_k dac_dac[k] * h[k-i] + sum_k dac_dac[k] * h[i-k]
    # The second term is the same as the first with k -> -k, so by symmetry of dac_dac,
    # grad_h[i] = 2 * sum_k dac_dac[k] * h[k-i]  (approximately, if dac_dac is symmetric)
    
    # Let's compute this via FFT.
    fsize = _get_fft_size(2*n)
    H = np.fft.rfft(h, fsize)
    D = np.fft.rfft(dac_dac, fsize)
    conv = np.fft.irfft(H * D, fsize)
    
    # conv[m] = sum_k dac_dac[k] * h[m-k]
    # We want sum_k dac_dac[k] * h[k-i] = conv[i] with appropriate indexing
    # Actually, conv[i] = sum_k dac_dac[k] * h[i-k]
    # We want sum_k dac_dac[k] * h[k-i]
    
    # Let's just compute directly for safety, but use a smarter approach.
    # For N=400, direct O(n^2) might be acceptable if done efficiently.
    
    # Let's use the fact that:
    # sum_k dac_dac[k] * h[k-i] for k in [0, 2n-2], i in [0, n-1]
    # h[k-i] is nonzero when 0 <= k-i < n, i.e., i <= k < i+n
    
    # This is getting complex. Let's use a simpler numerical approach for the gradient
    # since the analytical gradient is complex, but we can use finite differences efficiently.
    
    # Actually, let's try a different approach: use the fact that the gradient of
    # sum_k w[k] * ac[k] w.r.t. h is:
    # grad[i] = sum_k w[k] * (h[k-i] + h[i-k])
    
    # Let's compute this via FFT.
    # Define w = dac_dac, length 2n-1
    # grad[i] = sum_k w[k] * h[k-i] + sum_k w[k] * h[i-k]
    
    # First term: sum_k w[k] * h[k-i]
    # Let j = k-i, so k = j+i
    # sum_j w[j+i] * h[j] for j in [0, n-1], k=j+i in [0, 2n-2]
    # This is a cross-correlation-like operation.
    
    # For simplicity and correctness, let's use a direct computation with NumPy's
    # ability to handle this efficiently.
    
    # grad[i] = sum_{k} w[k] * (h[k-i] if 0<=k-i<n else 0) + (h[i-k] if 0<=i-k<n else 0)
    
    # Let's build a matrix or use broadcasting.
    
    # For N=400, 2n-1 = 799. n=400. So we need 400 x 799 operations, which is ~320k.
    # This is fine for Python if vectorized.
    
    k = np.arange(2*n-1)
    i = np.arange(n)
    
    # k - i: shape (n, 2n-1)
    ki = k[None, :] - i[:, None]
    ik = i[None, :] - k[:, None]  # same as -ki
    
    # h[ki] where 0 <= ki < n
    valid_ki = (ki >= 0) & (ki < n)
    h_ki = np.where(valid_ki, h[ki.clip(0, n-1)], 0.0)
    
    valid_ik = (ik >= 0) & (ik < n)
    h_ik = np.where(valid_ik, h[ik.clip(0, n-1)], 0.0)
    
    w = dac_dac[None, :]
    
    grad_h = np.sum(w * (h_ki + h_ik), axis=1)
    
    # Now apply the 1/s^2 factor and its derivative
    # f = sm / s^2
    # df/dh = (dsm/dh * s^2 - sm * 2 * s * ds/dh) / s^4
    #       = dsm/dh / s^2 - 2 * sm * ds/dh / s^3
    # where ds/dh = ones (since s = sum h)
    
    sm = _soft_max_abs(ac, beta)
    ds = np.ones(n)
    
    grad = grad_h / (s * s) - 2 * sm * ds / (s * s * s)
    
    return grad

def _lbfgs_optimize(h_init, beta, maxiter=200, deadline=None):
    """Run L-BFGS-B with analytical gradient."""
    def obj(x):
        return _smooth_objective(x, beta)
    
    def grad(x):
        return _smooth_gradient(x, beta)
    
    res = minimize(obj, h_init, jac=grad, method='L-BFGS-B',
                   options={'maxiter': maxiter, 'disp': False, 'gtol': 1e-6})
    return res.x

def _block_sign_flip_local_search(h, block_size=10, max_passes=5, deadline=None):
    """
    Local search that tries flipping signs in blocks.
    """
    h = h.copy()
    best_val = _raw_value(h)
    rng = np.random.default_rng(123)
    
    for pass_num in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        
        # Try flipping signs in random blocks
        for _ in range(50):
            if deadline is not None and time.time() > deadline:
                break
            
            start = rng.integers(0, N - block_size)
            h_new = h.copy()
            h_new[start:start+block_size] = -h_new[start:start+block_size]
            
            new_val = _raw_value(h_new)
            if new_val < best_val:
                h = h_new
                best_val = new_val
                improved = True
        
        if not improved:
            break
    
    return h, best_val

def _single_flip_local_search(h, max_passes=5, deadline=None):
    """
    Local search: try flipping individual signs.
    """
    h = h.copy()
    best_val = _raw_value(h)
    
    for pass_num in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            
            h[i] = -h[i]
            new_val = _raw_value(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = -h[i]
        
        if not improved:
            break
    
    return h, best_val

def _spectral_subspace_optimize(h, subspace_dim=20, maxiter=50, deadline=None):
    """
    Optimize in a low-dimensional spectral subspace.
    Use the top eigenvectors of the Hessian or a related matrix.
    """
    h0 = h.copy()
    s0 = np.sum(h0)
    if abs(s0) < 1e-9:
        return h0
    
    # Project into a subspace spanned by Fourier modes
    n = len(h0)
    x = np.arange(n)
    modes = []
    for freq in range(1, subspace_dim+1):
        mode = np.cos(2 * np.pi * freq * x / n)
        mode = mode - np.mean(mode)  # center
        mode = mode / np.linalg.norm(mode)
        modes.append(mode)
    
    modes = np.array(modes)  # shape (subspace_dim, n)
    
    # Initial coefficients
    c0 = modes @ h0  # shape (subspace_dim,)
    
    def obj_sub(c):
        h_sub = h0 + modes.T @ c
        return _smooth_objective(h_sub, beta=100.0)
    
    def grad_sub(c):
        h_sub = h0 + modes.T @ c
        g = _smooth_gradient(h_sub, beta=100.0)
        return modes @ g
    
    res = minimize(obj_sub, c0, jac=grad_sub, method='L-BFGS-B',
                   options={'maxiter': maxiter, 'disp': False})
    
    h_opt = h0 + modes.T @ res.x
    return h_opt

def _epigraph_polish(h, maxiter=50, deadline=None):
    """
    Epigraph formulation: minimize t subject to |ac_k| <= t * s^2 for all k.
    This is a smooth problem if we relax the constraints.
    """
    h0 = h.copy()
    s0 = np.sum(h0)
    if abs(s0) < 1e-9:
        return h0
    
    ac0 = _autoconv_fft(h0)
    t0 = np.max(np.abs(ac0)) / (s0 * s0)
    
    # Parameterize: x = [h, t]
    # Objective: t
    # Constraints: |ac_k(h)| - t * (sum h)^2 <= 0
    
    # Use a smooth penalty for the constraints
    beta = 50.0
    
    def obj_epigraph(x):
        h_x = x[:N]
        t_x = x[N]
        s = np.sum(h_x)
        if abs(s) < 1e-9:
            return 1e10
        ac = _autoconv_fft(h_x)
        violation = np.max(ac / (s*s) - t_x, ac / (s*s) + t_x)
        # Soft penalty
        return t_x + beta * max(0, violation)**2
    
    def grad_epigraph(x):
        h_x = x[:N]
        t_x = x[N]
        s = np.sum(h_x)
        if abs(s) < 1e-9:
            return np.zeros(N+1)
        
        ac = _autoconv_fft(h_x)
        ac_scaled = ac / (s * s)
        
        # Gradient of t
        gt = np.ones(1)
        
        # Gradient of penalty term
        violation = np.max(ac_scaled - t_x, ac_scaled + t_x)
        if violation > 0:
            # d(violation)/d(h) = (d(ac)/d(h) / s^2 - 2 * ac * s * 1 / s^4)
            # This is complex; use numerical gradient for simplicity
            eps = 1e-6
            grad_num = np.zeros(N)
            for i in range(N):
                h_plus = h_x.copy()
                h_plus[i] += eps
                h_minus = h_x.copy()
                h_minus[i] -= eps
                s_plus = np.sum(h_plus)
                s_minus = np.sum(h_minus)
                if abs(s_plus) < 1e-9 or abs(s_minus) < 1e-9:
                    grad_num[i] = 0
                    continue
                ac_plus = _autoconv_fft(h_plus)
                ac_minus = _autoconv_fft(h_minus)
                ac_scaled_plus = ac_plus / (s_plus * s_plus)
                ac_scaled_minus = ac_minus / (s_minus * s_minus)
                
                viol_plus = np.max(ac_scaled_plus - t_x, ac_scaled_plus + t_x)
                viol_minus = np.max(ac_scaled_minus - t_x, ac_scaled_minus + t_x)
                
                penalty_plus = max(0, viol_plus)**2
                penalty_minus = max(0, viol_minus)**2
                grad_num[i] = beta * (penalty_plus - penalty_minus) / (2 * eps)
            
            grad = np.concatenate([grad_num, gt])
        else:
            grad = np.concatenate([np.zeros(N), gt])
        
        return grad
    
    x0 = np.concatenate([h0, [t0]])
    
    res = minimize(obj_epigraph, x0, jac=grad_epigraph, method='L-BFGS-B',
                   options={'maxiter': maxiter, 'disp': False})
    
    h_opt = res.x[:N]
    return h_opt

def _gen_chirp(n):
    i = np.arange(n)
    phase = np.pi * i**2 / n
    return np.cos(phase)

def _gen_random(n, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n)

def _gen_window(n):
    x = np.arange(n)
    mu = n / 2
    sigma = n * 0.25
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def construct():
    t_start = time.time()
    deadline = t_start + 145.0
    
    best_score = np.inf
    best_h = np.ones(N)
    
    # Generate diverse initial points
    inits = [
        np.ones(N),
        _gen_chirp(N),
        _gen_random(N, seed=1),
        _gen_random(N, seed=2),
        _gen_window(N),
        np.cos(np.pi * np.arange(N) / N),
        _gen_chirp(N) * _gen_window(N),
    ]
    
    for h_init in inits:
        if time.time() > deadline - 40:
            break
        
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        
        h_norm = h_init / s0
        
        # 1. L-BFGS with analytical gradient
        if time.time() < deadline - 30:
            try:
                h_opt = _lbfgs_optimize(h_norm, beta=100.0, maxiter=100, deadline=deadline-25)
                if abs(np.sum(h_opt)) > 1e-6:
                    val = _raw_value(h_opt)
                    if val < best_score:
                        best_score = val
                        best_h = h_opt.copy()
            except Exception:
                pass
        
        # 2. Spectral subspace optimization
        if time.time() < deadline - 25:
            try:
                h_spec = _spectral_subspace_optimize(best_h, subspace_dim=15, maxiter=30, deadline=deadline-20)
                if abs(np.sum(h_spec)) > 1e-6:
                    val = _raw_value(h_spec)
                    if val < best_score:
                        best_score = val
                        best_h = h_spec.copy()
            except Exception:
                pass
        
        # 3. Block sign flip local search
        if time.time() < deadline - 15:
            h_block, val_block = _block_sign_flip_local_search(best_h, block_size=8, max_passes=3, deadline=deadline-10)
            if val_block < best_score:
                best_score = val_block
                best_h = h_block.copy()
        
        # 4. Single flip local search
        if time.time() < deadline - 10:
            h_flip, val_flip = _single_flip_local_search(best_h, max_passes=3, deadline=deadline-5)
            if val_flip < best_score:
                best_score = val_flip
                best_h = h_flip.copy()
    
    # Final epigraph polish
    if time.time() < deadline - 5:
        try:
            h_ep = _epigraph_polish(best_h, maxiter=30, deadline=deadline-2)
            if abs(np.sum(h_ep)) > 1e-6:
                val_ep = _raw_value(h_ep)
                if val_ep < best_score:
                    best_score = val_ep
                    best_h = h_ep.copy()
        except Exception:
            pass
    
    # Final single flip polish
    if time.time() < deadline - 2:
        h_final, val_final = _single_flip_local_search(best_h, max_passes=2, deadline=deadline-1)
        if val_final < best_score:
            best_score = val_final
            best_h = h_final.copy()
    
    if not np.all(np.isfinite(best_h)):
        return np.ones(N)
        
    return best_h

# EVOLVE-BLOCK-END
