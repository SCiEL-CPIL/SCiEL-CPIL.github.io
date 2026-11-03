"""Signed step heights minimizing max|autoconv|/(sum h)^2.

Scale-invariant: normalize sum(h)=1, minimize max|autoconv(h)|.
autoconv(h) = IFFT(|FFT(h)|^2) = h @ C @ h  (C circulant, C e_k = IFFT(FFT(e_k)^2)).
Gradient of peak term: d(ac[k])/dh = C @ e_k  (precomputed once, N FFTs).

Pipeline:
  1. Precompute G (N x N) where G[:,k] = C @ e_k  (gradient of ac[k] w.r.t. h)
  2. Global: DE on smooth log-sum-exp surrogate (beta annealing) + random inits
  3. Exact Newton coordinate descent (best coordinate per pass, Newton step + fallback)
  4. Block search (top-K gradient coordinates, L-BFGS-B on block, rest fixed)
  5. Gradient descent in full space (with sum-preserving projection)
  6. Final high-precision Newton coordinate descent
"""

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _fft_size(n):
    f = 1
    while f < n:
        f *= 2
    return f

def _autoconv(h):
    """Linear autoconvolution of h via FFT. Returns array of length 2N-1."""
    n = len(h)
    size = 2 * n - 1
    fsize = _fft_size(size)
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:size]

def _autoconv_max_abs(h):
    ac = _autoconv(h)
    return float(np.max(np.abs(ac)))

def _score(h):
    s = float(np.sum(h))
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _normalize(h):
    s = float(np.sum(h))
    if abs(s) < 1e-9:
        return h
    return h / s

def _soft_max_abs(ac, beta):
    """Smooth approximation of max|ac| via log-sum-exp."""
    abs_ac = np.abs(ac)
    max_val = float(np.max(abs_ac))
    if max_val < 1e-12:
        return 0.0
    shifted = abs_ac - max_val
    log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
    return max_val + log_sum

def _smooth_score(h, beta):
    s = float(np.sum(h))
    if abs(s) < 1e-9:
        return 1e10
    n = len(h)
    size = 2 * n - 1
    fsize = _fft_size(size)
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size]
    return _soft_max_abs(ac, beta) / (s * s)

def _build_gradient_matrix():
    """
    Build G (N x N) where G[:, k] = C @ e_k = IFFT(FFT(e_k)^2)[:N].
    G[:, k] is the gradient of ac[k] w.r.t. h (for the circulant autoconvolution).
    This is computed once and reused throughout the optimization.
    """
    n = N
    fsize = _fft_size(2 * n - 1)
    G = np.empty((n, n))
    for k in range(n):
        e = np.zeros(fsize)
        e[k] = 1.0
        E = np.fft.rfft(e)
        col = np.fft.irfft(E * E, fsize)[:n]
        G[:, k] = col
    return G

def _newton_coordinate_descent(h, G, max_passes=200, deadline=None, tol=1e-12):
    """
    Exact Newton coordinate descent on max|autoconv| with sum(h)=1 constraint.
    
    For each coordinate i, the quadratic in delta (h[i] -> h[i]+delta) is:
      ac[k] = ac0[k] + delta * G[i,k]
    We minimize max_k |ac0[k] + delta * G[i,k]| over delta, then adjust
    to maintain sum(h)=1.
    
    Actually, since we normalize after each step, we work with the
    unnormalized problem and project.
    """
    h = h.copy()
    ac = _autoconv(h)
    cur_max = float(np.max(np.abs(ac)))
    
    for _pass in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        
        best_i = -1
        best_new_max = cur_max
        best_delta = 0.0
        
        # Find the best coordinate to update
        # For each i, the peak index is where |ac[k]| is max
        # We focus on the current peak and nearby peaks
        abs_ac = np.abs(ac)
        peak_idx = int(np.argmax(abs_ac))
        peak_val = abs_ac[peak_idx]
        sign_peak = np.sign(ac[peak_idx])
        
        # Try updating each coordinate to reduce the peak
        # For coordinate i: ac[peak] + delta * G[i, peak]
        # We want to reduce |ac[peak] + delta * G[i, peak]|
        # Newton step: delta = -ac[peak] / G[i, peak] (if G[i,peak] != 0)
        
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            
            g_ik = G[i, peak_idx]
            if abs(g_ik) < 1e-15:
                continue
            
            # Newton step to zero out the peak
            delta_newton = -ac[peak_idx] / g_ik
            
            # Also try smaller steps
            candidates = [delta_newton, 0.5 * delta_newton, 0.25 * delta_newton,
                          0.75 * delta_newton, 0.125 * delta_newton]
            
            best_local_delta = 0.0
            best_local_max = cur_max
            
            for d in candidates:
                # Update ac: ac[k] += d * G[i, k]
                new_ac = ac + d * G[i, :]
                new_max = float(np.max(np.abs(new_ac)))
                
                # Also check: after normalizing, the score might change
                # But for now, just minimize max|ac|
                if new_max < best_local_max - 1e-14:
                    best_local_max = new_max
                    best_local_delta = d
            
            if best_local_max < best_new_max - 1e-13:
                best_new_max = best_local_max
                best_i = i
                best_delta = best_local_delta
        
        if best_i == -1 or best_new_max >= cur_max - 1e-13:
            break
        
        # Apply the best update
        h[best_i] += best_delta
        ac = ac + best_delta * G[best_i, :]
        cur_max = best_new_max
        
        # Renormalize to sum=1
        s = float(np.sum(h))
        if abs(s) < 1e-9:
            break
        h = h / s
        # After normalization, ac scales by 1/s^2
        ac = ac / (s * s)
        cur_max = float(np.max(np.abs(ac)))
    
    return h, cur_max

def _block_search(h, G, K=20, max_iters=50, deadline=None):
    """
    Block search: identify the top-K coordinates most involved in the peak,
    then optimize only those K coordinates (L-BFGS-B) while fixing the rest.
    """
    h = h.copy()
    ac = _autoconv(h)
    abs_ac = np.abs(ac)
    cur_max = float(np.max(abs_ac))
    
    for _iter in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Identify which coordinates have the largest gradient contribution
        # to the peak
        peak_idx = int(np.argmax(abs_ac))
        g_peak = G[:, peak_idx]  # gradient of ac[peak] w.r.t. each coordinate
        
        # Top K coordinates by |gradient at peak|
        top_k = np.argsort(np.abs(g_peak))[-K:]
        
        # Extract the block
        h_block = h[top_k].copy()
        G_block = G[:, top_k]  # (2N-1, K)
        
        # Optimize the block to minimize max|ac|
        def block_obj(params):
            # ac = ac_fixed + params @ G_block  (but ac_fixed depends on h_block)
            # Actually: ac = h @ C @ h, so changing h[top_k] changes ac quadratically
            # For the block: ac = ac_rest + 2 * h_rest @ C_block @ params + params @ C_block_block @ params
            # This is complex; use the linear approximation:
            # ac ≈ ac_current + (params - h_block) @ G_block
            delta = params - h_block
            new_ac = ac + delta @ G_block
            return float(np.max(np.abs(new_ac)))
        
        res = minimize(
            block_obj, h_block, method='L-BFGS-B',
            options={'maxiter': 100, 'gtol': 1e-8}
        )
        
        h[top_k] = res.x
        new_ac = _autoconv(h)
        new_max = float(np.max(np.abs(new_ac)))
        
        if new_max >= cur_max - 1e-12:
            break
        
        h = _normalize(h)
        ac = _autoconv(h)
        cur_max = float(np.max(np.abs(ac)))
        abs_ac = np.abs(ac)
    
    return h, cur_max

def _gradient_descent_subspace(h, G, max_iters=500, deadline=None, lr=0.01):
    """
    Gradient descent in the full space, projected to maintain sum(h)=1.
    Gradient of max|ac| at peak k: G[:, k] (the k-th column).
    """
    h = h.copy()
    ac = _autoconv(h)
    cur_max = float(np.max(np.abs(ac)))
    
    for _iter in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        abs_ac = np.abs(ac)
        peak_idx = int(np.argmax(abs_ac))
        sign_peak = np.sign(ac[peak_idx])
        
        # Gradient of ac[peak] w.r.t. h
        grad = G[:, peak_idx] * sign_peak
        
        # Project gradient to maintain sum(h)=1: subtract mean
        grad = grad - np.mean(grad)
        
        # Adaptive step size
        step = lr / (np.linalg.norm(grad) + 1e-10)
        
        h_new = h - step * grad
        h_new = _normalize(h_new)
        
        new_ac = _autoconv(h_new)
        new_max = float(np.max(np.abs(new_ac)))
        
        if new_max < cur_max - 1e-14:
            h = h_new
            ac = new_ac
            cur_max = new_max
        else:
            # Reduce step size
            lr *= 0.5
            if lr < 1e-10:
                break
    
    return h, cur_max

def _random_init(rng, n):
    """Generate a random initial condition."""
    h = rng.standard_normal(n)
    # Mix in some structure
    x = np.arange(n, dtype=float)
    h += 0.5 * np.cos(2 * np.pi * rng.uniform(1, 10) * x / n)
    h += 0.3 * np.sin(2 * np.pi * rng.uniform(1, 10) * x / n)
    return _normalize(h)

def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin before 150s limit
    
    best_h = np.ones(N) / N
    best_score = _score(best_h)
    
    # Precompute gradient matrix
    G = _build_gradient_matrix()
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Global optimization with DE on smooth surrogate
    if time.time() < deadline - 60:
        # Use a basis for DE (lower dimensional)
        n_basis = 30
        x = np.arange(N, dtype=float)
        basis = []
        basis.append(np.ones(N))
        for k in range(1, 10):
            basis.append(np.cos(2 * np.pi * k * x / N))
        for k in range(1, 10):
            basis.append(np.sin(2 * np.pi * k * x / N))
        for center_frac in [0.1, 0.3, 0.5, 0.7, 0.9]:
            mu = center_frac * N
            sigma = N / 6
            basis.append(np.exp(-(x - mu)**2 / (2 * sigma**2)))
        basis.append(np.cos(np.pi * x / N))
        basis.append((x - N/2)**2 / (N**2))
        basis.append(x / N - 0.5)
        basis = np.array(basis[:n_basis])
        
        def de_objective(params):
            if time.time() > deadline - 55:
                return 1e10
            h = np.dot(params, basis)
            s = float(np.sum(h))
            if abs(s) < 1e-9:
                return 1e10
            return _soft_max_abs(_autoconv(h), 50.0) / (s * s)
        
        try:
            bounds = [(-2.0, 2.0)] * n_basis
            res_de = differential_evolution(
                de_objective, bounds,
                seed=42, maxiter=30, tol=1e-5,
                popsize=12, mutation=(0.3, 1.2), recombination=0.7,
                polish=False
            )
            h_de = np.dot(res_de.x, basis)
            h_de = _normalize(h_de)
            val_de = _score(h_de)
            if val_de < best_score:
                best_score = val_de
                best_h = h_de.copy()
        except Exception:
            pass
        
        # Also try a few random inits
        for _ in range(5):
            if time.time() > deadline - 55:
                break
            h_rand = _random_init(rng, N)
            val_rand = _score(h_rand)
            if val_rand < best_score:
                best_score = val_rand
                best_h = h_rand.copy()
    
    # Phase 2: Newton coordinate descent
    if time.time() < deadline - 20:
        h_cd, val_cd = _newton_coordinate_descent(
            best_h, G, max_passes=300, deadline=deadline - 15
        )
        val_cd = _score(h_cd)
        if val_cd < best_score:
            best_score = val_cd
            best_h = h_cd.copy()
    
    # Phase 3: Block search
    if time.time() < deadline - 15:
        h_block, val_block = _block_search(
            best_h, G, K=20, max_iters=30, deadline=deadline - 10
        )
        val_block = _score(h_block)
        if val_block < best_score:
            best_score = val_block
            best_h = h_block.copy()
    
    # Phase 4: Gradient descent in subspace
    if time.time() < deadline - 10:
        h_gd, val_gd = _gradient_descent_subspace(
            best_h, G, max_iters=500, deadline=deadline - 5, lr=0.05
        )
        val_gd = _score(h_gd)
        if val_gd < best_score:
            best_score = val_gd
            best_h = h_gd.copy()
    
    # Phase 5: Final high-precision Newton coordinate descent
    if time.time() < deadline - 3:
        h_final, val_final = _newton_coordinate_descent(
            best_h, G, max_passes=200, deadline=deadline - 1, tol=1e-14
        )
        val_final = _score(h_final)
        if val_final < best_score:
            best_score = val_final
            best_h = h_final.copy()
    
    # Final polish: a few more coordinate descent passes
    if time.time() < deadline - 1:
        h_polish, val_polish = _newton_coordinate_descent(
            best_h, G, max_passes=50, deadline=deadline - 0.5, tol=1e-15
        )
        val_polish = _score(h_polish)
        if val_polish < best_score:
            best_score = val_polish
            best_h = h_polish.copy()
    
    # Ensure valid output
    best_h = _normalize(best_h)
    final_score = _score(best_h)
    
    # Fallback to baseline if something went wrong
    if not np.isfinite(final_score) or final_score > _score(np.ones(N) / N):
        return np.ones(N) / N
    
    return best_h

# EVOLVE-BLOCK-END
