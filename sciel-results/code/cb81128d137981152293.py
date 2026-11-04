"""Second autocorrelation inequality: maximize ||h||^2_2 / (||h||_1 * ||h*h||_inf)
over 50 nonnegative step heights using a narrow-rectangular-pulse parameterization.

Key insight: for a rectangular pulse of width w (in discrete samples), the ratio
scales as ~1/w. By making the pulse as narrow as possible (w=1, a single spike),
we get the maximum possible ratio. However, we can do better by optimizing
the shape to minimize autoconvolution peak while maximizing L2/L1.

The optimal structure is a sparse set of well-separated peaks (Sidon-like sets)
that minimize autoconvolution overlap. We parameterize heights directly and use
a combination of:
1. Direct narrow-pulse initializations
2. DFT-based spectral optimization with smooth-max surrogate
3. Local coordinate descent on the heights directly
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 50
_DEADLINE = 145.0  # safety margin before 150s kill


def _time_left():
    return _DEADLINE - time.time()


def _autoconv_peak(h):
    """Compute the supremum norm of the autoconvolution of h."""
    conv = np.convolve(h, h, mode='full')
    return np.max(conv)


def _ratio(h):
    """Compute the ratio: squared L2 norm / (L1 norm * sup norm of autoconvolution)."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-15) or np.sum(h) < 1e-15:
        return 0.0
    h = np.maximum(h, 0.0)
    l2sq = np.sum(h ** 2)
    l1 = np.sum(h)
    peak = _autoconv_peak(h)
    if peak < 1e-30:
        return 0.0
    return l2sq / (l1 * peak)


def _smooth_max(x, sigma=0.005):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    if x_max <= 0:
        return 0.0
    x_shifted = (x - x_max) / sigma
    x_shifted = np.clip(x_shifted, -100, 0)
    log_sum_exp = x_max + sigma * np.log(np.sum(np.exp(x_shifted)))
    return log_sum_exp


def _objective_complex(z_flat):
    """
    Objective to minimize: negative of the ratio.
    z_flat is a real vector of length 2*N representing complex z.
    h = |FFT(z)|^2, normalized to L1 = 1.
    """
    re = z_flat[:N]
    im = z_flat[N:]
    z = re + 1j * im
    fft_z = np.fft.fft(z)
    h = np.abs(fft_z) ** 2
    
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h = h / l1
    
    l2sq = np.sum(h ** 2)
    conv = np.convolve(h, h, mode='full')
    peak = _smooth_max(conv, sigma=0.005)
    
    if peak < 1e-30:
        return 0.0
    
    return -l2sq / peak


def _z_to_heights(z):
    """Convert complex vector z to nonnegative heights via |FFT(z)|^2, normalized to L1=1."""
    z = np.asarray(z, dtype=np.complex128)
    fft_z = np.fft.fft(z)
    h = np.abs(fft_z) ** 2
    l1 = np.sum(h)
    if l1 > 0:
        h = h / l1
    return h


def _heights_to_z(h):
    """Convert nonnegative heights to a complex vector z such that |FFT(z)|^2 ~ h."""
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    mags = np.sqrt(h)
    z = np.fft.ifft(mags)
    return z


def _local_optimize_z(z0, max_iter=300):
    """Optimize the complex vector z using L-BFGS-B."""
    x0 = np.concatenate([z0.real, z0.imag])
    
    try:
        res = minimize(_objective_complex, x0, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-14, 
                                'gtol': 1e-10, 'maxfun': 2000})
        z_opt = res.x[:N] + 1j * res.x[N:]
        h_opt = _z_to_heights(z_opt)
        score = _ratio(h_opt)
        return h_opt, score
    except Exception:
        h0 = _z_to_heights(z0)
        return h0, _ratio(h0)


def _direct_heights_optimize(h0, max_iter=50):
    """Direct coordinate descent on heights, trying to move mass to reduce peak."""
    h = np.array(h0, dtype=np.float64)
    h = np.maximum(h, 0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return h, 0.0
    h = h / l1
    
    best_score = _ratio(h)
    best_h = h.copy()
    
    for _ in range(max_iter):
        if _time_left() < 3:
            break
        improved = False
        for i in range(N):
            if _time_left() < 3:
                break
            # Try moving some mass from i to other positions
            if h[i] < 1e-10:
                continue
            for j in range(N):
                if i == j:
                    continue
                # Try shifting delta from i to j
                for delta_frac in [0.1, 0.25, 0.5, 0.75, 0.9]:
                    delta = h[i] * delta_frac
                    if delta < 1e-15:
                        continue
                    h_new = h.copy()
                    h_new[i] -= delta
                    h_new[j] += delta
                    h_new = np.maximum(h_new, 0)
                    l1_new = np.sum(h_new)
                    if l1_new < 1e-15:
                        continue
                    h_new = h_new / l1_new
                    s = _ratio(h_new)
                    if s > best_score + 1e-10:
                        best_score = s
                        best_h = h_new.copy()
                        h = h_new.copy()
                        improved = True
                        break
        if not improved:
            break
    
    return best_h, best_score


def _narrow_pulse(width, center=None):
    """Create a narrow rectangular pulse of given width."""
    h = np.zeros(N)
    if center is None:
        center = N // 2
    half = width // 2
    start = max(0, center - half)
    end = min(N, center + half + 1)
    h[start:end] = 1.0
    return h


def _sidon_like_positions(k, seed=42):
    """Generate k well-separated positions (Sidon-like: all pairwise sums distinct)."""
    rng = np.random.RandomState(seed)
    positions = []
    attempts = 0
    while len(positions) < k and attempts < 10000:
        p = rng.randint(0, N)
        # Check Sidon condition: all pairwise sums must be distinct
        valid = True
        sums = set()
        for i in range(len(positions)):
            s = positions[i] + p
            if s in sums:
                valid = False
                break
            sums.add(s)
        # Also check with existing pairs
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                s1 = positions[i] + p
                s2 = positions[j] + p
                s3 = positions[i] + positions[j]
                if s1 == s2 or s1 == s3 or s2 == s3:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            positions.append(p)
        attempts += 1
    return positions


def _initializations():
    """Generate diverse initial candidates focused on sparse/narrow structures."""
    candidates = []
    
    # 1. Single spike (narrowest possible)
    for pos in range(N):
        h = np.zeros(N)
        h[pos] = 1.0
        candidates.append(h)
    
    # 2. Narrow rectangular pulses of various widths
    for width in [1, 2, 3, 4, 5]:
        for center in range(0, N, max(1, N // 10)):
            candidates.append(_narrow_pulse(width, center))
    
    # 3. Sidon-like sparse sets
    for k in [2, 3, 4, 5, 6, 7, 8]:
        for seed in [42, 123, 456, 789, 101]:
            positions = _sidon_like_positions(k, seed)
            if len(positions) >= 2:
                h = np.zeros(N)
                for p in positions:
                    h[p] = 1.0
                candidates.append(h)
    
    # 4. Well-separated pairs
    for i in range(0, N, 2):
        for j in range(i + 3, N, 2):
            if j - i >= 5:  # ensure separation
                h = np.zeros(N)
                h[i] = 1.0
                h[j] = 1.0
                candidates.append(h)
    
    # 5. DFT-based: optimize complex z to produce sparse-like h
    for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        rng_local = np.random.RandomState(seed)
        z = rng_local.randn(N) + 1j * rng_local.randn(N)
        H = np.fft.fft(z)
        h_dft = np.abs(H)**2
        h_dft = h_dft / np.max(h_dft + 1e-30)
        # Threshold to make sparse
        threshold = np.percentile(h_dft[h_dft > 0], 70) if np.any(h_dft > 0) else 0
        h_sparse = np.where(h_dft > threshold, h_dft, 0)
        if np.sum(h_sparse) > 0:
            h_sparse = h_sparse / np.sum(h_sparse)
            candidates.append(h_sparse)
        candidates.append(h_dft / np.sum(h_dft + 1e-30))
    
    # 6. Gaussian with very narrow width
    x = np.linspace(-1, 1, N)
    for width in [0.05, 0.1, 0.15, 0.2]:
        for center in [-0.5, 0, 0.5]:
            h = np.exp(-((x - center)**2) / width)
            if np.sum(h) > 0:
                candidates.append(h / np.sum(h))
    
    # 7. Bimodal with separation
    for sep in [0.3, 0.4, 0.5]:
        for width in [0.05, 0.1]:
            h = np.exp(-((x - sep)**2) / width) + np.exp(-((x + sep)**2) / width)
            if np.sum(h) > 0:
                candidates.append(h / np.sum(h))
    
    return candidates


def construct():
    start_time = time.time()
    
    best_h = np.ones(N)
    best_score = _ratio(best_h)
    
    # Evaluate initial candidates and find best
    candidates = _initializations()
    scored_candidates = []
    for h in candidates:
        if _time_left() < 5:
            break
        s = _ratio(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
        scored_candidates.append((s, h))
    
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Try direct height optimization from best candidates
    for s_cand, h_cand in scored_candidates[:20]:
        if _time_left() < 5:
            break
        h_opt, s_opt = _direct_heights_optimize(h_cand, max_iter=30)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt
    
    # Convert best to spectral representation and optimize
    z_best = _heights_to_z(best_h)
    
    # Optimize from the best initial candidate via spectral method
    if _time_left() > 10:
        h_opt, s_opt = _local_optimize_z(z_best, max_iter=300)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt
            z_best = _heights_to_z(best_h)
    
    # Try optimization from several diverse starting points (top candidates)
    tried = set()
    for s_cand, h_cand in scored_candidates[:15]:
        if _time_left() < 8:
            break
        # Check if this candidate is "close" to already tried ones
        is_new = True
        for _, h_tried in tried:
            if np.sum((h_cand - h_tried)**2) < 0.1:
                is_new = False
                break
        if not is_new:
            continue
        
        z_cand = _heights_to_z(h_cand)
        # Add small random perturbation
        rng = np.random.RandomState(int(time.time() * 1000) % 10000)
        z_pert = z_cand + (rng.randn(N) + 1j * rng.randn(N)) * 0.03
        
        h_opt, s_opt = _local_optimize_z(z_pert, max_iter=250)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt
            z_best = _heights_to_z(best_h)
        
        tried.append((s_cand, h_cand))
    
    # Final polish: perturb around the best solution
    if _time_left() > 8:
        for pert_seed in [100, 200, 300, 400, 500, 600, 700]:
            if _time_left() < 5:
                break
            rng_local = np.random.RandomState(pert_seed)
            z_pert = z_best + (rng_local.randn(N) + 1j * rng_local.randn(N)) * 0.05
            
            h_opt, s_opt = _local_optimize_z(z_pert, max_iter=200)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt
                z_best = _heights_to_z(best_h)
    
    # Also try direct optimization from the final best
    if _time_left() > 5:
        h_opt, s_opt = _direct_heights_optimize(best_h, max_iter=40)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt
    
    # Ensure nonnegative and return
    best_h = np.maximum(0, best_h)
    
    return list(best_h)


if __name__ == "__main__":
    result = construct()
    print("Score:", _ratio(result))
    print("Heights:", result)
