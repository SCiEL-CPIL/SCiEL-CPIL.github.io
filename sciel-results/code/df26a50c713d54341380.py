"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.

Combined approach:
1. Use a Fourier-series representation (h_i = 0.5 + sum A_j cos(2*pi*j*i/N))
   which automatically satisfies the sum constraint (sum h = N/2) and keeps
   h in [0,1] for small enough coefficients.
2. Optimize the Fourier coefficients with L-BFGS-B using a smooth max objective
   and analytic gradient.
3. Polish the resulting vector with direct local search (pairwise mass moves).
4. Also try the 2-block golden-ratio construction and a few random starts.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE_MARGIN = 15.0  # seconds to spare before 150s kill
TARGET_SUM = N / 2.0


def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    if n == 0:
        return 0.0
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))


def _fix_sum(h, target=None):
    """Adjust h so that sum(h) == target (default N/2), preserving [0,1] bounds."""
    if target is None:
        target = TARGET_SUM
    h = np.asarray(h, dtype=float).copy()
    diff = target - h.sum()
    if abs(diff) > 1e-12:
        h = h + diff / len(h)
        h = np.clip(h, 0.0, 1.0)
        diff2 = target - h.sum()
        if abs(diff2) > 1e-12:
            mask = (h > 1e-9) & (h < 1.0 - 1e-9)
            if np.any(mask):
                h[mask] += diff2 / np.sum(mask)
                h = np.clip(h, 0.0, 1.0)
    return h


def _smooth_max(vals, beta=200.0):
    """Smooth max via log-sum-exp."""
    m = np.max(vals)
    if m < -1e-15:
        return 0.0
    return m + np.log(np.sum(np.exp(beta * (vals - m)))) / beta


def _smooth_max_grad(vals, beta=200.0):
    """Gradient of smooth max via log-sum-exp."""
    m = np.max(vals)
    if m < -1e-15:
        return np.zeros_like(vals)
    exp_vals = np.exp(beta * (vals - m))
    sum_exp = np.sum(exp_vals)
    return (exp_vals / sum_exp)


def _fourier_to_h(coeffs):
    """Convert Fourier coefficients (cosine series, no DC) to h vector.
    h_i = 0.5 + sum_{j=1}^{J} coeffs[j-1] * cos(2*pi*j*i/N)
    """
    i = np.arange(N)
    h = np.full(N, 0.5)
    J = len(coeffs)
    for j in range(1, J + 1):
        h += coeffs[j - 1] * np.cos(2 * np.pi * j * i / N)
    return h


def _fourier_obj_and_grad(coeffs, beta=200.0):
    """Objective and gradient for Fourier coefficient optimization.
    Objective: smooth_max(overlap) + penalty * (sum(h) - N/2)^2
    """
    h = _fourier_to_h(coeffs)
    # Penalty for bounds violation
    lower_viol = np.maximum(0.0, -h)
    upper_viol = np.maximum(0.0, h - 1.0)
    bound_pen = 1000.0 * (np.sum(lower_viol**2) + np.sum(upper_viol**2))
    
    # Clip for overlap computation (but keep raw h for gradient)
    h_clipped = np.clip(h, 0.0, 1.0)
    n = len(h_clipped)
    shifted = np.zeros(n)
    shifted[:-1] = h_clipped[1:]
    overlap = h_clipped * (1.0 - shifted)
    
    smax = _smooth_max(overlap, beta)
    grad_overlap = _smooth_max_grad(overlap, beta)
    
    # Sum constraint penalty
    sum_err = h_clipped.sum() - TARGET_SUM
    sum_pen = 50.0 * sum_err**2
    
    total_obj = smax + bound_pen + sum_pen
    
    # Gradient w.r.t. h
    grad_h = np.zeros(n)
    grad_h[:] += grad_overlap * (1.0 - shifted)
    if n > 1:
        grad_h[1:] += -grad_overlap[:-1] * h_clipped[:-1]
    
    # Bound penalty gradient
    grad_h += 2000.0 * (-lower_viol + upper_viol)
    
    # Sum penalty gradient
    grad_h += 100.0 * sum_err * np.ones(n)
    
    # Chain rule: grad_coeffs[j-1] = sum_i grad_h[i] * d(h_i)/d(coeffs[j-1])
    # d(h_i)/d(coeffs[j-1]) = cos(2*pi*j*i/N)
    i = np.arange(N)
    grad_coeffs = np.zeros(len(coeffs))
    for j in range(1, len(coeffs) + 1):
        grad_coeffs[j - 1] = np.dot(grad_h, np.cos(2 * np.pi * j * i / N))
    
    return total_obj, grad_coeffs


def _optimize_fourier(beta=200.0, max_iter=5000, J=10):
    """Optimize Fourier coefficients using L-BFGS-B."""
    rng = np.random.default_rng(42)
    best_score = 1e10
    best_h = None
    
    # Multiple random starts
    for _ in range(5):
        coeffs0 = rng.uniform(-0.05, 0.05, J)
        try:
            res = minimize(_fourier_obj_and_grad, coeffs0, method='L-BFGS-B',
                           jac=True, options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10})
            h = _fourier_to_h(res.x)
            h = _fix_sum(h)
            s = _score(h)
            if s < best_score:
                best_score = s
                best_h = h.copy()
        except Exception:
            pass
    
    return best_score, best_h


def _optimize_full_vector(h0, beta=200.0, max_iter=5000):
    """Optimize full vector h using L-BFGS-B with smooth objective and analytic gradient."""
    
    def obj_and_grad(x):
        h = np.clip(x, 0.0, 1.0)
        n = len(h)
        shifted = np.zeros(n)
        shifted[:-1] = h[1:]
        overlap = h * (1.0 - shifted)
        
        smax = _smooth_max(overlap, beta)
        grad_overlap = _smooth_max_grad(overlap, beta)
        
        sum_err = h.sum() - TARGET_SUM
        sum_pen = 50.0 * sum_err**2
        grad_sum_pen = 100.0 * sum_err * np.ones(n)
        
        grad_h = np.zeros(n)
        grad_h[:] += grad_overlap * (1.0 - shifted)
        if n > 1:
            grad_h[1:] += -grad_overlap[:-1] * h[:-1]
            
        total_grad = grad_h + grad_sum_pen
        return smax + sum_pen, total_grad

    h0_clipped = np.clip(h0, 0.0, 1.0)
    try:
        res = minimize(obj_and_grad, h0_clipped, method='L-BFGS-B', jac=True,
                       options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10})
        h_opt = np.clip(res.x, 0.0, 1.0)
        h_opt = _fix_sum(h_opt)
        return h_opt
    except Exception:
        return _fix_sum(h0_clipped)


def _local_search(h, max_iters=3000):
    """Fast local search: pair-wise mass moves to maintain sum exactly."""
    h = np.clip(h, 0.0, 1.0).copy()
    h = _fix_sum(h)
    best_score = _score(h)
    best_h = h.copy()

    indices = np.arange(N)
    
    for it in range(max_iters):
        improved = False
        rng = np.random.default_rng(it)
        perm = rng.permutation(N)
        
        for i in perm:
            if improved:
                break
            js = rng.choice(N, size=10, replace=False)
            for j in js:
                if i == j:
                    continue
                for delta in [0.01, 0.005, 0.001]:
                    if h[i] + delta <= 1.0 and h[j] - delta >= 0.0:
                        h_try = h.copy()
                        h_try[i] += delta
                        h_try[j] -= delta
                        s = _score(h_try)
                        if s < best_score - 1e-12:
                            best_score = s
                            best_h = h_try.copy()
                            h = h_try.copy()
                            improved = True
                            break
                    
                    if h[j] + delta <= 1.0 and h[i] - delta >= 0.0:
                        h_try = h.copy()
                        h_try[j] += delta
                        h_try[i] -= delta
                        s = _score(h_try)
                        if s < best_score - 1e-12:
                            best_score = s
                            best_h = h_try.copy()
                            h = h_try.copy()
                            improved = True
                            break
                if improved:
                    break
        if not improved:
            break
    return best_score, best_h


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN

    best_score = 1e10
    best_h = None

    # 1. Golden-ratio 2-block (from seed)
    A0 = (3.0 - math.sqrt(5.0)) / 2.0
    B0 = 1.0 - A0
    m0 = N // 2
    h_2block = np.zeros(N)
    h_2block[:m0] = A0
    h_2block[m0:] = B0
    h_2block = _fix_sum(h_2block)
    s = _score(h_2block)
    if s < best_score:
        best_score = s
        best_h = h_2block.copy()

    # 2. Uniform
    h_unif = np.full(N, 0.5)
    s = _score(h_unif)
    if s < best_score:
        best_score = s
        best_h = h_unif.copy()

    # 3. Fourier optimization (structural change from inspirations)
    if time.time() < deadline - 60:
        s_four, h_four = _optimize_fourier(beta=300.0, max_iter=3000, J=10)
        if h_four is not None and s_four < best_score:
            best_score = s_four
            best_h = h_four.copy()

    # 4. Full-vector L-BFGS-B optimization from multiple starts
    if time.time() < deadline - 40:
        starts = [best_h.copy(), h_2block.copy(), h_unif.copy()]
        for _ in range(3):
            h_rand = np.random.uniform(0.2, 0.8, N)
            h_rand = _fix_sum(h_rand)
            starts.append(h_rand)

        for h0 in starts:
            if time.time() > deadline - 25:
                break
            h_opt = _optimize_full_vector(h0, beta=300.0, max_iter=3000)
            s = _score(h_opt)
            if s < best_score:
                best_score = s
                best_h = h_opt.copy()

    # 5. Local search polish
    if time.time() < deadline - 10:
        s_ls, h_ls = _local_search(best_h, max_iters=2000)
        if s_ls < best_score:
            best_score = s_ls
            best_h = h_ls.copy()

    # 6. Another round of L-BFGS-B from the locally-searched result
    if time.time() < deadline - 15:
        h_opt2 = _optimize_full_vector(best_h, beta=500.0, max_iter=3000)
        s = _score(h_opt2)
        if s < best_score:
            best_score = s
            best_h = h_opt2.copy()
        if time.time() < deadline - 5:
            s_ls2, h_ls2 = _local_search(best_h, max_iters=1000)
            if s_ls2 < best_score:
                best_score = s_ls2
                best_h = h_ls2.copy()

    # Final fix
    best_h = np.clip(best_h, 0.0, 1.0)
    best_h = _fix_sum(best_h)

    return best_h
