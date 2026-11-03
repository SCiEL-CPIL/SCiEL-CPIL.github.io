"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Structural change: frequency-domain formulation. The overlap values C_k are the
inverse Fourier transform of the power spectrum. We optimize the power spectrum
coefficients (real, non-negative, symmetric) to minimize the peak of their inverse
transform, subject to the constraint that the resulting autocorrelation corresponds
to a valid h in [0,1] with sum N/2.

Key insight: C_k = (2/N)(N/2 - r_k) where r_k is the autocorrelation.
The Fourier transform of r_k is |H_k|^2 (the power spectrum).
So C_k = (2/N)(N/2 - ifft(|H|^2)_k).

We parametrize via the power spectrum P_k = |H_k|^2 >= 0, and the constraint
sum(h) = N/2 means H_0 = N/2 (DC component fixed).

The objective is to minimize max_k C_k = (2/N)(N/2 - r_k), which is equivalent
to maximizing min_k r_k, or equivalently making r_k as flat as possible (close to N/2).

We use a direct approach: optimize h in the time domain but use a different
optimization strategy - alternating between time-domain and frequency-domain
projections, combined with a subgradient method for the max constraint.

Actually, let me think more carefully. The key structural change should be:
1. Use the fact that C_k depends only on the autocorrelation r_k
2. The autocorrelation is determined by the power spectrum P_k = |FFT(h)|_k^2
3. We can optimize over P_k directly (convex in P_k for the objective)
4. Then recover h from P_k (up to a phase, which we can choose)

The constraint sum(h_i) = N/2 means P_0 = (N/2)^2 = 2500.
The constraint 0 <= h_i <= 1 is non-convex in the frequency domain.

Alternative structural approach: Use a semidefinite programming relaxation or
a different parameterization.

Let me try a cleaner approach:
- The overlap C_k = (2/N)(N/2 - r_k)
- We want to minimize max_k C_k
- This is equivalent to: find h such that r_k >= N/2 - (N/2)*t for all k, minimize t
- Where t is the maximum overlap value

This is a feasibility problem: r_k >= N/2(1-t) for all k, 0<=h<=1, sum(h)=N/2.

The autocorrelation constraint r_k >= c for all k can be written as:
sum_{i=k}^{N-1} h_i h_{i-k} >= c for k=0..N-1.

This is a set of quadratic constraints. We can use a bisection on t and solve
the feasibility problem.

For the feasibility problem, we can use projected gradient or a sequential
quadratic approach.

Let me implement:
1. Bisection on t (the target max overlap)
2. For each t, check if there exists h in [0,1]^N with sum(h)=N/2 and
   r_k >= N/2(1-t) for all k
3. Use projected gradient descent on the constraint violations

Actually, a simpler and more effective structural change:
- Use the fact that the problem is to minimize max_k C_k
- C_k = (2/N)(N/2 - r_k)
- So minimizing max C_k is maximizing min r_k
- r_k = sum_{i=k}^{N-1} h_i h_{i-k}

We can use a max-min formulation and optimize with a different method:
- Use a trust-region or interior-point method
- Or use a different smoothing: instead of log-sum-exp on C_k, use
  a polynomial smoothing or a different convex relaxation

Let me try a completely different approach: 
- Parametrize h as h_i = 0.5 + 0.5 * x_i where x_i in [-1,1] and sum(x_i) = 0
- Then r_k = sum_{i=k}^{N-1} (0.5+0.5 x_i)(0.5+0.5 x_{i-k})
         = (1/4) sum (1 + x_i + x_{i-k} + x_i x_{i-k})
         = (1/4)((N-k) + sum_{i=k}^{N-1} x_i + sum_{i=k}^{N-1} x_{i-k} + sum_{i=k}^{N-1} x_i x_{i-k})

This gets complicated. Let me try a cleaner structural change.

NEW APPROACH: Use alternating minimization between the time domain and the
frequency domain.

1. In the frequency domain, the overlap is determined by the power spectrum.
2. We can optimize the power spectrum to be as flat as possible (which minimizes
   the peak of the inverse transform, i.e., the autocorrelation variation).
3. Then recover h from the power spectrum.

The flat power spectrum corresponds to h being close to white noise, but we need
h in [0,1] with a specific sum.

Let me implement a practical approach:
1. Start with a good initial h (cosine wave)
2. Use a different optimization: alternating between
   a) Projecting onto the set {h: sum(h)=N/2, 0<=h<=1}
   b) Taking a step in the direction that reduces max C_k

For (b), instead of the log-sum-exp smoothing, use a different approach:
- Identify the active constraints (k values where C_k is near the max)
- Take a step that reduces those specific C_k values

This is essentially an active-set method for the max problem.
"""
import time
import numpy as np
from scipy.optimize import minimize, linprog

N = 100
_DEADLINE = time.time() + 140


def _linear_overlap_values(h):
    """Compute linear overlap values C_k for k=0..N-1 using FFT."""
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    """Compute the maximum linear overlap."""
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _fix_sum(h):
    """Adjust h to have sum exactly N/2, keeping values in [0, 1]."""
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
    return h


def _active_set_step(h, k_active, step_size):
    """
    Take a step to reduce C_k for k in k_active.
    C_k = (2/N)(N/2 - r_k)
    dC_k/dh_j = -(2/N)(h_{j-k} + h_{j+k}) for valid indices
    
    We want to decrease C_k, so we need to increase r_k.
    dr_k/dh_j = h_{j-k} (if j>=k) + h_{j+k} (if j+k<n)
    
    So the gradient direction to increase r_k is:
    For each j: grad_j = h_{j-k} (if j>=k) + h_{j+k} (if j+k<n)
    
    We take a step in this direction (increasing r_k, decreasing C_k).
    """
    n = len(h)
    grad = np.zeros(n)
    for k in k_active:
        if k > 0:
            grad[k:] += h[:n-k]
        grad[:n-k] += h[k:]
    
    # Normalize gradient
    norm = np.linalg.norm(grad)
    if norm < 1e-15:
        return h.copy()
    grad /= norm
    
    h_new = h + step_size * grad
    return _fix_sum(h_new)


def _active_set_optimize(h0, max_iter=500, tol=1e-10):
    """
    Active-set optimization: iteratively identify the active constraints
    (k values where C_k is near the max) and take steps to reduce them.
    """
    h = h0.copy()
    best_h = h.copy()
    best_score = _score(h)
    
    for it in range(max_iter):
        if time.time() - _DEADLINE > 10:
            break
        
        cs = _linear_overlap_values(h)
        current_max = np.max(cs)
        
        if current_max < best_score:
            best_score = current_max
            best_h = h.copy()
        
        # Find active set: k values where C_k is within tol of the max
        threshold = current_max - 0.05
        active = np.where(cs >= threshold)[0]
        
        if len(active) == 0:
            active = np.array([np.argmax(cs)])
        
        # Adaptive step size
        step_size = 0.1 / (1.0 + it * 0.01)
        
        h_new = _active_set_step(h, active, step_size)
        
        # Check if we improved
        new_score = _score(h_new)
        if new_score < current_max - 1e-12:
            h = h_new
        else:
            # Try smaller step
            step_size /= 4
            h_new = _active_set_step(h, active, step_size)
            new_score = _score(h_new)
            if new_score < current_max - 1e-12:
                h = h_new
            else:
                # No improvement, reduce active set to just the max
                active = np.array([np.argmax(cs)])
                step_size = 0.05
                h_new = _active_set_step(h, active, step_size)
                new_score = _score(h_new)
                if new_score < current_max - 1e-12:
                    h = h_new
                else:
                    break
        
        if current_max - best_score < tol:
            break
    
    return best_h, best_score


def _frequency_domain_optimize(h0, max_iter=300):
    """
    Frequency-domain optimization: optimize the power spectrum to be flatter.
    
    The autocorrelation r_k is the inverse FFT of the power spectrum P_k = |H_k|^2.
    We want r_k to be as close to N/2 as possible for all k.
    
    Approach: 
    1. Compute H = FFT(h)
    2. Modify P = |H|^2 to be flatter (reduce variance)
    3. Recover h from the modified P (using the original phase)
    4. Project back to [0,1] with sum constraint
    """
    h = h0.copy()
    best_h = h.copy()
    best_score = _score(h)
    
    for it in range(max_iter):
        if time.time() - _DEADLINE > 10:
            break
        
        n_fft = 2 * N
        h_pad = np.zeros(n_fft)
        h_pad[:N] = h
        H = np.fft.rfft(h_pad, n=n_fft)
        
        # Power spectrum
        P = np.abs(H) ** 2
        
        # We want to flatten P (make it more uniform) to make r_k more uniform
        # Target: P_k = P_0 / (N/2) for all k (uniform power)
        # But we need to preserve P_0 = (sum h)^2 = (N/2)^2
        
        P_0 = P[0]
        target_P = P_0 / (n_fft / 2.0)  # Average power level
        
        # Blend current P with target P
        alpha = 0.1 / (1.0 + it * 0.005)
        P_new = (1.0 - alpha) * P + alpha * target_P
        P_new[0] = P_0  # Preserve DC
        
        # Recover H from P_new, preserving phase
        phase = np.angle(H)
        H_new = np.sqrt(P_new) * np.exp(1j * phase)
        
        # Inverse FFT
        h_new = np.fft.irfft(H_new, n=n_fft)[:N]
        h_new = _fix_sum(h_new)
        
        new_score = _score(h_new)
        if new_score < best_score:
            best_score = new_score
            best_h = h_new.copy()
            h = h_new
        else:
            # Try smaller alpha
            alpha /= 4
            P_new = (1.0 - alpha) * P + alpha * target_P
            P_new[0] = P_0
            H_new = np.sqrt(P_new) * np.exp(1j * phase)
            h_new = np.fft.irfft(H_new, n=n_fft)[:N]
            h_new = _fix_sum(h_new)
            new_score = _score(h_new)
            if new_score < best_score:
                best_score = new_score
                best_h = h_new.copy()
                h = h_new
            else:
                break
    
    return best_h, best_score


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # 1. Analytical Start: Cosine Wave
    i = np.arange(N)
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    h_cos = _fix_sum(h_cos)
    _update_best(h_cos)
    
    # 2. Active-set optimization from cosine start
    if time.time() - t0 < _DEADLINE - 40:
        h_as, s_as = _active_set_optimize(h_cos, max_iter=500)
        _update_best(h_as)
    
    # 3. Frequency-domain optimization
    if time.time() - t0 < _DEADLINE - 30:
        h_fd, s_fd = _frequency_domain_optimize(h_cos, max_iter=300)
        _update_best(h_fd)
    
    # 4. Combine: active-set on frequency-domain result
    if time.time() - t0 < _DEADLINE - 25:
        if best_h is not None:
            h_as2, s_as2 = _active_set_optimize(best_h, max_iter=400)
            _update_best(h_as2)
    
    # 5. Try diverse structured starts with active-set
    if time.time() - t0 < _DEADLINE - 20:
        h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
        h_sin = _fix_sum(h_sin)
        h_as3, s_as3 = _active_set_optimize(h_sin, max_iter=300)
        _update_best(h_as3)
    
    if time.time() - t0 < _DEADLINE - 15:
        h_tri = 0.5 + 0.5 * np.abs(2.0 * ((i / N) % 1.0) - 1.0)
        h_tri = _fix_sum(h_tri)
        h_as4, s_as4 = _active_set_optimize(h_tri, max_iter=300)
        _update_best(h_as4)
    
    # 6. Random starts with active-set
    if time.time() - t0 < _DEADLINE - 10:
        for _ in range(3):
            if time.time() - t0 > _DEADLINE - 5:
                break
            h_rand = np.random.uniform(0, 1, N)
            h_rand = _fix_sum(h_rand)
            h_as5, s_as5 = _active_set_optimize(h_rand, max_iter=200)
            _update_best(h_as5)
    
    # 7. Final polish: frequency-domain + active-set on best
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        h_fd2, s_fd2 = _frequency_domain_optimize(best_h, max_iter=200)
        _update_best(h_fd2)
    
    if best_h is not None and time.time() - t0 < _DEADLINE - 2:
        h_as_final, s_as_final = _active_set_optimize(best_h, max_iter=300)
        _update_best(h_as_final)
    
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _fix_sum(best_h)
    return best_h
