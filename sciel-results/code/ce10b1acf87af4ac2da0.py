"""
Strategy: Global Random Search over Low-Dimensional Parametric Families
Instead of gradient-based local optimization in 400D, we search over a small set
of parametric families (smoothed step, polynomial, cosine-modulated) with a global
random search. The key insight is that the objective is scale-invariant, so only
the shape of h matters. We vectorize the evaluation to test thousands of candidates
quickly, avoiding the local minima trap of gradient methods.

Families tried:
1. Smoothed step: h[i] = tanh(k * (x - c)) with parameters k (sharpness), c (center)
2. Polynomial: h[i] = a0 + a1*x + a2*x^2 + a3*x^3 (with x in [-1/4, 1/4])
3. Cosine-modulated: h[i] = A*cos(w*x + phi) + B
4. Gaussian window: h[i] = exp(-((x-c)^2)/(2*sigma^2)) - offset
5. Multi-scale: combination of two smoothed steps
"""

import numpy as np
import time

N = 400
_DEADLINE = 140.0  # safety margin from 150s limit

def _evaluate(h):
    """Compute the task value."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _evaluate_batch(h_batch):
    """
    Vectorized evaluation of multiple candidates.
    h_batch: shape (M, N)
    Returns: array of M values
    """
    M = h_batch.shape[0]
    s = np.sum(h_batch, axis=1)  # (M,)
    # Avoid division by zero
    valid = np.abs(s) > 1e-12
    values = np.full(M, np.inf)
    
    if not np.any(valid):
        return values
    
    # For each valid candidate, compute convolve and max
    # We can't fully vectorize convolve, so we loop but with numpy ops inside
    for i in np.where(valid)[0]:
        h = h_batch[i]
        conv = np.convolve(h, h)
        max_conv = np.max(conv)
        values[i] = abs(2.0 * N * max_conv / (s[i] * s[i]))
    
    return values

def _make_smoothed_step(k, c, x):
    """h[i] = tanh(k * (x[i] - c))"""
    return np.tanh(k * (x - c))

def _make_polynomial(a0, a1, a2, a3, x):
    """h[i] = a0 + a1*x + a2*x^2 + a3*x^3"""
    return a0 + a1 * x + a2 * x * x + a3 * x * x * x

def _make_cosine(A, w, phi, B, x):
    """h[i] = A*cos(w*x + phi) + B"""
    return A * np.cos(w * x + phi) + B

def _make_gaussian(c, sigma, offset, x):
    """h[i] = exp(-((x-c)^2)/(2*sigma^2)) - offset"""
    return np.exp(-((x - c) ** 2) / (2 * sigma ** 2)) - offset

def _make_multi_step(k1, c1, k2, c2, x):
    """h[i] = tanh(k1*(x-c1)) + tanh(k2*(x-c2))"""
    return np.tanh(k1 * (x - c1)) + np.tanh(k2 * (x - c2))

def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    x = np.linspace(-0.25, 0.25, N)
    
    # Seed baseline
    best_h = np.ones(N)
    best_val = 2.0
    
    rng = np.random.default_rng(12345)
    
    def try_candidate(h):
        nonlocal best_h, best_val
        val = _evaluate(h)
        if np.isfinite(val) and val < best_val:
            best_val = val
            best_h = h.copy()
        return val
    
    # Family 1: Smoothed step - dense random search
    # Parameters: k in [1, 50], c in [-0.25, 0.25]
    n_trials_f1 = 5000
    for _ in range(n_trials_f1):
        if time.time() > deadline:
            break
        k = rng.uniform(1, 50)
        c = rng.uniform(-0.25, 0.25)
        h = _make_smoothed_step(k, c, x)
        try_candidate(h)
    
    # Family 2: Polynomial - random search
    # Parameters: a0 in [-1,1], a1 in [-10,10], a2 in [-10,10], a3 in [-10,10]
    n_trials_f2 = 3000
    for _ in range(n_trials_f2):
        if time.time() > deadline:
            break
        a0 = rng.uniform(-1, 1)
        a1 = rng.uniform(-10, 10)
        a2 = rng.uniform(-10, 10)
        a3 = rng.uniform(-10, 10)
        h = _make_polynomial(a0, a1, a2, a3, x)
        try_candidate(h)
    
    # Family 3: Cosine-modulated
    # Parameters: A in [0.1, 5], w in [0, 20], phi in [0, 2pi], B in [-1, 1]
    n_trials_f3 = 3000
    for _ in range(n_trials_f3):
        if time.time() > deadline:
            break
        A = rng.uniform(0.1, 5)
        w = rng.uniform(0, 20)
        phi = rng.uniform(0, 2 * np.pi)
        B = rng.uniform(-1, 1)
        h = _make_cosine(A, w, phi, B, x)
        try_candidate(h)
    
    # Family 4: Gaussian window
    # Parameters: c in [-0.25, 0.25], sigma in [0.01, 0.5], offset in [-1, 1]
    n_trials_f4 = 3000
    for _ in range(n_trials_f4):
        if time.time() > deadline:
            break
        c = rng.uniform(-0.25, 0.25)
        sigma = rng.uniform(0.01, 0.5)
        offset = rng.uniform(-1, 1)
        h = _make_gaussian(c, sigma, offset, x)
        try_candidate(h)
    
    # Family 5: Multi-step
    # Parameters: k1, k2 in [1, 30], c1, c2 in [-0.25, 0.25]
    n_trials_f5 = 3000
    for _ in range(n_trials_f5):
        if time.time() > deadline:
            break
        k1 = rng.uniform(1, 30)
        c1 = rng.uniform(-0.25, 0.25)
        k2 = rng.uniform(1, 30)
        c2 = rng.uniform(-0.25, 0.25)
        h = _make_multi_step(k1, c1, k2, c2, x)
        try_candidate(h)
    
    # Family 6: Fourier-based - sum of a few sinusoids with random phases
    # h[i] = sum_{j=1}^{K} A_j * cos(w_j * x + phi_j)
    n_trials_f6 = 3000
    for _ in range(n_trials_f6):
        if time.time() > deadline:
            break
        K = rng.integers(2, 6)
        h = np.zeros(N)
        for j in range(K):
            A = rng.uniform(0.1, 2)
            w = rng.uniform(1, 15)
            phi = rng.uniform(0, 2 * np.pi)
            h += A * np.cos(w * x + phi)
        # Add a DC offset to ensure sum != 0
        h += rng.uniform(-0.5, 0.5)
        try_candidate(h)
    
    # Family 7: Smoothed sign with random asymmetry
    # h[i] = sign(x) * exp(-alpha * |x|) + beta * x
    n_trials_f7 = 2000
    for _ in range(n_trials_f7):
        if time.time() > deadline:
            break
        alpha = rng.uniform(0, 20)
        beta = rng.uniform(-5, 5)
        h = np.sign(x) * np.exp(-alpha * np.abs(x)) + beta * x
        try_candidate(h)
    
    # Final refinement: local random perturbation around best found
    if best_val < 2.0 and time.time() < deadline - 5:
        # Do some local search around the best candidate
        for _ in range(2000):
            if time.time() > deadline:
                break
            h_perturbed = best_h + rng.normal(0, 0.05, N)
            val = _evaluate(h_perturbed)
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_h = h_perturbed.copy()
    
    # Final safety check
    final_val = _evaluate(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        best_h = np.ones(N)
    
    return best_h
