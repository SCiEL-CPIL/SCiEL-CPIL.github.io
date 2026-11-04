"""Rewritten program for the third autocorrelation inequality task."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_DEADLINE = 140.0  # wall-clock limit with safety margin


def _evaluate(h):
    """Compute the task value: abs(2 * N * max(convolve(h,h)) / sum(h)^2)."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def _softmax_value(h, beta=10.0):
    """
    Surrogate objective: 2*N * softmax(max(conv)) / sum(h)^2
    This is differentiable and close to the true max when beta is large.
    """
    conv = np.convolve(h, h)
    # Use a soft-max approximation
    m = np.max(conv)
    # For numerical stability
    conv_shifted = conv - m
    weights = np.exp(beta * conv_shifted)
    softmax_val = m + np.log(np.sum(weights)) / beta
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf, np.zeros_like(h)
    val = 2.0 * N * softmax_val / (s * s)
    
    # Gradient
    # d(softmax_val)/d(conv_k) = weights[k] / sum(weights)
    # d(conv_k)/d(h_j) is complex, use numerical gradient for reliability
    # Actually, let's use a simpler approach: just use the true max with a perturbation
    # For L-BFGS-B, we need a smooth function. Let's use the max of conv with a small regularization
    
    # Better approach: use the fact that we want to minimize the maximum convolution entry
    # We can use a trust-region approach or just use the true value with subgradient
    # For simplicity, use the true value and let L-BFGS-B handle it with finite differences
    
    return val, None


def _true_value_and_gradient(h):
    """
    Return the true value and a numerical gradient approximation.
    """
    val = _evaluate(h)
    if not np.isfinite(val):
        return val, None
    
    # Numerical gradient
    eps = 1e-8
    grad = np.zeros(N)
    for i in range(N):
        h_plus = h.copy()
        h_plus[i] += eps
        v_plus = _evaluate(h_plus)
        if not np.isfinite(v_plus):
            v_plus = val + 1.0
        grad[i] = (v_plus - val) / eps
    
    return val, grad


def _parametric_family(freqs, amps, phase, offset):
    """
    Generate a structured candidate: offset + sum(amps[k] * cos(2*pi*freqs[k]*n/N + phase))
    This ensures sum(h) can be controlled via offset.
    """
    n = np.arange(N)
    h = np.full(N, offset)
    for f, a in zip(freqs, amps):
        h += a * np.cos(2 * np.pi * f * n / N + phase)
    return h


def _optimize_with_lbgfgs(h_init, deadline):
    """
    Use L-BFGS-B to minimize the surrogate objective.
    """
    def objective(x):
        if time.time() > deadline:
            raise ValueError("Timeout")
        h = x
        s = np.sum(h)
        if s == 0 or not np.isfinite(s):
            return 1e10
        conv = np.convolve(h, h)
        # Use a smooth approximation of max
        beta = 50.0
        m = np.max(conv)
        conv_shifted = conv - m
        weights = np.exp(beta * conv_shifted)
        soft_max = m + np.log(np.sum(weights)) / beta
        val = 2.0 * N * soft_max / (s * s)
        return val
    
    try:
        res = minimize(
            objective,
            h_init,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-7, 'maxfun': 500}
        )
        h_opt = res.x
        if not np.all(np.isfinite(h_opt)):
            return h_init
        return h_opt
    except Exception:
        return h_init


def _refine_with_greedy(h, deadline):
    """
    Local greedy refinement: try small perturbations and keep improvements.
    """
    best_val = _evaluate(h)
    best_h = h.copy()
    rng = np.random.default_rng(123)
    
    for step in range(50000):
        if time.time() > deadline:
            break
        
        # Pick 1-3 indices to perturb
        n_perturb = rng.integers(1, 4)
        idx = rng.choice(N, size=n_perturb, replace=False)
        h_new = best_h.copy()
        # Random small perturbations
        h_new[idx] += rng.normal(0, 0.05, n_perturb)
        
        v_new = _evaluate(h_new)
        if np.isfinite(v_new) and v_new < best_val - 1e-12:
            best_val = v_new
            best_h = h_new
    
    return best_h


def construct():
    """
    Construct step heights to minimize the autocorrelation inequality value.
    """
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    # Start with the seed
    best_h = np.ones(N)
    best_val = 2.0
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Generate diverse initial candidates
    inits = [best_h.copy()]
    
    # Random vectors
    for _ in range(10):
        h_rand = rng.normal(0, 1, N)
        inits.append(h_rand)
    
    # Structured low-frequency patterns
    for _ in range(20):
        n_freqs = rng.integers(1, 4)
        freqs = rng.uniform(1, 10, size=n_freqs)
        amps = rng.uniform(0.1, 1.0, size=n_freqs)
        phase = rng.uniform(0, 2 * np.pi)
        offset = 1.0  # Ensure positive mean
        h_struct = _parametric_family(freqs, amps, phase, offset)
        inits.append(h_struct)
    
    # Sparse patterns
    for _ in range(10):
        h_sparse = np.ones(N)
        n_zero = rng.integers(10, 50)
        idx_zero = rng.choice(N, size=n_zero, replace=False)
        h_sparse[idx_zero] = 0.0
        inits.append(h_sparse)
    
    # Block patterns
    for _ in range(10):
        h_block = np.ones(N)
        n_blocks = rng.integers(4, 20)
        block_size = N // n_blocks
        for i in range(0, N, block_size):
            val = rng.uniform(0.5, 1.5)
            h_block[i:min(i + block_size, N)] = val
        inits.append(h_block)
    
    # Phase 2: Optimize each candidate with L-BFGS-B
    for h_init in inits:
        if time.time() > deadline - 5:
            break
        
        # Ensure finite and non-zero sum
        if not np.all(np.isfinite(h_init)) or np.sum(h_init) == 0:
            continue
        
        # Quick L-BFGS-B optimization
        h_opt = _optimize_with_lbgfgs(h_init, deadline - 5)
        
        if not np.all(np.isfinite(h_opt)):
            continue
        
        v_opt = _evaluate(h_opt)
        if np.isfinite(v_opt) and v_opt < best_val:
            best_val = v_opt
            best_h = h_opt.copy()
    
    # Phase 3: Local refinement of the best candidate
    if time.time() < deadline - 5:
        best_h = _refine_with_greedy(best_h, deadline - 5)
        best_val = _evaluate(best_h)
    
    # Phase 4: More structured search with different parameterizations
    if time.time() < deadline - 10:
        # Try to find a pattern where the autoconvolution is as flat as possible
        # Theoretical lower bound analysis:
        # value = 2*N*max(conv)/sum^2
        # For constant h=c: conv[k] = c^2*(N-|k-N+1|) for |k-N+1| < N, else 0
        # max(conv) = c^2*N, sum = c*N, so value = 2*N*c^2*N/(c^2*N^2) = 2
        # To go below 2, we need max(conv)/sum^2 < 1/N
        # This requires the autoconvolution to be very flat
        # 
        # Try h with alternating signs in a structured way
        # h = [1, -1, 1, -1, ...] gives conv with peaks at even lags
        # 
        # Let's try h = cos(2*pi*n/N + phase) + offset
        # and optimize offset and phase
        
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            
            freq = rng.uniform(1, 5)
            phase = rng.uniform(0, 2 * np.pi)
            offset = rng.uniform(0.5, 2.0)
            n = np.arange(N)
            h_test = offset + np.cos(2 * np.pi * freq * n / N + phase)
            
            # Ensure no zeros in sum
            if np.sum(h_test) == 0:
                continue
            
            h_opt = _optimize_with_lbgfgs(h_test, deadline - 5)
            if not np.all(np.isfinite(h_opt)):
                continue
            
            v_opt = _evaluate(h_opt)
            if np.isfinite(v_opt) and v_opt < best_val:
                best_val = v_opt
                best_h = h_opt.copy()
    
    # Final check
    final_val = _evaluate(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        # Fall back to seed if optimization failed
        best_h = np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
