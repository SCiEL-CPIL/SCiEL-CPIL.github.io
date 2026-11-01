"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _autocorr_sidelobe_max(x):
    """Compute the maximum absolute value of the autocorrelation sidelobes."""
    n = len(x)
    f = np.fft.fft(x, n=2*n)
    acf = np.fft.ifft(f * np.conj(f)).real[:n]
    return np.max(np.abs(acf[1:]))


def _softmax_sidelobe(x, beta=50.0):
    """
    Differentiable approximation to max |sidelobes| using log-sum-exp.
    Returns (objective, gradient).
    """
    n = len(x)
    f = np.fft.fft(x, n=2*n)
    power = f * np.conj(f)
    acf = np.fft.ifft(power).real[:n]
    sidelobes = acf[1:n]
    abs_sidelobes = np.abs(sidelobes)
    
    # Log-sum-exp for numerical stability
    m = np.max(abs_sidelobes)
    if m < 1e-15:
        return 0.0, np.zeros(n)
    log_sum = m + np.log(np.sum(np.exp(beta * (abs_sidelobes - m))))
    obj = log_sum / beta
    
    # Gradient: d/dx of soft-max
    weights = np.exp(beta * (abs_sidelobes - m))
    weights /= np.sum(weights)
    
    # d|sidelobes[k]|/dx = sign(sidelobes[k]) * dsidelobes[k]/dx
    # dsidelobes[k]/dx is the autocorrelation derivative
    grad = np.zeros(n)
    for k in range(1, n):
        c_k = sidelobes[k-1]
        if abs(c_k) < 1e-15:
            continue
        sign_c = np.sign(c_k)
        w = weights[k-1]
        # d acf[k] / dx[i] = x[i+k] + x[i-k] (for valid indices)
        grad[0:n-k] += 2.0 * w * sign_c * x[k:n]
        grad[k:n] += 2.0 * w * sign_c * x[0:n-k]
    
    return obj, grad


def _walsh_hadamard_init(n, seed=42):
    """
    Generate initial sequence using Walsh-Hadamard basis with sparse coefficients.
    This exploits the flat spectral properties to create low-autocorrelation sequences.
    """
    rng = np.random.RandomState(seed)
    
    # Build Hadamard matrix (Walsh-Hadamard)
    def hadamard(n):
        if n == 1:
            return np.array([[1.0]])
        H = hadamard(n // 2)
        return np.block([[H, H], [H, -H]])
    
    # Use next power of 2 >= n for Hadamard, then truncate
    h = 1
    while h < n:
        h *= 2
    H = hadamard(h)
    H = H[:n, :n] / np.sqrt(n)
    
    # Sparse coefficient selection
    n_sparse = max(1, n // 8)
    indices = rng.choice(n, size=n_sparse, replace=False)
    coeffs = rng.choice([-1.0, 1.0], size=n_sparse)
    
    x = np.zeros(n)
    for i, c in zip(indices, coeffs):
        x += c * H[:, i]
    
    # Normalize to ±1
    x = np.sign(x)
    x[x == 0] = 1.0
    return x


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    np.random.seed(42)
    
    # Multiple initialization strategies
    inits = []
    inits.append(np.random.choice([-1.0, 1.0], size=N))
    inits.append(_walsh_hadamard_init(N, seed=42))
    inits.append(_walsh_hadamard_init(N, seed=123))
    inits.append(_walsh_hadamard_init(N, seed=999))
    
    # Evaluate all inits
    best_x = None
    best_score = np.inf
    for x_init in inits:
        s = _autocorr_sidelobe_max(x_init)
        if s < best_score:
            best_score = s
            best_x = x_init.copy()
    
    # Phase 1: Continuous optimization with soft-max objective
    # Multiple restarts from different initial points
    best_x_cont = best_x.copy()
    best_score_cont = best_score
    
    n_restarts = 6
    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        
        # Choose initial point
        if restart == 0:
            x_start_base = best_x
        elif restart == 1:
            x_start_base = inits[1]
        elif restart == 2:
            x_start_base = inits[2]
        elif restart == 3:
            x_start_base = inits[3]
        else:
            x_start_base = best_x + np.random.normal(0, 0.3, N)
        
        # Map to continuous space: arctanh(0.999 * x)
        x_cont = np.arctanh(0.999 * np.clip(x_start_base, -0.999, 0.999))
        
        # Add small perturbation for restarts > 3
        if restart > 3:
            x_cont += np.random.normal(0, 0.5, N)
        
        def _obj(xc):
            h = np.tanh(xc)
            obj, _ = _softmax_sidelobe(h)
            return obj
        
        def _grad(xc):
            h = np.tanh(xc)
            obj, grad_h = _softmax_sidelobe(h)
            grad_xc = grad_h * (1 - h**2)
            return grad_xc
        
        result = minimize(
            _obj,
            x_cont,
            jac=_grad,
            method='L-BFGS-B',
            options={'maxiter': 500, 'gtol': 1e-5, 'maxfun': 2000}
        )
        
        # Project back to ±1
        h_opt = np.tanh(result.x)
        x_binary = np.sign(h_opt)
        x_binary[x_binary == 0] = 1.0
        
        score_opt = _autocorr_sidelobe_max(x_binary)
        if score_opt < best_score_cont:
            best_score_cont = score_opt
            best_x_cont = x_binary.copy()
    
    # Phase 2: Discrete refinement (single flips)
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        for idx in range(N):
            old_val = best_x_cont[idx]
            best_x_cont[idx] = -old_val
            new_score = _autocorr_sidelobe_max(best_x_cont)
            if new_score < best_score_cont:
                best_score_cont = new_score
                improved = True
            else:
                best_x_cont[idx] = old_val
    
    return best_x_cont
# EVOLVE-BLOCK-END
