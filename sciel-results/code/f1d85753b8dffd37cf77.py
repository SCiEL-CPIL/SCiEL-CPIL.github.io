import numpy as np
import time
import scipy.optimize

N = 400

def _autoconv_lin(x):
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    return np.fft.irfft(fx * fx, size)[:2 * n - 1]

def _metric_std(x):
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv_lin(x)
    return float(np.max(np.abs(c))) / (s * s)

def _metric_offdiag(x):
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv_lin(x)
    if len(c) > 1:
        return float(np.max(np.abs(c[1:]))) / (s * s)
    return 0.0

def _soft_max_abs(c, beta):
    """Smooth approximation to max|c| using log-sum-exp."""
    abs_c = np.abs(c)
    if len(abs_c) == 0:
        return 0.0
    m = np.max(abs_c)
    if m < 1e-15:
        return 0.0
    return m + np.log(np.sum(np.exp(beta * (abs_c - m)))) / beta

def _objective_soft(x, beta, deadline):
    """Objective function for L-BFGS-B with time guard."""
    if time.time() > deadline:
        return 1e10
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    c = _autoconv_lin(x)
    if len(c) <= 1:
        return 0.0
    abs_c = np.abs(c[1:])
    soft_max = _soft_max_abs(abs_c, beta)
    return soft_max / (s * s)

def _generate_random_phase_signal(n, seed):
    """Generate a signal with flat spectrum and random phases."""
    rng = np.random.default_rng(seed)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    phases = rng.uniform(-np.pi, np.pi, size // 2 + 1)
    mag = np.ones(size // 2 + 1)
    spectrum = mag * np.exp(1j * phases)
    spectrum[0] = mag[0]  # DC real
    if size % 2 == 0:
        spectrum[-1] = mag[-1]  # Nyquist real
    x_full = np.fft.irfft(spectrum, size)
    x = x_full[:n]
    return x

def _generate_sine_sum(n, K, seed):
    """Generate a signal as sum of K random sines."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    x = np.zeros(n)
    for _ in range(K):
        f = rng.uniform(1, n/2)
        p = rng.uniform(0, 2*np.pi)
        a = rng.uniform(0.5, 1.5)
        x += a * np.sin(2 * np.pi * f * t / n + p)
    return x

def _refine_lbfgs(x_init, deadline, beta_start=10.0, beta_end=100.0, n_anneal_steps=5):
    """
    Refine x_init using L-BFGS-B with annealed soft-max beta.
    """
    x = x_init.copy()
    current_beta = beta_start
    
    for step in range(n_anneal_steps):
        if time.time() > deadline - 2.0:
            break
        
        t_anneal = step / max(1, n_anneal_steps - 1)
        beta = beta_start + (beta_end - beta_start) * t_anneal
        
        def obj(x):
            return _objective_soft(x, beta, deadline)
        
        try:
            result = scipy.optimize.minimize(
                obj,
                x,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10},
                tol=1e-10
            )
            x = result.x
        except Exception:
            pass

    return x

def _polish_subgradient(x, deadline, max_iters=50):
    """
    Polish using subgradient descent on the exact max |c_k|.
    """
    x = x.copy()
    best_x = x.copy()
    best_score = _metric_offdiag(x)
    
    for i in range(max_iters):
        if time.time() > deadline - 1.0:
            break
            
        s = np.sum(x)
        if abs(s) < 1e-10:
            break
            
        c = _autoconv_lin(x)
        abs_c = np.abs(c[1:])
        
        # Find active lags (top 10%)
        if len(abs_c) > 0:
            threshold = np.percentile(abs_c, 90)
        else:
            threshold = 0
        active_mask = abs_c > threshold
        if not np.any(active_mask):
            break
            
        # Subgradient: sum of gradients of active terms
        # d(c_k)/dx_j = x_{j+k} + x_{j-k}
        grad = np.zeros_like(x)
        active_indices = np.where(active_mask)[0]
        
        for k_idx in active_indices:
            k = k_idx + 1
            sign_c = np.sign(c[k])
            if sign_c == 0:
                sign_c = 1.0
            
            # x[j+k]
            if k < N:
                grad[:N-k] += sign_c * x[k:]
            # x[j-k]
            if k < N:
                grad[k:] += sign_c * x[:N-k]
        
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-15:
            break
            
        # Step size
        step_size = 0.01 * (1.0 - i/max_iters) + 0.001
        x_new = x - step_size * grad / gnorm
        
        sc_new = _metric_offdiag(x_new)
        
        if sc_new < best_score:
            best_score = sc_new
            best_x = x_new.copy()
            x = x_new
        else:
            step_size *= 0.5
            x_new = x - step_size * grad / gnorm
            sc_new = _metric_offdiag(x_new)
            if sc_new < best_score:
                best_score = sc_new
                best_x = x_new.copy()
                x = x_new

    return best_x, best_score

def _cellular_automaton_trajectory(rule, init_state, steps):
    """
    Generate a trajectory from a 1D elementary cellular automaton.
    rule: integer 0-255
    init_state: binary array of length N
    steps: number of steps to evolve
    Returns: array of length N (the final state)
    """
    state = init_state.copy()
    # Map rule to lookup table for 8 possible neighbor configurations
    # Neighborhoods: (left, center, right) -> 0 or 1
    # Since we are on a finite ring, we handle boundaries with wrap-around
    for _ in range(steps):
        new_state = np.zeros_like(state)
        for i in range(len(state)):
            left = state[(i - 1) % len(state)]
            center = state[i]
            right = state[(i + 1) % len(state)]
            # Configuration index: left*4 + center*2 + right
            config = int(left * 4 + center * 2 + right)
            new_state[i] = (rule >> config) & 1
        state = new_state
    return state

def construct():
    deadline = time.time() + 140.0
    
    best_x = None
    best_score = float('inf')
    
    def update_best(x, score):
        nonlocal best_x, best_score
        if x is not None and score < best_score:
            best_score = score
            best_x = x.copy()
            
    # Phase 1: Generate diverse initial candidates
    candidates = []
    
    # 1. Random phase flat spectrum signals (good spectral spread)
    for seed in [1, 2, 3, 4, 5]:
        x = _generate_random_phase_signal(N, seed)
        candidates.append(x)
        
    # 2. Sum of sines
    for seed in [10, 11, 12]:
        x = _generate_sine_sum(N, K=5, seed=seed)
        candidates.append(x)
        
    # 3. Random noise
    rng = np.random.default_rng(42)
    candidates.append(rng.normal(0, 1, N))
    
    # 4. Binary +/-1 (Barker-like start)
    candidates.append(rng.choice([-1, 1], N))
    
    # 5. Cellular Automaton Trajectories (Structural Change)
    # CA rules known for complex behavior (e.g., Rule 30, 90, 110)
    ca_rules = [30, 90, 110, 150, 184, 225]
    for rule in ca_rules:
        # Random initial state
        init_state = rng.integers(0, 2, N).astype(float)
        # Evolve for a few steps
        steps = rng.integers(5, 20)
        x_ca = _cellular_automaton_trajectory(rule, init_state, steps)
        # Convert binary {0,1} to signed {-1,1} or keep as is?
        # The metric is scale invariant but not sign invariant in a simple way.
        # Let's try signed version: 2*x - 1
        x_signed = 2.0 * x_ca - 1.0
        candidates.append(x_signed)
        
    # Phase 2: Refine each candidate
    for i, x_init in enumerate(candidates):
        if time.time() > deadline - 15.0:
            break
            
        # Quick score check
        sc_init = _metric_offdiag(x_init)
        update_best(x_init, sc_init)
        
        # Refine with L-BFGS
        x_ref = _refine_lbfgs(x_init, deadline)
        sc_ref = _metric_offdiag(x_ref)
        update_best(x_ref, sc_ref)
        
        # Polish with subgradient
        if time.time() < deadline - 5.0:
            x_pol, sc_pol = _polish_subgradient(x_ref, deadline)
            update_best(x_pol, sc_pol)
            
    # If we have time, do a final polish on the best one
    if best_x is not None and time.time() < deadline - 2.0:
        x_final, sc_final = _polish_subgradient(best_x, deadline, max_iters=20)
        update_best(x_final, sc_final)
        
    if best_x is None:
        return np.ones(N)
        
    return best_x
