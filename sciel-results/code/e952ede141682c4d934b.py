"""Vectorized QAOA with analytical gradient, multi-restart Nelder-Mead, and high-precision polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 140.0


def _time_left():
    return _DEADLINE - time.time()


def _precompute_cut_sizes(n, edges):
    """Precompute cut size for each basis state."""
    dim = 1 << n
    cut_sizes = np.zeros(dim, dtype=np.float64)
    for (i, j) in edges:
        mask_i = 1 << i
        mask_j = 1 << j
        for state in range(dim):
            bi = (state >> i) & 1
            bj = (state >> j) & 1
            if bi != bj:
                cut_sizes[state] += 1.0
    return cut_sizes


def _apply_x_layer(state, beta):
    """Apply exp(-i beta sum_i X_i) to state vector efficiently."""
    dim = len(state)
    n = int(np.log2(dim))
    c = np.cos(beta)
    s = np.sin(beta)
    
    new_state = state.copy()
    for i in range(n):
        mask = 1 << i
        for state_idx in range(dim):
            if (state_idx & mask) == 0:
                partner = state_idx | mask
                a = new_state[state_idx]
                b = new_state[partner]
                new_state[state_idx] = c * a - 1j * s * b
                new_state[partner] = c * b - 1j * s * a
    return new_state


def _qaoa_state(n, edges, cut_sizes, angles):
    """Simulate QAOA circuit to get final state."""
    dim = 1 << n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    for gamma, beta in angles:
        # Apply cost layer: exp(-i gamma C)
        phase = np.exp(-1j * gamma * cut_sizes)
        state = state * phase
        # Apply X layer
        state = _apply_x_layer(state, beta)
    
    return state


def _expected_cut(state, cut_sizes):
    """Compute expected cut value."""
    probs = np.abs(state) ** 2
    return float(np.real(np.sum(probs * cut_sizes)))


def _objective(params, n, edges, cut_sizes, p):
    """Objective: minimize negative expected cut."""
    angles = params.reshape(p, 2)
    state = _qaoa_state(n, edges, cut_sizes, angles)
    return -_expected_cut(state, cut_sizes)


def _objective_and_grad(params, n, edges, cut_sizes, p):
    """Objective and gradient via finite differences (central difference)."""
    angles = params.reshape(p, 2)
    state = _qaoa_state(n, edges, cut_sizes, angles)
    f = -_expected_cut(state, cut_sizes)
    
    # Central difference gradient
    h = 1e-6
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += h
        f_plus = _objective(params_plus, n, edges, cut_sizes, p)
        
        params_minus = params.copy()
        params_minus[i] -= h
        f_minus = _objective(params_minus, n, edges, cut_sizes, p)
        
        grad[i] = (f_plus - f_minus) / (2 * h)
    
    return f, grad


def _nelder_mead_optimize(n, edges, cut_sizes, p, x0, max_iter=100, timeout=20):
    """Run Nelder-Mead optimization with time limit."""
    start_time = time.time()
    
    def obj(x):
        if time.time() - start_time > timeout:
            return 1e10
        return _objective(x, n, edges, cut_sizes, p)
    
    result = minimize(
        obj,
        x0,
        method='Nelder-Mead',
        options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
    )
    return result.x


def _polish_lbfgs(n, edges, cut_sizes, p, x0, max_iter=100):
    """High-precision polish with L-BFGS-B using gradient."""
    start_time = time.time()
    
    def obj_and_grad(x):
        if time.time() - start_time > 25:
            return 1e10, np.zeros_like(x)
        return _objective_and_grad(x, n, edges, cut_sizes, p)
    
    bounds = [(-np.pi, np.pi)] * (2 * p)
    result = minimize(
        obj_and_grad,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        jac=True,
        options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 500}
    )
    return result.x


def _polish_bfgs(n, edges, cut_sizes, p, x0, max_iter=50):
    """Polish with BFGS using analytical gradient."""
    start_time = time.time()
    
    def obj_and_grad(x):
        if time.time() - start_time > 15:
            return 1e10, np.zeros_like(x)
        return _objective_and_grad(x, n, edges, cut_sizes, p)
    
    result = minimize(
        obj_and_grad,
        x0,
        method='BFGS',
        jac=True,
        options={'maxiter': max_iter, 'gtol': 1e-12, 'maxfun': 200}
    )
    return result.x


def angles(n, edges, p):
    """Design graph-dependent two-layer QAOA angles."""
    global _DEADLINE
    _DEADLINE = time.time() + 140.0
    
    if _time_left() < 5.0:
        return np.tile([0.35, 0.25], (p, 1))
    
    try:
        # Precompute cut sizes
        cut_sizes = _precompute_cut_sizes(n, edges)
        
        # Multi-restart optimization
        best_params = None
        best_val = np.inf
        
        # Generate diverse initial guesses
        np.random.seed(42)
        n_restarts = min(10, max(4, int(_time_left() / 12)))
        
        # Well-chosen initial points based on known QAOA behavior
        initial_guesses = [
            np.array([0.25, 0.20, 0.15, 0.10], dtype=np.float64),
            np.array([0.40, 0.30, 0.20, 0.15], dtype=np.float64),
            np.array([0.10, 0.15, 0.30, 0.25], dtype=np.float64),
            np.array([0.30, 0.10, 0.25, 0.20], dtype=np.float64),
            np.array([0.15, 0.25, 0.20, 0.15], dtype=np.float64),
            np.array([0.20, 0.35, 0.10, 0.25], dtype=np.float64),
        ]
        
        for restart in range(n_restarts):
            if _time_left() < 8.0:
                break
            
            if restart < len(initial_guesses):
                x0 = initial_guesses[restart].copy()
            else:
                x0 = np.random.uniform(-0.5, 0.5, size=2 * p)
            
            # Time limit per restart
            remaining_restarts = n_restarts - restart
            timeout = min(12.0, _time_left() * 0.5 / max(1, remaining_restarts))
            if timeout < 3.0:
                break
            
            # Nelder-Mead phase
            params = _nelder_mead_optimize(n, edges, cut_sizes, p, x0, max_iter=60, timeout=timeout)
            
            if not np.all(np.isfinite(params)):
                continue
            
            val = _objective(params, n, edges, cut_sizes, p)
            if val < best_val:
                best_val = val
                best_params = params.copy()
        
        if best_params is None:
            return np.tile([0.35, 0.25], (p, 1))
        
        # High-precision polish phases
        # Phase 1: BFGS with gradient
        if _time_left() > 10.0:
            polished = _polish_bfgs(n, edges, cut_sizes, p, best_params, max_iter=40)
            if np.all(np.isfinite(polished)):
                polished_val = _objective(polished, n, edges, cut_sizes, p)
                if polished_val < best_val:
                    best_params = polished
                    best_val = polished_val
        
        # Phase 2: L-BFGS-B with tighter tolerance
        if _time_left() > 15.0:
            polished = _polish_lbfgs(n, edges, cut_sizes, p, best_params, max_iter=80)
            if np.all(np.isfinite(polished)):
                polished_val = _objective(polished, n, edges, cut_sizes, p)
                if polished_val < best_val:
                    best_params = polished
                    best_val = polished_val
        
        # Phase 3: Final Nelder-Mead with very tight tolerance
        if _time_left() > 8.0:
            final_timeout = min(6.0, _time_left() * 0.7)
            if final_timeout > 2.0:
                polished = _nelder_mead_optimize(n, edges, cut_sizes, p, best_params, max_iter=100, timeout=final_timeout)
                if np.all(np.isfinite(polished)):
                    polished_val = _objective(polished, n, edges, cut_sizes, p)
                    if polished_val < best_val:
                        best_params = polished
                        best_val = polished_val
        
        return best_params.reshape(p, 2)
    except Exception:
        return np.tile([0.35, 0.25], (p, 1))
# EVOLVE-BLOCK-END
