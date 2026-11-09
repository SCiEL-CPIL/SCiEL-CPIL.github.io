"""Combined program: vectorized X-layer, analytical gradients, and adaptive multi-restart optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 140.0


def _time_left():
    return _DEADLINE - time.time()


def _precompute_cut_sizes(n, edges):
    """Precompute cut size for each basis state (vectorized)."""
    dim = 1 << n
    cut_sizes = np.zeros(dim, dtype=np.float64)
    for (i, j) in edges:
        # Vectorized bit extraction
        states = np.arange(dim)
        bi = ((states >> i) & 1).astype(np.float64)
        bj = ((states >> j) & 1).astype(np.float64)
        cut_sizes += (bi != bj).astype(np.float64)
    return cut_sizes


def _apply_x_layer(state, beta, n):
    """Apply exp(-i*beta*sum(X_i)) to state vector using bitwise XOR indexing."""
    c = np.cos(beta)
    s = np.sin(beta)
    dim = len(state)
    for i in range(n):
        bit = 1 << i
        flipped_idx = np.arange(dim) ^ bit
        state = c * state + (-1j * s) * state[flipped_idx]
    return state


def _qaoa_forward(n, angles, cut_sizes):
    """
    Forward pass of QAOA circuit.
    Returns:
        state: final state vector
        intermediates: list of (state_before_layer, gamma, beta) for gradient computation
    """
    dim = 1 << n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    intermediates = []
    
    for k in range(len(angles)):
        gamma, beta = angles[k]
        # Store state before this layer for gradient computation
        intermediates.append((state.copy(), gamma, beta))
        
        # Apply cost layer
        state = state * np.exp(-1j * gamma * cut_sizes)
        # Apply X layer
        state = _apply_x_layer(state, beta, n)
        
    return state, intermediates


def _expected_cut(n, angles, cut_sizes):
    """Compute expected cut value."""
    state, _ = _qaoa_forward(n, angles, cut_sizes)
    probs = np.abs(state) ** 2
    return float(np.real(np.sum(probs * cut_sizes)))


def _objective(params, n, cut_sizes, p):
    """Objective function for minimization (negative expected cut)."""
    angles = params.reshape(p, 2)
    return -_expected_cut(n, angles, cut_sizes)


def _gradient(params, n, cut_sizes, p):
    """
    Compute gradient via parameter-shift rule (analytical).
    For each parameter theta_k, compute f(theta_k + pi/2) - f(theta_k - pi/2)
    """
    angles = params.reshape(p, 2).copy()
    grad = np.zeros_like(params)
    
    for k in range(p):
        for j in range(2):  # 0 for gamma, 1 for beta
            # Parameter shift rule
            angles_plus = angles.copy()
            angles_plus[k, j] += np.pi / 2
            
            angles_minus = angles.copy()
            angles_minus[k, j] -= np.pi / 2
            
            f_plus = _expected_cut(n, angles_plus, cut_sizes)
            f_minus = _expected_cut(n, angles_minus, cut_sizes)
            
            # Gradient is (f_plus - f_minus) / 2
            # Since we're minimizing -expected_cut, the gradient of the objective is negative
            grad[k * 2 + j] = -(f_plus - f_minus) / 2.0
            
    return grad


def _nelder_mead_optimize(x0, n, cut_sizes, p, maxiter=100, tol=1e-8):
    """Run Nelder-Mead optimization."""
    try:
        res = minimize(
            _objective,
            x0,
            args=(n, cut_sizes, p),
            method='Nelder-Mead',
            options={
                'maxiter': maxiter,
                'xatol': tol,
                'fatol': tol,
                'adaptive': True
            }
        )
        return res.x, -res.fun
    except Exception:
        return x0, _expected_cut(n, x0.reshape(p, 2), cut_sizes)


def _l_bfgs_b_optimize(x0, n, cut_sizes, p, maxiter=50, tol=1e-10):
    """Run L-BFGS-B optimization with analytical gradient."""
    try:
        res = minimize(
            _objective,
            x0,
            args=(n, cut_sizes, p),
            method='L-BFGS-B',
            jac=_gradient,
            options={'maxiter': maxiter, 'ftol': tol, 'gtol': 1e-8}
        )
        return res.x, -res.fun
    except Exception:
        return x0, _expected_cut(n, x0.reshape(p, 2), cut_sizes)


def _polish(x0, n, cut_sizes, p, maxiter=200, tol=1e-12):
    """High-precision polish pass using L-BFGS-B with analytical gradient."""
    try:
        res = minimize(
            _objective,
            x0,
            args=(n, cut_sizes, p),
            method='L-BFGS-B',
            jac=_gradient,
            options={'maxiter': maxiter, 'ftol': tol, 'gtol': 1e-10}
        )
        return res.x, -res.fun
    except Exception:
        return x0, _expected_cut(n, x0.reshape(p, 2), cut_sizes)


def angles(n, edges, p):
    """Design graph-dependent two-layer QAOA angles."""
    global _DEADLINE
    _DEADLINE = time.time() + 140.0
    
    # 1. Precompute cut sizes
    cut_sizes = _precompute_cut_sizes(n, edges)
    
    best_angles = None
    best_score = -np.inf
    
    # 2. Generate diverse initial guesses
    np.random.seed(42)
    initial_guesses = []
    
    # Standard QAOA angles
    initial_guesses.append(np.array([0.35, 0.25] * p))
    
    # Additional heuristic starts
    initial_guesses.append(np.array([0.5, 0.3] * p))
    initial_guesses.append(np.array([0.2, 0.4] * p))
    initial_guesses.append(np.array([0.7, 0.2] * p))
    initial_guesses.append(np.array([0.4, 0.35] * p))
    initial_guesses.append(np.array([0.6, 0.25] * p))
    
    # Random guesses
    for _ in range(20):
        gammas = np.random.uniform(0, np.pi, p)
        betas = np.random.uniform(0, np.pi, p)
        initial_guesses.append(np.column_stack((gammas, betas)).flatten())
    
    # 3. Multi-restart optimization
    # Use L-BFGS-B with analytical gradients for speed and accuracy
    for idx, x0 in enumerate(initial_guesses):
        if _time_left() < 15.0:
            break
        
        x_opt, score = _l_bfgs_b_optimize(x0, n, cut_sizes, p, maxiter=30, tol=1e-8)
        
        if score > best_score:
            best_score = score
            best_angles = x_opt.reshape(p, 2)
    
    # 4. Final high-precision polish on the best result
    if best_angles is not None and _time_left() > 20.0:
        x_polished, score_polished = _polish(
            best_angles.flatten(), n, cut_sizes, p, maxiter=100, tol=1e-12
        )
        if score_polished > best_score:
            best_score = score_polished
            best_angles = x_polished.reshape(p, 2)
    
    # Fallback if no valid optimization result
    if best_angles is None:
        best_angles = np.tile([0.35, 0.25], (p, 1))
        
    return best_angles
# EVOLVE-BLOCK-END
