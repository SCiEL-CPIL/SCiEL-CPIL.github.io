"""Rewritten program with model-and-optimize approach using efficient QAOA simulation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Safety deadline: 140s to ensure we finish before the 150s kill
_DEADLINE = time.time() + 140.0


def _time_left():
    return _DEADLINE - time.time()


def _precompute_cut_sizes(n, edges):
    """Precompute the cut size for all 2^n states using vectorized operations."""
    dim = 1 << n
    cut_sizes = np.zeros(dim, dtype=np.float64)
    for (i, j) in edges:
        states = np.arange(dim, dtype=np.int64)
        bi = (states >> i) & 1
        bj = (states >> j) & 1
        cut_sizes += (bi != bj).astype(np.float64)
    return cut_sizes


def _apply_x_layer_fast(state, beta, n):
    """Apply exp(-i beta sum X) to state vector using tensor operations."""
    c = np.cos(beta)
    s = np.sin(beta)
    tensor = state.reshape([2] * n)
    for i in range(n):
        moved = np.moveaxis(tensor, i, 0)
        flat = moved.reshape(2, -1)
        new_flat = np.empty_like(flat)
        new_flat[0] = c * flat[0] - 1j * s * flat[1]
        new_flat[1] = -1j * s * flat[0] + c * flat[1]
        tensor = np.moveaxis(new_flat.reshape([2] * n), 0, i)
    return tensor.flatten()


def _qaoa_state(n, cut_sizes, angles):
    """Simulate QAOA circuit and return final state."""
    dim = 1 << n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    for gamma, beta in angles:
        # Cost layer: diagonal
        state *= np.exp(-1j * gamma * cut_sizes)
        # X layer
        state = _apply_x_layer_fast(state, beta, n)
        
    return state


def _expected_cut(n, cut_sizes, angles):
    """Compute expected cut value."""
    state = _qaoa_state(n, cut_sizes, angles)
    probs = np.abs(state) ** 2
    return float(np.sum(probs * cut_sizes))


def _objective(params, n, cut_sizes, p):
    """Objective function for minimization (negative expected cut)."""
    angles = params.reshape(p, 2)
    return -_expected_cut(n, cut_sizes, angles)


def _objective_and_grad(params, n, cut_sizes, p):
    """Objective and numerical gradient for L-BFGS-B."""
    angles = params.reshape(p, 2)
    val = _expected_cut(n, cut_sizes, angles)
    
    # Central difference gradient
    eps = 1e-7
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += eps
        params_minus[i] -= eps
        val_plus = _expected_cut(n, cut_sizes, params_plus.reshape(p, 2))
        val_minus = _expected_cut(n, cut_sizes, params_minus.reshape(p, 2))
        grad[i] = (val_plus - val_minus) / (2 * eps)
    
    return -val, -grad


def _lbfgsb_optimize(x0, n, cut_sizes, p, maxiter=200):
    """Run L-BFGS-B optimization with time guard."""
    try:
        if _time_left() < 5.0:
            return x0, _expected_cut(n, cut_sizes, x0.reshape(p, 2))
        res = minimize(
            _objective_and_grad,
            x0,
            args=(n, cut_sizes, p),
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': maxiter,
                'gtol': 1e-10,
                'ftol': 1e-12,
            }
        )
        return res.x, -res.fun
    except Exception:
        return x0, _expected_cut(n, cut_sizes, x0.reshape(p, 2))


def _nelder_mead_optimize(x0, n, cut_sizes, p, maxiter=300, tol=1e-10):
    """Run Nelder-Mead optimization with time guard."""
    try:
        if _time_left() < 5.0:
            return x0, _expected_cut(n, cut_sizes, x0.reshape(p, 2))
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
        return x0, _expected_cut(n, cut_sizes, x0.reshape(p, 2))


def _polish(x0, n, cut_sizes, p, maxiter=500, tol=1e-12):
    """High-precision polish pass using L-BFGS-B with tighter tolerance."""
    try:
        if _time_left() < 5.0:
            return x0, _expected_cut(n, cut_sizes, x0.reshape(p, 2))
        res = minimize(
            _objective_and_grad,
            x0,
            args=(n, cut_sizes, p),
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': maxiter,
                'gtol': 1e-14,
                'ftol': 1e-15,
            }
        )
        return res.x, -res.fun
    except Exception:
        return x0, _expected_cut(n, cut_sizes, x0.reshape(p, 2))


def _coarse_to_fine_optimize(n, cut_sizes, p):
    """Multi-scale optimization: coarse grid then local refinement."""
    best_angles = None
    best_score = -np.inf
    
    # Phase 1: Reduced coarse grid search (5x5 = 25 per layer, 625 total for p=2)
    gamma_grid = np.linspace(0.1, np.pi * 0.85, 5)
    beta_grid = np.linspace(0.1, np.pi * 0.85, 5)
    
    best_coarse = None
    best_coarse_score = -np.inf
    
    for g1 in gamma_grid:
        for b1 in beta_grid:
            for g2 in gamma_grid:
                for b2 in beta_grid:
                    if _time_left() < 5.0:
                        break
                    angles = np.array([[g1, b1], [g2, b2]])
                    score = _expected_cut(n, cut_sizes, angles)
                    if score > best_coarse_score:
                        best_coarse_score = score
                        best_coarse = angles.copy()
            if _time_left() < 5.0:
                break
        if _time_left() < 5.0:
            break
    
    if best_coarse is not None:
        best_score = best_coarse_score
        best_angles = best_coarse
    
    # Phase 2: L-BFGS-B refinement from best coarse point
    if best_angles is not None and _time_left() > 10.0:
        x_opt, score = _lbfgsb_optimize(
            best_angles.flatten(), n, cut_sizes, p, maxiter=200
        )
        if score > best_score:
            best_score = score
            best_angles = x_opt.reshape(p, 2)
    
    # Phase 3: Polish
    if best_angles is not None and _time_left() > 10.0:
        x_polished, score_polished = _polish(
            best_angles.flatten(), n, cut_sizes, p, maxiter=300, tol=1e-12
        )
        if score_polished > best_score:
            best_score = score_polished
            best_angles = x_polished.reshape(p, 2)
    
    return best_angles, best_score


def _multi_restart_optimize(n, cut_sizes, p, num_restarts=15):
    """Multi-restart L-BFGS-B with diverse initial guesses."""
    best_angles = None
    best_score = -np.inf
    
    # Generate diverse initial guesses
    np.random.seed(123)
    initial_guesses = []
    
    # Known good QAOA angles for p=2
    initial_guesses.append(np.array([0.35, 0.25, 0.35, 0.25]))
    initial_guesses.append(np.array([0.5, 0.3, 0.5, 0.3]))
    initial_guesses.append(np.array([0.2, 0.4, 0.2, 0.4]))
    initial_guesses.append(np.array([0.7, 0.2, 0.7, 0.2]))
    initial_guesses.append(np.array([0.4, 0.35, 0.4, 0.35]))
    initial_guesses.append(np.array([0.6, 0.4, 0.6, 0.4]))
    
    # Random guesses
    for _ in range(num_restarts - len(initial_guesses)):
        gammas = np.random.uniform(0.1, np.pi * 0.85, p)
        betas = np.random.uniform(0.1, np.pi * 0.85, p)
        initial_guesses.append(np.column_stack((gammas, betas)).flatten())
    
    # Optimize from each initial guess using L-BFGS-B
    for x0 in initial_guesses:
        if _time_left() < 8.0:
            break
        
        x_opt, score = _lbfgsb_optimize(x0, n, cut_sizes, p, maxiter=100)
        
        if score > best_score:
            best_score = score
            best_angles = x_opt.reshape(p, 2)
    
    # Polish the best result
    if best_angles is not None and _time_left() > 10.0:
        x_polished, score_polished = _polish(
            best_angles.flatten(), n, cut_sizes, p, maxiter=300, tol=1e-12
        )
        if score_polished > best_score:
            best_score = score_polished
            best_angles = x_polished.reshape(p, 2)
    
    return best_angles, best_score


def angles(n, edges, p):
    """Design graph-dependent two-layer QAOA angles."""
    global _DEADLINE
    _DEADLINE = time.time() + 140.0
    
    # Precompute cut sizes
    cut_sizes = _precompute_cut_sizes(n, edges)
    
    # Strategy 1: Coarse-to-fine optimization
    angles_cf, score_cf = _coarse_to_fine_optimize(n, cut_sizes, p)
    
    # Strategy 2: Multi-restart optimization
    angles_mr, score_mr = _multi_restart_optimize(n, cut_sizes, p, num_restarts=15)
    
    # Pick the best result
    if angles_cf is None:
        best_angles = angles_mr
    elif angles_mr is None:
        best_angles = angles_cf
    else:
        best_angles = angles_cf if score_cf >= score_mr else angles_mr
    
    # Final polish with L-BFGS-B
    if best_angles is not None and _time_left() > 15.0:
        x_polished, score_polished = _polish(
            best_angles.flatten(), n, cut_sizes, p, maxiter=400, tol=1e-13
        )
        best_score = score_cf if angles_cf is not None else score_mr
        if score_polished > best_score:
            best_angles = x_polished.reshape(p, 2)
    
    # Fallback
    if best_angles is None:
        best_angles = np.tile([0.35, 0.25], (p, 1))
    
    return best_angles
# EVOLVE-BLOCK-END
