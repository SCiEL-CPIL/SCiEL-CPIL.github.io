"""QAOA parameter design: graph-dependent two-layer angles for MaxCut."""
import numpy as np
from scipy.optimize import minimize
import time

# Cache for exact MaxCut and graph features
_exact_cache = {}
_features_cache = {}


def _exact_maxcut(n, edges):
    """Compute exact MaxCut value by brute force."""
    key = (n, tuple(sorted(edges)))
    if key in _exact_cache:
        return _exact_cache[key]
    
    best = 0
    for mask in range(1 << n):
        cut = 0
        for u, v in edges:
            if ((mask >> u) ^ (mask >> v)) & 1:
                cut += 1
        if cut > best:
            best = cut
    _exact_cache[key] = best
    return best


def _precompute_cut_values(n, edges):
    """Precompute cut values for each mask."""
    key = (n, tuple(sorted(edges)))
    if key in _features_cache and 'cut_vals' in _features_cache[key]:
        return _features_cache[key]['cut_vals']
    
    dim = 1 << n
    cut_vals = np.zeros(dim, dtype=np.float64)
    for mask in range(dim):
        cut = 0
        for u, v in edges:
            if ((mask >> u) ^ (mask >> v)) & 1:
                cut += 1
        cut_vals[mask] = cut
    
    if key not in _features_cache:
        _features_cache[key] = {}
    _features_cache[key]['cut_vals'] = cut_vals
    return cut_vals


def _precompute_mixer_indices(n):
    """Precompute index arrays for fast mixer application via moveaxis."""
    key = ('mixer', n)
    if key in _features_cache:
        return _features_cache[key]
    
    dim = 1 << n
    # For each qubit j, create an array that maps index -> index with bit j flipped
    # We use reshape to (2, 2, ..., 2) and moveaxis to swap axis 0 of the j-th qubit
    # Actually, simpler: for each j, the flip is index ^ (1 << j)
    # But for fast application: reshape psi to (2,)*n, swap axes j and 0, 
    # then the mixer is: for each bit position j, apply a 2x2 rotation on that axis
    
    # The mixer H_M = sum_j X_j
    # exp(-i*beta*sum_j X_j) = product_j exp(-i*beta*X_j) since X_j commute
    # exp(-i*beta*X) = cos(beta)*I - i*sin(beta)*X
    # So on each qubit axis: [[cos, -i sin], [-i sin, cos]]
    
    # For fast application: reshape to (2,)*n, for each j apply the 2x2 gate on axis j
    # We precompute the reshape shape
    shape = (2,) * n
    
    _features_cache[key] = {'shape': shape, 'dim': dim}
    return _features_cache[key]


def _apply_mixer_fast(psi, beta, n, mixer_info):
    """Apply exp(-i*beta*sum_j X_j) to state vector using reshape and axis operations."""
    shape = mixer_info['shape']
    dim = mixer_info['dim']
    
    # Reshape to tensor
    psi_t = psi.reshape(shape)
    
    c = np.cos(beta)
    s = np.sin(beta)
    
    # For each qubit j, apply [[c, -i*s], [-i*s, c]] on axis j
    # psi_t[j, ...] -> c * psi_t[j, ...] - i*s * psi_t[1-j, ...]
    # We need to do this for all j sequentially
    for j in range(n):
        # Move axis j to front
        psi_t = np.moveaxis(psi_t, j, 0)
        # Apply the 2x2 gate: new[0] = c*old[0] - i*s*old[1], new[1] = -i*s*old[0] + c*old[1]
        old = psi_t
        psi_t = np.empty_like(old)
        psi_t[0] = c * old[0] - 1j * s * old[1]
        psi_t[1] = -1j * s * old[0] + c * old[1]
        # Move axis back
        psi_t = np.moveaxis(psi_t, 0, j)
    
    return psi_t.reshape(dim)


def _qaoa_expectation(n, edges, gammas, betas, features=None):
    """
    Compute the expected cut value of QAOA with given parameters.
    """
    if features is None:
        cut_vals = _precompute_cut_values(n, edges)
        mixer_info = _precompute_mixer_indices(n)
    else:
        cut_vals = features['cut_vals']
        mixer_info = features['mixer_info']
    
    dim = 1 << n
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    for gamma, beta in zip(gammas, betas):
        # Apply cost unitary: exp(-i*gamma*C) where C is diagonal
        psi = psi * np.exp(-1j * gamma * cut_vals)
        
        # Apply mixer unitary: exp(-i*beta*H_M)
        psi = _apply_mixer_fast(psi, beta, n, mixer_info)
    
    # Expected cut
    probs = np.abs(psi) ** 2
    expected_cut = np.sum(probs * cut_vals)
    
    return expected_cut


def _qaoa_gradient(n, edges, gammas, betas, features=None):
    """
    Compute gradient of expected cut w.r.t. (gamma_0, beta_0, gamma_1, beta_1)
    using the parameter-shift rule for exact gradients.
    
    For QAOA with diagonal cost and transverse-field mixer:
    d/dgamma_i <C> = 0.5 * (<C>_{gamma_i + pi/2} - <C>_{gamma_i - pi/2})
    d/dbeta_i <C> = 0.5 * (<C>_{beta_i + pi/2} - <C>_{beta_i - pi/2})
    """
    if features is None:
        cut_vals = _precompute_cut_values(n, edges)
        mixer_info = _precompute_mixer_indices(n)
        features = {'cut_vals': cut_vals, 'mixer_info': mixer_info}
    
    grad = np.zeros(4)
    half_pi = np.pi / 2
    
    for i in range(2):
        # gamma_i gradient via parameter-shift
        g_plus = gammas.copy()
        g_plus[i] += half_pi
        val_plus = _qaoa_expectation(n, edges, g_plus, betas, features)
        
        g_minus = gammas.copy()
        g_minus[i] -= half_pi
        val_minus = _qaoa_expectation(n, edges, g_minus, betas, features)
        
        grad[i] = 0.5 * (val_plus - val_minus)
        
        # beta_i gradient via parameter-shift
        b_plus = betas.copy()
        b_plus[i] += half_pi
        val_plus = _qaoa_expectation(n, edges, gammas, b_plus, features)
        
        b_minus = betas.copy()
        b_minus[i] -= half_pi
        val_minus = _qaoa_expectation(n, edges, gammas, b_minus, features)
        
        grad[2 + i] = 0.5 * (val_plus - val_minus)
    
    return grad


def _objective(params, n, edges, exact_opt, features):
    """Objective function to minimize (negative of ratio)."""
    gammas = params[:2]
    betas = params[2:]
    
    try:
        exp_cut = _qaoa_expectation(n, edges, gammas, betas, features)
    except Exception:
        return 1e6
    
    ratio = exp_cut / exact_opt if exact_opt > 0 else 0
    return -ratio


def _gradient(params, n, edges, exact_opt, features):
    """Gradient of the objective (negative ratio)."""
    gammas = params[:2]
    betas = params[2:]
    
    try:
        grad_cut = _qaoa_gradient(n, edges, gammas, betas, features)
    except Exception:
        return np.zeros(4)
    
    # d(-ratio)/dparam = -d(exp_cut/exact_opt)/dparam = -grad_cut / exact_opt
    if exact_opt > 0:
        return -grad_cut / exact_opt
    return np.zeros(4)


def _optimize_single(n, edges, x_start, exact_opt, features, deadline=None):
    """Run L-BFGS-B from a single starting point."""
    try:
        result = minimize(
            _objective,
            x_start,
            jac=_gradient,
            args=(n, edges, exact_opt, features),
            method='L-BFGS-B',
            bounds=[(-np.pi, np.pi)] * 4,
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 500}
        )
        return result
    except Exception:
        return None


def _polish(n, edges, x_start, exact_opt, features, deadline=None):
    """Nelder-Mead polish pass for higher precision."""
    try:
        result = minimize(
            _objective,
            x_start,
            args=(n, edges, exact_opt, features),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        return result
    except Exception:
        return None


def _optimize_for_graph(n, edges, p=2, deadline=None):
    """Optimize QAOA parameters for a specific graph."""
    exact_opt = _exact_maxcut(n, edges)
    cut_vals = _precompute_cut_values(n, edges)
    mixer_info = _precompute_mixer_indices(n)
    features = {'cut_vals': cut_vals, 'mixer_info': mixer_info}
    
    # Initial guesses based on known good QAOA values
    candidates = [
        np.array([0.35, 0.35, 0.25, 0.25]),
        np.array([0.5, 0.4, 0.3, 0.2]),
        np.array([0.25, 0.25, 0.35, 0.35]),
        np.array([0.4, 0.3, 0.2, 0.3]),
        np.array([0.3, 0.3, 0.3, 0.3]),
        np.array([0.2, 0.3, 0.4, 0.2]),
        np.array([0.45, 0.2, 0.35, 0.3]),
        np.array([0.15, 0.45, 0.25, 0.35]),
    ]
    
    best_params = candidates[0].copy()
    best_score = _objective(candidates[0], n, edges, exact_opt, features)
    
    # Multi-start L-BFGS-B optimization with analytic gradients
    n_starts = len(candidates)
    for i in range(n_starts):
        if deadline is not None and time.time() > deadline:
            break
        
        x_start = candidates[i].copy()
        
        result = _optimize_single(n, edges, x_start, exact_opt, features, deadline)
        if result is not None and result.fun < best_score:
            best_score = result.fun
            best_params = result.x.copy()
    
    # Additional random starts if time permits
    rng = np.random.RandomState(42)
    for i in range(4):
        if deadline is not None and time.time() > deadline:
            break
        
        x_start = rng.uniform(-np.pi/2, np.pi/2, size=4)
        
        result = _optimize_single(n, edges, x_start, exact_opt, features, deadline)
        if result is not None and result.fun < best_score:
            best_score = result.fun
            best_params = result.x.copy()
    
    # Nelder-Mead polish pass on the best result
    if deadline is None or time.time() < deadline - 5.0:
        polish_result = _polish(n, edges, best_params, exact_opt, features, deadline)
        if polish_result is not None and polish_result.fun < best_score:
            best_score = polish_result.fun
            best_params = polish_result.x.copy()
    
    # Second polish if time permits
    if deadline is None or time.time() < deadline - 3.0:
        polish_result = _polish(n, edges, best_params, exact_opt, features, deadline)
        if polish_result is not None and polish_result.fun < best_score:
            best_score = polish_result.fun
            best_params = polish_result.x.copy()
    
    gammas = best_params[:2]
    betas = best_params[2:]
    
    return np.array([[gammas[0], betas[0]], [gammas[1], betas[1]]])


def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles.
    
    Args:
        n: number of qubits/nodes
        edges: list of edges (u, v)
        p: number of QAOA layers (should be 2)
    
    Returns:
        Array of shape (p, 2) with rows (gamma, beta) in radians.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    if p != 2:
        return np.tile([0.35, 0.25], (p, 1))
    
    try:
        result = _optimize_for_graph(n, edges, p=2, deadline=deadline)
        return result
    except Exception:
        return np.array([[0.35, 0.25], [0.35, 0.25]])
