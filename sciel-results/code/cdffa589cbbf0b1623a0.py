"""Rewritten QAOA parameter design with graph-dependent optimization."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import itertools

# Deadline guard: 120s limit, use 110s safety margin
_DEADLINE = 110.0
_start_time = None


def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _build_graph(n, edges):
    """Build adjacency matrix from edge list."""
    adj = np.zeros((n, n), dtype=np.float64)
    for u, v in edges:
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    return adj


def _exhaustive_maxcut(n, edges):
    """Find exact MaxCut value by exhaustive search."""
    best = 0
    for bits in itertools.product([0, 1], repeat=n):
        cut = 0
        for u, v in edges:
            if bits[u] != bits[v]:
                cut += 1
        if cut > best:
            best = cut
    return best


def _expected_cut_value(n, edges, gammas, betas, p):
    """
    Compute expected cut value for given QAOA parameters.
    Uses exact statevector simulation for small n.
    """
    # State vector: 2^n complex amplitudes
    dim = 2 ** n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Precompute cost Hamiltonian action
    # C = sum_{(u,v) in E} (1 - Z_u Z_v) / 2
    # exp(-i gamma C) is diagonal in computational basis
    
    # For each layer: apply exp(-i gamma_k C) then exp(-i beta_k sum X)
    for k in range(p):
        gamma = gammas[k]
        beta = betas[k]
        
        # Apply cost Hamiltonian phase: diagonal
        # For each basis state, compute cut value, apply phase exp(-i * gamma * cut_val)
        phases = np.zeros(dim, dtype=np.complex128)
        for idx in range(dim):
            bits = (idx >> np.arange(n)) & 1
            cut_val = 0
            for u, v in edges:
                if bits[u] != bits[v]:
                    cut_val += 1
            phases[idx] = np.exp(-1j * gamma * cut_val)
        state = state * phases
        
        # Apply mixer: exp(-i beta * sum_i X_i)
        # This is a tensor product of single-qubit rotations
        # X rotation by angle 2*beta: [[cos(beta), -i*sin(beta)], [-i*sin(beta), cos(beta)]]
        # Actually exp(-i beta X) = cos(beta) I - i sin(beta) X
        # So each qubit: state_new = cos(beta)*state - i*sin(beta)*flip_state
        
        c = np.cos(beta)
        s = -1j * np.sin(beta)
        
        # Apply to each qubit
        for q in range(n):
            # Reshape to apply on qubit q
            shape = [2] * n
            new_state = np.zeros_like(state)
            # For each pair of states differing only in bit q
            mask = 1 << q
            for idx in range(dim):
                if idx & mask:
                    continue  # only process one of each pair
                partner = idx | mask
                new_state[idx] = c * state[idx] + s * state[partner]
                new_state[partner] = s * state[idx] + c * state[partner]
            state = new_state
    
    # Compute expected cut
    probs = np.abs(state) ** 2
    expected_cut = 0.0
    for idx in range(dim):
        if probs[idx] < 1e-15:
            continue
        bits = (idx >> np.arange(n)) & 1
        cut_val = 0
        for u, v in edges:
            if bits[u] != bits[v]:
                cut_val += 1
        expected_cut += probs[idx] * cut_val
    return expected_cut


def _objective(params, n, edges, p, exact_opt):
    """Objective to minimize: negative of expected_cut / exact_opt."""
    gammas = params[:p]
    betas = params[p:2*p]
    val = _expected_cut_value(n, edges, gammas, betas, p)
    return -val / exact_opt


def _fast_expected_cut(n, edges, gammas, betas, p):
    """
    Faster expected cut using probability propagation without full statevector.
    For n <= 10, statevector is feasible (2^10 = 1024 states).
    """
    return _expected_cut_value(n, edges, gammas, betas, p)


def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles maximizing the mean ratio
    of independently simulated expected cut to exact classical optimum.
    """
    global _start_time
    _start_time = time.time()
    
    # Build graph
    adj = _build_graph(n, edges)
    
    # Compute exact MaxCut
    exact_opt = _exhaustive_maxcut(n, edges)
    if exact_opt == 0:
        return np.zeros((p, 2))
    
    # Initial guess: uniform angles
    best_params = np.array([0.35] * p + [0.25] * p)
    best_score = _expected_cut_value(n, edges, best_params[:p], best_params[p:], p) / exact_opt
    
    # Try multiple optimization strategies within time budget
    strategies = [
        # Strategy 1: Local optimization from multiple starts
        lambda: _local_optimize(n, edges, p, exact_opt, best_params, best_score),
        # Strategy 2: Differential evolution (global)
        lambda: _global_optimize(n, edges, p, exact_opt, best_params, best_score),
    ]
    
    for strategy in strategies:
        if _elapsed() > _DEADLINE - 5:
            break
        try:
            best_params, best_score = strategy()
        except Exception:
            continue
    
    # Final polish if time permits
    if _elapsed() < _DEADLINE - 10:
        try:
            best_params = _polish(n, edges, p, exact_opt, best_params)
            val = _expected_cut_value(n, edges, best_params[:p], best_params[p:], p)
            best_score = val / exact_opt
        except Exception:
            pass
    
    # Return angles as (p, 2) array: rows are (gamma, beta)
    result = np.column_stack([best_params[:p], best_params[p:]])
    return result


def _local_optimize(n, edges, p, exact_opt, init_params, init_score):
    """Local optimization from multiple starting points."""
    best_params = init_params.copy()
    best_score = init_score
    
    # Generate multiple starting points
    rng = np.random.default_rng(42)
    starts = [init_params.copy()]
    
    for _ in range(8):
        start = rng.uniform(0, np.pi, size=2*p)
        starts.append(start)
    
    # Also try some structured starts
    starts.append(np.array([np.pi/(2*p)] * p + [np.pi/(2*p)] * p))
    starts.append(np.array([np.pi/(4*p)] * p + [np.pi/(4*p)] * p))
    starts.append(np.array([0.5] * p + [0.3] * p))
    starts.append(np.array([1.0] * p + [0.5] * p))
    
    bounds = [(0, np.pi)] * (2*p)
    
    for start in starts:
        if _elapsed() > _DEADLINE - 5:
            break
        try:
            res = minimize(
                _objective, start, args=(n, edges, p, exact_opt),
                method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.success and -res.fun > best_score:
                best_score = -res.fun
                best_params = res.x.copy()
        except Exception:
            continue
    
    return best_params, best_score


def _global_optimize(n, edges, p, exact_opt, init_params, init_score):
    """Global optimization using differential evolution."""
    best_params = init_params.copy()
    best_score = init_score
    
    if _elapsed() > _DEADLINE - 15:
        return best_params, best_score
    
    bounds = [(0, np.pi)] * (2*p)
    
    try:
        res = differential_evolution(
            _objective, bounds, args=(n, edges, p, exact_opt),
            maxiter=50, popsize=15, tol=1e-6, seed=42,
            polish=True, init='latinhypercube'
        )
        if -res.fun > best_score:
            best_score = -res.fun
            best_params = res.x.copy()
    except Exception:
        pass
    
    return best_params, best_score


def _polish(n, edges, p, exact_opt, params):
    """High-precision polish pass."""
    bounds = [(0, np.pi)] * (2*p)
    try:
        res = minimize(
            _objective, params, args=(n, edges, p, exact_opt),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if res.success:
            return res.x
    except Exception:
        pass
    return params
