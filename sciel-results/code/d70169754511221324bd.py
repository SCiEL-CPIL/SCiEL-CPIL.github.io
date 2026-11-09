"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

# Global cache for exact optima and graph properties
_graph_cache = {}


def _build_graph(n, edges):
    """Build adjacency structure from edge list."""
    if (n, tuple(sorted(edges))) in _graph_cache:
        return _graph_cache[(n, tuple(sorted(edges)))]
    
    # Build edge set and adjacency
    edge_set = set()
    for u, v in edges:
        if u < v:
            edge_set.add((u, v))
        else:
            edge_set.add((v, u))
    
    # Precompute cut values for all 2^n assignments (only for n <= 10)
    num_assignments = 1 << n
    cut_values = np.zeros(num_assignments, dtype=np.float64)
    
    if edge_set:
        edges_list = list(edge_set)
        # For each assignment, count cut edges
        # An edge (u,v) is cut if bit u != bit v
        # We can vectorize this
        for idx in range(num_assignments):
            bits = (idx >> np.arange(n)) & 1
            cut = 0
            for u, v in edges_list:
                if bits[u] != bits[v]:
                    cut += 1
            cut_values[idx] = cut
    
    optimum = np.max(cut_values) if num_assignments > 0 else 0
    _graph_cache[(n, tuple(sorted(edges)))] = (edge_set, cut_values, optimum)
    return _graph_cache[(n, tuple(sorted(edges)))]


def _expected_cut(n, edges, gammas, betas):
    """
    Compute expected cut for given QAOA parameters using state evolution.
    For small n (<=10), we can simulate the full state vector.
    
    The QAOA circuit:
    - Start in |+>^n
    - For each layer l:
        - Apply exp(-i * gamma_l * C) where C is the MaxCut cost operator
        - Apply exp(-i * beta_l * sum(X_i))
    
    C = sum_{(u,v) in E} (1 - Z_u Z_v) / 2
    sum(X_i) = sum of Pauli X on each qubit
    
    For n <= 10, we can represent the state as a 2^n vector and apply
    the unitaries directly.
    """
    edge_set, cut_values, optimum = _build_graph(n, edges)
    num_assignments = 1 << n
    
    # Initial state: |+>^n = (1/sqrt(2^n)) * [1, 1, ..., 1]
    state = np.ones(num_assignments, dtype=np.complex128) / np.sqrt(num_assignments)
    
    p = len(gammas)
    for l in range(p):
        gamma = gammas[l]
        beta = betas[l]
        
        # Apply exp(-i * gamma * C)
        # C |x> = cut_value(x) |x> in computational basis
        # So exp(-i * gamma * C) |x> = exp(-i * gamma * cut_value(x)) |x>
        if gamma != 0:
            phase = np.exp(-1j * gamma * cut_values)
            state = state * phase
        
        # Apply exp(-i * beta * sum(X_i))
        # sum(X_i) applies X to each qubit independently
        # X flips each bit, so X_i |x> = |x with bit i flipped>
        # sum(X_i) |x> = sum_i |x with bit i flipped>
        # exp(-i * beta * sum(X_i)) = product over i of exp(-i * beta * X_i)
        # since X_i commute with X_j for i != j
        # exp(-i * beta * X_i) = cos(beta) * I - i * sin(beta) * X_i
        # So the full mixer is product_i (cos(beta) * I - i * sin(beta) * X_i)
        
        if beta != 0:
            # Apply each X_i rotation
            for i in range(n):
                c = np.cos(beta)
                s = -1j * np.sin(beta)
                # New state: for each x, new[x] = c * state[x] + s * state[x ^ (1 << i)]
                flipped = state.copy()
                # Flip bit i
                mask = 1 << i
                # We need to swap elements where bit i is 0 vs 1
                new_state = np.empty_like(state)
                for idx in range(num_assignments):
                    if idx & mask == 0:
                        new_state[idx] = c * state[idx] + s * state[idx | mask]
                    else:
                        new_state[idx] = c * state[idx] + s * state[idx ^ mask]
                state = new_state
        
        # Normalize to prevent numerical drift
        norm = np.linalg.norm(state)
        if norm > 0:
            state = state / norm
    
    # Expected cut = sum_x |<x|psi>|^2 * cut_value(x)
    probs = np.abs(state) ** 2
    expected = np.sum(probs * cut_values)
    return expected


def _exact_optimum(n, edges):
    """Get the exact classical optimum cut value."""
    edge_set, cut_values, optimum = _build_graph(n, edges)
    return optimum


def _objective(params, n, edges):
    """
    Objective: maximize mean of expected_cut / exact_optimum.
    Here we just compute the ratio for a single graph.
    """
    gammas = params[:len(params)//2]
    betas = params[len(params)//2:]
    expected = _expected_cut(n, edges, gammas, betas)
    optimum = _exact_optimum(n, edges)
    if optimum == 0:
        return 0.0
    return expected / optimum


def _nelder_mead(func, x0, bounds=None, maxiter=200, tol=1e-6, time_limit=None, start_time=None):
    """
    Simple Nelder-Mead implementation with time limit.
    """
    n = len(x0)
    # Initial simplex
    simplex = [x0.astype(np.float64)]
    for i in range(n):
        xi = x0.copy()
        step = 0.1 if x0[i] == 0 else 0.1 * abs(x0[i])
        xi[i] += step
        simplex.append(xi)
    
    # Evaluate initial simplex
    fvals = [func(x) for x in simplex]
    
    iter_count = 0
    while iter_count < maxiter:
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        # Sort by function value
        idx = np.argsort(fvals)
        simplex = [simplex[i] for i in idx]
        fvals = [fvals[i] for i in idx]
        
        if abs(fvals[-1] - fvals[0]) < tol:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]
        f_worst = fvals[-1]
        
        # Reflection
        alpha = 1.0
        x_reflect = centroid + alpha * (centroid - worst)
        f_reflect = func(x_reflect)
        
        if f_reflect < fvals[0]:
            # Expansion
            gamma_exp = 2.0
            x_expand = centroid + gamma_exp * (centroid - worst)
            f_expand = func(x_expand)
            if f_expand < f_reflect:
                simplex[-1] = x_expand
                fvals[-1] = f_expand
            else:
                simplex[-1] = x_reflect
                fvals[-1] = f_reflect
        elif f_reflect < fvals[-2]:
            simplex[-1] = x_reflect
            fvals[-1] = f_reflect
        else:
            # Contraction
            rho = 0.5
            x_contract = centroid + rho * (worst - centroid)
            f_contract = func(x_contract)
            if f_contract < f_worst:
                simplex[-1] = x_contract
                fvals[-1] = f_contract
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = func(simplex[i])
        
        iter_count += 1
    
    # Return best point
    idx = np.argmin(fvals)
    return simplex[idx], fvals[idx]


def _optimize_for_graph(n, edges, p, time_budget=10.0):
    """
    Optimize QAOA parameters for a specific graph.
    Uses multiple restarts and Nelder-Mead.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Initial guesses based on literature
    # For p=2, typical good values are around gamma ~ 0.2-0.5, beta ~ 0.1-0.4
    best_params = np.array([0.35, 0.35, 0.25, 0.25])  # [g1, g2, b1, b2]
    best_val = -np.inf
    
    # Try several initial points
    initial_points = [
        np.array([0.35, 0.35, 0.25, 0.25]),
        np.array([0.5, 0.5, 0.3, 0.3]),
        np.array([0.2, 0.4, 0.15, 0.35]),
        np.array([0.4, 0.2, 0.3, 0.2]),
        np.array([0.6, 0.4, 0.25, 0.4]),
        np.array([0.3, 0.5, 0.4, 0.2]),
        np.array([0.1, 0.3, 0.1, 0.2]),
        np.array([0.7, 0.3, 0.35, 0.15]),
    ]
    
    # Add some random initial points
    rng = np.random.RandomState(42)
    for _ in range(4):
        g = rng.uniform(0.1, 0.8, 2)
        b = rng.uniform(0.1, 0.5, 2)
        initial_points.append(np.concatenate([g, b]))
    
    for x0 in initial_points:
        if time.time() > deadline:
            break
        
        def func(params):
            return _objective(params, n, edges)
        
        try:
            result, fval = _nelder_mead(
                func, x0, maxiter=100, tol=1e-7,
                time_limit=time_budget, start_time=start_time
            )
            if fval > best_val:
                best_val = fval
                best_params = result
        except Exception:
            continue
    
    # Final polish with tighter tolerance if time permits
    if time.time() < deadline - 2.0:
        def func(params):
            return _objective(params, n, edges)
        
        try:
            result, fval = _nelder_mead(
                func, best_params, maxiter=150, tol=1e-9,
                time_limit=max(0, deadline - time.time() - 1.0),
                start_time=start_time
            )
            if fval > best_val:
                best_val = fval
                best_params = result
        except Exception:
            pass
    
    return best_params


def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles.
    
    Args:
        n: number of vertices
        edges: list of (u, v) tuples
        p: number of layers (should be 2)
    
    Returns:
        Array of shape (p, 2) with rows (gamma, beta) in radians.
    """
    global _graph_cache
    
    # Clear cache if too large to prevent memory issues
    if len(_graph_cache) > 100:
        _graph_cache.clear()
    
    start_time = time.time()
    
    # Optimize parameters for this specific graph
    params = _optimize_for_graph(n, edges, p, time_budget=110.0)
    
    # params = [gamma_1, gamma_2, beta_1, beta_2]
    # Reshape to (p, 2) with rows (gamma, beta)
    result = np.array([[params[0], params[2]],
                       [params[1], params[3]]])
    
    return result
# EVOLVE-BLOCK-END
