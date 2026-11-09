"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

# Cache for exact optima
_optimum_cache = {}
# Cache for best angles found per graph signature
_angles_cache = {}


def _graph_signature(n, edges):
    """Create a hashable signature for the graph."""
    edge_set = frozenset(tuple(sorted(e)) for e in edges)
    return (n, edge_set)


def _exact_maxcut(n, edges):
    """Compute exact MaxCut value via brute force."""
    sig = _graph_signature(n, edges)
    if sig in _optimum_cache:
        return _optimum_cache[sig]
    
    # For small n, brute force all 2^n assignments
    best = 0
    edge_list = [(u, v) for u, v in edges]
    m = len(edge_list)
    
    for mask in range(1 << n):
        cut = 0
        for u, v in edge_list:
            if (mask >> u) & 1 != (mask >> v) & 1:
                cut += 1
        if cut > best:
            best = cut
    
    _optimum_cache[sig] = best
    return best


def _expected_cut(n, edges, gammas, betas):
    """
    Compute expected cut size for QAOA with given angles.
    Uses the standard QAOA expectation formula.
    
    For a graph with edges E, the expected cut is:
    sum_{(u,v) in E} (1 - <Z_u Z_v>) / 2
    
    where <Z_u Z_v> is computed from the QAOA state.
    
    For p=2 QAOA on general graphs, we need to compute the expectation
    <Z_u Z_v> for each edge. This requires tracking correlations.
    
    The state is: |psi> = prod_{l=p}^{1} exp(-i beta_l sum_i X_i) exp(-i gamma_l sum_{(j,k) in E} Z_j Z_k) |+>^n
    
    For p=2, we can compute <Z_u Z_v> using the following approach:
    - Start with |+>^n where <Z_i> = 0, <X_i> = 1, <Y_i> = 0
    - Apply the unitaries and track the relevant correlators.
    
    A more efficient approach for small n: simulate the full state vector.
    For n up to 10, the state space is 2^10 = 1024, which is manageable.
    """
    # Full state vector simulation for small n
    dim = 1 << n
    # Initial state |+>^n = (1/sqrt(2^n)) * [1, 1, ..., 1]
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    edge_list = [(u, v) for u, v in edges]
    
    for l in range(len(gammas)):
        gamma = gammas[l]
        beta = betas[l]
        
        # Apply exp(-i gamma * C) where C = sum_{(u,v) in E} Z_u Z_v
        # C is diagonal in the computational basis
        # C|s> = c(s)|s> where c(s) = sum_{(u,v) in E} z_u * z_v, z_i = +1 if bit i is 0, -1 if bit i is 1
        phases = np.zeros(dim, dtype=np.float64)
        for s in range(dim):
            c_val = 0.0
            for u, v in edge_list:
                zu = 1.0 if ((s >> u) & 1) == 0 else -1.0
                zv = 1.0 if ((s >> v) & 1) == 0 else -1.0
                c_val += zu * zv
            phases[s] = -gamma * c_val
        
        psi = psi * np.exp(1j * phases)
        
        # Apply exp(-i beta * sum_i X_i)
        # This is a product of single-qubit rotations: prod_i exp(-i beta X_i)
        # exp(-i beta X) = cos(beta) I - i sin(beta) X
        # In the computational basis, X flips the bit.
        # For each qubit i, the operation maps |s> -> cos(beta)|s> - i sin(beta)|s_flip_i>
        
        # We can do this efficiently by processing each qubit
        for i in range(n):
            # For each state, the amplitude gets mixed with the state where bit i is flipped
            # new_psi[s] = cos(beta) * old_psi[s] - i sin(beta) * old_psi[s ^ (1 << i)]
            new_psi = np.empty_like(psi)
            for s in range(dim):
                s_flip = s ^ (1 << i)
                new_psi[s] = np.cos(beta) * psi[s] - 1j * np.sin(beta) * psi[s_flip]
            psi = new_psi
    
    # Now compute <Z_u Z_v> for each edge
    # <Z_u Z_v> = sum_s psi[s]^* * (z_u * z_v) * psi[s]
    # where z_i = +1 if bit i is 0, -1 if bit i is 1
    expected_cut = 0.0
    for u, v in edge_list:
        zz = np.zeros(dim, dtype=np.float64)
        for s in range(dim):
            zu = 1.0 if ((s >> u) & 1) == 0 else -1.0
            zv = 1.0 if ((s >> v) & 1) == 0 else -1.0
            zz[s] = zu * zv
        zz_psi = zz * psi
        # <Z_u Z_v> = sum_s conj(psi[s]) * zz[s] * psi[s]
        correlator = np.sum(np.conj(psi) * zz_psi).real
        expected_cut += (1.0 - correlator) / 2.0
    
    return expected_cut


def _objective(params, n, edges):
    """Objective function to maximize (we minimize the negative)."""
    gammas = params[:2]
    betas = params[2:4]
    cut = _expected_cut(n, edges, gammas, betas)
    return -cut


def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles.
    
    Args:
        n: number of qubits
        edges: list of edges (u, v)
        p: number of layers (should be 2)
    
    Returns:
        Array of shape (p, 2) with rows (gamma, beta) in radians
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    sig = _graph_signature(n, edges)
    if sig in _angles_cache:
        return _angles_cache[sig]
    
    # Get exact optimum for normalization (not directly used in optimization but good to have)
    # Actually, we just need to maximize expected cut, so we don't need the optimum for optimization
    
    # Initial guess based on known good values for QAOA
    # Typical good starting points for p=2
    x0 = np.array([0.4, 0.3, 0.5, 0.4])  # gamma1, beta1, gamma2, beta2
    
    best_params = x0.copy()
    best_obj = _objective(best_params, n, edges)
    
    # Try multiple starting points
    start_points = [
        np.array([0.4, 0.3, 0.5, 0.4]),
        np.array([0.3, 0.2, 0.4, 0.3]),
        np.array([0.5, 0.4, 0.6, 0.5]),
        np.array([0.2, 0.1, 0.3, 0.2]),
        np.array([0.6, 0.5, 0.7, 0.6]),
        np.array([0.35, 0.25, 0.45, 0.35]),
        np.array([0.45, 0.35, 0.55, 0.45]),
        np.array([0.25, 0.15, 0.35, 0.25]),
    ]
    
    for x_start in start_points:
        if time.time() > deadline:
            break
        
        try:
            # Use Nelder-Mead for local optimization
            result = minimize(
                _objective,
                x_start,
                args=(n, edges),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # If we have time, try differential evolution for global search
    if time.time() < deadline - 20:
        try:
            bounds = [(0, np.pi), (0, np.pi), (0, np.pi), (0, np.pi)]
            de_result = differential_evolution(
                _objective,
                bounds,
                args=(n, edges),
                maxiter=30,
                popsize=10,
                tol=1e-8,
                seed=42
            )
            if de_result.fun < best_obj:
                best_obj = de_result.fun
                best_params = de_result.x
        except Exception:
            pass
    
    # Final polish with tighter tolerance if time permits
    if time.time() < deadline - 5:
        try:
            result = minimize(
                _objective,
                best_params,
                args=(n, edges),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Extract gammas and betas
    gammas = best_params[:2]
    betas = best_params[2:4]
    
    # Clamp to reasonable range
    gammas = np.clip(gammas, 0, np.pi)
    betas = np.clip(betas, 0, np.pi)
    
    result_array = np.column_stack((gammas, betas))
    
    _angles_cache[sig] = result_array
    
    return result_array
# EVOLVE-BLOCK-END
