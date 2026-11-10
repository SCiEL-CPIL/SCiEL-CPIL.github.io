"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 145.0  # wall-clock safety margin (evaluator kills at 150s)
_START = time.time()


def _check_deadline():
    """Return True if we're running out of time."""
    return (time.time() - _START) > _DEADLINE


def _maxcut_exact(n, edges):
    """Brute-force MaxCut for small n (n <= 20 or so)."""
    if n == 0:
        return 0
    # Get unique edges
    edge_set = set()
    edge_list = []
    for u, v in edges:
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) not in edge_set:
            edge_set.add((a, b))
            edge_list.append((a, b))
    
    if not edge_list:
        return 0
    
    best = 0
    for mask in range(1 << n):
        cut = 0
        for i, j in edge_list:
            ai = (mask >> i) & 1
            aj = (mask >> j) & 1
            if ai != aj:
                cut += 1
        if cut > best:
            best = cut
    return best


def _qaoa_expectation_fast(n, edges, gammas, betas):
    """
    Compute expected cut for p-layer QAOA via exact statevector simulation.
    Optimized version using tensor operations.
    """
    dim = 1 << n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Get unique edges
    edge_set = set()
    edge_list = []
    for u, v in edges:
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) not in edge_set:
            edge_set.add((a, b))
            edge_list.append((a, b))
    
    if not edge_list:
        return 0.0
    
    E = len(edge_list)
    
    # Precompute index arrays for bit extraction to avoid repeated allocation
    indices = np.arange(dim)
    
    # Precompute diff masks for each edge to avoid recomputation
    edge_diff_masks = []
    for i, j in edge_list:
        bit_i = (indices >> i) & 1
        bit_j = (indices >> j) & 1
        edge_diff_masks.append(bit_i != bit_j)
    
    # Precompute zero/one masks for each qubit
    qubit_zero_masks = []
    qubit_one_masks = []
    qubit_zero_indices = []
    qubit_one_indices = []
    for q in range(n):
        bit_q = (indices >> q) & 1
        zero_mask = (bit_q == 0)
        one_mask = (bit_q == 1)
        qubit_zero_masks.append(zero_mask)
        qubit_one_masks.append(one_mask)
        qubit_zero_indices.append(indices[zero_mask])
        qubit_one_indices.append(indices[one_mask])
    
    # Apply QAOA layers
    for k in range(len(gammas)):
        gamma = gammas[k]
        beta = betas[k]
        
        # Apply exp(-i gamma C): for each edge, apply the edge gate
        e_gamma = np.exp(-1j * gamma)
        for idx, (i, j) in enumerate(edge_list):
            state[edge_diff_masks[idx]] *= e_gamma
        
        # Apply exp(-i beta Sx): for each qubit, apply R_x(beta)
        c = np.cos(beta)
        s = np.sin(beta)
        for q in range(n):
            zero_indices = qubit_zero_indices[q]
            one_indices = qubit_one_indices[q]
            
            old_zero = state[zero_indices].copy()
            old_one = state[one_indices].copy()
            
            state[zero_indices] = c * old_zero - 1j * s * old_one
            state[one_indices] = -1j * s * old_zero + c * old_one
    
    # Compute expected cut: <psi| C |psi> = E - sum_{edges} <Z_i Z_j>/2
    prob = np.abs(state) ** 2
    
    sum_zz = 0.0
    for idx, (i, j) in enumerate(edge_list):
        p_diff = np.sum(prob[edge_diff_masks[idx]])
        zz = 1.0 - 2.0 * p_diff
        sum_zz += zz
    
    expected_cut = E - sum_zz / 2.0
    return float(expected_cut)


def _objective(params, n, edges, optimum):
    """Negative ratio for minimization."""
    p = len(params) // 2
    gammas = params[:p]
    betas = params[p:]
    
    # Clip to reasonable range
    gammas = np.clip(gammas, 0.0, np.pi)
    betas = np.clip(betas, 0.0, np.pi)
    
    exp_cut = _qaoa_expectation_fast(n, edges, gammas, betas)
    ratio = exp_cut / optimum
    return -ratio


def _nelder_mead(f, x0, args=(), maxiter=200, xatol=1e-5, fatol=1e-5):
    """
    Simple Nelder-Mead implementation.
    """
    n = len(x0)
    # Create initial simplex
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        xi[i] += 0.1 if xi[i] == 0 else 0.1 * xi[i]
        simplex.append(xi)
    
    # Evaluate initial simplex
    values = [f(x, *args) for x in simplex]
    
    for iteration in range(maxiter):
        if _check_deadline():
            break
        
        # Sort by function value
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        
        # Check convergence
        if abs(values[0] - values[-1]) < fatol:
            break
        
        # Centroid of all but worst point
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        alpha = 1.0
        x_reflect = centroid + alpha * (centroid - simplex[-1])
        f_reflect = f(x_reflect, *args)
        
        if values[0] <= f_reflect < values[-2]:
            # Accept reflection
            simplex[-1] = x_reflect
            values[-1] = f_reflect
            continue
        
        if f_reflect < values[0]:
            # Expansion
            gamma = 2.0
            x_expand = centroid + gamma * (x_reflect - centroid)
            f_expand = f(x_expand, *args)
            if f_expand < f_reflect:
                simplex[-1] = x_expand
                values[-1] = f_expand
            else:
                simplex[-1] = x_reflect
                values[-1] = f_reflect
            continue
        
        # Contraction
        rho = 0.5
        if f_reflect < values[-1]:
            # Outside contraction
            x_contract = centroid + rho * (x_reflect - centroid)
        else:
            # Inside contraction
            x_contract = centroid - rho * (x_reflect - centroid)
        f_contract = f(x_contract, *args)
        
        if f_contract < min(f_reflect, values[-1]):
            simplex[-1] = x_contract
            values[-1] = f_contract
        else:
            # Shrink
            for i in range(1, n + 1):
                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                values[i] = f(simplex[i], *args)
    
    # Return best point
    best_idx = np.argmin(values)
    return simplex[best_idx], values[best_idx]


def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles maximizing the mean ratio
    of the independently simulated expected cut to the exact classical optimum.
    
    Args:
        n: number of qubits
        edges: list of edges
        p: number of QAOA layers
    
    Returns:
        np.array of shape (p, 2), rows (gamma, beta) in radians
    """
    # Compute exact optimum
    optimum = _maxcut_exact(n, edges)
    if optimum == 0:
        return np.zeros((p, 2))
    
    # Initial guesses based on known QAOA heuristics
    # For p=2, common good starting points
    start_points = [
        np.array([np.pi/4, np.pi/8, np.pi/8, np.pi/16])[:2*p],
        np.array([np.pi/6, np.pi/12, np.pi/12, np.pi/24])[:2*p],
        np.array([np.pi/3, np.pi/6, np.pi/6, np.pi/12])[:2*p],
        np.array([0.5, 0.25, 0.25, 0.125])[:2*p],
        np.array([0.7, 0.35, 0.35, 0.175])[:2*p],
        np.array([0.4, 0.2, 0.2, 0.1])[:2*p],
    ]
    
    # Add some random starts
    np.random.seed(42)
    for _ in range(5):
        start_points.append(np.random.uniform(0.1, np.pi, 2*p))
    
    best_params = start_points[0]
    best_score = -np.inf
    
    # Phase 1: Quick Nelder-Mead from multiple starts
    for x_init in start_points:
        if _check_deadline():
            break
        
        try:
            result_x, result_f = _nelder_mead(
                _objective,
                x_init,
                args=(n, edges, optimum),
                maxiter=150,
                xatol=1e-4,
                fatol=1e-4
            )
            
            score = -result_f
            if score > best_score:
                best_score = score
                best_params = result_x.copy()
        except Exception:
            continue
    
    # Phase 2: Refine best point with L-BFGS-B if time permits
    if not _check_deadline():
        try:
            bounds = [(0.0, np.pi)] * (2 * p)
            result = minimize(
                _objective,
                best_params,
                args=(n, edges, optimum),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-6, 'gtol': 1e-5}
            )
            
            if result.success or result.fun < -best_score:
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_params = result.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final local refinement with Nelder-Mead from best point
    if not _check_deadline():
        try:
            result_x, result_f = _nelder_mead(
                _objective,
                best_params,
                args=(n, edges, optimum),
                maxiter=100,
                xatol=1e-5,
                fatol=1e-5
            )
            
            score = -result_f
            if score > best_score:
                best_score = score
                best_params = result_x.copy()
        except Exception:
            pass
    
    # Extract gammas and betas
    gammas = np.clip(best_params[:p], 0.0, np.pi)
    betas = np.clip(best_params[p:], 0.0, np.pi)
    
    # Format as (p, 2) array
    angles_out = np.column_stack((gammas, betas))
    
    return angles_out

# EVOLVE-BLOCK-END
