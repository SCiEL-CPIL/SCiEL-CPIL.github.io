"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

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
    
    # Apply QAOA layers
    for k in range(len(gammas)):
        gamma = gammas[k]
        beta = betas[k]
        
        # Apply exp(-i gamma C): for each edge, apply the edge gate
        # exp(-i*gamma*(1-Z_iZ_j)/2) = diag(1, exp(-i*gamma), exp(-i*gamma), 1) in basis |00>,|01>,|10>,|11>
        # This means: multiply by exp(-i*gamma) when bits i and j differ
        e_gamma = np.exp(-1j * gamma)
        for i, j in edge_list:
            mask_i = 1 << i
            mask_j = 1 << j
            bit_i = (indices >> i) & 1
            bit_j = (indices >> j) & 1
            diff_mask = (bit_i != bit_j)
            state[diff_mask] *= e_gamma
        
        # Apply exp(-i beta Sx): for each qubit, apply R_x(beta)
        # R_x(beta) = cos(beta)*I - i*sin(beta)*X
        c = np.cos(beta)
        s = np.sin(beta)
        for q in range(n):
            mask_q = 1 << q
            bit_q = (indices >> q) & 1
            zero_mask = (bit_q == 0)
            one_mask = (bit_q == 1)
            zero_indices = indices[zero_mask]
            one_indices = indices[one_mask]
            
            old_zero = state[zero_indices].copy()
            old_one = state[one_indices].copy()
            
            state[zero_indices] = c * old_zero - 1j * s * old_one
            state[one_indices] = -1j * s * old_zero + c * old_one
    
    # Compute expected cut: <psi| C |psi> = E - sum_{edges} <Z_i Z_j>/2
    prob = np.abs(state) ** 2
    
    sum_zz = 0.0
    for i, j in edge_list:
        bit_i = (indices >> i) & 1
        bit_j = (indices >> j) & 1
        diff_mask = (bit_i != bit_j)
        p_diff = np.sum(prob[diff_mask])
        zz = 1.0 - 2.0 * p_diff
        sum_zz += zz
    
    expected_cut = E - sum_zz / 2.0
    return float(expected_cut)


def _nelder_mead(f, x0, args=(), maxiter=200, xatol=1e-4, fatol=1e-4):
    """
    Optimized Nelder-Mead for 4-dimensional problem.
    """
    n = len(x0)
    simplex = [x0.copy()]
    # Initial step size
    step = 0.2
    for i in range(n):
        xi = x0.copy()
        xi[i] += step
        simplex.append(xi)
    
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
            x_contract = centroid + rho * (x_reflect - centroid)
        else:
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
    
    best_idx = np.argmin(values)
    return simplex[best_idx], values[best_idx]


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
    
    # Generate diverse initial guesses
    # 1. Heuristic points based on known QAOA behavior
    # For p=2, typical good ranges are gamma ~ [0.2, 0.8], beta ~ [0.1, 0.5]
    heuristic_starts = [
        np.array([0.5, 0.25, 0.5, 0.25]),
        np.array([0.3, 0.15, 0.3, 0.15]),
        np.array([0.8, 0.4, 0.8, 0.4]),
        np.array([0.1, 0.05, 0.1, 0.05]),
        np.array([0.6, 0.3, 0.4, 0.2]),
    ]
    
    # 2. Random starts
    rng = np.random.default_rng(42)
    random_starts = []
    for _ in range(20):
        g1 = rng.uniform(0.1, 1.5)
        b1 = rng.uniform(0.05, 0.8)
        g2 = rng.uniform(0.1, 1.5)
        b2 = rng.uniform(0.05, 0.8)
        random_starts.append(np.array([g1, b1, g2, b2]))
    
    # Combine and deduplicate roughly
    all_starts = heuristic_starts + random_starts
    
    best_params = all_starts[0]
    best_score = -np.inf
    
    # Dynamic loop: run as many restarts as time allows
    # We prioritize heuristic starts first, then random
    for x_init in all_starts:
        if _check_deadline():
            break
        
        try:
            result_x, result_f = _nelder_mead(
                _objective,
                x_init,
                args=(n, edges, optimum),
                maxiter=150,  # Reduced maxiter to allow more restarts
                xatol=1e-4,
                fatol=1e-4
            )
            
            score = -result_f
            if score > best_score:
                best_score = score
                best_params = result_x
        except Exception:
            continue
    
    # If we still have time, do a few more random restarts
    # This helps if the initial 25 weren't enough
    restart_count = 0
    while not _check_deadline() and restart_count < 20:
        x_init = rng.uniform(0.1, 1.5, 2*p)
        # Clip to valid range
        x_init[0] = np.clip(x_init[0], 0, np.pi)
        x_init[2] = np.clip(x_init[2], 0, np.pi)
        x_init[1] = np.clip(x_init[1], 0, np.pi)
        x_init[3] = np.clip(x_init[3], 0, np.pi)
        
        try:
            result_x, result_f = _nelder_mead(
                _objective,
                x_init,
                args=(n, edges, optimum),
                maxiter=100, # Even shorter for additional random restarts
                xatol=1e-4,
                fatol=1e-4
            )
            
            score = -result_f
            if score > best_score:
                best_score = score
                best_params = result_x
        except Exception:
            pass
        restart_count += 1

    # Extract gammas and betas
    gammas = np.clip(best_params[:p], 0.0, np.pi)
    betas = np.clip(best_params[p:], 0.0, np.pi)
    
    # Format as (p, 2) array
    angles_out = np.column_stack((gammas, betas))
    
    return angles_out

# EVOLVE-BLOCK-END
