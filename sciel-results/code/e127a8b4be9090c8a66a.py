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
    
    # Precompute bit masks for Z_i Z_j computation
    # For each edge (i,j), precompute which indices have bit_i != bit_j
    edge_diff_masks = []
    for i, j in edge_list:
        mask_i = 1 << i
        mask_j = 1 << j
        # Create boolean mask: True where bit_i != bit_j
        # We'll compute this on the fly using vectorized operations
        pass
    
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
            # Create index arrays for vectorized operation
            indices = np.arange(dim)
            bit_i = (indices >> i) & 1
            bit_j = (indices >> j) & 1
            diff_mask = (bit_i != bit_j)
            state[diff_mask] *= e_gamma
        
        # Apply exp(-i beta Sx): for each qubit, apply R_x(beta)
        # R_x(beta) = cos(beta)*I - i*sin(beta)*X
        # In computational basis, X|0> = |1>, X|1> = |0>
        # So for qubit q: new_state[|0...0>] = c*old[|0...0>] - i*s*old[|1...1>]
        #              new_state[|1...1>] = -i*s*old[|0...0>] + c*old[|1...1>]
        c = np.cos(beta)
        s = np.sin(beta)
        for q in range(n):
            mask_q = 1 << q
            indices = np.arange(dim)
            bit_q = (indices >> q) & 1
            # For states where bit_q = 0, partner is indices with bit_q = 1
            # For states where bit_q = 1, partner is indices with bit_q = 0
            zero_mask = (bit_q == 0)
            one_mask = (bit_q == 1)
            zero_indices = indices[zero_mask]
            one_indices = indices[one_mask]
            
            old_zero = state[zero_indices].copy()
            old_one = state[one_indices].copy()
            
            state[zero_indices] = c * old_zero - 1j * s * old_one
            state[one_indices] = -1j * s * old_zero + c * old_one
    
    # Compute expected cut: <psi| C |psi> = E - sum_{edges} <Z_i Z_j>/2
    # <Z_i Z_j> = 1 - 2*P(bit_i != bit_j)
    prob = np.abs(state) ** 2
    indices = np.arange(dim)
    
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


def _nelder_mead(f, x0, args=(), maxiter=300, xatol=1e-5, fatol=1e-5):
    """Simple Nelder-Mead implementation."""
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
    for _ in range(3):
        start_points.append(np.random.uniform(0.1, np.pi, 2*p))
    
    best_params = start_points[0]
    best_score = -np.inf
    
    for x_init in start_points:
        if _check_deadline():
            break
        
        try:
            result_x, result_f = _nelder_mead(
                _objective,
                x_init,
                args=(n, edges, optimum),
                maxiter=300,
                xatol=1e-5,
                fatol=1e-5
            )
            
            score = -result_f
            if score > best_score:
                best_score = score
                best_params = result_x
        except Exception:
            continue
    
    # Extract gammas and betas
    gammas = np.clip(best_params[:p], 0.0, np.pi)
    betas = np.clip(best_params[p:], 0.0, np.pi)
    
    # Format as (p, 2) array
    angles_out = np.column_stack((gammas, betas))
    
    return angles_out

# EVOLVE-BLOCK-END
