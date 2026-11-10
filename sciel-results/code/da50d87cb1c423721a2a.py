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


def _qaoa_expectation(n, edges, gammas, betas):
    """
    Compute expected cut for p-layer QAOA via exact statevector simulation.
    C = sum_{(i,j) in edges} (1 - Z_i Z_j) / 2
    State: start |+>^n, apply exp(-i gamma_k C) exp(-i beta_k Sx) for k=1..p
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
    
    # Precompute Z_i values for all states
    # Z_i(x) = 1 if bit i is 0, -1 if bit i is 1
    # We'll compute <Z_i Z_j> directly
    
    # Apply QAOA layers
    for k in range(len(gammas)):
        gamma = gammas[k]
        beta = betas[k]
        
        # Apply exp(-i gamma C): for each edge, apply the edge gate
        # exp(-i gamma (1-Z_iZ_j)/2) = diag(1, e^{-igamma}, e^{-igamma}, 1)
        e = np.exp(-1j * gamma)
        
        for i, j in edge_list:
            # Apply two-qubit gate for edge (i, j)
            # We need to apply diag(1, e, e, 1) on qubits i, j
            # This is equivalent to: if bits i and j are different, multiply by e
            
            # Efficient approach: iterate through all states and apply phase
            # For each state x, if bit_i != bit_j, multiply state[x] by e
            for x in range(dim):
                bi = (x >> i) & 1
                bj = (x >> j) & 1
                if bi != bj:
                    state[x] *= e
        
        # Apply exp(-i beta Sx): for each qubit, apply rx gate
        # exp(-i beta X) = cos(beta) I - i sin(beta) X
        c = np.cos(beta)
        s = np.sin(beta)
        
        for q in range(n):
            # Apply single-qubit rotation to qubit q
            # For each state, split into pairs where bit q is 0 and 1
            for x in range(dim):
                if (x >> q) & 1 == 0:
                    x1 = x | (1 << q)
                    # [state[x], state[x1]] -> [c*state[x] - i*s*state[x1], -i*s*state[x] + c*state[x1]]
                    old_x = state[x]
                    old_x1 = state[x1]
                    state[x] = c * old_x - 1j * s * old_x1
                    state[x1] = -1j * s * old_x + c * old_x1
    
    # Compute expected cut: <psi| C |psi> = E - sum_{edges} <Z_i Z_j>/2
    prob = np.abs(state) ** 2
    
    sum_zz = 0.0
    for i, j in edge_list:
        # <Z_i Z_j> = sum_x prob[x] * Z_i(x) * Z_j(x)
        # Z_i(x) * Z_j(x) = 1 if bits i and j are the same, -1 if different
        zz = 0.0
        for x in range(dim):
            bi = (x >> i) & 1
            bj = (x >> j) & 1
            if bi == bj:
                zz += prob[x]
            else:
                zz -= prob[x]
        sum_zz += zz
    
    expected_cut = E - sum_zz / 2.0
    return expected_cut


def _objective(params, n, edges, optimum):
    """Negative ratio for minimization."""
    p = len(params) // 2
    gammas = params[:p]
    betas = params[p:]
    
    # Clip to reasonable range
    gammas = np.clip(gammas, 0.0, np.pi)
    betas = np.clip(betas, 0.0, np.pi)
    
    exp_cut = _qaoa_expectation(n, edges, gammas, betas)
    ratio = exp_cut / optimum
    return -ratio


def _nelder_mead(f, x0, args=(), maxiter=100, xatol=1e-4, fatol=1e-4, adaptive=True):
    """Simple Nelder-Mead implementation."""
    n = len(x0)
    # Initialize simplex
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        x1 = np.array(x0, dtype=float)
        step = 0.25 if x1[i] == 0 else 0.05 * abs(x1[i])
        x1[i] += step
        simplex.append(x1)
    
    # Evaluate initial simplex
    fvals = [f(x, *args) for x in simplex]
    
    # Sort simplex by function value
    order = np.argsort(fvals)
    simplex = [simplex[i] for i in order]
    fvals = [fvals[i] for i in order]
    
    alpha = 1.0  # reflection
    gamma = 2.0  # expansion
    rho = 0.5   # contraction
    sigma = 0.5 # shrink
    
    for it in range(maxiter):
        if adaptive and (time.time() - _START) > _DEADLINE - 5:
            break
        
        # Check convergence
        if abs(fvals[0] - fvals[-1]) < fatol:
            break
        if max(np.abs(simplex[0] - x) for x in simplex) < xatol:
            break
        
        # Reflection
        centroid = np.mean(simplex[:-1], axis=0)
        x_reflect = centroid + alpha * (centroid - simplex[-1])
        f_reflect = f(x_reflect, *args)
        
        if fvals[0] <= f_reflect < fvals[-2]:
            simplex[-1] = x_reflect
            fvals[-1] = f_reflect
        elif f_reflect < fvals[0]:
            # Expansion
            x_expand = centroid + gamma * (x_reflect - centroid)
            f_expand = f(x_expand, *args)
            if f_expand < f_reflect:
                simplex[-1] = x_expand
                fvals[-1] = f_expand
            else:
                simplex[-1] = x_reflect
                fvals[-1] = f_reflect
        else:
            # Contraction
            if f_reflect < fvals[-1]:
                x_contract = centroid + rho * (x_reflect - centroid)
                f_contract = f(x_contract, *args)
                if f_contract <= f_reflect:
                    simplex[-1] = x_contract
                    fvals[-1] = f_contract
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = f(simplex[i], *args)
            else:
                x_contract = centroid - rho * (simplex[-1] - centroid)
                f_contract = f(x_contract, *args)
                if f_contract < fvals[-1]:
                    simplex[-1] = x_contract
                    fvals[-1] = f_contract
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = f(simplex[i], *args)
        
        # Re-sort simplex
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
    
    return simplex[0], fvals[0]


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
    
    # Use the known good values from inspiration 1 as a strong starting point
    # gamma ~ 0.35, beta ~ 0.25
    x0 = np.array([0.35, 0.25, 0.35, 0.25])[:2*p]
    
    # Try multiple starting points including the known good values
    start_points = [
        x0,  # Known good from inspiration
        np.array([np.pi/4, np.pi/8, np.pi/8, np.pi/16])[:2*p],
        np.array([np.pi/6, np.pi/12, np.pi/12, np.pi/24])[:2*p],
        np.array([np.pi/3, np.pi/6, np.pi/6, np.pi/12])[:2*p],
        np.random.uniform(0.1, np.pi, 2*p),
        np.random.uniform(0.1, np.pi, 2*p),
    ]
    
    best_params = x0
    best_score = _objective(x0, n, edges, optimum)
    
    for x_init in start_points:
        if _check_deadline():
            break
        
        try:
            result_x, result_f = _nelder_mead(
                _objective,
                x_init,
                args=(n, edges, optimum),
                maxiter=150,
                xatol=1e-5,
                fatol=1e-5,
                adaptive=True
            )
            
            score = -result_f
            if score > -best_score:
                best_score = result_f
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
