"""Quantum information: fixed-depth QAOA parameter design.
Graph-dependent two-layer QAOA angles via efficient state-vector simulation,
multi-start local optimization with physics-informed initializations, and
strict wall-clock deadline guard.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 150.0
_START = time.time()
_MARGIN = 15.0  # safety margin seconds

def _time_left():
    return _DEADLINE - _MARGIN - (time.time() - _START)


def _build_adj(edges, n):
    adj = np.zeros((n, n), dtype=np.float64)
    for e in edges:
        a, b = int(e[0]), int(e[1])
        adj[a, b] = 1.0
        adj[b, a] = 1.0
    return adj


def _precompute_graph_data(adj, n):
    """Precompute cut values and edge list."""
    num_states = 1 << n
    states = np.arange(num_states, dtype=np.int64)
    
    cuts = np.zeros(num_states, dtype=np.float64)
    edge_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0:
                edge_list.append((i, j))
                bits_i = (states >> i) & 1
                bits_j = (states >> j) & 1
                mask = bits_i != bits_j
                cuts[mask] += 1
    
    m = len(edge_list)
    return cuts, edge_list, m


def _expected_cut_fast(cuts, edge_list, m_edges, n, gammas, betas):
    """Compute expected cut using efficient state-vector evolution."""
    num_states = 1 << n
    
    # Start in |+>^n
    psi = np.ones(num_states, dtype=np.complex128) / np.sqrt(num_states)
    
    for layer in range(2):
        g = gammas[layer]
        b = betas[layer]
        
        # Apply cost phase: exp(-i * gamma * cut_size)
        psi = psi * np.exp(-1j * g * cuts)
        
        # Apply mixer: exp(-i beta sum_i X_i)
        c = np.cos(b)
        s = np.sin(b)
        
        # Apply U = [[c, -i*s], [-i*s, c]] on each qubit
        psi_t = psi.reshape([2] * n)
        for q in range(n):
            psi_t = np.moveaxis(psi_t, q, 0)
            flat = psi_t.reshape(2, -1)
            new = np.empty_like(flat)
            new[0, :] = c * flat[0, :] - 1j * s * flat[1, :]
            new[1, :] = -1j * s * flat[0, :] + c * flat[1, :]
            psi_t = new.reshape([2] * n)
            psi_t = np.moveaxis(psi_t, 0, q)
        psi = psi_t.reshape(num_states)
    
    # Compute expected cut: E[cut] = sum_edges P(diff)
    probs = np.abs(psi) ** 2
    expected_cut = 0.0
    for (i, j) in edge_list:
        # P(x_i != x_j) = 0.5 - 0.5 * <Z_i Z_j>
        # <Z_i Z_j> = sum_x probs[x] * Z_i[x] * Z_j[x]
        # Z_i[x] = +1 if bit i is 0, -1 if bit i is 1
        # So Z_i * Z_j = +1 if bits equal, -1 if bits differ
        # P(diff) = 0.5 * (1 - <Z_i Z_j>)
        # Alternatively: P(diff) = sum over states where bits differ of probs
        states = np.arange(num_states, dtype=np.int64)
        bits_i = (states >> i) & 1
        bits_j = (states >> j) & 1
        diff_mask = bits_i != bits_j
        expected_cut += np.sum(probs[diff_mask])
    
    return expected_cut


def _expected_cut_vectorized(cuts, edge_list, m_edges, n, gammas, betas):
    """Compute expected cut using precomputed difference masks for speed."""
    num_states = 1 << n
    states = np.arange(num_states, dtype=np.int64)
    
    psi = np.ones(num_states, dtype=np.complex128) / np.sqrt(num_states)
    
    for layer in range(2):
        g = gammas[layer]
        b = betas[layer]
        
        psi = psi * np.exp(-1j * g * cuts)
        
        c = np.cos(b)
        s = np.sin(b)
        
        psi_t = psi.reshape([2] * n)
        for q in range(n):
            psi_t = np.moveaxis(psi_t, q, 0)
            flat = psi_t.reshape(2, -1)
            new = np.empty_like(flat)
            new[0, :] = c * flat[0, :] - 1j * s * flat[1, :]
            new[1, :] = -1j * s * flat[0, :] + c * flat[1, :]
            psi_t = new.reshape([2] * n)
            psi_t = np.moveaxis(psi_t, 0, q)
        psi = psi_t.reshape(num_states)
    
    probs = np.abs(psi) ** 2
    expected_cut = 0.0
    for (i, j) in edge_list:
        bits_i = (states >> i) & 1
        bits_j = (states >> j) & 1
        diff_mask = bits_i != bits_j
        expected_cut += np.sum(probs[diff_mask])
    
    return expected_cut


def _negative_expected_cut(params, cuts, edge_list, m_edges, n):
    """Negative expected cut for minimization."""
    gammas = np.array([params[0], params[2]])
    betas = np.array([params[1], params[3]])
    return -_expected_cut_vectorized(cuts, edge_list, m_edges, n, gammas, betas)


def _physics_informed_init(n, m_edges):
    """Generate physics-informed initial angle sets based on QAOA asymptotics."""
    inits = []
    
    # Standard asymptotic values for p=2 MaxCut
    # From literature: gamma1 ~ 0.35-0.45, beta1 ~ 0.25-0.35, gamma2 ~ 0.25-0.40, beta2 ~ 0.15-0.30
    base_sets = [
        [0.40, 0.30, 0.35, 0.25],
        [0.35, 0.25, 0.30, 0.20],
        [0.45, 0.35, 0.40, 0.30],
        [0.30, 0.20, 0.25, 0.15],
        [0.50, 0.40, 0.45, 0.35],
        [0.25, 0.15, 0.20, 0.10],
    ]
    
    # Add some perturbations
    rng = np.random.RandomState(42)
    for base in base_sets:
        inits.append(np.array(base))
        # Small perturbation
        perturbed = np.array(base) + rng.uniform(-0.05, 0.05, 4)
        inits.append(perturbed)
    
    # Graph-dependent scaling: larger graphs may need slightly different angles
    density = 2.0 * m_edges / (n * (n - 1)) if n > 1 else 0.0
    scale = 1.0 + 0.1 * (density - 0.5)
    for base in base_sets[:3]:
        scaled = np.array(base) * scale
        inits.append(scaled)
    
    return inits


def _coarse_grid_search(cuts, edge_list, m_edges, n, max_time=3.0):
    """Coarse grid search to find a good starting point."""
    t0 = time.time()
    best_val = -1.0
    best_angles = np.array([[0.35, 0.25], [0.30, 0.20]])
    
    # Coarser grid for speed
    gamma_grid = np.linspace(0.1, 0.8, 8)
    beta_grid = np.linspace(0.05, 0.6, 8)
    
    for g1 in gamma_grid:
        if time.time() - t0 > max_time:
            break
        for b1 in beta_grid:
            if time.time() - t0 > max_time:
                break
            for g2 in gamma_grid:
                if time.time() - t0 > max_time:
                    break
                for b2 in beta_grid:
                    if time.time() - t0 > max_time:
                        break
                    val = _expected_cut_vectorized(cuts, edge_list, m_edges, n, 
                                                   np.array([g1, g2]), np.array([b1, b2]))
                    if val > best_val:
                        best_val = val
                        best_angles = np.array([[g1, b1], [g2, b2]])
    
    return best_angles, best_val


def _multi_start_optimize(cuts, edge_list, m_edges, n, max_time=10.0):
    """Multi-start local optimization with physics-informed initializations."""
    t0 = time.time()
    
    # Get initializations
    inits = _physics_informed_init(n, m_edges)
    
    # Add grid search result
    grid_angles, grid_val = _coarse_grid_search(cuts, edge_list, m_edges, n, max_time=2.0)
    inits.append(np.array([grid_angles[0,0], grid_angles[0,1], grid_angles[1,0], grid_angles[1,1]]))
    
    best_val = -1.0
    best_x = None
    
    bounds = [(-1.0, 1.5), (-1.0, 1.5), (-1.0, 1.5), (-1.0, 1.5)]
    
    for x0 in inits:
        if time.time() - t0 > max_time:
            break
        
        # Ensure within bounds
        x0_clipped = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        
        def obj(x):
            return _negative_expected_cut(x, cuts, edge_list, m_edges, n)
        
        try:
            # L-BFGS-B
            res = minimize(obj, x0_clipped, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res.fun < obj(x0_clipped):
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_x = res.x.copy()
        except Exception:
            pass
        
        # Try Nelder-Mead from the best point found so far
        if best_x is not None and time.time() - t0 < max_time * 0.7:
            try:
                res2 = minimize(obj, best_x, method='Nelder-Mead',
                               options={'maxiter': 50, 'xatol': 1e-12, 'fatol': 1e-12})
                val2 = -res2.fun
                if val2 > best_val:
                    best_val = val2
                    best_x = res2.x.copy()
            except Exception:
                pass
    
    if best_x is None:
        # Fallback to grid result
        best_x = np.array([grid_angles[0,0], grid_angles[0,1], grid_angles[1,0], grid_angles[1,1]])
        best_val = grid_val
    
    angles = np.array([[best_x[0], best_x[1]], [best_x[2], best_x[3]]])
    return angles, best_val


# Cache for per-graph optimization results
_graph_cache = {}


def angles(n, edges, p):
    if time.time() - _START > _DEADLINE - _MARGIN:
        # Emergency fallback
        return np.tile([0.35, 0.25], (p, 1))

    adj = _build_adj(edges, n)
    m = len(edges)

    # Check cache
    cache_key = (n, tuple(sorted(tuple(e) for e in edges)))
    if cache_key in _graph_cache:
        return _graph_cache[cache_key]

    # Default good angles
    default = np.array([[0.40, 0.30], [0.35, 0.25]], dtype=np.float64)

    # Try per-graph optimization if time permits
    if _time_left() > 15.0 and n <= 10:
        try:
            cuts, edge_list, m_edges = _precompute_graph_data(adj, n)
            
            max_opt_time = min(12.0, _time_left() * 0.6)
            angles_opt, val_opt = _multi_start_optimize(cuts, edge_list, m_edges, n, 
                                                        max_time=max_opt_time)
            result = angles_opt
        except Exception:
            result = default
    else:
        result = default

    # Ensure shape (p, 2)
    if result.shape != (p, 2):
        result = default

    _graph_cache[cache_key] = result
    return result
# EVOLVE-BLOCK-END
