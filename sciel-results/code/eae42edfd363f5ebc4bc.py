"""QAOA angle design for MaxCut with p=2 layers using graph-aware lookup and refinement."""
import numpy as np
from scipy.optimize import minimize
import time
import itertools

# Deadline guard: 150s wall-clock kill, use 140s as safety margin
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0

def _remaining_time():
    return _DEADLINE - time.time()

def _build_adjacency(n, edges):
    """Build adjacency matrix from edge list."""
    adj = np.zeros((n, n), dtype=np.float64)
    for u, v in edges:
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    return adj

def _exact_maxcut(n, edges):
    """Compute exact MaxCut via brute force for small n (n <= 10)."""
    m = len(edges)
    if m == 0:
        return 0.0
    best = 0
    # For n=10, 2^10 = 1024 subsets, feasible
    for mask in range(1 << n):
        cut = 0
        for u, v in edges:
            if (mask >> u) & 1 != (mask >> v) & 1:
                cut += 1
        if cut > best:
            best = cut
    return float(best)

def _expected_cut(n, edges, gammas, betas):
    """
    Compute expected cut for QAOA with given angles.
    Uses state vector simulation for n <= 10.
    C = sum_{(u,v) in edges} (1 - Z_u Z_v)/2, so <C> = m/2 - 0.5 * sum <Z_u Z_v>
    """
    m = len(edges)
    if m == 0:
        return 0.0
    
    # Initialize state: |+>^n
    # State vector of size 2^n
    dim = 1 << n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Precompute Z_u Z_v expectation helper
    # For each edge (u,v), <Z_u Z_v> = sum_x |psi_x|^2 * (-1)^{x_u + x_v}
    
    def apply_cost(gamma):
        nonlocal state
        # Apply exp(-i * gamma * C) where C = sum (1-Z_u Z_v)/2
        # C has eigenvalues 0 or 1 for each edge term, but they don't commute in general
        # Actually C = sum_edges (1 - Z_u Z_v)/2, so exp(-i*gamma*C) 
        # We need to compute this unitary. For small n, we can diagonalize or use the fact that
        # C is a sum of commuting terms? No, Z_u Z_v terms commute only if edges don't share vertices.
        # For general graph, we need to apply the full unitary.
        # Alternative: compute <C> directly without full state evolution? 
        # Actually, for expected value, we can use:
        # <psi| C |psi> where |psi> is the state after mixer
        # But we need the state after both cost and mixer.
        # Let's do full state vector simulation.
        
        # Build C matrix (2^n x 2^n) - expensive for n=10 (1024x1024)
        # Better: apply exp(-i*gamma*C) using the spectral decomposition or direct matrix exp
        # For n=10, 2^10 = 1024, so 1024x1024 matrix is 8MB, feasible
        
        # Build C as a matrix
        C = np.zeros((dim, dim), dtype=np.complex128)
        for u, v in edges:
            # (1 - Z_u Z_v)/2
            for x in range(dim):
                xu = (x >> u) & 1
                xv = (x >> v) & 1
                zuzv = 1 if xu == xv else -1
                C[x, x] += (1 - zuzv) / 2.0
        
        # Apply exp(-i * gamma * C) to state
        # Use scipy for matrix exponential
        from scipy.linalg import expm
        U_cost = expm(-1j * gamma * C)
        state = U_cost @ state
    
    def apply_mixer(beta):
        nonlocal state
        # Apply exp(-i * beta * sum X_j)
        # sum X_j = X_0 + X_1 + ... + X_{n-1}
        # Each X_j flips bit j
        # Build the mixer Hamiltonian
        H_mix = np.zeros((dim, dim), dtype=np.complex128)
        for j in range(n):
            for x in range(dim):
                y = x ^ (1 << j)
                H_mix[x, y] += 1.0
        
        from scipy.linalg import expm
        U_mix = expm(-1j * beta * H_mix)
        state = U_mix @ state
    
    # Apply layers
    for i in range(p):
        apply_cost(gammas[i])
        apply_mixer(betas[i])
    
    # Compute expected cut: <C> = sum_edges <(1 - Z_u Z_v)/2>
    prob = np.abs(state) ** 2
    expected = 0.0
    for u, v in edges:
        zz = 0.0
        for x in range(dim):
            xu = (x >> u) & 1
            xv = (x >> v) & 1
            zuzv = 1 if xu == xv else -1
            zz += prob[x] * zuzv
        expected += (1 - zz) / 2.0
    return expected

def _expected_cut_fast(n, edges, gammas, betas):
    """
    Faster expected cut using the fact that for p=2, we can compute <Z_u Z_v> 
    more efficiently. Actually, let's use a different approach:
    For QAOA, <Z_u Z_v> after the circuit can be computed from the state.
    But for n=10, the state vector approach is O(2^n * n) per expectation, which is fine.
    The bottleneck is the matrix exponential. Let's precompute or use a faster method.
    
    Alternative: use the fact that exp(-i*beta*sum X) has a known form.
    sum X_j = X_0 + ... + X_{n-1}. This is not diagonal in computational basis.
    But exp(-i*beta*sum X) = product? No, X_j don't commute.
    
    Actually, let's just use the state vector approach but optimize the matrix exponential.
    For n=10, 1024x1024 expm is fast enough.
    """
    return _expected_cut(n, edges, gammas, betas)

def _objective(params, n, edges, exact_opt):
    """Objective to minimize: negative of the ratio."""
    gammas = params[:2]
    betas = params[2:]
    # Keep angles in reasonable range
    expected = _expected_cut_fast(n, edges, gammas, betas)
    if exact_opt == 0:
        return 0.0
    return -expected / exact_opt

def _optimize_angles_for_graph(n, edges):
    """Optimize angles for a single graph."""
    exact_opt = _exact_maxcut(n, edges)
    if exact_opt == 0:
        return np.array([[0.35, 0.25], [0.35, 0.25]])
    
    best_params = np.array([0.35, 0.35, 0.25, 0.25])
    best_val = _objective(best_params, n, edges, exact_opt)
    
    # Try multiple initial points
    init_points = [
        np.array([0.35, 0.35, 0.25, 0.25]),
        np.array([0.5, 0.5, 0.3, 0.3]),
        np.array([0.2, 0.2, 0.1, 0.1]),
        np.array([0.7, 0.7, 0.5, 0.5]),
        np.array([0.1, 0.4, 0.2, 0.1]),
        np.array([0.4, 0.1, 0.1, 0.4]),
    ]
    
    for init in init_points:
        if _remaining_time() < 5.0:
            break
        try:
            res = minimize(_objective, init, args=(n, edges, exact_opt),
                          method='Nelder-Mead',
                          options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-5})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    return np.array([[best_params[0], best_params[2]],
                     [best_params[1], best_params[3]]])

def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles.
    Returns array of shape (p, 2) with rows (gamma, beta) in radians.
    """
    if p != 2:
        return np.tile([0.35, 0.25], (p, 1))
    
    # Check time budget
    if _remaining_time() < 10.0:
        # Not enough time for optimization, return heuristic
        return np.array([[0.35, 0.25], [0.35, 0.25]])
    
    try:
        return _optimize_angles_for_graph(n, edges)
    except Exception:
        return np.array([[0.35, 0.25], [0.35, 0.25]])
