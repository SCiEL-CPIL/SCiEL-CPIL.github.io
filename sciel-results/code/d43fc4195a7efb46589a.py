"""Improved Lennard-Jones cluster global minimum, 55 atoms."""
N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

# Deadline guard: 150s wall-clock limit, use 145s for safety
_DEADLINE = 145.0
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _DEADLINE - _elapsed()


def _lj_energy_vec(pos):
    """Compute total Lennard-Jones energy using vectorized operations."""
    # pos shape: (N, 3)
    # Compute all pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (N, N, 3)
    r2 = np.sum(diff * diff, axis=2)  # (N, N)
    mask = ~np.eye(len(pos), dtype=bool)
    r2 = np.where(mask, r2, 1e10)
    
    r6 = r2 ** -3
    r12 = r6 ** 2
    
    e = 4.0 * np.sum(r12 - r6) / 2.0
    return e


def _lj_grad_vec(pos):
    """Compute gradient of LJ energy using vectorized operations."""
    N = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (N, N, 3)
    r2 = np.sum(diff * diff, axis=2)  # (N, N)
    
    mask = ~np.eye(N, dtype=bool)
    r2_safe = np.where(mask, r2, 1.0)
    
    r6 = r2_safe ** -3
    r12 = r6 ** 2
    
    coeff = 4.0 * (-12.0 * r12 + 6.0 * r6) / r2_safe  # (N, N)
    coeff = np.where(mask, coeff, 0.0)
    
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return grad


def _lj_energy_and_grad(pos):
    """Combined energy and gradient for L-BFGS."""
    e = _lj_energy_vec(pos)
    g = _lj_grad_vec(pos)
    return e, g.ravel()


def _check_min_dist(pos, min_d=0.5):
    """Check all pair distances >= min_d."""
    d = pdist(pos)
    return d.min() >= min_d - 1e-9


def _relax_lbfgs(pos, maxiter=1000, tol=1e-10):
    """L-BFGS-B local relaxation with vectorized objective."""
    if _remaining() < 5.0:
        return pos
    
    def fun(x):
        x = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(x)
        return e, g
    
    x0 = pos.reshape(-1)
    try:
        result = minimize(fun, x0, method='L-BFGS-B', jac=True,
                          options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': tol, 'maxls': 50})
        if result.success or result.fun < _lj_energy_vec(pos):
            return result.x.reshape(-1, 3)
    except Exception:
        pass
    return pos


def _make_mackay_icosahedron_55(r1=1.122, r2=2.244):
    """Construct a Mackay icosahedron with specified shell radii."""
    phi = (1 + np.sqrt(5)) / 2.0
    
    # Icosahedron vertices (12)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # Shell 1: 12 icosahedron vertices at radius r1
    shell1 = verts * r1
    
    # Find edges of the icosahedron
    from scipy.spatial.distance import cdist
    D = cdist(verts, verts)
    np.fill_diagonal(D, np.inf)
    min_dist = D.min()
    
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(D[i, j] - min_dist) < 1e-10:
                edges.append((i, j))
    
    # Edge midpoint directions (30 edges)
    edge_mids = []
    for i, j in edges:
        mid_dir = (verts[i] + verts[j]) / np.linalg.norm(verts[i] + verts[j])
        edge_mids.append(mid_dir)
    edge_mids = np.array(edge_mids)  # shape (30, 3)
    
    # Vertex directions for outer shell (12 points)
    vertex_outer = verts * r2
    
    # Combine: 12 vertex-outer + 30 edge-midpoint-outer = 42
    shell2 = np.vstack([vertex_outer, edge_mids * r2])
    
    # Center atom
    center = np.array([[0.0, 0.0, 0.0]])
    
    positions = np.vstack([center, shell1, shell2])
    return positions


def _optimize_radial_parameters():
    """Optimize the radial parameters of the Mackay icosahedron."""
    if _remaining() < 10.0:
        return 1.122, 2.244
    
    def radial_energy(params):
        r1, r2 = params
        if r1 < 0.5 or r2 < 0.5:
            return 1e10
        pos = _make_mackay_icosahedron_55(r1, r2)
        return _lj_energy_vec(pos)
    
    try:
        result = minimize(radial_energy, [1.122, 2.244], method='Nelder-Mead',
                          options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10})
        r1, r2 = result.x
        if r1 > 0.5 and r2 > 0.5:
            return r1, r2
    except Exception:
        pass
    return 1.122, 2.244


def _multi_start_search(base_pos, n_starts=20, noise_scale=0.05):
    """Multi-start stochastic search with Gaussian perturbations."""
    best_pos = None
    best_energy = np.inf
    rng = np.random.default_rng(42)
    
    for i in range(n_starts):
        if _remaining() < 15.0:
            break
        
        # Create perturbed version
        noise = rng.normal(0, noise_scale, base_pos.shape)
        perturbed = base_pos + noise
        
        # Relax
        relaxed = _relax_lbfgs(perturbed, maxiter=300, tol=1e-10)
        e = _lj_energy_vec(relaxed)
        
        if _check_min_dist(relaxed) and e < best_energy:
            best_energy = e
            best_pos = relaxed.copy()
    
    return best_pos, best_energy


def _make_fcc_cluster(n=55, seed=None):
    """Generate an FCC-like cluster."""
    rng = np.random.default_rng(seed)
    a = 1.12
    pos = []
    for i in range(-2, 5):
        for j in range(-2, 5):
            for k in range(-2, 5):
                for offset in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    p = np.array([i, j, k]) * a + np.array(offset) * a
                    pos.append(p)
    
    pos = np.array(pos)
    dists = np.linalg.norm(pos, axis=1)
    indices = np.argsort(dists)
    pos = pos[indices[:n]]
    pos += rng.normal(0, 0.01, pos.shape)
    return pos


def construct():
    global _start_time
    _start_time = time.time()
    
    best_pos = None
    best_energy = np.inf
    
    # Step 1: Optimize radial parameters of Mackay icosahedron
    if _remaining() > 30.0:
        try:
            r1_opt, r2_opt = _optimize_radial_parameters()
        except Exception:
            r1_opt, r2_opt = 1.122, 2.244
    else:
        r1_opt, r2_opt = 1.122, 2.244
    
    # Step 2: Create optimized Mackay icosahedron
    if _remaining() > 20.0:
        try:
            mackay_opt = _make_mackay_icosahedron_55(r1_opt, r2_opt)
            relaxed = _relax_lbfgs(mackay_opt, maxiter=500)
            e = _lj_energy_vec(relaxed)
            if _check_min_dist(relaxed) and e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
        except Exception:
            pass
    
    # Step 3: Multi-start search from optimized Mackay
    if best_pos is not None and _remaining() > 30.0:
        try:
            # Use remaining time to determine number of starts
            time_per_start = 8.0  # Estimated time per start
            n_starts = max(1, int((_remaining() - 20.0) / time_per_start))
            n_starts = min(n_starts, 30)  # Cap at 30 starts
            
            multi_pos, multi_energy = _multi_start_search(best_pos, n_starts=n_starts, noise_scale=0.03)
            if multi_pos is not None and multi_energy < best_energy:
                best_energy = multi_energy
                best_pos = multi_pos.copy()
        except Exception:
            pass
    
    # Step 4: Try FCC as alternative
    if _remaining() > 20.0:
        try:
            fcc = _make_fcc_cluster(55, seed=42)
            relaxed = _relax_lbfgs(fcc, maxiter=300)
            e = _lj_energy_vec(relaxed)
            if _check_min_dist(relaxed) and e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
        except Exception:
            pass
    
    # Final Polish on the best result
    if best_pos is not None and _remaining() > 10.0:
        try:
            # Multi-stage polish with progressively tighter tolerance
            polished = _relax_lbfgs(best_pos, maxiter=500, tol=1e-11)
            e = _lj_energy_vec(polished)
            if _check_min_dist(polished) and e < best_energy:
                best_energy = e
                best_pos = polished.copy()
            
            if _remaining() > 5.0:
                polished2 = _relax_lbfgs(best_pos, maxiter=1000, tol=1e-13)
                e2 = _lj_energy_vec(polished2)
                if _check_min_dist(polished2) and e2 < best_energy:
                    best_energy = e2
                    best_pos = polished2.copy()
        except Exception:
            pass
    
    if best_pos is None:
        # Fallback: original cubic lattice
        side = int(np.ceil(N ** (1 / 3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        best_pos = np.array(points[:N], dtype=float) * 1.12
    
    return best_pos
# EVOLVE-BLOCK-END
