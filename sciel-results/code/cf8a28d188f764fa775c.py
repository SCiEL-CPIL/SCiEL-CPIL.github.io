"""LJ38 cluster global minimum search with basin hopping from FCC and icosahedral seeds."""
import numpy as np
import time
from scipy.optimize import minimize

N = 38
DEADLINE = 145.0  # safety margin before 150s kill


def _lj_energy_and_grad(pos):
    """Compute total Lennard-Jones energy and gradient."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=-1)
    r2 = np.where(r2 < 1e-12, 1e-12, r2)
    r6 = r2 ** 3
    r12 = r6 ** 2
    energy = 4.0 * np.sum(1.0 / r12 - 1.0 / r6) - 4.0 * n * (1.0 / 1.0 - 1.0)  # remove self terms
    # Actually, let's do it properly:
    mask = ~np.eye(n, dtype=bool)
    r2_m = r2[mask]
    r6_m = r2_m ** 3
    r12_m = r6_m ** 2
    energy = 4.0 * np.sum(1.0 / r12_m - 1.0 / r6_m)
    
    coeff = 4.0 * (-12.0 / (r2 ** 7) + 6.0 / (r2 ** 4))
    np.fill_diagonal(coeff, 0.0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return energy, grad


def _lj_energy(pos):
    """Compute total Lennard-Jones energy only."""
    e, _ = _lj_energy_and_grad(pos)
    return e


def _lj_grad(pos):
    """Compute gradient of total Lennard-Jones energy only."""
    _, g = _lj_energy_and_grad(pos)
    return g


def _relax(pos, max_iter=2000, tol=1e-10):
    """L-BFGS local relaxation."""
    res = minimize(
        _lj_energy,
        pos,
        jac=_lj_grad,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol, 'maxls': 50}
    )
    return res.x


def _make_fcc_truncated_octahedron():
    """Generate FCC truncated octahedron positions for 38 atoms.
    
    The FCC truncated octahedron for N=38 consists of:
    - 1 center atom
    - 12 atoms at face positions (±1, ±1, 0) type in FCC
    - 24 atoms at edge positions
    
    Using FCC lattice with nearest neighbor distance = 1:
    FCC basis: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5) * a
    where a = sqrt(2) gives NN distance = 1.
    
    Alternatively, use the integer FCC representation:
    Points where i+j+k is even, scaled by 1/sqrt(2).
    """
    a = np.sqrt(2.0)  # lattice constant for NN = 1
    
    points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                if (i + j + k) % 2 == 0:
                    points.append([i, j, k])
    
    pts = np.array(points, dtype=float) * (a / np.sqrt(2.0))  # scale so NN = 1
    
    # Sort by distance to origin
    dists = np.sum(pts ** 2, axis=1)
    sorted_indices = np.argsort(dists)
    
    cluster = pts[sorted_indices[:N]].copy()
    
    # Center of mass to origin
    cluster -= np.mean(cluster, axis=0)
    
    # Optimal scale: r_min = 2^(1/6), and NN in our construction = 1
    # So we need to scale by r_min
    r_min = 2.0 ** (1.0 / 6.0)
    cluster *= r_min
    
    return cluster


def _make_icosahedral_structure():
    """Generate icosahedral structure for 38 atoms.
    
    Icosahedral shell: 1 center + 12 inner + 30 outer = 43 (too many)
    Or: 1 center + 12 inner + 25 outer = 38
    
    Standard icosahedron vertices (radius R):
    (0, ±1, ±φ), (±1, ±φ, 0), (±φ, 0, ±1) where φ = (1+√5)/2
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    # 12 icosahedron vertices
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize to unit radius
    norms = np.linalg.norm(ico_verts, axis=1, keepdims=True)
    ico_verts = ico_verts / norms
    
    # 1 center atom
    center = np.array([[0.0, 0.0, 0.0]])
    
    # 12 inner atoms (icosahedron vertices)
    inner = ico_verts.copy()
    
    # 30 outer atoms: positions at midpoints of icosahedron edges, pushed out
    # 30 edges of icosahedron
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(ico_verts[i] - ico_verts[j])
            if d < 1.0:  # edge length ~ 2 in unit sphere
                edges.append((i, j))
    
    # Actually, let's use a simpler approach: place 30 atoms on the next shell
    # The 30 vertices of a truncated icosahedron or just use edge midpoints
    edge_midpoints = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(ico_verts[i] - ico_verts[j])
            if d < 0.5:  # adjacent vertices
                mid = (ico_verts[i] + ico_verts[j]) / 2.0
                mid_norm = np.linalg.norm(mid)
                if mid_norm > 1e-6:
                    edge_midpoints.append(mid / mid_norm)
    
    # Deduplicate
    unique_mids = []
    for m in edge_midpoints:
        if not any(np.linalg.norm(m - u) < 0.01 for u in unique_mids):
            unique_mids.append(m)
    
    outer = np.array(unique_mids[:30]) if len(unique_mids) >= 30 else np.array(unique_mids)
    
    # If we don't have exactly 30, pad with more
    if len(outer) < 30:
        # Add more points on the sphere
        rng = np.random.RandomState(42)
        while len(outer) < 30:
            v = rng.randn(3)
            v = v / np.linalg.norm(v)
            if not any(np.linalg.norm(v - u) < 0.1 for u in outer):
                outer = np.vstack([outer, v.reshape(1, -1)])
    
    outer = outer[:30]
    
    # Scale: inner shell radius = 1, outer shell radius = ~1.6
    inner *= 1.0
    outer *= 1.6
    
    cluster = np.vstack([center, inner, outer])
    
    # Center of mass
    cluster -= np.mean(cluster, axis=0)
    
    # Scale for LJ optimum
    r_min = 2.0 ** (1.0 / 6.0)
    cluster *= r_min
    
    return cluster


def _ensure_min_sep(pos, min_sep=0.5):
    """Ensure all atoms are at least min_sep apart."""
    n = len(pos)
    for _ in range(100):
        min_d = np.inf
        min_pair = (0, 1)
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_d:
                    min_d = d
                    min_pair = (i, j)
        
        if min_d >= min_sep:
            break
        
        # Push apart
        i, j = min_pair
        diff = pos[i] - pos[j]
        d = np.linalg.norm(diff)
        if d < 1e-10:
            diff = np.array([1.0, 0.0, 0.0])
            d = 1.0
        push = (min_sep - d) / d * 0.5
        pos[i] += diff * push
        pos[j] -= diff * push
    
    return pos


def _basin_hop(pos, rng, n_hops=50, step_size=0.3):
    """Basin hopping: perturb + relax, keep best."""
    best_pos = pos.copy()
    best_energy = _lj_energy(pos)
    
    for hop in range(n_hops):
        # Perturb
        perturbed = best_pos + rng.randn(*best_pos.shape) * step_size
        perturbed = _ensure_min_sep(perturbed)
        
        # Relax
        relaxed = _relax(perturbed, max_iter=500, tol=1e-8)
        energy = _lj_energy(relaxed)
        
        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed
    
    return best_pos, best_energy


def construct():
    start_time = time.time()
    rng = np.random.RandomState(12345)
    
    # Start with FCC truncated octahedron
    fcc_pos = _make_fcc_truncated_octahedron()
    fcc_pos = _ensure_min_sep(fcc_pos)
    
    # Relax FCC
    relaxed_fcc = _relax(fcc_pos, max_iter=3000, tol=1e-12)
    best_pos = relaxed_fcc.copy()
    best_energy = _lj_energy(relaxed_fcc)
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 20:
        return best_pos
    
    # Basin hopping from FCC
    bh_pos, bh_energy = _basin_hop(relaxed_fcc, rng, n_hops=30, step_size=0.2)
    if bh_energy < best_energy:
        best_energy = bh_energy
        best_pos = bh_pos
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 20:
        return best_pos
    
    # Try icosahedral start
    ico_pos = _make_icosahedral_structure()
    ico_pos = _ensure_min_sep(ico_pos)
    relaxed_ico = _relax(ico_pos, max_iter=2000, tol=1e-10)
    ico_energy = _lj_energy(relaxed_ico)
    
    if ico_energy < best_energy:
        best_energy = ico_energy
        best_pos = relaxed_ico
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 20:
        return best_pos
    
    # Basin hopping from icosahedral
    bh_ico_pos, bh_ico_energy = _basin_hop(relaxed_ico, rng, n_hops=20, step_size=0.3)
    if bh_ico_energy < best_energy:
        best_energy = bh_ico_energy
        best_pos = bh_ico_pos
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 15:
        return best_pos
    
    # Additional basin hopping from best with smaller steps
    bh2_pos, bh2_energy = _basin_hop(best_pos, rng, n_hops=30, step_size=0.15)
    if bh2_energy < best_energy:
        best_energy = bh2_energy
        best_pos = bh2_pos
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 10:
        return best_pos
    
    # Final high-precision polish
    polished = _relax(best_pos, max_iter=5000, tol=1e-14)
    polished_energy = _lj_energy(polished)
    
    if polished_energy < best_energy:
        best_pos = polished
        best_energy = polished_energy
    
    # Ensure minimum separation
    best_pos = _ensure_min_sep(best_pos)
    
    return best_pos
