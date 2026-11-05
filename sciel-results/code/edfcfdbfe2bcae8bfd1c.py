"""Dzugutov cluster global minima via annealed gradient descent with icosahedral seeds."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax

_DEADLINE = 330.0
_START = time.time()

def _remaining():
    return _DEADLINE - time.time()

def _dzugutov_energy(pos, A=5.82, B=1.28, a=1.87, b=1.94, c=1.10, d=0.27, m=16):
    """Vectorized Dzugutov energy."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    iu = np.triu_indices(n, 1)
    r = dist[iu]
    V = np.zeros(len(r))
    ma = r < a
    mb = (r < b) & ~ma
    if np.any(ma):
        ra = r[ma]
        V[ma] = A * (ra**(-m) - B) * np.exp(c / (ra - a))
    if np.any(mb):
        rb = r[mb]
        V[mb] = B * np.exp(d / (rb - b))
    return float(np.sum(V))

def _dzugutov_grad(pos, c=1.10, A=5.82, B=1.28, a=1.87, b=1.94, d=0.27, m=16):
    """Fully vectorized Dzugutov gradient."""
    n = len(pos)
    if n < 2:
        return np.zeros_like(pos)
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    iu = np.triu_indices(n, 1)
    r = dist[iu]
    rij = diff[iu]  # shape (n_pairs, 3)
    
    force = np.zeros((len(r), 3))
    ma = r < a
    mb = (r < b) & ~ma
    
    if np.any(ma):
        ra = r[ma]
        exp_c = np.exp(c / (ra - a))
        dVdr = A * (-m * ra**(-m-1)) * exp_c + A * (ra**(-m) - B) * exp_c * (-c / (ra-a)**2)
        force[ma] = -dVdr[:, None] * rij[ma] / ra[:, None]
    
    if np.any(mb):
        rb = r[mb]
        exp_d = np.exp(d / (rb - b))
        dVdr = B * exp_d * (-d / (rb-b)**2)
        force[mb] = -dVdr[:, None] * rij[mb] / rb[:, None]
    
    # Scatter forces back
    grad = np.zeros_like(pos)
    for idx in range(len(r)):
        i, j = iu[0][idx], iu[1][idx]
        grad[i] += force[idx]
        grad[j] -= force[idx]
    return grad

def _icosahedron_seed(n, rng):
    """Generate an icosahedral-like seed structure."""
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [-1, 0, phi], [1, 0, phi], [-1, 0, -phi], [1, 0, -phi],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [0, phi, -1], [0, phi, 1], [0, -phi, -1], [0, -phi, 1]
    ], dtype=float)
    norm = np.linalg.norm(ico[0])
    ico = ico / norm * 1.05
    
    points = list(ico[:min(12, n)])
    
    while len(points) < n:
        added = False
        for _ in range(20):
            if len(points) > 0:
                base = points[rng.integers(0, len(points))]
                p = base + rng.normal(0, 0.5, 3)
            else:
                p = rng.normal(0, 1.0, 3)
            if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                points.append(p)
                added = True
                break
        if not added:
            # Force add with larger displacement
            if len(points) > 0:
                base = points[rng.integers(0, len(points))]
                p = base + rng.normal(0, 1.0, 3)
            else:
                p = rng.normal(0, 2.0, 3)
            points.append(p)
    
    return np.array(points[:n])

def _anneal_grad_descent(pos0, n, rng):
    """Annealed gradient descent with cosine schedule on c."""
    c_start = 4.0
    c_end = 1.10
    d_val = 0.27
    steps = 400
    
    pos = pos0.copy()
    best_pos = pos.copy()
    best_e = _dzugutov_energy(pos)
    
    for step in range(steps):
        if _remaining() < 8.0:
            break
        t = step / steps
        c_val = c_end + (c_start - c_end) * 0.5 * (1 + np.cos(np.pi * t))
        lr = 0.003 * (1.0 - 0.7 * t) + 0.0003
        
        grad = _dzugutov_grad(pos, c=c_val)
        pos = pos - lr * grad
        
        # Enforce minimum distance
        for i in range(n):
            for j in range(i+1, n):
                rij = pos[i] - pos[j]
                d = np.linalg.norm(rij)
                if d < 0.5 and d > 1e-10:
                    corr = (0.5 - d) / (2*d) * rij
                    pos[i] += corr
                    pos[j] -= corr
        
        e = _dzugutov_energy(pos)
        if e < best_e:
            best_e = e
            best_pos = pos.copy()
    
    return best_pos, best_e

def construct(instance):
    """Find low-energy Dzugutov cluster using annealed gradient descent."""
    n = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(n * 13 + 7)
    
    best_energy = np.inf
    best_positions = None
    
    n_seeds = 4
    for s in range(n_seeds):
        if _remaining() < 12.0:
            break
        
        seed = _icosahedron_seed(n, rng)
        seed = seed + rng.normal(0, 0.08 * (s+1), 3)
        
        pos, energy = _anneal_grad_descent(seed, n, rng)
        
        # Final local relaxation
        try:
            e_rel, pos_rel = relax(pos, potential, parameter)
        except:
            e_rel, pos_rel = energy, pos
        
        if e_rel < best_energy:
            best_energy = e_rel
            best_positions = pos_rel
    
    if best_positions is None:
        best_positions = rng.normal(0, 1.0, (n, 3))
    
    return best_positions
# EVOLVE-BLOCK-END
