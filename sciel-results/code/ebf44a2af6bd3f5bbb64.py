"""Structure-motif + adaptive basin hopping for Sutton-Chen cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = 325.0  # safety margin before 330s kill


def _time_left():
    """Return remaining time in seconds."""
    return _DEADLINE - (time.time() - _START_TIME)


def _fcc_cluster(n, a=0.72):
    """Generate an FCC-like cluster with n atoms."""
    pts = []
    max_idx = 12
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                # FCC lattice points
                for di, dj, dk in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    p = a * np.array([i + di, j + dj, k + dk])
                    pts.append(p)
    pts = np.array(pts)
    if len(pts) < n:
        return pts
    # Sort by distance from center and take n closest
    center = pts.mean(axis=0)
    dists = np.linalg.norm(pts - center, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx]


def _hcp_cluster(n, a=0.72):
    """Generate an HCP-like cluster with n atoms."""
    pts = []
    max_idx = 10
    c = a * np.sqrt(2.0 / 3.0)
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                # HCP: ABAB stacking
                if k % 2 == 0:
                    p = a * np.array([i, j, k * c / a])
                else:
                    p = a * np.array([i + 0.5, j + 0.5, k * c / a])
                pts.append(p)
    pts = np.array(pts)
    if len(pts) < n:
        return pts
    center = pts.mean(axis=0)
    dists = np.linalg.norm(pts - center, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx]


def _icosa_cluster(n, a=0.72):
    """Generate an icosahedral-like cluster with n atoms."""
    pts = []
    phi = (1 + np.sqrt(5)) / 2
    # First shell: 12 vertices of icosahedron
    shell1 = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]
    for s in shell1:
        pts.append(a * np.array(s))
    # Second shell: 30 atoms at midpoints of edges
    for i in range(12):
        for j in range(i + 1, 12):
            p1 = np.array(shell1[i], dtype=float)
            p2 = np.array(shell1[j], dtype=float)
            d = np.linalg.norm(p1 - p2)
            if 1.0 < d < 1.5:  # Only add edge midpoints
                mid = a * (p1 + p2) / 2
                pts.append(mid)
    # Third shell: 20 face centers
    faces = [
        [0, 4, 8], [0, 8, 10], [0, 10, 1], [0, 1, 4],
        [2, 5, 9], [2, 9, 11], [2, 11, 3], [2, 3, 5],
        [4, 6, 8], [6, 7, 8], [8, 10, 9], [10, 11, 9],
        [1, 3, 10], [3, 5, 11], [5, 7, 9], [7, 6, 9],
        [4, 6, 2], [6, 7, 11], [1, 4, 8], [3, 5, 9]
    ]
    for face in faces:
        if all(f < 12 for f in face):
            p1 = np.array(shell1[face[0]], dtype=float)
            p2 = np.array(shell1[face[1]], dtype=float)
            p3 = np.array(shell1[face[2]], dtype=float)
            center = a * (p1 + p2 + p3) / 3
            pts.append(center)
    pts = np.array(pts)
    if len(pts) < n:
        # Add more atoms in concentric shells
        rng = np.random.default_rng(42)
        shell_r = a * 2.5
        while len(pts) < n:
            theta = rng.uniform(0, 2 * np.pi)
            phi_angle = rng.uniform(-np.pi / 2, np.pi / 2)
            p = shell_r * np.array([np.cos(theta) * np.cos(phi_angle),
                                    np.sin(theta) * np.cos(phi_angle),
                                    np.sin(phi_angle)])
            pts = np.vstack([pts, p])
            shell_r += 0.1
    center = pts.mean(axis=0)
    dists = np.linalg.norm(pts - center, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx]


def _bcc_cluster(n, a=0.72):
    """Generate a BCC-like cluster with n atoms."""
    pts = []
    max_idx = 10
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                # BCC lattice points
                for di, dj, dk in [(0, 0, 0), (0.5, 0.5, 0.5)]:
                    p = a * np.array([i + di, j + dj, k + dk])
                    pts.append(p)
    pts = np.array(pts)
    if len(pts) < n:
        return pts
    center = pts.mean(axis=0)
    dists = np.linalg.norm(pts - center, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx]


def _random_compact(n, spacing=0.72, seed=42):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.7 * spacing for q in points):
                points.append(p)
        attempts += 1
    if len(points) < n:
        # Fill remaining with random points
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    return np.array(points)


def _basin_hopping_step(positions, kind, n_exp, m_exp, c, rng):
    """Perform one basin hopping step: perturb, then relax."""
    n = len(positions)
    # Perturbation magnitude
    scale = 0.3 + 0.2 * rng.random()
    perturbed = positions + scale * rng.standard_normal((n, 3))
    # Relax
    energy, relaxed = relax(perturbed, 'sutton_chen', kind)
    return energy, relaxed


def construct(instance):
    """Find global minimum of Sutton-Chen cluster using structure motifs + basin hopping."""
    n = instance["n"]
    kind = instance["kind"]
    n_exp = instance["n_exp"]
    m_exp = instance["m_exp"]
    c = instance["c"]
    
    # Generate initial structures from different motifs
    initial_structures = []
    
    # FCC-like
    try:
        fcc = _fcc_cluster(n)
        if len(fcc) >= n:
            fcc = fcc[:n]
            initial_structures.append(fcc)
    except:
        pass
    
    # HCP-like
    try:
        hcp = _hcp_cluster(n)
        if len(hcp) >= n:
            hcp = hcp[:n]
            initial_structures.append(hcp)
    except:
        pass
    
    # Icosahedral-like
    try:
        ico = _icosa_cluster(n)
        if len(ico) >= n:
            ico = ico[:n]
            initial_structures.append(ico)
    except:
        pass
    
    # BCC-like
    try:
        bcc = _bcc_cluster(n)
        if len(bcc) >= n:
            bcc = bcc[:n]
            initial_structures.append(bcc)
    except:
        pass
    
    # Random compact with multiple seeds
    for seed in [42, 123, 456, 789, 1011]:
        try:
            rand = _random_compact(n, seed=seed)
            if len(rand) == n:
                initial_structures.append(rand)
        except:
            pass
    
    if not initial_structures:
        # Fallback: simple random
        rng = np.random.default_rng(n)
        initial_structures.append(rng.uniform(-1, 1, (n, 3)))
    
    # Relax each initial structure
    best_energy = np.inf
    best_positions = None
    relaxed_structures = []
    
    for idx, init in enumerate(initial_structures):
        if _time_left() < 5.0:
            break
        
        try:
            energy, positions = relax(init, 'sutton_chen', kind)
            if np.isfinite(energy):
                relaxed_structures.append((energy, positions.copy()))
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()
        except:
            continue
    
    # Basin hopping from multiple starting points
    if best_positions is not None and _time_left() > 10.0:
        rng = np.random.default_rng(int(time.time() * 1000) % 10000)
        
        # Use top 3 relaxed structures as starting points for basin hopping
        relaxed_structures.sort(key=lambda x: x[0])
        start_points = [s[1] for s in relaxed_structures[:min(3, len(relaxed_structures))]]
        
        for start_pos in start_points:
            if _time_left() < 15.0:
                break
            
            current_energy = best_energy
            current_positions = start_pos.copy()
            
            # Adaptive basin hopping
            base_scale = 0.3
            for it in range(30):
                if _time_left() < 10.0:
                    break
                
                # Adaptive perturbation: larger when stuck, smaller when improving
                recent_improvement = 1 if it > 0 and current_energy < best_energy else 0
                scale = base_scale * (0.7 if recent_improvement else 1.3) * (0.8 + 0.4 * rng.random())
                
                perturbed = current_positions + scale * rng.standard_normal((n, 3))
                
                # Relax
                try:
                    energy, relaxed = relax(perturbed, 'sutton_chen', kind)
                except:
                    continue
                
                # Metropolis-like acceptance: accept if better or with some probability
                if energy < current_energy:
                    current_energy = energy
                    current_positions = relaxed.copy()
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = relaxed.copy()
                elif _time_left() > 20.0:
                    # Accept worse solutions with decreasing probability
                    delta_e = energy - current_energy
                    accept_prob = np.exp(-delta_e / (0.05 + 0.1 * it / 30))
                    if rng.random() < accept_prob:
                        current_energy = energy
                        current_positions = relaxed.copy()
    
    if best_positions is None:
        # Last resort: return first initial structure
        best_positions = initial_structures[0]
    
    # High-precision polish step
    if _time_left() > 5.0:
        try:
            energy, polished = relax(best_positions, 'sutton_chen', kind)
            if np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_positions = polished.copy()
        except:
            pass
    
    # Validate: check no atoms closer than 0.35
    min_dist = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(best_positions[i] - best_positions[j])
            if d < min_dist:
                min_dist = d
    
    if min_dist < 0.35:
        # Spread out atoms iteratively
        for _ in range(10):
            center = best_positions.mean(axis=0)
            best_positions = best_positions - center
            best_positions *= 1.1
            min_dist = np.inf
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.linalg.norm(best_positions[i] - best_positions[j])
                    if d < min_dist:
                        min_dist = d
            if min_dist >= 0.35:
                break
    
    return best_positions
# EVOLVE-BLOCK-END
