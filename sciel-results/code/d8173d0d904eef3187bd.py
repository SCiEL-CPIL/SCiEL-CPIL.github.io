"""Program for Sutton-Chen TM cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = 290.0  # Safety margin before 300s kill


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _is_valid(positions, min_dist=0.35):
    """Check if positions are valid (finite, no atoms closer than min_dist)."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    for i in range(n - 1):
        d = np.linalg.norm(positions[i + 1:] - positions[i], axis=1)
        if np.any(d < min_dist):
            return False
    return True


def _generate_fcc(n):
    """Generate a cluster of n atoms in an FCC-like structure."""
    max_r = int(np.ceil(n ** (1 / 3))) + 3
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k]) * (1.0 / np.sqrt(2.0))
                    candidates.append((np.linalg.norm(p), p))
    
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)


def _generate_bcc(n):
    """Generate a cluster of n atoms in a BCC-like structure."""
    max_r = int(np.ceil(n ** (1 / 3))) + 3
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k])
                p2 = np.array([i + 0.5, j + 0.5, k + 0.5])
                candidates.append((np.linalg.norm(p1), p1))
                candidates.append((np.linalg.norm(p2), p2))
    
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)


def _generate_hcp(n):
    """Generate a cluster of n atoms in an HCP-like structure."""
    a = 1.0
    c = np.sqrt(8.0 / 3.0) * a
    pts = []
    max_r = int(np.ceil(n ** (1 / 3))) + 3
    candidates = []
    
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                x = a * (i + 0.5 * j)
                y = a * np.sqrt(3) / 2 * j
                z = c * k
                p1 = np.array([x, y, z])
                candidates.append((np.linalg.norm(p1), p1))
                
                x2 = a * (i + 0.5 * j + 1.0 / np.sqrt(3.0))
                y2 = a * np.sqrt(3) / 2 * (j + 1.0 / 3.0)
                z2 = c * (k + 1.0 / 3.0)
                p2 = np.array([x2, y2, z2])
                candidates.append((np.linalg.norm(p2), p2))
    
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)


def _generate_icosahedron(n):
    """Generate a cluster of n atoms with icosahedral symmetry."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])
    
    pts = [np.zeros(3)]
    pts.extend(verts.tolist())
    
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < 0.5:
                mid = (verts[i] + verts[j]) / 2
                mid = mid / np.linalg.norm(mid) * 1.05
                pts.append(mid)
    
    while len(pts) < n:
        for i in range(12):
            for j in range(i + 1, 12):
                for k in range(j + 1, 12):
                    if len(pts) >= n:
                        break
                    d1 = np.linalg.norm(verts[i] - verts[j])
                    d2 = np.linalg.norm(verts[j] - verts[k])
                    d3 = np.linalg.norm(verts[k] - verts[i])
                    if d1 < 0.5 and d2 < 0.5 and d3 < 0.5:
                        centroid = (verts[i] + verts[j] + verts[k]) / 3
                        centroid = centroid / np.linalg.norm(centroid) * 1.15
                        pts.append(centroid)
                if len(pts) >= n:
                    break
            if len(pts) >= n:
                break
        if len(pts) < n:
            for v in verts:
                if len(pts) >= n:
                    break
                outer = v * 1.3
                pts.append(outer)
    
    return np.array(pts[:n])


def _generate_decahedral(n):
    """Generate a cluster with decahedral (D5h) symmetry."""
    # 5-fold axis along z, with 5-fold rotational symmetry
    pts = [np.zeros(3)]  # center
    
    # First shell: 5 atoms in a ring + 1 on top + 1 on bottom
    r1 = 1.0
    for i in range(5):
        angle = 2 * np.pi * i / 5
        pts.append(np.array([r1 * np.cos(angle), r1 * np.sin(angle), 0]))
    pts.append(np.array([0, 0, r1 * 0.5]))
    pts.append(np.array([0, 0, -r1 * 0.5]))
    
    # Second shell: 10 atoms
    r2 = 1.8
    for i in range(5):
        angle = 2 * np.pi * i / 5
        pts.append(np.array([r2 * np.cos(angle), r2 * np.sin(angle), 0.3]))
        pts.append(np.array([r2 * np.cos(angle + np.pi/5), r2 * np.sin(angle + np.pi/5), -0.3]))
    
    # Third shell: 15 atoms
    r3 = 2.5
    for i in range(5):
        angle = 2 * np.pi * i / 5
        pts.append(np.array([r3 * np.cos(angle), r3 * np.sin(angle), 0]))
        pts.append(np.array([r3 * np.cos(angle + np.pi/5), r3 * np.sin(angle + np.pi/5), 0.5]))
        pts.append(np.array([r3 * np.cos(angle + np.pi/5), r3 * np.sin(angle + np.pi/5), -0.5]))
    
    # Fill remaining with random compact points
    while len(pts) < n:
        angle = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(2.5, 3.5)
        z = np.random.uniform(-1.0, 1.0)
        pts.append(np.array([r * np.cos(angle), r * np.sin(angle), z]))
    
    return np.array(pts[:n])


def _generate_random_compact(n, rng, radius=2.0):
    """Generate n random points within a sphere, compactly distributed."""
    pts = []
    while len(pts) < n:
        p = rng.uniform(-1, 1, size=(10, 3))
        norms = np.linalg.norm(p, axis=1)
        valid = p[norms <= 1.0]
        for v in valid:
            if len(pts) < n:
                pts.append(v * radius)
    return np.array(pts[:n])


def _generate_deformed(n, base, rng, scale=0.3):
    """Generate a deformed version of a base structure."""
    pos = base.copy()
    for i in range(len(pos)):
        pos[i] += rng.normal(0, scale, 3)
    return pos


def _symmetry_constrained_optimize(n, kind, rng, n_shells=3):
    """
    Symmetry-constrained optimization using icosahedral shell parameters.
    Parameterize positions by shell radii and angular offsets, reducing dimensionality.
    """
    phi = (1 + np.sqrt(5)) / 2
    base_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    base_verts /= np.linalg.norm(base_verts[0])
    
    best_energy = np.inf
    best_pos = None
    
    # Try different numbers of shells
    for n_shell in range(1, n_shells + 1):
        # Generate shell radii
        shell_radii = np.linspace(0.8, 2.5, n_shell)
        
        # For each shell configuration, try different angular offsets
        for trial in range(5):
            if _time_left() < 30.0:
                return best_energy, best_pos
            
            # Random angular rotation for this trial
            theta = rng.uniform(0, 2 * np.pi)
            rot = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
            
            pts = [np.zeros(3)]  # center atom
            
            # Add shells
            atoms_per_shell = [12, 30, 60, 120]  # Icosahedral shell sizes
            for shell_idx in range(n_shell):
                r = shell_radii[shell_idx]
                if shell_idx == 0:
                    # First shell: icosahedron vertices
                    shell_pts = base_verts * r
                elif shell_idx == 1:
                    # Second shell: edge midpoints (30 atoms)
                    shell_pts = []
                    for i in range(12):
                        for j in range(i + 1, 12):
                            d = np.linalg.norm(base_verts[i] - base_verts[j])
                            if d < 0.5:
                                mid = (base_verts[i] + base_verts[j]) / 2
                                mid = mid / np.linalg.norm(mid) * r
                                shell_pts.append(mid)
                    shell_pts = np.array(shell_pts)
                else:
                    # Higher shells: face centroids and beyond
                    shell_pts = []
                    for i in range(12):
                        for j in range(i + 1, 12):
                            for k in range(j + 1, 12):
                                d1 = np.linalg.norm(base_verts[i] - base_verts[j])
                                d2 = np.linalg.norm(base_verts[j] - base_verts[k])
                                d3 = np.linalg.norm(base_verts[k] - base_verts[i])
                                if d1 < 0.5 and d2 < 0.5 and d3 < 0.5:
                                    centroid = (base_verts[i] + base_verts[j] + base_verts[k]) / 3
                                    centroid = centroid / np.linalg.norm(centroid) * r
                                    shell_pts.append(centroid)
                    if len(shell_pts) == 0:
                        # Fallback: scale up base vertices
                        shell_pts = (base_verts * r * (1.0 + 0.3 * shell_idx)).tolist()
                    else:
                        shell_pts = np.array(shell_pts)
                
                # Apply rotation
                shell_pts = shell_pts @ rot.T
                pts.extend(shell_pts.tolist())
                
                if len(pts) >= n:
                    break
            
            if len(pts) < n:
                # Fill remaining with random points
                while len(pts) < n:
                    angle = rng.uniform(0, 2 * np.pi)
                    elev = rng.uniform(-np.pi/2, np.pi/2)
                    r = rng.uniform(2.0, 3.0)
                    pts.append(np.array([
                        r * np.cos(elev) * np.cos(angle),
                        r * np.cos(elev) * np.sin(angle),
                        r * np.sin(elev)
                    ]))
            
            pos = np.array(pts[:n])
            
            if not _is_valid(pos):
                continue
            
            # Relax
            try:
                energy, relaxed_pos = relax(pos, 'sutton_chen', kind)
                if np.isfinite(energy) and _is_valid(relaxed_pos):
                    if energy < best_energy:
                        best_energy = energy
                        best_pos = relaxed_pos
            except Exception:
                continue
    
    return best_energy, best_pos


def _polish(positions, kind):
    """High-precision polish pass with tighter tolerance."""
    try:
        energy, pos = relax(positions, 'sutton_chen', kind)
        if not np.isfinite(energy):
            return None, np.inf
        
        for _ in range(5):
            e, g = sutton_chen_energy_gradient(pos, kind)
            if not np.isfinite(e):
                break
            lr = 0.001
            pos = pos - lr * g
            if not _is_valid(pos):
                pos = pos + 0.01 * np.random.default_rng(12345).normal(0, 1, pos.shape)
        
        energy, pos = relax(pos, 'sutton_chen', kind)
        if np.isfinite(energy) and _is_valid(pos):
            return pos, energy
    except Exception:
        pass
    return None, np.inf


rng_global = np.random.default_rng(12345)


def construct(instance):
    n = instance["n"]
    kind = instance["kind"]
    
    best_energy = np.inf
    best_pos = None
    
    rng = np.random.default_rng(42 + n)
    
    # Base structures to try
    base_structures = []
    base_structures.append(_generate_fcc(n))
    base_structures.append(_generate_bcc(n))
    base_structures.append(_generate_hcp(n))
    base_structures.append(_generate_icosahedron(n))
    base_structures.append(_generate_decahedral(n))
    base_structures.append(_generate_random_compact(n, rng, radius=2.0))
    
    # Relax each base structure first to get better starting points
    relaxed_bases = []
    for base in base_structures:
        if _time_left() < 30.0:
            break
        try:
            energy, relaxed_pos = relax(base, 'sutton_chen', kind)
            if np.isfinite(energy) and _is_valid(relaxed_pos):
                relaxed_bases.append(relaxed_pos)
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed_pos
        except Exception:
            continue
    
    # Symmetry-constrained optimization
    if _time_left() > 60.0:
        sym_energy, sym_pos = _symmetry_constrained_optimize(n, kind, rng, n_shells=3)
        if sym_pos is not None and sym_energy < best_energy:
            best_energy = sym_energy
            best_pos = sym_pos
    
    # Now do multi-start from relaxed bases with various perturbations
    max_restarts = 150
    
    for i in range(max_restarts):
        if _time_left() < 20.0:
            break
        
        # Choose a base structure
        if relaxed_bases:
            base = relaxed_bases[i % len(relaxed_bases)].copy()
        else:
            base = base_structures[i % len(base_structures)].copy()
        
        # Vary noise level
        noise_scale = 0.05 + 0.15 * ((i // 5) % 4)
        noise = rng.normal(0, noise_scale, base.shape)
        pos = base + noise
        
        try:
            energy, relaxed_pos = relax(pos, 'sutton_chen', kind)
            
            if not np.isfinite(energy):
                continue
                
            if _is_valid(relaxed_pos):
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed_pos
        except Exception:
            continue
    
    # Polish pass on the best result
    if best_pos is not None and _time_left() > 30.0:
        polished_pos, polished_energy = _polish(best_pos, kind)
        if polished_pos is not None and polished_energy < best_energy:
            best_energy = polished_energy
            best_pos = polished_pos
    
    # Fallback: try more random starts
    if best_pos is None:
        for i in range(100):
            if _time_left() < 15.0:
                break
            pos = _generate_random_compact(n, rng, radius=2.5)
            try:
                energy, relaxed_pos = relax(pos, 'sutton_chen', kind)
                if np.isfinite(energy) and _is_valid(relaxed_pos):
                    if energy < best_energy:
                        best_energy = energy
                        best_pos = relaxed_pos
            except Exception:
                continue
    
    # Final fallback
    if best_pos is None:
        best_pos = _generate_fcc(n)
    
    return best_pos

# EVOLVE-BLOCK-END
