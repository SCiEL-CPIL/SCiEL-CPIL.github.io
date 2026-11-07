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


def _generate_icosahedron(n):
    """Generate a Mackay icosahedral cluster of n atoms."""
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    # Mackay icosahedron layers: 1, 12, 42, 92, 165, ...
    # We'll build by adding shells
    points = [np.zeros(3)]
    
    # Shell 1: 12 icosahedron vertices
    for v in ico_verts:
        points.append(v.copy())
    
    # Shell 2: 42 points (edges and face centers of icosahedron)
    # Edge midpoints
    icosahedron_edges = [
        (0, 4), (0, 5), (0, 8), (0, 9),
        (1, 2), (1, 3), (1, 6), (1, 7),
        (2, 3), (2, 10), (2, 11),
        (3, 4), (3, 5),
        (4, 8), (4, 9),
        (5, 8), (5, 9),
        (6, 7), (6, 10), (6, 11),
        (7, 10), (7, 11),
        (8, 10), (8, 11),
        (9, 10), (9, 11),
        (0, 4), (0, 5), (0, 8), (0, 9),
        (1, 6), (1, 7), (1, 2), (1, 3),
        (2, 10), (2, 11), (2, 3),
        (3, 10), (3, 11),
        (4, 8), (4, 9), (4, 3),
        (5, 8), (5, 9), (5, 3),
        (6, 10), (6, 11), (6, 7),
        (7, 10), (7, 11), (7, 6),
        (8, 10), (8, 11),
        (9, 10), (9, 11)
    ]
    
    # Actually, let's use a simpler approach: generate points on icosahedral shells
    # and sort by distance
    max_r = int(np.ceil(n ** (1 / 3))) + 2
    candidates = []
    
    # Add icosahedral lattice points
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p = np.array([i, j, k]) * (1.0 / np.sqrt(2.0))
                candidates.append((np.linalg.norm(p), p))
    
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)


def _growth_construct(n, kind, rng):
    """
    Deterministic growth algorithm: build cluster by adding atoms one at a time
    to the lowest-energy interstitial site of the (N-1)-atom cluster.
    """
    if n <= 1:
        return np.zeros((n, 3))
    
    # Start with 2 atoms at optimal distance
    # For Sutton-Chen, optimal distance is around 1.0-1.1
    best_energy = np.inf
    best_pos = None
    
    # Try initial distances
    for d0 in [0.9, 1.0, 1.1, 1.2]:
        pos = np.array([[0, 0, 0], [d0, 0, 0]])
        try:
            energy, relaxed = relax(pos, 'sutton_chen', kind)
            if np.isfinite(energy) and _is_valid(relaxed):
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed
        except Exception:
            continue
    
    if best_pos is None:
        best_pos = np.array([[0, 0, 0], [1.0, 0, 0]])
    
    # Grow from 2 to n atoms
    current = best_pos.copy()
    
    for target_n in range(3, n + 1):
        if _time_left() < 30.0:
            break
        
        # Generate candidate positions for new atom
        # Candidates: near existing atoms, in interstitial sites
        candidates = []
        
        # For each existing atom, try positions at various offsets
        for i in range(len(current)):
            # Random directions
            for trial in range(8):
                direction = rng.normal(0, 1, 3)
                direction /= np.linalg.norm(direction)
                for dist in [0.8, 0.9, 1.0, 1.1, 1.2]:
                    new_pos = current[i] + direction * dist
                    candidates.append(new_pos)
        
        # Also try positions in "gaps" - midpoints between close pairs
        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                midpoint = (current[i] + current[j]) / 2
                # Offset perpendicular to the bond
                bond_vec = current[j] - current[i]
                bond_len = np.linalg.norm(bond_vec)
                if bond_len > 0.1:
                    perp = np.cross(bond_vec, [0, 0, 1])
                    if np.linalg.norm(perp) < 1e-6:
                        perp = np.cross(bond_vec, [1, 0, 0])
                    perp /= np.linalg.norm(perp)
                    for offset in [0.3, 0.5, 0.7]:
                        candidates.append(midpoint + perp * offset)
                        candidates.append(midpoint - perp * offset)
        
        # Relax the full cluster with each candidate
        best_e = np.inf
        best_p = None
        
        # Shuffle candidates and try up to 15
        rng.shuffle(candidates)
        for candidate in candidates[:15]:
            if _time_left() < 25.0:
                break
            
            test_pos = np.vstack([current, candidate.reshape(1, 3)])
            try:
                energy, relaxed = relax(test_pos, 'sutton_chen', kind)
                if np.isfinite(energy) and _is_valid(relaxed):
                    if energy < best_e:
                        best_e = energy
                        best_p = relaxed
            except Exception:
                continue
        
        if best_p is not None:
            current = best_p
        else:
            # Fallback: add atom at a random position
            new_atom = current[0] + rng.normal(0, 0.5, 3)
            current = np.vstack([current, new_atom.reshape(1, 3)])
    
    return current


def construct(instance):
    n = instance["n"]
    kind = instance["kind"]
    
    best_energy = np.inf
    best_pos = None
    
    rng = np.random.default_rng(42 + n)
    
    # Strategy 1: Deterministic growth (primary)
    try:
        growth_pos = _growth_construct(n, kind, rng)
        if _is_valid(growth_pos):
            energy, relaxed = relax(growth_pos, 'sutton_chen', kind)
            if np.isfinite(energy) and _is_valid(relaxed):
                best_energy = energy
                best_pos = relaxed
    except Exception:
        pass
    
    # Strategy 2: FCC/BCC templates with relaxation
    for gen_func in [_generate_fcc, _generate_bcc, _generate_icosahedron]:
        if _time_left() < 20.0:
            break
        
        base = gen_func(n)
        if not _is_valid(base):
            continue
        
        # Try multiple noise levels
        for noise_scale in [0.0, 0.05, 0.1, 0.2]:
            if _time_left() < 15.0:
                break
            
            noise = rng.normal(0, noise_scale, base.shape) if noise_scale > 0 else np.zeros_like(base)
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
    
    # Strategy 3: Random restarts if time remains
    if _time_left() > 30.0:
        for i in range(50):
            if _time_left() < 15.0:
                break
            
            # Random compact sphere
            radius = 1.5 + 0.5 * (n / 50)
            pos = rng.uniform(-radius, radius, (n, 3))
            
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
    
    # Final fallback
    if best_pos is None:
        best_pos = _generate_fcc(n)
    
    return best_pos

# EVOLVE-BLOCK-END
