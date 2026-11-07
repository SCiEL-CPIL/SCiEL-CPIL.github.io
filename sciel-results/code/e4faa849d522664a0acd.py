"""Sutton-Chen transition-metal cluster global minima - deterministic growth construction."""
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _icosahedron_vertices():
    """Return 12 vertices of a regular icosahedron (unit edge length)."""
    phi = (1 + np.sqrt(5)) / 2
    # Icosahedron with edge length 2 (standard coordinates)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    # Normalize to unit edge length
    # Edge length of this icosahedron is 2, so divide by 2
    verts /= 2.0
    return verts


def _fcc_lattice_points(n, a=1.0):
    """Generate n closest FCC lattice points to origin with lattice constant a."""
    max_coord = int(np.ceil((3 * n / (16 * np.pi)) ** (1/3) * a)) + 3
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for (di, dj, dk) in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    x = (i + di) * a
                    y = (j + dj) * a
                    z = (k + dk) * a
                    points.append([x, y, z])
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    return points[order[:n]]


def _mackay_shells(n):
    """Generate Mackay icosahedron positions up to n atoms.
    
    Shell 0: 1 atom at center
    Shell 1: 12 atoms (icosahedron vertices)  
    Shell 2: 42 atoms (edge midpoints + face centers)
    Shell 3: 92 atoms
    """
    ico_verts = _icosahedron_vertices()
    
    positions = [np.zeros(3)]  # Shell 0
    
    # Shell 1: 12 icosahedron vertices
    for v in ico_verts:
        positions.append(v.copy())
    
    if n <= 13:
        return np.array(positions[:n])
    
    # Shell 2: 42 atoms
    # Edge midpoints (30 edges) and face centers (20 faces)
    # Find edges: pairs of vertices at minimum distance
    verts = ico_verts
    n_verts = len(verts)
    min_dist = np.inf
    for i in range(n_verts):
        for j in range(i + 1, n_verts):
            d = np.linalg.norm(verts[i] - verts[j])
            min_dist = min(min_dist, d)
    
    # Edge midpoints
    edge_mids = []
    for i in range(n_verts):
        for j in range(i + 1, n_verts):
            d = np.linalg.norm(verts[i] - verts[j])
            if abs(d - min_dist) < 1e-6:
                mid = (verts[i] + verts[j]) / 2
                edge_mids.append(mid)
    
    # Face centers (20 triangular faces)
    face_centers = []
    for i in range(n_verts):
        for j in range(i + 1, n_verts):
            for k in range(j + 1, n_verts):
                dij = np.linalg.norm(verts[i] - verts[j])
                djk = np.linalg.norm(verts[j] - verts[k])
                dki = np.linalg.norm(verts[k] - verts[i])
                if (abs(dij - min_dist) < 1e-6 and 
                    abs(djk - min_dist) < 1e-6 and 
                    abs(dki - min_dist) < 1e-6):
                    center = (verts[i] + verts[j] + verts[k]) / 3
                    face_centers.append(center)
    
    shell2_points = np.array(edge_mids + face_centers)
    # Scale to appropriate radius (should be ~sqrt(3)/2 * edge_length from center for face centers)
    # For unit edge icosahedron, shell 2 radius is approximately 0.707
    for pt in shell2_points:
        positions.append(pt.copy())
    
    if n <= 55:
        return np.array(positions[:n])
    
    # Shell 3: 92 atoms - approximate with points on a slightly larger icosahedral grid
    # For simplicity, generate additional points on a sphere
    n_needed = n - len(positions)
    if n_needed > 0:
        shell3_radius = 1.15  # approximate
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(n_needed):
            z = 1 - 2 * (i + 0.5) / n_needed
            r = np.sqrt(max(0, 1 - z**2))
            theta = golden_angle * i
            pt = np.array([r * np.cos(theta), r * np.sin(theta), z]) * shell3_radius
            positions.append(pt)
    
    return np.array(positions[:n])


def _relaxed_growth(instance, max_time=45.0):
    """Deterministic growth: build cluster atom-by-atom at lowest-energy interstitial sites."""
    import time
    start_time = time.time()
    n = instance["n"]
    kind = instance["kind"]
    
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    if n == 2:
        pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        try:
            _, pos = relax(pos, 'sutton_chen', kind)
        except:
            pass
        return pos
    
    # Start with 2-atom cluster
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    try:
        _, positions = relax(positions, 'sutton_chen', kind)
    except:
        pass
    
    for current_n in range(2, n):
        if time.time() - start_time > max_time:
            break
        
        # Generate candidate interstitial positions
        candidates = []
        
        # 1. FCC-type interstitials: near existing atoms at typical bond distances
        for p in positions:
            for d in np.array([
                [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5],
                [-0.5, 0, 0], [0, -0.5, 0], [0, 0, -0.5],
                [0.35, 0.35, 0], [0.35, -0.35, 0], [-0.35, 0.35, 0], [-0.35, -0.35, 0],
                [0.35, 0, 0.35], [0.35, 0, -0.35], [-0.35, 0, 0.35], [-0.35, 0, -0.35],
                [0, 0.35, 0.35], [0, 0.35, -0.35], [0, -0.35, 0.35], [0, -0.35, -0.35],
            ]):
                candidates.append(p + d)
        
        # 2. Spherical shell candidates at various radii
        centroid = np.mean(positions, axis=0)
        max_r = np.max(np.linalg.norm(positions - centroid, axis=1))
        for r in np.linspace(0.5, max_r + 0.5, 8):
            n_pts = 24
            golden_angle = np.pi * (3 - np.sqrt(5))
            for i in range(n_pts):
                z = 1 - 2 * (i + 0.5) / n_pts
                rr = np.sqrt(max(0, 1 - z**2))
                theta = golden_angle * i
                candidates.append(centroid + np.array([rr * np.cos(theta), rr * np.sin(theta), z]) * r)
        
        # 3. Icosahedral-type positions
        ico_verts = _icosahedron_vertices()
        for v in ico_verts:
            for scale in [0.8, 1.0, 1.2]:
                candidates.append(centroid + v * scale)
        
        candidates = np.array(candidates)
        
        # Filter: minimum distance 0.35 from all existing atoms
        valid_indices = []
        for i, cand in enumerate(candidates):
            dists = np.linalg.norm(positions - cand, axis=1)
            if np.min(dists) >= 0.35:
                valid_indices.append(i)
        
        if not valid_indices:
            # Fallback: add far away
            positions = np.vstack([positions, np.array([max_r + 1.0, 0, 0])])
            continue
        
        # Evaluate energy for each valid candidate (limit for speed)
        max_evals = min(len(valid_indices), 40)
        if len(valid_indices) > max_evals:
            # Sample
            rng = np.random.default_rng(current_n)
            valid_indices = list(rng.choice(valid_indices, max_evals, replace=False))
        
        best_energy = np.inf
        best_cand = None
        
        for idx in valid_indices:
            cand = candidates[idx]
            test_pos = np.vstack([positions, cand.reshape(1, 3)])
            try:
                e, _ = sutton_chen_energy_gradient(test_pos, kind)
                if np.isfinite(e) and e < best_energy:
                    best_energy = e
                    best_cand = cand
            except:
                continue
        
        if best_cand is None:
            # Fallback
            best_cand = candidates[valid_indices[0]]
        
        positions = np.vstack([positions, best_cand.reshape(1, 3)])
        
        # Relax periodically
        if (current_n + 1) % 4 == 0 or current_n + 1 == n:
            try:
                _, positions = relax(positions, 'sutton_chen', kind)
            except:
                pass
    
    return positions


def _template_relax(instance, max_time=30.0):
    """Relax from multiple structural templates (FCC, Mackay, mixed)."""
    import time
    start_time = time.time()
    n = instance["n"]
    kind = instance["kind"]
    
    best_energy = np.inf
    best_positions = None
    
    # Template 1: FCC
    try:
        fcc = _fcc_lattice_points(n, a=1.0)
        e, pos = relax(fcc, 'sutton_chen', kind)
        if np.isfinite(e) and e < best_energy:
            best_energy = e
            best_positions = pos
    except:
        pass
    
    # Template 2: Mackay icosahedron
    if time.time() - start_time < max_time:
        try:
            mackay = _mackay_shells(n)
            e, pos = relax(mackay, 'sutton_chen', kind)
            if np.isfinite(e) and e < best_energy:
                best_energy = e
                best_positions = pos
        except:
            pass
    
    # Template 3: Growth-based
    if time.time() - start_time < max_time:
        try:
            growth = _relaxed_growth(instance, max_time=max_time - (time.time() - start_time))
            e, pos = relax(growth, 'sutton_chen', kind)
            if np.isfinite(e) and e < best_energy:
                best_energy = e
                best_positions = pos
        except:
            pass
    
    # Template 4: Slightly perturbed FCC
    if time.time() - start_time < max_time:
        try:
            fcc = _fcc_lattice_points(n, a=1.0)
            rng = np.random.default_rng(123)
            perturbed = fcc + rng.normal(0, 0.05, fcc.shape)
            e, pos = relax(perturbed, 'sutton_chen', kind)
            if np.isfinite(e) and e < best_energy:
                best_energy = e
                best_positions = pos
        except:
            pass
    
    # Template 5: Slightly perturbed Mackay
    if time.time() - start_time < max_time:
        try:
            mackay = _mackay_shells(n)
            rng = np.random.default_rng(456)
            perturbed = mackay + rng.normal(0, 0.05, mackay.shape)
            e, pos = relax(perturbed, 'sutton_chen', kind)
            if np.isfinite(e) and e < best_energy:
                best_energy = e
                best_positions = pos
        except:
            pass
    
    if best_positions is None:
        best_positions = _fcc_lattice_points(n, a=1.0)
    
    return best_positions


def construct(instance):
    """Construct a low-energy Sutton-Chen cluster configuration using deterministic growth."""
    n = instance["n"]
    kind = instance["kind"]
    
    import time
    start_time = time.time()
    
    # Primary: template-based relaxation with multiple structural guides
    positions = _template_relax(instance, max_time=50.0)
    
    # Verify validity
    if not np.all(np.isfinite(positions)):
        positions = _fcc_lattice_points(n, a=1.0)
    
    # Check minimum distance constraint
    min_dist = np.inf
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < min_dist:
                min_dist = d
    
    if min_dist < 0.35:
        # Try growth-based as fallback
        try:
            positions = _relaxed_growth(instance, max_time=40.0)
        except:
            positions = _fcc_lattice_points(n, a=1.0)
        
        # Re-check
        if not np.all(np.isfinite(positions)):
            positions = _fcc_lattice_points(n, a=1.0)
        
        min_dist = np.inf
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < min_dist:
                    min_dist = d
        
        if min_dist < 0.35:
            # Last resort: spread out FCC
            positions = _fcc_lattice_points(n, a=1.2)
    
    return positions
