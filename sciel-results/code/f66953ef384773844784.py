"""Sutton-Chen transition-metal cluster global minima - differential evolution with multi-template init."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Global deadline: 320s (10s safety margin before 330s kill)
_DEADLINE = time.time() + 320.0


def _fcc_positions(n):
    """Generate n positions on an FCC lattice, packed as compactly as possible."""
    a = np.sqrt(2.0)
    R = (n / 4.0) ** (1.0 / 3.0) * a * 1.4
    nmax = int(R / a) + 3
    points = []
    for i in range(-nmax, nmax + 1):
        for j in range(-nmax, nmax + 1):
            for k in range(-nmax, nmax + 1):
                for dx, dy, dz in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    x = (i + dx) * a
                    y = (j + dy) * a
                    z = (k + dz) * a
                    if x*x + y*y + z*z <= R*R:
                        points.append([x, y, z])
    points = np.array(points)
    if len(points) >= n:
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:n]
        return points[idx]
    else:
        while len(points) < n:
            p = np.random.uniform(-R, R, 3)
            points = np.vstack([points, p])
        return points[:n]


def _mackay_positions(n):
    """Generate positions using Mackay icosahedral shells."""
    phi = (1 + np.sqrt(5)) / 2.0
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    R1 = 1.0 / (2.0 * np.sin(np.pi / 5.0))
    
    shells = [
        (1, np.array([[0.0, 0.0, 0.0]])),
        (12, ico_verts * R1),
    ]
    
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    
    R2 = R1 * np.sqrt(3.0)
    shell2_points = []
    for i in range(42):
        z = 1.0 - 2.0 * i / 41.0
        r = np.sqrt(max(0.0, 1.0 - z*z))
        theta = golden_angle * i
        shell2_points.append([r*np.cos(theta), r*np.sin(theta), z])
    shell2_points = np.array(shell2_points) * R2
    shells.append((42, shell2_points))
    
    R3 = R1 * np.sqrt(6.0)
    shell3_points = []
    for i in range(92):
        z = 1.0 - 2.0 * i / 91.0
        r = np.sqrt(max(0.0, 1.0 - z*z))
        theta = golden_angle * i
        shell3_points.append([r*np.cos(theta), r*np.sin(theta), z])
    shell3_points = np.array(shell3_points) * R3
    shells.append((92, shell3_points))
    
    R4 = R1 * 3.0
    shell4_points = []
    for i in range(162):
        z = 1.0 - 2.0 * i / 161.0
        r = np.sqrt(max(0.0, 1.0 - z*z))
        theta = golden_angle * i
        shell4_points.append([r*np.cos(theta), r*np.sin(theta), z])
    shell4_points = np.array(shell4_points) * R4
    shells.append((162, shell4_points))
    
    positions = []
    for count, pts in shells:
        if len(positions) + count <= n:
            positions.append(pts)
        else:
            remaining = n - len(positions)
            positions.append(pts[:remaining])
            break
    
    if not positions:
        positions = [shells[0][1]]
    
    return np.vstack(positions)[:n]


def _hcp_positions(n):
    """Generate n positions in HCP arrangement."""
    a = 1.0
    c = a * 1.633
    layer_spacing = c / 2.0
    R_est = (n / 2.0) ** (1.0 / 3.0) * a
    n_layers = max(1, int(2 * R_est / layer_spacing) + 1)
    
    layer_counts = []
    for i in range(n_layers):
        d = abs(i - n_layers // 2)
        max_ring = max(0, int((R_est - d * layer_spacing) / a) + 1)
        count = 1 + 3 * max_ring * (max_ring + 1) if max_ring > 0 else 1
        layer_counts.append(count)
    
    total = sum(layer_counts)
    if total > n:
        for i in range(len(layer_counts) - 1, -1, -1):
            if total > n and layer_counts[i] > 1:
                layer_counts[i] -= 1
                total -= 1
    elif total < n:
        for i in range(n_layers):
            if total < n:
                layer_counts[i] += 1
                total += 1
    
    positions = []
    for layer_idx, count in enumerate(layer_counts):
        z = (layer_idx - n_layers // 2) * layer_spacing
        offset = (a / 2.0, a / np.sqrt(3.0)) if layer_idx % 2 == 1 else (0.0, 0.0)
        
        hex_points = [[0.0, 0.0]]
        ring = 1
        while len(hex_points) < count:
            for i in range(6 * ring):
                angle = i * np.pi / (3.0 * ring)
                x = ring * a * np.cos(angle) + offset[0]
                y = ring * a * np.sin(angle) + offset[1]
                hex_points.append([x, y])
                if len(hex_points) >= count:
                    break
            ring += 1
        
        for x, y in hex_points[:count]:
            positions.append([x, y, z])
    
    return np.array(positions[:n])


def _random_compact(n, rng, seed_offset=0):
    """Generate a random compact cluster."""
    spacing = 0.72
    radius = spacing * n ** (1.0 / 3.0) * 0.9
    points = []
    max_attempts = 15000
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.6 * spacing for q in points):
                points.append(p)
        attempts += 1
    
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points[:n])


def _sphere_positions(n, rng):
    """Generate positions on a sphere surface."""
    radius = (n / 4.0) ** (1.0 / 3.0) * 1.1
    points = []
    for _ in range(n):
        # Random direction
        z = rng.uniform(-1, 1)
        theta = rng.uniform(0, 2 * np.pi)
        r = np.sqrt(max(0, 1 - z*z))
        p = radius * np.array([r*np.cos(theta), r*np.sin(theta), z])
        points.append(p)
    return np.array(points)


def _energy_from_positions(positions, kind, n_exp, m_exp, c):
    """Compute Sutton-Chen energy from positions."""
    n = len(positions)
    if n < 2:
        return 0.0
    
    # Compute all pair distances
    diffs = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Compute sums
    repulsive = np.sum(dists ** (-n_exp), axis=1)
    attractive = np.sum(dists ** (-m_exp), axis=1)
    
    energy = 0.0
    for i in range(n):
        energy += 0.5 * repulsive[i] - c * np.sqrt(attractive[i])
    
    return energy


def _de_energy(x, kind, n_exp, m_exp, c, n):
    """Energy function for differential evolution."""
    positions = x.reshape(n, 3)
    
    # Check for invalid configurations
    if not np.all(np.isfinite(positions)):
        return 1e10
    
    # Check minimum distance
    diffs = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 0.35:
        return 1e10
    
    try:
        e = _energy_from_positions(positions, kind, n_exp, m_exp, c)
        if not np.isfinite(e):
            return 1e10
        return e
    except Exception:
        return 1e10


def construct(instance):
    """Differential evolution with multi-template initialization for Sutton-Chen clusters."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    
    # Get parameters based on kind
    if kind == "12-6":
        n_exp, m_exp, c = 12, 6, 144.41
    elif kind == "9-6":
        n_exp, m_exp, c = 9, 6, 39.432
    elif kind == "10-8":
        n_exp, m_exp, c = 10, 8, 34.408
    else:
        n_exp, m_exp, c = 12, 6, 144.41
    
    rng = np.random.default_rng(size * 1000 + 42)
    
    # Generate diverse templates
    templates = []
    
    # FCC template
    try:
        fcc = _fcc_positions(size)
        if np.all(np.isfinite(fcc)) and len(fcc) == size:
            templates.append(fcc)
    except Exception:
        pass
    
    # Mackay icosahedral template
    try:
        mackay = _mackay_positions(size)
        if np.all(np.isfinite(mackay)) and len(mackay) == size:
            templates.append(mackay)
    except Exception:
        pass
    
    # HCP template
    try:
        hcp = _hcp_positions(size)
        if np.all(np.isfinite(hcp)) and len(hcp) == size:
            templates.append(hcp)
    except Exception:
        pass
    
    # Random compact templates
    for i in range(4):
        try:
            rand = _random_compact(size, rng, seed_offset=i)
            if np.all(np.isfinite(rand)) and len(rand) == size:
                templates.append(rand)
        except Exception:
            pass
    
    # Sphere surface template
    try:
        sphere = _sphere_positions(size, rng)
        if np.all(np.isfinite(sphere)) and len(sphere) == size:
            templates.append(sphere)
    except Exception:
        pass
    
    # Relax all templates to get good starting points
    relaxed_templates = []
    for i, template in enumerate(templates):
        if time.time() > _DEADLINE - 40:
            break
        try:
            energy, positions = relax(template, potential, kind)
            if np.all(np.isfinite(positions)) and np.isfinite(energy):
                relaxed_templates.append(positions)
        except Exception:
            pass
    
    # If we have relaxed templates, use them for DE initialization
    # Otherwise, use the raw templates
    
    # Determine bounds for DE
    # Cluster radius estimate
    R_est = (size / 4.0) ** (1.0 / 3.0) * 1.5
    bounds = [(-R_est, R_est)] * (size * 3)
    
    # Run differential evolution
    best_energy = np.inf
    best_positions = None
    
    # First, try to find a good solution from relaxed templates
    if relaxed_templates:
        for positions in relaxed_templates:
            try:
                e = _energy_from_positions(positions, kind, n_exp, m_exp, c)
                if np.isfinite(e) and e < best_energy:
                    best_energy = e
                    best_positions = positions.copy()
            except Exception:
                pass
    
    # Now run DE
    if time.time() < _DEADLINE - 20:
        # Use population size based on available time
        remaining_time = _DEADLINE - 20 - time.time()
        
        # Estimate time per DE iteration
        # DE with popsize P does P evaluations per generation
        # Each evaluation is an energy computation (~1ms for n<100)
        # Relaxation after DE adds overhead
        
        # Start with moderate population
        popsize = min(30, max(15, int(remaining_time / 2.0)))
        
        try:
            de_result = differential_evolution(
                _de_energy,
                bounds,
                args=(kind, n_exp, m_exp, c, size),
                popsize=popsize,
                maxiter=50,
                tol=1e-6,
                seed=int(size * 1000 + 42),
                polish=True,
                init='sobol',
                update_interval=10,
                callback=None
            )
            
            if de_result.success and np.isfinite(de_result.fun):
                de_positions = de_result.x.reshape(size, 3)
                de_energy = de_result.fun
                
                # Check validity
                if np.all(np.isfinite(de_positions)):
                    diffs = de_positions[:, np.newaxis, :] - de_positions[np.newaxis, :, :]
                    dists = np.sqrt(np.sum(diffs**2, axis=2))
                    np.fill_diagonal(dists, np.inf)
                    if np.min(dists) >= 0.35:
                        if de_energy < best_energy:
                            best_energy = de_energy
                            best_positions = de_positions.copy()
        except Exception:
            pass
    
    # If we still don't have a good solution, use the best template
    if best_positions is None:
        if templates:
            # Use the first valid template
            for t in templates:
                if np.all(np.isfinite(t)) and len(t) == size:
                    best_positions = t.copy()
                    break
        if best_positions is None:
            best_positions = np.random.uniform(-2, 2, (size, 3))
    
    # Final relaxation to polish the result
    if time.time() < _DEADLINE - 5:
        try:
            final_energy, final_positions = relax(best_positions, potential, kind)
            if np.all(np.isfinite(final_positions)) and np.isfinite(final_energy):
                # Check validity
                diffs = final_positions[:, np.newaxis, :] - final_positions[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diffs**2, axis=2))
                np.fill_diagonal(dists, np.inf)
                if np.min(dists) >= 0.35:
                    best_positions = final_positions
        except Exception:
            pass
    
    # Final validity check
    if not np.all(np.isfinite(best_positions)):
        # Fallback to a simple valid configuration
        spacing = 0.8
        R = spacing * size ** (1.0 / 3.0) * 0.8
        best_positions = np.zeros((size, 3))
        for i in range(size):
            z = 1.0 - 2.0 * i / max(1, size - 1)
            r = np.sqrt(max(0, 1 - z*z))
            theta = 2 * np.pi * i / max(1, size)
            best_positions[i] = R * np.array([r*np.cos(theta), r*np.sin(theta), z])
    
    return best_positions
