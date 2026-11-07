"""Sutton-Chen transition-metal cluster global minima - multi-template with basin hopping."""
import time
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Global deadline: 320s (10s safety margin before 330s kill)
_DEADLINE = time.time() + 320.0


def _fcc_positions(n):
    """Generate n positions on an FCC lattice, packed as compactly as possible."""
    # FCC lattice vectors (conventional cell)
    # Generate points in FCC: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) in units of a
    # Scale so nearest neighbor distance ~ 1.0 (equilibrium for SC)
    # In FCC, nearest neighbor = a/sqrt(2), so set a = sqrt(2) for nn=1
    
    a = np.sqrt(2.0)  # lattice constant for nn distance = 1.0
    
    # Generate candidate positions in a bounding sphere
    # For n atoms, radius R ~ (n/4)^(1/3) * a (FCC packing: 4 atoms per a^3)
    R = (n / 4.0) ** (1.0 / 3.0) * a * 1.3
    
    # Generate all FCC points within radius
    nmax = int(R / a) + 2
    points = []
    for i in range(-nmax, nmax + 1):
        for j in range(-nmax, nmax + 1):
            for k in range(-nmax, nmax + 1):
                # Four FCC basis positions
                for dx, dy, dz in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    x = (i + dx) * a
                    y = (j + dy) * a
                    z = (k + dz) * a
                    if x*x + y*y + z*z <= R*R:
                        points.append([x, y, z])
    
    points = np.array(points)
    
    # Sort by distance from center, take n closest
    if len(points) >= n:
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:n]
        return points[idx]
    else:
        # Pad with random points if not enough
        while len(points) < n:
            p = np.random.uniform(-R, R, 3)
            points = np.vstack([points, p])
        return points[:n]


def _mackay_positions(n):
    """Generate positions using Mackay icosahedral shells."""
    # Icosahedral shell radii and atom counts
    # Shell 0: 1 atom at center
    # Shell 1: 12 atoms (vertices of icosahedron)
    # Shell 2: 42 atoms
    # Shell 3: 92 atoms
    # Shell 4: 162 atoms
    
    # Icosahedron vertices (unit radius)
    phi = (1 + np.sqrt(5)) / 2.0
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    # Scale for nearest neighbor ~ 1.0
    # In icosahedron, edge length = 2*sin(pi/5) * R_ico
    # For nn=1: R_ico = 1/(2*sin(pi/5))
    R1 = 1.0 / (2.0 * np.sin(np.pi / 5.0))
    
    # Shell radii (approximate, for Mackay icosahedra)
    # Shell 1: R1
    # Shell 2: R1 * sqrt(3) approx (actually more complex)
    # Use known ratios: shell 2 at ~1.732*R1, shell 3 at ~2.449*R1
    
    shells = [
        (1, np.array([[0.0, 0.0, 0.0]])),  # center
        (12, ico_verts * R1),  # shell 1
    ]
    
    # For shell 2, use a larger icosahedron-like arrangement
    # Approximate with 42 points on a sphere of radius R2
    R2 = R1 * np.sqrt(3.0)
    # Generate 42 points on sphere using golden spiral
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    shell2_points = []
    for i in range(42):
        z = 1.0 - 2.0 * i / 41.0
        r = np.sqrt(max(0.0, 1.0 - z*z))
        theta = golden_angle * i
        shell2_points.append([r*np.cos(theta), r*np.sin(theta), z])
    shell2_points = np.array(shell2_points) * R2
    shells.append((42, shell2_points))
    
    # Shell 3: 92 points
    R3 = R1 * np.sqrt(6.0)
    shell3_points = []
    for i in range(92):
        z = 1.0 - 2.0 * i / 91.0
        r = np.sqrt(max(0.0, 1.0 - z*z))
        theta = golden_angle * i
        shell3_points.append([r*np.cos(theta), r*np.sin(theta), z])
    shell3_points = np.array(shell3_points) * R3
    shells.append((92, shell3_points))
    
    # Shell 4: 162 points
    R4 = R1 * 3.0
    shell4_points = []
    for i in range(162):
        z = 1.0 - 2.0 * i / 161.0
        r = np.sqrt(max(0.0, 1.0 - z*z))
        theta = golden_angle * i
        shell4_points.append([r*np.cos(theta), r*np.sin(theta), z])
    shell4_points = np.array(shell4_points) * R4
    shells.append((162, shell4_points))
    
    # Build up to n atoms
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
    # HCP: ABAB stacking
    # Layer spacing = 0.95 * nn (for ideal HCP, c/a = 1.633)
    a = 1.0  # in-plane nearest neighbor
    c = a * 1.633  # c-axis
    layer_spacing = c / 2.0
    
    # In-plane: hexagonal lattice
    # Each layer has ~ n / n_layers atoms
    # Estimate number of layers
    # Area per atom in hex layer = (sqrt(3)/2) * a^2
    # Total area ~ n * (sqrt(3)/2) * a^2
    # If cluster is roughly spherical: R ~ (n/2)^(1/3) * a
    # Number of layers ~ 2*R / layer_spacing
    
    R_est = (n / 2.0) ** (1.0 / 3.0) * a
    n_layers = max(1, int(2 * R_est / layer_spacing) + 1)
    
    # Distribute atoms among layers
    # Central layers have more atoms
    layer_counts = []
    for i in range(n_layers):
        # Distance from center layer
        d = abs(i - n_layers // 2)
        # Number of hex rings in this layer
        max_ring = max(0, int((R_est - d * layer_spacing) / a) + 1)
        # Atoms in hex packing: 1 + 3*r*(r+1) for r rings
        count = 1 + 3 * max_ring * (max_ring + 1) if max_ring > 0 else 1
        layer_counts.append(count)
    
    # Adjust to get exactly n
    total = sum(layer_counts)
    if total > n:
        # Remove from outer layers
        for i in range(len(layer_counts) - 1, -1, -1):
            if total > n and layer_counts[i] > 1:
                layer_counts[i] -= 1
                total -= 1
    elif total < n:
        # Add to central layers
        for i in range(n_layers):
            if total < n:
                layer_counts[i] += 1
                total += 1
    
    positions = []
    for layer_idx, count in enumerate(layer_counts):
        z = (layer_idx - n_layers // 2) * layer_spacing
        # Hexagonal lattice in xy plane
        # Offset for B layers
        offset = (a / 2.0, a / np.sqrt(3.0)) if layer_idx % 2 == 1 else (0.0, 0.0)
        
        # Generate hex lattice points
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
    radius = spacing * n ** (1.0 / 3.0) * 0.85
    points = []
    max_attempts = 10000
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.65 * spacing for q in points):
                points.append(p)
        attempts += 1
    
    # If not enough points, add random ones
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points[:n])


def _basin_hopping_step(positions, potential, kind, rng, step_size=0.5):
    """Perform a single basin hopping step: perturb and relax."""
    # Add random perturbation
    perturbation = rng.normal(0, step_size, positions.shape)
    perturbed = positions + perturbation
    
    # Ensure no atoms are too close
    # If any pair is too close, increase perturbation
    for _ in range(10):
        # Check minimum distance
        n = len(perturbed)
        min_dist = np.inf
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(perturbed[i] - perturbed[j])
                if d < min_dist:
                    min_dist = d
        if min_dist > 0.4:
            break
        perturbation = rng.normal(0, step_size * 1.5, positions.shape)
        perturbed = positions + perturbation
    
    # Relax
    try:
        energy, new_positions = relax(perturbed, potential, kind)
        if np.all(np.isfinite(new_positions)):
            return energy, new_positions
    except Exception:
        pass
    
    return None, None


def construct(instance):
    """Multi-template construction with basin hopping for Sutton-Chen clusters."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    
    rng = np.random.default_rng(size * 1000 + 42)
    
    # Generate multiple template starting points
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
    for i in range(3):
        try:
            rand = _random_compact(size, rng, seed_offset=i)
            if np.all(np.isfinite(rand)) and len(rand) == size:
                templates.append(rand)
        except Exception:
            pass
    
    # Relax all templates
    best_energy = np.inf
    best_positions = None
    
    for i, template in enumerate(templates):
        if time.time() > _DEADLINE - 30:
            break
        try:
            energy, positions = relax(template, potential, kind)
            if np.all(np.isfinite(positions)) and energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            pass
    
    if best_positions is None:
        # Fallback: use first template
        best_positions = templates[0] if templates else np.random.uniform(-2, 2, (size, 3))
    
    # Basin hopping from the best local minimum
    if time.time() < _DEADLINE - 10:
        current_energy = best_energy
        current_positions = best_positions
        step_size = 0.4
        max_bh_steps = 50
        bh_steps_done = 0
        
        for bh_step in range(max_bh_steps):
            if time.time() > _DEADLINE - 10:
                break
            
            # Perturb and relax
            new_energy, new_positions = _basin_hopping_step(
                current_positions, potential, kind, rng, step_size
            )
            
            if new_energy is not None and np.all(np.isfinite(new_positions)):
                # Accept if lower energy (or with Metropolis for exploration)
                if new_energy < current_energy:
                    current_energy = new_energy
                    current_positions = new_positions
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_positions = current_positions.copy()
                    # Reduce step size on success
                    step_size = max(0.1, step_size * 0.95)
                else:
                    # Metropolis acceptance
                    delta_e = new_energy - current_energy
                    # Temperature decreases over time
                    T = 0.5 * (1.0 - bh_step / max_bh_steps)
                    if T > 0 and rng.random() < np.exp(-delta_e / T):
                        current_energy = new_energy
                        current_positions = new_positions
            
            bh_steps_done += 1
    
    return best_positions
