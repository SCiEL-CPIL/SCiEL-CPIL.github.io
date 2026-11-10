# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Dzugutov potential parameters
A_POT = 5.82
B_POT = 1.28
a_POT = 1.87
b_POT = 1.94
c_POT = 1.10
d_POT = 0.27
m_POT = 16


def dzugutov_energy(positions):
    """Compute Dzugutov potential energy."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(r, np.inf)
    iu = np.triu_indices(n, k=1)
    r_ij = r[iu]
    # V(r) = A(r^-m - B)exp(c/(r-a)) for r < a, B*exp(d/(r-b)) for r < b
    # For r >= max(a,b), V = 0 (no interaction)
    e = np.zeros_like(r_ij)
    mask_a = r_ij < a_POT
    mask_b = r_ij < b_POT
    # For r < a: full expression
    ra = r_ij[mask_a]
    e[mask_a] = A_POT * (ra**(-m_POT) - B_POT) * np.exp(c_POT / (ra - a_POT))
    # For a <= r < b: only the second term (B*exp(d/(r-b)))
    mask_b_only = mask_b & ~mask_a
    rb = r_ij[mask_b_only]
    e[mask_b_only] = B_POT * np.exp(d_POT / (rb - b_POT))
    return float(np.sum(e))


def icosahedron(center, radius=1.0):
    """Generate 12 vertices of a regular icosahedron at given center and radius."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms * radius
    return center + verts


def decahedron(center, radius=1.0):
    """Generate 20 vertices of a regular decahedron at given center and radius."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, phi, 1/phi], [0, phi, -1/phi], [0, -phi, 1/phi], [0, -phi, -1/phi],
        [1/phi, 0, phi], [1/phi, 0, -phi], [-1/phi, 0, phi], [-1/phi, 0, -phi],
        [phi, 1/phi, 0], [phi, -1/phi, 0], [-phi, 1/phi, 0], [-phi, -1/phi, 0]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms * radius
    return center + verts


def mackay_icosahedron(center, shell=2, radius=1.0):
    """Generate Mackay icosahedral cluster (13 + 42 + ... atoms)."""
    # 13-atom icosahedron (1 center + 12 vertices)
    if shell == 1:
        return np.vstack([center.reshape(1, 3), icosahedron(center, radius)])
    # 55-atom Mackay icosahedron: 13 + 42 + 140... but we need specific counts
    # For shell=2: 13 + 42 = 55 atoms
    # Build by adding second shell
    base = icosahedron(center, radius)
    # Second shell: atoms at positions that are sums of two edge vectors
    # Simpler: use known 55-atom Mackay icosahedron coordinates
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    # 55-atom Mackay icosahedron
    coords = np.array([
        [0, 0, 0],
        # 12 first-shell
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
        # 30 second-shell (face centers of icosahedron edges)
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
        [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
        [phi, 1, 1], [phi, 1, -1], [phi, -1, 1], [phi, -1, -1],
        [-phi, 1, 1], [-phi, 1, -1], [-phi, -1, 1], [-phi, -1, -1],
        [1, phi, 1], [1, phi, -1], [1, -phi, 1], [1, -phi, -1],
        [-1, phi, 1], [-1, phi, -1], [-1, -phi, 1], [-1, -phi, -1],
        # 12 more to reach 55
        [1, 1, phi], [1, -1, phi], [1, 1, -phi], [1, -1, -phi],
        [1, phi, 1], [-1, phi, 1], [1, phi, -1], [-1, phi, -1],
        [1, 1, 1], [-1, 1, 1], [1, 1, -1], [-1, 1, -1]
    ], dtype=float)
    norms = np.linalg.norm(coords, axis=1, keepdims=True)
    coords = coords / norms * radius
    return center + coords


def generate_seeds(n, rng):
    """Generate diverse polytetrahedral seed structures for n atoms."""
    seeds = []
    
    # Strategy 1: Icosahedral clusters
    # 13-atom icosahedron
    if n == 13:
        seeds.append(icosahedron(np.zeros(3), 1.0) * 0.55)
    # For larger n, build from icosahedral fragments
    
    # Strategy 2: Compact polytetrahedral packing
    # Place atoms in a dense arrangement favoring icosahedral coordination
    for trial in range(8):
        points = []
        # Start with a central atom
        points.append(np.zeros(3))
        
        # Add atoms in shells with icosahedral preference
        shell_radii = [1.0, 1.7, 2.4, 3.0, 3.6, 4.2]
        shell_counts = [12, 30, 55, 80, 110, 140]
        
        idx = 1
        for s, (rad, count) in enumerate(zip(shell_radii, shell_counts)):
            if idx >= n:
                break
            # Generate points on sphere at this radius
            for _ in range(count):
                if idx >= n:
                    break
                # Random point on sphere
                theta = rng.uniform(0, 2 * np.pi)
                z = rng.uniform(-1, 1)
                r = np.sqrt(1 - z**2)
                p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * rad
                # Check minimum distance
                if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                    points.append(p)
                    idx += 1
        
        # Fill remaining with random compact points
        while len(points) < n:
            p = rng.uniform(-4.0, 4.0, 3)
            if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                points.append(p)
        
        if len(points) == n:
            seeds.append(np.array(points))
    
    # Strategy 3: icosahedral fragment assembly
    # For n around 13, 55, 147
    if n >= 13 and n <= 60:
        # Use icosahedron as core, add atoms
        core = icosahedron(np.zeros(3), 1.0)
        core = np.vstack([np.zeros(3), core])  # 13 atoms
        if n >= 13:
            # Add remaining atoms in outer shell
            extra = []
            for _ in range(n - 13):
                theta = rng.uniform(0, 2 * np.pi)
                z = rng.uniform(-1, 1)
                r = np.sqrt(1 - z**2)
                p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * 1.8
                if all(np.linalg.norm(p - q) >= 0.55 for q in list(core) + extra):
                    extra.append(p)
            if len(extra) == n - 13:
                seeds.append(np.vstack([core, np.array(extra)]))
    
    # Strategy 4: Random compact with icosahedral bias
    for trial in range(4):
        points = []
        for _ in range(n):
            # Bias toward icosahedral positions
            if trial % 2 == 0:
                # Spherical shell
                theta = rng.uniform(0, 2 * np.pi)
                z = rng.uniform(-1, 1)
                r = np.sqrt(1 - z**2)
                p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * (2.5 + rng.uniform(0, 1))
            else:
                p = rng.uniform(-3.0, 3.0, 3)
            if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                points.append(p)
        if len(points) == n:
            seeds.append(np.array(points))
    
    return seeds


def basin_hopping_step(positions, rng, perturbation_scale=0.3):
    """Apply a small random perturbation and relax."""
    perturbed = positions + rng.uniform(-perturbation_scale, perturbation_scale, positions.shape)
    # Ensure minimum distance
    for i in range(len(perturbed)):
        for j in range(i + 1, len(perturbed)):
            d = np.linalg.norm(perturbed[i] - perturbed[j])
            if d < 0.5:
                # Push apart
                diff = perturbed[i] - perturbed[j]
                if np.linalg.norm(diff) < 1e-10:
                    diff = np.array([0.01, 0, 0])
                diff = diff / np.linalg.norm(diff)
                push = (0.5 - d) / 2 + 0.01
                perturbed[i] += diff * push
                perturbed[j] -= diff * push
    energy, relaxed = relax(perturbed, 'dzugutov', None)
    return energy, relaxed


def construct(instance):
    """Find low-energy Dzugutov cluster structure."""
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    n = instance["n"]
    potential = instance["potential"]
    rng = np.random.default_rng(n * 1000 + 42)
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse seeds
    seeds = generate_seeds(n, rng)
    
    # Relax each seed
    for seed in seeds:
        if time.time() > deadline:
            break
        try:
            energy, positions = relax(seed, potential, None)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            continue
    
    # Basin hopping on the best structure
    if best_positions is not None:
        current_energy = best_energy
        current_positions = best_positions
        no_improve_count = 0
        pert_scale = 0.3
        
        while time.time() < deadline:
            # Try a perturbation
            for _ in range(3):
                if time.time() > deadline:
                    break
                try:
                    e, pos = basin_hopping_step(current_positions, rng, pert_scale)
                    if e < current_energy:
                        current_energy = e
                        current_positions = pos
                        no_improve_count = 0
                        # Reduce perturbation scale on improvement
                        pert_scale = max(0.1, pert_scale * 0.9)
                    else:
                        no_improve_count += 1
                except Exception:
                    continue
            
            # Increase perturbation if stuck
            if no_improve_count > 5:
                pert_scale = min(0.5, pert_scale * 1.2)
                no_improve_count = 0
            
            # Update best
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = current_positions
    
    # Final safety check
    if best_positions is None:
        # Fallback: random compact
        points = []
        for _ in range(n):
            p = rng.uniform(-3.0, 3.0, 3)
            if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                points.append(p)
        best_positions = np.array(points)
        best_energy, best_positions = relax(best_positions, potential, None)
    
    return best_positions
# EVOLVE-BLOCK-END
