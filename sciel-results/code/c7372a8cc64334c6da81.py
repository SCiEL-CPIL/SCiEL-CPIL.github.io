"""Dzugutov cluster global minima search using Mackay icosahedron-based initialization."""
# EVOLVE-BLOCK-START
import numpy as np
from autoresearch_suite.chem_clusters import relax
from autoresearch_suite.chem_clusters import dzugutov_energy_gradient
import time

# Global deadline tracking
_START_TIME = None
_DEADLINE = 330.0  # 300s limit + safety margin


def _get_icosahedron_vertices():
    """Return the 12 vertices of a regular icosahedron centered at origin."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    return verts


def _generate_mackay_icosahedron(n_target, spacing=1.1):
    """
    Generate a Mackay icosahedral cluster with exactly n_target atoms.
    Shell structure:
      - Core: 1 atom at origin
      - Shell 1 (12 atoms): icosahedron vertices at radius r1
      - Shell 2 (42 atoms): icosahedral second shell at radius r2
    
    For n up to 60, we use core + shell1 + part of shell2.
    """
    # Core atom
    points = [np.array([0.0, 0.0, 0.0])]
    
    # Shell 1: 12 icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(ico_verts, axis=1)
    ico_verts = ico_verts / norms[:, None]
    
    r1 = spacing * 1.0  # First shell radius
    shell1 = ico_verts * r1
    
    # Add shell 1 atoms
    for v in shell1:
        if len(points) < n_target:
            points.append(v)
    
    if n_target <= 13:
        return np.array(points[:n_target])
    
    # Shell 2: 42 atoms - generated from icosahedral symmetry
    # The 42 atoms are at:
    # - 12 vertices of a larger icosahedron (at icosahedron vertex positions, scaled)
    # - 20 face centers
    # - 10 edge midpoints
    
    # Generate all 42 positions using icosahedral symmetry
    # Vertices of the icosahedron (12)
    v1 = np.array([0, 1, phi], dtype=float)
    v1 = v1 / np.linalg.norm(v1)
    
    # We need to generate 42 points with icosahedral symmetry
    # Using the known positions for the second shell of a Mackay icosahedron
    
    # The 42 positions can be generated as follows:
    # Take the 12 icosahedron vertices and create 3 orbits under the icosahedral group
    
    # Orbit 1: 12 points at the icosahedron vertices (scaled to r2)
    # Orbit 2: 20 points at face centers
    # Orbit 3: 10 points at edge midpoints
    
    r2 = spacing * np.sqrt(2)  # Second shell radius (approximate)
    
    # For simplicity, use a Fibonacci sphere for the second shell
    # This gives a good approximation of icosahedral symmetry
    
    n_shell2 = min(42, n_target - len(points))
    if n_shell2 > 0:
        # Fibonacci sphere for quasi-uniform distribution
        golden_ratio = (1 + np.sqrt(5)) / 2
        angles = []
        for i in range(n_shell2):
            theta = np.pi * (1 - 2 * (i + 0.5) / n_shell2)
            phi_angle = 2 * np.pi * i / golden_ratio
            x = np.sin(theta) * np.cos(phi_angle)
            y = np.sin(theta) * np.sin(phi_angle)
            z = np.cos(theta)
            angles.append(np.array([x, y, z]))
        
        for v in angles:
            if len(points) < n_target:
                points.append(v * r2)
    
    return np.array(points[:n_target])


def _generate_defect_icosahedron(n_target, spacing=1.1, rng=None):
    """
    Generate a defect-seeded icosahedral cluster by removing atoms from a full Mackay structure.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Start with a full 55-atom Mackay icosahedron (1 + 12 + 42)
    full_mackay = _generate_mackay_icosahedron(55, spacing)
    
    if n_target >= 55:
        # For 57, 60: add extra atoms in a shell
        extra = n_target - 55
        # Add atoms on a sphere of radius r3
        r3 = spacing * 2.0
        golden_ratio = (1 + np.sqrt(5)) / 2
        added = []
        for i in range(extra):
            theta = np.pi * (1 - 2 * (i + 0.5) / max(extra, 1))
            phi_angle = 2 * np.pi * i / golden_ratio
            x = np.sin(theta) * np.cos(phi_angle)
            y = np.sin(theta) * np.sin(phi_angle)
            z = np.cos(theta)
            added.append(np.array([x, y, z]) * r3)
        return np.vstack([full_mackay, np.array(added)])
    
    # For n < 55: remove atoms from the full structure
    # Prefer removing from the outer shell (least coordinated)
    # Atoms 13-54 are in shell 2, atoms 1-12 are in shell 1, atom 0 is core
    
    # Shuffle outer shell atoms and remove
    outer_indices = list(range(13, 55))
    rng.shuffle(outer_indices)
    
    n_to_remove = 55 - n_target
    remove_set = set(outer_indices[:n_to_remove])
    
    keep_indices = [i for i in range(55) if i not in remove_set]
    return full_mackay[keep_indices]


def _relax_with_gradient(positions, potential='dzugutov', max_iter=500, tol=1e-8):
    """Local minimization using gradient-based optimization."""
    pos = positions.copy()
    n = len(pos)
    
    for iteration in range(max_iter):
        # Check deadline
        if _START_TIME is not None:
            elapsed = time.time() - _START_TIME
            if elapsed > _DEADLINE - 5:
                break
        
        try:
            energy, grad = dzugutov_energy_gradient(pos)
        except Exception:
            break
        
        # Simple gradient descent with adaptive step size
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        # Adaptive step size
        step_size = min(0.01, 0.1 / (grad_norm + 1e-10))
        pos = pos - step_size * grad
        
        # Ensure minimum distance constraint
        # Simple check: if any pair is too close, push apart
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < 0.5:
                    # Push apart
                    if d > 1e-10:
                        direction = (pos[i] - pos[j]) / d
                        push = (0.5 - d) / 2
                        pos[i] += direction * push
                        pos[j] -= direction * push
                    else:
                        # Random small displacement
                        pos[i] += rng.uniform(-0.1, 0.1, 3)
                        pos[j] += rng.uniform(-0.1, 0.1, 3)
    
    return pos


def _basin_hopping(positions, potential='dzugutov', n_iterations=50, step_scale=0.3, rng=None):
    """Basin hopping algorithm for global optimization."""
    if rng is None:
        rng = np.random.default_rng()
    
    best_pos = positions.copy()
    best_energy = None
    
    try:
        best_energy = relax(positions, potential)[0]
    except Exception:
        return positions
    
    current_pos = positions.copy()
    current_energy = best_energy
    
    for i in range(n_iterations):
        # Check deadline
        if _START_TIME is not None:
            elapsed = time.time() - _START_TIME
            if elapsed > _DEADLINE - 10:
                break
        
        # Perturbation
        perturbation = rng.normal(0, step_scale, size=current_pos.shape)
        perturbed_pos = current_pos + perturbation
        
        # Relax
        try:
            energy, relaxed_pos = relax(perturbed_pos, potential)
        except Exception:
            continue
        
        # Metropolis accept/reject
        if best_energy is None:
            best_energy = energy
            best_pos = relaxed_pos.copy()
        
        if energy < current_energy or rng.random() < np.exp(-(energy - current_energy) / 0.1):
            current_pos = relaxed_pos.copy()
            current_energy = energy
            
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed_pos.copy()
        
        # Adapt step size
        step_scale *= 0.98
    
    return best_pos


def construct(instance):
    """
    Find low-energy structures for Dzugutov clusters using Mackay icosahedron-based initialization.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    
    rng = np.random.default_rng(size)
    
    best_energy = np.inf
    best_positions = None
    
    # Time budget per instance
    per_instance_time = 50.0  # ~50 seconds per instance, 6 instances = 300s
    
    start_instance = time.time()
    
    # Strategy 1: Defect-seeded Mackay icosahedron
    n_attempts = 8
    
    for attempt in range(n_attempts):
        # Check time
        if time.time() - start_instance > per_instance_time:
            break
        
        # Generate initial structure
        if attempt == 0:
            # Perfect Mackay icosahedron (or closest)
            positions = _generate_defect_icosahedron(size, spacing=1.1, rng=rng)
        elif attempt == 1:
            # Slightly perturbed Mackay
            positions = _generate_defect_icosahedron(size, spacing=1.1, rng=rng)
            positions += rng.normal(0, 0.05, size=positions.shape)
        elif attempt == 2:
            # Different spacing
            positions = _generate_defect_icosahedron(size, spacing=1.05, rng=rng)
        elif attempt == 3:
            # Another spacing
            positions = _generate_defect_icosahedron(size, spacing=1.15, rng=rng)
        else:
            # Random perturbations of the base structure
            base = _generate_defect_icosahedron(size, spacing=1.1, rng=rng)
            pert_scale = 0.1 * (1 + attempt * 0.2)
            positions = base + rng.normal(0, pert_scale, size=base.shape)
        
        # Ensure minimum distance
        # Simple fix: if any pair is too close, displace
        n = len(positions)
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < 0.5:
                    if d > 1e-10:
                        direction = (positions[i] - positions[j]) / d
                        push = (0.5 - d) / 2
                        positions[i] += direction * push
                        positions[j] -= direction * push
                    else:
                        positions[i] += rng.uniform(-0.1, 0.1, 3)
        
        # Relax
        try:
            energy, relaxed_positions = relax(positions, potential)
        except Exception:
            continue
        
        # Basin hopping from this local minimum
        bh_positions = _basin_hopping(
            relaxed_positions, potential, 
            n_iterations=20, 
            step_scale=0.2, 
            rng=rng
        )
        
        try:
            bh_energy, bh_final = relax(bh_positions, potential)
        except Exception:
            bh_energy = energy
            bh_final = relaxed_positions
        
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_positions = bh_final.copy()
    
    # If no valid result found, fall back to random
    if best_positions is None:
        # Fallback: random compact cluster
        radius = 1.1 * size ** (1.0 / 3.0) * 0.75
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(
                np.linalg.norm(p - q) >= 0.8 * 1.1 for q in points
            ):
                points.append(p)
        
        try:
            energy, positions = relax(np.array(points), potential)
            best_positions = positions
            best_energy = energy
        except Exception:
            # Last resort: simple cubic grid
            n_side = int(np.ceil(size ** (1.0 / 3.0)))
            grid = []
            for i in range(n_side):
                for j in range(n_side):
                    for k in range(n_side):
                        if len(grid) < size:
                            grid.append(np.array([i, j, k]) * 1.1)
            best_positions = np.array(grid[:size])
    
    return best_positions
# EVOLVE-BLOCK-END
