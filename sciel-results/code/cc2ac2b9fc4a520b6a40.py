"""Morse cluster global minima search using structured seeds and basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_DEADLINE_MARGIN = 5.0  # seconds to spare before 330s kill


def _morse_energy_and_grad(positions, rho):
    """Compute Morse energy and gradient."""
    n = len(positions)
    energy = 0.0
    grad = np.zeros_like(positions)
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[i] - positions[j]
            r = np.linalg.norm(rij)
            if r < 1e-12:
                continue
            x = rho * (1.0 - r)
            ex = np.exp(x)
            v = ex * (ex - 2.0)
            dv_dr = -rho * ex * (ex - 1.0)
            energy += v
            # dV/dri = dv_dr * (-rij/r)
            factor = -dv_dr / r
            grad[i] += factor * rij
            grad[j] -= factor * rij
    return energy, grad


def _icosahedron_shell(n_atoms, rho):
    """Generate an icosahedral-like cluster with shells."""
    # Icosahedron: 12 vertices, then add shells
    phi = (1 + np.sqrt(5)) / 2
    # Base icosahedron vertices (edge length ~1 for rho=1)
    base = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    # Scale so nearest neighbor distance ~ 1
    # In this construction, edge length is 2, so divide by 2
    base = base / 2.0
    
    points = [np.zeros(3)]  # center atom
    points.extend(base.tolist())
    
    # Add second shell (20 faces -> face centers, but use icosahedral second shell)
    # Second shell of icosahedron has 30 or 42 atoms depending on packing
    # For simplicity, add atoms at face centers of the icosahedron
    if n_atoms > 13:
        # Face centers of icosahedron (20 faces)
        # Find triangles on the icosahedron
        # Simpler: just add more layers with increasing radius
        remaining = n_atoms - len(points)
        if remaining > 0:
            # Add a spherical shell
            n_shell = min(remaining, 30)  # max 30 in this shell
            shell_radius = 1.0 * np.sqrt(2)  # approximate
            for k in range(n_shell):
                # Fibonacci sphere for even distribution
                t = (k + 0.5) / n_shell
                z = 1 - 2 * t
                r = np.sqrt(1 - z * z)
                theta = 2 * np.pi * k / n_shell
                p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * shell_radius
                points.append(p)
            remaining -= n_shell
            if remaining > 0:
                # Third shell
                shell_radius2 = 1.0 * np.sqrt(3)
                for k in range(remaining):
                    t = (k + 0.5) / remaining
                    z = 1 - 2 * t
                    r = np.sqrt(1 - z * z)
                    theta = 2 * np.pi * k / remaining
                    p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * shell_radius2
                    points.append(p)
    
    return np.array(points[:n_atoms])


def _fcc_cluster(n_atoms, rho):
    """Generate an FCC-like cluster."""
    points = []
    # FCC lattice points
    a = 1.0  # lattice constant for nearest neighbor = 1
    # Generate FCC points in a sphere
    max_coord = int(np.ceil(n_atoms ** (1/3)) + 2)
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # FCC: add face-centered positions
                for offset in [
                    [0, 0, 0],
                    [0.5, 0.5, 0],
                    [0.5, 0, 0.5],
                    [0, 0.5, 0.5],
                ]:
                    p = np.array([i, j, k]) * a + np.array(offset) * a
                    if np.linalg.norm(p) < 1.5 * n_atoms ** (1/3):
                        points.append(p)
    # Deduplicate and take nearest n_atoms to center
    if len(points) > 0:
        points = np.array(points)
        norms = np.linalg.norm(points, axis=1)
        idx = np.argsort(norms)
        points = points[idx[:n_atoms]]
    else:
        points = np.zeros((n_atoms, 3))
    return points


def _decahedral_cluster(n_atoms, rho):
    """Generate a decahedral-like cluster (5-fold symmetric)."""
    points = [np.zeros(3)]
    # 5-fold axis: 10 atoms in two rings
    for layer in range(1, 6):
        r = layer * 0.8
        n_in_ring = 10
        for k in range(n_in_ring):
            theta = 2 * np.pi * k / n_in_ring + (np.pi / 10 if layer % 2 else 0)
            p = np.array([r * np.cos(theta), r * np.sin(theta), 0])
            points.append(p)
            if len(points) >= n_atoms:
                break
        if len(points) >= n_atoms:
            break
    return np.array(points[:n_atoms])


def _random_compact(n_atoms, rng, spacing=1.0):
    """Generate a random compact cluster."""
    radius = spacing * n_atoms ** (1/3) * 0.8
    points = []
    attempts = 0
    while len(points) < n_atoms and attempts < n_atoms * 100:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.55 * spacing for q in points):
                points.append(p)
        attempts += 1
    # Fill remaining with random positions
    while len(points) < n_atoms:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)


def _basin_hop_step(positions, rho, rng, step_size):
    """Perform one basin hopping step: perturb, then relax."""
    n = len(positions)
    # Random perturbation
    perturbed = positions + rng.normal(0, step_size, (n, 3))
    # Ensure minimum separation
    # Relax
    try:
        energy, relaxed = relax(perturbed, 'morse', rho)
        return energy, relaxed
    except:
        return np.inf, positions


def construct(instance):
    """Find global minimum of Morse cluster using structured seeds and basin hopping."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 6))
    start_time = time.time()
    
    rng = np.random.default_rng(hash((n, rho)) % (2**32))
    
    # Generate diverse initial configurations
    seeds = []
    
    # 1. Icosahedral
    try:
        seeds.append(_icosahedron_shell(n, rho))
    except:
        pass
    
    # 2. FCC
    try:
        seeds.append(_fcc_cluster(n, rho))
    except:
        pass
    
    # 3. Decahedral
    try:
        seeds.append(_decahedral_cluster(n, rho))
    except:
        pass
    
    # 4-8. Random compact
    for _ in range(5):
        seeds.append(_random_compact(n, rng))
    
    # Evaluate and relax all seeds
    best_energy = np.inf
    best_positions = None
    
    for seed in seeds:
        elapsed = time.time() - start_time
        if elapsed > 330 - _DEADLINE_MARGIN:
            break
        
        try:
            energy, positions = relax(seed, 'morse', rho)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except:
            pass
    
    if best_positions is None:
        # Fallback
        best_positions = _random_compact(n, rng)
        best_energy, best_positions = relax(best_positions, 'morse', rho)
    
    # Basin hopping from best configuration
    step_size = 0.3
    max_hops = 50
    hop_count = 0
    
    while hop_count < max_hops:
        elapsed = time.time() - start_time
        if elapsed > 330 - _DEADLINE_MARGIN:
            break
        
        # Try basin hopping from current best
        try:
            energy, new_pos = _basin_hop_step(best_positions, rho, rng, step_size)
            if energy < best_energy:
                best_energy = energy
                best_positions = new_pos.copy()
                # Reduce step size slightly on success
                step_size *= 0.95
            else:
                # Occasionally try larger step
                if rng.random() < 0.2:
                    step_size = min(step_size * 1.2, 0.5)
        except:
            pass
        
        hop_count += 1
    
    # Final verification: ensure minimum separation
    # If any atoms are too close, try to fix
    for _ in range(10):
        elapsed = time.time() - start_time
        if elapsed > 330 - _DEADLINE_MARGIN:
            break
        dmin = np.inf
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(best_positions[i] - best_positions[j])
                if d < dmin:
                    dmin = d
        if dmin >= 0.5:
            break
        # Nudge atoms apart
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(best_positions[i] - best_positions[j])
                if d < 0.5 and d > 1e-10:
                    direction = (best_positions[i] - best_positions[j]) / d
                    shift = (0.5 - d) / 2 + 0.01
                    best_positions[i] += direction * shift
                    best_positions[j] -= direction * shift
        try:
            energy, best_positions = relax(best_positions, 'morse', rho)
            if energy < best_energy:
                best_energy = energy
        except:
            pass
    
    return best_positions
# EVOLVE-BLOCK-END
