"""Rewritten program: systematic multi-structure initialization + shake-relax refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _time_left(deadline, margin=5.0):
    return deadline - time.time() - margin > 0


def _is_valid(positions):
    """Check that positions are finite and no two atoms are closer than 0.35."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(positions[i] - positions[j]) < 0.35:
                return False
    return True


def _make_fcc(n, spacing=1.0):
    """Generate n points from an FCC lattice, centered, sorted by radius."""
    pts = []
    max_coord = int(np.ceil(n ** (1.0 / 3.0) * 1.5)) + 3
    fcc_pts = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # FCC lattice points in conventional cell
                # Basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
                for basis in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    p = np.array([i + basis[0], j + basis[1], k + basis[2]]) * spacing
                    fcc_pts.append(p)
    fcc_pts = np.array(fcc_pts)
    # Center
    fcc_pts -= fcc_pts.mean(axis=0)
    # Sort by distance from origin
    dists = np.linalg.norm(fcc_pts, axis=1)
    order = np.argsort(dists)
    pts = fcc_pts[order[:n]]
    return pts


def _make_hcp(n, spacing=1.0):
    """Generate n points from an HCP lattice, centered, sorted by radius."""
    pts = []
    max_coord = int(np.ceil(n ** (1.0 / 3.0) * 1.5)) + 3
    a = spacing
    c = spacing * np.sqrt(8.0 / 3.0)
    hcp_pts = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # HCP: two atoms per layer
                # Layer positions: (i*a + j*a/2, j*a*sqrt(3)/2, k*c)
                # Second atom in layer offset
                p1 = np.array([i * a + j * a / 2, j * a * np.sqrt(3) / 2, k * c])
                p2 = np.array([i * a + j * a / 2 + a / 2, j * a * np.sqrt(3) / 2 + a * np.sqrt(3) / 6, k * c + c / 2])
                hcp_pts.append(p1)
                hcp_pts.append(p2)
    hcp_pts = np.array(hcp_pts)
    hcp_pts -= hcp_pts.mean(axis=0)
    dists = np.linalg.norm(hcp_pts, axis=1)
    order = np.argsort(dists)
    pts = hcp_pts[order[:n]]
    return pts


def _make_icosahedral_mackay(n, spacing=1.0):
    """Generate points from nested icosahedral shells (Mackay icosahedra)."""
    # Icosahedral vertices
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
    ])
    # Normalize to unit radius
    ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    # Shell radii for Mackay icosahedra: 1, 3, 5, 7, 9...
    # Number of atoms per shell: 12, 42, 92, 162, ...
    # Actually Mackay: shell k has 12*k*(k+1)/2 atoms? No.
    # Shell 1: 12 atoms (icosahedron vertices)
    # Shell 2: 42 atoms
    # Shell 3: 92 atoms
    # Shell 4: 162 atoms
    # Total: 1+12=13, +42=55, +92=147, +162=309
    
    shell_counts = [12, 42, 92, 162, 252, 362]
    shell_radii = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0]
    
    pts = [np.zeros(3)]  # center atom
    remaining = n - 1
    
    for shell_idx, (count, radius) in enumerate(zip(shell_counts, shell_radii)):
        if remaining <= 0:
            break
        take = min(remaining, count)
        # For each shell, use icosahedral symmetry
        # For shell k, the points are at positions given by icosahedral group
        # Simplified: just take first 'take' icosahedral vertices scaled
        if take <= 12:
            shell_pts = ico_verts[:take] * radius * spacing
        else:
            # Generate more points using icosahedral symmetry
            # For now, repeat icosahedral vertices with small perturbations
            shell_pts = []
            for i in range(take):
                base = ico_verts[i % 12]
                # Add small perturbation to avoid overlap
                pert = np.random.default_rng(i).uniform(-0.01, 0.01, 3)
                shell_pts.append((base + pert) * radius * spacing)
            shell_pts = np.array(shell_pts)
        pts.append(shell_pts)
        remaining -= take
    
    if len(pts) > 1:
        pts = np.vstack(pts)
    else:
        pts = np.array(pts)
    
    # Center
    pts -= pts.mean(axis=0)
    return pts


def _make_spherical_shells(n, spacing=1.0):
    """Generate points on concentric spherical shells."""
    # Use Fibonacci sphere for each shell
    def fibonacci_sphere(num_points, radius):
        """Distribute points evenly on a sphere using Fibonacci sequence."""
        points = []
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(num_points):
            z = 1 - 2 * i / (num_points - 1) if num_points > 1 else 0
            r = np.sqrt(1 - z * z)
            theta = golden_angle * i
            x = np.cos(theta) * r
            y = np.sin(theta) * r
            points.append([x * radius, y * radius, z * radius])
        return np.array(points)
    
    # Choose shell structure
    # Try to distribute atoms among shells
    shell_counts = []
    remaining = n
    shell_idx = 0
    while remaining > 0:
        # Max atoms on shell k is approximately 2*k*(k+1) for a reasonable density
        max_atoms = 2 * (shell_idx + 1) * (shell_idx + 2)
        take = min(remaining, max_atoms)
        if shell_idx == 0:
            take = min(take, 12)  # Icosahedral for first shell
        shell_counts.append(take)
        remaining -= take
        shell_idx += 1
    
    pts = []
    for i, count in enumerate(shell_counts):
        radius = (i + 1) * spacing * 0.8
        if count <= 12 and i == 0:
            # Use icosahedral vertices for first shell
            phi = (1 + np.sqrt(5)) / 2
            ico_verts = np.array([
                [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
                [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
                [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
            ])
            ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
            pts.append(ico_verts[:count] * radius)
        else:
            pts.append(fibonacci_sphere(count, radius))
    
    pts = np.vstack(pts)
    pts -= pts.mean(axis=0)
    return pts


def _shake_relax(positions, potential, kind, n_shakes=5, shake_amp=0.3):
    """Try multiple shake perturbations and relax each, return best."""
    best_energy = np.inf
    best_positions = positions.copy()
    rng = np.random.default_rng(42)
    
    for _ in range(n_shakes):
        perturbed = positions + rng.normal(0, shake_amp, positions.shape)
        try:
            energy, relaxed = relax(perturbed, potential, kind)
            if _is_valid(relaxed) and energy < best_energy:
                best_energy = energy
                best_positions = relaxed
        except:
            pass
    
    return best_energy, best_positions


def construct(instance):
    """Systematic multi-structure initialization with shake-relax refinement."""
    start_time = time.time()
    deadline = start_time + 280  # Safety margin for 300s limit
    
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    
    # Generate candidate structures
    candidates = []
    
    # FCC fragment
    try:
        fcc_pts = _make_fcc(size)
        if _is_valid(fcc_pts):
            candidates.append(fcc_pts)
    except:
        pass
    
    # HCP fragment
    try:
        hcp_pts = _make_hcp(size)
        if _is_valid(hcp_pts):
            candidates.append(hcp_pts)
    except:
        pass
    
    # Icosahedral Mackay
    try:
        ico_pts = _make_icosahedral_mackay(size)
        if _is_valid(ico_pts):
            candidates.append(ico_pts)
    except:
        pass
    
    # Spherical shells
    try:
        sph_pts = _make_spherical_shells(size)
        if _is_valid(sph_pts):
            candidates.append(sph_pts)
    except:
        pass
    
    # If no valid candidates, use random
    if not candidates:
        rng = np.random.default_rng(size)
        radius = 0.72 * size ** (1.0 / 3.0) * 0.75
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 * 0.72 for q in points):
                points.append(p)
        candidates.append(np.array(points))
    
    # Relax each candidate and keep the best
    best_energy = np.inf
    best_positions = None
    
    for i, candidate in enumerate(candidates):
        if not _time_left(deadline):
            break
        
        # First, direct relaxation
        try:
            energy, relaxed = relax(candidate, potential, kind)
            if _is_valid(relaxed) and energy < best_energy:
                best_energy = energy
                best_positions = relaxed
        except:
            pass
        
        # Then shake-relax on the best so far
        if _time_left(deadline) and best_positions is not None:
            try:
                shake_energy, shake_positions = _shake_relax(
                    best_positions, potential, kind,
                    n_shakes=3, shake_amp=0.2
                )
                if shake_energy < best_energy:
                    best_energy = shake_energy
                    best_positions = shake_positions
            except:
                pass
    
    # Final high-precision polish if time permits
    if _time_left(deadline, margin=30) and best_positions is not None:
        try:
            # Use minimize with tighter tolerance
            def obj_grad(x):
                positions = x.reshape(-1, 3)
                energy, grad = sutton_chen_energy_gradient(positions, kind)
                return energy, grad.flatten()
            
            result = minimize(
                obj_grad,
                best_positions.flatten(),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-12}
            )
            polished = result.x.reshape(-1, 3)
            if _is_valid(polished) and result.fun < best_energy:
                best_positions = polished
        except:
            pass
    
    if best_positions is None:
        # Fallback: return the first valid candidate
        for candidate in candidates:
            if _is_valid(candidate):
                return candidate
        return candidates[0]
    
    return best_positions
# EVOLVE-BLOCK-END
