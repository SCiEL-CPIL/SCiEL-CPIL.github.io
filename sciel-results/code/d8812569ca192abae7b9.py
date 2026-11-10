"""Morse cluster global minima: basin hopping with diverse crystalline/shell starts and deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy(pos, rho):
    """Compute Morse energy E = sum_{i<j} exp(rho(1-r_ij))(exp(rho(1-r_ij)) - 2)."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(r, np.inf)
    iu = np.triu_indices(n, k=1)
    r_ij = r[iu]
    x = rho * (1.0 - r_ij)
    e = np.exp(x) * (np.exp(x) - 2.0)
    return float(np.sum(e))


def _is_valid(pos, min_dist=0.5):
    """Check that all interatomic distances are >= min_dist."""
    n = len(pos)
    if n < 2:
        return True
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(r, np.inf)
    return np.min(r) >= min_dist


def _generate_fcc(n, spacing=1.0):
    """Generate an FCC-like cluster of n atoms."""
    # FCC lattice points in a cube, then pick closest to origin
    a = spacing
    points = []
    # Generate FCC lattice points in a region
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                # FCC: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) * a
                for dx, dy, dz in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    points.append(p)
    points = np.array(points)
    # Sort by distance from origin and take first n
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx].copy()


def _generate_bcc(n, spacing=1.0):
    """Generate a BCC-like cluster of n atoms."""
    a = spacing
    points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                # BCC: (0,0,0), (0.5,0.5,0.5) * a
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0.5)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    points.append(p)
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx].copy()


def _generate_hcp(n, spacing=1.0):
    """Generate an HCP-like cluster of n atoms."""
    a = spacing
    c = a * np.sqrt(8.0 / 3.0)
    points = []
    for layer in range(-3, 4):
        z = layer * c
        # AB stacking: even layers at (0,0), odd at (a/2, a/(2*sqrt(3)))
        offset = np.array([0.0, 0.0]) if layer % 2 == 0 else np.array([a/2.0, a/(2.0*np.sqrt(3.0))])
        for i in range(-4, 5):
            for j in range(-4, 5):
                # Hexagonal lattice
                x = (i + 0.5*(j % 2)) * a
                y = j * a * np.sqrt(3.0) / 2.0
                p = np.array([x + offset[0], y + offset[1], z])
                points.append(p)
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx].copy()


def _generate_shell(n, spacing=1.0):
    """Generate a shell-like cluster: inner core + outer shell."""
    rng = np.random.default_rng(42)
    # Inner core: small FCC-like
    core_n = max(1, int(n ** (1.0/3.0)))
    core_n = min(core_n, n)
    core = _generate_fcc(core_n, spacing)
    
    remaining = n - len(core)
    if remaining <= 0:
        return core[:n]
    
    # Outer shell: points on a sphere
    shell_radius = spacing * (n / (4.0/3.0 * np.pi)) ** (1.0/3.0) * 1.2
    shell_points = []
    # Fibonacci sphere for even distribution
    golden = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(remaining):
        y = 1.0 - (i / max(remaining - 1, 1)) * 2.0
        radius_y = np.sqrt(max(1.0 - y**2, 0.0))
        theta = golden * i
        x = np.cos(theta) * radius_y
        z = np.sin(theta) * radius_y
        p = np.array([x, y, z]) * shell_radius
        shell_points.append(p)
    shell_points = np.array(shell_points)
    
    # Combine and ensure minimum distance
    combined = np.vstack([core, shell_points])
    # Nudge to satisfy min distance
    for i in range(len(combined)):
        for j in range(i+1, len(combined)):
            d = np.linalg.norm(combined[i] - combined[j])
            if d < 0.5 * spacing:
                diff = combined[j] - combined[i]
                if d > 1e-10:
                    diff = diff / d * (0.5 * spacing - d + 0.01)
                else:
                    diff = np.array([0.01, 0.01, 0.01])
                combined[j] += diff
    return combined


def _generate_random(n, spacing=1.0, seed=0):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.75
    points = []
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 * spacing for q in points):
            points.append(p)
    return np.array(points)


def _perturb(pos, scale, rng):
    """Apply random perturbation to positions."""
    return pos + scale * rng.standard_normal(pos.shape)


def construct(instance):
    """Find low-energy Morse cluster via basin hopping from diverse starts."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 6))
    potential = instance["potential"]
    spacing = 1.0
    
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 330s kill
    
    # Generate diverse starting configurations
    starts = []
    try:
        starts.append(_generate_fcc(n, spacing))
    except:
        pass
    try:
        starts.append(_generate_bcc(n, spacing))
    except:
        pass
    try:
        starts.append(_generate_hcp(n, spacing))
    except:
        pass
    try:
        starts.append(_generate_shell(n, spacing))
    except:
        pass
    # Add a few random starts
    for seed in [0, 1, 2]:
        try:
            starts.append(_generate_random(n, spacing, seed))
        except:
            pass
    
    # Filter valid starts
    valid_starts = []
    for s in starts:
        if len(s) == n and _is_valid(s, 0.5):
            valid_starts.append(s)
    
    if not valid_starts:
        # Fallback: simple random
        rng = np.random.default_rng(0)
        radius = spacing * n ** (1.0/3.0) * 0.8
        points = []
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
        valid_starts = [np.array(points)]
    
    # Basin hopping from each start
    best_energy = np.inf
    best_pos = valid_starts[0].copy()
    
    for start_idx, start_pos in enumerate(valid_starts):
        if time.time() > deadline:
            break
        
        # Local relax
        try:
            energy, pos = relax(start_pos.copy(), potential, rho)
            if energy < best_energy:
                best_energy = energy
                best_pos = pos.copy()
        except:
            continue
        
        # Basin hopping iterations
        rng = np.random.default_rng(start_idx * 1000 + 42)
        current_energy = energy
        current_pos = pos.copy()
        bh_scale = 0.3  # Initial perturbation scale
        
        for bh_iter in range(50):
            if time.time() > deadline:
                break
            
            # Perturb
            new_pos = _perturb(current_pos, bh_scale, rng)
            
            # Relax
            try:
                new_energy, new_pos = relax(new_pos, potential, rho)
            except:
                continue
            
            # Metropolis accept
            if new_energy < current_energy:
                current_energy = new_energy
                current_pos = new_pos
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pos = current_pos.copy()
                # Reduce scale slightly on success
                bh_scale = max(0.1, bh_scale * 0.95)
            else:
                # Accept with probability exp(-dE/T)
                # Temperature decreases over iterations
                T = 0.5 * (1.0 - bh_iter / 50.0) + 0.01
                if T > 0.01:
                    dE = new_energy - current_energy
                    if dE < 0 or rng.random() < np.exp(-dE / T):
                        current_energy = new_energy
                        current_pos = new_pos
                        # Increase scale on rejection
                        bh_scale = min(0.8, bh_scale * 1.1)
                    else:
                        bh_scale = min(0.8, bh_scale * 1.05)
                else:
                    bh_scale = min(0.8, bh_scale * 1.05)
    
    # Final polish: high-precision relax from best
    if time.time() < deadline - 5.0:
        try:
            final_energy, final_pos = relax(best_pos.copy(), potential, rho)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pos = final_pos
        except:
            pass
    
    return best_pos
# EVOLVE-BLOCK-END
