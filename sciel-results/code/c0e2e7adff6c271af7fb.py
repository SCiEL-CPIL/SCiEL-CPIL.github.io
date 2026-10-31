"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

# Deadline: kill before 150s wall-clock, leave safety margin
DEADLINE = 145.0
START_TIME = time.time()


def lj_energy(positions):
    """Compute total Lennard-Jones energy: 4 * sum(r^-12 - r^-6) over all pairs."""
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            r2 = np.dot(rij, rij)
            if r2 < 1e-12:
                return 1e20
            inv_r2 = 1.0 / r2
            inv_r6 = inv_r2 * inv_r2 * inv_r2
            inv_r12 = inv_r6 * inv_r6
            energy += 4.0 * (inv_r12 - inv_r6)
    return energy


def lj_energy_vectorized(positions):
    """Vectorized LJ energy computation."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    return energy


def check_validity(positions, min_dist=0.5):
    """Check that all atoms are at least min_dist apart."""
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            r = np.sqrt(np.dot(rij, rij))
            if r < min_dist:
                return False
    return True


def make_fcc_truncated_octahedron():
    """
    Create an FCC truncated octahedron cluster of 38 atoms.
    We generate FCC lattice points and select the 38 closest to the origin.
    """
    scale = 1.0 / math.sqrt(2.0)
    
    points = []
    max_coord = 5
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    r2 = i*i + j*j + k*k
                    if r2 <= 50: # Large enough to include 38 points
                        points.append((i * scale, j * scale, k * scale, r2))
    
    points.sort(key=lambda x: x[3])
    selected = [p[:3] for p in points[:38]]
    return np.array(selected, dtype=float)


def make_icosahedral():
    """Create an icosahedral cluster (local minimum)."""
    phi = (1 + math.sqrt(5)) / 2.0
    r = 1.0 / math.sqrt(1 + phi * phi)
    
    ico1 = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]) * r
    
    center = np.array([0.0, 0.0, 0.0])
    shell1 = ico1 * 1.12
    
    # Add more shells to reach 38
    # 1 + 12 = 13. Need 25 more.
    # Add 24 vertices of a truncated octahedron? Or just random?
    # Let's add 24 points on a sphere with radius 1.8
    rng = np.random.default_rng(42)
    shell2 = []
    for _ in range(24):
        theta = rng.uniform(0, 2 * math.pi)
        phi_angle = rng.uniform(0, math.pi)
        r_val = 1.8
        x = r_val * math.sin(phi_angle) * math.cos(theta)
        y = r_val * math.sin(phi_angle) * math.sin(theta)
        z = r_val * math.cos(phi_angle)
        shell2.append([x, y, z])
    
    all_points = np.vstack([center.reshape(1, 3), shell1, np.array(shell2)])
    return all_points[:38]


def random_cluster(radius=2.0, seed=None):
    """Generate a random cluster of 38 atoms within a sphere."""
    rng = np.random.default_rng(seed)
    points = []
    attempts = 0
    while len(points) < N and attempts < 10000:
        theta = rng.uniform(0, 2 * math.pi)
        phi_angle = rng.uniform(0, math.pi)
        r = radius * rng.uniform(0, 1) ** (1/3)
        
        x = r * math.sin(phi_angle) * math.cos(theta)
        y = r * math.sin(phi_angle) * math.sin(theta)
        z = r * math.cos(phi_angle)
        
        p = np.array([x, y, z])
        
        valid = True
        for q in points:
            if np.linalg.norm(p - q) < 0.5:
                valid = False
                break
        if valid:
            points.append(p)
        attempts += 1
    
    if len(points) < N:
        side = int(np.ceil(N ** (1/3)))
        grid = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        grid.sort(key=lambda p: (p[0] - (side-1)/2)**2 + (p[1] - (side-1)/2)**2 + (p[2] - (side-1)/2)**2)
        points = [np.array(p, dtype=float) * 1.12 for p in grid[:N]]
    
    return np.array(points, dtype=float)


def fcc_lattice_fragment(scale=1.12):
    """Generate an FCC lattice fragment."""
    max_coord = 4
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    points.append((i * scale, j * scale, k * scale))
    
    points.sort(key=lambda p: p[0]**2 + p[1]**2 + p[2]**2)
    return np.array(points[:N], dtype=float)


def bcc_lattice_fragment(scale=1.12):
    """Generate a BCC lattice fragment."""
    max_coord = 3
    points = []
    s = scale / math.sqrt(2.0)
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                points.append((i*s, j*s, k*s))
                points.append(((i+0.5)*s, (j+0.5)*s, (k+0.5)*s))
    
    unique_points = []
    for p in points:
        dup = False
        for q in unique_points:
            if np.allclose(p, q, atol=1e-6):
                dup = True
                break
        if not dup:
            unique_points.append(p)
    
    unique_points.sort(key=lambda p: p[0]**2 + p[1]**2 + p[2]**2)
    return np.array(unique_points[:N], dtype=float)


def relax_ljfg(positions, high_precision=False):
    """Relax using L-BFGS if available, otherwise return as-is."""
    try:
        from science_chemistry import relax
        return relax(positions)
    except ImportError:
        def energy_func(pos):
            pos = pos.reshape(N, 3)
            return lj_energy(pos)
        
        options = {'maxiter': 100, 'gtol': 1e-6}
        if high_precision:
            options = {'maxiter': 500, 'gtol': 1e-8}
            
        result = minimize(energy_func, positions.flatten(), method='L-BFGS-B', 
                         options=options)
        return result.x.reshape(N, 3)
    except Exception:
        return positions


def construct():
    """Main entry point: generate many candidates and return the best one found."""
    global START_TIME
    START_TIME = time.time()
    
    best_positions = None
    best_energy = float('inf')
    
    candidates = []
    
    # 1. FCC truncated octahedron (primary candidate for global min)
    try:
        fcc = make_fcc_truncated_octahedron()
        if check_validity(fcc):
            candidates.append(fcc)
    except Exception:
        pass
    
    # 2. FCC lattice fragment (alternative scale)
    try:
        fcc_frag = fcc_lattice_fragment(scale=1.12)
        if check_validity(fcc_frag):
            candidates.append(fcc_frag)
    except Exception:
        pass
    
    # 3. BCC lattice fragment
    try:
        bcc_frag = bcc_lattice_fragment(scale=1.12)
        if check_validity(bcc_frag):
            candidates.append(bcc_frag)
    except Exception:
        pass
    
    # 4. Icosahedral structure
    try:
        ico = make_icosahedral()
        if check_validity(ico):
            candidates.append(ico)
    except Exception:
        pass
    
    # 5. Simple cubic lattice
    side = int(np.ceil(N ** (1/3)))
    points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
    points.sort(key=lambda p: (p[0] - (side-1)/2)**2 + (p[1] - (side-1)/2)**2 + (p[2] - (side-1)/2)**2)
    cubic = np.array(points[:N], dtype=float) * 1.12
    if check_validity(cubic):
        candidates.append(cubic)
    
    # 6. Random clusters
    for seed in range(20):
        try:
            rand = random_cluster(radius=2.5, seed=seed)
            if check_validity(rand):
                candidates.append(rand)
        except Exception:
            pass
    
    # Relax all candidates and find the best
    for i, cand in enumerate(candidates):
        if time.time() - START_TIME > DEADLINE - 10:
            break
        
        try:
            relaxed = relax_ljfg(cand)
            if not check_validity(relaxed):
                continue
            
            energy = lj_energy_vectorized(relaxed)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed
        except Exception:
            try:
                energy = lj_energy_vectorized(cand)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = cand
            except Exception:
                pass
    
    # If we have time, do more random restarts
    elapsed = time.time() - START_TIME
    if best_positions is not None and elapsed < DEADLINE - 30:
        for seed in range(1000):
            if time.time() - START_TIME > DEADLINE - 15:
                break
            
            try:
                rand = random_cluster(radius=2.0 + np.random.default_rng(seed).uniform(-0.5, 1.0), seed=seed)
                if not check_validity(rand):
                    continue
                
                relaxed = relax_ljfg(rand)
                if not check_validity(relaxed):
                    continue
                
                energy = lj_energy_vectorized(relaxed)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed
            except Exception:
                pass
    
    # Final high-precision polish on the best candidate
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 5:
        try:
            polished = relax_ljfg(best_positions, high_precision=True)
            if check_validity(polished):
                energy = lj_energy_vectorized(polished)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = polished
        except Exception:
            pass

    if best_positions is None:
        best_positions = cubic
    
    if not check_validity(best_positions):
        best_positions = cubic
    
    return best_positions

# EVOLVE-BLOCK-END
