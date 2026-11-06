import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient
import scipy.optimize

def _morse_energy(pos, rho):
    """Compute Morse energy."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    triu_idx = np.triu_indices(n, 1)
    r = dists[triu_idx]
    x = rho * (1.0 - r)
    e = np.exp(x)
    energy = np.sum(e * (e - 2.0))
    return energy

def _morse_gradient(pos, rho):
    """Compute Morse gradient."""
    n = len(pos)
    grad = np.zeros_like(pos)
    for i in range(n):
        for j in range(i+1, n):
            rij = pos[i] - pos[j]
            r = np.linalg.norm(rij)
            if r < 1e-10:
                continue
            x = rho * (1.0 - r)
            e = np.exp(x)
            # dE/dr = rho * e * (2e - 2) * (-1) = -2*rho*e*(e-1)
            dE_dr = -2.0 * rho * e * (e - 1.0)
            # dE/dri = dE/dr * (ri - rj)/r
            force = dE_dr * rij / r
            grad[i] += force
            grad[j] -= force
    return grad

def _min_dist_check(pos, min_d=0.5):
    """Check minimum distance constraint."""
    n = len(pos)
    if n < 2:
        return True
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= min_d

def _fibonacci_sphere(n, radius):
    """Generate n points on a sphere using Fibonacci lattice, then project to 3D cluster."""
    if n <= 1:
        return np.zeros((n, 3))
    
    # Fibonacci sphere for even distribution on sphere surface
    golden_ratio = (1 + np.sqrt(5)) / 2
    indices = np.arange(n)
    theta = 2 * np.pi * indices / golden_ratio
    # For a 3D cluster, use a shell approach: place points at various radii
    # Use a spiral/radial distribution for compact 3D packing
    
    # Better: use a 3D spiral (Fibonacci-like) for compact packing
    # Place points along a radial spiral
    t = np.linspace(0, 1, n)
    r = radius * np.cbrt(t)  # uniform volume distribution
    phi = 2 * np.pi * t * golden_ratio * 10  # angular spread
    theta = np.arccos(1 - 2 * t)
    
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    return np.column_stack([x, y, z])

def _compact_seed(n, radius, rng):
    """Generate a compact initial configuration using Fibonacci-like packing."""
    pos = _fibonacci_sphere(n, radius)
    # Center it
    pos -= np.mean(pos, axis=0)
    # Add tiny noise to break exact symmetry
    pos += rng.normal(0, 0.01, size=pos.shape)
    return pos

def _fcc_seed(n, rng):
    """Generate FCC-like seed."""
    basis = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5]
    ])
    side = int(np.ceil((0.25 * n) ** (1.0/3.0))) + 2
    pts = []
    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                for b in basis:
                    p = np.array([i, j, k]) + b
                    pts.append(p)
    pts = np.array(pts)
    pts -= np.mean(pts, axis=0)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    cluster = pts[idx]
    cluster += rng.normal(0, 0.03, size=cluster.shape)
    return cluster

def _simulated_annealing(pos0, rho, n, rng, time_budget, T_init=1.0, T_final=0.001, cooling=0.9995):
    """
    Run simulated annealing with adaptive kick sizes.
    Returns best position and energy found.
    """
    current_pos = pos0.copy()
    current_energy = _morse_energy(current_pos, rho)
    
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    # Adaptive kick parameters
    kick_sigma = 0.15
    T = T_init
    
    start_t = time.time()
    steps = 0
    no_improve = 0
    
    while time.time() - start_t < time_budget:
        # Generate perturbation
        kick = rng.normal(0, kick_sigma, size=(n, 3))
        new_pos = current_pos + kick
        
        # Check min distance; if violated, shrink kick
        if not _min_dist_check(new_pos):
            # Try smaller kick
            for _ in range(5):
                kick_sigma *= 0.7
                kick = rng.normal(0, kick_sigma, size=(n, 3))
                new_pos = current_pos + kick
                if _min_dist_check(new_pos):
                    break
            if not _min_dist_check(new_pos):
                T *= cooling
                continue
        
        # Evaluate new energy
        new_energy = _morse_energy(new_pos, rho)
        
        # Metropolis acceptance
        if new_energy < current_energy:
            current_pos = new_pos
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
                no_improve = 0
            else:
                no_improve += 1
        else:
            delta_e = new_energy - current_energy
            if delta_e / T < 50:  # avoid overflow
                if rng.random() < np.exp(-delta_e / T):
                    current_pos = new_pos
                    current_energy = new_energy
            no_improve += 1
        
        # Cool down
        T *= cooling
        
        # Adaptive kick: shrink if too many rejections, grow if stuck
        if no_improve > 50:
            kick_sigma *= 0.8
            no_improve = 0
        elif steps % 100 == 0 and kick_sigma < 0.3:
            kick_sigma *= 1.1
        
        steps += 1
    
    return best_pos, best_energy

def _basin_hopping(pos0, rho, n, rng, time_budget, kick_sigma_init=0.1, n_restarts=1):
    """
    Basin hopping: kick + local minimize, accept/reject.
    Uses relax() for local minimization.
    """
    current_pos = pos0.copy()
    current_energy = _morse_energy(current_pos, rho)
    
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    kick_sigma = kick_sigma_init
    start_t = time.time()
    
    while time.time() - start_t < time_budget:
        # Kick
        new_pos = current_pos + rng.normal(0, kick_sigma, size=(n, 3))
        
        if not _min_dist_check(new_pos):
            # Try to fix by shrinking
            for _ in range(3):
                new_pos = current_pos + rng.normal(0, kick_sigma * 0.5, size=(n, 3))
                if _min_dist_check(new_pos):
                    break
            if not _min_dist_check(new_pos):
                kick_sigma *= 0.8
                continue
        
        # Local minimize using relax
        try:
            energy, positions = relax(new_pos, 'morse', rho)
        except Exception:
            continue
        
        if energy < current_energy:
            current_pos = positions
            current_energy = energy
            if energy < best_energy:
                best_energy = energy
                best_pos = positions.copy()
            kick_sigma = max(0.05, kick_sigma * 0.95)
        else:
            delta_e = energy - current_energy
            T = 0.1
            if delta_e / T < 50 and rng.random() < np.exp(-delta_e / T):
                current_pos = positions
                current_energy = energy
            kick_sigma = min(0.3, kick_sigma * 1.05)
    
    return best_pos, best_energy

def _polish_lbfgs(pos, rho, max_iter=2000):
    """Final L-BFGS-B polish."""
    try:
        def objective(x):
            p = x.reshape(-1, 3)
            if not _min_dist_check(p):
                return 1e10
            e, g = morse_energy_gradient(p, rho)
            return e
        
        def gradient(x):
            p = x.reshape(-1, 3)
            if not _min_dist_check(p):
                return np.zeros_like(x)
            e, g = morse_energy_gradient(p, rho)
            return g.flatten()
        
        x0 = pos.flatten()
        res = scipy.optimize.minimize(
            objective, x0, jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'gtol': 1e-10, 'ftol': 1e-12}
        )
        if res.fun < 1e9:
            return res.fun, res.x.reshape(-1, 3)
    except Exception:
        pass
    return _morse_energy(pos, rho), pos

def construct(instance):
    """
    Find the lowest-energy arrangement of n atoms bound by Morse potentials.
    Strategy: Simulated annealing + basin hopping with Fibonacci-sphere and FCC seeds.
    """
    n = instance["n"]
    rho = instance.get("rho", 3)
    potential = instance["potential"]
    
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin for 330s total / 6 instances
    
    rng = np.random.default_rng(12345 + n * 100 + int(rho * 10))
    
    best_energy = np.inf
    best_positions = None
    
    # Estimate cluster radius
    radius = (n / 4.0) ** (1.0/3.0) * 1.2
    
    # Generate initial seeds
    seeds = []
    
    # Seed 1: Fibonacci sphere compact packing
    try:
        s1 = _compact_seed(n, radius, rng)
        if _min_dist_check(s1):
            seeds.append(s1)
        else:
            # Fix by scaling
            s1_fixed = s1 * 1.2
            if _min_dist_check(s1_fixed):
                seeds.append(s1_fixed)
    except Exception:
        pass
    
    # Seed 2: FCC
    try:
        s2 = _fcc_seed(n, rng)
        if _min_dist_check(s2):
            seeds.append(s2)
    except Exception:
        pass
    
    # Seed 3: Random compact
    try:
        s3 = rng.uniform(-radius, radius, size=(n, 3))
        # Check and fix
        if not _min_dist_check(s3):
            # Scale up
            for scale in [1.1, 1.3, 1.5]:
                if _min_dist_check(s3 * scale):
                    s3 = s3 * scale
                    break
        if _min_dist_check(s3):
            seeds.append(s3)
    except Exception:
        pass
    
    if not seeds:
        seeds.append(rng.uniform(-radius, radius, size=(n, 3)))
    
    # Phase 1: Quick relax of all seeds to find good starting point
    for seed in seeds:
        if time.time() > deadline - 10.0:
            break
        try:
            energy, positions = relax(seed, potential, rho)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            continue
    
    if best_positions is None:
        return seeds[0]
    
    # Phase 2: Basin hopping from best configuration (main search)
    remaining = deadline - time.time() - 5.0  # Reserve 5s for polish
    if remaining > 5.0:
        # Split time: 60% basin hopping, 40% simulated annealing
        bh_time = remaining * 0.55
        sa_time = remaining * 0.35
        
        # Basin hopping
        if bh_time > 2.0:
            bh_pos, bh_energy = _basin_hopping(
                best_positions, rho, n, rng, bh_time,
                kick_sigma_init=0.08
            )
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_positions = bh_pos
        
        # Simulated annealing from current best
        if time.time() < deadline - 5.0:
            sa_time_actual = min(sa_time, deadline - time.time() - 5.0)
            if sa_time_actual > 2.0:
                sa_pos, sa_energy = _simulated_annealing(
                    best_positions, rho, n, rng, sa_time_actual,
                    T_init=0.5, T_final=0.001, cooling=0.9998
                )
                if sa_energy < best_energy:
                    best_energy = sa_energy
                    best_positions = sa_pos
    
    # Phase 3: Final polish
    if time.time() < deadline - 2.0:
        polished_energy, polished_pos = _polish_lbfgs(best_positions, rho, max_iter=1000)
        if polished_energy < best_energy:
            best_energy = polished_energy
            best_positions = polished_pos
    
    # Final validation
    if not _min_dist_check(best_positions):
        # Try to fix
        for scale in [1.05, 1.1, 1.2]:
            if _min_dist_check(best_positions * scale):
                best_positions = best_positions * scale
                break
        else:
            # Fallback to first seed
            best_positions = seeds[0]
    
    return best_positions
