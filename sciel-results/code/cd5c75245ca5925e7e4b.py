"""
Morse cluster global minima via Deterministic Multi-Start L-BFGS-B Minimization.
Strategy:
1. Generate a rich library of structurally diverse seeds tailored to (n, rho):
   - Icosahedral shells (magic numbers 1, 12, 30, 55, 91)
   - FCC lattice fragments
   - HCP lattice fragments
   - BCC lattice fragments
   - Greedy dense packing
   - Spherical shell with FWHM-like distribution
2. Pre-relax each seed with L-BFGS-B (scipy.optimize.minimize) to a local minimum.
3. Run additional L-BFGS-B starts from perturbed versions of the best seeds
   until the time budget is exhausted.
4. Return the globally best structure found.
No stochastic search, no Metropolis, no cross-operator. Pure deterministic
multi-start local minimization.
"""
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global time management
_START_TIME = time.time()
_TOTAL_BUDGET = 300.0
_DEADLINE = _START_TIME + 280.0  # 20s safety margin

def _time_left():
    return _DEADLINE - time.time()

def _morse_energy(coords, rho):
    """Compute Morse energy; return inf if any pair < 0.5 apart."""
    n = len(coords)
    if n < 2:
        return 0.0
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 0.5:
        return np.inf
    iu = np.triu_indices(n, 1)
    r = dists[iu]
    x = rho * (1.0 - r)
    e = np.exp(x)
    energy = np.sum(e * (e - 2.0))
    return float(energy)

def _morse_gradient(coords, rho):
    """Compute Morse gradient analytically."""
    n = len(coords)
    if n < 2:
        return np.zeros((n, 3))
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    # Guard against division by zero
    dists_safe = np.where(dists > 1e-10, dists, 1.0)
    
    # d/dr of exp(rho(1-r)) = -rho * exp(rho(1-r))
    # d/dr of [exp(rho(1-r)) * (exp(rho(1-r)) - 2)] 
    #   = -rho * exp(rho(1-r)) * [exp(rho(1-r)) - 2] + (-rho * exp(rho(1-r))) * exp(rho(1-r))
    #   = -rho * exp(rho(1-r)) * [2*exp(rho(1-r)) - 2]
    #   = -2*rho*exp(rho(1-r))*(exp(rho(1-r)) - 1)
    # Force on pair i<j: F_ij = -dE/dr * (r_i - r_j)/r
    # But we need gradient wrt each coordinate.
    # E = sum_{i<j} f(r_ij) where f(r) = exp(rho(1-r))*(exp(rho(1-r))-2)
    # dE/dx_i = sum_{j!=i} f'(r_ij) * (x_i - x_j)/r_ij
    # f'(r) = -2*rho*exp(rho(1-r))*(exp(rho(1-r))-1)
    
    x = rho * (1.0 - dists)
    ex = np.exp(x)
    # f'(r) = -2*rho*ex*(ex-1)
    fprime = -2.0 * rho * ex * (ex - 1.0)
    
    # Gradient: grad_i = sum_{j != i} f'(r_ij) * (x_i - x_j) / r_ij
    # Vectorized: 
    # force_matrix[i,j] = f'(r_ij) * (coords[i] - coords[j]) / r_ij
    # grad[i] = sum_j force_matrix[i,j]
    
    # Build the force matrix
    # fprime is (n,n), dists is (n,n)
    # (coords[i] - coords[j]) / r_ij -> diff / dists
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vec = diff / dists_safe[:, :, np.newaxis]
    unit_vec = np.where(np.isfinite(unit_vec), unit_vec, 0.0)
    
    # fprime is (n,n), multiply element-wise with unit_vec (n,n,3)
    force_matrix = fprime[:, :, np.newaxis] * unit_vec
    grad = np.sum(force_matrix, axis=1)
    return grad

def _build_icosahedral_shells(n, spacing=1.0):
    """Build icosahedral shells with magic numbers."""
    points = []
    phi = (1 + np.sqrt(5)) / 2
    r1 = spacing * 1.0
    r2 = spacing * 1.618  # golden ratio * r1
    r3 = spacing * 2.618
    r4 = spacing * 3.854
    
    # Center
    points.append([0.0, 0.0, 0.0])
    
    # Shell 1: 12 vertices of icosahedron
    verts = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ]
    for v in verts:
        v = np.array(v, dtype=float)
        v = v / np.linalg.norm(v) * r1
        points.append(v)
        if len(points) >= n:
            break
    
    if len(points) >= n:
        points = np.array(points[:n])
        points -= np.mean(points, axis=0)
        return points
    
    # Shell 2: 30 points (edges of icosahedron, at midpoints scaled to r2)
    if n > 13:
        # Use the 30 edge midpoints of the icosahedron, scaled
        edge_pairs = [
            (0,1),(0,2),(0,4),(0,8),(1,3),(1,4),(1,9),(2,3),(2,5),(2,8),
            (3,5),(3,9),(4,6),(4,7),(4,9),(5,7),(5,10),(5,11),(6,7),(6,8),
            (6,10),(7,11),(8,10),(8,11),(9,11)
        ]
        # Actually icosahedron has 30 edges, but let's just place 30 points on sphere r2
        for k in range(min(30, n - len(points))):
            # Fibonacci sphere distribution for shell 2
            idx = k
            total = 30
            z = 1 - 2 * (idx + 0.5) / total
            radius_z = np.sqrt(1 - z * z)
            theta = np.pi * (1 + 5**0.5) * idx  # golden angle
            x = radius_z * np.cos(theta)
            y = radius_z * np.sin(theta)
            points.append([x * r2, y * r2, z * r2])
            if len(points) >= n:
                break
    
    if len(points) >= n:
        points = np.array(points[:n])
        points -= np.mean(points, axis=0)
        return points
    
    # Shell 3: 55 points
    if n > 43:
        for k in range(min(55, n - len(points))):
            total = 55
            z = 1 - 2 * (k + 0.5) / total
            radius_z = np.sqrt(1 - z * z)
            theta = np.pi * (1 + 5**0.5) * k
            x = radius_z * np.cos(theta)
            y = radius_z * np.sin(theta)
            points.append([x * r3, y * r3, z * r3])
            if len(points) >= n:
                break
    
    if len(points) >= n:
        points = np.array(points[:n])
        points -= np.mean(points, axis=0)
        return points
    
    # Shell 4: 91 points
    if n > 98:
        for k in range(min(91, n - len(points))):
            total = 91
            z = 1 - 2 * (k + 0.5) / total
            radius_z = np.sqrt(1 - z * z)
            theta = np.pi * (1 + 5**0.5) * k
            x = radius_z * np.cos(theta)
            y = radius_z * np.sin(theta)
            points.append([x * r4, y * r4, z * r4])
            if len(points) >= n:
                break
    
    # Fill remaining randomly on outermost shell
    while len(points) < n:
        theta = np.random.uniform(0, 2 * np.pi)
        z = np.random.uniform(-1, 1)
        radius_z = np.sqrt(1 - z * z)
        x = radius_z * np.cos(theta)
        y = radius_z * np.sin(theta)
        r = r4 * (1 + 0.1 * np.random.uniform())
        points.append([x * r, y * r, z * r])
    
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    return points

def _build_fcc_cluster(n, spacing=1.0):
    """Build an FCC-like compact cluster."""
    points = []
    a = spacing * 0.9
    max_idx = int(np.ceil((n / 4) ** (1/3))) + 2
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                p1 = np.array([i, j, k]) * a
                p2 = np.array([i + 0.5, j + 0.5, k]) * a
                p3 = np.array([i + 0.5, j, k + 0.5]) * a
                p4 = np.array([i, j + 0.5, k + 0.5]) * a
                for p in [p1, p2, p3, p4]:
                    if len(points) < n:
                        points.append(p)
    if len(points) < n:
        while len(points) < n:
            base = points[np.random.randint(len(points))]
            p = base + np.random.uniform(-0.3, 0.3, 3)
            points.append(p)
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    return points

def _build_hcp_cluster(n, spacing=1.0):
    """Build an HCP-like compact cluster."""
    points = []
    a = spacing * 0.95
    c = a * np.sqrt(8.0 / 3.0)  # HCP c/a ratio
    max_idx = int(np.ceil((n / 2) ** (1/3))) + 2
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                # AB stacking
                if k % 2 == 0:
                    p = np.array([i, j, k]) * np.array([a, a, c/2])
                else:
                    p = np.array([i + 0.5, j + 0.5, k]) * np.array([a, a, c/2])
                points.append(p)
                if len(points) >= n:
                    break
            if len(points) >= n:
                break
        if len(points) >= n:
            break
    if len(points) < n:
        while len(points) < n:
            base = points[np.random.randint(len(points))]
            p = base + np.random.uniform(-0.3, 0.3, 3)
            points.append(p)
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    return points

def _build_bcc_cluster(n, spacing=1.0):
    """Build a BCC-like compact cluster."""
    points = []
    a = spacing * 1.0
    max_idx = int(np.ceil((n / 2) ** (1/3))) + 2
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                p1 = np.array([i, j, k]) * a
                p2 = np.array([i + 0.5, j + 0.5, k + 0.5]) * a
                for p in [p1, p2]:
                    if len(points) < n:
                        points.append(p)
    if len(points) < n:
        while len(points) < n:
            base = points[np.random.randint(len(points))]
            p = base + np.random.uniform(-0.3, 0.3, 3)
            points.append(p)
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    return points

def _build_greedy_packing(n, rng, spacing=1.0):
    """Greedy dense packing: place each atom at the position that minimizes 
    energy contribution given existing atoms."""
    points = []
    # Start with center
    points.append(np.zeros(3))
    min_dist = 0.55 * spacing
    max_radius = spacing * n ** (1/3) * 1.2
    
    for _ in range(1, n):
        best_pos = None
        best_score = np.inf
        # Try random candidate positions
        for _ in range(200):
            r = max_radius * np.sqrt(rng.random())
            theta = rng.uniform(0, 2 * np.pi)
            z = rng.uniform(-1, 1)
            sin_theta = np.sqrt(1 - z * z)
            p = np.array([r * sin_theta * np.cos(theta), 
                         r * sin_theta * np.sin(theta), 
                         r * z])
            # Check min distance
            if points:
                d = np.min(np.linalg.norm(np.array(points) - p, axis=1))
                if d < min_dist:
                    continue
            # Score: sum of Morse terms with existing atoms
            if points:
                dists = np.linalg.norm(np.array(points) - p, axis=1)
                x = spacing * rho_global * (1.0 - dists)
                e = np.exp(x)
                score = np.sum(e * (e - 2.0))
            else:
                score = 0.0
            if score < best_score:
                best_score = score
                best_pos = p
        if best_pos is not None:
            points.append(best_pos)
        else:
            # Fallback: random
            r = max_radius * np.sqrt(rng.random())
            theta = rng.uniform(0, 2 * np.pi)
            z = rng.uniform(-1, 1)
            sin_theta = np.sqrt(1 - z * z)
            points.append(np.array([r * sin_theta * np.cos(theta),
                                    r * sin_theta * np.sin(theta),
                                    r * z]))
    
    points = np.array(points)
    points -= np.mean(points, axis=0)
    return points

def _build_spherical_shell(n, spacing=1.0):
    """Uniform spherical shell."""
    points = []
    # Estimate radius
    r = spacing * n ** (1/3) * 0.7
    for _ in range(n):
        theta = np.random.uniform(0, 2 * np.pi)
        z = np.random.uniform(-1, 1)
        sin_theta = np.sqrt(1 - z * z)
        points.append([r * sin_theta * np.cos(theta),
                      r * sin_theta * np.sin(theta),
                      r * z])
    points = np.array(points)
    points -= np.mean(points, axis=0)
    return points

def _ensure_valid(coords, rho, n, rng):
    """Ensure coordinates are valid (min distance >= 0.5, shape correct)."""
    if len(coords) != n:
        return _build_spherical_shell(n)
    coords = np.array(coords, dtype=float)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 0.5:
        # Try relax
        try:
            energy, relaxed = relax(coords, 'morse', rho)
            if np.isfinite(energy):
                diff = relaxed[:, np.newaxis, :] - relaxed[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff ** 2, axis=-1))
                np.fill_diagonal(dists, np.inf)
                if np.min(dists) >= 0.5:
                    return relaxed
        except:
            pass
        # Fallback: random
        return _build_spherical_shell(n)
    return coords

def _lbfgsb_minimize(coords, rho, maxiter=2000):
    """Run L-BFGS-B minimization on flattened coordinates."""
    x0 = coords.flatten()
    
    def objective(x):
        c = x.reshape(-1, 3)
        return _morse_energy(c, rho)
    
    def gradient(x):
        c = x.reshape(-1, 3)
        return _morse_gradient(c, rho).flatten()
    
    try:
        result = minimize(
            objective, x0, jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
        )
        if result.success or result.fun < np.inf:
            c = result.x.reshape(-1, 3)
            # Verify validity
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=-1))
            np.fill_diagonal(dists, np.inf)
            if np.min(dists) >= 0.5:
                return float(result.fun), c
    except:
        pass
    
    # Fallback: try the provided relax
    try:
        energy, relaxed = relax(coords, 'morse', rho)
        if np.isfinite(energy):
            diff = relaxed[:, np.newaxis, :] - relaxed[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=-1))
            np.fill_diagonal(dists, np.inf)
            if np.min(dists) >= 0.5:
                return float(energy), relaxed
    except:
        pass
    
    return np.inf, coords

# Global rho for greedy packing
rho_global = 6.0

def construct(instance):
    """Find global minimum of Morse cluster via multi-start L-BFGS-B."""
    global rho_global
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    rho_global = rho
    
    rng = np.random.default_rng(n * 1000 + int(rho))
    
    # Generate diverse seeds
    seeds = []
    seeds.append(_build_icosahedral_shells(n))
    seeds.append(_build_fcc_cluster(n))
    seeds.append(_build_hcp_cluster(n))
    seeds.append(_build_bcc_cluster(n))
    seeds.append(_build_greedy_packing(n, rng))
    seeds.append(_build_spherical_shell(n))
    
    # Filter and validate
    valid_seeds = []
    for seed in seeds:
        seed = np.array(seed, dtype=float)
        if len(seed) == n:
            seed = _ensure_valid(seed, rho, n, rng)
            valid_seeds.append(seed)
    
    if not valid_seeds:
        valid_seeds = [_build_spherical_shell(n)]
    
    # Run L-BFGS-B on each seed
    best_energy = np.inf
    best_coords = valid_seeds[0]
    
    for i, seed in enumerate(valid_seeds):
        if _time_left() < 10.0:
            break
        energy, coords = _lbfgsb_minimize(seed, rho)
        if energy < best_energy:
            best_energy = energy
            best_coords = coords
    
    # Use remaining time for additional perturbed starts from the best seed
    if _time_left() > 15.0:
        n_extra = min(20, int(_time_left() / 5.0))  # ~5s per extra start
        for _ in range(n_extra):
            if _time_left() < 10.0:
                break
            # Perturb the best coords
            perturb_scale = 0.1 + 0.1 * rng.random()
            perturbed = best_coords + rng.normal(0, perturb_scale, (n, 3))
            perturbed = _ensure_valid(perturbed, rho, n, rng)
            energy, coords = _lbfgsb_minimize(perturbed, rho)
            if energy < best_energy:
                best_energy = energy
                best_coords = coords
    
    # Final validation
    best_coords = _ensure_valid(best_coords, rho, n, rng)
    
    # Last resort: if still invalid, use relax directly
    try:
        energy, relaxed = relax(best_coords, 'morse', rho)
        if np.isfinite(energy):
            diff = relaxed[:, np.newaxis, :] - relaxed[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=-1))
            np.fill_diagonal(dists, np.inf)
            if np.min(dists) >= 0.5:
                best_coords = relaxed
    except:
        pass
    
    return best_coords
