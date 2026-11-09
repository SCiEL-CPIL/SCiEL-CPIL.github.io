"""Morse cluster global minima search: combining diverse structural prototypes with basin hopping."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline management
_START_TIME = time.time()
_TOTAL_BUDGET = 300.0
# We aim to finish by 290s to be safe, leaving 40s buffer for overhead/evaluation
_DEADLINE = _START_TIME + 290.0


def _time_remaining():
    return _DEADLINE - time.time()


def _morse_energy(rho, coords):
    """Compute Morse energy. Returns huge value if any pair is closer than 0.5."""
    n = len(coords)
    if n < 2:
        return 0.0
    d = cdist(coords, coords)
    np.fill_diagonal(d, np.inf)
    min_d = d.min()
    if min_d < 0.5:
        return 1e10
    x = rho * (1.0 - d)
    # Avoid overflow for large negative x (large distances) by clipping, 
    # though for clusters distances are usually small.
    # exp(x) where x is large positive (small r) can overflow.
    # rho is 3, 6, 10. r_min=0.5. x_max = 10*(0.5)=5. exp(5) is fine.
    ex = np.exp(x)
    e_ij = ex * (ex - 2.0)
    return 0.5 * e_ij.sum()


def _morse_energy_grad(rho, coords):
    """Compute Morse energy and gradient."""
    n = len(coords)
    if n < 2:
        return 0.0, np.zeros_like(coords)
    d = cdist(coords, coords)
    np.fill_diagonal(d, np.inf)
    min_d = d.min()
    if min_d < 0.5:
        return 1e10, np.zeros_like(coords)
    x = rho * (1.0 - d)
    ex = np.exp(x)
    e_ij = ex * (ex - 2.0)
    energy = 0.5 * e_ij.sum()

    # Gradient computation
    # dE/dr = -2*rho*ex*(ex-1)
    # grad[i] = sum_j dE/dr_ij * (r_i - r_j)/r_ij
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # (n,n,3)
    d_safe = np.where(d > 1e-8, d, 1.0)
    
    # coeff[i,j] = dE/dr_ij / r_ij
    # dE/dr = -2*rho*exp(x)*(exp(x)-1)
    de_dr = -2.0 * rho * ex * (ex - 1.0)
    coeff = de_dr / d_safe
    np.fill_diagonal(coeff, 0.0)
    
    # grad[i] = sum_j coeff[i,j] * (r_i - r_j)
    # Using einsum for efficiency
    grad = np.einsum('ij,ijc->ic', coeff, diff)
    return energy, grad


def _local_minimize(coords, rho, time_limit=5.0):
    """Minimize energy from given starting point using L-BFGS-B."""
    n = len(coords)

    def obj_grad(x):
        c = x.reshape(-1, 3)
        e, g = _morse_energy_grad(rho, c)
        return e, g

    x0 = coords.flatten()
    start_t = time.time()
    try:
        # Use a moderate maxiter to avoid hanging on difficult landscapes
        res = minimize(obj_grad, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': 200, 'maxls': 30, 'ftol': 1e-10, 'gtol': 1e-6})
        if res.fun < 1e9:
            return res.fun, res.x.reshape(-1, 3)
        else:
            e0, _ = _morse_energy_grad(rho, coords)
            return e0, coords
    except Exception:
        e0, _ = _morse_energy_grad(rho, coords)
        return e0, coords


def _generate_fcc_cluster(n, rng):
    """Generate FCC-like cluster by truncating an FCC lattice."""
    a = 1.0  # equilibrium distance
    points = []
    # FCC basis: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    basis = np.array([[0,0,0],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5]])
    # Generate grid of lattice points
    extent = int(np.ceil(n ** (1/3))) + 3
    for i in range(-extent, extent+1):
        for j in range(-extent, extent+1):
            for k in range(-extent, extent+1):
                origin = np.array([i*a, j*a, k*a])
                for b in basis:
                    p = origin + b * a
                    points.append(p)
    
    points = np.array(points)
    if len(points) <= n:
        # If not enough points, pad with random
        remaining = n - len(points)
        rand_pts = rng.normal(0, 2, (remaining, 3))
        points = np.vstack([points, rand_pts])
    
    # Center and select nearest n points
    center = points.mean(axis=0)
    points -= center
    dists = np.sum(points**2, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]


def _generate_hcp_cluster(n, rng):
    """Generate HCP-like cluster."""
    a = 1.0
    c = a * np.sqrt(8.0/3.0)  # HCP c/a ratio
    points = []
    extent = int(np.ceil(n ** (1/3))) + 3
    for layer in range(-extent, extent+1):
        z = layer * c
        offset = (0.5, 0.0, 0.0) if layer % 2 == 0 else (0.0, 0.0, 0.0)
        for i in range(-extent, extent+1):
            for j in range(-extent, extent+1):
                x = i * a + offset[0] * a
                y = j * a * np.sqrt(3.0)/2.0 + offset[1] * a
                points.append([x, y, z])
    
    points = np.array(points)
    if len(points) <= n:
        remaining = n - len(points)
        rand_pts = rng.normal(0, 2, (remaining, 3))
        points = np.vstack([points, rand_pts])

    center = points.mean(axis=0)
    points -= center
    dists = np.sum(points**2, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]


def _generate_icosahedral_cluster(n, rng):
    """Generate icosahedral shell cluster."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    points = []
    # Layer 1: center
    points.append(np.array([0.0, 0.0, 0.0]))
    # Layer 2: icosahedron (12 points)
    r2 = 1.0
    for v in verts:
        points.append(v * r2)
    
    # Add more shells if needed
    shells_radii = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    for r in shells_radii:
        if len(points) >= n:
            break
        # Approximate number of points on shell ~ 4*pi*r^2
        m = min(n - len(points), max(12, int(4 * np.pi * r**2)))
        if m <= 0:
            continue
        # Fibonacci sphere for this shell
        phi_angle = np.pi * (3.0 - np.sqrt(5.0))
        i = np.arange(m) + 0.5
        cos_theta = 1.0 - (i / m)
        sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)
        theta = i * phi_angle
        x = sin_theta * np.cos(theta)
        y = sin_theta * np.sin(theta)
        z = cos_theta
        shell_pts = np.column_stack((x, y, z)) * r
        points.extend(shell_pts)
        
    points = np.array(points)
    if len(points) < n:
        remaining = n - len(points)
        rand_pts = rng.normal(0, 2, (remaining, 3))
        points = np.vstack([points, rand_pts])

    center = points.mean(axis=0)
    points -= center
    dists = np.sum(points**2, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]


def _generate_shell_cluster(n, rng):
    """Generate multi-shell cluster with Fibonacci spheres."""
    points = [np.array([0.0, 0.0, 0.0])]
    shells = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    for r in shells:
        if len(points) >= n:
            break
        # Fibonacci sphere for this shell
        m = min(n - len(points), max(12, int(4 * np.pi * r**2)))
        if m <= 0:
            continue
        phi_angle = np.pi * (3.0 - np.sqrt(5.0))
        i = np.arange(m) + 0.5
        cos_theta = 1.0 - (i / m)
        sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)
        theta = i * phi_angle
        x = sin_theta * np.cos(theta)
        y = sin_theta * np.sin(theta)
        z = cos_theta
        shell_pts = np.column_stack((x, y, z)) * r
        points.extend(shell_pts)
        
    points = np.array(points)
    if len(points) < n:
        remaining = n - len(points)
        rand_pts = rng.normal(0, 2, (remaining, 3))
        points = np.vstack([points, rand_pts])

    center = points.mean(axis=0)
    points -= center
    dists = np.sum(points**2, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]


def _make_initial_config(n, rng, rho, method):
    """Create a reasonable initial configuration using one of the methods."""
    if method == 'fcc':
        coords = _generate_fcc_cluster(n, rng)
    elif method == 'hcp':
        coords = _generate_hcp_cluster(n, rng)
    elif method == 'ico':
        coords = _generate_icosahedral_cluster(n, rng)
    elif method == 'shell':
        coords = _generate_shell_cluster(n, rng)
    else: # random
        # Random sphere
        radius = (n / 8.0) ** (1.0 / 3.0) * 1.5
        coords = rng.normal(0, radius, (n, 3))
        
    # Small random perturbation to break symmetry and avoid singularities
    coords += rng.normal(0, 0.05, (n, 3))
    return coords


def _basin_hopping_step(coords, rho, displacement, rng):
    """One basin hopping step: displace + minimize."""
    displaced = coords + rng.normal(0, displacement, coords.shape)
    e, new_coords = _local_minimize(displaced, rho)
    return e, new_coords


def _solve_one_instance(instance, time_budget):
    """Solve one instance within time budget."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 6))

    start_time = time.time()
    deadline = start_time + time_budget

    # Generate diverse initial configurations
    rng = np.random.default_rng(hash((n, rho)) % (2**32))

    best_energy = float('inf')
    best_coords = None
    
    methods = ['fcc', 'hcp', 'ico', 'shell', 'random']
    
    # 1. Try local minimization from each structural prototype
    for m in methods:
        if time.time() > deadline - 10:
            break
        coords = _make_initial_config(n, rng, rho, m)
        e, coords = _local_minimize(coords, rho)
        if e < best_energy:
            best_energy = e
            best_coords = coords.copy()
            
    if best_coords is None:
        # Fallback
        coords = _make_initial_config(n, rng, rho, 'random')
        best_energy, best_coords = _local_minimize(coords, rho)

    # 2. Basin Hopping from the best found so far
    current_energy = best_energy
    current_coords = best_coords.copy()
    
    # Adaptive displacement
    base_disp = 0.5
    max_hopping_steps = 200 # Will be limited by time
    
    for step in range(max_hopping_steps):
        if time.time() > deadline - 15:
            break

        # Vary displacement slightly to help escape
        disp = base_disp * (1.0 + 0.2 * np.sin(step * 0.3))
        
        e_new, c_new = _basin_hopping_step(current_coords, rho, disp, rng)

        # Metropolis accept
        if e_new < current_energy:
            current_energy = e_new
            current_coords = c_new
            if e_new < best_energy:
                best_energy = e_new
                best_coords = c_new.copy()
        else:
            # Accept with probability (Metropolis criterion)
            delta = e_new - current_energy
            if delta < 5.0: # Cap delta to avoid underflow/overflow issues in exp
                # Temperature scales with energy magnitude
                temp = max(0.1, abs(current_energy) * 0.05)
                if rng.random() < np.exp(-delta / temp):
                    current_coords = c_new

        # Periodically try a fresh start if stuck or to explore new basins
        if step % 20 == 0 and step > 0:
            if time.time() > deadline - 10:
                break
            m = methods[step % len(methods)]
            coords = _make_initial_config(n, rng, rho, m)
            e, coords = _local_minimize(coords, rho)
            if e < best_energy:
                best_energy = e
                best_coords = coords.copy()
                current_coords = coords.copy()
                current_energy = e

    return best_coords


def construct(instance):
    """Find the lowest-energy arrangement of n atoms bound by Morse potentials."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 6))

    # Allocate time budget for this instance
    # We have 6 instances total, so roughly 45s each with safety margin
    # But we must check global deadline
    time_budget = min(45.0, _time_remaining() - 5.0)
    if time_budget < 5.0:
        # Not enough time, just do a quick local minimization from a good prototype
        rng = np.random.default_rng(n)
        coords = _make_initial_config(n, rng, rho, 'fcc')
        e, coords = _local_minimize(coords, rho)
        return coords

    result = _solve_one_instance(instance, time_budget)

    # Final validation: ensure minimum distance constraint
    if result is not None and len(result) >= 2:
        d = cdist(result, result)
        np.fill_diagonal(d, np.inf)
        if d.min() < 0.5:
            # If constraint violated, try to fix by spreading out
            # Simple fix: scale coordinates from center
            center = result.mean(axis=0)
            result = (result - center) * 1.2 + center
            # Check again
            d = cdist(result, result)
            np.fill_diagonal(d, np.inf)
            if d.min() < 0.5:
                # Last resort: use a fresh random start and minimize
                rng = np.random.default_rng(n + 1)
                coords = _make_initial_config(n, rng, rho, 'random')
                e, coords = _local_minimize(coords, rho)
                result = coords

    return result
