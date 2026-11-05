"""Search for global minima of Morse clusters using diverse initializations and L-BFGS-B refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _time_left(deadline):
    return deadline - time.time()


def _morse_energy_fast(pos, rho):
    """Compute Morse energy for a set of positions (vectorized)."""
    n = pos.shape[0]
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    iu = np.triu_indices(n, k=1)
    r = dist[iu]
    exp_term = np.exp(rho * (1.0 - r))
    return float(np.sum(exp_term * (exp_term - 2.0)))


def _morse_gradient_fast(pos, rho):
    """Compute Morse gradient for a set of positions (vectorized)."""
    n = pos.shape[0]
    grad = np.zeros_like(pos)
    if n < 2:
        return grad
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    iu = np.triu_indices(n, k=1)
    r = dist[iu]
    exp_term = np.exp(rho * (1.0 - r))
    # dE/dr = rho * exp_term * (2*exp_term - 1)
    dE_dr = rho * exp_term * (2.0 * exp_term - 1.0)
    # grad_i = -dE_dr * (pos_i - pos_j) / r  for each pair
    # Vectorized: for each pair (i,j), contribution to grad[i] is -dE_dr * diff[i,j] / r
    # and to grad[j] is +dE_dr * diff[i,j] / r
    # diff[i,j] = pos[i] - pos[j]
    diff_iu = diff[iu]  # shape (n_pairs, 3)
    r_safe = np.where(r < 1e-10, 1.0, r)
    factor = dE_dr / r_safe  # shape (n_pairs,)
    # Contribution to each atom
    for k in range(len(iu[0])):
        i, j = iu[0][k], iu[1][k]
        grad[i] -= factor[k] * diff_iu[k]
        grad[j] += factor[k] * diff_iu[k]
    return grad


def _make_fcc_start(n, spacing=1.0):
    """Generate FCC-like configuration."""
    points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                for offset in [
                    np.array([0, 0, 0]),
                    np.array([0.5, 0.5, 0]),
                    np.array([0.5, 0, 0.5]),
                    np.array([0, 0.5, 0.5]),
                ]:
                    p = (np.array([i, j, k]) + offset) * spacing
                    if np.linalg.norm(p) <= 5.0 * spacing:
                        points.append(p)
    unique_points = []
    for p in points:
        if not any(np.linalg.norm(p - q) < 0.1 * spacing for q in unique_points):
            unique_points.append(p)
    unique_points.sort(key=lambda p: np.linalg.norm(p))
    return np.array(unique_points[:n])


def _make_bcc_start(n, spacing=1.0):
    """Generate BCC-like configuration."""
    points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                p = np.array([i, j, k]) * spacing
                if np.linalg.norm(p) <= 5.0 * spacing:
                    points.append(p)
                p2 = (np.array([i + 0.5, j + 0.5, k + 0.5])) * spacing
                if np.linalg.norm(p2) <= 5.0 * spacing:
                    points.append(p2)
    unique_points = []
    for p in points:
        if not any(np.linalg.norm(p - q) < 0.1 * spacing for q in unique_points):
            unique_points.append(p)
    unique_points.sort(key=lambda p: np.linalg.norm(p))
    return np.array(unique_points[:n])


def _make_hcp_start(n, spacing=1.0):
    """Generate HCP-like configuration."""
    points = []
    a = spacing
    c = spacing * 2 * math.sqrt(2.0 / 3.0)
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-2, 3):
                # ABAB stacking
                if k % 2 == 0:
                    offset = np.array([0, 0, 0])
                else:
                    offset = np.array([a / 2, a * math.sqrt(3) / 6, 0])
                p = np.array([i * a, j * a * math.sqrt(3), k * c]) + offset
                if np.linalg.norm(p) <= 5.0 * spacing:
                    points.append(p)
    unique_points = []
    for p in points:
        if not any(np.linalg.norm(p - q) < 0.1 * spacing for q in unique_points):
            unique_points.append(p)
    unique_points.sort(key=lambda p: np.linalg.norm(p))
    return np.array(unique_points[:n])


def _make_icosahedral_shell(n, radius=1.0):
    """Generate icosahedral shell configuration."""
    # Icosahedron vertices
    t = (1.0 + math.sqrt(5.0)) / 2.0
    verts = [
        np.array([1, 1, 1]), np.array([1, 1, -1]), np.array([1, -1, 1]), np.array([1, -1, -1]),
        np.array([-1, 1, 1]), np.array([-1, 1, -1]), np.array([-1, -1, 1]), np.array([-1, -1, -1]),
        np.array([0, t, 1/t]), np.array([0, t, -1/t]), np.array([0, -t, 1/t]), np.array([0, -t, -1/t]),
        np.array([1/t, 0, t]), np.array([1/t, 0, -t]), np.array([-1/t, 0, t]), np.array([-1/t, 0, -t]),
        np.array([t, 1/t, 0]), np.array([t, -1/t, 0]), np.array([-t, 1/t, 0]), np.array([-t, -1/t, 0]),
    ]
    verts = [v / np.linalg.norm(v) * radius for v in verts]
    
    points = [np.array([0.0, 0.0, 0.0])]  # center
    points.extend(verts)
    
    # Add more points on the surface if needed
    rng = np.random.default_rng(42)
    while len(points) < n:
        # Random point on sphere
        theta = rng.uniform(0, 2 * math.pi)
        phi = rng.uniform(0, math.pi)
        p = radius * np.array([
            math.sin(phi) * math.cos(theta),
            math.sin(phi) * math.sin(theta),
            math.cos(phi)
        ])
        # Check minimum distance
        if min(np.linalg.norm(p - q) for q in points) > 0.5:
            points.append(p)
        else:
            break
    
    # If still not enough, add random points
    while len(points) < n:
        theta = rng.uniform(0, 2 * math.pi)
        phi = rng.uniform(0, math.pi)
        p = radius * np.array([
            math.sin(phi) * math.cos(theta),
            math.sin(phi) * math.sin(theta),
            math.cos(phi)
        ])
        points.append(p)
    
    return np.array(points[:n])


def _make_random_compact(n, spacing=1.0, seed=42):
    """Generate random compact configuration with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.9
    points = []
    attempts = 0
    max_attempts = 20000
    
    while len(points) < n and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            min_dist = min((np.linalg.norm(p - q) for q in points), default=np.inf)
            if min_dist >= 0.55 * spacing:
                points.append(p)
    
    # If we couldn't place all points, fill remaining with random positions
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points)


def _make_jittered_crystal(n, crystal_type, spacing=1.0, jitter=0.1, seed=42):
    """Generate crystal with random jitter."""
    rng = np.random.default_rng(seed)
    if crystal_type == "fcc":
        base = _make_fcc_start(n, spacing)
    elif crystal_type == "bcc":
        base = _make_bcc_start(n, spacing)
    elif crystal_type == "hcp":
        base = _make_hcp_start(n, spacing)
    else:
        base = _make_fcc_start(n, spacing)
    
    if len(base) < n:
        radius = spacing * n ** (1.0 / 3.0) * 0.9
        while len(base) < n:
            p = rng.uniform(-radius, radius, 3)
            base = np.vstack([base, p])
    
    jittered = base + rng.normal(0, jitter, base.shape)
    return jittered


def _lbfgs_refine(pos, rho, maxiter=500, gtol=1e-8):
    """Refine using L-BFGS-B with gradient."""
    try:
        def objective(x):
            pos_2d = x.reshape(-1, 3)
            return _morse_energy_fast(pos_2d, rho)
        
        def gradient(x):
            pos_2d = x.reshape(-1, 3)
            return _morse_gradient_fast(pos_2d, rho).flatten()
        
        result = minimize(
            objective,
            pos.flatten(),
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': gtol}
        )
        
        if result.success or result.fun < float('inf'):
            return result.x.reshape(-1, 3), float(result.fun)
        else:
            return pos, _morse_energy_fast(pos, rho)
    except Exception:
        return pos, _morse_energy_fast(pos, rho)


def _check_min_distance(pos, min_dist=0.5):
    """Check if all pairs are at least min_dist apart."""
    n = len(pos)
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(pos[i] - pos[j]) < min_dist:
                return False
    return True


def _enforce_min_distance(pos, min_dist=0.5, max_iter=100):
    """Nudge atoms to satisfy minimum distance constraint."""
    pos = pos.copy()
    for _ in range(max_iter):
        violated = False
        n = len(pos)
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist and d > 1e-10:
                    direction = (pos[i] - pos[j]) / d
                    push = (min_dist - d) / 2.0 + 0.01
                    pos[i] += direction * push
                    pos[j] -= direction * push
                    violated = True
        if not violated:
            break
    return pos


def construct(instance):
    """Find low-energy Morse cluster configuration."""
    start_time = time.time()
    deadline = start_time + 280.0  # 50s safety margin
    
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 6.0))
    
    best_energy = float('inf')
    best_positions = None
    
    # Generate diverse initial configurations
    initializations = []
    
    # Crystalline starts
    for crystal_type in ["fcc", "bcc", "hcp"]:
        try:
            if crystal_type == "fcc":
                init = _make_fcc_start(size)
            elif crystal_type == "bcc":
                init = _make_bcc_start(size)
            else:
                init = _make_hcp_start(size)
            if len(init) == size:
                initializations.append(init)
        except Exception:
            pass
    
    # Icosahedral
    try:
        init = _make_icosahedral_shell(size)
        if len(init) == size:
            initializations.append(init)
    except Exception:
        pass
    
    # Jittered crystals
    for seed in [42, 123, 456, 789, 1011]:
        for crystal_type in ["fcc", "bcc"]:
            try:
                init = _make_jittered_crystal(size, crystal_type, jitter=0.12, seed=seed)
                if len(init) == size:
                    initializations.append(init)
            except Exception:
                pass
    
    # Random compact
    for seed in [42, 123, 456, 789, 1011, 1013, 1019]:
        try:
            init = _make_random_compact(size, seed=seed)
            if len(init) == size:
                initializations.append(init)
        except Exception:
            pass
    
    # Relax each initialization
    for idx, init in enumerate(initializations):
        if time.time() > deadline - 15:
            break
        
        try:
            # First try the built-in relax
            energy, positions = relax(init.copy(), potential, rho)
            
            if energy < best_energy:
                if _check_min_distance(positions, 0.5):
                    best_energy = energy
                    best_positions = positions.copy()
                else:
                    positions = _enforce_min_distance(positions, 0.5)
                    energy, positions = relax(positions, potential, rho)
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = positions.copy()
                        
        except Exception:
            pass
    
    # Try L-BFGS-B on the best found so far
    if best_positions is not None and time.time() < deadline - 25:
        try:
            relaxed_pos, relaxed_energy = _lbfgs_refine(best_positions, rho, maxiter=300, gtol=1e-8)
            if relaxed_energy < best_energy:
                if _check_min_distance(relaxed_pos, 0.5):
                    best_energy = relaxed_energy
                    best_positions = relaxed_pos.copy()
        except Exception:
            pass
    
    # Try L-BFGS-B on a few other initializations
    if time.time() < deadline - 30:
        for init in initializations[:5]:
            if time.time() > deadline - 25:
                break
            try:
                relaxed_pos, relaxed_energy = _lbfgs_refine(init, rho, maxiter=200, gtol=1e-7)
                if relaxed_energy < best_energy:
                    if _check_min_distance(relaxed_pos, 0.5):
                        best_energy = relaxed_energy
                        best_positions = relaxed_pos.copy()
            except Exception:
                pass
    
    # If no valid configuration found, generate one
    if best_positions is None:
        init = _make_random_compact(size, seed=42)
        init = _enforce_min_distance(init, 0.5)
        best_positions = init
        best_energy = _morse_energy_fast(init, rho)
    
    # Final check and enforcement
    best_positions = _enforce_min_distance(best_positions, 0.5)
    
    return best_positions
# EVOLVE-BLOCK-END
