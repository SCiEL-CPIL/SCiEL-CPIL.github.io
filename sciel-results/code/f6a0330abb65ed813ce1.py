"""Rewritten program using symmetry-constrained variational optimization for Dzugutov clusters."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = time.time()
TIME_LIMIT = 295.0  # Safety margin before 300s limit


def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)


def _dzugutov_energy(pos):
    """Compute Dzugutov potential energy."""
    n = len(pos)
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(pos[i] - pos[j])
            if r < 1e-10:
                continue
            if r < a:
                total += A * (r ** (-m) - B) * np.exp(c / (r - a))
            elif r < b:
                total += B * np.exp(d / (r - b))
    return total


def _dzugutov_gradient(pos):
    """Compute gradient of Dzugutov potential energy."""
    n = len(pos)
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    grad = np.zeros_like(pos)
    for i in range(n):
        for j in range(i + 1, n):
            rij = pos[i] - pos[j]
            r = np.linalg.norm(rij)
            if r < 1e-10:
                continue
            if r < a:
                # V = A(r^-m - B)exp(c/(r-a))
                # dV/dr = A[(-m*r^(-m-1))*exp(c/(r-a)) + (r^-m - B)*exp(c/(r-a))*(-c/(r-a)^2)]
                exp_term = np.exp(c / (r - a))
                dV_dr = A * (-m * r ** (-m - 1)) * exp_term + A * (r ** (-m) - B) * exp_term * (-c / (r - a) ** 2)
            elif r < b:
                # V = B*exp(d/(r-b))
                # dV/dr = B*exp(d/(r-b))*(-d/(r-b)^2)
                exp_term = np.exp(d / (r - b))
                dV_dr = B * exp_term * (-d / (r - b) ** 2)
            else:
                continue
            # grad_i += dV/dr * (pos_i - pos_j)/r, grad_j -= dV/dr * (pos_i - pos_j)/r
            factor = dV_dr / r
            grad[i] += factor * rij
            grad[j] -= factor * rij
    return grad


def _generate_macky_shells(n, scale=1.0):
    """Generate Mackay icosahedral cluster with exactly n atoms.
    
    Shells: center(1), icosahedron(12), M2(42), M3(120)
    Total complete: 1, 13, 55, 175
    """
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 vertices of regular icosahedron on unit sphere
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    points = [np.array([0.0, 0.0, 0.0])]  # center
    
    # Shell 1: icosahedron vertices
    if n > 1:
        shell1 = verts * scale
        points.extend(shell1)
    
    if len(points) >= n:
        return np.array(points[:n])
    
    # Shell 2: 42 atoms (30 edge midpoints + 12 face centers)
    if n > 13:
        # Find edges (distance ~ 1.0515 for unit icosahedron)
        edge_mids = []
        for i in range(12):
            for j in range(i + 1, 12):
                d = np.linalg.norm(verts[i] - verts[j])
                if 0.95 < d < 1.15:
                    mid = (verts[i] + verts[j]) / 2.0
                    mid = mid / np.linalg.norm(mid)
                    edge_mids.append(mid)
        
        # Find faces
        face_centers = []
        for i in range(12):
            for j in range(i + 1, 12):
                for k in range(j + 1, 12):
                    d1 = np.linalg.norm(verts[i] - verts[j])
                    d2 = np.linalg.norm(verts[j] - verts[k])
                    d3 = np.linalg.norm(verts[k] - verts[i])
                    if 0.95 < d1 < 1.15 and 0.95 < d2 < 1.15 and 0.95 < d3 < 1.15:
                        center = (verts[i] + verts[j] + verts[k]) / 3.0
                        center = center / np.linalg.norm(center)
                        face_centers.append(center)
        
        # Scale shell 2 to appropriate radius
        shell2_radius = scale * 1.7
        shell2 = np.array(edge_mids + face_centers) * shell2_radius
        points.extend(shell2)
    
    if len(points) >= n:
        return np.array(points[:n])
    
    # Shell 3: 120 atoms - use Fibonacci sphere distribution
    if n > 55:
        shell3_radius = scale * 2.3
        shell3 = []
        for i in range(120):
            y = 1 - 2 * (i + 0.5) / 120
            r = np.sqrt(max(0, 1 - y ** 2))
            theta = 2 * np.pi * i * 0.381966  # golden angle
            p = np.array([r * np.cos(theta), y, r * np.sin(theta)])
            shell3.append(p * shell3_radius)
        points.extend(shell3)
    
    if len(points) >= n:
        return np.array(points[:n])
    
    # Fill remaining with perturbed points
    rng = np.random.default_rng(42)
    while len(points) < n:
        idx = rng.integers(0, len(points))
        p = points[idx] + rng.normal(0, 0.05, 3)
        points.append(p)
    
    return np.array(points[:n])


def _generate_twisted_shells(n, scale=1.0, twist=0.1):
    """Generate icosahedral shells with slight twist for exploration."""
    base = _generate_macky_shells(n, scale=scale)
    
    # Apply small rotational perturbations to outer shells
    rng = np.random.default_rng(123)
    result = base.copy()
    
    # Determine shell boundaries
    center_idx = 0
    shell1_end = min(13, n)
    shell2_end = min(55, n)
    
    # Twist shell 2 slightly
    if n > 13:
        rot_angle = twist * rng.uniform(0.5, 1.5)
        axis = rng.normal(0, 1, 3)
        axis = axis / np.linalg.norm(axis)
        
        # Rotation matrix
        c, s = np.cos(rot_angle), np.sin(rot_angle)
        R = np.array([
            [c + axis[0]**2 * (1 - c), axis[0] * axis[1] * (1 - c) - axis[2] * s, axis[0] * axis[2] * (1 - c) + axis[1] * s],
            [axis[1] * axis[0] * (1 - c) + axis[2] * s, c + axis[1]**2 * (1 - c), axis[1] * axis[2] * (1 - c) - axis[0] * s],
            [axis[2] * axis[0] * (1 - c) - axis[1] * s, axis[2] * axis[1] * (1 - c) + axis[0] * s, c + axis[2]**2 * (1 - c)]
        ])
        
        for i in range(shell1_end, shell2_end):
            if i < len(result):
                result[i] = R @ result[i]
    
    # Twist shell 3 if present
    if n > 55:
        rot_angle = twist * rng.uniform(0.3, 1.0)
        axis = rng.normal(0, 1, 3)
        axis = axis / np.linalg.norm(axis)
        
        c, s = np.cos(rot_angle), np.sin(rot_angle)
        R = np.array([
            [c + axis[0]**2 * (1 - c), axis[0] * axis[1] * (1 - c) - axis[2] * s, axis[0] * axis[2] * (1 - c) + axis[1] * s],
            [axis[1] * axis[0] * (1 - c) + axis[2] * s, c + axis[1]**2 * (1 - c), axis[1] * axis[2] * (1 - c) - axis[0] * s],
            [axis[2] * axis[0] * (1 - c) - axis[1] * s, axis[2] * axis[1] * (1 - c) + axis[0] * s, c + axis[2]**2 * (1 - c)]
        ])
        
        for i in range(shell2_end, len(result)):
            result[i] = R @ result[i]
    
    return result


def _generate_bcc_like(n, scale=1.0):
    """Generate BCC-like cluster as alternative structure."""
    rng = np.random.default_rng(999)
    points = []
    
    a = scale * 0.45
    max_coord = int(scale * 2.5)
    
    # BCC lattice: corners + body centers
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            for z in range(-max_coord, max_coord + 1):
                if len(points) >= n:
                    break
                pos = np.array([x, y, z]) * a
                if np.linalg.norm(pos) <= scale * 1.3:
                    points.append(pos)
                # Body center
                if len(points) < n:
                    pos_bc = np.array([x + 0.5, y + 0.5, z + 0.5]) * a
                    if np.linalg.norm(pos_bc) <= scale * 1.3:
                        points.append(pos_bc)
    
    # Fill remaining with random points
    while len(points) < n:
        p = rng.uniform(-scale, scale, 3)
        if np.linalg.norm(p) <= scale:
            points.append(p)
    
    return np.array(points[:n])


def _optimize_with_gradient(pos, max_iter=200):
    """Optimize positions using gradient-based method."""
    def objective(x):
        p = x.reshape(-1, 3)
        # Add penalty for atoms too close
        min_dist_violation = 0.0
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                d = np.linalg.norm(p[i] - p[j])
                if d < 0.5:
                    min_dist_violation += (0.5 - d) ** 2 * 100
        return _dzugutov_energy(p) + min_dist_violation
    
    def gradient(x):
        p = x.reshape(-1, 3)
        g = _dzugutov_gradient(p)
        # Add gradient for min distance penalty
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                d = np.linalg.norm(p[i] - p[j])
                if d < 0.5 and d > 1e-10:
                    force = 2 * (0.5 - d) * 100 / d
                    rij = p[i] - p[j]
                    g[i] += force * rij
                    g[j] -= force * rij
        return g.flatten()
    
    try:
        result = minimize(
            objective, pos.flatten(), jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if result.success or result.fun < objective(pos.flatten()):
            return result.fun, result.x.reshape(-1, 3)
    except Exception:
        pass
    
    return _dzugutov_energy(pos), pos


def _relax_with_library(pos, potential, parameter):
    """Use the provided relax function."""
    try:
        energy, positions = relax(pos, potential, parameter)
        return energy, positions
    except Exception:
        return _dzugutov_energy(pos), pos


def construct(instance):
    """Construct low-energy Dzugutov cluster using symmetry-constrained optimization."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    best_energy = np.inf
    best_positions = None
    
    def _try_candidate(points):
        nonlocal best_energy, best_positions
        if points is None or len(points) != size:
            return
        if _time_remaining() < 30:
            return
        
        # First try library relaxation
        try:
            energy, positions = relax(points, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
                return
        except Exception:
            pass
        
        # Then try our own gradient optimization
        if _time_remaining() < 20:
            return
        energy, positions = _optimize_with_gradient(points, max_iter=100)
        if energy < best_energy:
            best_energy = energy
            best_positions = positions.copy()
    
    # Base scale for this cluster size
    base_scale = size ** (1.0 / 3.0) * 0.38
    
    # Strategy 1: Mackay icosahedron
    ico_points = _generate_macky_shells(size, scale=base_scale)
    _try_candidate(ico_points)
    
    if _time_remaining() < 60:
        return best_positions
    
    # Strategy 2: Twisted Mackay
    twisted_points = _generate_twisted_shells(size, scale=base_scale, twist=0.08)
    _try_candidate(twisted_points)
    
    if _time_remaining() < 60:
        return best_positions
    
    # Strategy 3: Different scale Mackay
    ico_points2 = _generate_macky_shells(size, scale=base_scale * 0.92)
    _try_candidate(ico_points2)
    
    if _time_remaining() < 60:
        return best_positions
    
    # Strategy 4: BCC-like
    bcc_points = _generate_bcc_like(size, scale=base_scale * 1.05)
    _try_candidate(bcc_points)
    
    if _time_remaining() < 60:
        return best_positions
    
    # Strategy 5: Perturbed Mackay with multiple restarts
    rng = np.random.default_rng(size)
    for _ in range(3):
        if _time_remaining() < 40:
            break
        perturbed = ico_points + rng.normal(0, 0.1, ico_points.shape)
        _try_candidate(perturbed)
    
    if _time_remaining() < 50:
        return best_positions
    
    # Strategy 6: Larger scale Mackay
    ico_points3 = _generate_macky_shells(size, scale=base_scale * 1.08)
    _try_candidate(ico_points3)
    
    if _time_remaining() < 50:
        return best_positions
    
    # Strategy 7: Another twisted variant
    twisted_points2 = _generate_twisted_shells(size, scale=base_scale * 0.95, twist=0.12)
    _try_candidate(twisted_points2)
    
    if _time_remaining() < 40:
        return best_positions
    
    # Strategy 8: Smaller scale
    ico_points4 = _generate_macky_shells(size, scale=base_scale * 0.85)
    _try_candidate(ico_points4)
    
    if _time_remaining() < 40:
        return best_positions
    
    # Strategy 9: Random compact with minimum distance
    rng2 = np.random.default_rng(size + 100)
    radius = base_scale * 1.2
    points = []
    min_dist = 0.6
    attempts = 0
    max_attempts = size * 500
    while len(points) < size and attempts < max_attempts:
        p = rng2.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    
    if len(points) < size:
        min_dist = 0.5
        attempts = 0
        while len(points) < size and attempts < max_attempts:
            p = rng2.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= min_dist for q in points):
                    points.append(p)
            attempts += 1
    
    while len(points) < size:
        p = rng2.uniform(-radius * 0.5, radius * 0.5, 3)
        points.append(p)
    
    _try_candidate(np.array(points[:size]))
    
    if _time_remaining() < 30:
        return best_positions
    
    # Strategy 10: Final perturbation of best found
    if best_positions is not None:
        for _ in range(2):
            if _time_remaining() < 20:
                break
            perturbed = best_positions + rng.normal(0, 0.08, best_positions.shape)
            _try_candidate(perturbed)
    
    return best_positions
# EVOLVE-BLOCK-END
