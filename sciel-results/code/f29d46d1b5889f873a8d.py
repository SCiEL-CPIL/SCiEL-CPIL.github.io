"""Morse cluster global minima search with multi-start relaxation and gradient polishing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline shared across all six construct() calls
_START_TIME = time.time()
_TOTAL_BUDGET = 320.0  # 10s safety margin before 330s kill

def _time_left():
    return _TOTAL_BUDGET - (time.time() - _START_TIME)

def _morse_energy(pos, rho):
    """Compute total Morse energy."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    e = np.exp(rho * (1.0 - d))
    return float(np.sum(e * (e - 2.0)))

def _build_close_packing(n, spacing=1.0):
    """Build a cluster using close-packing lattice points (FCC/HCP-like)."""
    points = []
    # FCC lattice generation: points where i+j+k is even, scaled so nearest neighbor = 1
    a = spacing * np.sqrt(2.0)
    
    max_coord = int(np.ceil(n / 3.0)) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k]) * a
                    if np.linalg.norm(p) <= spacing * np.sqrt(n) * 1.3:
                        points.append(p)
    if not points:
        return np.zeros((n, 3))
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _build_hcp_like(n, spacing=1.0):
    """Build HCP-like cluster with hexagonal layers."""
    a = spacing
    layer_spacing = a * np.sqrt(8.0 / 3.0)
    
    def hex_points(r):
        pts = []
        for q in range(-r, r + 1):
            for rr in range(-r, r + 1):
                if abs(q) + abs(rr) <= r and abs(q + rr) <= r:
                    x = a * (q + 0.5 * rr)
                    y = a * np.sqrt(3) / 2 * rr
                    pts.append([x, y])
        return np.array(pts) if pts else np.zeros((0, 2))
    
    points = []
    layer = 0
    while len(points) < n:
        if layer == 0:
            layer_pts = np.array([[0.0, 0.0]])
        else:
            needed = n - len(points)
            r_est = max(1, int(np.ceil(np.sqrt(needed / 3.0))))
            layer_pts = hex_points(r_est)
            if layer % 2 == 1:
                layer_pts[:, 0] += a / 2.0
        
        z = layer * layer_spacing
        for pt in layer_pts:
            if len(points) >= n:
                break
            points.append([pt[0], pt[1], z])
        layer += 1
    return np.array(points[:n])

def _build_icosahedral(n, spacing=1.0):
    """Build icosahedral cluster with shells."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico12 = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico12 /= np.linalg.norm(ico12[0])
    
    # Second shell: 30 points (icosidodecahedron)
    shell2 = []
    for perm in [(0, 1, phi), (1, phi, 0), (phi, 0, 1)]:
        for signs in [(1,1,1),(1,1,-1),(1,-1,1),(1,-1,-1),(-1,1,1),(-1,1,-1),(-1,-1,1),(-1,-1,-1)]:
            p = np.array([perm[0]*signs[0], perm[1]*signs[1], perm[2]*signs[2]])
            norm = np.linalg.norm(p)
            if norm > 0:
                shell2.append(p / norm)
    shell2 = np.array(shell2)
    
    # Third shell: approximate 92 points
    shell3 = []
    for perm in [(0, 2, 3*phi), (2, 3*phi, 0), (3*phi, 0, 2)]:
        for signs in [(1,1,1),(1,1,-1),(1,-1,1),(1,-1,-1),(-1,1,1),(-1,1,-1),(-1,-1,1),(-1,-1,-1)]:
            p = np.array([perm[0]*signs[0], perm[1]*signs[1], perm[2]*signs[2]])
            norm = np.linalg.norm(p)
            if norm > 0:
                shell3.append(p / norm)
    shell3 = np.array(shell3)
    
    points = [np.zeros(3)]
    for v in ico12:
        points.append(v * 1.0)
    for v in shell2:
        points.append(v * 1.7)
    for v in shell3:
        points.append(v * 2.5)
    
    # Add more if needed
    if len(points) < n:
        rng = np.random.default_rng(42)
        radius = 3.2
        attempts = 0
        while len(points) < n and attempts < 8000:
            p = rng.normal(0, 1, 3)
            p = p / np.linalg.norm(p) * radius
            min_d = min(np.linalg.norm(p - q) for q in points) if points else 10
            if min_d >= 0.55:
                points.append(p)
            attempts += 1
            if attempts % 400 == 0:
                radius += 0.15
    
    return np.array(points[:n])

def _build_fibonacci_sphere(n, radius=3.0):
    """Build points on a Fibonacci sphere."""
    if n <= 1:
        return np.zeros((n, 3))
    golden = (1 + np.sqrt(5)) / 2
    i = np.arange(n)
    phi = np.arccos(1 - 2 * (i + 0.5) / n)
    theta = 2 * np.pi * i / golden
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.stack([x, y, z], axis=1)

def _build_random_dense(n, seed=42):
    """Build random compact cluster via rejection sampling."""
    rng = np.random.default_rng(seed)
    radius = 1.0 * (n ** (1.0/3.0)) * 1.1
    points = [np.zeros(3)]
    max_attempts = 12000
    for _ in range(max_attempts):
        if len(points) >= n:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        min_d = min(np.linalg.norm(p - q) for q in points)
        if min_d >= 0.65:
            points.append(p)
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points[:n])

def _relax_single(pos, potential, param, timeout=15.0):
    """Relax configuration with timeout protection."""
    start = time.time()
    try:
        energy, relaxed = relax(pos, potential, param)
        elapsed = time.time() - start
        return energy, relaxed
    except Exception:
        return float('inf'), pos

def _gradient_refinement(pos, potential, param, max_iter=50, tol=1e-8):
    """Polish with gradient descent using L-BFGS from scipy."""
    from scipy.optimize import minimize
    
    def objective(x):
        pos_2d = x.reshape(-1, 3)
        e, _ = _relax_single(pos_2d, potential, param, timeout=5.0)
        return e
    
    def gradient(x):
        pos_2d = x.reshape(-1, 3)
        try:
            e, g = morse_energy_gradient(pos_2d, param)
            return g.flatten()
        except:
            # Finite difference fallback
            eps = 1e-6
            grad = np.zeros_like(x)
            for i in range(len(x)):
                xp = x.copy()
                xm = x.copy()
                xp[i] += eps
                xm[i] -= eps
                grad[i] = (objective(xp) - objective(xm)) / (2 * eps)
            return grad
    
    x0 = pos.flatten()
    try:
        result = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                         options={'maxiter': max_iter, 'gtol': tol, 'disp': False})
        if result.success or result.fun < objective(x0):
            return result.x.reshape(-1, 3), result.fun
    except:
        pass
    return pos, _morse_energy(pos, param)

def _ensure_min_distance(pos, min_d=0.5):
    """Ensure all pairs are at least min_d apart."""
    n = len(pos)
    pos = pos.copy()
    for _ in range(50):
        violated = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_d:
                    violated = True
                    diff = pos[i] - pos[j]
                    if np.linalg.norm(diff) < 1e-10:
                        diff = np.array([1.0, 0.0, 0.0])
                    diff /= np.linalg.norm(diff)
                    push = (min_d - d) * 0.5 + 0.005
                    pos[i] += diff * push
                    pos[j] -= diff * push
        if not violated:
            break
    return pos

def construct(instance):
    """Find low-energy Morse cluster configuration using multi-start relaxation and gradient polishing."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind", 10))
    
    tl = _time_left()
    if tl < 5.0:
        pos = _build_close_packing(size)
        return _ensure_min_distance(pos)
    
    # Generate diverse initial configurations
    configs = []
    try:
        configs.append(_build_close_packing(size))
    except: pass
    try:
        configs.append(_build_hcp_like(size))
    except: pass
    try:
        configs.append(_build_icosahedral(size))
    except: pass
    try:
        configs.append(_build_fibonacci_sphere(size, radius=2.5 * (size ** (1.0/3.0))))
    except: pass
    try:
        configs.append(_build_random_dense(size, seed=size))
    except: pass
    try:
        configs.append(_build_random_dense(size, seed=size + 13))
    except: pass
    try:
        configs.append(_build_random_dense(size, seed=size + 29))
    except: pass
    
    if not configs:
        pos = np.zeros((size, 3))
        return _ensure_min_distance(pos)
    
    # Relax all initial configs and track best
    best_energy = float('inf')
    best_pos = configs[0].copy()
    relaxed_configs = []
    
    for cfg in configs:
        tl = _time_left()
        if tl < 10.0:
            break
        e, relaxed = _relax_single(cfg, potential, parameter, timeout=min(12.0, tl * 0.4))
        if e < best_energy:
            best_energy = e
            best_pos = relaxed.copy()
        if np.isfinite(e):
            relaxed_configs.append((e, relaxed.copy()))
    
    # Sort by energy and keep top candidates
    relaxed_configs.sort(key=lambda x: x[0])
    top_configs = relaxed_configs[:min(5, len(relaxed_configs))]
    
    # Gradient polish the top candidates
    tl = _time_left()
    if tl > 15.0 and top_configs:
        for e, pos in top_configs:
            tl = _time_left()
            if tl < 10.0:
                break
            polished, e_polished = _gradient_refinement(pos, potential, parameter, max_iter=30)
            if e_polished < best_energy:
                best_energy = e_polished
                best_pos = polished.copy()
    
    # Final basin hopping from best configuration if time permits
    tl = _time_left()
    if tl > 20.0:
        rng = np.random.default_rng(int(time.time() * 1000) % 100000)
        step_size = 0.12
        n_bh_steps = int(min(tl * 0.6, 50.0) / 10.0)
        n_bh_steps = max(1, n_bh_steps)
        
        current_pos = best_pos.copy()
        current_energy = best_energy
        
        for step in range(n_bh_steps):
            tl = _time_left()
            if tl < 15.0:
                break
            
            # Adaptive step size
            if step % 5 == 0 and step > 0:
                step_size *= 0.85
            step_size = max(0.05, step_size)
            
            # Perturb
            noise = rng.normal(0, step_size, current_pos.shape)
            pos_perturbed = current_pos + noise
            
            # Relax
            e_new, pos_new = _relax_single(pos_perturbed, potential, parameter, timeout=min(10.0, tl * 0.5))
            
            # Metropolis acceptance
            if e_new < current_energy:
                current_pos = pos_new
                current_energy = e_new
                if e_new < best_energy:
                    best_energy = e_new
                    best_pos = pos_new.copy()
            else:
                delta = e_new - current_energy
                T = 0.1 * (1.0 + 0.5 * step / max(1, n_bh_steps))
                if rng.random() < np.exp(-delta / T):
                    current_pos = pos_new
                    current_energy = e_new
    
    # Final gradient polish
    tl = _time_left()
    if tl > 10.0:
        polished, e_polished = _gradient_refinement(best_pos, potential, parameter, max_iter=50)
        if e_polished < best_energy:
            best_pos = polished
            best_energy = e_polished
    
    # Final distance constraint
    return _ensure_min_distance(best_pos, min_d=0.5)
# EVOLVE-BLOCK-END
