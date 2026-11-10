"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

DEADLINE = 140.0  # wall-clock safety margin (evaluator kills at 150 s)


def _normalize(pts):
    n = np.linalg.norm(pts, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return pts / n


def _covering_radius_softmax(pts, alpha=50.0):
    """Differentiable soft-max approximation of covering radius using hull facets.
    
    Uses log-sum-exp of facet circumradii to create a smooth objective
    that focuses on the largest facet while remaining differentiable.
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 1.0, np.zeros(N * 3)
    
    radii = []
    facet_indices = []
    for f in hull.simplices:
        tri = pts[f]
        a, b, c = tri[0], tri[1], tri[2]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        cc = n / nn
        cosv = np.clip(np.dot(cc, a), -1.0, 1.0)
        r = np.arccos(cosv)
        radii.append(r)
        facet_indices.append(f)
    
    if not radii:
        return 1.0, np.zeros(N * 3)
    
    radii = np.array(radii)
    
    # Log-sum-exp for numerical stability
    max_r = radii.max()
    if max_r < 1e-12:
        return 0.0, np.zeros(N * 3)
    
    # Soft-max: emphasizes the largest facets
    shifted = radii - max_r
    weights = np.exp(alpha * shifted)
    weights /= weights.sum()
    
    # Weighted sum approximates the max
    soft_max_val = max_r + np.log(weights.sum()) / alpha
    
    # Compute gradient via finite differences or analytical
    # For now, return just the value; gradient computed externally
    return soft_max_val, np.zeros(N * 3)


def _covering_radius_hull(pts):
    """Exact covering radius (radians) via convex hull facets."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi
    best = 0.0
    for f in hull.simplices:
        tri = pts[f]
        a, b, c = tri[0], tri[1], tri[2]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        cc = n / nn
        cosv = np.clip(np.dot(cc, a), -1.0, 1.0)
        r = np.arccos(cosv)
        if r > best:
            best = r
    if best == 0.0:
        G = np.clip(pts @ pts.T, -1.0, 1.0)
        np.fill_diagonal(G, -1.0)
        best = np.arccos(G.min())
    return best


def _repulsive_force_optimize(pts, time_budget):
    """Repulsive force dynamics: points repel each other and are attracted to 
    the sphere surface. The covering radius is minimized by balancing these forces.
    
    Different structural approach from gradient-based methods.
    """
    pts = _normalize(pts).copy()
    start = time.time()
    
    # Parameters
    dt = 0.005
    repulsion_strength = 0.5
    attraction_strength = 0.1
    cooling_rate = 0.995
    temperature = 1.0
    
    best_pts = pts.copy()
    best_r = _covering_radius_hull(pts)
    
    iteration = 0
    while time.time() - start < time_budget:
        iteration += 1
        
        # Compute pairwise distances and angles
        G = pts @ pts.T
        G = np.clip(G, -1.0, 1.0)
        np.fill_diagonal(G, 0.0)
        
        # Angular distances
        angles = np.arccos(G)
        
        # Repulsive force: stronger for closer points
        # Use inverse square law in angular space
        mask = angles > 1e-6
        repulsive = np.zeros_like(angles)
        repulsive[mask] = 1.0 / (angles[mask] ** 2)
        np.fill_diagonal(repulsive, 0.0)
        
        # Attraction to maintain uniform distribution
        # Points should be roughly equally spaced
        target_angle = np.pi * np.sqrt(3.0 / (4.0 * N))  # Approximate optimal spacing
        
        # Net force on each point
        forces = np.zeros_like(pts)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                # Repulsion from close points
                if angles[i, j] < target_angle:
                    force_dir = (pts[i] - pts[j]) / np.linalg.norm(pts[i] - pts[j] + 1e-10)
                    forces[i] += repulsion_strength * repulsive[i, j] * force_dir
                else:
                    # Mild attraction to spread out
                    force_dir = (pts[i] - pts[j]) / np.linalg.norm(pts[i] - pts[j] + 1e-10)
                    forces[i] -= 0.1 * repulsion_strength * force_dir
            
            # Keep on sphere: project force tangent to sphere
            radial = np.dot(forces[i], pts[i]) * pts[i]
            forces[i] -= radial
        
        # Update positions with cooling
        for i in range(N):
            pts[i] += dt * temperature * forces[i]
            pts[i] = _normalize(pts[i].reshape(1, -1)).flatten()
        
        # Cool down
        temperature *= cooling_rate
        
        # Evaluate and keep best
        r = _covering_radius_hull(pts)
        if r < best_r:
            best_r = r
            best_pts = pts.copy()
        
        # Check for convergence
        if temperature < 0.01:
            break
    
    return best_pts, best_r


def _covering_radius_fast(pts, n_grid=150):
    """Fast approximate covering radius using a grid on the sphere."""
    pts = _normalize(pts)
    theta = np.linspace(0, np.pi, n_grid)
    phi = np.linspace(0, 2 * np.pi, 2 * n_grid)
    theta, phi = np.meshgrid(theta, phi)
    grid_x = np.sin(theta) * np.cos(phi)
    grid_y = np.sin(theta) * np.sin(phi)
    grid_z = np.cos(theta)
    grid = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
    grid = _normalize(grid)
    
    G = grid @ pts.T
    G = np.clip(G, -1.0, 1.0)
    max_cos = G.max(axis=1)
    angles = np.arccos(max_cos)
    return angles.max()


def _build_hss50():
    """Build a configuration based on truncated icosahedron structure."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    
    ico = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1],
    ], dtype=float)
    ico = _normalize(ico)
    
    D = 1.0 - ico @ ico.T
    min_d = D.min()
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(D[i, j] - min_d) < 1e-8:
                edges.append((i, j))
    
    verts = []
    for (i, j) in edges:
        verts.append(ico[i] * (2.0 / 3.0) + ico[j] * (1.0 / 3.0))
        verts.append(ico[i] * (1.0 / 3.0) + ico[j] * (2.0 / 3.0))
    verts = np.array(verts)
    verts = _normalize(verts)
    
    # Remove 10 vertices to get 50
    vmap = []
    for (i, j) in edges:
        vmap.append((i, j))
        vmap.append((i, j))
    
    G = ico @ ico.T
    np.fill_diagonal(G, -2.0)
    pair_i, pair_j = np.unravel_index(np.argmin(G), G.shape)
    
    remove = set()
    for idx, (i, j) in enumerate(vmap):
        if i == pair_i or j == pair_i or i == pair_j or j == pair_j:
            remove.add(idx)
    
    keep = [k for k in range(len(verts)) if k not in remove]
    out = verts[keep]
    if len(out) != 50:
        # If we don't have exactly 50, take first 50
        out = out[:50]
    return _normalize(out)


def _fibonacci_spiral():
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _softmax_lbgfs_optimize(pts, time_budget):
    """L-BFGS-B optimization using log-sum-exp soft-max surrogate for covering radius.
    
    The soft-max focuses optimization on the largest facet while remaining
    differentiable, allowing efficient gradient-based optimization.
    """
    pts = _normalize(pts).copy()
    start = time.time()
    alpha = 50.0  # Soft-max sharpness
    
    def objective_and_grad(x):
        if time.time() - start > time_budget:
            return np.inf, np.zeros_like(x)
        
        p = x.reshape(N, 3)
        p = _normalize(p)
        
        try:
            hull = ConvexHull(p)
        except Exception:
            return 1.0, np.zeros_like(x)
        
        radii = []
        facet_normals = []
        facet_vertices = []
        for f in hull.simplices:
            tri = p[f]
            a, b, c = tri[0], tri[1], tri[2]
            n = np.cross(b - a, c - a)
            nn = np.linalg.norm(n)
            if nn < 1e-12:
                continue
            cc = n / nn
            cosv = np.clip(np.dot(cc, a), -1.0, 1.0)
            r = np.arccos(cosv)
            radii.append(r)
            facet_normals.append(cc)
            facet_vertices.append(f)
        
        if not radii:
            return 1.0, np.zeros_like(x)
        
        radii = np.array(radii)
        max_r = radii.max()
        
        if max_r < 1e-12:
            return 0.0, np.zeros_like(x)
        
        # Log-sum-exp soft-max
        shifted = radii - max_r
        exp_vals = np.exp(alpha * shifted)
        weights = exp_vals / exp_vals.sum()
        
        soft_max_val = max_r + np.log(exp_vals.sum()) / alpha
        
        # Approximate gradient: weight each facet's gradient by its soft-max weight
        grad = np.zeros_like(x)
        for k, (r, cc, f) in enumerate(zip(radii, facet_normals, facet_vertices)):
            w = weights[k]
            if w < 1e-8:
                continue
            # Gradient of arccos(dot(cc, a)) w.r.t. a is -cc/sqrt(1-dot^2)
            # But cc depends on the triangle vertices
            # Simplified: push vertices away from the facet normal
            a, b, c = p[f[0]], p[f[1]], p[f[2]]
            # Move each vertex in direction that increases the angle
            # d(arccos(v))/dv = -1/sqrt(1-v^2)
            v_a = np.clip(np.dot(cc, a), -1.0 + 1e-10, 1.0 - 1e-10)
            v_b = np.clip(np.dot(cc, b), -1.0 + 1e-10, 1.0 - 1e-10)
            v_c = np.clip(np.dot(cc, c), -1.0 + 1e-10, 1.0 - 1e-10)
            
            scale_a = -1.0 / np.sqrt(max(1.0 - v_a**2, 1e-10))
            scale_b = -1.0 / np.sqrt(max(1.0 - v_b**2, 1e-10))
            scale_c = -1.0 / np.sqrt(max(1.0 - v_c**2, 1e-10))
            
            grad[f[0]*3:(f[0]*3+3)] += w * scale_a * cc
            grad[f[1]*3:(f[1]*3+3)] += w * scale_b * cc
            grad[f[2]*3:(f[2]*3+3)] += w * scale_c * cc
        
        # Project gradient to tangent space of sphere
        for i in range(N):
            idx = slice(i*3, i*3+3)
            p_i = p[i]
            radial = np.dot(grad[idx], p_i) * p_i
            grad[idx] -= radial
        
        return soft_max_val, grad
    
    x0 = pts.ravel()
    f0, g0 = objective_and_grad(x0)
    
    try:
        result = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True,
                         options={'maxiter': 1000, 'gtol': 1e-6, 'maxls': 100,
                                  'ftol': 1e-8, 'maxfun': 2000})
        
        if result.fun < f0:
            best_pts = _normalize(result.x.reshape(N, 3))
        else:
            best_pts = pts
    except Exception:
        best_pts = pts
    
    return best_pts


def _local_polish(pts, time_budget):
    """Coordinate-wise local optimization with adaptive steps."""
    pts = _normalize(pts).copy()
    best_r = _covering_radius_hull(pts)
    start = time.time()
    
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    current_step = 0
    
    while time.time() - start < time_budget:
        improved = False
        for i in range(N):
            for axis in range(3):
                for sgn in (1.0, -1.0):
                    for ss in step_sizes[current_step:current_step+2]:
                        cand = pts.copy()
                        cand[i, axis] += sgn * ss
                        norm = np.linalg.norm(cand[i])
                        if norm < 1e-10:
                            continue
                        cand[i] /= norm
                        r = _covering_radius_hull(cand)
                        if r < best_r - 1e-12:
                            pts = cand
                            best_r = r
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            current_step += 1
            if current_step >= len(step_sizes):
                break
    
    return pts, best_r


def construct():
    t0 = time.time()
    
    # Generate multiple initial configurations
    configs = []
    
    # 1. Hardin-Sloane-Smith based
    try:
        hss = _build_hss50()
        if len(hss) == 50 and np.all(np.isfinite(hss)):
            configs.append(hss)
    except Exception:
        pass
    
    # 2. Fibonacci spiral
    fib = _fibonacci_spiral()
    configs.append(fib)
    
    # 3. Random configurations with icosahedral seed
    rng = np.random.RandomState(42)
    for _ in range(2):
        ico = np.array([
            [0, 1, (1+np.sqrt(5))/2], [0, -1, (1+np.sqrt(5))/2],
            [0, 1, -(1+np.sqrt(5))/2], [0, -1, -(1+np.sqrt(5))/2],
            [1, (1+np.sqrt(5))/2, 0], [-1, (1+np.sqrt(5))/2, 0],
            [1, -(1+np.sqrt(5))/2, 0], [-1, -(1+np.sqrt(5))/2, 0],
            [(1+np.sqrt(5))/2, 0, 1], [-(1+np.sqrt(5))/2, 0, 1],
            [(1+np.sqrt(5))/2, 0, -1], [-(1+np.sqrt(5))/2, 0, -1],
        ], dtype=float)
        ico = _normalize(ico)
        
        rand_pts = rng.randn(38, 3)
        rand_pts = rand_pts / np.linalg.norm(rand_pts, axis=1, keepdims=True)
        
        all_pts = np.vstack([ico, rand_pts])
        configs.append(all_pts)
    
    # Evaluate all configs and pick the best
    best_pts = None
    best_r = np.inf
    
    for cfg in configs:
        cfg = _normalize(cfg)
        r = _covering_radius_fast(cfg)
        if r < best_r:
            best_r = r
            best_pts = cfg
    
    if best_pts is None:
        best_pts = fib
        best_r = _covering_radius_fast(best_pts)
    
    # Soft-max L-BFGS-B optimization (primary strategy - different from BFGS)
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 40.0:
        try:
            softmax_pts = _softmax_lbgfs_optimize(best_pts, min(remaining - 40.0, 50.0))
            softmax_r = _covering_radius_hull(softmax_pts)
            if softmax_r < best_r:
                best_pts, best_r = softmax_pts, softmax_r
        except Exception:
            pass
    
    # Repulsive force dynamics (secondary strategy - different optimization family)
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 30.0:
        try:
            repulsive_pts, repulsive_r = _repulsive_force_optimize(best_pts, min(remaining - 30.0, 25.0))
            if repulsive_r < best_r:
                best_pts, best_r = repulsive_pts, repulsive_r
        except Exception:
            pass
    
    # Local polish
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 15.0:
        try:
            polish_pts, polish_r = _local_polish(best_pts, min(remaining - 15.0, 15.0))
            if polish_r < best_r:
                best_pts, best_r = polish_pts, polish_r
        except Exception:
            pass
    
    # Ensure valid output
    best_pts = _normalize(best_pts)
    if not np.all(np.isfinite(best_pts)) or len(best_pts) != N:
        best_pts = fib
    
    return best_pts
# EVOLVE-BLOCK-END
