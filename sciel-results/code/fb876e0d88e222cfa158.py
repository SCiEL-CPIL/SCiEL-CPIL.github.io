"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log(|xi-xj|)"""
    n = pts.shape[0]
    dots = pts @ pts.T
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, 1e-15)
    dist = np.sqrt(d2)
    iu = np.triu_indices(n, k=1)
    log_d = np.log(dist[iu])
    return -math.fsum(log_d)

def _gradient_log_energy(pts):
    """Tangent-space gradient of logarithmic energy, fully vectorized."""
    n = pts.shape[0]
    dots = pts @ pts.T
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, 1e-15)
    
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    
    sum_inv_d2 = inv_d2.sum(axis=1)
    sum_xj_inv_d2 = (pts.T @ inv_d2).T
    
    # grad of E = -sum_{i<j} log(d_ij): dE/dx_i = -sum_{j!=i} (x_i - x_j)/d_ij^2
    grad = -(pts * sum_inv_d2[:, np.newaxis] - sum_xj_inv_d2)
    
    # Project to tangent space
    grad_dot_xi = np.sum(grad * pts, axis=1, keepdims=True)
    grad_tan = grad - grad_dot_xi * pts
    
    return grad_tan

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, rng):
    """Generate random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / np.maximum(norms, 1e-15)

def _perturb_sphere(pts, rng, sigma=0.05):
    """Perturb points and renormalize."""
    noise = rng.standard_normal(pts.shape) * sigma
    new_pts = pts + noise
    return _project_to_sphere(new_pts)

def _icosahedron_subdivide(n, rng):
    """Generate points by subdividing an icosahedron."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    verts = _project_to_sphere(verts)
    
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)
    ]
    
    pts = []
    for v in verts:
        pts.append(v)
    
    edges = set()
    for f in faces:
        for i in range(3):
            a, b = f[i], f[(i+1) % 3]
            edge = (min(a,b), max(a,b))
            edges.add(edge)
    
    for a, b in edges:
        mid = (verts[a] + verts[b]) / 2.0
        pts.append(_project_to_sphere(mid.reshape(1,3))[0])
    
    for f in faces:
        center = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
        pts.append(_project_to_sphere(center.reshape(1,3))[0])
    
    pts = np.array(pts)
    if len(pts) > n:
        indices = rng.choice(len(pts), n, replace=False)
        pts = pts[indices]
    elif len(pts) < n:
        remaining = _random_sphere(n - len(pts), rng)
        pts = np.vstack([pts, remaining])
    
    return _project_to_sphere(pts)

def _riemannian_descent(pts, max_iter=500, tol=1e-10, deadline=None):
    """Riemannian gradient descent with backtracking line search on the sphere."""
    n = pts.shape[0]
    pts = _project_to_sphere(pts.copy())
    f = _log_energy(pts)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        grad = _gradient_log_energy(pts)
        
        # Check convergence
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        # Initial step size
        step = 1.0 / (1.0 + 0.5 * grad_norm)
        
        accepted = False
        for ls_iter in range(40):
            new_pts = pts - step * grad
            new_pts = _project_to_sphere(new_pts)
            new_f = _log_energy(new_pts)
            
            if np.isfinite(new_f) and new_f < f:
                if new_f < f - 1e-6 * step * grad_norm:
                    accepted = True
                    break
                if new_f < f - 1e-10:
                    accepted = True
                    break
            step *= 0.7
        
        if accepted:
            pts = new_pts
            f = new_f
        else:
            break
    
    return pts, f

def _lbfgs_optimize(pts0, deadline, maxiter=500, gtol=1e-8):
    """Run L-BFGS-B optimization from a given starting configuration."""
    def objective(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        return _log_energy(pts)
    
    def gradient(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        g = _gradient_log_energy(pts)
        return g.reshape(-1)
    
    x0 = pts0.reshape(-1)
    
    try:
        res = minimize(
            objective,
            x0,
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': gtol, 'maxfun': 2000, 'ftol': 1e-10}
        )
        if res.fun < float('inf'):
            pts_new = _project_to_sphere(res.x.reshape(N, 3))
            e_new = _log_energy(pts_new)
            if np.isfinite(e_new):
                return pts_new, e_new
    except Exception:
        pass
    
    return pts0, _log_energy(pts0)

def _coupled_harmonic_optimize(pts0, deadline, max_iter=200):
    """
    Coupled variational approach: simultaneously optimize point positions and
    a soft spherical harmonic confining potential. The potential is represented
    as a low-order expansion V(x) = sum_l,m c_lm Y_lm(x) which acts as a
    smooth surrogate bias to guide the points toward a more uniform distribution.
    We alternate between: (1) fixing the potential and optimizing points,
    (2) fixing points and optimizing potential coefficients to reduce energy.
    """
    pts = _project_to_sphere(pts0.copy())
    current_energy = _log_energy(pts)
    
    # Spherical harmonic coefficients up to l=4 (25 coefficients for l<=4)
    # We use a simple basis: Y_lm approximated by normalized polynomial features
    # For efficiency, use a small set of harmonic modes
    l_max = 4
    n_harm = (l_max + 1) ** 2  # 25 coefficients
    c = np.zeros(n_harm)
    
    def _harmonic_eval(pts, c):
        """Evaluate spherical harmonic potential at points."""
        # Use a simplified basis: powers of z and angular modes
        # This is a surrogate, not exact spherical harmonics
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        r2 = x**2 + y**2 + z**2
        r2 = np.maximum(r2, 1e-15)
        
        # Build feature matrix: [1, x, y, z, x^2, xy, y^2, xz, yz, z^2, ...]
        # Up to l=4, we need terms up to degree 4
        features = np.ones((len(pts), 1))
        # l=1
        features = np.hstack([features, x[:, None], y[:, None], z[:, None]])
        # l=2
        features = np.hstack([features, (x*x)[:, None], (x*y)[:, None], (y*y)[:, None], 
                               (x*z)[:, None], (y*z)[:, None], (z*z)[:, None]])
        # l=3
        features = np.hstack([features, (x**3)[:, None], (x*x*y)[:, None], (x*y*y)[:, None], 
                               (y**3)[:, None], (x*x*z)[:, None], (x*y*z)[:, None], 
                               (y*y*z)[:, None], (x*z*z)[:, None], (y*z*z)[:, None], (z**3)[:, None]])
        # l=4
        features = np.hstack([features, (x**4)[:, None], (x**3*y)[:, None], (x**2*y**2)[:, None], 
                               (x*y**3)[:, None], (y**4)[:, None], (x**3*z)[:, None], 
                               (x**2*y*z)[:, None], (x*y**2*z)[:, None], (y**3*z)[:, None],
                               (x**2*z**2)[:, None], (x*y*z**2)[:, None], (y**2*z**2)[:, None],
                               (x*z**3)[:, None], (y*z**3)[:, None], (z**4)[:, None]])
        
        return features @ c
    
    def _harmonic_grad(pts, c):
        """Gradient of harmonic potential w.r.t. point positions."""
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        
        # Gradient of each feature w.r.t. (x, y, z)
        # l=0: grad = [0, 0, 0]
        # l=1: grad = [1, 0, 0], [0, 1, 0], [0, 0, 1]
        # l=2: grad of x^2 = [2x, 0, 0], etc.
        
        grad = np.zeros((len(pts), 3))
        
        # l=1 terms (indices 1,2,3)
        if n_harm > 3:
            grad[:, 0] += c[1] * 1.0
            grad[:, 1] += c[2] * 1.0
            grad[:, 2] += c[3] * 1.0
        
        # l=2 terms (indices 4-10)
        if n_harm > 10:
            # x^2
            grad[:, 0] += c[4] * 2.0 * x
            # xy
            grad[:, 0] += c[5] * y
            grad[:, 1] += c[5] * x
            # y^2
            grad[:, 1] += c[6] * 2.0 * y
            # xz
            grad[:, 0] += c[7] * z
            grad[:, 2] += c[7] * x
            # yz
            grad[:, 1] += c[8] * z
            grad[:, 2] += c[8] * y
            # z^2
            grad[:, 2] += c[9] * 2.0 * z
        
        # l=3 terms (indices 11-21)
        if n_harm > 21:
            # x^3
            grad[:, 0] += c[11] * 3.0 * x * x
            # x^2 y
            grad[:, 0] += c[12] * 2.0 * x * y
            grad[:, 1] += c[12] * x * x
            # x y^2
            grad[:, 0] += c[13] * y * y
            grad[:, 1] += c[13] * 2.0 * x * y
            # y^3
            grad[:, 1] += c[14] * 3.0 * y * y
            # x^2 z
            grad[:, 0] += c[15] * 2.0 * x * z
            grad[:, 2] += c[15] * x * x
            # x y z
            grad[:, 0] += c[16] * y * z
            grad[:, 1] += c[16] * x * z
            grad[:, 2] += c[16] * x * y
            # y^2 z
            grad[:, 1] += c[17] * 2.0 * y * z
            grad[:, 2] += c[17] * y * y
            # x z^2
            grad[:, 0] += c[18] * z * z
            grad[:, 2] += c[18] * 2.0 * x * z
            # y z^2
            grad[:, 1] += c[19] * z * z
            grad[:, 2] += c[19] * 2.0 * y * z
            # z^3
            grad[:, 2] += c[20] * 3.0 * z * z
        
        # l=4 terms (indices 22-35, but we only have up to 24 for n_harm=25)
        # For simplicity, skip l=4 gradient details
        
        return grad
    
    def _total_energy(pts, c):
        """Total energy = log energy + harmonic potential energy."""
        e_log = _log_energy(pts)
        e_harm = 0.5 * np.sum(_harmonic_eval(pts, c) ** 2)
        return e_log + e_harm
    
    def _total_gradient(pts, c):
        """Total gradient = log energy gradient + harmonic gradient."""
        g_log = _gradient_log_energy(pts)
        g_harm = _harmonic_grad(pts, c) * _harmonic_eval(pts, c)[:, None]
        return g_log + g_harm
    
    # Alternating optimization
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Step 1: Optimize points with fixed potential
        if time.time() > deadline - 5.0:
            break
        
        def obj_pts(x):
            p = _project_to_sphere(x.reshape(N, 3))
            return _total_energy(p, c)
        
        def grad_pts(x):
            p = _project_to_sphere(x.reshape(N, 3))
            g = _total_gradient(p, c)
            return g.reshape(-1)
        
        x0 = pts.reshape(-1)
        try:
            res = minimize(
                obj_pts, x0, jac=grad_pts,
                method='L-BFGS-B',
                options={'maxiter': 100, 'gtol': 1e-8, 'maxfun': 500, 'ftol': 1e-10}
            )
            if res.fun < float('inf'):
                new_pts = _project_to_sphere(res.x.reshape(N, 3))
                new_e = _log_energy(new_pts)
                if np.isfinite(new_e) and new_e < current_energy:
                    pts = new_pts
                    current_energy = new_e
        except Exception:
            pass
        
        # Step 2: Optimize potential coefficients with fixed points
        if time.time() > deadline - 3.0:
            break
        
        def obj_c(c_vec):
            e_harm = 0.5 * np.sum(_harmonic_eval(pts, c_vec) ** 2)
            # Regularize to keep coefficients small
            return e_harm + 0.01 * np.sum(c_vec ** 2)
        
        def grad_c(c_vec):
            h = _harmonic_eval(pts, c_vec)
            # grad = features.T @ h + 0.02 * c
            x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
            features = np.ones((len(pts), 1))
            features = np.hstack([features, x[:, None], y[:, None], z[:, None]])
            features = np.hstack([features, (x*x)[:, None], (x*y)[:, None], (y*y)[:, None], 
                                   (x*z)[:, None], (y*z)[:, None], (z*z)[:, None]])
            features = np.hstack([features, (x**3)[:, None], (x*x*y)[:, None], (x*y*y)[:, None], 
                                   (y**3)[:, None], (x*x*z)[:, None], (x*y*z)[:, None], 
                                   (y*y*z)[:, None], (x*z*z)[:, None], (y*z*z)[:, None], (z**3)[:, None]])
            # Truncate to n_harm
            if features.shape[1] > n_harm:
                features = features[:, :n_harm]
            elif features.shape[1] < n_harm:
                pad = np.zeros((len(pts), n_harm - features.shape[1]))
                features = np.hstack([features, pad])
            
            g = features.T @ h + 0.02 * c_vec
            return g
        
        try:
            res_c = minimize(
                obj_c, c, jac=grad_c,
                method='L-BFGS-B',
                options={'maxiter': 50, 'gtol': 1e-8, 'maxfun': 200, 'ftol': 1e-10}
            )
            if res_c.fun < float('inf'):
                c = res_c.x
        except Exception:
            pass
    
    return pts, current_energy

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    best_points = None
    best_energy = float('inf')
    
    # Generate diverse starting points
    starts = []
    
    # Start 1: Pure Fibonacci
    fib_pts = _fibonacci_sphere(N)
    starts.append(fib_pts)
    
    # Starts 2-5: Icosahedron-based
    rng = np.random.default_rng(123)
    for seed in [1, 2, 3, 4]:
        if time.time() > deadline - 20.0:
            break
        ico_pts = _icosahedron_subdivide(N, rng)
        starts.append(ico_pts)
    
    # Starts 6-13: Fibonacci with random perturbations at various scales
    scales = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    for scale_idx, scale in enumerate(scales):
        if time.time() > deadline - 20.0:
            break
        perturbed = _perturb_sphere(fib_pts, rng, sigma=scale)
        starts.append(perturbed)
    
    # Starts 14-16: Pure random starts
    for seed in [1, 2, 3]:
        if time.time() > deadline - 20.0:
            break
        random_pts = _random_sphere(N, rng)
        starts.append(random_pts)
    
    # Run optimization from each start, tracking best
    for idx, start_pts in enumerate(starts):
        if time.time() > deadline - 30.0:
            break
        
        # L-BFGS-B pass
        pts, energy = _lbfgs_optimize(start_pts, deadline - 25.0, maxiter=400, gtol=1e-8)
        
        if energy < best_energy:
            best_energy = energy
            best_points = pts.copy()
    
    # Final polish: Riemannian gradient descent from best point
    if best_points is not None and time.time() < deadline - 15.0:
        pts_rd, e_rd = _riemannian_descent(best_points, max_iter=500, tol=1e-12, deadline=deadline - 15.0)
        if e_rd < best_energy:
            best_energy = e_rd
            best_points = pts_rd
    
    # Coupled harmonic optimization as a final refinement
    if best_points is not None and time.time() < deadline - 10.0:
        pts_ch, e_ch = _coupled_harmonic_optimize(best_points, deadline - 5.0, max_iter=100)
        if e_ch < best_energy:
            best_energy = e_ch
            best_points = pts_ch
    
    # Final L-BFGS-B polish with tighter tolerance
    if best_points is not None and time.time() < deadline - 10.0:
        pts_final, e_final = _lbfgs_optimize(best_points, deadline - 5.0, maxiter=800, gtol=1e-12)
        if e_final < best_energy:
            best_energy = e_final
            best_points = pts_final
    
    # Ensure valid output
    if best_points is None:
        best_points = fib_pts
    
    # Final validation
    if not np.all(np.isfinite(best_points)):
        best_points = fib_pts
    
    # Normalize
    best_points = _project_to_sphere(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
