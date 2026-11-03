"""Logarithmic energy minimization of 150 points on the unit sphere.
Uses a multi-start strategy with Adam, L-BFGS-B polishing, and simulated annealing."""

import numpy as np
import math
import time
from scipy.optimize import minimize

N = 150

# EVOLVE-BLOCK-START

def _energy_and_grad(pts):
    """
    Compute logarithmic energy and its gradient for points on the unit sphere.
    Energy: E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient: grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    The gradient is projected to the tangent space of the sphere.
    """
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero for i==j
    np.fill_diagonal(dist2, 1.0)
    
    # Compute energy
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dist2[iu])
    energy = -math.fsum(np.log(dists).tolist())
    
    # Compute gradient of E = -sum_{i<j} log(d_ij)
    # dE/dx_i = sum_{j != i} (x_i - x_j) / d_ij^2
    inv_dist2 = 1.0 / dist2
    np.fill_diagonal(inv_dist2, 0.0)
    
    # grad_i = sum_j (x_i - x_j) / d_ij^2 = x_i * sum_j 1/d_ij^2 - sum_j x_j / d_ij^2
    row_sums = np.sum(inv_dist2, axis=1)
    grad = pts * row_sums[:, np.newaxis] - inv_dist2 @ pts
    
    # Project gradient to tangent space
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts
    
    return energy, grad


def _energy_only(pts):
    """Compute only the energy."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dist2[iu])
    return -math.fsum(np.log(dists).tolist())


def _project_to_sphere(pts):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return pts / norms


def _fibonacci_spiral(n, offset=0.0, twist=1.0):
    """Generate points using a Fibonacci spiral."""
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i * twist
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _geodesic_dome(n, seed=0):
    """
    Generate points via recursive subdivision of an icosahedron (geodesic dome),
    then project to sphere. Provides a different structure than Fibonacci spiral.
    """
    rng = np.random.RandomState(seed)
    phi_g = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, phi_g, 0], [1, phi_g, 0], [-1, -phi_g, 0], [1, -phi_g, 0],
        [0, -1, phi_g], [0, 1, phi_g], [0, -1, -phi_g], [0, 1, -phi_g],
        [phi_g, 0, -1], [phi_g, 0, 1], [-phi_g, 0, -1], [-phi_g, 0, 1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    edges = [(0,11),(0,5),(0,1),(0,7),(0,10),(1,5),(5,11),(11,10),(10,7),
             (7,1),(1,6),(6,10),(2,11),(2,5),(2,9),(2,4),(2,8),(3,5),
             (3,11),(3,4),(3,9),(3,6),(4,9),(5,4),(11,4),(10,7),(6,1),
             (8,4),(9,8),(8,5),(6,9),(7,2),(8,3)]
    
    while len(verts) < n:
        new_verts = []
        for i, j in edges:
            mid = (verts[i] + verts[j]) / 2.0
            mid /= np.linalg.norm(mid)
            new_verts.append(mid)
        verts = np.vstack([verts, np.array(new_verts)])
    
    if len(verts) > n:
        idx = rng.choice(len(verts), size=n, replace=False)
        verts = verts[idx]
    
    return verts


def _random_points(n, seed=42):
    """Generate random points on the sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _adam_optimize(pts0, max_iter=3000, lr_start=0.01, lr_end=0.001,
                   beta1=0.9, beta2=0.999, eps=1e-8, deadline=None, check_interval=20):
    """
    Adam optimization on the sphere with learning rate decay.
    """
    pts = _project_to_sphere(pts0.copy())
    best_pts = pts.copy()
    best_energy = _energy_only(pts)

    m = np.zeros_like(pts)
    v = np.zeros_like(pts)
    
    def lr_schedule(t):
        return lr_end + 0.5 * (lr_start - lr_end) * (1 + math.cos(math.pi * t / max_iter))

    for t in range(1, max_iter + 1):
        if deadline is not None and (t % check_interval == 0 or t == 1):
            if time.time() > deadline:
                break

        energy, grad = _energy_and_grad(pts)
        
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        update = m_hat / (np.sqrt(v_hat) + eps)
        
        lr = lr_schedule(t)
        
        pts_new = pts - lr * update
        pts_new = _project_to_sphere(pts_new)

        if not np.all(np.isfinite(pts_new)):
            break

        energy_new = _energy_only(pts_new)
        if energy_new < best_energy:
            best_energy = energy_new
            best_pts = pts_new.copy()
            
        pts = pts_new

    return best_pts, best_energy


def _lbfgs_polish(pts, deadline, maxiter=500):
    """
    L-BFGS-B polish using flat Cartesian coordinates with projection.
    """
    def fun(x):
        p = _project_to_sphere(x.reshape(N, 3))
        e, g = _energy_and_grad(p)
        return e, g.flatten()

    x0 = pts.flatten()
    try:
        res = minimize(fun, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'maxfun': maxiter * 20,
                                'ftol': 1e-14, 'gtol': 1e-14, 'maxls': 40})
        p_res = _project_to_sphere(res.x.reshape(N, 3))
        e_check = _energy_only(p_res)
        return p_res, e_check
    except Exception:
        return pts, _energy_only(pts)


def _conjugate_gradient_polish(pts, deadline, max_iter=200):
    """
    Conjugate gradient polish on the sphere for final high-precision refinement.
    Uses exact line search (golden section) on the geodesic.
    """
    pts = _project_to_sphere(pts.copy())
    current_energy = _energy_only(pts)
    best_pts = pts.copy()
    best_energy = current_energy
    
    p = np.zeros_like(pts)  # search direction
    
    for k in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        
        energy, grad = _energy_and_grad(pts)
        current_energy = energy
        
        # Check convergence
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-12:
            break
        
        if k == 0:
            p = -grad
        else:
            # Fletcher-Reeves beta
            prev_grad_norm2 = np.sum(prev_grad * prev_grad)
            curr_grad_norm2 = np.sum(grad * grad)
            beta = curr_grad_norm2 / prev_grad_norm2 if prev_grad_norm2 > 0 else 0.0
            p = -grad + beta * p
            # Project p to tangent space
            dot_p = np.sum(p * pts, axis=1, keepdims=True)
            p = p - dot_p * pts
        
        # Store for next beta computation
        prev_grad = grad.copy()
        
        # Geodesic line search: move along p on the sphere
        # x(t) = cos(t*||p||) * x + sin(t*||p||)/||p|| * p
        p_norm = np.linalg.norm(p)
        if p_norm < 1e-14:
            break
        
        p_unit = p / p_norm
        
        # Golden section search for t in [0, 1]
        gr = (np.sqrt(5.0) - 1.0) / 2.0
        a, b = 0.0, 1.0
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        
        def eval_on_geodesic(t):
            if t < 1e-15:
                return current_energy
            angle = t * p_norm
            ca = np.cos(angle)
            sa = np.sin(angle) / p_norm
            new_pts = ca * pts + sa * p
            new_pts = _project_to_sphere(new_pts)
            return _energy_only(new_pts)
        
        fc = eval_on_geodesic(c)
        fd = eval_on_geodesic(d)
        
        for _ in range(30):
            if time.time() > deadline - 2.0:
                break
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = eval_on_geodesic(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = eval_on_geodesic(d)
            if abs(b - a) < 1e-10:
                break
        
        t_best = (a + b) / 2.0
        angle = t_best * p_norm
        ca = np.cos(angle)
        sa = np.sin(angle) / p_norm
        new_pts = ca * pts + sa * p
        new_pts = _project_to_sphere(new_pts)
        
        new_energy = _energy_only(new_pts)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pts = new_pts.copy()
        
        pts = new_pts
    
    return best_pts, best_energy


def _simulated_annealing(points, deadline, rng, max_iter=5000):
    """Simulated Annealing to escape local minima."""
    current_points = points.copy()
    current_energy = _energy_only(current_points)
    best_points = current_points.copy()
    best_energy = current_energy
    
    T = 0.5
    T_min = 1e-4
    cooling = 0.995
    
    for it in range(max_iter):
        if time.time() > deadline - 5.0:
            break
        if T < T_min:
            break
            
        n_perturb = rng.randint(1, 10)
        idxs = rng.choice(N, n_perturb, replace=False)
        
        new_points = current_points.copy()
        for idx in idxs:
            rand_vec = rng.randn(3)
            tangent = rand_vec - np.dot(rand_vec, new_points[idx]) * new_points[idx]
            norm_t = np.linalg.norm(tangent)
            if norm_t > 1e-15:
                tangent /= norm_t
            else:
                continue
            step_size = T * rng.uniform(0.1, 1.0)
            new_pt = new_points[idx] + step_size * tangent
            new_points[idx] = new_pt / np.linalg.norm(new_pt)
        
        new_energy = _energy_only(new_points)
        
        delta_e = new_energy - current_energy
        if delta_e < 0 or (T > 0 and rng.rand() < math.exp(-delta_e / T)):
            current_points = new_points
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = current_points.copy()
        
        T *= cooling

    return best_points, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    rng = np.random.RandomState(42)
    
    # Generate diverse initializations
    inits = []
    inits.append(_fibonacci_spiral(N))
    inits.append(_geodesic_dome(N, seed=0))
    
    for seed in range(5):
        r = np.random.RandomState(seed)
        inits.append(_random_points(N, seed=seed))
        
    for k in range(5):
        perturbed = _fibonacci_spiral(N) + rng.randn(N, 3) * 0.2
        inits.append(_project_to_sphere(perturbed))

    best_points = inits[0].copy()
    best_energy = _energy_only(best_points)
    
    # Phase 1: Local minimize all inits with Adam
    for init in inits:
        if time.time() > deadline - 40.0:
            break
        p_res, e_res = _adam_optimize(init, max_iter=2000, deadline=deadline - 40.0)
        if e_res < best_energy:
            best_energy = e_res
            best_points = p_res.copy()

    # Phase 2: Simulated Annealing from best candidates
    candidates = [best_points]
    # Add top-3 Adam results as SA starting points
    adam_results = []
    for init in inits:
        if time.time() > deadline - 40.0:
            break
        p_res, e_res = _adam_optimize(init, max_iter=1500, deadline=deadline - 40.0)
        adam_results.append((e_res, p_res))
    
    adam_results.sort(key=lambda x: x[0])
    for e, p in adam_results[:3]:
        candidates.append(p)
        
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 15.0:
            break
        sa_rng = np.random.RandomState(i * 100 + 123)
        sa_points, sa_energy = _simulated_annealing(cand, deadline, sa_rng, max_iter=2000)
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_points = sa_points.copy()

    # Phase 3: Final polishing with L-BFGS-B
    if time.time() < deadline - 10.0:
        final_points, final_energy = _lbfgs_polish(best_points, deadline, maxiter=2000)
        if final_energy < best_energy:
            best_energy = final_energy
            best_points = final_points.copy()

    # Phase 4: Conjugate gradient polish for high precision
    if time.time() < deadline - 3.0:
        cg_points, cg_energy = _conjugate_gradient_polish(best_points, deadline, max_iter=300)
        if cg_energy < best_energy:
            best_energy = cg_energy
            best_points = cg_points.copy()

    best_points = _project_to_sphere(best_points)

    # Safety checks
    if not np.all(np.isfinite(best_points)):
        return _fibonacci_spiral(N)

    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    iu = np.triu_indices(N, k=1)
    if np.min(dists[iu]) < 1e-9:
        return _fibonacci_spiral(N)

    return best_points
# EVOLVE-BLOCK-END
