"""Optimize logarithmic energy of 150 points on the unit sphere."""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 150
DEADLINE = 145.0  # seconds, with 5s safety margin before 150s limit


def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for points on unit sphere.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|d_ij|
    where d_ij is chordal distance.
    
    Gradient: dE/dx_i = -sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)  # (N, N)
    np.fill_diagonal(dist_sq, 1.0)  # avoid division by zero
    dist = np.sqrt(dist_sq)
    
    # Energy: sum over i<j of -log(d_ij)
    iu = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: dE/dx_i = -sum_{j!=i} (x_i - x_j) / dist_sq[i,j]
    # = sum_{j!=i} (x_j - x_i) / dist_sq[i,j]
    
    inv_dist_sq = 1.0 / dist_sq  # (N, N)
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    row_sums = np.sum(inv_dist_sq, axis=1)  # (N,)
    weighted_sum = inv_dist_sq @ pts  # (N, 3)
    
    # grad[i] = sum_j (x_j - x_i) / dist_sq[i,j] = weighted_sum[i] - x_i * row_sums[i]
    grad = weighted_sum - pts * row_sums[:, np.newaxis]  # (N, 3)
    
    return energy, grad


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _tangent_project(pts, grad):
    """Project gradient onto tangent space of sphere at each point."""
    # Tangent projection: g_tan = g - (g·n)n where n = pts (unit normal)
    g_dot_n = np.sum(grad * pts, axis=1, keepdims=True)
    return grad - g_dot_n * pts


def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _icosahedron_based(n):
    """Generate points using icosahedron subdivision for better initial distribution."""
    # Start with icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    # If n == 12, return icosahedron
    if n == 12:
        return ico
    
    # Subdivide faces and add more points
    # For simplicity, use Fibonacci for the rest but perturb near icosahedron
    fib = _fibonacci_sphere(n)
    # Blend: use icosahedron structure as anchor
    result = fib.copy()
    for i in range(12):
        # Find closest point in fib to each ico vertex
        dists = np.linalg.norm(fib - ico[i], axis=1)
        closest = np.argmin(dists)
        # Pull it toward the icosahedron vertex
        result[closest] = ico[i]
    return _project_to_sphere(result)


def _random_sphere(n, rng):
    """Generate random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    return _project_to_sphere(pts)


def _so2_rotate(pts, rng):
    """Apply random SO(3) rotation to points."""
    # Generate random rotation matrix
    u = rng.standard_normal(3)
    v = rng.standard_normal(3)
    w = rng.standard_normal(3)
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    w = np.cross(u, v)
    w = w / np.linalg.norm(w)
    R = np.column_stack([u, v, w])
    return pts @ R.T


def _optimize_single(pts0, max_time):
    """Optimize using L-BFGS-B with spherical constraint via parameterization."""
    start_time = time.time()
    
    # Use spherical coordinates or direct optimization with projection
    # We'll use L-BFGS-B on the raw 3D coordinates with a penalty for non-unit norm
    # Actually, better: use the fact that we can optimize in R^3 and project
    
    pts = _project_to_sphere(pts0.copy())
    
    def objective_and_grad(x_flat):
        x = x_flat.reshape(N, 3)
        # Project to sphere to ensure constraint
        x = _project_to_sphere(x)
        e, g = _energy_and_grad(x)
        # For L-BFGS-B, we need gradient w.r.t. x_flat
        # Since we project, the effective gradient is the tangent component
        g_tan = _tangent_project(x, g)
        # But L-BFGS-B operates in flat space, so we return the projected gradient
        # Actually, this won't work well. Let's use a different approach.
        return e, g_tan.reshape(-1)
    
    # Use L-BFGS-B with the raw coordinates
    x0 = pts.reshape(-1)
    
    best_pts = pts.copy()
    best_e, _ = _energy_and_grad(pts)
    
    def callback(xk):
        nonlocal best_pts, best_e
        if time.time() - start_time >= max_time:
            raise RuntimeError("timeout")
        xk_3d = _project_to_sphere(xk.reshape(N, 3))
        e_k, _ = _energy_and_grad(xk_3d)
        if e_k < best_e:
            best_e = e_k
            best_pts = xk_3d.copy()
        return time.time() - start_time >= max_time
    
    try:
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            callback=callback,
            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 2000}
        )
        if result.success or result.fun < best_e:
            res_pts = _project_to_sphere(result.x.reshape(N, 3))
            res_e, _ = _energy_and_grad(res_pts)
            if res_e < best_e:
                best_e = res_e
                best_pts = res_pts.copy()
    except RuntimeError:
        pass
    except Exception:
        pass
    
    # Final polish with tighter tolerance
    if time.time() - start_time < max_time - 1:
        try:
            x0 = best_pts.reshape(-1)
            def obj2(x_flat):
                x = _project_to_sphere(x_flat.reshape(N, 3))
                e, _ = _energy_and_grad(x)
                return e
            def grad2(x_flat):
                x = _project_to_sphere(x_flat.reshape(N, 3))
                _, g = _energy_and_grad(x)
                g_tan = _tangent_project(x, g)
                return g_tan.reshape(-1)
            
            result2 = minimize(
                obj2,
                x0,
                jac=grad2,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 1000}
            )
            res_pts = _project_to_sphere(result2.x.reshape(N, 3))
            res_e, _ = _energy_and_grad(res_pts)
            if res_e < best_e:
                best_e = res_e
                best_pts = res_pts.copy()
        except Exception:
            pass
    
    return best_pts, best_e


def _simulated_annealing(pts, max_time, rng):
    """Simulated annealing with batch moves to escape local minima."""
    start_time = time.time()
    current = pts.copy()
    current_e, _ = _energy_and_grad(current)
    best = current.copy()
    best_e = current_e
    
    T = 0.02
    T_min = 1e-10
    cooling = 0.998
    
    while time.time() - start_time < max_time and T > T_min:
        # Move a single point for faster evaluation
        idx = rng.integers(N)
        
        new_pts = current.copy()
        # Random direction in tangent space
        rand_dir = rng.standard_normal(3)
        rand_dir -= np.dot(rand_dir, current[idx]) * current[idx]
        norm = np.linalg.norm(rand_dir)
        if norm > 1e-15:
            rand_dir /= norm
        
        # Move along random direction with adaptive step
        step_size = T * 0.5
        new_pt = current[idx].copy()
        new_pt += step_size * rand_dir
        new_pt = new_pt / np.linalg.norm(new_pt)
        new_pts[idx] = new_pt
        
        new_e, _ = _energy_and_grad(new_pts)
        delta_e = new_e - current_e
        
        if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 1e-12)):
            current = new_pts
            current_e = new_e
            if current_e < best_e:
                best_e = current_e
                best = current.copy()
        
        T *= cooling
    
    return best, best_e


def _pso_optimize(starts, max_time, rng):
    """Particle Swarm Optimization in tangent space.
    
    Each particle is a configuration of N points on the sphere.
    Velocity is maintained in tangent space at each particle's position.
    """
    start_time = time.time()
    n_particles = len(starts)
    
    # Initialize particles
    particles = np.array([_project_to_sphere(s.copy()) for s in starts])
    velocities = [np.zeros_like(p) for p in particles]
    pbests = [p.copy() for p in particles]
    pbest_energies = [None] * n_particles
    gbest = None
    gbest_energy = float('inf')
    
    # Compute initial energies
    for i in range(n_particles):
        e, _ = _energy_and_grad(particles[i])
        pbest_energies[i] = e
        if e < gbest_energy:
            gbest_energy = e
            gbest = particles[i].copy()
    
    # PSO parameters
    w = 0.7  # inertia
    c1 = 1.5  # cognitive
    c2 = 1.5  # social
    max_vel = 0.05  # max step size
    
    for iteration in range(10000):
        if time.time() - start_time >= max_time:
            break
        
        for i in range(n_particles):
            # Compute random coefficients
            r1 = rng.random((N, 3))
            r2 = rng.random((N, 3))
            
            # pbest - x (tangent direction)
            diff_pbest = pbests[i] - particles[i]
            # Project to tangent space
            diff_pbest_tan = diff_pbest - np.sum(diff_pbest * particles[i], axis=1, keepdims=True) * particles[i]
            
            # gbest - x (tangent direction)
            diff_gbest = gbest - particles[i]
            diff_gbest_tan = diff_gbest - np.sum(diff_gbest * particles[i], axis=1, keepdims=True) * particles[i]
            
            # Update velocity
            velocities[i] = w * velocities[i] + c1 * r1 * diff_pbest_tan + c2 * r2 * diff_gbest_tan
            
            # Clamp velocity
            vel_norms = np.linalg.norm(velocities[i], axis=1, keepdims=True)
            too_large = vel_norms > max_vel
            if np.any(too_large):
                velocities[i] = np.where(too_large, velocities[i] * (max_vel / np.maximum(vel_norms, 1e-15)), velocities[i])
            
            # Update position
            new_pts = particles[i] + velocities[i]
            new_pts = _project_to_sphere(new_pts)
            
            # Evaluate
            new_e, _ = _energy_and_grad(new_pts)
            
            if new_e < pbest_energies[i]:
                pbest_energies[i] = new_e
                pbests[i] = new_pts.copy()
            
            if new_e < gbest_energy:
                gbest_energy = new_e
                gbest = new_pts.copy()
            
            particles[i] = new_pts
    
    return gbest, gbest_energy


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # Generate diverse starting points
    starts = []
    
    # 1. Fibonacci spiral
    starts.append(_fibonacci_sphere(N))
    
    # 2. Icosahedron-based
    starts.append(_icosahedron_based(N))
    
    # 3. Perturbed Fibonacci
    for _ in range(3):
        fib = _fibonacci_sphere(N)
        noise = 0.05 * rng.standard_normal((N, 3))
        perturbed = fib + noise
        starts.append(_project_to_sphere(perturbed))
    
    # 4. Random points
    for _ in range(4):
        starts.append(_random_sphere(N, rng))
    
    # 5. Rotated Fibonacci
    for _ in range(2):
        fib = _fibonacci_sphere(N)
        starts.append(_so2_rotate(fib, rng))
    
    # Optimize each start with L-BFGS
    best_pts = None
    best_energy = float('inf')
    
    n_starts = len(starts)
    # Allocate time: ~60% for initial optimizations, ~25% for SA, ~15% for final polish
    opt_budget = (DEADLINE - (time.time() - start_time)) * 0.6
    sa_budget = (DEADLINE - (time.time() - start_time)) * 0.25
    polish_budget = (DEADLINE - (time.time() - start_time)) * 0.15
    
    time_per_opt = opt_budget / n_starts
    time_per_opt = max(3.0, min(15.0, time_per_opt))
    
    for i, s in enumerate(starts):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        remaining = DEADLINE - (time.time() - start_time)
        if remaining < 30:
            break
        
        opt_time = min(time_per_opt, remaining - 25)
        if opt_time < 3:
            break
        
        try:
            pts_opt, e_opt = _optimize_single(s, opt_time)
            
            if e_opt < best_energy:
                best_energy = e_opt
                best_pts = pts_opt.copy()
        except Exception:
            continue
    
    # Simulated annealing on best result
    if best_pts is not None and time.time() - start_time < DEADLINE - 20:
        remaining = DEADLINE - (time.time() - start_time) - 15
        if remaining > 5:
            try:
                pts_sa, e_sa = _simulated_annealing(best_pts, remaining, rng)
                if e_sa < best_energy:
                    best_energy = e_sa
                    best_pts = pts_sa.copy()
            except Exception:
                pass
    
    # Final high-precision polish
    if best_pts is not None and time.time() - start_time < DEADLINE - 5:
        remaining = DEADLINE - (time.time() - start_time) - 3
        if remaining > 3:
            try:
                pts_final, e_final = _optimize_single(best_pts, remaining)
                if e_final < best_energy:
                    best_energy = e_final
                    best_pts = pts_final.copy()
            except Exception:
                pass
    
    if best_pts is None:
        # Fallback to Fibonacci
        best_pts = _fibonacci_sphere(N)
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    assert best_pts.shape == (N, 3)
    assert np.all(np.isfinite(best_pts))
    assert np.all(np.linalg.norm(best_pts, axis=1) > 1e-15)
    
    # Check no two points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Separate them slightly
                best_pts[j] += 1e-8 * rng.standard_normal(3)
                best_pts[j] = best_pts[j] / np.linalg.norm(best_pts[j])
    
    return best_pts
