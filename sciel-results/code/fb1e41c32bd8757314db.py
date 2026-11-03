"""Rewritten program for logarithmic energy minimization of 150 points on sphere."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _log_energy(points):
    """Compute log(1/d) energy for points on unit sphere using math.fsum."""
    n = len(points)
    iu = np.triu_indices(n, k=1)
    diff = points[iu[0]] - points[iu[1]]
    dists_sq = np.sum(diff * diff, axis=-1)
    dists = np.sqrt(np.maximum(dists_sq, 1e-30))
    log_dists = np.log(dists)
    return -math.fsum(log_dists.tolist())

def _energy_and_grad(x):
    """Energy and gradient for L-BFGS-B. x is flat array of shape (N*3,)."""
    points = x.reshape(N, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    points = points / norms

    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    
    inv_dists_sq = 1.0 / dists_sq
    np.fill_diagonal(inv_dists_sq, 0.0)
    
    sum_inv_dists_sq = np.sum(inv_dists_sq, axis=1)
    weighted_points = inv_dists_sq @ points
    
    grad = weighted_points - points * sum_inv_dists_sq[:, np.newaxis]
    
    # Project gradient to tangent space
    dot_products = np.sum(grad * points, axis=1, keepdims=True)
    grad_tangent = grad - dot_products * points
    
    energy = _log_energy(points)
    
    return energy, grad_tangent.flatten()

def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_uniform_sphere(n, rng):
    """Generate random uniform points on sphere."""
    points = rng.randn(n, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms

def _icosahedron_subdivided(n, rng):
    """Generate points from a subdivided icosahedron, then jitter and project."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    points_list = [v.tolist() for v in verts]
    
    for level in range(3):
        new_points = []
        for i in range(len(points_list)):
            new_points.append(points_list[i])
            for j in range(i+1, len(points_list)):
                d = np.linalg.norm(np.array(points_list[i]) - np.array(points_list[j]))
                if d < 1.5:
                    mid = (np.array(points_list[i]) + np.array(points_list[j])) / 2.0
                    mid = mid / np.linalg.norm(mid)
                    new_points.append(mid.tolist())
        points_list = new_points
    
    pts = np.array(points_list)
    if len(pts) >= n:
        idx = rng.choice(len(pts), n, replace=False)
        pts = pts[idx]
    else:
        extra = _random_uniform_sphere(n - len(pts), rng)
        pts = np.vstack([pts, extra])
    
    pts = pts + rng.randn(n, 3) * 0.01
    return _project_to_sphere(pts)

def _velocity_verlet_minimize(points, deadline, max_iter=20000, dt_init=0.01, damping=0.999):
    """
    Velocity-Verlet integration of the repulsive force field on the sphere.
    Treats points as particles with logarithmic repulsion, integrated with
    adaptive timestep and velocity damping to settle into a minimum.
    """
    pos = points.copy().astype(float)
    vel = np.zeros_like(pos)
    
    # Initial forces
    def compute_forces(positions):
        n = len(positions)
        forces = np.zeros_like(positions)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                diff = positions[i] - positions[j]
                dist_sq = np.dot(diff, diff)
                if dist_sq < 1e-12:
                    # Random perturbation for coincident points
                    forces[i] += rng.randn(3) * 0.1
                    continue
                dist = math.sqrt(dist_sq)
                # Force is -grad of energy = -(-1/dist * d/dist) = d/dist * (1/dist)
                # Actually: energy = sum log(1/d) = -sum log(d)
                # grad_i energy = sum_{j!=i} -1/d_ij * (x_i - x_j)/d_ij = -sum_{j!=i} (x_i - x_j)/d_ij^2
                # Force = -grad = sum_{j!=i} (x_i - x_j)/d_ij^2
                force = diff / dist_sq
                forces[i] += force
        return forces
    
    rng_local = np.random.RandomState(12345)
    
    dt = dt_init
    min_dt = 1e-6
    max_dt = 0.05
    
    energy = _log_energy(pos)
    best_pos = pos.copy()
    best_energy = energy
    
    for it in range(max_iter):
        if time.time() > deadline - 10.0:
            break
        
        # Compute forces
        forces = compute_forces(pos)
        
        # Project forces to tangent space
        for i in range(len(pos)):
            radial_comp = np.dot(forces[i], pos[i])
            forces[i] -= radial_comp * pos[i]
        
        # Velocity-Verlet: half-step velocity
        vel += 0.5 * dt * forces
        
        # Update position
        pos = pos + dt * vel
        
        # Project to sphere
        pos = _project_to_sphere(pos)
        
        # Compute new forces
        forces_new = compute_forces(pos)
        for i in range(len(pos)):
            radial_comp = np.dot(forces_new[i], pos[i])
            forces_new[i] -= radial_comp * pos[i]
        
        # Second half-step velocity
        vel += 0.5 * dt * forces_new
        
        # Damping
        vel *= damping
        
        # Adaptive timestep: if energy increases too much, reduce dt
        new_energy = _log_energy(pos)
        if new_energy > energy + 0.1:
            dt *= 0.5
            # Rollback
            pos = _project_to_sphere(pos - dt * vel)
            vel = vel * 0.5
            new_energy = _log_energy(pos)
        else:
            # If stable, can slowly increase dt
            if dt < max_dt:
                dt = min(dt * 1.01, max_dt)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pos = pos.copy()
        
        energy = new_energy
        
        # Check convergence
        if it > 1000 and abs(new_energy - energy) < 1e-12:
            break
    
    return best_pos, best_energy

def _local_minimize(points, deadline, maxiter=1000, tol=1e-12):
    """Run L-BFGS-B to find local minimum."""
    x0 = points.flatten()
    try:
        res = minimize(
            _energy_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'maxfun': maxiter * 20, 'ftol': tol, 'gtol': tol, 'maxls': 40}
        )
        p_res = _project_to_sphere(res.x.reshape(N, 3))
        e_check = _log_energy(p_res)
        return p_res, e_check
    except Exception:
        return points, _log_energy(points)

def _basin_hopping(points, deadline, rng, n_outer=50, n_inner=200):
    """Basin hopping: perturb then locally minimize."""
    current_points = points.copy()
    current_energy = _log_energy(current_points)
    best_points = current_points.copy()
    best_energy = current_energy
    
    T = 0.3
    T_min = 0.01
    cooling = 0.98
    
    for outer in range(n_outer):
        if time.time() > deadline - 5.0:
            break
        if T < T_min:
            break
        
        # Perturb all points slightly
        perturbed = current_points.copy()
        for i in range(N):
            rand_vec = rng.randn(3)
            tangent = rand_vec - np.dot(rand_vec, perturbed[i]) * perturbed[i]
            norm_t = np.linalg.norm(tangent)
            if norm_t > 1e-15:
                tangent /= norm_t
            else:
                continue
            step = T * rng.uniform(0.5, 1.5)
            new_pt = perturbed[i] + step * tangent
            perturbed[i] = new_pt / np.linalg.norm(new_pt)
        
        # Local minimize from perturbed
        p_res, e_res = _local_minimize(perturbed, deadline, maxiter=n_inner, tol=1e-10)
        
        # Metropolis
        delta_e = e_res - current_energy
        if delta_e < 0 or (T > 0 and rng.rand() < math.exp(-delta_e / T)):
            current_points = p_res
            current_energy = e_res
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = current_points.copy()
        
        T *= cooling
    
    return best_points, best_energy

def _simulated_annealing(points, deadline, rng, max_iter=5000):
    """Simulated Annealing to escape local minima."""
    current_points = points.copy()
    current_energy = _log_energy(current_points)
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
            
        # Perturb a random subset of points
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
        
        new_energy = _log_energy(new_points)
        
        # Metropolis criterion
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
    deadline = start_time + 145.0

    rng = np.random.RandomState(42)
    
    # Generate diverse initializations
    inits = []
    inits.append(_fibonacci_sphere(N))
    inits.append(_icosahedron_subdivided(N, rng))
    
    for seed in range(4):
        r = np.random.RandomState(seed)
        inits.append(_random_uniform_sphere(N, r))
        
    for k in range(4):
        perturbed = _fibonacci_sphere(N) + rng.randn(N, 3) * 0.15
        inits.append(_project_to_sphere(perturbed))
    
    for k in range(2):
        r2 = np.random.RandomState(100 + k)
        inits.append(_icosahedron_subdivided(N, r2))

    best_points = inits[0].copy()
    best_energy = _log_energy(best_points)
    
    # Phase 1: Velocity-Verlet dynamics on all initializations
    for init in inits:
        if time.time() > deadline - 35.0:
            break
        vv_points, vv_energy = _velocity_verlet_minimize(init, deadline, max_iter=5000)
        if vv_energy < best_energy:
            best_energy = vv_energy
            best_points = vv_points.copy()

    # Phase 2: Local minimize all inits (quick pass)
    for init in inits:
        if time.time() > deadline - 25.0:
            break
        p_res, e_res = _local_minimize(init, deadline, maxiter=100, tol=1e-10)
        if e_res < best_energy:
            best_energy = e_res
            best_points = p_res.copy()

    # Phase 3: Basin hopping from best candidate
    if time.time() < deadline - 20.0:
        bh_rng = np.random.RandomState(777)
        bh_points, bh_energy = _basin_hopping(best_points, deadline, bh_rng, n_outer=25, n_inner=100)
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_points = bh_points.copy()

    # Phase 4: Simulated Annealing from best
    if time.time() < deadline - 15.0:
        sa_rng = np.random.RandomState(123)
        sa_points, sa_energy = _simulated_annealing(best_points, deadline, sa_rng, max_iter=1000)
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_points = sa_points.copy()

    # Phase 5: Velocity-Verlet again from best (refinement)
    if time.time() < deadline - 10.0:
        vv_points, vv_energy = _velocity_verlet_minimize(best_points, deadline, max_iter=10000)
        if vv_energy < best_energy:
            best_energy = vv_energy
            best_points = vv_points.copy()

    # Phase 6: Final polishing with L-BFGS-B
    if time.time() < deadline - 5.0:
        final_points, final_energy = _local_minimize(best_points, deadline, maxiter=500, tol=1e-13)
        if final_energy < best_energy:
            best_energy = final_energy
            best_points = final_points.copy()
    
    if time.time() < deadline - 3.0:
        final_points, final_energy = _local_minimize(best_points, deadline, maxiter=1000, tol=1e-15)
        if final_energy < best_energy:
            best_energy = final_energy
            best_points = final_points.copy()

    best_points = _project_to_sphere(best_points)

    # Safety checks
    if not np.all(np.isfinite(best_points)):
        return _fibonacci_sphere(N)

    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    iu = np.triu_indices(N, k=1)
    if np.min(dists[iu]) < 1e-9:
        return _fibonacci_sphere(N)

    return best_points
# EVOLVE-BLOCK-END
