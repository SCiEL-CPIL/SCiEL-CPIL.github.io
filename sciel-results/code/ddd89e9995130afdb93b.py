"""Optimize logarithmic energy of 150 points on the unit sphere."""
import numpy as np
import math
import time

N = 150
DEADLINE = 145.0  # seconds, with 5s safety margin before 150s limit


def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for points on unit sphere.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|d_ij|
    where d_ij is chordal distance.
    Gradient: dE/dx_i = -sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)  # (N, N)
    np.fill_diagonal(dist_sq, 1.0)  # avoid division by zero
    dist = np.sqrt(dist_sq)
    
    # Energy: sum over i<j of -log(d_ij)
    iu = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: dE/dx_i = -sum_{j!=i} (x_i - x_j) / dist_sq[i,j]
    inv_dist_sq = 1.0 / dist_sq  # (N, N)
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    row_sums = np.sum(inv_dist_sq, axis=1)  # (N,)
    weighted_sum = inv_dist_sq @ pts  # (N, 3)
    
    grad = -pts * row_sums[:, np.newaxis] + weighted_sum  # (N, 3)
    
    return energy, grad


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
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng):
    """Generate random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    return _project_to_sphere(pts)


def _tangent_move(pt, direction, step):
    """Move point along tangent direction and renormalize."""
    new_pt = pt + step * direction
    norm = np.linalg.norm(new_pt)
    if norm < 1e-15:
        return pt
    return new_pt / norm


def _local_minimize(pts, max_time, rng):
    """Local minimization using gradient descent with momentum on the sphere."""
    start_time = time.time()
    current = pts.copy()
    current_e, _ = _energy_and_grad(current)
    best = current.copy()
    best_e = current_e
    
    momentum = np.zeros_like(current)
    lr = 0.01
    beta = 0.9
    step_count = 0
    
    while time.time() - start_time < max_time:
        e, g = _energy_and_grad(current)
        
        # Project gradient to tangent space
        g_tan = g - np.sum(g * current, axis=1, keepdims=True) * current
        
        # Momentum update
        momentum = beta * momentum - lr * g_tan
        new_pts = current + momentum
        new_pts = _project_to_sphere(new_pts)
        
        new_e, _ = _energy_and_grad(new_pts)
        
        if new_e < current_e:
            current = new_pts
            current_e = new_e
            if current_e < best_e:
                best_e = current_e
                best = current.copy()
            step_count += 1
        else:
            # Reduce learning rate if stuck
            lr *= 0.95
            momentum *= 0.5
        
        # Check convergence
        if lr < 1e-12:
            break
        
        # Periodic restart with smaller noise
        if step_count > 100 and time.time() - start_time < max_time * 0.7:
            noise = 0.01 * rng.standard_normal_like(current)
            # Project noise to tangent space
            noise_tan = noise - np.sum(noise * current, axis=1, keepdims=True) * current
            current = _project_to_sphere(current + noise_tan)
            current_e, _ = _energy_and_grad(current)
            step_count = 0
            lr = 0.01
            momentum = np.zeros_like(current)
    
    return best, best_e


def _pso_optimize(initial_particles, max_time, rng, n_particles=10):
    """Particle swarm optimization on the sphere.
    
    Each particle is a configuration of N points.
    Particles move based on their personal best and the global best.
    """
    start_time = time.time()
    
    # Initialize particles
    particles = np.array([p.copy() for p in initial_particles])
    n_p = len(particles)
    
    # Compute initial energies
    energies = np.array([_energy_and_grad(p)[0] for p in particles])
    
    # Personal and global bests
    personal_best = particles.copy()
    personal_best_e = energies.copy()
    global_best_idx = np.argmin(energies)
    global_best = particles[global_best_idx].copy()
    global_best_e = energies[global_best_idx]
    
    # PSO parameters
    w = 0.7  # inertia weight
    c1 = 1.5  # cognitive coefficient
    c2 = 1.5  # social coefficient
    max_step = 0.1  # maximum step size
    
    iteration = 0
    while time.time() - start_time < max_time:
        iteration += 1
        
        # Update each particle
        for i in range(n_p):
            # Random coefficients
            r1 = rng.random(3)
            r2 = rng.random(3)
            
            # Velocity-like update in tangent space
            # Move towards personal best
            to_personal = personal_best[i] - particles[i]
            # Project to tangent space
            to_personal_tan = to_personal - np.sum(to_personal * particles[i], axis=1, keepdims=True) * particles[i]
            
            # Move towards global best
            to_global = global_best - particles[i]
            to_global_tan = to_global - np.sum(to_global * particles[i], axis=1, keepdims=True) * particles[i]
            
            # Combined update
            step = w * momentum[i] + c1 * r1 * to_personal_tan + c2 * r2 * to_global_tan
            
            # Limit step size
            step_norm = np.linalg.norm(step, axis=1, keepdims=True)
            step_max = np.maximum(step_norm, 1e-15)
            scale = np.minimum(1.0, max_step / step_max)
            step = step * scale
            
            # Update position
            new_pts = particles[i] + step
            new_pts = _project_to_sphere(new_pts)
            
            new_e, _ = _energy_and_grad(new_pts)
            
            # Update personal best
            if new_e < personal_best_e[i]:
                personal_best_e[i] = new_e
                personal_best[i] = new_pts.copy()
            
            # Update global best
            if new_e < global_best_e:
                global_best_e = new_e
                global_best = new_pts.copy()
            
            particles[i] = new_pts
        
        # Decay inertia weight
        w = max(0.4, w * 0.995)
        
        # Occasionally restart stuck particles
        if iteration % 50 == 0:
            for i in range(n_p):
                if personal_best_e[i] - global_best_e > 1.0:
                    # Restart particle near global best with noise
                    noise = 0.1 * rng.standard_normal_like(global_best)
                    noise_tan = noise - np.sum(noise * global_best, axis=1, keepdims=True) * global_best
                    particles[i] = _project_to_sphere(global_best + noise_tan)
                    personal_best[i] = particles[i].copy()
                    personal_best_e[i] = _energy_and_grad(particles[i])[0]
    
    return global_best, global_best_e


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # Generate diverse starting points
    starts = []
    
    # 1. Fibonacci spiral
    starts.append(_fibonacci_sphere(N))
    
    # 2. Perturbed Fibonacci
    for _ in range(5):
        fib = _fibonacci_sphere(N)
        noise = 0.05 * rng.standard_normal((N, 3))
        perturbed = fib + noise
        starts.append(_project_to_sphere(perturbed))
    
    # 3. Random points
    for _ in range(5):
        starts.append(_random_sphere(N, rng))
    
    # 4. Clustered starts
    for _ in range(3):
        centers = _random_sphere(10, rng)
        pts = []
        for c in centers:
            local = c + 0.1 * rng.standard_normal((15, 3))
            pts.append(_project_to_sphere(local))
        starts.append(np.vstack(pts))
    
    # Phase 1: Local minimization of each start
    best_pts = None
    best_energy = float('inf')
    
    # Allocate time: 40% for local minimization, 50% for PSO, 10% for final polish
    time_local = DEADLINE * 0.4
    time_pso = DEADLINE * 0.5
    time_polish = DEADLINE * 0.1
    
    local_start = time.time()
    time_per_local = (time_local - (local_start - start_time)) / len(starts)
    time_per_local = max(3.0, min(15.0, time_per_local))
    
    local_results = []
    for i, s in enumerate(starts):
        if time.time() - start_time > DEADLINE - time_pso - time_polish - 5:
            break
        
        remaining = DEADLINE - (time.time() - start_time) - time_pso - time_polish - 5
        if remaining < 3:
            break
        
        opt_time = min(time_per_local, remaining)
        if opt_time < 2:
            break
        
        try:
            pts_opt, e_opt = _local_minimize(s, opt_time, rng)
            local_results.append((pts_opt, e_opt))
            
            if e_opt < best_energy:
                best_energy = e_opt
                best_pts = pts_opt.copy()
        except Exception:
            continue
    
    # Phase 2: PSO using the best local results as initial particles
    if time.time() - start_time < DEADLINE - time_polish - 3:
        pso_start = time.time()
        pso_time = min(time_pso, DEADLINE - (pso_start - start_time) - time_polish - 3)
        
        if pso_time > 5:
            # Use top local results as PSO initial particles
            local_results.sort(key=lambda x: x[1])
            n_pso_particles = min(10, len(local_results))
            
            if n_pso_particles >= 2:
                pso_initial = [local_results[i][0] for i in range(n_pso_particles)]
                # Add some random particles for diversity
                for _ in range(max(0, 10 - n_pso_particles)):
                    pso_initial.append(_random_sphere(N, rng))
                
                try:
                    pso_pts, pso_e = _pso_optimize(pso_initial, pso_time, rng, n_particles=len(pso_initial))
                    if pso_e < best_energy:
                        best_energy = pso_e
                        best_pts = pso_pts.copy()
                except Exception:
                    pass
    
    # Phase 3: Final high-precision polish
    if best_pts is not None and time.time() - start_time < DEADLINE - 5:
        polish_start = time.time()
        polish_time = DEADLINE - (polish_start - start_time) - 3
        if polish_time > 3:
            try:
                pts_final, e_final = _local_minimize(best_pts, polish_time, rng)
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
                best_pts[j] += 1e-8 * rng.standard_normal(3)
                best_pts[j] = best_pts[j] / np.linalg.norm(best_pts[j])
    
    return best_pts
