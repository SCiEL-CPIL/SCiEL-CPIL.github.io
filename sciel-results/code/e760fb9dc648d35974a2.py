import numpy as np
from scipy.optimize import minimize
import time
import math

N = 150

def _fibonacci_sphere(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _geodesic_dome(n, seed=0):
    """Generate points via recursive subdivision of an icosahedron."""
    rng = np.random.RandomState(seed)
    # Icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Edges of icosahedron (12 vertices, 30 edges)
    edges = [
        (0,1),(0,5),(0,7),(0,10),(0,11),
        (1,5),(1,7),(1,8),(1,9),
        (2,3),(2,4),(2,6),(2,10),(2,11),
        (3,4),(3,6),(3,8),(3,9),
        (4,5),(4,9),(4,11),
        (5,9),(5,11),
        (6,7),(6,8),(6,10),
        (7,8),(7,10),
        (8,9),
        (10,11)
    ]
    
    points = verts.copy()
    midpoints = {}
    
    def midpoint(i, j):
        key = (min(i,j), max(i,j))
        if key in midpoints:
            return midpoints[key]
        p = (points[i] + points[j]) / 2.0
        p = p / np.linalg.norm(p)
        midpoints[key] = len(points)
        points = np.vstack([points, p])
        return midpoints[key]
    
    # Subdivide until we have at least n points
    while len(points) < n:
        new_edges = []
        for i, j in edges:
            mid = midpoint(i, j)
            new_edges.append((i, mid))
            new_edges.append((mid, j))
        edges = new_edges
    
    # If we have more than n points, randomly select n
    if len(points) > n:
        indices = rng.choice(len(points), size=n, replace=False)
        points = points[indices]
    
    return points

def _energy_and_grad(X):
    """Compute logarithmic energy and gradient for points on unit sphere."""
    n = X.shape[0]
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(n, k=1)
    energy = -math.fsum(np.log(dist[iu]).tolist())
    
    with np.errstate(divide='ignore', invalid='ignore'):
        grad_term = diff / (dist_sq[:, :, np.newaxis] + 1e-30)
    np.fill_diagonal(grad_term[:, :, 0], 0)
    np.fill_diagonal(grad_term[:, :, 1], 0)
    np.fill_diagonal(grad_term[:, :, 2], 0)
    
    grad = -np.sum(grad_term, axis=1)
    
    return energy, grad

def _energy_only(X):
    """Compute only the logarithmic energy with fsum for precision."""
    n = X.shape[0]
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    return -math.fsum(np.log(dist[iu]).tolist())

def _project_to_sphere(X):
    """Project points back to unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms

def _exp_map(X, V):
    """Exponential map on the unit sphere."""
    v_norms = np.linalg.norm(V, axis=1, keepdims=True)
    small_mask = (v_norms < 1e-10).flatten()
    
    result = np.zeros_like(X)
    
    if np.any(small_mask):
        V_small = V[small_mask]
        X_small = X[small_mask]
        result[small_mask] = X_small + V_small + 0.5 * X_small * np.sum(V_small * V_small, axis=1, keepdims=True)
    
    if np.any(~small_mask):
        V_large = V[~small_mask]
        X_large = X[~small_mask]
        v_norms_large = v_norms[~small_mask].flatten()
        cos_theta = np.cos(v_norms_large)
        sin_theta = np.sin(v_norms_large)
        result[~small_mask] = X_large * cos_theta[:, np.newaxis] + V_large * (sin_theta / v_norms_large)[:, np.newaxis]
    
    return _project_to_sphere(result)

def _riemannian_cg(X, max_iter=200, tol=1e-12, time_limit=None, start_time=None):
    """Riemannian conjugate gradient method on the unit sphere."""
    if time_limit is None:
        time_limit = 100.0
    if start_time is None:
        start_time = time.time()
    
    n = X.shape[0]
    X = _project_to_sphere(X.copy())
    
    energy, grad = _energy_and_grad(X)
    grad_t = grad - np.sum(grad * X, axis=1, keepdims=True) * X
    
    d = -grad_t
    d_norm = np.linalg.norm(d)
    if d_norm < 1e-15:
        return X, energy
    
    d = d / d_norm
    prev_grad_t = grad_t.copy()
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        if np.linalg.norm(grad_t) < tol:
            break
        
        alpha = 0.1
        best_alpha = 0.0
        best_energy = energy
        best_X = X
        
        for ls in range(20):
            X_new = _exp_map(X, alpha * d)
            e_new, g_new = _energy_and_grad(X_new)
            
            if e_new < energy - 1e-4 * alpha * np.sum(grad_t * d):
                best_alpha = alpha
                best_energy = e_new
                best_X = X_new
                break
            else:
                best_alpha = alpha
                best_energy = min(best_energy, e_new)
                best_X = X_new if e_new < best_energy else best_X
                alpha *= 0.5
        
        X = best_X
        energy = best_energy
        
        _, grad_new = _energy_and_grad(X)
        grad_new_t = grad_new - np.sum(grad_new * X, axis=1, keepdims=True) * X
        
        if energy >= best_energy - 1e-15:
            d = -grad_new_t
            d_norm = np.linalg.norm(d)
            if d_norm < 1e-15:
                break
            d = d / d_norm
            prev_grad_t = grad_new_t.copy()
            continue
        
        beta = np.sum(grad_new_t * grad_new_t) / max(np.sum(prev_grad_t * prev_grad_t), 1e-30)
        beta = max(beta, 0.0)
        
        d_new = -grad_new_t + beta * d
        
        if np.sum(d_new * grad_new_t) > 0:
            d_new = -grad_new_t
        
        d_norm = np.linalg.norm(d_new)
        if d_norm < 1e-15:
            break
        d = d_new / d_norm
        
        prev_grad_t = grad_new_t.copy()
    
    return X, energy

def _lbfgs_polish(X, time_limit=30.0, start_time=None):
    """Final L-BFGS polish using scipy's minimize."""
    if start_time is None:
        start_time = time.time()
    
    if time.time() - start_time > time_limit:
        return X
    
    def objective_flat(x_flat):
        X_local = x_flat.reshape(N, 3)
        X_local = _project_to_sphere(X_local)
        e, _ = _energy_and_grad(X_local)
        return e
    
    def gradient_flat(x_flat):
        X_local = x_flat.reshape(N, 3)
        X_local = _project_to_sphere(X_local)
        _, g = _energy_and_grad(X_local)
        g_t = g - np.sum(g * X_local, axis=1, keepdims=True) * X_local
        return g_t.flatten()
    
    x0 = X.flatten()
    
    try:
        res = minimize(
            objective_flat,
            x0,
            jac=gradient_flat,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 10000}
        )
        
        if res.success or res.fun < _energy_only(X):
            X_new = res.x.reshape(N, 3)
            X_new = _project_to_sphere(X_new)
            return X_new
    except Exception:
        pass
    
    return X

def _psO_search(X_best, E_best, time_budget, start_time):
    """Particle swarm optimization to explore configuration space."""
    rng = np.random.RandomState(42)
    n_particles = 10
    max_iters = 30
    
    # Initialize particles around best solution with varying perturbation scales
    particles = []
    pbests = []
    pbest_energies = []
    
    for k in range(n_particles):
        scale = 0.02 * (k + 1) / n_particles
        X_k = X_best + scale * rng.randn(N, 3)
        X_k = _project_to_sphere(X_k)
        E_k = _energy_only(X_k)
        particles.append(X_k)
        pbests.append(X_k.copy())
        pbest_energies.append(E_k)
    
    gbest = X_best.copy()
    gbest_energy = E_best
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        for k in range(n_particles):
            if time.time() - start_time > time_budget:
                break
            
            # Update velocity (in tangent space conceptually, but we use simple Euclidean for PSO)
            r1 = rng.rand(N, 3)
            r2 = rng.rand(N, 3)
            
            # Inertia
            v = 0.7 * (particles[k] - pbests[k])  # This is a simplification
            # Actually, let's do proper PSO in Cartesian then project
            
            # Standard PSO: v_k = w*v_k + c1*r1*(pbest_k - x_k) + c2*r2*(gbest - x_k)
            # But we need to be careful with the sphere constraint
            
            # Let's use a simpler approach: generate new positions as convex combinations
            # and project to sphere
            
            w = 0.7
            c1 = 1.5
            c2 = 1.5
            
            # Velocity update in Cartesian (will be projected)
            # We'll track velocities implicitly through position updates
            # For simplicity, let's just do random perturbations weighted by pbest and gbest
            
            # New position = w * current + c1*r1*(pbest - current) + c2*r2*(gbest - current)
            # This is a linear combination that we then project to sphere
            
            X_new = (w * particles[k] + 
                    c1 * r1 * (pbests[k] - particles[k]) + 
                    c2 * r2 * (gbest - particles[k]))
            X_new = _project_to_sphere(X_new)
            
            E_new = _energy_only(X_new)
            
            # Update personal best
            if E_new < pbest_energies[k]:
                pbests[k] = X_new.copy()
                pbest_energies[k] = E_new
            
            # Update global best
            if E_new < gbest_energy:
                gbest = X_new.copy()
                gbest_energy = E_new
            
            particles[k] = X_new
        
        # Every few iterations, do a local refinement on the best particle
        if it % 5 == 4 and time.time() - start_time < time_budget - 5:
            X_ref, E_ref = _riemannian_cg(gbest, max_iter=50, tol=1e-11,
                                          time_limit=5.0, start_time=time.time())
            if E_ref < gbest_energy:
                gbest = X_ref.copy()
                gbest_energy = E_ref
    
    return gbest, gbest_energy

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    # Start with multiple initial configurations
    X_fib = _fibonacci_sphere(N)
    X_fib = _project_to_sphere(X_fib)
    
    X_dome = _geodesic_dome(N, seed=0)
    X_dome = _project_to_sphere(X_dome)
    
    # Evaluate initial energies
    E_fib = _energy_only(X_fib)
    E_dome = _energy_only(X_dome)
    
    # Start with the better initial configuration
    if E_fib <= E_dome:
        X = X_fib
        best_X = X_fib.copy()
        best_energy = E_fib
    else:
        X = X_dome
        best_X = X_dome.copy()
        best_energy = E_dome
    
    # Phase 1: Riemannian conjugate gradient on best initial
    X_cg, energy_cg = _riemannian_cg(X, max_iter=500, tol=1e-12, 
                                       time_limit=60.0, start_time=start_time)
    if energy_cg < best_energy:
        best_energy = energy_cg
        best_X = X_cg.copy()
    
    # Also try RCG on the other initial if time permits
    other_X = X_dome if E_fib <= E_dome else X_fib
    if time.time() < deadline - 30:
        X_cg2, energy_cg2 = _riemannian_cg(other_X, max_iter=300, tol=1e-11,
                                            time_limit=30.0, start_time=time.time())
        if energy_cg2 < best_energy:
            best_energy = energy_cg2
            best_X = X_cg2.copy()
    
    # Phase 2: PSO exploration
    remaining_time = deadline - time.time()
    if remaining_time > 20:
        pso_budget = min(40.0, remaining_time - 10)
        X_pso, E_pso = _psO_search(best_X, best_energy, pso_budget, start_time)
        if E_pso < best_energy:
            best_energy = E_pso
            best_X = X_pso.copy()
    
    # Phase 3: Random restarts with RCG
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        n_restarts = min(5, int(remaining_time / 15))
        for r in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            X_pert = best_X + 0.05 * np.random.randn(N, 3)
            X_pert = _project_to_sphere(X_pert)
            
            e_pert, _ = _energy_and_grad(X_pert)
            
            if e_pert < best_energy + 5.0:
                X_cg_r, energy_cg_r = _riemannian_cg(X_pert, max_iter=200, tol=1e-11,
                                                       time_limit=12.0, start_time=time.time())
                if energy_cg_r < best_energy:
                    best_energy = energy_cg_r
                    best_X = X_cg_r.copy()
    
    # Phase 4: Final L-BFGS polish
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        X_final = _lbfgs_polish(best_X, time_limit=min(30.0, remaining_time - 2.0),
                                 start_time=start_time)
        e_final = _energy_only(X_final)
        if e_final < best_energy:
            best_energy = e_final
            best_X = X_final.copy()
    
    # Ensure output is valid
    result = best_X.copy()
    result = _project_to_sphere(result)
    
    # Check for very close points
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(result[i] - result[j])
            if d < 1e-9:
                result[i] = result[i] + 1e-8 * np.random.randn(3)
                result[i] = result[i] / np.linalg.norm(result[i])
    
    return result
