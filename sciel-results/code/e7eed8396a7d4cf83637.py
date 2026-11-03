"""Logarithmic energy minimization of 150 points on the unit sphere."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150
DEADLINE = 140.0  # seconds, safety margin before 150s kill


def _log_energy_and_grad(pts):
    """Compute log(1/|x_i - x_j|) energy sum over i<j and gradient.
    
    pts: (N, 3) array on unit sphere.
    Returns: (energy, grad) where grad is (N, 3).
    """
    n = pts.shape[0]
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist = np.sqrt(np.sum(diff * diff, axis=2))  # (n, n)
    
    # Mask diagonal
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d = dist[mask]  # (n*(n-1)/2,)
    
    # Energy: sum of log(1/d) = -sum(log(d))
    energy = np.sum(np.log(1.0 / d))
    
    # Gradient: dE/dx_i = -sum_{j != i} (x_i - x_j)/|x_i - x_j|^2
    # = sum_{j != i} (x_j - x_i)/|x_i - x_j|^2
    inv_d2 = np.zeros_like(dist)
    np.fill_diagonal(inv_d2, 0.0)
    inv_d2[mask] = 1.0 / (d ** 2)
    
    # grad[i] = sum_j inv_d2[i,j] * (pts[j] - pts[i])
    # = sum_j inv_d2[i,j] * pts[j] - pts[i] * sum_j inv_d2[i,j]
    row_sum = inv_d2.sum(axis=1)  # (n,)
    grad = inv_d2 @ pts - pts * row_sum[:, np.newaxis]
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _fibonacci_sphere(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_uniform_sphere(n, seed=None):
    """Generate n random points on unit sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    # Generate random directions
    angles = rng.uniform(0, 2 * np.pi, n)
    cos_z = rng.uniform(-1, 1, n)
    sin_z = np.sqrt(1 - cos_z ** 2)
    
    pts = np.column_stack([
        sin_z * np.cos(angles),
        sin_z * np.sin(angles),
        cos_z
    ])
    return _project_to_sphere(pts)


def _geodesic_dome_init(n_target, subdivisions=4):
    """Generate points on sphere using geodesic dome (icosahedron subdivision)."""
    # Start with icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    vertices = _project_to_sphere(vertices)
    
    # Faces of icosahedron
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    # Subdivide faces
    for _ in range(subdivisions):
        new_faces = []
        for face in faces:
            a, b, c = face
            # Midpoints
            ab = _project_to_sphere((vertices[a] + vertices[b]) / 2.0)[0]
            bc = _project_to_sphere((vertices[b] + vertices[c]) / 2.0)[0]
            ca = _project_to_sphere((vertices[c] + vertices[a]) / 2.0)[0]
            
            # Add new vertices
            vertices = np.vstack([vertices, [ab, bc, ca]])
            new_a = len(vertices) - 3
            new_b = len(vertices) - 2
            new_c = len(vertices) - 1
            
            # Create 4 sub-faces
            new_faces.extend([
                [a, new_a, new_c],
                [b, new_b, new_a],
                [c, new_c, new_b],
                [new_a, new_b, new_c]
            ])
        faces = new_faces
    
    # If we have more points than needed, randomly select
    if len(vertices) > n_target:
        indices = np.random.choice(len(vertices), n_target, replace=False)
        vertices = vertices[indices]
    elif len(vertices) < n_target:
        # Pad with random points
        extra = _random_uniform_sphere(n_target - len(vertices))
        vertices = np.vstack([vertices, extra])
    
    return vertices


def _local_minimize(pts, max_iter=500, tol=1e-12):
    """Use L-BFGS-B to minimize energy on the sphere."""
    
    def energy_fn(x):
        p = _project_to_sphere(x.reshape(-1, 3))
        e, _ = _log_energy_and_grad(p)
        return e
    
    def grad_fn(x):
        p = _project_to_sphere(x.reshape(-1, 3))
        _, g = _log_energy_and_grad(p)
        return g.flatten()
    
    x0 = pts.flatten()
    result = minimize(energy_fn, x0, jac=grad_fn, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol})
    pts_opt = _project_to_sphere(result.x.reshape(-1, 3))
    return pts_opt


def _basin_hopping(pts, n_restarts=3, max_iter=300, tol=1e-10, t0=1.0, seed=None):
    """Basin hopping: random perturbations + local minimization."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    best_pts = pts.copy()
    best_energy, _ = _log_energy_and_grad(pts)
    
    for restart in range(n_restarts):
        # Random perturbation
        noise = rng.randn(*pts.shape) * t0
        perturbed = pts + noise
        perturbed = _project_to_sphere(perturbed)
        
        # Local minimization
        refined = _local_minimize(perturbed, max_iter=max_iter, tol=tol)
        e_ref, _ = _log_energy_and_grad(refined)
        
        if e_ref < best_energy:
            best_energy = e_ref
            best_pts = refined.copy()
        
        # Anneal: reduce perturbation size
        t0 *= 0.7
    
    return best_pts


def _simulated_annealing(pts, n_steps=2000, t_start=0.1, t_end=0.001, seed=None):
    """Simulated annealing on the sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    best_pts = pts.copy()
    best_energy, _ = _log_energy_and_grad(pts)
    current_energy = best_energy
    
    for step in range(n_steps):
        # Temperature schedule
        t = t_start * (t_end / t_start) ** (step / n_steps)
        
        # Random perturbation
        noise = rng.randn(*pts.shape) * np.sqrt(t)
        perturbed = pts + noise
        perturbed = _project_to_sphere(perturbed)
        
        e_new, _ = _log_energy_and_grad(perturbed)
        delta_e = e_new - current_energy
        
        # Accept if better or with probability exp(-delta_e/t)
        if delta_e < 0 or rng.rand() < math.exp(-delta_e / t):
            pts = perturbed
            current_energy = e_new
            
            if e_new < best_energy:
                best_energy = e_new
                best_pts = pts.copy()
    
    return best_pts


def _final_polish(pts, max_iter=1000, tol=1e-14):
    """High-precision final polish."""
    return _local_minimize(pts, max_iter=max_iter, tol=tol)


def construct():
    t_start = time.time()
    
    # Generate multiple initializations
    initializations = []
    
    # 1. Fibonacci sphere
    fib_pts = _fibonacci_sphere(N)
    initializations.append(fib_pts)
    
    # 2. Geodesic dome
    dome_pts = _geodesic_dome_init(N, subdivisions=4)
    initializations.append(dome_pts)
    
    # 3. Random points (multiple seeds)
    for seed in [42, 123, 456]:
        rand_pts = _random_uniform_sphere(N, seed=seed)
        initializations.append(rand_pts)
    
    # Local minimize each initialization
    best_pts = None
    best_energy = np.inf
    
    for i, init_pts in enumerate(initializations):
        if time.time() - t_start > DEADLINE - 60:
            break
        
        # Local minimization
        refined = _local_minimize(init_pts, max_iter=300, tol=1e-10)
        e_ref, _ = _log_energy_and_grad(refined)
        
        if e_ref < best_energy:
            best_energy = e_ref
            best_pts = refined.copy()
    
    # Basin hopping from best
    if time.time() - t_start < DEADLINE - 30:
        bh_pts = _basin_hopping(best_pts, n_restarts=5, max_iter=200, tol=1e-10, t0=0.5, seed=42)
        e_bh, _ = _log_energy_and_grad(bh_pts)
        
        if e_bh < best_energy:
            best_energy = e_bh
            best_pts = bh_pts
    
    # Simulated annealing
    if time.time() - t_start < DEADLINE - 20:
        sa_pts = _simulated_annealing(best_pts, n_steps=1000, t_start=0.05, t_end=0.001, seed=42)
        e_sa, _ = _log_energy_and_grad(sa_pts)
        
        if e_sa < best_energy:
            best_energy = e_sa
            best_pts = sa_pts
    
    # Final high-precision polish
    if time.time() - t_start < DEADLINE - 10:
        polished = _final_polish(best_pts, max_iter=500, tol=1e-14)
        e_pol, _ = _log_energy_and_grad(polished)
        
        if e_pol < best_energy:
            best_energy = e_pol
            best_pts = polished
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    
    # Check for near-duplicate points
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Nudge point j slightly
                best_pts[j] += 1e-8 * np.random.randn(3)
                best_pts[j] = _project_to_sphere(best_pts[j].reshape(1, 3))[0]
    
    return best_pts
