"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 14
D = 3

# EVOLVE-BLOCK-START

def _pairwise_distances(points):
    """Compute all pairwise distances between points."""
    # points: (n, d)
    # Returns (n, n) matrix of distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    return dists

def _score(points):
    """Compute the squared ratio of max to min pairwise distance."""
    dists = _pairwise_distances(points)
    # Get upper triangle indices (excluding diagonal)
    iu = np.triu_indices(len(points), k=1)
    d = dists[iu]
    d_min = np.min(d)
    d_max = np.max(d)
    if d_min < 1e-12:
        return 1e10
    return (d_max / d_min) ** 2

def _objective(params):
    """Objective function for optimization."""
    points = params.reshape(N, D)
    return _score(points)

def _gradient(params):
    """Numerical gradient of the objective."""
    eps = 1e-7
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        params_minus = params.copy()
        params_minus[i] -= eps
        grad[i] = (_objective(params_plus) - _objective(params_minus)) / (2 * eps)
    return grad

def _tetrahedron_initial():
    """
    Generate initial configuration based on regular tetrahedron vertices
    plus midpoints of edges, then add additional points.
    
    A regular tetrahedron has 4 vertices and 6 edges.
    4 vertices + 6 midpoints = 10 points.
    We need 14, so we add 4 more points.
    """
    # Regular tetrahedron vertices (centered at origin)
    a = 1.0
    verts = np.array([
        [a, a, a],
        [a, -a, -a],
        [-a, a, -a],
        [-a, -a, a]
    ], dtype=float)
    
    # Center at origin
    verts -= np.mean(verts, axis=0)
    
    # Scale to unit edge length for consistency
    edge_len = np.linalg.norm(verts[0] - verts[1])
    verts /= edge_len
    
    # Edge midpoints
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    midpoints = []
    for i, j in edges:
        midpoints.append((verts[i] + verts[j]) / 2)
    midpoints = np.array(midpoints)
    
    # We have 4 + 6 = 10 points. Need 4 more.
    # Add points at face centers (4 faces)
    faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    face_centers = []
    for i, j, k in faces:
        fc = (verts[i] + verts[j] + verts[k]) / 3
        face_centers.append(fc)
    face_centers = np.array(face_centers)
    
    # Combine: 4 vertices + 6 midpoints + 4 face centers = 14 points
    points = np.vstack([verts, midpoints, face_centers])
    
    # Center the whole configuration
    points -= np.mean(points, axis=0)
    
    return points

def construct():
    """Construct 14 points in 3D to minimize squared max/min distance ratio."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Initial configuration from tetrahedron-based construction
    points0 = _tetrahedron_initial()
    best_points = points0.copy()
    best_score = _score(best_points)
    
    # Try multiple random initializations and local optimizations
    np.random.seed(42)
    
    # Also try a few random initializations
    n_attempts = 0
    while time.time() < deadline - 10.0:
        if n_attempts == 0:
            init_points = points0
        else:
            # Random initialization with some structure
            # Start with tetrahedron-based and add noise
            noise_scale = 0.1 * n_attempts
            init_points = points0 + np.random.randn(N, D) * noise_scale
            init_points -= np.mean(init_points, axis=0)
        
        params0 = init_points.flatten()
        
        # Use Nelder-Mead for derivative-free optimization
        result = minimize(
            _objective,
            params0,
            method='Nelder-Mead',
            options={
                'maxiter': 5000,
                'xatol': 1e-10,
                'fatol': 1e-10,
                'adaptive': True
            }
        )
        
        if result.success or result.fun < best_score:
            candidate = result.x.reshape(N, D)
            # Ensure distinctness
            dists = _pairwise_distances(candidate)
            iu = np.triu_indices(N, k=1)
            if np.min(dists[iu]) > 1e-8:
                candidate_score = _score(candidate)
                if candidate_score < best_score:
                    best_score = candidate_score
                    best_points = candidate.copy()
        
        n_attempts += 1
        
        # Check time
        if time.time() > deadline - 15.0:
            break
        
        # If we've done a few attempts, try a different strategy
        if n_attempts == 3:
            # Try BFGS with gradient
            for trial in range(3):
                if time.time() > deadline - 10.0:
                    break
                noise_scale = 0.05 * (trial + 1)
                init_points = points0 + np.random.randn(N, D) * noise_scale
                init_points -= np.mean(init_points, axis=0)
                params0 = init_points.flatten()
                
                result = minimize(
                    _objective,
                    params0,
                    jac=_gradient,
                    method='L-BFGS-B',
                    options={
                        'maxiter': 2000,
                        'gtol': 1e-12
                    }
                )
                
                if result.success or result.fun < best_score:
                    candidate = result.x.reshape(N, D)
                    dists = _pairwise_distances(candidate)
                    iu = np.triu_indices(N, k=1)
                    if np.min(dists[iu]) > 1e-8:
                        candidate_score = _score(candidate)
                        if candidate_score < best_score:
                            best_score = candidate_score
                            best_points = candidate.copy()
        
        if n_attempts >= 6:
            break
    
    # Final check: ensure all points are distinct
    dists = _pairwise_distances(best_points)
    iu = np.triu_indices(N, k=1)
    if np.min(dists[iu]) < 1e-8:
        # Fallback to initial
        best_points = points0
    
    return best_points

# EVOLVE-BLOCK-END
