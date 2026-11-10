import numpy as np
import time
from scipy.spatial.distance import pdist

def _score(points):
    """Compute (d_min / d_max)^2 for a set of points."""
    if points.shape != (14, 3):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _fibonacci_sphere(n):
    points = []
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        r = np.sqrt(1 - z**2)
        theta = 2 * np.pi * i / golden_ratio
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append([x, y, z])
    return np.array(points)

def _random_rotation_matrix(rng):
    """Generate a random rotation matrix using QR decomposition."""
    Q, R = np.linalg.qr(rng.standard_normal((3, 3)))
    d = np.sign(np.diag(R))
    Q = Q * d
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q

def _normalize(points):
    """Center and normalize points to unit sphere."""
    points = points - np.mean(points, axis=0)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1e-12
    points = points / norms
    return points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Maximizes (d_min / d_max)^2 for 14 points in 3D using Simulated Annealing 
    with global rotations and local perturbations.
    """
    start_time = time.time()
    deadline = start_time + 600  # 30s safety margin
    
    n = 14
    rng = np.random.default_rng(42)
    
    # Initialize with Fibonacci sphere
    best_points = _fibonacci_sphere(n)
    best_score = _score(best_points)
    
    # Simulated Annealing parameters
    T = 1.0
    T_min = 1e-6
    alpha = 0.9995
    max_iter = 200000
    
    current_points = best_points.copy()
    current_score = best_score
    
    # Precompute some random rotations for speed? No, generating is fast enough.
    
    for it in range(max_iter):
        if time.time() > deadline - 10:
            break
            
        # Generate a random perturbation
        # 1. Global rotation
        R = _random_rotation_matrix(rng)
        
        # 2. Local perturbation (small noise)
        # Scale of noise decreases with temperature
        noise_scale = 0.1 * T
        perturbation = rng.standard_normal((n, 3)) * noise_scale
        
        # Apply rotation and perturbation
        candidate = (current_points @ R.T) + perturbation
        
        # Re-normalize to sphere
        candidate = _normalize(candidate)
        
        # Evaluate
        candidate_score = _score(candidate)
        
        # Metropolis acceptance criterion
        # We maximize score, so delta = new - old
        delta = candidate_score - current_score
        
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current_points = candidate
            current_score = candidate_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
    
    # Final local polish: simple hill climbing on the best found so far
    # to squeeze out any remaining gains
    polish_points = best_points.copy()
    polish_score = best_score
    
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        
        improved = False
        for i in range(n):
            if time.time() > deadline - 5:
                break
            
            # Try to improve point i by moving it slightly
            # Find direction of maximum clearance
            dists = np.linalg.norm(polish_points - polish_points[i], axis=1)
            dists[i] = np.inf
            nearest = np.argmin(dists)
            
            # Move away from nearest neighbor
            direction = polish_points[i] - polish_points[nearest]
            direction /= np.linalg.norm(direction)
            
            # Try a few steps
            for step in [0.05, 0.1, 0.2, 0.5]:
                candidate_pos = polish_points[i] + step * direction
                candidate_pos = candidate_pos / np.linalg.norm(candidate_pos)
                
                new_points = polish_points.copy()
                new_points[i] = candidate_pos
                
                new_score = _score(new_points)
                
                if new_score > polish_score + 1e-10:
                    polish_points = new_points
                    polish_score = new_score
                    improved = True
                    break
        
        if not improved:
            break
    
    if polish_score > best_score:
        best_score = polish_score
        best_points = polish_points
    
    return best_points
