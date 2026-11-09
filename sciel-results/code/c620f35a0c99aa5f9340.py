# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist

def _evaluate(points: np.ndarray) -> float:
    """Compute the metric (d_min / d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    dists = pdist(points)
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio

def _normalize(points: np.ndarray) -> np.ndarray:
    """Normalize points to lie on the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1.0
    return points / norms

def _icosahedron_vertices() -> np.ndarray:
    """Return the 12 vertices of a regular icosahedron on the unit sphere."""
    phi = (1 + np.sqrt(5)) / 2
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    return _normalize(vertices)

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses Simulated Annealing on the unit sphere.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin for 600s limit
    
    # Initial configuration: Icosahedron + 2 poles
    ico = _icosahedron_vertices()
    p1 = np.array([0.0, 0.0, 1.0])
    p2 = np.array([0.0, 0.0, -1.0])
    
    # We need 14 points. Icosahedron has 12.
    # Adding poles creates a configuration with high symmetry.
    initial_points = np.vstack([ico, p1.reshape(1, -1), p2.reshape(1, -1)])
    initial_points = _normalize(initial_points)
    
    best_points = initial_points.copy()
    best_metric = _evaluate(best_points)
    
    # Simulated Annealing Parameters
    # We operate on the unit sphere, so scale is fixed.
    # The metric is scale invariant, so we can fix points to unit sphere.
    # However, d_max is also scale invariant in the ratio, so fixing to unit sphere is valid 
    # and simplifies the search space to 42 dims (actually 42-1=41 effective due to rotation invariance, 
    # but we don't need to enforce rotation invariance explicitly, just that the metric is invariant).
    
    # Note: The metric (d_min/d_max)^2 is invariant under scaling and rotation.
    # By fixing points to the unit sphere, we remove the scaling degree of freedom.
    # Rotation invariance means the landscape has flat directions, but SA handles this okay 
    # because random moves will eventually break symmetry if it helps, or just wander.
    
    T = 1.0  # Initial temperature (in terms of metric difference)
    T_min = 1e-6
    alpha = 0.9995  # Cooling rate
    # We need to estimate how many steps we can afford.
    # Let's run a fixed number of steps or until time runs out.
    
    rng = np.random.RandomState(42)
    
    current_points = best_points.copy()
    current_metric = best_metric
    
    # Precompute some things? No, pdist is fast enough for 14 points (91 distances).
    
    step_count = 0
    
    while time.time() < deadline - 10:
        # Generate a neighbor
        # Perturb each point with a small Gaussian noise
        # Step size should be related to temperature? 
        # Or fixed small step size?
        # Let's use a step size that decays slightly with temperature to refine.
        # But on the sphere, large steps can jump to other basins.
        # Let's try a base step size of 0.1 and scale it by sqrt(T) or similar?
        # Actually, for SA on a bounded domain (sphere), a fixed small step is often safer.
        # Let's use 0.05 * (T + 0.1) to allow larger moves early on.
        
        step_size = 0.05 * (1.0 + T)
        
        new_points = current_points + rng.randn(14, 3) * step_size
        new_points = _normalize(new_points)
        
        new_metric = _evaluate(new_points)
        
        # Acceptance criterion
        delta_metric = new_metric - current_metric
        
        if delta_metric > 0:
            # Better, always accept
            current_points = new_points
            current_metric = new_metric
        else:
            # Worse, accept with probability exp(delta_metric / T)
            # Note: delta_metric is negative.
            # If T is small, exp(large negative) is ~0.
            prob = np.exp(delta_metric / T) if T > 0 else 0.0
            if rng.rand() < prob:
                current_points = new_points
                current_metric = new_metric
        
        # Update best
        if current_metric > best_metric:
            best_metric = current_metric
            best_points = current_points.copy()
            
        # Cool down
        T *= alpha
        
        step_count += 1
        
        # Restart if stuck? 
        # If we haven't improved in a long time, maybe kick it?
        # For now, simple SA.
        
        # Check time more frequently in tight loops? 
        # The loop body is fast (pdist on 14x3 is ~microseconds).
        # So we can do many iterations.
        
    # Final polish: Local search with very small steps?
    # Or just return best.
    
    # Ensure finite
    if not np.all(np.isfinite(best_points)):
        best_points = initial_points
        
    return best_points

# EVOLVE-BLOCK-END
