import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit (actually 150s, but keep safe)

    # Helper to compute covering radius from points (High Precision, used for final scoring)
    def compute_covering_radius(pts):
        try:
            hull = ConvexHull(pts)
            max_rad = 0.0
            for simplex in hull.simplices:
                v1, v2, v3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
                
                normal = np.cross(v2 - v1, v3 - v1)
                norm = np.linalg.norm(normal)
                if norm < 1e-10:
                    continue
                c_s = normal / norm
                
                # Calculate angular distances from circumcenter to vertices
                d1 = np.arccos(np.clip(np.dot(c_s, v1), -1.0, 1.0))
                d2 = np.arccos(np.clip(np.dot(c_s, v2), -1.0, 1.0))
                d3 = np.arccos(np.clip(np.dot(c_s, v3), -1.0, 1.0))
                
                max_d_circ = max(d1, d2, d3)
                
                # Check edges
                e12 = v1 + v2
                e12_norm = np.linalg.norm(e12)
                if e12_norm > 1e-10:
                    m12 = e12 / e12_norm
                    d_e12 = np.arccos(np.clip(np.dot(m12, v1), -1.0, 1.0))
                else:
                    d_e12 = 0.0
                    m12 = v1
                    
                e23 = v2 + v3
                e23_norm = np.linalg.norm(e23)
                if e23_norm > 1e-10:
                    m23 = e23 / e23_norm
                    d_e23 = np.arccos(np.clip(np.dot(m23, v2), -1.0, 1.0))
                else:
                    d_e23 = 0.0
                    m23 = v2

                e31 = v3 + v1
                e31_norm = np.linalg.norm(e31)
                if e31_norm > 1e-10:
                    m31 = e31 / e31_norm
                    d_e31 = np.arccos(np.clip(np.dot(m31, v3), -1.0, 1.0))
                else:
                    d_e31 = 0.0
                    m31 = v3

                max_d_edge = max(d_e12, d_e23, d_e31)

                # Determine local radius
                # If the circumcenter is "inside" (acute triangle), the radius is max_d_circ.
                # If it's "outside" (obtuse), the radius is max_d_edge.
                if max_d_circ <= np.pi / 2.0:
                    local_rad = max_d_circ
                    center = c_s
                else:
                    local_rad = max_d_edge
                    if d_e12 == max_d_edge:
                        center = m12
                    elif d_e23 == max_d_edge:
                        center = m23
                    else:
                        center = m31

                if local_rad > max_rad:
                    max_rad = local_rad
            return max_rad
        except Exception:
            return np.inf

    def compute_facet_data(pts):
        """
        Computes facet data for gradient calculation.
        Returns:
        - max_rad: maximum covering radius
        - grads: array of shape (N, 3) containing the Riemannian gradient for each point
        - active_mask: boolean array indicating which points are part of active facets (near max radius)
        """
        try:
            hull = ConvexHull(pts)
            max_rad = 0.0
            grads = np.zeros_like(pts)
            active_mask = np.zeros(N, dtype=bool)
            
            facet_radii = []
            facet_centers = []
            facet_indices = []
            
            for simplex in hull.simplices:
                v1, v2, v3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
                
                normal = np.cross(v2 - v1, v3 - v1)
                norm = np.linalg.norm(normal)
                if norm < 1e-10:
                    continue
                c_s = normal / norm
                
                d1 = np.arccos(np.clip(np.dot(c_s, v1), -1.0, 1.0))
                d2 = np.arccos(np.clip(np.dot(c_s, v2), -1.0, 1.0))
                d3 = np.arccos(np.clip(np.dot(c_s, v3), -1.0, 1.0))
                
                max_d_circ = max(d1, d2, d3)
                
                e12 = v1 + v2
                e12_norm = np.linalg.norm(e12)
                if e12_norm > 1e-10:
                    m12 = e12 / e12_norm
                    d_e12 = np.arccos(np.clip(np.dot(m12, v1), -1.0, 1.0))
                else:
                    d_e12 = 0.0
                    m12 = v1
                    
                e23 = v2 + v3
                e23_norm = np.linalg.norm(e23)
                if e23_norm > 1e-10:
                    m23 = e23 / e23_norm
                    d_e23 = np.arccos(np.clip(np.dot(m23, v2), -1.0, 1.0))
                else:
                    d_e23 = 0.0
                    m23 = v2

                e31 = v3 + v1
                e31_norm = np.linalg.norm(e31)
                if e31_norm > 1e-10:
                    m31 = e31 / e31_norm
                    d_e31 = np.arccos(np.clip(np.dot(m31, v3), -1.0, 1.0))
                else:
                    d_e31 = 0.0
                    m31 = v3

                max_d_edge = max(d_e12, d_e23, d_e31)

                if max_d_circ <= np.pi / 2.0:
                    local_rad = max_d_circ
                    center = c_s
                else:
                    local_rad = max_d_edge
                    if d_e12 == max_d_edge:
                        center = m12
                    elif d_e23 == max_d_edge:
                        center = m23
                    else:
                        center = m31

                if local_rad > max_rad:
                    max_rad = local_rad
                    
                facet_radii.append(local_rad)
                facet_centers.append(center)
                facet_indices.append(simplex)

            if not facet_radii:
                return np.inf, np.zeros_like(pts), np.zeros(N, dtype=bool)
                
            facet_radii = np.array(facet_radii)
            
            # Identify active facets (those within a small epsilon of the max radius)
            # Using a soft threshold to include multiple facets for smoother gradient
            epsilon = 0.01 # radians
            active_facs = facet_radii > (max_rad - epsilon)
            
            # Weight active facets by their proximity to max radius
            # We want to minimize the max, so we focus on the worst ones.
            # A simple approach: weight = 1 if active, else 0.
            # A better approach: weight = exp(-k * (max_rad - r_i))
            # Let's use a simple binary mask first, or a soft max weight.
            # For stability, let's use weights proportional to how close they are to the max.
            weights = np.zeros_like(facet_radii)
            for i, r in enumerate(facet_radii):
                if r > max_rad - epsilon:
                    # Linear weight: 1.0 at max, 0.0 at max-epsilon
                    weights[i] = (r - (max_rad - epsilon)) / epsilon
            
            # Normalize weights
            sum_w = np.sum(weights)
            if sum_w < 1e-10:
                # Fallback to just the max facet
                max_idx = np.argmax(facet_radii)
                weights = np.zeros_like(facet_radii)
                weights[max_idx] = 1.0
            else:
                weights /= sum_w

            # Accumulate gradients
            for i, w in enumerate(weights):
                if w < 1e-6:
                    continue
                center = facet_centers[i]
                simplex = facet_indices[i]
                for idx in simplex:
                    v = pts[idx]
                    # Riemannian gradient direction towards center
                    # We want to move v towards center to reduce distance.
                    # The direction of steepest ascent of distance is away from center.
                    # So gradient of radius w.r.t v is roughly (v - proj_v(center)) / dist
                    # Wait, we want to MINIMIZE radius.
                    # Radius is distance from center to v.
                    # d(dist)/d(v) = (v - center) / dist (in Euclidean, projected to tangent)
                    # So gradient direction is (v - center) projected to tangent.
                    # We want to move in -gradient direction.
                    # -gradient = (center - v) projected to tangent.
                    
                    # Project (center - v) onto tangent space at v
                    # grad_dir = center - (center . v) v
                    grad_dir = center - np.dot(center, v) * v
                    grad_norm = np.linalg.norm(grad_dir)
                    if grad_norm > 1e-10:
                        grads[idx] += w * grad_dir / grad_norm
                    active_mask[idx] = True

            return max_rad, grads, active_mask
        except Exception:
            return np.inf, np.zeros_like(pts), np.zeros(N, dtype=bool)

    # Generate initial points
    def get_initial_points(seed):
        np.random.seed(seed)
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        # Add some randomness to break symmetry
        points += np.random.normal(0, 0.1, points.shape)
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        return points

    best_points = None
    best_radius = np.inf

    # Multi-start strategy
    num_restarts = 5
    time_per_restart = (time_limit - 5.0) / num_restarts

    for restart in range(num_restarts):
        if time.time() - start_time > time_limit - 5.0:
            break
            
        pts = get_initial_points(restart)
        current_rad, grads, active_mask = compute_facet_data(pts)
        
        if current_rad < best_radius:
            best_radius = current_rad
            best_points = pts.copy()

        lr = 0.05
        min_lr = 0.0001
        patience = 0
        max_patience = 50
        
        # Riemannian Gradient Flow
        while time.time() - start_time < time_limit - 5.0 and patience < max_patience:
            new_rad, grads, active_mask = compute_facet_data(pts)
            
            if new_rad < current_rad:
                current_rad = new_rad
                patience = 0
                if current_rad < best_radius:
                    best_radius = current_rad
                    best_points = pts.copy()
            else:
                patience += 1
                lr *= 0.9
                if lr < min_lr:
                    # Random perturbation to escape
                    pts += np.random.normal(0, 0.05, pts.shape)
                    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
                    current_rad, grads, active_mask = compute_facet_data(pts)
                    lr = 0.05
                    patience = 0
                    continue

            # Apply gradient step
            # Normalize gradients to prevent large steps
            grad_norms = np.linalg.norm(grads, axis=1, keepdims=True)
            grad_norms[grad_norms < 1e-10] = 1.0
            normalized_grads = grads / grad_norms
            
            # Only move active points
            new_pts = pts.copy()
            for i in range(N):
                if active_mask[i]:
                    # Move in negative gradient direction
                    step = lr * normalized_grads[i]
                    new_v = pts[i] - step
                    new_v /= np.linalg.norm(new_v)
                    new_pts[i] = new_v
            
            pts = new_pts

    # Final polish: Check if the best found is valid and return
    if best_points is None:
        # Fallback to Fibonacci if something went wrong
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
    # Ensure points are on unit sphere
    best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points

# EVOLVE-BLOCK-END
