import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit
    
    # 1. Initialization: Fibonacci Sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = np.inf

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
                
                e23 = v2 + v3
                e23_norm = np.linalg.norm(e23)
                if e23_norm > 1e-10:
                    m23 = e23 / e23_norm
                    d_e23 = np.arccos(np.clip(np.dot(m23, v2), -1.0, 1.0))
                else:
                    d_e23 = 0.0

                e31 = v3 + v1
                e31_norm = np.linalg.norm(e31)
                if e31_norm > 1e-10:
                    m31 = e31 / e31_norm
                    d_e31 = np.arccos(np.clip(np.dot(m31, v3), -1.0, 1.0))
                else:
                    d_e31 = 0.0

                max_d_edge = max(d_e12, d_e23, d_e31)

                # Determine local radius
                # If the circumcenter is "inside" (acute triangle), the radius is max_d_circ.
                # If it's "outside" (obtuse), the radius is max_d_edge.
                if max_d_circ <= np.pi / 2.0:
                    local_rad = max_d_circ
                else:
                    local_rad = max_d_edge

                if local_rad > max_rad:
                    max_rad = local_rad
            return max_rad
        except Exception:
            return np.inf

    def compute_energy(pts):
        # Fast approximate energy using probe points for initial coarse search
        M = 2000
        i_probe = np.arange(M) + 0.5
        z_probe = 1.0 - 2.0 * i_probe / M
        phi_probe = np.pi * (1.0 + np.sqrt(5.0)) * i_probe
        r_probe = np.sqrt(1.0 - z_probe * z_probe)
        probe_points = np.column_stack([r_probe * np.cos(phi_probe), r_probe * np.sin(phi_probe), z_probe])
        
        dots = probe_points @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        max_dots = np.max(dots, axis=1)
        min_dists = np.arccos(max_dots)
        return np.max(min_dists)

    # Phase 1: Coarse Search with Simulated Annealing (kept from seed to ensure good starting basin)
    current_points = points.copy()
    current_energy = compute_energy(current_points)
    
    T = 0.1
    T_min = 0.001
    cooling_rate = 0.999
    step_size = 0.1
    
    n_iter = 0
    while time.time() - start_time < time_limit - 30.0: # Leave 30s for Riemannian Gradient Flow
        idx = np.random.randint(N)
        perturbation = np.random.normal(0, step_size, 3)
        new_point = current_points[idx] + perturbation
        new_point /= np.linalg.norm(new_point)
        
        old_point = current_points[idx].copy()
        current_points[idx] = new_point
        
        new_energy = compute_energy(current_points)
        
        if new_energy < current_energy:
            current_energy = new_energy
            if current_energy < best_radius:
                 best_radius = current_energy
                 best_points = current_points.copy()
        else:
            delta_E = new_energy - current_energy
            if delta_E > 0:
                if np.random.rand() < np.exp(-delta_E / T):
                    current_energy = new_energy
                else:
                    current_points[idx] = old_point
            
        T *= cooling_rate
        
        if time.time() - start_time > time_limit - 30.0:
            break
            
    # Phase 2: Riemannian Gradient Flow / Steepest Descent on the Manifold
    # We directly minimize the exact covering radius by identifying the worst facet
    # and moving its vertices in the direction that reduces the circumradius.
    
    pts = best_points.copy()
    current_rad = compute_covering_radius(pts)
    
    # We use a soft-max surrogate for the objective to get gradients, 
    # but since we are doing discrete steps on the worst facet, 
    # we can use the exact geometry to determine the gradient direction.
    
    # The gradient of the covering radius w.r.t. a vertex v_i of the worst facet 
    # points towards the circumcenter of that facet (if the facet is the binding constraint).
    # Specifically, moving v_i towards the circumcenter c reduces the distance from c to v_i.
    
    # We will iterate:
    # 1. Find worst facet and its circumcenter.
    # 2. Compute Riemannian gradient for vertices involved.
    # 3. Take a step in the negative gradient direction (projected to tangent space).
    
    lr = 0.05
    min_lr = 0.0001
    patience = 0
    best_pts_phase2 = pts.copy()
    best_rad_phase2 = current_rad
    
    t_start_phase2 = time.time()
    
    while time.time() - t_start_phase2 < 25.0:
        try:
            hull = ConvexHull(pts)
            max_rad = 0.0
            worst_simplex = None
            worst_center = None
            worst_rad_val = 0.0
            
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
                
                # Edge midpoints
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
                    worst_simplex = simplex
                    worst_center = center
                    worst_rad_val = local_rad
                    
            if worst_simplex is None:
                break
                
            # Identify the vertices of the worst facet
            v_idx = worst_simplex
            
            # The "gradient" direction for each vertex is towards the worst_center.
            # We want to move the vertices such that the distance to the center decreases.
            # On the sphere, the steepest descent direction towards a target point c from v 
            # is the projection of (c - v) onto the tangent space at v.
            # Tangent space at v: { w | w . v = 0 }
            # proj(v, c-v) = (c - v) - ((c-v) . v) v = c - (c . v) v
            
            # We apply a small step in this direction for each vertex.
            # To avoid breaking the convex hull topology or creating degenerate facets,
            # we use a small learning rate.
            
            new_pts = pts.copy()
            for idx in v_idx:
                v = pts[idx]
                c = worst_center
                # Project (c - v) onto tangent space at v
                # dot_cv = np.dot(c, v)
                # grad = c - dot_cv * v
                # This grad is already in the tangent space because grad . v = c.v - (c.v)(v.v) = 0.
                
                # Normalize gradient for consistent step size?
                grad_norm = np.linalg.norm(worst_center - np.dot(worst_center, v) * v)
                if grad_norm < 1e-10:
                    continue
                    
                # Step
                step = lr * (worst_center - np.dot(worst_center, v) * v) / grad_norm
                new_v = v + step
                new_v /= np.linalg.norm(new_v)
                new_pts[idx] = new_v
            
            # Evaluate new configuration
            new_rad = compute_covering_radius(new_pts)
            
            if new_rad < current_rad:
                pts = new_pts
                current_rad = new_rad
                patience = 0
                if current_rad < best_rad_phase2:
                    best_rad_phase2 = current_rad
                    best_pts_phase2 = pts.copy()
            else:
                patience += 1
                # Reduce learning rate if stuck
                lr *= 0.95
                if lr < min_lr:
                    # Try a random perturbation to escape local minimum
                    pts += np.random.normal(0, 0.01, pts.shape)
                    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
                    current_rad = compute_covering_radius(pts)
                    lr = 0.01
                    patience = 0
                    
                if patience > 100:
                    break
                    
        except Exception:
            break
            
    # Update global best
    if best_rad_phase2 < best_radius:
        best_points = best_pts_phase2
        best_radius = best_rad_phase2
        
    # Final polish: Check if Fibonacci was better (unlikely after SA+RGD, but safe)
    fib_rad = compute_covering_radius(points)
    if fib_rad < best_radius:
        best_points = points
        best_radius = fib_rad

    return best_points

# EVOLVE-BLOCK-END
