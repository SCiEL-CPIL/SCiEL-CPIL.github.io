"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit
    
    # 1. Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    
    best_pts = pts.copy()
    best_radius = np.inf
    
    # Helper to compute covering radius using convex hull
    def get_covering_radius(points):
        try:
            hull = ConvexHull(points)
            max_ang = 0.0
            for simplex in hull.simplices:
                v1 = points[simplex[0]]
                v2 = points[simplex[1]]
                v3 = points[simplex[2]]
                
                # Compute normal
                edge1 = v2 - v1
                edge2 = v3 - v1
                normal = np.cross(edge1, edge2)
                norm = np.linalg.norm(normal)
                if norm < 1e-10:
                    continue
                normal = normal / norm
                
                # Ensure normal points outward (away from origin)
                if np.dot(normal, v1) < 0:
                    normal = -normal
                    
                # The angular radius is the max angle between normal and vertices
                d1 = np.dot(normal, v1)
                d2 = np.dot(normal, v2)
                d3 = np.dot(normal, v3)
                
                # Clamp for numerical stability
                d1 = np.clip(d1, -1.0, 1.0)
                d2 = np.clip(d2, -1.0, 1.0)
                d3 = np.clip(d3, -1.0, 1.0)
                
                ang1 = np.arccos(d1)
                ang2 = np.arccos(d2)
                ang3 = np.arccos(d3)
                
                ang = max(ang1, ang2, ang3)
                if ang > max_ang:
                    max_ang = ang
                    
            return max_ang
        except Exception:
            return np.pi # Worst case
            
    current_radius = get_covering_radius(pts)
    if current_radius < best_radius:
        best_radius = current_radius
        best_pts = pts.copy()
        
    # Phase 1: Fast gradient-based refinement using Gram matrix / energy minimization
    # We minimize a smooth surrogate for the covering radius: 
    # Sum of exp(beta * d_ij) or similar, but simpler: minimize max distance to nearest neighbor is hard.
    # Instead, we use a potential energy model where points repel each other.
    # Specifically, we minimize the sum of squared distances to the nearest neighbor? No.
    # Let's use the "Thomson problem" style repulsive potential: U = sum_{i != j} 1/d_ij^k
    # But we want to minimize the covering radius, which is related to the largest gap.
    # A good surrogate is the sum of exp(alpha * d_max) where d_max is the max distance to nearest neighbor?
    # Actually, let's just do a simple repulsive force simulation.
    # Force on i is sum_{j != i} (pts[i] - pts[j]) / ||pts[i] - pts[j]||^2
    # This tends to spread points out evenly.
    
    # However, the current program already does a local search on the hull.
    # The inspiration suggests using spectral properties of the Gram matrix to enforce uniformity.
    # This is essentially minimizing the variance of pairwise distances or maximizing the minimum distance.
    # Let's implement a fast gradient descent step on the "energy" E = sum_{i,j} f(||p_i - p_j||)
    # where f is a repulsive function.
    
    # We will alternate between:
    # 1. Fast gradient steps on a smooth energy function (repulsive) to get close to a good configuration.
    # 2. Precise local search on the convex hull to fine-tune the worst facets.
    
    # Let's define the energy function: E = -sum_{i != j} log(||p_i - p_j||)
    # This is the Coulomb energy. Minimizing it maximizes the minimum distance.
    # Gradient of E with respect to p_i:
    # dE/dp_i = sum_{j != i} (p_i - p_j) / ||p_i - p_j||^2
    
    # We need to project the gradient onto the tangent space of the sphere.
    # grad_tangent = grad - (grad . p_i) * p_i
    
    # Let's do this for a few hundred steps to speed up the initial convergence.
    lr = 0.01
    for step in range(500):
        if time.time() > deadline - 20: # Leave time for hull search
            break
            
        # Compute forces
        # P is N x 3
        # D is N x N matrix of squared distances
        D = np.sum(pts**2, axis=1)[:, None] + np.sum(pts**2, axis=1)[None, :] - 2 * np.dot(pts, pts.T)
        D = np.maximum(D, 1e-12) # Avoid division by zero
        inv_D = 1.0 / D
        np.fill_diagonal(inv_D, 0.0)
        
        # Force F_i = sum_j (p_i - p_j) / D_ij
        # F = pts * sum_j inv_D - pts @ inv_D
        # Wait, (p_i - p_j) = p_i - p_j
        # sum_j (p_i - p_j) / D_ij = p_i * sum_j (1/D_ij) - sum_j p_j / D_ij
        # sum_j (1/D_ij) is the row sum of inv_D
        row_sums = np.sum(inv_D, axis=1)
        # sum_j p_j / D_ij is pts @ inv_D
        force = pts * row_sums[:, None] - pts @ inv_D
        
        # Project to tangent space
        # grad . p_i
        dot_grad_p = np.sum(force * pts, axis=1)
        force_tangent = force - dot_grad_p[:, None] * pts
        
        # Update
        pts += lr * force_tangent
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        
        # Check improvement in the actual objective (covering radius) occasionally
        # But computing hull is expensive. Let's just do this for a fixed number of steps
        # and then switch to hull-based search.
        
    # After the fast phase, we might have improved or degraded the covering radius.
    # Let's check and update best if improved.
    current_radius = get_covering_radius(pts)
    if current_radius < best_radius:
        best_radius = current_radius
        best_pts = pts.copy()
    else:
        # If it got worse, revert to best_pts for the local search phase?
        # Or continue from current state?
        # Usually, the repulsive force helps escape local minima of the hull search.
        # But if it degraded significantly, we might want to restart from best_pts.
        # Let's keep the current state if it's not too bad, or revert if it's much worse.
        if current_radius > best_radius * 1.05:
            pts = best_pts.copy()
            current_radius = best_radius

    # Phase 2: Precise local search on the convex hull (similar to original)
    step_size = 0.05
    learning_rate = 1.0
    no_improve_count = 0
    max_no_improve = 20
    
    while time.time() < deadline:
        # Find the worst facet
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
        worst_facet_idx = -1
        max_ang = -1.0
        worst_normal = None
        
        for idx, simplex in enumerate(hull.simplices):
            v1 = pts[simplex[0]]
            v2 = pts[simplex[1]]
            v3 = pts[simplex[2]]
            
            edge1 = v2 - v1
            edge2 = v3 - v1
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal = normal / norm
            if np.dot(normal, v1) < 0:
                normal = -normal
            
            d1 = np.clip(np.dot(normal, v1), -1.0, 1.0)
            d2 = np.clip(np.dot(normal, v2), -1.0, 1.0)
            d3 = np.clip(np.dot(normal, v3), -1.0, 1.0)
            
            ang = max(np.arccos(d1), np.arccos(d2), np.arccos(d3))
            
            if ang > max_ang:
                max_ang = ang
                worst_facet_idx = idx
                worst_normal = normal
        
        if worst_facet_idx == -1:
            break
            
        # Get the vertices of the worst facet
        simplex = hull.simplices[worst_facet_idx]
        v1 = pts[simplex[0]]
        v2 = pts[simplex[1]]
        v3 = pts[simplex[2]]
        
        # Move these points towards the worst_normal (the center of the hole)
        alpha = step_size * learning_rate
        
        # Update points
        pts[simplex[0]] = v1 + alpha * worst_normal
        pts[simplex[1]] = v2 + alpha * worst_normal
        pts[simplex[2]] = v3 + alpha * worst_normal
        
        # Normalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        
        # Check new radius
        new_radius = get_covering_radius(pts)
        
        if new_radius < current_radius:
            current_radius = new_radius
            if current_radius < best_radius:
                best_radius = current_radius
                best_pts = pts.copy()
            no_improve_count = 0
            # Decay step size slowly
            step_size *= 0.99
        else:
            # Revert
            pts[simplex[0]] = v1
            pts[simplex[1]] = v2
            pts[simplex[2]] = v3
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms
            
            # Reduce step size
            step_size *= 0.9
            no_improve_count += 1
            
            if step_size < 1e-5 or no_improve_count > max_no_improve:
                # Try a small random perturbation to escape local minimum?
                # Or just break.
                # Let's try a few random restarts of the step size if stuck.
                if no_improve_count > max_no_improve:
                    step_size = 0.02
                    no_improve_count = 0
                    # Perturb points slightly
                    pts += np.random.randn(*pts.shape) * 0.01
                    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
                    current_radius = get_covering_radius(pts)
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_pts = pts.copy()
                    else:
                        pts = best_pts.copy()
                        current_radius = best_radius
                    max_no_improve = 10 # Reduce patience
                    if max_no_improve < 5:
                        break
                if step_size < 1e-5:
                    break

    return best_pts
# EVOLVE-BLOCK-END
