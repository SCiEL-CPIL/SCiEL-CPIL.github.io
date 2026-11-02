import numpy as np
from scipy.spatial import ConvexHull
import random
import time

N = 13

# EVOLVE-BLOCK-START
def construct():
    n = N
    
    # Precompute triangle indices
    tri_indices = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                tri_indices.append((i, j, k))
    tri_indices = np.array(tri_indices)
    n_tris = len(tri_indices)
    
    def get_score(pts):
        """Calculate minimum triangle area / convex hull area"""
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume
            if hull_area <= 1e-10:
                return 0.0
        except:
            return 0.0
            
        p1 = pts[tri_indices[:, 0]]
        p2 = pts[tri_indices[:, 1]]
        p3 = pts[tri_indices[:, 2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        return min_area / hull_area

    def compute_areas(pts):
        """Compute all triangle areas (unnormalized)"""
        p1 = pts[tri_indices[:, 0]]
        p2 = pts[tri_indices[:, 1]]
        p3 = pts[tri_indices[:, 2]]
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        return 0.5 * np.abs(cross)

    def force_based_optimization(pts, max_iter=1000, start_time=0, max_time=100):
        """
        Deterministic force-directed optimization.
        Pushes vertices of small-area triangles apart to maximize the minimum area.
        """
        pts = pts.copy()
        # Center and normalize initial scale
        pts -= np.mean(pts, axis=0)
        
        # Adaptive step size
        step_size = 0.1
        best_score = get_score(pts)
        best_pts = pts.copy()
        no_improve = 0
        
        for it in range(max_iter):
            if time.time() - start_time > max_time:
                break
            
            # 1. Compute areas
            areas = compute_areas(pts)
            min_area = np.min(areas)
            
            # 2. Identify worst triangles (those with area < min_area + threshold)
            # We want to push points in the worst triangles apart.
            threshold = min_area + 0.01 * min_area
            worst_mask = areas < threshold
            worst_indices = tri_indices[worst_mask]
            
            if len(worst_indices) == 0:
                # If no "worst" found (all equal?), pick the absolute minimum
                min_idx = np.argmin(areas)
                worst_indices = tri_indices[min_idx:min_idx+1]
            
            # 3. Calculate forces
            # Force on point i from triangle (i, j, k) to increase area.
            # Area = 0.5 * | (Pj-Pi) x (Pk-Pi) |
            # Gradient w.r.t Pi: -0.5 * ( (Pj-Pi) x (Pk-Pi) ) / Area * (Pk-Pi)_perp ?
            # Simpler: Move Pi in direction that increases cross product.
            # Let v1 = Pj-Pi, v2 = Pk-Pi. Cross = v1_x v2_y - v1_y v2_x.
            # d(Cross)/d(Pi_x) = -v2_y, d(Cross)/d(Pi_y) = v2_x.
            # So grad w.r.t Pi is (-v2_y, v2_x).
            # To increase |Cross|, move Pi in direction of sign(Cross) * grad.
            
            forces = np.zeros_like(pts)
            
            for idx in worst_indices:
                i, j, k = idx
                pi = pts[i]
                pj = pts[j]
                pk = pts[k]
                
                v1 = pj - pi
                v2 = pk - pi
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                
                if abs(cross) < 1e-9:
                    continue
                    
                sign = 1.0 if cross > 0 else -1.0
                
                # Gradient of area w.r.t Pi
                # Area = 0.5 * cross
                # d(Area)/d(Pi) = 0.5 * (-v2_y, v2_x) * sign
                grad_i = np.array([-v2[1], v2[0]]) * sign * 0.5
                
                # Gradient w.r.t Pj
                # v1 = Pj-Pi. d(v1)/d(Pj) = I.
                # Cross = v1_x v2_y - v1_y v2_x
                # d(Cross)/d(Pj_x) = v2_y, d(Cross)/d(Pj_y) = -v2_x
                grad_j = np.array([v2[1], -v2[0]]) * sign * 0.5
                
                # Gradient w.r.t Pk
                # v2 = Pk-Pi. d(v2)/d(Pk) = I.
                # Cross = v1_x v2_y - v1_y v2_x
                # d(Cross)/d(Pk_x) = -v1_y, d(Cross)/d(Pk_y) = v1_x
                grad_k = np.array([-v1[1], v1[0]]) * sign * 0.5
                
                # Add forces
                forces[i] += grad_i
                forces[j] += grad_j
                forces[k] += grad_k
            
            # 4. Add global repulsion to prevent overlap/clustering
            # Weak repulsion between all pairs
            repulsion = np.zeros_like(pts)
            for i in range(n):
                for j in range(i+1, n):
                    diff = pts[i] - pts[j]
                    dist_sq = np.dot(diff, diff)
                    if dist_sq < 1e-6:
                        # Random kick if too close
                        diff = np.random.randn(2) * 0.01
                        dist_sq = 0.01
                    # Inverse square law repulsion
                    force_mag = 0.01 / (dist_sq + 0.1)
                    repulsion[i] += diff * force_mag
                    repulsion[j] -= diff * force_mag
            
            # 5. Add centering force to prevent drift
            centering = -0.01 * pts
            
            # 6. Update points
            # Normalize forces
            force_norm = np.linalg.norm(forces)
            if force_norm > 0:
                forces = forces / force_norm
            
            # Combined update
            # Scale factor
            update = forces * step_size + repulsion * step_size * 0.5 + centering * step_size * 0.1
            
            new_pts = pts + update
            
            # Center and normalize scale to keep numbers stable
            new_pts -= np.mean(new_pts, axis=0)
            
            # Check score
            new_score = get_score(new_pts)
            
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
                no_improve = 0
                # Reduce step size slowly
                step_size *= 0.99
            else:
                no_improve += 1
                # Reduce step size faster if stuck
                step_size *= 0.95
                if no_improve > 50:
                    # Restart with slightly perturbed best
                    pts = best_pts + np.random.randn(n, 2) * 0.05
                    step_size = 0.1
                    no_improve = 0
                    continue
                
            pts = new_pts
            
            # Periodic normalization to unit area to help score calculation consistency
            try:
                hull = ConvexHull(pts)
                area = hull.volume
                if area > 1e-10:
                    pts /= np.sqrt(area)
            except:
                pass
        
        return best_pts, best_score

    def generate_initial_points(method='fermat'):
        """Generate diverse initial point configurations"""
        if method == 'fermat':
            i = np.arange(n)
            r = np.sqrt(i / n)
            theta = i * 2.399963229728653  # Golden angle
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points = np.column_stack((x, y))
        elif method == 'circle':
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            r = np.ones(n)
            x = r * np.cos(angles)
            y = r * np.sin(angles)
            points = np.column_stack((x, y))
        elif method == 'grid':
            # 4x4 grid with one point removed, then adjust
            xs = np.array([0, 1, 2, 3])
            ys = np.array([0, 1, 2, 3])
            grid = [(x, y) for x in xs for y in ys]
            random.shuffle(grid)
            points = np.array(grid[:n], dtype=float)
            points -= points.mean(axis=0)
        else:  # random
            points = np.random.randn(n, 2)
            points -= points.mean(axis=0)
            points /= np.max(np.abs(points))
        
        # Normalize to unit area
        try:
            hull = ConvexHull(points)
            area = hull.volume
            if area > 1e-10:
                points /= np.sqrt(area)
        except:
            pass
        
        return points

    start_time = time.time()
    max_time = 100  # Leave 20s buffer
    
    best_points = None
    best_score = -np.inf
    
    # Try multiple initial configurations
    init_methods = ['fermat', 'circle', 'random']
    
    for idx, method in enumerate(init_methods):
        elapsed = time.time() - start_time
        if elapsed > max_time * 0.8:
            break
        
        points = generate_initial_points(method)
        
        # Run force-based optimization
        fb_pts, fb_score = force_based_optimization(points, max_iter=2000, start_time=start_time, max_time=max_time)
        
        if fb_score > best_score:
            best_score = fb_score
            best_points = fb_pts.copy()
            
        # Check time
        elapsed = time.time() - start_time
        if elapsed > max_time * 0.9:
            break
    
    if best_points is None:
        # Fallback
        i = np.arange(n)
        r = np.sqrt(i / n)
        theta = i * 2.399963229728653
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        best_points = np.column_stack((x, y))
        best_points -= np.mean(best_points, axis=0)
        
    return best_points
# EVOLVE-BLOCK-END
