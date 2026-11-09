# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Maximizes (d_min / d_max)^2 for 14 points in 3D.
    
    Strategy:
    1. Generate diverse initial configurations based on known high-symmetry polyhedra
       and their extensions (Icosahedron+2, Octahedron+8, etc.).
    2. For each configuration, perform a local optimization in the full 42D Cartesian
       space using Nelder-Mead to maximize the metric. This allows the points to 
       break symmetry if beneficial, which is often necessary for the true optimum.
    3. Use a time-budgeted approach to refine the best candidates.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin

    def evaluate_config(points):
        """
        Takes an array of shape (14, 3), centers, normalizes, and returns the metric.
        Returns (points_normalized, metric_value).
        """
        if not np.all(np.isfinite(points)):
            return None, 0.0
            
        pts = points.copy()
        # Center
        pts = pts - np.mean(pts, axis=0)
        
        # Check for coincident points (d_min = 0)
        dists = pdist(pts)
        d_max = np.max(dists)
        
        if d_max < 1e-12:
            return None, 0.0
            
        d_min = np.min(dists)
        
        # Normalize to unit max distance
        pts_norm = pts / d_max
        
        ratio = d_min / d_max
        metric = ratio ** 2
        
        return pts_norm, metric

    def objective_flat(params):
        """
        Objective function for flat 42D array. Returns negative metric.
        """
        try:
            pts = params.reshape((14, 3))
            _, metric = evaluate_config(pts)
            return -metric
        except Exception:
            return 0.0

    def generate_icosahedron_plus_2():
        """
        Icosahedron (12 verts) + 2 poles.
        Icosahedron has high symmetry. Adding poles might disrupt it, but it's a strong start.
        """
        phi = (1 + np.sqrt(5)) / 2
        verts = np.array([
            [0, 1, phi], [0, -1, phi],
            [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0],
            [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1],
            [phi, 0, -1], [-phi, 0, -1]
        ])
        verts = verts / np.linalg.norm(verts, axis=1)[:, np.newaxis]
        # Add poles at (0,0,1) and (0,0,-1) which are already vertices of icosahedron?
        # No, icosahedron vertices are not at poles unless rotated.
        # Let's just add two points at the poles of the coordinate system.
        pole1 = np.array([0, 0, 1])
        pole2 = np.array([0, 0, -1])
        pts = np.vstack([verts, pole1, pole2])
        return pts

    def generate_octahedron_plus_8():
        """
        Octahedron (6 verts) + 8 points (cube vertices).
        This creates a cuboctahedron-like structure if scaled appropriately.
        """
        oct_verts = np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ])
        cube_verts = np.array([
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
        ])
        # Scale cube to have same circumradius as octahedron (which is 1)
        # Cube circumradius is sqrt(3). So scale by 1/sqrt(3)
        cube_verts = cube_verts / np.sqrt(3)
        pts = np.vstack([oct_verts, cube_verts])
        return pts

    def generate_two_rings_6_6():
        """
        Two rings of 6 points, twisted.
        """
        angles1 = np.linspace(0, 2*np.pi, 6, endpoint=False)
        angles2 = angles1 + np.pi/6 # 30 degree twist
        r = 1.0
        z = 0.5
        ring1 = np.column_stack([r*np.cos(angles1), r*np.sin(angles1), np.full(6, z)])
        ring2 = np.column_stack([r*np.cos(angles2), r*np.sin(angles2), np.full(6, -z)])
        return np.vstack([ring1, ring2])

    def generate_tetrahedron_expansion():
        """
        Tetrahedron (4) + 4 edge midpoints + 4 face centroids + 2 poles?
        Just a generic symmetric start.
        """
        # Regular tetrahedron
        t = np.array([
            [1, 1, 1],
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1]
        ])
        t = t / np.sqrt(3)
        
        # Add points at origin? No, need 10 more.
        # Let's add 3 points on each axis?
        # Or just random symmetric perturbation of icosahedron.
        # Let's use a different seed for random symmetric points.
        rng = np.random.RandomState(42)
        # 14 random points, then symmetrize?
        # Better: Start with icosahedron and perturb.
        phi = (1 + np.sqrt(5)) / 2
        verts = np.array([
            [0, 1, phi], [0, -1, phi],
            [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0],
            [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1],
            [phi, 0, -1], [-phi, 0, -1]
        ])
        verts = verts / np.linalg.norm(verts, axis=1)[:, np.newaxis]
        
        # Perturb slightly to break symmetry and see if local opt finds a better spot
        noise = rng.normal(0, 0.1, verts.shape)
        verts_pert = verts + noise
        
        # Add 2 random points near poles
        p1 = np.array([0.1, 0.1, 1.0])
        p2 = np.array([-0.1, -0.1, -1.0])
        pts = np.vstack([verts_pert, p1, p2])
        return pts

    # List of initial configurations
    init_configs = [
        generate_icosahedron_plus_2(),
        generate_octahedron_plus_8(),
        generate_two_rings_6_6(),
        generate_tetrahedron_expansion()
    ]
    
    # Add a few random symmetric configs
    rng = np.random.RandomState(123)
    for i in range(3):
        # Random points on sphere, then center
        pts = rng.normal(0, 1, (14, 3))
        # Normalize to unit sphere
        pts = pts / np.linalg.norm(pts, axis=1)[:, np.newaxis]
        init_configs.append(pts)

    best_points = None
    best_score = 0.0

    # Evaluate initial scores
    for i, config in enumerate(init_configs):
        if time.time() > deadline - 30:
            break
        pts_norm, score = evaluate_config(config)
        if score > best_score:
            best_score = score
            best_points = pts_norm

    # Local refinement for each initial config
    for i, config in enumerate(init_configs):
        if time.time() > deadline - 10:
            break
            
        # Flatten for optimizer
        x0 = config.flatten()
        
        try:
            # Nelder-Mead in 42D can be slow, but 14 points is small enough.
            # Use a reasonable maxiter.
            res = minimize(
                objective_flat,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 500, 
                    'xatol': 1e-8, 
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            # Reconstruct and evaluate
            pts_opt = res.x.reshape((14, 3))
            pts_norm, score = evaluate_config(pts_opt)
            
            if score > best_score:
                best_score = score
                best_points = pts_norm
                
        except Exception:
            pass

    # Fallback if no valid points found or score is 0
    if best_points is None or not np.all(np.isfinite(best_points)) or best_score < 1e-6:
        # Use the icosahedron + 2 poles as a safe fallback
        pts = generate_icosahedron_plus_2()
        best_points, _ = evaluate_config(pts)
        if best_points is None:
             # Absolute fallback: grid
             best_points = np.zeros((14, 3))
             # Just return something finite
             best_points[:, 0] = np.linspace(-1, 1, 14)

    return best_points

# EVOLVE-BLOCK-END
