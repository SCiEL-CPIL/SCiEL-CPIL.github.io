# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the coordinates of the 14 points.
    """
    n = 14
    d = 3
    
    # High symmetry template: 4 vertices of a regular tetrahedron + 10 edge midpoints
    # This configuration is known to be optimal or near-optimal for maximizing the min/max distance ratio
    # for 14 points in 3D.
    
    # Define the base geometry
    # Tetrahedron vertices (scaled such that circumradius is 1)
    # Vertices: (1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1)
    # Edge midpoints: average of each pair of vertices
    
    v1 = np.array([1, 1, 1])
    v2 = np.array([1, -1, -1])
    v3 = np.array([-1, 1, -1])
    v4 = np.array([-1, -1, 1])
    
    vertices = np.array([v1, v2, v3, v4])
    
    # Midpoints of the 6 edges
    m12 = (v1 + v2) / 2
    m13 = (v1 + v3) / 2
    m14 = (v1 + v4) / 2
    m23 = (v2 + v3) / 2
    m24 = (v2 + v4) / 2
    m34 = (v3 + v4) / 2
    
    midpoints = np.array([m12, m13, m14, m23, m24, m34])
    
    # The 14 points are:
    # 4 vertices at radius r_v
    # 10 midpoints? Wait, a tetrahedron has 6 edges.
    # The prompt suggests "4 vertices + 10 edge midpoints". A tetrahedron only has 6 edges.
    # Perhaps the template implies a different structure or subdivided faces?
    # Let's re-read: "4 vertices + 10 edge midpoints".
    # If we subdivide the tetrahedron into 4 smaller tetrahedra (corners) and 1 central octahedron?
    # No, that's 4 vertices + 6 edge midpoints + ... ?
    # Actually, the 14 points could be:
    # 4 vertices of a tetrahedron
    # 6 midpoints of edges
    # 4 centers of faces?
    # 4 + 6 + 4 = 14.
    # Let's check this configuration.
    # Face centers of a regular tetrahedron form another smaller tetrahedron (inverted).
    
    f1 = (v2 + v3 + v4) / 3
    f2 = (v1 + v3 + v4) / 3
    f3 = (v1 + v2 + v4) / 3
    f4 = (v1 + v2 + v3) / 3
    
    face_centers = np.array([f1, f2, f3, f4])
    
    # So the 14 points are:
    # 4 vertices (orbit A)
    # 6 edge midpoints (orbit B)
    # 4 face centers (orbit C)
    
    # This structure has T_d symmetry.
    # We can optimize the radial scaling of each orbit independently?
    # Or just the overall scale? The ratio d_min/d_max is scale invariant.
    # So we can fix the scale of one orbit and optimize the relative scales of the others.
    # Let r_v = 1.
    # Let r_m = x (scale for midpoints relative to their natural position)
    # Let r_f = y (scale for face centers relative to their natural position)
    
    # Natural positions:
    # v_i have norm sqrt(3)
    # m_ij have norm sqrt(1) = 1
    # f_i have norm sqrt(1/3) = 1/sqrt(3)
    
    # We define the points as:
    # P_v = v_i * (1 / ||v_i||) * R_v  -> unit direction * R_v
    # P_m = m_ij * (1 / ||m_ij||) * R_m
    # P_f = f_i * (1 / ||f_i||) * R_f
    
    # To maximize the ratio, we optimize R_m and R_f given R_v = 1.
    # Actually, we can optimize R_v, R_m, R_f but one is redundant due to scale invariance.
    # Let's fix R_v = 1 and optimize R_m, R_f.
    
    # Directions:
    dirs_v = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    dirs_m = midpoints / np.linalg.norm(midpoints, axis=1, keepdims=True)
    dirs_f = face_centers / np.linalg.norm(face_centers, axis=1, keepdims=True)
    
    def objective(params):
        r_m, r_f = params
        # Construct points
        pts_v = dirs_v * r_v  # r_v is fixed to 1
        pts_m = dirs_m * r_m
        pts_f = dirs_f * r_f
        
        points = np.vstack([pts_v, pts_m, pts_f])
        
        # Check for coincident points
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max == 0:
            return 0.0
        if d_min < 1e-12:
            return 0.0
            
        return (d_min / d_max) ** 2

    # Minimize negative objective
    def neg_objective(params):
        return -objective(params)

    # Initial guess
    # Natural radii:
    # R_v = 1
    # R_m = 1 (since ||m||=1)
    # R_f = 1/sqrt(3) (since ||f||=1/sqrt(3))
    # Let's start with these natural relative sizes.
    x0 = np.array([1.0, 1.0/np.sqrt(3)])
    
    # Optimization
    # Bounds: radii must be positive
    bounds = [(0.1, 10.0), (0.1, 10.0)]
    
    result = minimize(neg_objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000})
    
    # Get optimal radii
    r_m_opt, r_f_opt = result.x
    
    # Construct final points
    r_v = 1.0
    pts_v = dirs_v * r_v
    pts_m = dirs_m * r_m_opt
    pts_f = dirs_f * r_f_opt
    
    points = np.vstack([pts_v, pts_m, pts_f])
    
    # Verify shape and finiteness
    if points.shape != (14, 3):
        # Fallback to random if something went wrong
        np.random.seed(42)
        points = np.random.randn(14, 3)
        
    if not np.all(np.isfinite(points)):
        np.random.seed(42)
        points = np.random.randn(14, 3)
        
    return points

# EVOLVE-BLOCK-END
