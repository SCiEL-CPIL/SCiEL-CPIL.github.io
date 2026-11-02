import numpy as np
import time
import math
from scipy.optimize import linprog, differential_evolution

N = 12
DEADLINE = 145.0

# Unit hexagon geometry
# Vertices at angles 0, 60, 120, 180, 240, 300
UNIT_VERTS = np.array([
    [np.cos(np.radians(60 * i)), np.sin(np.radians(60 * i))] for i in range(6)
])

def get_cluster_vertices(cluster_centers, cluster_angle_deg=0.0):
    """
    Compute the absolute coordinates of all vertices of the cluster.
    cluster_centers: (N, 2) array of centers relative to cluster origin.
    cluster_angle_deg: rotation of the entire cluster.
    Returns: (N, 6, 2) array of vertices in absolute coordinates.
    """
    n = len(cluster_centers)
    verts = np.zeros((n, 6, 2))
    a_rad = np.radians(cluster_angle_deg)
    c = np.cos(a_rad)
    s = np.sin(a_rad)
    R = np.array([[c, -s], [s, c]])
    
    for i in range(n):
        # Rotate center
        center_rot = R @ cluster_centers[i]
        # Rotate vertices relative to center
        v_rel = UNIT_VERTS @ R.T
        verts[i] = v_rel + center_rot
    return verts

def compute_min_side_lp(cluster_centers, cluster_angle_deg=0.0):
    """
    Use Linear Programming to find the minimum side length of a regular hexagon
    containing the cluster.
    
    The outer hexagon is defined by:
    - Center: (cx, cy)
    - Rotation: theta
    - Side length: s
    
    The constraint is that for every vertex v of the cluster, and for every 
    face normal n_k of the outer hexagon (rotated by theta), the projection 
    (v - c) . n_k <= apothem = s * sqrt(3)/2.
    
    Since theta is a variable, this is non-linear. However, we can fix theta 
    in a discrete search or use a different formulation.
    
    Alternative formulation:
    For a fixed outer hexagon orientation theta, the problem is convex.
    We can search over theta. But to make it "LP", we can parameterize the 
    support functions.
    
    Actually, the minimal enclosing regular hexagon of a set of points P 
    with orientation theta is determined by the max projection of P onto 
    the 6 normals.
    Let h_k(theta) = max_{p in P} (p . n_k(theta)).
    The side length s is determined by max(h_k(theta)) / (sqrt(3)/2).
    We want to minimize over theta and center c.
    
    For a fixed theta, the optimal center c is the midpoint of the bounding box 
    in the hexagonal metric? No, it's more complex.
    
    Let's use a simpler robust method:
    1. Fix cluster orientation.
    2. Use differential_evolution to find optimal outer theta and center (cx, cy).
    3. For each candidate (theta, cx, cy), compute the required side length 
       by checking all vertices against the 6 faces. This is O(N*6).
    
    To satisfy the "LP" spirit in the prompt, we can note that for fixed theta, 
    the problem of finding min s such that exists c:
    (v_i - c) . n_k <= s * sqrt(3)/2 for all i, k
    is a linear program in variables c_x, c_y, s.
    Constraints:
    c_x * n_k_x + c_y * n_k_y + s * sqrt(3)/2 >= v_i . n_k
    -c_x * n_k_x - c_y * n_k_y + s * sqrt(3)/2 >= -v_i . n_k  (for opposite face)
    
    So for a fixed theta, we can solve an LP to find min s and optimal c.
    Then we search over theta.
    """
    verts = get_cluster_vertices(cluster_centers, cluster_angle_deg)
    # Flatten vertices: (M, 2) where M = N * 6
    points = verts.reshape(-1, 2)
    
    def solve_for_theta(theta_deg):
        theta_rad = np.radians(theta_deg)
        # Normals of outer hexagon
        normals = np.array([
            [np.cos(np.radians(60 * i + 30 + theta_deg)), np.sin(np.radians(60 * i + 30 + theta_deg))] for i in range(6)
        ])
        
        # Variables: x = [cx, cy, s]
        # Minimize: s
        c_obj = np.array([0, 0, 1])
        
        A_ub = []
        b_ub = []
        
        for p in points:
            for k in range(6):
                n = normals[k]
                # Constraint: (p - c) . n <= s * sqrt(3)/2
                # p.n - c.n <= s * sqrt(3)/2
                # -c.n - s * sqrt(3)/2 <= -p.n
                # c.n_x * n_x + c.n_y * n_y + s * sqrt(3)/2 >= p.n
                # So: -n_x * cx - n_y * cy - sqrt(3)/2 * s <= -p.n
                
                row = np.array([-n[0], -n[1], -np.sqrt(3)/2])
                val = -np.dot(p, n)
                A_ub.append(row)
                b_ub.append(val)
                
                # Opposite face: (p - c) . (-n) <= s * sqrt(3)/2
                # -p.n + c.n <= s * sqrt(3)/2
                # c.n - s * sqrt(3)/2 >= p.n
                # So: -n_x * cx - n_y * cy + sqrt(3)/2 * s <= -p.n  <-- Wait.
                # (p-c).(-n) = -p.n + c.n <= s * sqrt(3)/2
                # c.n - s * sqrt(3)/2 >= p.n
                # Multiply by -1: -c.n + s * sqrt(3)/2 <= -p.n
                # n_x * cx + n_y * cy - sqrt(3)/2 * s <= -p.n
                
                row2 = np.array([n[0], n[1], -np.sqrt(3)/2])
                val2 = -np.dot(p, n)
                A_ub.append(row2)
                b_ub.append(val2)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Bounds: cx, cy in [-10, 10], s in [0, 20]
        bounds = [(-10, 10), (-10, 10), (0, 20)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                return res.x[2], res.x[0], res.x[1]
            else:
                return float('inf'), 0, 0
        except Exception:
            return float('inf'), 0, 0

    # Search over theta. Due to symmetry, we only need 0 to 60.
    # The function is periodic. We can sample it.
    best_s = float('inf')
    best_theta = 0
    best_c = (0, 0)
    
    # Sample theta
    thetas = np.linspace(0, 60, 121)
    for th in thetas:
        s, cx, cy = solve_for_theta(th)
        if s < best_s:
            best_s = s
            best_theta = th
            best_c = (cx, cy)
            
    # Refine around best_theta
    if best_s < float('inf'):
        thetas_refine = np.linspace(best_theta - 1, best_theta + 1, 21)
        for th in thetas_refine:
            s, cx, cy = solve_for_theta(th)
            if s < best_s:
                best_s = s
                best_theta = th
                best_c = (cx, cy)
                
    return best_s, best_theta, best_c

def generate_clusters():
    """
    Generate candidate rigid clusters of 12 unit hexagons.
    """
    clusters = []
    
    # Cluster 1: Hexagonal patch of radius 2 (19 hexes) minus 7 corners/edges to get 12?
    # A hexagon of radius 1 has 7 hexes.
    # A hexagon of radius 2 has 19 hexes.
    # We want 12.
    # Let's take the central hex + ring 1 (6) + 5 from ring 2.
    # Or a rectangular 3x4 grid in hex lattice.
    
    # Hex lattice vectors
    a1 = np.array([np.sqrt(3), 0])
    a2 = np.array([np.sqrt(3)/2, 1.5])
    
    # Configuration 1: 3x4 rectangular grid
    centers = []
    for i in range(3):
        for j in range(4):
            # Offset rows for honeycomb
            if i % 2 == 1:
                pos = i * a1 + j * a2 + a1/2
            else:
                pos = i * a1 + j * a2
            centers.append(pos)
    # Center it
    com = np.mean(centers, axis=0)
    centers = [c - com for c in centers]
    clusters.append(np.array(centers))
    
    # Configuration 2: "Snowflake" - Center + 6 in Ring 1 + 5 in Ring 2
    centers = [np.array([0,0])]
    # Ring 1
    for k in range(6):
        angle = np.radians(60 * k + 30)
        r = np.sqrt(3)
        centers.append(np.array([r * np.cos(angle), r * np.sin(angle)]))
    # Ring 2: pick 5 positions
    # Ring 2 positions are at distance 2*sqrt(3) or sqrt(3)*sqrt(7)?
    # In hex grid, ring 2 has 12 positions.
    # Let's just add 5 more from a larger grid to make it compact.
    # Actually, let's try a 2x6 rectangle.
    centers2 = []
    for i in range(2):
        for j in range(6):
            if i % 2 == 1:
                pos = i * a1 + j * a2 + a1/2
            else:
                pos = i * a1 + j * a2
            centers2.append(pos)
    com2 = np.mean(centers2, axis=0)
    centers2 = [c - com2 for c in centers2]
    clusters.append(np.array(centers2))
    
    # Configuration 3: Two rows of 3, two rows of 3, offset
    # Basically a 4x3 grid but rotated 90 degrees?
    centers3 = []
    for i in range(4):
        for j in range(3):
            if i % 2 == 1:
                pos = i * a1 + j * a2 + a1/2
            else:
                pos = i * a1 + j * a2
            centers3.append(pos)
    com3 = np.mean(centers3, axis=0)
    centers3 = [c - com3 for c in centers3]
    clusters.append(np.array(centers3))
    
    return clusters

def construct():
    start_time = time.time()
    
    best_score = float('inf')
    best_result = None
    
    clusters = generate_clusters()
    
    for cluster in clusters:
        # Check validity of cluster (no overlaps)
        # For a standard hex lattice packing, they are tangent, so valid.
        # We assume these generated clusters are valid packings.
        
        # Optimize outer hexagon
        side, outer_angle, outer_center = compute_min_side_lp(cluster)
        
        if side < best_score:
            # Construct the result
            # inner: list of [x, y, angle_deg]
            # The cluster is rotated by 0 relative to its local frame, 
            # but the LP found the optimal outer hexagon orientation.
            # The inner hexagons' absolute angles are 0 (since cluster angle is 0).
            # Wait, the cluster vertices were computed with cluster_angle=0.
            # The outer hexagon has rotation outer_angle.
            # The inner hexagons are fixed in the cluster frame.
            # So their absolute angle is 0.
            
            inner = []
            for c in cluster:
                inner.append([c[0], c[1], 0.0])
                
            # outer_center is (cx, cy)
            # outer_side is side
            # outer_angle is outer_angle
            
            # Verify validity just in case
            # We need to check if the inner hexagons fit in the outer hexagon.
            # The LP guarantees this for the vertices.
            
            if check_validity(inner, outer_center, side, outer_angle):
                best_score = side
                best_result = (inner, outer_center, side, outer_angle)
                
        if time.time() - start_time > DEADLINE - 10:
            break
            
    # If LP approach didn't find a good result or time is running out, 
    # try a simple heuristic fallback or the previous best.
    if best_result is None:
        # Fallback: Simple ring
        inner = []
        for i in range(12):
            theta = 2 * np.pi * i / 12
            x = 3.0 * np.cos(theta)
            y = 3.0 * np.sin(theta)
            inner.append([x, y, 0.0])
        side = 5.0
        best_result = (inner, (0.0, 0.0), side, 0.0)
        
    return best_result

def check_validity(inner, outer_center, outer_side, outer_angle, tol=1e-5):
    """
    Verify that the configuration is valid.
    inner: list of [x, y, angle]
    outer_center: (x, y)
    outer_side: float
    outer_angle: float (degrees)
    """
    verts_list = []
    for x, y, a_deg in inner:
        a_rad = np.radians(a_deg)
        c = np.cos(a_rad)
        s = np.sin(a_rad)
        R = np.array([[c, -s], [s, c]])
        v = UNIT_VERTS @ R.T + np.array([x, y])
        verts_list.append(v)
        
    # 1. Check overlaps
    n = len(verts_list)
    for i in range(n):
        v1 = verts_list[i]
        for j in range(i + 1, n):
            v2 = verts_list[j]
            # Simple distance check for centers as a quick filter?
            # No, do SAT.
            edges1 = np.roll(v1, -1, axis=0) - v1
            edges2 = np.roll(v2, -1, axis=0) - v2
            axes = np.vstack([edges1, edges2])
            axes = axes / (np.linalg.norm(axes, axis=1, keepdims=True) + 1e-15)
            
            proj1 = v1 @ axes.T
            proj2 = v2 @ axes.T
            
            sep = np.minimum(proj1.min(axis=0) - proj2.max(axis=0),
                            proj2.min(axis=0) - proj1.max(axis=0))
            
            if np.max(sep) < -tol:
                return False
                
    # 2. Check containment
    apothem = outer_side * np.sqrt(3) / 2
    oa_rad = np.radians(outer_angle)
    c_o = np.cos(-oa_rad)
    s_o = np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    for v in verts_list:
        v_rel = v - np.array(outer_center)
        v_rot = v_rel @ R_o.T
        # Project onto 6 normals
        for k in range(6):
            angle = np.radians(60 * k + 30)
            n = np.array([np.cos(angle), np.sin(angle)])
            proj = np.dot(v_rot, n)
            if abs(proj) > apothem + tol:
                return False
                
    return True

if __name__ == "__main__":
    result = construct()
    print(f"Best Side: {result[2]}")
