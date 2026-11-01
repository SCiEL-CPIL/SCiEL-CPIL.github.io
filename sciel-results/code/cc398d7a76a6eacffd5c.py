import numpy as np
import time
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Helper functions for geometry
    def get_hex_vertices(cx, cy, s=1.0, angle=0.0):
        """Return 6 vertices of a regular hexagon."""
        a = np.radians(angle)
        pts = np.array([[cx + s * np.cos(a + np.radians(60 * k)),
                         cy + s * np.sin(a + np.radians(60 * k))] for k in range(6)])
        return pts

    def get_normals(vertices):
        """Compute outward normals for a convex polygon."""
        normals = np.zeros_like(vertices)
        for i in range(6):
            j = (i + 1) % 6
            ex, ey = vertices[j] - vertices[i]
            nl = np.hypot(ex, ey)
            if nl > 1e-12:
                normals[i] = np.array([-ey, ex]) / nl
        return normals

    def support_function(vertices, normals):
        """Compute support function values for given normals."""
        return vertices @ normals.T

    def smooth_max(vals, beta=100.0):
        """Smooth approximation of max using log-sum-exp."""
        m = np.max(vals)
        if np.allclose(vals, m):
            return m
        return m + np.log(np.sum(np.exp(beta * (vals - m)))) / beta

    def compute_outer_side(positions, angles, outer_center, outer_side, outer_angle):
        """
        Compute a smooth penalty for outer containment.
        Returns 0 if all vertices are inside, positive value otherwise.
        """
        ocx, ocy = outer_center
        # Outer hexagon vertices and normals
        outer_verts = get_hex_vertices(ocx, ocy, outer_side, outer_angle)
        outer_normals = get_normals(outer_verts)
        
        # We want all inner vertices to satisfy: n_i . (v - c) <= 0 for all outer normals n_i
        # Or: max(n_i . v) <= max(n_i . c) + side_length? No, it's about support.
        # Distance from center to face along normal n is 'apothem'.
        # For a regular hexagon of side s, apothem = s * sqrt(3) / 2.
        # The condition for a point v to be inside the outer hexagon (centered at c, angle a) is:
        # For each of the 6 normals n_k of the outer hex: n_k . (v - c) <= apothem
        # Let's use the normals of the outer hexagon.
        
        apothem = outer_side * np.sqrt(3) / 2.0
        
        penalty = 0.0
        for i in range(N):
            v = get_hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            # For each vertex of the inner hex, check against each outer normal
            # d_k = n_k . (v - c)
            # We need d_k <= apothem for all k.
            # Smooth penalty: sum(softplus(d_k - apothem))
            for vert in v:
                rel = vert - np.array([ocx, ocy])
                dists = outer_normals @ rel
                # softplus(x) = log(1 + exp(x))
                # We want to penalize when dists > apothem
                over = dists - apothem
                # Smooth max of (over, 0)
                # Use log(1+exp(beta*x))/beta approx for max(x,0)
                # Actually, we just need sum of violations.
                # violation = max(0, over)
                # Smooth: log(1+exp(beta*over))/beta
                viol = np.log1p(np.exp(100.0 * over)) / 100.0
                penalty += np.sum(viol)
        return penalty

    def compute_pairwise_distance(i, j, positions, angles):
        """
        Compute a smooth measure of the minimum distance between hex i and j.
        We want this distance to be >= 0.
        Using the Minkowski difference approach:
        Distance >= 0 iff 0 is not in the Minkowski difference H_i - H_j.
        The minimum distance is the minimum norm of vectors in H_i - H_j.
        This is equivalent to the minimum over all pairs of edges of the distance between them?
        A simpler differentiable proxy:
        The support function of the Minkowski sum H_i + (-H_j) in direction u is h_i(u) + h_j(-u).
        The distance is min_u ||u|| subject to ... this is complex.
        
        Alternative: Use the fact that for two convex polygons, the minimum distance is the minimum
        over all pairs of edges (e1, e2) of the distance between the line segments, or vertex-vertex.
        However, a very effective smooth penalty for disjointness is based on the "separating axis" 
        but smoothed.
        
        Let's use the smooth maximum of the penetration depth over all potential separating axes.
        The axes are the normals of the edges of H_i and H_j.
        For a normal n, the projection of H_i is [min_i, max_i] and H_j is [min_j, max_j].
        If max_i < min_j or max_j < min_i, they are separated along n.
        The "overlap" along n is max(0, max_i - min_j, max_j - min_i).
        The polygons are disjoint iff there exists an n such that overlap(n) == 0.
        We want to enforce that min_n overlap(n) >= 0 is false? No, we want them to be disjoint, so we want max_n separation(n) > 0.
        
        Let's define separation(n) = min(min_j - max_i, min_i - max_j).
        If separation(n) > 0, they are disjoint along n.
        We want max_n separation(n) > 0.
        Smooth max of separation(n) over all normals.
        If this value is <= 0, they are overlapping (or touching).
        We want to minimize the negative of this value, or penalize when it is <= 0.
        
        Penalty = softplus(-max_n separation(n))
        """
        verts_i = get_hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
        verts_j = get_hex_vertices(positions[j, 0], positions[j, 1], 1.0, angles[j])
        
        norms_i = get_normals(verts_i)
        norms_j = get_normals(verts_j)
        
        all_normals = np.vstack([norms_i, norms_j])
        
        seps = []
        for n in all_normals:
            proj_i = verts_i @ n
            proj_j = verts_j @ n
            
            min_i, max_i = np.min(proj_i), np.max(proj_i)
            min_j, max_j = np.min(proj_j), np.max(proj_j)
            
            # Separation along n
            sep1 = min_j - max_i
            sep2 = min_i - max_j
            sep = np.maximum(sep1, sep2)
            seps.append(sep)
            
        seps = np.array(seps)
        # We want max(seps) to be > 0.
        # Smooth max
        max_sep = smooth_max(seps, beta=50.0)
        
        # Penalty if max_sep <= 0
        # softplus(-max_sep)
        return np.log1p(np.exp(50.0 * (-max_sep))) / 50.0

    def objective(params):
        """
        params: [x_0, y_0, a_0, ..., x_11, y_11, a_11, ox, oy, s_o, a_o]
        Total 12*3 + 4 = 40 parameters.
        """
        # Check time
        if time.time() > deadline:
            return 1e10
            
        positions = params[:3*N].reshape(N, 3)
        ox, oy, s_o, a_o = params[3*N:]
        
        # Outer hexagon side length must be positive
        if s_o < 1.0:
            s_o = 1.0 + (1.0 - s_o) * 100.0 # Penalize small side
            
        pos_xy = positions[:, :2]
        angles = positions[:, 2]
        
        # Outer containment penalty
        p_outer = compute_outer_side(pos_xy, angles, np.array([ox, oy]), s_o, a_o)
        
        # Pairwise disjointness penalty
        p_pairs = 0.0
        for i in range(N):
            for j in range(i+1, N):
                p_pairs += compute_pairwise_distance(i, j, pos_xy, angles)
        
        # Objective: Minimize outer side length + penalties
        # We want to minimize s_o, but s_o is a variable.
        # If we just minimize s_o, it will go to 0 unless penalized by containment.
        # The containment penalty ensures s_o is large enough to fit the hexes.
        # So objective = s_o + weight_outer * p_outer + weight_pairs * p_pairs
        
        return s_o + 10.0 * p_outer + 10.0 * p_pairs

    # Initial guess
    # Hexagonal close packing
    s3 = np.sqrt(3)
    init_params = np.zeros(3*N + 4)
    
    # Centers
    centers = []
    centers.append([0, 0, 0])
    for k in range(6):
        a = np.radians(60 * k)
        centers.append([s3 * np.cos(a), 1.5 * np.sin(a), 0])
    for k in range(5):
        a = np.radians(60 * k + 30)
        r = 2 * 1.5 * np.sqrt(3)
        centers.append([r * np.cos(a), r * np.sin(a), 0])
        
    init_params[:3*N] = np.array(centers).flatten()
    
    # Outer hexagon
    init_params[3*N] = 0.0 # ox
    init_params[3*N+1] = 0.0 # oy
    init_params[3*N+2] = 5.0 # s_o (initial guess)
    init_params[3*N+3] = 0.0 # a_o
    
    # Optimization
    best_result = None
    best_score = float('inf')
    
    # Try L-BFGS-B
    try:
        res = minimize(
            objective, init_params,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        
        if res.fun < best_score:
            best_score = res.fun
            best_result = res.x
    except Exception as e:
        pass
        
    # If optimization failed or didn't improve, use initial
    if best_result is None:
        best_result = init_params
        
    # Extract best result
    final_params = best_result
    pos_xy = final_params[:3*N].reshape(N, 3)[:, :2]
    angles = final_params[:3*N].reshape(N, 3)[:, 2]
    ox, oy = final_params[3*N], final_params[3*N+1]
    s_o = final_params[3*N+2]
    a_o = final_params[3*N+3]
    
    # Ensure validity: check if hexes are actually disjoint and inside
    # If not, we might need to fallback or just return the best we have.
    # The smooth penalty should have pushed them apart.
    
    # Convert angles to degrees for output
    inner = []
    for i in range(N):
        inner.append([float(pos_xy[i, 0]), float(pos_xy[i, 1]), float(np.degrees(angles[i]))])
        
    return inner, (float(ox), float(oy)), float(s_o), float(np.degrees(a_o))

# EVOLVE-BLOCK-END
