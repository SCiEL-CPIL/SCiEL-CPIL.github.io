"""Hexagon packing: 11 unit hexagons inside a regular hexagon."""
import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11

# EVOLVE-BLOCK-START

# Precompute hexagon vertex offsets for unit hexagon at angle 0
_HEX_VERTS_0 = np.array([[math.cos(i * math.pi / 3.0), math.sin(i * math.pi / 3.0)] for i in range(6)])

def _hex_vertices(angle_rad, center):
    """Return 6 vertices of a unit regular hexagon rotated by angle_rad, translated by center."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    return _HEX_VERTS_0 @ R.T + center

def _support_function(angle_rad, center, direction):
    """Support function: max dot(v, direction) over vertices of hexagon at angle_rad, center."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    verts = _HEX_VERTS_0 @ R.T
    return np.max(verts @ direction) + center @ direction

def _hex_distance(a1, c1, a2, c2):
    """
    Compute distance between two regular unit hexagons using Minkowski sum / support function.
    Returns 0 if overlapping.
    For two convex polygons A and B, dist(A,B) = max(0, min over directions of (h_A(-d) + h_B(d) - |c1-c2|_d))
    where h is support function. Actually: dist = max(0, -max_d [h_A(-d) + h_B(d) - (c1-c2)·d])
    
    More precisely: dist(A,B) = max(0, -max_{d} [h_A(-d) + h_B(d) - (c1-c2)·d])
    where the max is over unit directions. For polygons, we only need to check normals of both polygons.
    """
    d_center = c1 - c2
    # Candidate directions: edge normals of hex1 and hex2
    # Edge normals of hex at angle a: a + k*pi/3 for k=0..5
    dirs = []
    for k in range(6):
        dirs.append(a1 + k * math.pi / 3.0)
        dirs.append(a2 + k * math.pi / 3.0)
    
    max_val = -float('inf')
    for d_angle in dirs:
        d_vec = np.array([math.cos(d_angle), math.sin(d_angle)])
        # h_A(-d): support of A in direction -d
        h_A_neg = _support_function(a1, c1, -d_vec)
        # h_B(d): support of B in direction d
        h_B = _support_function(a2, c2, d_vec)
        # (c1-c2)·d
        cd = d_center @ d_vec
        val = h_A_neg + h_B - cd
        if val > max_val:
            max_val = val
    
    # Distance = max(0, -max_val) if max_val < 0 means separated
    # Actually: dist = max(0, -max_val) is wrong.
    # The gap in direction d is: h_A(-d) + h_B(d) - (c1-c2)·d
    # If this is negative for some d, they overlap.
    # dist(A,B) = max(0, min_d [h_A(-d) + h_B(d) - (c1-c2)·d]) ... no.
    # 
    # Correct: dist(A,B) = max(0, -max_d [h_A(-d) + h_B(d) - (c1-c2)·d])
    # Wait, let me think again.
    # h_A(-d) = max_{x in A} (-d)·x = -min_{x in A} d·x
    # h_B(d) = max_{y in B} d·y
    # So h_A(-d) + h_B(d) - (c1-c2)·d = max_y d·y - min_x d·x - d·(c1-c2)
    # = max_y d·(y-c2) - min_x d·(x-c1) + d·(c2-c1)
    # Let u = x - c1 (in hex1 centered at origin), v = y - c2 (in hex2 centered at origin)
    # = max_v d·v - min_u d·u + d·(c2-c1)
    # = max_v d·v - min_u d·u - d·(c1-c2)
    # If this is <= 0, then in direction d, the hexagons overlap or touch.
    # dist = max(0, min_d [h_A(-d) + h_B(d) - (c1-c2)·d]) ... no, that's not right either.
    
    # Let me use the standard formula:
    # dist(A,B) = max(0, -max_{d unit} [h_A(-d) + h_B(d) - (c1-c2)·d])
    # Hmm, I think the correct formula is:
    # The separation in direction d is: h_B(d) - h_A(-d) + (c1-c2)·d ... 
    # 
    # Let me just use a simpler approach: check if they overlap via SAT, and if not, compute min edge distance.
    # But that's what we had before. Let me try a different fast approach.
    
    # Actually for regular hexagons, I can use the fact that the Minkowski sum of two unit hexagons
    # is also a hexagon (or 12-gon), and check if the center difference is inside.
    
    # Simpler: use SAT to check overlap, and if not overlapping, do a quick edge check.
    # But let's just use the support function approach correctly.
    
    # dist(A,B) = max(0, -max_d [h_A(-d) + h_B(d) - (c1-c2)·d])
    # Let's verify: if A and B are far apart, h_A(-d) + h_B(d) - (c1-c2)·d will be very negative
    # for all d, so max_d will be negative, and -max_d will be positive = distance. Good.
    # If they overlap, there exists d such that h_A(-d) + h_B(d) - (c1-c2)·d >= 0, so max_d >= 0,
    # and max(0, -max_d) = 0. Good.
    
    return max(0.0, -max_val)

def _compute_side_length(centers, angles, outer_angle_rad=0.0):
    """
    Compute the minimum side length of an outer hexagon (at given rotation)
    that contains all inner hexagons.
    """
    outer_normals = []
    for k in range(6):
        theta = outer_angle_rad + k * math.pi / 3.0
        outer_normals.append([math.cos(theta), math.sin(theta)])
    outer_normals = np.array(outer_normals)
    
    max_extent = 0.0
    for i in range(N):
        c, s = math.cos(angles[i]), math.sin(angles[i])
        R = np.array([[c, -s], [s, c]])
        verts = _HEX_VERTS_0 @ R.T
        projections = (verts + centers[i]) @ outer_normals.T
        abs_proj = np.abs(projections)
        max_extent = max(max_extent, np.max(abs_proj))
    
    S = max_extent / (math.sqrt(3) / 2.0)
    return S

def _is_valid(centers, angles, outer_side, outer_angle_rad=0.0):
    """Check if the configuration is valid (no overlaps, all inside outer)."""
    # Check containment using projection method
    outer_normals = []
    for k in range(6):
        theta = outer_angle_rad + k * math.pi / 3.0
        outer_normals.append([math.cos(theta), math.sin(theta)])
    outer_normals = np.array(outer_normals)
    limit = outer_side * math.sqrt(3) / 2.0
    for i in range(N):
        c, s = math.cos(angles[i]), math.sin(angles[i])
        R = np.array([[c, -s], [s, c]])
        verts = _HEX_VERTS_0 @ R.T
        projections = (verts + centers[i]) @ outer_normals.T
        if np.max(np.abs(projections)) > limit + 1e-6:
            return False
    
    # Check no overlaps using fast hex distance
    for i in range(N):
        for j in range(i + 1, N):
            if _hex_distance(angles[i], centers[i], angles[j], centers[j]) < 1e-6:
                return False
    return True

def _seed_solution():
    """Generate a valid seed solution using a hexagonal lattice."""
    d = math.sqrt(3)
    centers = []
    angles = []
    
    # Central hexagon
    centers.append([0, 0])
    angles.append(0.0)
    
    # 6 surrounding
    for i in range(6):
        theta = i * math.pi / 3.0
        centers.append([d * math.cos(theta), d * math.sin(theta)])
        angles.append(0.0)
    
    # 4 more from second ring
    ring2_positions = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                x = d * (q + r / 2.0)
                y = d * (math.sqrt(3) / 2.0) * r
                ring2_positions.append([x, y])
    
    # Pick 4 that are well-separated
    selected = [ring2_positions[i] for i in [0, 3, 6, 9]]
    for pos in selected:
        centers.append(pos)
        angles.append(0.0)
    
    centers = np.array(centers)
    angles = np.array(angles)
    
    S = _compute_side_length(centers, angles)
    S *= 1.01
    
    return centers, angles, S

def construct():
    start_time = time.time()
    deadline = 145.0
    
    centers0, angles0, S0 = _seed_solution()
    best_S = S0
    best_centers = centers0.copy()
    best_angles = angles0.copy()
    best_outer_angle = 0.0
    
    if not _is_valid(best_centers, best_angles, best_S):
        d = math.sqrt(3) * 1.5
        centers_fb = []
        angles_fb = []
        centers_fb.append([0, 0])
        angles_fb.append(0.0)
        for i in range(6):
            theta = i * math.pi / 3.0
            centers_fb.append([d * math.cos(theta), d * math.sin(theta)])
            angles_fb.append(0.0)
        for i in range(4):
            theta = (2*i + 1) * math.pi / 12.0
            centers_fb.append([2*d * math.cos(theta), 2*d * math.sin(theta)])
            angles_fb.append(0.0)
        best_centers = np.array(centers_fb)
        best_angles = np.array(angles_fb)
        best_S = _compute_side_length(best_centers, best_angles)
        best_S *= 1.01
    
    def objective(params):
        """
        params: 2N+1 array: [x1, y1, ..., xN, yN, outer_angle]
        Inner angles fixed to 0.
        """
        if time.time() > deadline - 5.0:
            return 1e10
        
        pos = params[:2*N].reshape(N, 2)
        outer_angle = params[2*N]
        
        # Compute S
        S = _compute_side_length(pos, angles0, outer_angle)
        
        # Compute penalty using fast hex distance
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                d = _hex_distance(angles0[i], pos[i], angles0[j], pos[j])
                if d < 0.01:
                    penalty += (0.01 - d) ** 2
        
        return S + 100.0 * penalty
    
    # Differential Evolution
    bounds = [(-5, 5)] * (2*N) + [(0, math.pi/3)]
    
    try:
        if time.time() < deadline - 30.0:
            res = differential_evolution(objective, bounds, seed=42, maxiter=200, popsize=20, tol=1e-5, polish=True)
            if res.fun < best_S:
                pos = res.x[:2*N].reshape(N, 2)
                outer_angle = res.x[2*N]
                S_val = _compute_side_length(pos, angles0, outer_angle)
                if _is_valid(pos, angles0, S_val * 1.001, outer_angle):
                    best_S = S_val
                    best_centers = pos
                    best_outer_angle = outer_angle
    except Exception:
        pass
    
    # Local refinement with Nelder-Mead
    if time.time() < deadline - 10.0:
        x0 = np.hstack([best_centers.flatten(), [best_outer_angle]])
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                           options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 10000, 'maxfev': 20000})
            if res.fun < best_S:
                pos = res.x[:2*N].reshape(N, 2)
                outer_angle = res.x[2*N]
                S_val = _compute_side_length(pos, angles0, outer_angle)
                if _is_valid(pos, angles0, S_val * 1.001, outer_angle):
                    best_S = S_val
                    best_centers = pos
                    best_outer_angle = outer_angle
        except Exception:
            pass
    
    # Format output
    inner = []
    for i in range(N):
        inner.append([best_centers[i][0], best_centers[i][1], 0.0])
    
    return inner, (0.0, 0.0), best_S, math.degrees(best_outer_angle)

# EVOLVE-BLOCK-END
