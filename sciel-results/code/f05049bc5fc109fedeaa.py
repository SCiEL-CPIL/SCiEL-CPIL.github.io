"""
Improved 11-hexagon packing: Topological constraint satisfaction with rigid body optimization.
Strategy:
1. Define candidate contact graphs (topologies) for 11 unit hexagons.
2. Compute the vertices for each topology.
3. Treat the cluster as a rigid body. Optimize the global orientation (rotation) and container orientation/position.
4. This reduces the search space from ~36 dimensions to ~3-4 dimensions, allowing for much deeper and more precise optimization.
5. Include a fallback 'soft' optimization phase if needed, but prioritize the rigid solution which is likely optimal for small N.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)
APOTHEM = math.sqrt(3) / 2.0

# EVOLVE-BLOCK-START

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon as numpy array (6,2)."""
    thetas = angle_rad + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + side * np.cos(thetas), cy + side * np.sin(thetas)])

def compute_required_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_deg)
    outer_center: (x, y)
    outer_angle_deg: rotation of the outer hexagon
    """
    if not inner:
        return 0.0

    ox, oy = outer_center
    # Normals for the outer hexagon sides (relative to its local frame)
    # A regular hexagon with flat top has normals at 30, 90, 150, 210, 270, 330 degrees
    normals = np.array([[math.cos(math.radians(30 + i * 60)),
                         math.sin(math.radians(30 + i * 60))] for i in range(6)])

    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    # Rotation matrix to transform from global to outer-local frame
    # Note: We want to project points onto the normals.
    # If outer is rotated by alpha, its normals are rotated by alpha.
    # It is equivalent to rotating the points by -alpha and projecting onto fixed normals.
    R = np.array([[cos_a, sin_a], [-sin_a, cos_a]])

    max_projection = -np.inf
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        # Translate to outer center
        local = (verts - np.array([ox, oy])) @ R.T
        projs = local @ normals.T
        cur = projs.max()
        if cur > max_projection:
            max_projection = cur

    # Side length S relates to projection P by P = S * sqrt(3) / 2
    # So S = 2 * P / sqrt(3)
    return 2.0 * max_projection / SQRT3

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=100.0):
    """
    Smooth surrogate for compute_required_side using log-sum-exp soft-max.
    """
    if not inner:
        return 0.0

    ox, oy = outer_center
    normals = np.array([[math.cos(math.radians(30 + i * 60)),
                         math.sin(math.radians(30 + i * 60))] for i in range(6)])

    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    R = np.array([[cos_a, sin_a], [-sin_a, cos_a]])

    all_projs = []
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        local = (verts - np.array([ox, oy])) @ R.T
        projs = local @ normals.T
        all_projs.append(projs)

    P = np.concatenate([p.ravel() for p in all_projs])
    if P.size == 0:
        return 0.0

    m = np.max(P)
    if m < -1e6:
        return 0.0
    max_proj = float(m + np.log(np.sum(np.exp(beta * (P - m)))) / beta)

    return 2.0 * max_proj / SQRT3

def sat_overlap_check(v1, v2, eps=1e-9):
    """Check if two convex polygons overlap using SAT."""
    axes = []
    for poly in (v1, v2):
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i + 1) % len(poly)]
            edge = (p2[0] - p1[0], p2[1] - p1[1])
            normal = (-edge[1], edge[0])
            norm = math.sqrt(normal[0]**2 + normal[1]**2)
            if norm > 1e-9:
                axes.append((normal[0] / norm, normal[1] / norm))

    for axis in axes:
        min1 = min(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        max1 = max(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        min2 = min(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        max2 = max(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    return True

def check_all_disjoint(inner, eps=1e-6):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner)
    polys = []
    for (x, y, a) in inner:
        polys.append(get_hex_vertices(x, y, 1.0, math.radians(a)))
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap_check(polys[i], polys[j], eps):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Define candidate topologies (contact graphs) for 11 hexagons.
    # Coordinates are in axial coordinates (q, r) for a unit hex grid.
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    
    def axial_to_cartesian(coords):
        pts = []
        for q, r in coords:
            x = 1.5 * q
            y = SQRT3 * (r + q / 2.0)
            pts.append((x, y))
        return pts

    # Topology 1: Center + 6 Ring + 4 attached to ring (symmetric cross-like extension)
    # Ring indices: 0:(1,0), 1:(0,1), 2:(-1,1), 3:(-1,0), 4:(0,-1), 5:(1,-1)
    # Attach 4 to indices 0, 1, 3, 4? Or 0, 2, 4, 5?
    # Let's try a few specific configurations.
    
    candidates = []

    # Config A: "Plus" shape extension (4 extra at N, S, E, W of the ring)
    # Ring: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Extra at: (2,0) [E of (1,0)], (0,2) [N of (0,1)], (-2,0) [W of (-1,0)], (0,-2) [S of (0,-1)]
    # Wait, (0,2) is neighbor of (0,1) and (-1,1).
    # Let's define explicit sets of axial coords.
    
    # 1. Center (0,0)
    # 2-7. Ring
    # 8-11. Extensions
    
    # Candidate 1: Symmetric "X" extensions (diagonals of the ring)
    # Ring: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ext: (2,-1) [se of (1,0)], (1,1) [ne of (0,1)], (-1,2) [nw of (-1,1)], (-2,1) [sw of (-1,0)]
    # This creates a 12-gon like shape? No, 1 center + 6 ring + 4 ext = 11.
    axial_1 = [(0,0)] + [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)] + [(2,-1), (1,1), (-1,2), (-2,1)]
    candidates.append(axial_1)

    # Candidate 2: "T" shape extensions (3 in one direction, 1 opposite)
    # Ext: (2,0), (3,0) [chain E], (0,2) [N], (0,-2) [S]
    axial_2 = [(0,0)] + [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)] + [(2,0), (3,0), (0,2), (0,-2)]
    candidates.append(axial_2)

    # Candidate 3: "L" shape extensions (2 E, 2 N)
    # Ext: (2,0), (3,0) [E], (0,2), (-1,2) [N]
    axial_3 = [(0,0)] + [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)] + [(2,0), (3,0), (0,2), (-1,2)]
    candidates.append(axial_3)

    # Candidate 4: Compact cluster (no long chains, all within radius 2)
    # Ext: (2,0), (1,1), (0,2), (-1,1) -- wait (-1,1) is in ring.
    # Let's pick 4 from the "second ring" that are most compact.
    # Second ring neighbors of the ring:
    # (2,0), (2,-1), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2)
    # Pick (2,0), (1,1), (0,2), (-1,2) ?
    axial_4 = [(0,0)] + [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)] + [(2,0), (1,1), (0,2), (-1,2)]
    candidates.append(axial_4)

    # Candidate 5: Another compact one
    axial_5 = [(0,0)] + [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)] + [(2,0), (1,1), (0,2), (-2,1)]
    candidates.append(axial_5)

    best_global_side = float('inf')
    best_global_inner = None
    best_global_center = None
    best_global_angle = 0.0

    for idx, axial_coords in enumerate(candidates):
        if time.time() > deadline - 20:
            break

        if len(axial_coords) != N:
            continue

        # Convert to Cartesian
        centers = axial_to_cartesian(axial_coords)
        
        # Initial inner layout (all aligned)
        initial_inner = [[x, y, 0.0] for x, y in centers]

        # Verify disjointness of this rigid topology
        if not check_all_disjoint(initial_inner, eps=1e-5):
            continue

        # The cluster is a rigid body.
        # We optimize:
        # 1. Global rotation of the cluster (theta_c)
        # 2. Outer center (cx, cy) - usually (0,0) if we center the cluster
        # 3. Outer rotation (theta_o)
        
        # Let's center the cluster at origin first.
        cx_avg = sum(c[0] for c in centers) / N
        cy_avg = sum(c[1] for c in centers) / N
        
        centered_centers = [(x - cx_avg, y - cy_avg) for x, y in centers]

        # Function to evaluate a specific rotation of the cluster and container
        def objective_rigid(params):
            """
            params: [theta_c_deg, theta_o_deg]
            cx, cy are fixed at 0,0 for the container, and cluster is centered at 0,0.
            """
            theta_c = params[0]
            theta_o = params[1]
            
            rad_c = math.radians(theta_c)
            cos_c = math.cos(rad_c)
            sin_c = math.sin(rad_c)
            
            rotated_inner = []
            for x, y in centered_centers:
                rx = x * cos_c - y * sin_c
                ry = x * sin_c + y * cos_c
                # Inner hexagons also rotate with the cluster
                rotated_inner.append([rx, ry, theta_c])
            
            side = compute_required_side(rotated_inner, (0, 0), theta_o)
            return side

        # We can also vary the container center slightly, but for a rigid cluster,
        # centering both at origin is optimal for the "minimum bounding hexagon" 
        # if the cluster is symmetric. If not, we might need to shift.
        # However, the support function max projection is minimized when the centroid
        # of the support points aligns. For a convex container, shifting the cluster
        # relative to the container center can reduce the max projection if the cluster
        # is not centrally symmetric.
        
        # Let's add cx, cy to the optimization for robustness.
        def objective_rigid_full(params):
            theta_c = params[0]
            cx = params[1]
            cy = params[2]
            theta_o = params[3]
            
            rad_c = math.radians(theta_c)
            cos_c = math.cos(rad_c)
            sin_c = math.sin(rad_c)
            
            rotated_inner = []
            for x, y in centered_centers:
                rx = x * cos_c - y * sin_c
                ry = x * sin_c + y * cos_c
                rotated_inner.append([rx, ry, theta_c])
            
            side = compute_required_side(rotated_inner, (cx, cy), theta_o)
            return side

        best_val = float('inf')
        best_params = None

        # Multi-start optimization
        # The function is periodic with 60 degrees for both rotations.
        # We will sample a grid of starting points.
        
        for start_theta_c in range(0, 60, 5):
            for start_theta_o in range(0, 60, 5):
                if time.time() > deadline - 10:
                    break
                
                init_params = [float(start_theta_c), 0.0, 0.0, float(start_theta_o)]
                
                try:
                    # Use Nelder-Mead for small dimensional problems
                    res = minimize(objective_rigid_full, init_params, method='Nelder-Mead',
                                   options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 5000})
                    
                    if res.fun < best_val:
                        best_val = res.fun
                        best_params = res.x
                except Exception:
                    pass

        if best_params is not None:
            theta_c, cx, cy, theta_o = best_params
            
            # Reconstruct the final inner layout
            rad_c = math.radians(theta_c)
            cos_c = math.cos(rad_c)
            sin_c = math.sin(rad_c)
            
            final_inner = []
            for x, y in centered_centers:
                rx = x * cos_c - y * sin_c
                ry = x * sin_c + y * cos_c
                final_inner.append([rx, ry, theta_c])
            
            final_center = (cx, cy)
            final_angle = theta_o
            final_side = best_val

            # Verify disjointness (should be true as it's a rigid rotation of a valid layout)
            if check_all_disjoint(final_inner, eps=1e-5):
                if final_side < best_global_side:
                    best_global_side = final_side
                    best_global_inner = final_inner
                    best_global_center = final_center
                    best_global_angle = final_angle
                    # Print progress for debugging if needed
                    # print(f"Candidate {idx}: Side {final_side:.6f}")

    # If no rigid candidate worked, fall back to the previous free-form approach (simplified)
    if best_global_inner is None:
        # Use the first candidate as a loose start for a quick free-form optimization
        axial_coords = candidates[0]
        centers = axial_to_cartesian(axial_coords)
        initial_inner = [[x, y, 0.0] for x, y in centers]
        
        # Quick local polish on free positions
        def objective_free(params):
            inner_list = []
            for i in range(N):
                inner_list.append([params[i*3], params[i*3+1], params[i*3+2]])
            cx = params[N*3]
            cy = params[N*3+1]
            angle = params[N*3+2]
            return compute_required_side(inner_list, (cx, cy), angle)

        init_params = []
        for x, y, a in initial_inner:
            init_params.extend([x, y, a])
        init_params.extend([0, 0, 0])
        
        try:
            res = minimize(objective_free, init_params, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 2000})
            
            final_inner = []
            for i in range(N):
                final_inner.append([res.x[i*3], res.x[i*3+1], res.x[i*3+2]])
            
            if check_all_disjoint(final_inner, eps=1e-5):
                best_global_inner = final_inner
                best_global_center = (res.x[N*3], res.x[N*3+1])
                best_global_angle = res.x[N*3+2]
                best_global_side = res.fun
        except Exception:
            # Absolute fallback
            best_global_inner = initial_inner
            best_global_center = (0, 0)
            best_global_angle = 0.0
            best_global_side = compute_required_side(initial_inner, (0,0), 0.0)

    return best_global_inner, best_global_center, best_global_side, best_global_angle

# EVOLVE-BLOCK-END
