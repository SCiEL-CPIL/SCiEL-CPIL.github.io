"""
Improved program: Topology-based rigid body packing optimization.
Uses specific 11-hexagon topologies and optimizes global rotation/translation
against the outer hexagon. Much faster and more robust than generic SA.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon as a numpy array (6,2)."""
    thetas = angle_rad + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(thetas), cy + side * np.sin(thetas)])

def compute_required_side_vectorized(centers, alpha_rad, cx, cy, beta_rad):
    """
    Compute minimum outer side length to contain all inner hexagons.
    centers: (N, 2) array of relative centers.
    alpha_rad: Global rotation of the inner cluster.
    cx, cy: Center of the outer hexagon.
    beta_rad: Rotation of the outer hexagon.
    """
    # Rotate centers by alpha
    c_alpha = np.cos(alpha_rad)
    s_alpha = np.sin(alpha_rad)
    R_inner = np.array([[c_alpha, -s_alpha], [s_alpha, c_alpha]])
    rotated_centers = centers @ R_inner.T
    
    # Translate to outer center frame
    # The outer hexagon is centered at (cx, cy) in the global frame.
    # We work in a frame where outer center is (0,0) for projection calculation?
    # No, let's keep it simple.
    # Inner hexagon i has center C_i_global = rotated_centers[i] + (0,0)? 
    # Wait, the "centers" are relative to the cluster center.
    # Let's define the cluster center as (0,0) in the local frame.
    # After rotation, centers are at rotated_centers.
    # We need to place this cluster such that it fits in the outer hexagon.
    # Let the outer hexagon center be at (cx, cy) in the global frame.
    # Let the cluster center be at (0,0) in the global frame? No, that's not right.
    
    # Let's define the configuration in the "Outer Hexagon Frame".
    # Outer Hexagon: Center (0,0), Rotation beta_rad, Side S.
    # Inner Cluster: Centers at P_i. 
    # We want to find S_min such that all inner hexagons (centered at P_i, rotated by alpha_rad) 
    # are inside the outer hexagon.
    
    # But we also have freedom to translate the cluster?
    # Yes, the cluster can be translated relative to the outer hexagon center.
    # Let's add translation (tx, ty) to the rotated centers.
    # So P_i = rotated_centers[i] + (tx, ty).
    
    # However, the function signature above doesn't have tx, ty.
    # Let's modify the logic: The optimization will handle tx, ty.
    # For now, let's assume the function takes the final global centers.
    
    # Actually, let's rewrite this helper to take global centers directly.
    pass

def compute_side_from_global_centers(global_centers, outer_center, outer_angle_deg):
    """
    Fast vectorized computation of required side length.
    global_centers: (N, 2) array of (x, y) for each inner hexagon center.
    outer_center: (x, y) tuple.
    outer_angle_deg: Rotation of outer hexagon in degrees.
    """
    if len(global_centers) == 0:
        return 0.0
    
    ox, oy = outer_center
    normals = np.array([[math.cos(math.radians(30 + i * 60)),
                         math.sin(math.radians(30 + i * 60))] for i in range(6)])
    
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    R_outer = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    
    # Vertices of each inner hexagon
    # Since all inner hexagons are identical and we can rotate them individually?
    # The task says "each inner hexagon may translate and rotate".
    # However, in a tight packing, the orientation of each hexagon is often constrained 
    # by its neighbors. For a triangular lattice packing, they all usually have the 
    # same orientation. Let's assume they all have the same orientation alpha_inner.
    # We need to include alpha_inner in the optimization.
    
    # This function needs to know the inner orientation.
    # Let's add an argument for inner_angle_deg.
    pass

def compute_side_fast(inner, outer_center, outer_angle_deg):
    """
    Compute minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_degrees)
    """
    if not inner:
        return 0.0
    ox, oy = outer_center
    normals = np.array([[math.cos(math.radians(30 + i * 60)),
                         math.sin(math.radians(30 + i * 60))] for i in range(6)])
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    R = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    
    max_proj = -np.inf
    for (ix, iy, iang) in inner:
        verts = hex_vertices(ix, iy, 1.0, math.radians(iang))
        local = (verts - np.array([ox, oy])) @ R.T
        projs = local @ normals.T
        cur = projs.max()
        if cur > max_proj:
            max_proj = cur
    if max_proj < 0:
        max_proj = 0.0
    return 2.0 * max_proj / SQRT3

def sat_overlap(v1, v2, eps=1e-9):
    """SAT overlap check for two convex polygons (lists of (x,y))."""
    axes = []
    for poly in (v1, v2):
        for i in range(len(poly)):
            p1 = poly[i]; p2 = poly[(i + 1) % len(poly)]
            e = (p2[0] - p1[0], p2[1] - p1[1])
            n = (-e[1], e[0])
            ln = math.hypot(n[0], n[1])
            if ln > 1e-12:
                axes.append((n[0] / ln, n[1] / ln))
    for ax in axes:
        min1 = min(p[0] * ax[0] + p[1] * ax[1] for p in v1)
        max1 = max(p[0] * ax[0] + p[1] * ax[1] for p in v1)
        min2 = min(p[0] * ax[0] + p[1] * ax[1] for p in v2)
        max2 = max(p[0] * ax[0] + p[1] * ax[1] for p in v2)
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    return True

def check_disjoint(inner, eps=1e-6):
    """Pairwise SAT check that all inner hexagons are disjoint."""
    n = len(inner)
    polys = []
    for (x, y, a) in inner:
        polys.append(hex_vertices(x, y, 1.0, math.radians(a)))
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap(polys[i], polys[j], eps):
                return False
    return True

def get_topology_centers(topo_id):
    """
    Return relative centers for a specific 11-hexagon topology.
    Centers are on a triangular lattice with spacing sqrt(3).
    """
    s = SQRT3
    if topo_id == 0:
        # Flower: 1 center, 6 ring, 4 outer
        # Axial coords: (q, r) -> x = s * 1.5 * q, y = s * sqrt(3) * (r + q/2)
        # Wait, standard axial to cartesian:
        # x = size * 3/2 * q
        # y = size * sqrt(3) * (r + q/2)
        # Here "size" is the distance between centers? No, side length is 1.
        # Distance between centers of touching hexagons is sqrt(3).
        # So axial step corresponds to distance sqrt(3).
        # Let's use axial coordinates (q, r) where step is 1.
        # x = sqrt(3) * 1.5 * q
        # y = sqrt(3) * sqrt(3) * (r + q/2) = 3 * (r + q/2) ? No.
        # Distance between (0,0) and (1,0) is sqrt(3).
        # Formula: x = sqrt(3) * 1.5 * q, y = sqrt(3) * (sqrt(3)/2) * (2r + q) ?
        # Let's stick to the standard:
        # x = sqrt(3) * 1.5 * q
        # y = sqrt(3) * (r + 0.5 * q) * sqrt(3) ? No.
        # y = 1.5 * sqrt(3) * (r + q/2) ? No.
        
        # Correct conversion for hexagon side length 1:
        # Center distance d = sqrt(3).
        # x = d * 1.5 * q
        # y = d * (r + q/2) * sqrt(3) ? No.
        # y = d * sqrt(3) / 2 * (2r + q) ?
        # Let's just use the vectors.
        # e1 = (sqrt(3), 0)
        # e2 = (sqrt(3)/2, 3/2)
        # pos = q * e1 + r * e2
        
        coords = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            # 4 outer ones attached to the ring
            # Attached to (1,0): (2, -1), (2, 0)? No, (2,0) is not touching (1,0) directly in a way that keeps it compact?
            # Let's pick 4 neighbors of the ring that don't overlap.
            # (2, -1), (1, 1), (-2, 1), (-1, -1)
            (2, -1), (1, 1), (-2, 1), (-1, -1)
        ]
    elif topo_id == 1:
        # 3-4-3 rows
        # Row 0: 3 hexes
        # Row 1: 4 hexes
        # Row 2: 3 hexes
        coords = [
            # Row 0 (y=0 relative to some base)
            (-1, 0), (0, 0), (1, 0),
            # Row 1
            (-1, 1), (0, 1), (1, 1), (2, 1),
            # Row 2
            (-1, 2), (0, 2), (1, 2)
        ]
        # Note: This might not be a valid polyhex if they don't touch correctly.
        # Let's verify connectivity.
        # (-1,0) touches (0,0), (0,1), (-1,1).
        # (0,0) touches (-1,0), (1,0), (0,1), (1,1)? No, (1,0) and (0,1) are distance sqrt(3) apart?
        # Distance (1,0) to (0,1): q diff -1, r diff 1. dist = sqrt(3). Yes.
        # So this is a valid connected cluster.
    elif topo_id == 2:
        # 4-3-4 rows
        coords = [
            # Row 0
            (-1, 0), (0, 0), (1, 0), (2, 0),
            # Row 1
            (-1, 1), (0, 1), (1, 1),
            # Row 2
            (-1, 2), (0, 2), (1, 2), (2, 2)
        ]
    else:
        return []
        
    centers = []
    for q, r in coords:
        x = SQRT3 * 1.5 * q
        y = SQRT3 * (r + 0.5 * q) * SQRT3 / SQRT3 # Wait, y component of e2 is 1.5?
        # e1 = (sqrt(3), 0)
        # e2 = (sqrt(3)/2, 1.5)
        # pos = q * e1 + r * e2
        x = SQRT3 * q + SQRT3/2 * r
        y = 0 + 1.5 * r
        centers.append((x, y))
    return np.array(centers)

def construct():
    t0 = time.time()
    deadline = t0 + 140.0 # Safety margin
    
    topologies = [0, 1, 2]
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = (0, 0)
    best_outer_angle = 0.0
    
    for topo_id in topologies:
        if time.time() > deadline - 20:
            break
            
        rel_centers = get_topology_centers(topo_id)
        if len(rel_centers) != N:
            continue
            
        # Center the topology at origin
        center_of_mass = np.mean(rel_centers, axis=0)
        rel_centers -= center_of_mass
        
        # Optimization variables:
        # alpha: rotation of inner cluster (0 to 60 deg)
        # beta: rotation of outer hexagon (0 to 60 deg)
        # tx, ty: translation of inner cluster relative to outer center
        # Note: Due to symmetry, we can restrict alpha and beta to [0, 60).
        
        def objective(params):
            alpha, beta, tx, ty = params
            alpha_rad = math.radians(alpha)
            beta_rad = math.radians(beta)
            
            # Rotate relative centers
            c_a = math.cos(alpha_rad)
            s_a = math.sin(alpha_rad)
            R = np.array([[c_a, -s_a], [s_a, c_a]])
            rotated = rel_centers @ R.T
            
            # Translate
            global_centers = rotated + np.array([tx, ty])
            
            # Construct inner list for compute_side_fast
            # Assume all inner hexagons have the same orientation alpha
            # Is this optimal? For a rigid cluster, yes.
            inner_list = []
            for i in range(N):
                inner_list.append([float(global_centers[i, 0]), float(global_centers[i, 1]), alpha])
            
            side = compute_side_fast(inner_list, (0.0, 0.0), beta)
            return side
        
        # Multi-start optimization
        best_local_val = float('inf')
        best_local_params = None
        
        # Grid search for initial guesses
        for a0 in [0, 15, 30, 45]:
            for b0 in [0, 15, 30, 45]:
                for tx0 in [-1, 0, 1]:
                    for ty0 in [-1, 0, 1]:
                        p0 = np.array([a0, b0, tx0, ty0])
                        try:
                            res = minimize(objective, p0, method='Nelder-Mead',
                                           options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
                            if res.fun < best_local_val:
                                best_local_val = res.fun
                                best_local_params = res.x
                        except Exception:
                            pass
                        
        if best_local_params is not None:
            # Final polish
            try:
                res_final = minimize(objective, best_local_params, method='Nelder-Mead',
                                     options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-12})
                if res_final.fun < best_local_val:
                    best_local_val = res_final.fun
                    best_local_params = res_final.x
            except Exception:
                pass
            
            # Extract solution
            alpha, beta, tx, ty = best_local_params
            alpha_rad = math.radians(alpha)
            c_a = math.cos(alpha_rad)
            s_a = math.sin(alpha_rad)
            R = np.array([[c_a, -s_a], [s_a, c_a]])
            rotated = rel_centers @ R.T
            global_centers = rotated + np.array([tx, ty])
            
            inner_list = []
            for i in range(N):
                inner_list.append([float(global_centers[i, 0]), float(global_centers[i, 1]), alpha])
            
            # Verify disjointness
            if check_disjoint(inner_list, eps=1e-5):
                if best_local_val < best_side:
                    best_side = best_local_val
                    best_inner = inner_list
                    best_outer_center = (0.0, 0.0)
                    best_outer_angle = beta
    
    if best_inner is None:
        # Fallback to simple lattice
        axial = [(0, 0),
                 (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
                 (2, -1), (1, 1), (-2, 1), (-1, -1)]
        s0 = 1.0 + 1e-7
        init_centers = [(SQRT3 * 1.5 * q, SQRT3 * (r + q / 2.0)) for q, r in axial]
        best_inner = [[x, y, 0.0] for x, y in init_centers]
        cx0 = sum(c[0] for c in init_centers) / len(init_centers)
        cy0 = sum(c[1] for c in init_centers) / len(init_centers)
        best_outer_center = (cx0, cy0)
        best_outer_angle = 0.0
        best_side = compute_side_fast(best_inner, best_outer_center, best_outer_angle)
        
    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
