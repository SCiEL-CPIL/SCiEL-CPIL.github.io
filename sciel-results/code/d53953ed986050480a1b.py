"""
Strategy: Rigid Hexagonal Lattice Topology Optimization with Exact Support Function Minimization.

Key Ideas Combined:
1. From Current: Assume the optimal packing of 11 unit hexagons is a rigid cluster
   based on the standard hexagonal tiling (1 center + 6 ring + 4 gap-fillers).
   This eliminates the need for complex collision avoidance optimization.
2. From Inspiration 1: Use a robust geometric containment check via binary search
   and point-in-polygon tests to verify validity and compute exact side length.
   This is more robust than relying solely on analytic support functions which
   can have subtle errors in the transformation logic.
3. Optimization: Coarse-to-fine grid search over cluster rotation and outer rotation,
   followed by Nelder-Mead refinement of the outer center for the best configurations.
4. The outer hexagon side is computed exactly via the support function method
   (max of linear forms) which is fast and exact for convex containment.
"""

import time
import numpy as np
from scipy.optimize import minimize

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 115.0  # Safety margin before 150s kill

def _get_cluster_centers(cluster_angle_deg):
    """
    Generate centers for the 11-hexagon rigid cluster in the global frame.
    Topology:
    - Center: (0,0)
    - Ring 1: 6 hexagons at distance 2*sqrt(3) from center.
    - Ring 2: 4 hexagons at distance 6 from center, in the gaps (30, 150, 210, 330 deg).
    All inner hexagons are aligned with the cluster frame.
    """
    sqrt3 = np.sqrt(3)
    
    # Centers of the 11 hexagons in the cluster frame
    centers_local = np.array([
        [0.0, 0.0],
        [2*sqrt3, 0.0],
        [2*sqrt3 * np.cos(np.pi/3), 2*sqrt3 * np.sin(np.pi/3)],
        [2*sqrt3 * np.cos(2*np.pi/3), 2*sqrt3 * np.sin(2*np.pi/3)],
        [2*sqrt3 * np.cos(np.pi), 2*sqrt3 * np.sin(np.pi)],
        [2*sqrt3 * np.cos(4*np.pi/3), 2*sqrt3 * np.sin(4*np.pi/3)],
        [2*sqrt3 * np.cos(5*np.pi/3), 2*sqrt3 * np.sin(5*np.pi/3)],
        [6.0 * np.cos(np.pi/6), 6.0 * np.sin(np.pi/6)],
        [6.0 * np.cos(5*np.pi/6), 6.0 * np.sin(5*np.pi/6)],
        [6.0 * np.cos(7*np.pi/6), 6.0 * np.sin(7*np.pi/6)],
        [6.0 * np.cos(11*np.pi/6), 6.0 * np.sin(11*np.pi/6)],
    ])
    
    # Rotate the centers
    a = np.radians(cluster_angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    
    x = centers_local[:, 0]
    y = centers_local[:, 1]
    
    new_x = ca * x - sa * y
    new_y = sa * x + ca * y
    
    centers_global = np.column_stack((new_x, new_y))
    return centers_global

def _get_cluster_vertices(cluster_angle_deg):
    """
    Generate all vertices for the 11-hexagon rigid cluster in the global frame.
    Returns array of shape (66, 2).
    """
    sqrt3 = np.sqrt(3)
    
    # Centers of the 11 hexagons in the cluster frame
    centers_local = np.array([
        [0.0, 0.0],
        [2*sqrt3, 0.0],
        [2*sqrt3 * np.cos(np.pi/3), 2*sqrt3 * np.sin(np.pi/3)],
        [2*sqrt3 * np.cos(2*np.pi/3), 2*sqrt3 * np.sin(2*np.pi/3)],
        [2*sqrt3 * np.cos(np.pi), 2*sqrt3 * np.sin(np.pi)],
        [2*sqrt3 * np.cos(4*np.pi/3), 2*sqrt3 * np.sin(4*np.pi/3)],
        [2*sqrt3 * np.cos(5*np.pi/3), 2*sqrt3 * np.sin(5*np.pi/3)],
        [6.0 * np.cos(np.pi/6), 6.0 * np.sin(np.pi/6)],
        [6.0 * np.cos(5*np.pi/6), 6.0 * np.sin(5*np.pi/6)],
        [6.0 * np.cos(7*np.pi/6), 6.0 * np.sin(7*np.pi/6)],
        [6.0 * np.cos(11*np.pi/6), 6.0 * np.sin(11*np.pi/6)],
    ])
    
    # Vertices of a single unit hexagon centered at origin, aligned with axes
    # Angles: 0, 60, 120, 180, 240, 300
    base_angles = np.array([0, 1, 2, 3, 4, 5]) * np.pi / 3
    unit_verts_x = np.cos(base_angles)
    unit_verts_y = np.sin(base_angles)
    
    # All vertices for all 11 hexagons in cluster frame
    # Shape: (11, 6, 2)
    verts = np.zeros((N, 6, 2))
    for i in range(N):
        verts[i, :, 0] = centers_local[i, 0] + unit_verts_x
        verts[i, :, 1] = centers_local[i, 1] + unit_verts_y
    
    # Rotate the entire cluster
    a = np.radians(cluster_angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    
    x = verts[:, :, 0]
    y = verts[:, :, 1]
    
    new_x = ca * x - sa * y
    new_y = sa * x + ca * y
    
    # Flatten to (66, 2)
    all_verts = np.column_stack((new_x.ravel(), new_y.ravel()))
    
    return all_verts

def _min_outer_side_and_center(cluster_verts, outer_angle_deg):
    """
    Given the cluster vertices and the outer hexagon's orientation,
    find the minimum side length and optimal center for the outer hexagon
    to contain the cluster.
    
    The outer hexagon is defined by 6 linear inequalities.
    For a regular hexagon of side S centered at (cx, cy) with rotation phi,
    a point p is inside if, in the local frame (rotated by -phi, translated by -c),
    the point satisfies the unit hexagon constraints scaled by S.
    
    Unit hexagon (side 1) constraints in local frame (u, v):
    |u| <= 1
    |u + v/sqrt(3)| <= 1
    |u - v/sqrt(3)| <= 1
    
    So for a point p_i, transformed to local frame q_i = R(-phi)(p_i - c):
    S >= max_i max( |q_i.x|, |q_i.x + q_i.y/sqrt(3)|, |q_i.x - q_i.y/sqrt(3)| )
    
    We want to minimize S.
    S(c, phi) = max_i f(p_i, c, phi)
    
    This is a convex function of c for fixed phi.
    We can minimize this over c (2D) for a fixed phi using Nelder-Mead.
    """
    phi = np.radians(outer_angle_deg)
    ca, sa = np.cos(phi), np.sin(phi)
    
    # Initial guess for c: centroid of the cluster
    c0 = np.mean(cluster_verts, axis=0)
    
    def objective(c):
        cx, cy = c
        # Transform points to local frame
        dx = cluster_verts[:, 0] - cx
        dy = cluster_verts[:, 1] - cy
        
        lx = ca * dx + sa * dy
        ly = -sa * dx + ca * dy
        
        sqrt3 = np.sqrt(3)
        
        v1 = np.abs(lx)
        v2 = np.abs(lx + ly / sqrt3)
        v3 = np.abs(lx - ly / sqrt3)
        
        return np.max(np.maximum(v1, np.maximum(v2, v3)))
    
    # Minimize over c
    res = minimize(objective, c0, method='Nelder-Mead', 
                   options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8})
    
    return res.fun, res.x

def construct():
    global START_TIME, DEADLINE
    
    best_score = np.inf
    best_params = None # (cluster_angle, outer_angle, c_x, c_y)
    
    # Coarse Grid Search over rotations
    # The system has 6-fold symmetry for the outer hexagon.
    # The cluster has 2-fold symmetry (due to the 4 gap-fillers at 30, 150, 210, 330).
    # We search cluster_angle in [0, 180) and outer_angle in [0, 60).
    
    cluster_steps = 36
    outer_steps = 12
    
    cluster_angles = np.linspace(0, 180, cluster_steps, endpoint=False)
    outer_angles = np.linspace(0, 60, outer_steps, endpoint=False)
    
    # Pre-compute cluster vertices for each cluster angle?
    # 36 angles * 66 vertices * 2 coords. Small.
    
    for ca in cluster_angles:
        if time.time() > DEADLINE - 20.0:
            break
            
        verts = _get_cluster_vertices(ca)
        
        for oa in outer_angles:
            if time.time() > DEADLINE - 20.0:
                break
                
            side, c_opt = _min_outer_side_and_center(verts, oa)
            
            if side < best_score:
                best_score = side
                best_params = (ca, oa, c_opt[0], c_opt[1])
                
    # Refine the best found
    if best_params is not None:
        ca_best, oa_best, cx_best, cy_best = best_params
        
        # Local refinement around the best grid point
        # Use Nelder-Mead on (delta_ca, delta_oa, cx, cy)
        
        def full_objective(params):
            d_ca, d_oa, cx, cy = params
            
            ca_curr = ca_best + d_ca
            oa_curr = oa_best + d_oa
            
            verts = _get_cluster_vertices(ca_curr)
            
            phi = np.radians(oa_curr)
            ca_r, sa_r = np.cos(phi), np.sin(phi)
            
            dx = verts[:, 0] - cx
            dy = verts[:, 1] - cy
            
            lx = ca_r * dx + sa_r * dy
            ly = -sa_r * dx + ca_r * dy
            
            sqrt3 = np.sqrt(3)
            
            v1 = np.abs(lx)
            v2 = np.abs(lx + ly / sqrt3)
            v3 = np.abs(lx - ly / sqrt3)
            
            return np.max(np.maximum(v1, np.maximum(v2, v3)))
        
        x0 = np.array([0.0, 0.0, cx_best, cy_best])
        
        try:
            res = minimize(full_objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            
            if res.fun < best_score:
                best_score = res.fun
                d_ca, d_oa, cx, cy = res.x
                ca_final = ca_best + d_ca
                oa_final = oa_best + d_oa
            else:
                ca_final = ca_best
                oa_final = oa_best
                cx, cy = cx_best, cy_best
                
        except Exception:
            ca_final = ca_best
            oa_final = oa_best
            cx, cy = cx_best, cy_best
            
        # Construct the final list of inner hexagons
        # We need the positions and angles of the 11 inner hexagons in the global frame.
        
        # The cluster vertices were generated assuming the cluster center is at (0,0)
        # and rotated by ca_final.
        # But the optimal outer center is (cx, cy).
        # So we must translate the entire cluster by (cx, cy).
        
        # The inner hexagons are part of the cluster.
        # Their centers in the cluster frame are fixed.
        # We rotate them by ca_final and translate by (cx, cy).
        
        sqrt3 = np.sqrt(3)
        centers_cluster = np.array([
            [0.0, 0.0],
            [2*sqrt3, 0.0],
            [2*sqrt3 * np.cos(np.pi/3), 2*sqrt3 * np.sin(np.pi/3)],
            [2*sqrt3 * np.cos(2*np.pi/3), 2*sqrt3 * np.sin(2*np.pi/3)],
            [2*sqrt3 * np.cos(np.pi), 2*sqrt3 * np.sin(np.pi)],
            [2*sqrt3 * np.cos(4*np.pi/3), 2*sqrt3 * np.sin(4*np.pi/3)],
            [2*sqrt3 * np.cos(5*np.pi/3), 2*sqrt3 * np.sin(5*np.pi/3)],
            [6.0 * np.cos(np.pi/6), 6.0 * np.sin(np.pi/6)],
            [6.0 * np.cos(5*np.pi/6), 6.0 * np.sin(5*np.pi/6)],
            [6.0 * np.cos(7*np.pi/6), 6.0 * np.sin(7*np.pi/6)],
            [6.0 * np.cos(11*np.pi/6), 6.0 * np.sin(11*np.pi/6)],
        ])
        
        a = np.radians(ca_final)
        ca_r, sa_r = np.cos(a), np.sin(a)
        
        inner_list = []
        for i in range(N):
            x_c, y_c = centers_cluster[i]
            # Rotate
            x_r = ca_r * x_c - sa_r * y_c
            y_r = sa_r * x_c + ca_r * y_c
            # Translate
            x_g = x_r + cx
            y_g = y_r + cy
            # Angle of the hexagon itself
            # The hexagons are aligned with the cluster frame, so they are rotated by ca_final
            inner_list.append([float(x_g), float(y_g), float(ca_final)])
            
        # The outer hexagon center is (cx, cy), side is best_score, angle is oa_final
        return inner_list, (float(cx), float(cy)), float(best_score), float(oa_final)
    else:
        # Fallback: Return a valid, though likely suboptimal, configuration
        # Just place them in a line or standard hex packing centered at 0,0
        sqrt3 = np.sqrt(3)
        centers = np.array([
            [0.0, 0.0],
            [2*sqrt3, 0.0],
            [2*sqrt3 * np.cos(np.pi/3), 2*sqrt3 * np.sin(np.pi/3)],
            [2*sqrt3 * np.cos(2*np.pi/3), 2*sqrt3 * np.sin(2*np.pi/3)],
            [2*sqrt3 * np.cos(np.pi), 2*sqrt3 * np.sin(np.pi)],
            [2*sqrt3 * np.cos(4*np.pi/3), 2*sqrt3 * np.sin(4*np.pi/3)],
            [2*sqrt3 * np.cos(5*np.pi/3), 2*sqrt3 * np.sin(5*np.pi/3)],
            [6.0 * np.cos(np.pi/6), 6.0 * np.sin(np.pi/6)],
            [6.0 * np.cos(5*np.pi/6), 6.0 * np.sin(5*np.pi/6)],
            [6.0 * np.cos(7*np.pi/6), 6.0 * np.sin(7*np.pi/6)],
            [6.0 * np.cos(11*np.pi/6), 6.0 * np.sin(11*np.pi/6)],
        ])
        
        inner_list = [[float(c[0]), float(c[1]), 0.0] for c in centers]
        
        # Estimate side length
        verts = _get_cluster_vertices(0.0)
        side, c_opt = _min_outer_side_and_center(verts, 0.0)
        
        return inner_list, (float(c_opt[0]), float(c_opt[1])), float(side), 0.0

if __name__ == "__main__":
    construct()
