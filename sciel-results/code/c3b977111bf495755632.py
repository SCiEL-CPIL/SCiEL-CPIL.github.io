"""
Strategy: Rigid Hexagonal Lattice Topology Optimization.

Key Differences from Current Program:
1. Problem Framing: Assumes the optimal packing of 11 unit hexagons is a rigid
   cluster based on the standard hexagonal tiling (1 center + 6 ring + 4 gap-fillers).
   This eliminates the need for complex collision avoidance optimization, as the
   topology guarantees disjointness.
2. Optimization Variables: Reduces the search space from 33+ dimensions (positions,
   angles) to 3-4 dimensions (cluster rotation, outer rotation, outer center, outer side).
3. Objective Function: Uses the exact support function of the rigid cluster's vertices
   to determine the minimum enclosing regular hexagon. This is a smooth, convex
   maximization of linear functions, amenable to high-precision grid search and
   local refinement.
4. Search Method: Employs a coarse-to-fine grid search over the rotational degrees
   of freedom, followed by a local Nelder-Mead refinement for the outer hexagon's
   position and size. This is far more robust than gradient-based methods on the
   non-convex landscape of the original formulation.
"""

import time
import numpy as np
from scipy.optimize import minimize

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 115.0  # Safety margin before 150s kill

def _get_cluster_vertices(cluster_angle_deg):
    """
    Generate vertices for the 11-hexagon rigid cluster.
    Topology:
    - Center: (0,0)
    - Ring 1: 6 hexagons at distance 2*sqrt(3) from center.
    - Ring 2: 4 hexagons at distance 6 from center, in the gaps (30, 150, 210, 330 deg).
    All inner hexagons are aligned with the cluster frame.
    """
    sqrt3 = np.sqrt(3)
    
    # Centers of the 11 hexagons in the cluster frame
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
    
    # Vertices of a single unit hexagon centered at origin, aligned with axes
    # Angles: 0, 60, 120, 180, 240, 300
    base_angles = np.array([0, 1, 2, 3, 4, 5]) * np.pi / 3
    unit_verts_x = np.cos(base_angles)
    unit_verts_y = np.sin(base_angles)
    
    # All vertices for all 11 hexagons in cluster frame
    # Shape: (11, 6, 2)
    verts = np.zeros((N, 6, 2))
    for i in range(N):
        verts[i, :, 0] = centers[i, 0] + unit_verts_x
        verts[i, :, 1] = centers[i, 1] + unit_verts_y
    
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
    the support function in direction theta is S * h(theta - phi), where h is
    the support function of a unit hexagon centered at origin.
    
    Actually, it's easier to work in the outer hexagon's local frame.
    Let the outer hexagon be centered at (cx, cy) and rotated by phi.
    A point p is inside if, in the local frame (rotated by -phi, translated by -c),
    the point satisfies the unit hexagon constraints scaled by S.
    
    Unit hexagon (side 1) constraints in local frame (u, v):
    |u| <= 1
    |u + v/sqrt(3)| <= 1
    |u - v/sqrt(3)| <= 1
    
    So for a point p_i, transformed to local frame q_i = R(-phi)(p_i - c):
    S >= max_i max( |q_i.x|, |q_i.x + q_i.y/sqrt(3)|, |q_i.x - q_i.y/sqrt(3)| )
    
    We want to minimize S.
    S(c, phi) = max_i f(p_i, c, phi)
    
    This is a convex function of c for fixed phi? No, it's the max of convex functions, so it's convex.
    We can minimize this over c (2D) for a fixed phi.
    """
    phi = np.radians(outer_angle_deg)
    ca, sa = np.cos(phi), np.sin(phi)
    
    # Rotation matrix for transforming global to local frame
    # q = R(-phi) (p - c)
    # R(-phi) = [[cos(phi), sin(phi)], [-sin(phi), cos(phi)]]
    
    # Let p_i = (x_i, y_i)
    # q_i.x = ca * (x_i - cx) + sa * (y_i - cy)
    # q_i.y = -sa * (x_i - cx) + ca * (y_i - cy)
    
    # We want to minimize max over i and k of |L_k(q_i)| where L_k are the 3 linear forms.
    # L_1(q) = q.x
    # L_2(q) = q.x + q.y/sqrt(3)
    # L_3(q) = q.x - q.y/sqrt(3)
    
    # Substituting q:
    # L_1 = ca*x - ca*cx + sa*y - sa*cy
    # L_2 = (ca - sa/sqrt3)*x - (ca - sa/sqrt3)*cx + (sa + ca/sqrt3)*y - (sa + ca/sqrt3)*cy
    # L_3 = (ca + sa/sqrt3)*x - (ca + sa/sqrt3)*cx + (sa - ca/sqrt3)*y - (sa - ca/sqrt3)*cy
    
    # Let A_k be the vector coefficients for (x, y) and B_k be the constant terms involving c.
    # L_k = A_k . p_i - A_k . c
    
    # We want to minimize max_i |A_k . p_i - A_k . c| for k=1,2,3.
    # This is equivalent to finding the center c that minimizes the width of the projection
    # of the point set in the 3 directions defined by the hexagon normals.
    
    # For a single direction n, the optimal center projection is the midpoint of the min and max projections.
    # The width is max(n.p) - min(n.p).
    # However, c affects all 3 directions simultaneously.
    
    # Let's use a simple optimization for c for a fixed phi.
    # The function is convex, so we can use a small grid or Nelder-Mead.
    
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
    res = minimize(objective, c0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8})
    
    return res.fun

def construct():
    global START_TIME, DEADLINE
    
    best_score = np.inf
    best_params = None # (cluster_angle, outer_angle, c_x, c_y)
    
    # Coarse Grid Search over rotations
    # The system has 6-fold symmetry for the cluster and outer, but the gap-fillers
    # in the cluster break the 6-fold symmetry of the cluster itself?
    # Cluster: Center + 6 (6-fold sym) + 4 (3-fold sym? No, 4 points at 30, 150, 210, 330).
    # The set of 4 points has symmetry of order 2 (180 deg rotation).
    # So the cluster has 2-fold symmetry.
    # The outer hexagon has 6-fold symmetry.
    # We should search cluster_angle in [0, 180) and outer_angle in [0, 60).
    
    # To be safe, search [0, 360) for cluster and [0, 60) for outer with coarse steps.
    
    cluster_steps = 36
    outer_steps = 12
    
    cluster_angles = np.linspace(0, 360, cluster_steps, endpoint=False)
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
                
            side = _min_outer_side_and_center(verts, oa)
            
            if side < best_score:
                # We need the center c that achieved this.
                # _min_outer_side_and_center doesn't return c.
                # Let's modify it to return c or re-run to get c.
                # For efficiency, let's just store the score and angles, and find c later for the best.
                # Or better, modify the function.
                
                # To avoid changing the function signature in this thought process,
                # I will re-run the minimization for the best candidate at the end.
                # But for tracking, I need to know which (ca, oa) was best.
                
                best_score = side
                best_params = (ca, oa)
                
    # Refine the best found
    if best_params is not None:
        ca_best, oa_best = best_params
        
        # Local refinement around the best grid point
        # Use Nelder-Mead on (delta_ca, delta_oa, cx, cy)
        
        verts_best = _get_cluster_vertices(ca_best)
        c0 = np.mean(verts_best, axis=0)
        
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
        
        x0 = np.array([0.0, 0.0, c0[0], c0[1]])
        
        try:
            res = minimize(full_objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            
            if res.fun < best_score:
                best_score = res.fun
                d_ca, d_oa, cx, cy = res.x
                ca_final = ca_best + d_ca
                oa_final = oa_best + d_oa
            else:
                # Fall back to grid best, need to find c for it
                ca_final = ca_best
                oa_final = oa_best
                # Find c for this config
                def obj_c(c):
                    cx, cy = c
                    phi = np.radians(oa_final)
                    ca_r, sa_r = np.cos(phi), np.sin(phi)
                    dx = verts_best[:, 0] - cx
                    dy = verts_best[:, 1] - cy
                    lx = ca_r * dx + sa_r * dy
                    ly = -sa_r * dx + ca_r * dy
                    sqrt3 = np.sqrt(3)
                    v1 = np.abs(lx)
                    v2 = np.abs(lx + ly / sqrt3)
                    v3 = np.abs(lx - ly / sqrt3)
                    return np.max(np.maximum(v1, np.maximum(v2, v3)))
                
                res_c = minimize(obj_c, c0, method='Nelder-Mead', options={'maxiter': 200})
                cx, cy = res_c.x
                
        except Exception:
            ca_final = ca_best
            oa_final = oa_best
            cx, cy = c0
            
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
        side = _min_outer_side_and_center(verts, 0.0)
        
        return inner_list, (0.0, 0.0), float(side), 0.0

if __name__ == "__main__":
    construct()
