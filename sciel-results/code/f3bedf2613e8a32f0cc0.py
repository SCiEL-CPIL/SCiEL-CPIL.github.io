import time
import numpy as np
from scipy.optimize import minimize

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 115.0  # Safety margin

def _get_rigid_body_positions(cluster_angle_deg):
    """
    Generate positions of the 11 unit hexagons in a rigid cluster.
    The cluster is centered at the origin.
    Internal angles are all 0 relative to the cluster frame, but the whole cluster is rotated.
    Wait, the inner hexagons can rotate individually? The task says "each inner hexagon may translate and rotate".
    However, the "rigid body" strategy assumes the internal geometry is fixed.
    For hexagons, the optimal packing is usually on a triangular lattice where all are aligned.
    So we assume all inner hexagons have the same orientation as the cluster frame.
    """
    sqrt3 = np.sqrt(3)
    # Relative positions in the cluster frame (unrotated)
    # 0: Center
    # 1-6: Ring 1
    # 7-10: Ring 2 (Gaps)
    
    rel_pos = np.zeros((N, 2))
    rel_pos[0] = [0, 0]
    
    for i in range(6):
        a = i * np.pi / 3
        rel_pos[i+1] = [2*sqrt3 * np.cos(a), 2*sqrt3 * np.sin(a)]
        
    ring2_angles = [np.pi/6, 5*np.pi/6, 7*np.pi/16, 11*np.pi/6] # Correction: 7pi/6, 11pi/6
    ring2_angles = [np.pi/6, 5*np.pi/6, 7*np.pi/6, 11*np.pi/6]
    for i in range(4):
        a = ring2_angles[i]
        rel_pos[i+7] = [6.0 * np.cos(a), 6.0 * np.sin(a)]
        
    # Rotate the cluster
    theta = np.radians(cluster_angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    
    return rel_pos @ rot.T

def _min_outer_side_rigid(cluster_angle_deg, outer_center, outer_angle_deg):
    """
    Compute the minimum side length of the outer hexagon to contain the rigid cluster.
    """
    pos = _get_rigid_body_positions(cluster_angle_deg)
    cx, cy = outer_center
    
    # Vertices of all inner hexagons
    # Since they are aligned with the cluster, their local angles are 0 relative to cluster frame.
    # But wait, the inner hexagons are rotated by cluster_angle_deg.
    # So their global angle is cluster_angle_deg.
    # Vertices of a unit hexagon with angle alpha:
    # v_k = center + (cos(alpha + k*pi/3), sin(alpha + k*pi/3))
    
    base_angles = np.array([0, 1, 2, 3, 4, 5]) * np.pi / 3
    cluster_rad = np.radians(cluster_angle_deg)
    
    # Global angles for vertices
    total_angles = cluster_rad + base_angles
    
    offsets_x = np.cos(total_angles)
    offsets_y = np.sin(total_angles)
    
    verts_x = pos[:, 0:1] + offsets_x
    verts_y = pos[:, 1:2] + offsets_y
    
    all_verts_x = verts_x.ravel()
    all_verts_y = verts_y.ravel()
    
    dx = all_verts_x - cx
    dy = all_verts_y - cy
    
    a_outer = np.radians(outer_angle_deg)
    ca, sa = np.cos(a_outer), np.sin(a_outer)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    
    sqrt3 = np.sqrt(3)
    
    # For a regular hexagon centered at origin with side S and rotation 0,
    # the condition for a point (x,y) to be inside is:
    # max(|x|, |x + y/sqrt(3)|, |x - y/sqrt(3)|) <= S
    # Here (lx, ly) are coordinates in the outer hexagon's frame.
    
    c1 = np.abs(lx)
    c2 = np.abs(lx + ly / sqrt3)
    c3 = np.abs(lx - ly / sqrt3)
    
    required = np.maximum(c1, np.maximum(c2, c3))
    return float(np.max(required))

def _objective_rigid(params):
    """
    Objective: Minimize the outer side length.
    Params: [cluster_angle_deg, outer_cx, outer_cy, outer_angle_deg]
    """
    cluster_angle = params[0]
    outer_cx = params[1]
    outer_cy = params[2]
    outer_angle = params[3]
    
    side = _min_outer_side_rigid(cluster_angle, (outer_cx, outer_cy), outer_angle)
    
    # Add a small regularization to keep the outer center near the cluster center (0,0)
    # to avoid trivial solutions where the outer hexagon is huge and offset.
    # Actually, the side length will naturally penalize offset if it's large.
    # But to help the optimizer, we can add a small penalty for moving the center.
    reg = 0.01 * (outer_cx**2 + outer_cy**2)
    
    return side + reg

def construct():
    global START_TIME, DEADLINE
    
    best_score = np.inf
    best_params = None
    
    # Initial guess
    # Cluster angle 0, Outer center (0,0), Outer angle 0
    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    
    # Try a few initial rotations for the cluster to escape symmetry
    initial_angles = [0.0, 10.0, 20.0, 30.0, 45.0, 60.0]
    
    for ang in initial_angles:
        if time.time() > DEADLINE - 5.0:
            break
            
        x_start = np.array([ang, 0.0, 0.0, 0.0])
        
        try:
            # Use Nelder-Mead for non-smooth-ish function (max is non-diff, but scipy handles it okay usually)
            # Or use L-BFGS-B if we smooth it?
            # The function is continuous but not differentiable at vertices.
            # Nelder-Mead is robust for this.
            
            res = minimize(_objective_rigid, x_start, method='Nelder-Mead',
                           options={'maxiter': 1000, 'fatol': 1e-12, 'xatol': 1e-12})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
                
        except Exception:
            pass

    if best_params is None:
        best_params = x0
        
    # Extract best solution
    cluster_angle = best_params[0]
    outer_cx = best_params[1]
    outer_cy = best_params[2]
    outer_angle = best_params[3]
    
    # Generate inner positions
    pos = _get_rigid_body_positions(cluster_angle)
    
    # Inner angles are all equal to cluster_angle
    inner_list = [[float(pos[i,0]), float(pos[i,1]), float(cluster_angle)] for i in range(N)]
    
    # Calculate final side length
    final_side = _min_outer_side_rigid(cluster_angle, (outer_cx, outer_cy), outer_angle)
    
    return inner_list, (float(outer_cx), float(outer_cy)), float(final_side), float(outer_angle)

if __name__ == "__main__":
    construct()
