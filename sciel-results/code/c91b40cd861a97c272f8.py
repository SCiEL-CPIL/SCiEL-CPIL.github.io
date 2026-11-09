# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 630s limit
    
    SQRT3 = np.sqrt(3.0)
    
    # Precompute hexagon vertices for unit side (angle 0)
    v_angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    hex_verts = np.array([[np.cos(a), np.sin(a)] for a in v_angles])
    
    # Edge normals for the 3 unique directions (due to symmetry)
    # The 6 edge normals are at 30, 90, 150, 210, 270, 330 degrees
    # But for projection separation, we just need to check all 12 (6 from each hex)
    # Actually, for two hexagons with the same orientation, the 12 normals are just the 6 unique directions
    # The task says: "on at least one of their 12 edge normals the projected intervals are apart"
    # If both hexagons have angle 0, the 12 normals are:
    # Hex 1 normals: 30, 90, 150, 210, 270, 330
    # Hex 2 normals: 30, 90, 150, 210, 270, 330
    # So we just need to check these 6 directions.
    
    normal_angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    normals = np.array([[np.cos(a), np.sin(a)] for a in normal_angles])
    
    def compute_separation_penalty(pts):
        """
        Compute penalty for overlapping hexagons.
        For each pair, check if they are separated on any of the 6 normal directions.
        If not separated on any, add penalty based on minimum separation deficit.
        """
        penalty = 0.0
        n_pts = len(pts)
        
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                # Get vertices of both hexagons
                v1 = pts[i] + hex_verts
                v2 = pts[j] + hex_verts
                
                # Check separation on each of the 6 normals
                is_separated = False
                min_gap = np.inf
                
                for k in range(6):
                    n_k = normals[k]
                    proj1 = v1 @ n_k
                    proj2 = v2 @ n_k
                    
                    gap1 = proj1.max() - proj2.min()
                    gap2 = proj2.max() - proj1.min()
                    
                    # If either gap is > 1e-6, they are separated on this normal
                    # The task requires separation by MORE than 1e-6
                    if gap1 > 1e-6 or gap2 > 1e-6:
                        is_separated = True
                        break
                    
                    # Track the minimum gap for penalty
                    min_gap = min(min_gap, max(gap1, gap2))
                
                if not is_separated:
                    # They overlap. Add penalty based on how much they violate
                    # min_gap is the maximum over normals of the minimum gap
                    # If min_gap < 0, they are deeply overlapping
                    # We want to penalize based on how much separation is needed
                    # The separation needed is 1e-6 - min_gap (if min_gap < 1e-6)
                    deficit = 1e-6 - min_gap
                    if deficit > 0:
                        penalty += deficit * deficit * 10000  # Large penalty
        
        return penalty
    
    def compute_enclosing_side(pts):
        """
        Compute the minimum side length of a regular hexagon centered at origin
        that encloses all inner hexagons (all with angle 0).
        """
        # All vertices of all inner hexagons
        all_verts = np.vstack([pts[i] + hex_verts for i in range(len(pts))])
        
        # For each of the 6 edges of the outer hexagon, compute the required side length
        # The outer hexagon has vertices at angles 0, 60, 120, 180, 240, 300 (for angle 0)
        # Edge i goes from vertex i to vertex i+1
        # The constraint is that all inner vertices must be inside the outer hexagon
        
        max_required_s = 0.0
        
        for i in range(6):
            a_i = np.deg2rad(60 * i)
            a_ip1 = np.deg2rad(60 * (i + 1))
            
            # Edge vector
            e = np.array([np.cos(a_ip1) - np.cos(a_i), np.sin(a_ip1) - np.sin(a_i)])
            
            # For a regular hexagon with side s and angle 0, the edge from vertex i to i+1
            # is at a distance of s * sqrt(3) / 2 from the center (apothem)
            # The cross product constraint: for point p, cross(p2 - p1, p - p1) >= -1e-6
            # where p1 and p2 are the outer hexagon vertices
            
            # Actually, let's use the formula from inspiration 2:
            # required_s = -2 * min_cross / SQRT3
            # where min_cross = min over all inner vertices of cross(e, v - p1)
            
            p1 = s_i = np.array([np.cos(a_i), np.sin(a_i)])  # This is wrong, s is unknown
            # Let me re-derive this.
            
            # For a regular hexagon with side length s and angle 0:
            # Vertex i is at s * (cos(60i), sin(60i))
            # Edge i goes from s*(cos(60i), sin(60i)) to s*(cos(60(i+1)), sin(60(i+1)))
            # The inward normal to this edge points towards the center.
            
            # The constraint is: for all inner vertices v, the point v must satisfy
            # cross(p2 - p1, v - p1) >= -1e-6
            # where p1 = s * (cos(60i), sin(60i)) and p2 = s * (cos(60(i+1)), sin(60(i+1)))
            
            # p2 - p1 = s * ((cos(60(i+1)) - cos(60i)), (sin(60(i+1)) - sin(60i)))
            # Let d = (cos(60(i+1)) - cos(60i), sin(60(i+1)) - sin(60i))
            # Then p2 - p1 = s * d
            # v - p1 = v - s * (cos(60i), sin(60i))
            
            # cross(p2 - p1, v - p1) = cross(s*d, v - s*u_i) where u_i = (cos(60i), sin(60i))
            # = s * cross(d, v - s*u_i)
            # = s * (cross(d, v) - s * cross(d, u_i))
            
            # We need this >= -1e-6 for all v
            # So s * (cross(d, v) - s * cross(d, u_i)) >= -1e-6
            # This is a quadratic in s.
            
            # Let's just compute it numerically for a given s and find the minimum s that works.
            # Actually, let's use the projection method.
            
            # The apothem (distance from center to edge) of a regular hexagon with side s is s * sqrt(3) / 2.
            # The edge is perpendicular to the direction from center to the midpoint of the edge.
            # For edge i, the midpoint direction is at angle 60i + 30 degrees.
            # So the inward normal is at angle 60i + 30 + 180 = 60i + 210 degrees.
            # Wait, the inward normal points towards the center, so it's opposite to the outward normal.
            # The outward normal to edge i is at angle 60i + 30 degrees.
            # So the inward normal is at angle 60i + 30 + 180 = 60i + 210 degrees.
            
            # The constraint is that all inner vertices v satisfy:
            # v dot inward_normal <= apothem = s * sqrt(3) / 2
            
            # So s * sqrt(3) / 2 >= max_v (v dot inward_normal)
            # s >= 2 / sqrt(3) * max_v (v dot inward_normal)
            
            inward_normal_angle = np.deg2rad(60 * i + 210)
            inward_normal = np.array([np.cos(inward_normal_angle), np.sin(inward_normal_angle)])
            
            max_proj = np.max(all_verts @ inward_normal)
            required_s = 2.0 / SQRT3 * max_proj
            
            if required_s > max_required_s:
                max_required_s = required_s
        
        return max_required_s
    
    def objective(x):
        pts = x.reshape(11, 2)
        side = compute_enclosing_side(pts)
        penalty = compute_separation_penalty(pts)
        return side + penalty
    
    # Initial guess: flower + 4 spikes
    d = SQRT3  # Distance from center to surrounding hexagons in a flower arrangement
    pts = [[0, 0]]
    for k in range(6):
        angle = np.deg2rad(60 * k)
        pts.append([d * np.cos(angle), d * np.sin(angle)])
    
    spike_angles = [0, 120, 180, 300]
    for angle_deg in spike_angles:
        angle = np.deg2rad(angle_deg)
        pts.append([2 * d * np.cos(angle), 2 * d * np.sin(angle)])
    
    x0 = np.array(pts).flatten()
    
    best_x = x0
    best_val = objective(x0)
    
    # Run optimization
    try:
        res = minimize(objective, x0, method='Powell', options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
        if res.fun < best_val:
            best_x = res.x
            best_val = res.fun
    except Exception:
        pass
    
    # Check time and run more if possible
    if time.time() < deadline - 120:
        try:
            # Try a different initial guess
            x0_pert = x0 + np.random.normal(0, 0.05, x0.shape)
            res2 = minimize(objective, x0_pert, method='Powell', options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res2.fun < best_val:
                best_x = res2.x
                best_val = res2.fun
        except Exception:
            pass
    
    pts_final = best_x.reshape(11, 2)
    
    # Compute final side length
    side = compute_enclosing_side(pts_final)
    
    inner_hex_data = np.zeros((11, 3))
    inner_hex_data[:, 0] = pts_final[:, 0]
    inner_hex_data[:, 1] = pts_final[:, 1]
    inner_hex_data[:, 2] = 0
    
    outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, side

# EVOLVE-BLOCK-END
