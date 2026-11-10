import numpy as np
from scipy.optimize import minimize
import time

def _hex_vertices(cx, cy, angle_deg):
    """
    Compute vertices of a unit regular hexagon.
    """
    a = np.deg2rad(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack((cx + np.cos(angles), cy + np.sin(angles)))

def _min_gap(v1, v2):
    """
    Compute the minimum gap between two convex polygons v1 and v2.
    v1, v2 are (6, 2) arrays of vertices.
    Returns a positive value if separated, negative if overlapping.
    We check the 12 edge normals.
    """
    # Get edges and normals for v1
    # Edge k is v[k] to v[(k+1)%6]
    # Normal direction is angle of edge + 90 deg
    # For regular hexagon, we can just use the vertex angles + 30 deg
    
    # However, to be robust for any convex polygon (though here they are regular), 
    # we calculate normals from edges.
    
    def get_normals(v):
        normals = []
        for i in range(6):
            p1 = v[i]
            p2 = v[(i+1)%6]
            edge = p2 - p1
            # Normal is perpendicular to edge
            # For CCW vertices, outward normal is (edge_y, -edge_x)
            # Check orientation: (1,0) -> (0.5, 0.866). Edge (-0.5, 0.866). 
            # Normal (0.866, 0.5). 
            # edge_y = 0.866, -edge_x = 0.5. Correct.
            n = np.array([edge[1], -edge[0]])
            n /= np.linalg.norm(n)
            normals.append(n)
        return np.array(normals)

    n1 = get_normals(v1)
    n2 = get_normals(v2)
    
    min_gap = np.inf
    
    # Check all normals from v1 and v2
    for n in np.vstack((n1, n2)):
        # Project v1
        proj1 = np.dot(v1, n)
        min1 = np.min(proj1)
        max1 = np.max(proj1)
        
        # Project v2
        proj2 = np.dot(v2, n)
        min2 = np.min(proj2)
        max2 = np.max(proj2)
        
        # Gap is min2 - max1 (if v2 is to the right of v1 along n)
        # Or min1 - max2
        # We want the distance between the intervals [min1, max1] and [min2, max2]
        # If they overlap, gap is negative.
        gap1 = min2 - max1
        gap2 = min1 - max2
        
        # The actual separation along this axis is the minimum distance between intervals
        # If gap1 > 0, separation is gap1.
        # If gap2 > 0, separation is gap2.
        # If both < 0, they overlap. The depth of overlap is max(-gap1, -gap2) ? 
        # No, if they overlap, the "gap" is negative.
        # We define gap = min(max1, max2) - max(min1, min2) ? No.
        
        # Let's use:
        # If max1 < min2: gap = min2 - max1
        # If max2 < min1: gap = min1 - max2
        # Else: gap = -min(max1 - min2, max2 - min1)  (depth of penetration)
        
        if max1 < min2:
            gap = min2 - max1
        elif max2 < min1:
            gap = min1 - max2
        else:
            # Overlap
            # Penetration depth
            p1 = max1 - min2
            p2 = max2 - min1
            gap = -max(p1, p2)
            
        if gap < min_gap:
            min_gap = gap
            
    return min_gap

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 500  # 500s limit
    
    # Initial guess: 7 hexagons in a flower + 4 extensions
    # Centers for 7 hexagons (side 1):
    # (0,0)
    # (1.5, 0), (-1.5, 0)
    # (0.75, 1.299), (-0.75, 1.299)
    # (0.75, -1.299), (-0.75, -1.299)
    # Extensions:
    # (3, 0), (-3, 0), (0, 2.598), (0, -2.598) -- these might be too far
    # Let's try a tighter initial guess.
    
    init_coords = np.array([
        [0, 0],
        [1.5, 0],
        [-1.5, 0],
        [0.75, 1.299],
        [-0.75, 1.299],
        [0.75, -1.299],
        [-0.75, -1.299],
        [3.0, 0.8],
        [-3.0, 0.8],
        [0.0, 2.6],
        [0.0, -2.6]
    ])
    
    best_side = np.inf
    best_params = None
    
    n_restarts = 20
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Perturb initial guess
        noise = np.random.normal(0, 0.5, init_coords.shape)
        x0_coords = init_coords + noise
        x0_angle = np.random.uniform(0, 60)
        
        x0 = np.concatenate([x0_coords.flatten(), [x0_angle]])
        
        def cost(params):
            coords = params[:22].reshape(11, 2)
            theta = params[22]
            
            # Compute vertices
            verts = []
            for i in range(11):
                cx, cy = coords[i]
                # Angle 0 for all inner hexagons for now
                v = _hex_vertices(cx, cy, 0)
                verts.append(v)
            
            # Check overlaps
            penalty = 0.0
            for i in range(11):
                for j in range(i+1, 11):
                    gap = _min_gap(verts[i], verts[j])
                    if gap < 0:
                        # Penalty for overlap
                        # Use squared penalty to be smooth
                        penalty += gap * gap
            
            # Compute outer hex side length
            # Support function of the union
            # Max projection of any vertex of any inner hexagon
            # in the direction of the outer hex normals
            
            # Outer hex normals are at theta + k*60
            # We need the max projection of ALL vertices of ALL inner hexagons
            # in each of the 6 directions
            
            all_verts = np.vstack(verts)
            
            max_proj = 0.0
            for k in range(6):
                angle_rad = np.deg2rad(theta + k * 60)
                n = np.array([np.cos(angle_rad), np.sin(angle_rad)])
                proj = np.dot(all_verts, n)
                m = np.max(proj)
                if m > max_proj:
                    max_proj = m
            
            # Apothem A = max_proj
            # Side S = 2 * A / sqrt(3)
            S = 2 * max_proj / np.sqrt(3)
            
            return S + 100 * penalty
        
        try:
            res = minimize(cost, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            
            # Check if the result is valid
            final_cost = res.fun
            # If penalty is significant, final_cost will be large
            # We need to verify validity separately to be sure
            
            # Re-evaluate penalty at res.x
            coords = res.x[:22].reshape(11, 2)
            theta = res.x[22]
            
            verts = []
            for i in range(11):
                cx, cy = coords[i]
                v = _hex_vertices(cx, cy, 0)
                verts.append(v)
            
            is_valid = True
            for i in range(11):
                for j in range(i+1, 11):
                    gap = _min_gap(verts[i], verts[j])
                    if gap < -1e-6: # Allow tiny numerical error
                        is_valid = False
                        break
                if not is_valid:
                    break
            
            if is_valid:
                # Calculate exact S
                all_verts = np.vstack(verts)
                max_proj = 0.0
                for k in range(6):
                    angle_rad = np.deg2rad(theta + k * 60)
                    n = np.array([np.cos(angle_rad), np.sin(angle_rad)])
                    proj = np.dot(all_verts, n)
                    m = np.max(proj)
                    if m > max_proj:
                        max_proj = m
                S = 2 * max_proj / np.sqrt(3)
                
                if S < best_side:
                    best_side = S
                    best_params = res.x.copy()
                    
        except Exception as e:
            pass
            
    if best_params is None:
        # Fallback to initial guess if optimization failed
        best_params = np.concatenate([init_coords.flatten(), [0.0]])
        # Calculate S for fallback
        verts = []
        for i in range(11):
            cx, cy = init_coords[i]
            v = _hex_vertices(cx, cy, 0)
            verts.append(v)
        all_verts = np.vstack(verts)
        max_proj = 0.0
        for k in range(6):
            angle_rad = np.deg2rad(k * 60)
            n = np.array([np.cos(angle_rad), np.sin(angle_rad)])
            proj = np.dot(all_verts, n)
            m = np.max(proj)
            if m > max_proj:
                max_proj = m
        best_side = 2 * max_proj / np.sqrt(3)

    # Construct output
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i, 0] = best_params[2*i]
        inner_hex_data[i, 1] = best_params[2*i+1]
        inner_hex_data[i, 2] = 0.0  # Angle 0
        
    outer_hex_data = np.array([0.0, 0.0, best_params[22]])
    
    return inner_hex_data, outer_hex_data, best_side

