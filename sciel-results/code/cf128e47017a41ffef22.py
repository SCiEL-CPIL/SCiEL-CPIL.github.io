import numpy as np
import time
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Leave 10s margin
    
    # Helper: Get vertices of a hexagon
    def get_hex_vertices(cx, cy, side, angle_deg):
        angle_rad = np.deg2rad(angle_deg)
        # Vertices are at angles angle_rad + k * 60 deg
        # Distance from center to vertex is 'side'
        # Wait, for a regular hexagon of side s, circumradius is s.
        # Inradius is s * sqrt(3) / 2.
        # The distance from center to a vertex is s.
        # The distance from center to a side midpoint is s * sqrt(3) / 2.
        # The half-plane check uses the side midpoints/normal distance.
        # Normal to side k is at angle angle_rad + k * 60 deg?
        # Let's check orientation.
        # If angle is 0, vertex 0 is at (s, 0).
        # Side 0 connects vertex 5 and vertex 0? No.
        # Standard orientation: Pointy top or flat top?
        # Let's assume vertices at angles 0, 60, 120, 180, 240, 300.
        # Then side 0 (between 300 and 0) has normal at -30 (or 330).
        # Side 1 (between 0 and 60) has normal at 30.
        # Generally, normal to side k is at angle (k * 60 - 30) + angle_rad.
        # Distance from center to side is s * sqrt(3) / 2.
        
        verts = []
        for k in range(6):
            theta = angle_rad + k * np.pi / 3
            x = cx + side * np.cos(theta)
            y = cy + side * np.sin(theta)
            verts.append([x, y])
        return np.array(verts)

    def max_violation(verts, L, phi_out_rad):
        # Calculate max violation for all vertices of this inner hexagon
        # against the outer hexagon of side L, rotation phi_out_rad
        # Outer hexagon half-planes:
        # Normal k: angle = phi_out_rad + (k * 60 - 30) degrees?
        # Let's align the outer hexagon definition with the inner one.
        # If inner hex angle is 0, vertices at 0, 60...
        # Outer hex angle is phi_out.
        # The "width" of the outer hexagon in direction theta is 2 * L * cos(delta) where delta is angle diff to normal?
        # No, the distance from center to the side face is R_in = L * sqrt(3) / 2.
        # The side faces are perpendicular to normals at angles:
        # phi_out - 30, phi_out + 30, phi_out + 90, etc.
        # Basically, the normals are at phi_out + (k * 60 - 30) for k=0..5?
        # Let's verify.
        # If phi_out = 0, normals are at -30, 30, 90, 150, 210, 270.
        # Distance to side is L * sqrt(3) / 2.
        # Condition: v . n <= L * sqrt(3) / 2.
        
        normals = []
        for k in range(6):
            ang = phi_out_rad + (k * np.pi / 3 - np.pi / 6)
            normals.append([np.cos(ang), np.sin(ang)])
        normals = np.array(normals)
        
        # Project all vertices onto all normals
        # verts shape (6, 2), normals shape (6, 2)
        # projections shape (6, 6)
        projections = verts @ normals.T
        # Max projection for each vertex
        max_proj_per_vertex = np.max(projections, axis=1)
        # Max projection overall
        max_proj = np.max(max_proj_per_vertex)
        
        limit = L * np.sqrt(3) / 2
        violation = max_proj - limit
        return max(0, violation)

    def objective(params):
        # params: [x1, y1, a1, x2, y2, a2, ..., x12, y12, a12, L, phi_out]
        # 12 * 3 + 2 = 38 params
        
        inner_params = params[:36].reshape(12, 3)
        L = params[36]
        phi_out = params[37]
        
        if L < 0:
            return 1e6
        
        total_violation = 0.0
        for i in range(12):
            x, y, a_deg = inner_params[i]
            verts = get_hex_vertices(x, y, 1.0, a_deg)
            v = max_violation(verts, L, np.deg2rad(phi_out))
            total_violation += v * v
            
        # Also add a small penalty if L is very large to encourage shrinking
        # But the main driver is L + penalty.
        return L + 1000.0 * total_violation

    # Initial Guess
    # Create a compact cluster
    # Hex grid positions
    # dx = sqrt(3), dy = 1.5
    # Let's place 12 hexagons in a 3x4 grid-like structure but adjusted for hex packing
    # Or just use a random cluster within radius 4
    
    def random_initial():
        # Center at 0,0
        # 1 in center
        # 6 around
        # 5 more
        pts = []
        pts.append([0.0, 0.0, 0.0])
        for k in range(6):
            ang = k * 60
            r = np.sqrt(3) # distance between centers of adjacent hexes
            x = r * np.cos(np.deg2rad(ang))
            y = r * np.sin(np.deg2rad(ang))
            pts.append([x, y, 0.0])
        # Add 5 more on the next ring
        # Next ring radius is 2*sqrt(3)
        # We need to pick 5 positions from the 12 available on the second ring
        # Or just scatter them randomly nearby
        for k in range(5):
            ang = (k * 72 + 30) # spread them out
            r = 2 * np.sqrt(3)
            x = r * np.cos(np.deg2rad(ang))
            y = r * np.sin(np.deg2rad(ang))
            pts.append([x, y, 0.0])
            
        # Flatten
        flat = []
        for p in pts:
            flat.extend(p)
        # Add L and phi_out
        # Estimate L. Max dist from origin is ~ 2*sqrt(3) + 1 (radius of hex)
        # 2*1.732 + 1 = 4.46.
        # Let's start with L = 4.0
        flat.append(4.0)
        flat.append(0.0)
        return np.array(flat)

    best_result = None
    best_score = float('inf')
    
    # Try a few restarts
    num_restarts = 5
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        x0 = random_initial()
        # Add some noise to avoid symmetry
        if i > 0:
            noise = np.random.normal(0, 0.5, 38)
            # Don't noise L and phi too much
            noise[36] *= 0.1
            noise[37] *= 10
            x0 = x0 + noise
            
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if time.time() > deadline:
                break
                
            if res.success or res.fun < best_score:
                # Check if valid (low violation)
                # Re-evaluate violation to be sure
                inner_params = res.x[:36].reshape(12, 3)
                L = res.x[36]
                phi_out = res.x[37]
                
                total_violation = 0.0
                for k in range(12):
                    x, y, a_deg = inner_params[k]
                    verts = get_hex_vertices(x, y, 1.0, a_deg)
                    v = max_violation(verts, L, np.deg2rad(phi_out))
                    total_violation += v * v
                    
                if total_violation < 1e-6:
                    best_score = L
                    best_result = res.x
        except Exception as e:
            pass

    if best_result is None:
        # Fallback: Return the initial guess or a known safe value
        # Use the last x0 or a default
        x0 = random_initial()
        best_result = x0
        best_score = x0[36]

    # Extract final parameters
    inner_params = best_result[:36].reshape(12, 3)
    L = best_result[36]
    phi_out = best_result[37]
    
    # Construct return values
    inner = []
    for i in range(12):
        x, y, a_deg = inner_params[i]
        inner.append([x, y, a_deg])
        
    center = [0.0, 0.0]
    side = L
    angle = phi_out
    
    return inner, center, side, angle

