import numpy as np
from scipy.optimize import minimize

N = 12

def _hex_normals():
    """Return the 6 outward normals of a unit regular hexagon (pointy-top)."""
    angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    return np.column_stack((np.cos(angles), np.sin(angles)))

def _min_separation(centers, normals):
    """Return the minimum separation violation across all pairs.
    The Minkowski difference of two unit hexagons is a hexagon of side 2.
    The distance between centers must be outside this hexagon.
    The hexagon of side 2 has inradius sqrt(3). So we need:
    max_u |u · (c_i - c_j)| >= sqrt(3)
    """
    min_viol = 0.0
    n = len(centers)
    for i in range(n):
        for j in range(i+1, n):
            diff = centers[i] - centers[j]
            d_dots = np.abs(diff @ normals.T)
            max_d_dot = np.max(d_dots)
            if max_d_dot < np.sqrt(3.0):
                viol = np.sqrt(3.0) - max_d_dot
                min_viol = max(min_viol, viol)
    return min_viol

def construct():
    normals = _hex_normals()
    sqrt3 = np.sqrt(3.0)
    
    # New Radial Slicing: 1 center, 5 inner, 6 outer
    # Parameters: [R1, R2, theta]
    # R1: radius of 5-inner ring
    # R2: radius of 6-outer ring
    # theta: global rotation
    
    def build_centers(params):
        R1, R2, theta = params
        theta_rad = np.deg2rad(theta)
        centers = np.zeros((N, 2))
        
        # Center hexagon
        centers[0] = [0.0, 0.0]
        
        # Inner ring: 5 hexagons
        # Angles: 0, 72, 144, 216, 288 + theta
        inner_angles = np.deg2rad([0, 72, 144, 216, 288]) + theta_rad
        for k in range(5):
            centers[1 + k] = [R1 * np.cos(inner_angles[k]), R1 * np.sin(inner_angles[k])]
            
        # Outer ring: 6 hexagons
        # Angles: 0, 60, 120, 180, 240, 300 + theta
        outer_angles = np.deg2rad([0, 60, 120, 180, 240, 300]) + theta_rad
        for k in range(6):
            centers[6 + k] = [R2 * np.cos(outer_angles[k]), R2 * np.sin(outer_angles[k])]
            
        return centers

    def objective(params):
        R1, R2, theta = params
        if R1 < 1.0 or R2 < 2.0:
            return 1e10
            
        centers = build_centers(params)
        
        # Check separation
        viol = _min_separation(centers, normals)
        if viol > 0:
            return 1000.0 * viol * viol + 100.0 * viol
            
        # Calculate required outer side length L
        # Assume outer hexagon is fixed orientation (normals at 0, 60, ...)
        # Wait, _hex_normals returns pointy-top normals (30, 90...).
        # Let's use the same normals for the outer hexagon to be consistent.
        # The outer hexagon must contain all inner hexagons.
        # Condition: For each normal u of the outer hexagon, 
        # max_i (u . c_i + h_inner(u)) <= R_outer
        # h_inner(u) for a unit hexagon (side 1) is the support function.
        # If the inner hexagon is aligned with the outer hexagon (angle 0),
        # its support in direction u is:
        # If u is a normal of the hexagon, h = sqrt(3)/2.
        # If u is between normals, it's larger? No, support function is max dot product.
        # For a regular hexagon with inradius r, support in direction u is r / cos(delta) where delta is angle to nearest normal?
        # Actually, if we fix the outer hexagon orientation to match the normals we use for checking,
        # and the inner hexagons are also aligned with that orientation,
        # then h_inner(u_k) = sqrt(3)/2 for all k if u_k are the normals of the inner hexagons.
        # But the inner hexagons are at different positions, not rotated relative to each other?
        # The directive implies fixed relative offsets. If we fix inner hexagon rotation to 0,
        # their normals are the same as the outer hexagon's normals (if outer is also 0).
        # So h_inner(u_k) = sqrt(3)/2 for all k.
        
        # So R_outer >= max_i (u_k . c_i + sqrt(3)/2) for all k.
        # R_outer = max_k ( max_i (u_k . c_i) + sqrt(3)/2 )
        
        dots = centers @ normals.T # (N, 6)
        max_dots = np.max(dots, axis=0) # (6,)
        R_outer = np.max(max_dots) + sqrt3 / 2.0
        L = 2.0 * R_outer / sqrt3
        
        return L

    best_L = float('inf')
    best_params = None
    
    # Optimization
    # Initial guesses
    # R1 should be ~ sqrt(3) ~ 1.732 to touch center
    # R2 should be ~ 2*sqrt(3) ~ 3.46 or more
    
    for seed in range(50):
        np.random.seed(seed)
        x0 = np.array([
            np.random.uniform(1.6, 2.2),   # R1
            np.random.uniform(3.0, 4.5),   # R2
            np.random.uniform(0, 360)      # theta
        ])
        
        try:
            # Use L-BFGS-B or Nelder-Mead?
            # Objective is non-smooth (max of linear), so Nelder-Mead is safer.
            result = minimize(objective, x0, method='Nelder-Mead', 
                             options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            params = result.x
            
            # Verify
            if params[0] < 1.0 or params[1] < 2.0:
                continue
                
            centers = build_centers(params)
            viol = _min_separation(centers, normals)
            if viol > 1e-6:
                continue
                
            dots = centers @ normals.T
            max_dots = np.max(dots, axis=0)
            R_outer = np.max(max_dots) + sqrt3 / 2.0
            L_final = 2.0 * R_outer / sqrt3
            
            if L_final < best_L:
                best_L = L_final
                best_params = params.copy()
                best_inner = [[centers[i, 0], centers[i, 1], 0.0] for i in range(N)]
        except Exception:
            continue

    if best_inner is None:
        # Fallback
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

    return best_inner, (0.0, 0.0), best_L, 0.0
