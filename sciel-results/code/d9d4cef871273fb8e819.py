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
    
    best_L = float('inf')
    best_inner = None
    best_params = None
    
    # New Radial Slicing: 1 center, 6 inner ring, 5 outer ring
    # Parameters: [r_inner, r_outer, global_rotation, outer_offset]
    # Inner ring: 6 hexagons at angles 0, 60, ..., 300 + global_rot
    # Outer ring: 5 hexagons at angles 0, 72, ..., 288 + global_rot + outer_offset
    
    def build_centers(params):
        r_in, r_out, rot, offset = params
        rot_rad = np.deg2rad(rot)
        offset_rad = np.deg2rad(offset)
        centers = np.zeros((N, 2))
        # Center hexagon
        centers[0] = [0.0, 0.0]
        
        # Inner ring: 6 hexagons
        inner_angles = np.deg2rad([0, 60, 120, 180, 240, 300]) + rot_rad
        for k in range(6):
            centers[1 + k] = [r_in * np.cos(inner_angles[k]), r_in * np.sin(inner_angles[k])]
            
        # Outer ring: 5 hexagons
        outer_angles = np.deg2rad([0, 72, 144, 216, 288]) + rot_rad + offset_rad
        for k in range(5):
            centers[7 + k] = [r_out * np.cos(outer_angles[k]), r_out * np.sin(outer_angles[k])]
            
        return centers
    
    def objective(params):
        centers = build_centers(params)
        
        # Calculate required outer side length L
        # For a regular hexagon with side L centered at origin, inradius is L*sqrt(3)/2.
        # Inner hexagon centered at c, side 1, inradius sqrt(3)/2.
        # Support in direction u: u·c + sqrt(3)/2 <= L*sqrt(3)/2
        # L >= (u·c + sqrt(3)/2) / (sqrt(3)/2) = 2*u·c/sqrt(3) + 1
        # Also for -u: -u·c + sqrt(3)/2 <= L*sqrt(3)/2 => L >= -2*u·c/sqrt(3) + 1
        # So L >= max( 2*max(u·c)/sqrt(3) + 1, 2*max(-u·c)/sqrt(3) + 1 )
        # L >= 2 * max(|u·c|) / sqrt(3) + 1
        
        max_abs_dot = np.max(np.abs(centers @ normals.T))
        L_req = 2.0 * max_abs_dot / sqrt3 + 1.0
        
        # Separation constraint
        viol = _min_separation(centers, normals)
        
        return L_req + 1000.0 * viol * viol
    
    # Try multiple random starts
    # r_in should be around sqrt(3) ~ 1.732 for touching inner ring
    # r_out should be larger
    
    for seed in range(100):
        np.random.seed(seed)
        x0 = np.array([
            np.random.uniform(1.6, 2.5),   # r_inner
            np.random.uniform(3.0, 5.0),   # r_outer
            np.random.uniform(0, 60),      # rotation (symmetry period 60)
            np.random.uniform(0, 72)       # offset (symmetry period 72)
        ])
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                             options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            params = result.x
            
            # Clip parameters to reasonable ranges to avoid weird local minima
            # r_in > 0, r_out > 0
            if params[0] < 1.0 or params[1] < 2.0:
                continue
                
            centers = build_centers(params)
            
            # Verify validity
            viol = _min_separation(centers, normals)
            if viol > 1e-6:
                continue
            
            # Compute final L
            max_abs_dot = np.max(np.abs(centers @ normals.T))
            L_final = 2.0 * max_abs_dot / sqrt3 + 1.0
            
            if L_final < best_L:
                best_L = L_final
                best_params = params.copy()
                best_inner = [[centers[i, 0], centers[i, 1], 0.0] for i in range(N)]
        except Exception:
            continue
    
    if best_inner is None:
        # Fallback: Grid search
        best_L = float('inf')
        for r_in in np.linspace(1.6, 2.2, 10):
            for r_out in np.linspace(3.0, 4.5, 10):
                for rot in np.linspace(0, 60, 5):
                    for offset in np.linspace(0, 72, 5):
                        centers = build_centers(np.array([r_in, r_out, rot, offset]))
                        viol = _min_separation(centers, normals)
                        if viol > 1e-6:
                            continue
                        max_abs_dot = np.max(np.abs(centers @ normals.T))
                        L_final = 2.0 * max_abs_dot / sqrt3 + 1.0
                        if L_final < best_L:
                            best_L = L_final
                            best_inner = [[centers[i, 0], centers[i, 1], 0.0] for i in range(N)]
                            
        if best_inner is None:
            # Last resort fallback
            inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
            return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, (0.0, 0.0), best_L, 0.0
