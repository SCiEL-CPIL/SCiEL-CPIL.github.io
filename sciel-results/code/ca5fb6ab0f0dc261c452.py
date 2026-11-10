import numpy as np
from scipy.optimize import minimize
import time

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # safety margin before 630s kill
    
    # Sofa grid points
    sofa_x = np.arange(-3, 1.5, 0.01)
    sofa_y = np.arange(-1.5, 1.5, 0.01)
    sofa_X, sofa_Y = np.meshgrid(sofa_x, sofa_y, indexing='ij')
    sofa_x_flat = sofa_X.ravel()
    sofa_y_flat = sofa_Y.ravel()
    n_sofa = len(sofa_x_flat)
    
    def _densify(path):
        """Add the evaluator's start/end and interpolate between consecutive rows."""
        full = np.vstack([[[0.0, 0.0, 0.0]], path, [[-2.0, 0.0, 90.0]]])
        result = [full[0].copy()]
        for i in range(len(full) - 1):
            p1, p2 = full[i], full[i + 1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(da)))
            for k in range(1, m + 1):
                t = k / m
                result.append([p1[0] + t * (p2[0] - p1[0]),
                               p1[1] + t * (p2[1] - p1[1]),
                               p1[2] + t * (p2[2] - p1[2])])
        return np.array(result)
    
    def _count_inside(path):
        """Count sofa points inside all placements in the densified path."""
        densified = _densify(path)
        mask = np.ones(n_sofa, dtype=bool)
        
        for x, y, a_deg in densified:
            a_rad = np.deg2rad(a_deg)
            cos_a = np.cos(a_rad)
            sin_a = np.sin(a_rad)
            
            dx = sofa_x_flat - x
            dy = sofa_y_flat - y
            
            u = cos_a * dx + sin_a * dy
            v = -sin_a * dx + cos_a * dy
            
            # Base corridor: union of two rectangles
            p1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            p2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            inside = p1 | p2
            
            mask &= inside
            # Early termination if no points remain
            if not np.any(mask):
                return 0
        
        return np.sum(mask)
    
    def _score(path):
        """Compute the raw value (area) for a path."""
        count = _count_inside(path)
        return count * 0.0001
    
    def _arc_path(cx, cy, r, n_points=8):
        """
        Generate a path that follows an arc from angle 0 to 90 degrees.
        The corridor's inner corner traces an arc of radius r centered at (cx, cy).
        """
        angles = np.linspace(0, 90, n_points + 1)[1:]  # exclude start (0,0,0)
        # Convert to radians for computation
        angles_rad = np.deg2rad(angles)
        
        # Position of inner corner on arc
        # At angle 0: corner at (0,0), so center is at (cx, cy)
        # The corner position relative to center: (r*cos(theta), r*sin(theta))
        # But we need to figure out the correct offset
        # At a=0, corner is at (0,0). At a=90, corner is at (-2,0).
        # Let's parameterize: corner = center + r * (cos(alpha), sin(alpha))
        # At a=0: (0,0) = (cx, cy) + r*(cos(a0), sin(a0))
        # At a=90: (-2,0) = (cx, cy) + r*(cos(a0+pi/2), sin(a0+pi/2))
        
        # Simpler approach: use a bump-like path
        # The path goes from (0,0) to (-2,0) with a bump upward
        # Parameterize by a bump height and horizontal position of the bump
        
        # Let's use a different parameterization:
        # The path is defined by control points that form a smooth curve
        # from (0,0,0) to (-2,0,90)
        
        # Actually, let's use a simpler approach:
        # Define the path as a sequence of points where:
        # - x goes from 0 to -2
        # - y has a bump (goes up then comes back)
        # - a goes from 0 to 90
        
        # Use a sine bump for y
        t = np.linspace(0, 1, n_points + 1)[1:]  # exclude t=0 (start)
        x_vals = -2.0 * t
        y_vals = r * np.sin(np.pi * t)  # bump height r
        a_vals = 90.0 * t
        
        path = np.column_stack([x_vals, y_vals, a_vals])
        return path
    
    def _bump_path(h, x_peak, n_points=8):
        """
        Generate a path with a parabolic bump.
        h: bump height
        x_peak: x-coordinate of the bump peak (between -2 and 0)
        """
        t = np.linspace(0, 1, n_points + 1)[1:]  # exclude start
        x_vals = -2.0 * t
        
        # Parabolic bump centered at x_peak
        # Normalize x_peak to [0,1] range
        x_peak_t = (-x_peak) / 2.0  # convert to t coordinate
        y_vals = h * (1 - ((t - x_peak_t) / 0.5)**2)
        y_vals = np.maximum(y_vals, 0)  # ensure non-negative
        
        a_vals = 90.0 * t
        
        path = np.column_stack([x_vals, y_vals, a_vals])
        return path
    
    # Initial guess: use a bump path
    best_path = _bump_path(0.5, -1.0, n_points=8)
    best_area = _score(best_path)
    
    # Local optimization using Nelder-Mead
    def _objective_bump(params):
        if time.time() > deadline - 10:
            return -best_area
        h, x_peak = params
        h = np.clip(h, 0.0, 2.0)
        x_peak = np.clip(x_peak, -1.8, -0.2)
        path = _bump_path(h, x_peak, n_points=8)
        try:
            area = _score(path)
            return -area
        except:
            return -best_area
    
    # Optimize bump parameters
    x0 = np.array([0.5, -1.0])
    result = minimize(_objective_bump, x0, method='Nelder-Mead',
                      options={'maxiter': 200, 'xatol': 0.01, 'fatol': 0.0001})
    
    if result.fun < -best_area:
        h, x_peak = result.x
        best_path = _bump_path(h, x_peak, n_points=8)
        best_area = -result.fun
    
    # Now try to optimize individual points for finer control
    def _objective_path(params):
        if time.time() > deadline - 10:
            return -best_area
        path = params.reshape(-1, 3)
        # Ensure angles are monotonically increasing from 0 to 90
        path[:, 2] = np.clip(path[:, 2], 0, 90)
        # Ensure x is monotonically decreasing from 0 to -2
        path[:, 0] = np.clip(path[:, 0], -2.1, 0.1)
        # Ensure y is non-negative (bump goes up)
        path[:, 1] = np.clip(path[:, 1], -0.1, 2.0)
        
        try:
            area = _score(path)
            return -area
        except:
            return -best_area
    
    # Optimize the full path
    x0 = best_path.ravel()
    result = minimize(_objective_path, x0, method='Nelder-Mead',
                      options={'maxiter': 300, 'xatol': 0.005, 'fatol': 0.0001})
    
    if result.fun < -best_area:
        best_path = result.x.reshape(-1, 3)
        best_area = -result.fun
    
    # Try different numbers of points
    for n_pts in [6, 10, 12]:
        if time.time() > deadline - 60:
            break
        
        # Create initial path with more/fewer points
        t = np.linspace(0, 1, n_pts + 1)[1:]
        x_vals = -2.0 * t
        # Use the best bump shape as guide
        y_vals = 0.5 * np.sin(np.pi * t)
        a_vals = 90.0 * t
        init_path = np.column_stack([x_vals, y_vals, a_vals])
        
        def _objective_npts(params):
            if time.time() > deadline - 10:
                return -best_area
            path = params.reshape(-1, 3)
            path[:, 2] = np.clip(path[:, 2], 0, 90)
            path[:, 0] = np.clip(path[:, 0], -2.1, 0.1)
            path[:, 1] = np.clip(path[:, 1], -0.1, 2.0)
            try:
                area = _score(path)
                return -area
            except:
                return -best_area
        
        result = minimize(_objective_npts, init_path.ravel(), method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 0.005, 'fatol': 0.0001})
        
        if result.fun < -best_area:
            best_path = result.x.reshape(-1, 3)
            best_area = -result.fun
    
    # Final verification
    final_area = _score(best_path)
    if final_area > best_area:
        best_area = final_area
    
    return best_path
