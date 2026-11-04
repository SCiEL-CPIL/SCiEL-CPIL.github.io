import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = 135.0  # Safety margin before 150s limit
    
    # Scale factor: hexagons on triangular lattice touch at edges, so small scale gives gap
    scale = 1.002
    
    # Centers of the 11 hexagons
    # 1 center
    centers = np.array([
        [0.0, 0.0]
    ])
    
    # 6 neighbors at dist sqrt(3) * scale
    r1 = np.sqrt(3) * scale
    for k in range(6):
        angle = np.deg2rad(k * 60)
        centers = np.vstack([centers, [r1 * np.cos(angle), r1 * np.sin(angle)]])
        
    # 4 more at dist 3 * scale
    # Angles 30, 90, 150, 210
    r2 = 3.0 * scale
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        centers = np.vstack([centers, [r2 * np.cos(angle), r2 * np.sin(angle)]])
        
    n_hex = len(centers)
    
    # Vertices of a unit hexagon (side 1)
    # Base vertices at angle 0 (pointy right)
    base_vertices = np.array([
        [np.cos(np.deg2rad(60*k)), np.sin(np.deg2rad(60*k))] for k in range(6)
    ])
    
    def get_all_vertices(params):
        """
        params: [theta_out, phi_0, ..., phi_10]
        Returns: Array of shape (n_hex * 6, 2) containing all vertices of all inner hexagons
        """
        theta_out = params[0]
        phis = params[1:]
        
        all_vertices = []
        for i in range(n_hex):
            phi = phis[i]
            c = np.cos(phi)
            s = np.sin(phi)
            rot = np.array([[c, -s], [s, c]])
            verts = base_vertices @ rot.T + centers[i]
            all_vertices.append(verts)
            
        return np.vstack(all_vertices)
    
    def objective(params):
        """
        Minimizes the side length S of the outer hexagon.
        Outer hexagon is centered at (0,0), rotated by theta_out.
        """
        theta_out = params[0]
        
        verts = get_all_vertices(params)
        
        normals = []
        for k in range(6):
            angle = np.deg2rad(30 + k * 60) + theta_out
            normals.append([np.cos(angle), np.sin(angle)])
        normals = np.array(normals)
        
        dots = verts @ normals.T
        max_proj = np.max(dots)
        S = max_proj / (np.sqrt(3) / 2)
        
        return S

    def smooth_objective(params, t=50.0):
        """
        Smooth epigraph formulation using log-sum-exp for the max over projections.
        This provides gradients that help the optimizer converge more tightly.
        """
        theta_out = params[0]
        
        verts = get_all_vertices(params)
        
        normals = []
        for k in range(6):
            angle = np.deg2rad(30 + k * 60) + theta_out
            normals.append([np.cos(angle), np.sin(angle)])
        normals = np.array(normals)
        
        dots = verts @ normals.T
        
        # Log-sum-exp approximation of max
        max_dots = np.max(dots)
        log_sum_exp = max_dots + np.log(np.sum(np.exp(dots - max_dots)))
        smooth_max = log_sum_exp / t
        
        S = smooth_max / (np.sqrt(3) / 2)
        
        return S

    # Initial guess
    x0 = np.zeros(n_hex + 1)
    
    bounds = [(0, np.pi/3)] * (n_hex + 1)
    
    best_score = None
    best_params = None
    
    # Phase 1: Multi-start with Powell for coarse optimization
    n_starts = 5
    for i in range(n_starts):
        if time.time() > deadline - 30:
            break
            
        if i == 0:
            x_start = x0
        else:
            x_start = np.random.uniform(0, np.pi/3, n_hex + 1)
            
        try:
            res = minimize(objective, x_start, method='Powell', bounds=bounds, 
                          options={'maxiter': 300, 'disp': False, 'ftol': 1e-12})
            
            if res.fun < (best_score if best_score is not None else float('inf')):
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
            
    if best_params is None:
        return _fallback_solution()
    
    # Phase 2: High-precision polish with Nelder-Mead on the smooth objective
    if time.time() < deadline - 10:
        try:
            # Use smooth objective with large t for near-exact behavior
            res2 = minimize(smooth_objective, best_params, method='Nelder-Mead',
                           options={'xatol': 1e-14, 'fatol': 1e-14, 'maxiter': 5000, 'disp': False})
            
            # Evaluate the true (nonsmooth) objective at the polished point
            true_score = objective(res2.x)
            if true_score < best_score:
                best_score = true_score
                best_params = res2.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final scalar polish on each parameter individually
    if time.time() < deadline - 5:
        for j in range(n_hex + 1):
            if time.time() > deadline - 5:
                break
            
            # Try small perturbations around the current best
            current_val = best_params[j]
            best_local = best_score
            best_local_param = best_params[j].copy()
            
            for delta in [1e-6, -1e-6, 1e-5, -1e-5, 1e-4, -1e-4]:
                test_param = best_params.copy()
                test_param[j] = current_val + delta
                if test_param[j] < 0 or test_param[j] > np.pi/3:
                    continue
                try:
                    test_score = objective(test_param)
                    if test_score < best_local:
                        best_local = test_score
                        best_local_param = test_param.copy()
                except Exception:
                    continue
            
            if best_local < best_score:
                best_score = best_local
                best_params = best_local_param
    
    # Extract best solution
    theta_out = best_params[0]
    phis = best_params[1:]
    
    inner = []
    for i in range(n_hex):
        inner.append([centers[i][0], centers[i][1], np.rad2deg(phis[i])])
    
    # The side length is the objective value
    side = best_score
    
    # Add tiny margin for safety
    side += 1e-9
    
    return inner, (0.0, 0.0), side, np.rad2deg(theta_out)

def _fallback_solution():
    eps = 1e-9
    scale = 1.002
    centers = []
    centers.append((0.0, 0.0))
    r1 = np.sqrt(3) * scale
    for k in range(6):
        angle = np.deg2rad(k * 60)
        x = (r1 + eps) * np.cos(angle)
        y = (r1 + eps) * np.sin(angle)
        centers.append((x, y))
    r2 = 3.0 * scale
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        x = (r2 + 2*eps) * np.cos(angle)
        y = (r2 + 2*eps) * np.sin(angle)
        centers.append((x, y))
        
    inner = []
    for x, y in centers:
        inner.append([x, y, 30.0])
        
    # Compute the actual required side length for this configuration
    # With all hexagons rotated 30 degrees and outer hexagon rotated 30 degrees
    # The critical constraint is the outermost vertices
    max_x = (r2 + eps) * np.cos(np.deg2rad(30)) + 1  # vertex extends 1 unit from center
    max_y = (r2 + eps) * np.sin(np.deg2rad(90)) + 1
    
    # For a hexagon with rotation 30 deg, the side length S satisfies:
    # The apothem in each direction must be >= max projection
    # h = S * sqrt(3)/2
    # For direction 30+30=60 deg: projection = max_x*cos(60) + max_y*sin(60)
    # Actually let's just compute it properly
    theta_out = np.deg2rad(30)
    normals = []
    for k in range(6):
        angle = np.deg2rad(30 + k * 60) + theta_out
        normals.append([np.cos(angle), np.sin(angle)])
    normals = np.array(normals)
    
    all_verts = []
    for x, y in centers:
        phi = np.deg2rad(30)
        c = np.cos(phi)
        s = np.sin(phi)
        rot = np.array([[c, -s], [s, c]])
        base_v = np.array([[np.cos(np.deg2rad(60*k)), np.sin(np.deg2rad(60*k))] for k in range(6)])
        verts = base_v @ rot.T + np.array([x, y])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    dots = all_verts @ normals.T
    max_proj = np.max(dots)
    side = max_proj / (np.sqrt(3) / 2) + 1e-9
    
    return inner, (0.0, 0.0), side, 30.0

if __name__ == "__main__":
    construct()
