import numpy as np
import time
from scipy.optimize import minimize, basinhopping

N = 11
MIN_DIST = np.sqrt(3)
NORMALS = np.array([[np.cos(a), np.sin(a)] for a in np.linspace(0, 2*np.pi, 7)[:-1]])

def construct():
    def cost_func(params):
        # params: [x1, y1, ..., xN, yN, cx, cy]
        pts = params[:2*N].reshape(N, 2)
        center = params[2*N:]
        
        # Overlap penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < MIN_DIST:
                    penalty += (MIN_DIST - d) ** 2
        
        # Enclosing hexagon size
        # Distance from center to side is L * sqrt(3) / 2
        # Constraint: |(P_i - C) . n_k| <= L * sqrt(3) / 2
        # L = max( max_i |(P_i - C) . n_k| ) / (sqrt(3) / 2)
        
        rel_pts = pts - center
        projections = np.abs(rel_pts @ NORMALS.T)
        max_proj = np.max(projections)
        L_enc = max_proj / (np.sqrt(3) / 2)
        
        return L_enc + 100.0 * penalty

    # Initialization
    def get_initial_points():
        # Hexagonal lattice for 11 points
        lattice_pts = []
        for q in range(-2, 3):
            for r in range(-2, 3):
                x = 1.5 * q
                y = np.sqrt(3)/2 * q + np.sqrt(3) * r
                lattice_pts.append((x, y))
        
        lattice_pts = np.array(lattice_pts)
        dists = np.linalg.norm(lattice_pts, axis=1)
        indices = np.argsort(dists)
        init_pts = lattice_pts[indices[:N]]
        
        # Add noise
        init_pts += np.random.normal(0, 0.1, init_pts.shape)
        return init_pts

    best_L = float('inf')
    best_params = None
    
    # Use basinhopping for global search
    x0 = np.concatenate([get_initial_points().flatten(), [0.0, 0.0]])
    
    # Limit iterations to ensure speed
    for i in range(5):
        res = basinhopping(cost_func, x0, niter=20, minimizer_kwargs={'method': 'Nelder-Mead', 'options': {'maxiter': 200}})
        
        final_params = res.x
        final_cost = res.fun
        
        # Extract L
        pts = final_params[:2*N].reshape(N, 2)
        center = final_params[2*N:]
        rel_pts = pts - center
        projections = np.abs(rel_pts @ NORMALS.T)
        max_proj = np.max(projections)
        L_enc = max_proj / (np.sqrt(3) / 2)
        
        # Check validity
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                if np.linalg.norm(pts[i] - pts[j]) < MIN_DIST - 1e-5:
                    valid = False
                    break
            if not valid: break
            
        if valid and L_enc < best_L:
            best_L = L_enc
            best_params = final_params
            x0 = final_params # Restart from best

    if best_params is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    pts = best_params[:2*N].reshape(N, 2)
    center = best_params[2*N:]
    
    inner = [[float(x), float(y), 0.0] for x, y in pts]
    outer_side = best_L + 1.0
    
    return inner, (float(center[0]), float(center[1])), outer_side, 0.0
