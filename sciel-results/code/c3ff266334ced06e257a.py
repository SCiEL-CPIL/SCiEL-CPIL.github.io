import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    # Normals for unit hexagon (side 1)
    angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    
    best_L = float('inf')
    best_inner = None
    
    # Try a few random starts
    for seed in range(10):
        np.random.seed(seed)
        # Initial guess: random points in a circle of radius 2
        x0 = np.random.uniform(-2, 2, (2 * N,))
        
        def objective(params):
            centers = params.reshape(N, 2)
            dots = centers @ normals.T
            max_dot = np.max(dots)
            L_req = (2.0 / np.sqrt(3.0)) * max_dot
            
            viol_sum = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    diff = centers[i] - centers[j]
                    d_dots = diff @ normals.T
                    max_d_dot = np.max(d_dots)
                    if max_d_dot < np.sqrt(3.0):
                        viol = np.sqrt(3.0) - max_d_dot
                        viol_sum += viol * viol
            return L_req + 100.0 * viol_sum

        try:
            result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000})
            best_centers = result.x.reshape(N, 2)
            
            # Verify validity
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    diff = best_centers[i] - best_centers[j]
                    d_dots = diff @ normals.T
                    if np.max(d_dots) < np.sqrt(3.0) - 1e-5:
                        valid = False
                        break
                if not valid: break
            
            if valid:
                dots = best_centers @ normals.T
                max_dot = np.max(dots)
                L_final = (2.0 / np.sqrt(3.0)) * max_dot
                
                if L_final < best_L:
                    best_L = L_final
                    best_inner = [[best_centers[i, 0], best_centers[i, 1], 0.0] for i in range(N)]
        except Exception:
            continue
            
    if best_inner is None:
        # Fallback to a known valid configuration if optimization fails
        # e.g. simple grid
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return best_inner, (0.0, 0.0), best_L, 0.0
