import numpy as np
from scipy.optimize import differential_evolution
import itertools

N = 11

def construct():
    # Vertices of the equilateral triangle
    V1 = np.array([0.0, 0.0])
    V2 = np.array([1.0, 0.0])
    V3 = np.array([0.5, np.sqrt(3) / 2])
    
    # Precompute indices for all triangles
    tri_indices = []
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                tri_indices.append((i, j, k))
    tri_indices = np.array(tri_indices)
    I = tri_indices[:, 0]
    J = tri_indices[:, 1]
    K = tri_indices[:, 2]

    def barycentric_to_cartesian(params):
        # params shape (N, 2) containing (a, b) for each point
        a = params[:, 0]
        b = params[:, 1]
        
        ea = np.exp(a)
        eb = np.exp(b)
        denom = ea + eb + 1.0
        
        u = ea / denom
        v = eb / denom
        w = 1.0 / denom
        
        px = v + 0.5 * w
        py = (np.sqrt(3) / 2) * w
        
        return np.column_stack((px, py))

    def min_area_vectorized(points):
        # points shape (N, 2)
        P_I = points[I]
        P_J = points[J]
        P_K = points[K]
        
        D_IJ = P_J - P_I
        D_IK = P_K - P_I
        
        Cross = D_IJ[:, 0] * D_IK[:, 1] - D_IJ[:, 1] * D_IK[:, 0]
        Areas = 0.5 * np.abs(Cross)
        
        return np.min(Areas)

    def objective(params_flat):
        params = params_flat.reshape(N, 2)
        points = barycentric_to_cartesian(params)
        return -min_area_vectorized(points)

    bounds = [(-10, 10)] * (2 * N)
    
    result = differential_evolution(
        objective, 
        bounds, 
        maxiter=1000, 
        popsize=20, 
        tol=1e-8, 
        seed=42,
        polish=False
    )
    
    best_params = result.x
    best_points = barycentric_to_cartesian(best_params.reshape(N, 2))
    
    return best_points

# EVOLVE-BLOCK-END
