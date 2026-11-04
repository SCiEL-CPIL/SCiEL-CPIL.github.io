import numpy as np
from scipy.optimize import minimize
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

N = 13

def construct():
    # Parameters: 13 edge weights for the cycle graph
    # We will optimize these weights.
    # Initial weights: 1.0
    
    def get_points(weights):
        # Construct weighted adjacency matrix for cycle graph
        # L = D - A
        # A has weights w_i on (i, i+1) and (i+1, i)
        # D has sum of weights on diagonal
        
        # Use dense matrix for small N for simplicity in eigendecomposition
        # or use scipy.sparse.linalg.eigsh
        
        A = np.zeros((N, N))
        for i in range(N):
            j = (i + 1) % N
            A[i, j] = weights[i]
            A[j, i] = weights[i]
            
        D = np.diag(np.sum(A, axis=1))
        L = D - A
        
        # Compute eigenvectors
        # We want the smallest non-zero eigenvalues
        # eigsh is for sparse, but N=13 is small, use np.linalg.eigh
        evals, evecs = np.linalg.eigh(L)
        
        # evecs are columns. evecs[:, 0] is the constant mode (eigenvalue ~0)
        # We take the next two: evecs[:, 1] and evecs[:, 2]
        # To get more flexibility, we might mix more modes, but let's start with 2.
        
        v1 = evecs[:, 1]
        v2 = evecs[:, 2]
        
        # Normalize the vectors to have unit norm to keep points bounded
        # Actually, the scale of the points doesn't matter for the ratio,
        # but it helps with numerical stability.
        
        x = v1
        y = v2
        
        return np.column_stack((x, y))

    def score(points):
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(points)
            hull_area = hull.volume
            if hull_area < 1e-10:
                return 0.0
        except Exception:
            return 0.0
            
        min_area = float('inf')
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    p1 = points[i]
                    p2 = points[j]
                    p3 = points[k]
                    area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
                    if area < min_area:
                        min_area = area
        
        if min_area == float('inf'):
            return 0.0
            
        return min_area / hull_area

    def objective(weights):
        pts = get_points(weights)
        # Check convexity: all points must be on the hull
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(pts)
            if len(hull.vertices) < N:
                return 1.0 # Penalty
        except Exception:
            return 1.0
            
        s = score(pts)
        return -s

    x0 = np.ones(N)
    
    # Weights must be positive
    # We can optimize log-weights or use bounds
    # Let's use bounds (0.1, 5.0)
    bounds = [(0.1, 5.0)] * N
    
    res = minimize(objective, x0, method='Nelder-Mead', 
                   options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    best_weights = res.x
    best_points = get_points(best_weights)
    
    # Validate
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(best_points)
        if len(hull.vertices) < N:
             # Fallback to regular polygon
             angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
             return np.column_stack((np.cos(angles), np.sin(angles)))
    except Exception:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_points
