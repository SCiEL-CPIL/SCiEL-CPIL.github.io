import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

def construct():
    # Spectral Graph Theory Approach:
    # Points are defined by the first two non-trivial eigenvectors
    # of a learnable banded symmetric matrix.
    
    def get_points(params):
        d, o1, o2 = params
        
        # Construct symmetric banded matrix M
        M = np.zeros((N, N))
        for i in range(N):
            M[i, i] = d
            if i + 1 < N:
                M[i, i+1] = o1
                M[i+1, i] = o1
            if i + 2 < N:
                M[i, i+2] = o2
                M[i+2, i] = o2
        
        # Compute eigenvectors
        # eigh returns eigenvalues in ascending order
        eigenvalues, eigenvectors = np.linalg.eigh(M)
        
        # Use eigenvectors at indices 1 and 2 (skipping the trivial/constant mode at 0)
        x = eigenvectors[:, 1].copy()
        y = eigenvectors[:, 2].copy()
        
        # Fix sign ambiguity to ensure smooth optimization landscape
        # Align the first significant element to be positive
        for vec in [x, y]:
            idx = np.argmax(np.abs(vec) > 1e-6)
            if vec[idx] < 0:
                vec[:] = -vec
        
        return np.column_stack((x, y))

    def score(points):
        # Calculate convex hull area
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
            if hull_area < 1e-10:
                return 0.0
        except Exception:
            return 0.0
            
        # Calculate minimum triangle area
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

    def objective(params):
        pts = get_points(params)
        # Check convexity
        try:
            hull = ConvexHull(pts)
            if len(hull.vertices) < N:
                return 1.0 # Penalty
        except Exception:
            return 1.0
            
        s = score(pts)
        return -s

    # Initial guess
    # Laplacian-like: d=2, o1=-1, o2=0
    x0 = np.array([2.0, -1.0, 0.0])
    
    # Bounds
    bounds = [(-2.0, 4.0), (-2.0, 2.0), (-2.0, 2.0)]
    
    # Optimize
    res = minimize(objective, x0, method='Nelder-Mead', 
                   options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    best_params = res.x
    best_points = get_points(best_params)
    
    # Ensure the result is valid (convex)
    try:
        hull = ConvexHull(best_points)
        if len(hull.vertices) < N:
             # Fallback to regular polygon
             angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
             return np.column_stack((np.cos(angles), np.sin(angles)))
    except Exception:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_points
