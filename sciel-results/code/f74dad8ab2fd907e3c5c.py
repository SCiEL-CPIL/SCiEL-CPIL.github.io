import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50

def construct():
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    start_time = time.time()
    time_limit = 110.0  # Leave some margin for the evaluator
    
    # Local search
    for it in range(10000):
        if time.time() - start_time > time_limit:
            break
        
        try:
            hull = ConvexHull(X)
            facets = hull.simplices
        except Exception:
            break
            
        worst_radius = 0
        worst_center = None
        worst_facet = None
        
        for f in facets:
            A, B, C = X[f[0]], X[f[1]], X[f[2]]
            # Compute circumcenter and circumradius
            a = np.linalg.norm(B - C)
            b = np.linalg.norm(A - C)
            c = np.linalg.norm(A - B)
            
            if a < 1e-10 or b < 1e-10 or c < 1e-10:
                continue
                
            cross = np.cross(B - A, C - A)
            area = 0.5 * np.linalg.norm(cross)
            if area < 1e-10:
                continue
                
            R = (a * b * c) / (4 * area)
            
            # Circumcenter
            n = cross
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                continue
            c1 = n / n_norm
            c2 = -c1
            centroid = (A + B + C) / 3
            if np.dot(c1, centroid) > np.dot(c2, centroid):
                center = c1
            else:
                center = c2
                
            radius = np.arccos(np.clip(np.dot(center, A), -1, 1))
            
            if radius > worst_radius:
                worst_radius = radius
                worst_center = center
                worst_facet = f
                
        if worst_facet is None:
            break
            
        # Move the vertices of the worst facet towards the circumcenter
        alpha = 0.1
        X[worst_facet[0]] = X[worst_facet[0]] + alpha * (worst_center - X[worst_facet[0]])
        X[worst_facet[1]] = X[worst_facet[1]] + alpha * (worst_center - X[worst_facet[1]])
        X[worst_facet[2]] = X[worst_facet[2]] + alpha * (worst_center - X[worst_facet[2]])
        
        # Normalize
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        
    return X
