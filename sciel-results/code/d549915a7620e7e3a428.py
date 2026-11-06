import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50

def construct():
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    P = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    start_time = time.time()
    deadline = 110.0  # Keep safe margin from 120s limit
    
    # Initial evaluation
    # We need a function to compute the covering radius
    def compute_covering_radius(P):
        try:
            hull = ConvexHull(P)
            max_r = 0.0
            # Iterate over facets
            for simplex in hull.simplices:
                p1, p2, p3 = P[simplex[0]], P[simplex[1]], P[simplex[2]]
                # Compute circumcenter of triangle p1, p2, p3 on sphere
                # Circumcenter is the unit vector c such that c.p1 = c.p2 = c.p3
                # c is proportional to cross(p2-p1, p3-p1) ? No.
                # c is in the plane of p1, p2, p3.
                # c = (p1 + p2 + p3) is not necessarily the center.
                # The circumcenter of a triangle on the unit sphere is the normalized vector:
                # c ~ (p2 - p1) x (p3 - p1) is the normal.
                # Actually, the circumcenter is the intersection of the plane and the sphere.
                # It is the unit vector c that satisfies c dot p1 = c dot p2 = c dot p3.
                # This means c is orthogonal to p2-p1 and p3-p1.
                # So c is parallel to (p2-p1) x (p3-p1).
                # Let n = (p2-p1) x (p3-p1).
                # Then c = n / ||n||.
                # But we must check if c is on the "same side" as the triangle centroid.
                # The centroid g = (p1+p2+p3)/3.
                # If dot(c, g) < 0, then c = -c.
                
                v1 = p2 - p1
                v2 = p3 - p1
                n = np.cross(v1, v2)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-10:
                    continue # Degenerate
                c = n / norm_n
                g = (p1 + p2 + p3) / 3.0
                if np.dot(c, g) < 0:
                    c = -c
                
                # The angular distance is arccos(c dot p1)
                # We want to maximize this to find the worst facet
                cos_theta = np.dot(c, p1)
                # Clip for numerical safety
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                theta = np.arccos(cos_theta)
                if theta > max_r:
                    max_r = theta
            return max_r
        except Exception:
            return np.pi # Invalid hull, return large radius
    
    # Optimization Loop
    # We will use a simple gradient descent on the max circumradius.
    # To make it smoother, we can use a soft-max of the top few radii.
    # But let's stick to the max for simplicity and robustness, with a small step.
    
    step_size = 0.05
    # Adaptive step size: if radius increases, reduce step.
    
    for _ in range(10000): # Max iterations
        if time.time() > deadline:
            break
            
        # Compute hull and worst facet
        try:
            hull = ConvexHull(P)
        except Exception:
            # If hull fails, jitter points slightly
            P += np.random.normal(0, 1e-5, P.shape)
            P /= np.linalg.norm(P, axis=1, keepdims=True)
            continue
            
        worst_idx = -1
        worst_r = 0.0
        worst_c = None
        worst_simplex = None
        
        # We need to find the facet with max circumradius
        # And also store the circumcenter for the gradient step
        for simplex in hull.simplices:
            p1, p2, p3 = P[simplex[0]], P[simplex[1]], P[simplex[2]]
            v1 = p2 - p1
            v2 = p3 - p1
            n = np.cross(v1, v2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                continue
            c = n / norm_n
            g = (p1 + p2 + p3) / 3.0
            if np.dot(c, g) < 0:
                c = -c
            
            cos_theta = np.dot(c, p1)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta = np.arccos(cos_theta)
            
            if theta > worst_r:
                worst_r = theta
                worst_idx = simplex
                worst_c = c
                worst_simplex = simplex
                
        if worst_idx == -1:
            break
            
        # Gradient step
        # Move p1, p2, p3 towards c
        # Gradient direction for pi is: c - (c dot pi) pi
        # This is the projection of c onto the tangent plane at pi.
        
        p1, p2, p3 = P[worst_simplex[0]], P[worst_simplex[1]], P[worst_simplex[2]]
        
        d1 = worst_c - np.dot(worst_c, p1) * p1
        d2 = worst_c - np.dot(worst_c, p2) * p2
        d3 = worst_c - np.dot(worst_c, p3) * p3
        
        # Normalize directions to ensure consistent step size
        n1 = np.linalg.norm(d1)
        n2 = np.linalg.norm(d2)
        n3 = np.linalg.norm(d3)
        
        if n1 > 1e-10:
            P[worst_simplex[0]] += step_size * d1 / n1
        if n2 > 1e-10:
            P[worst_simplex[1]] += step_size * d2 / n2
        if n3 > 1e-10:
            P[worst_simplex[2]] += step_size * d3 / n3
        
        # Renormalize all points to unit sphere
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        
        # Optional: Check if radius improved, if not, reduce step size
        # For now, we keep step size constant or slightly decay it
        # step_size *= 0.999
        
    return P
