"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from itertools import combinations


def construct():
    """
    Place 13 points in a planar convex region normalized to unit area to maximize
    the minimum area over all triples.
    
    Approach:
    1. Parameterize points in convex position using angular ordering.
    2. Use logarithmic barrier to enforce convexity (positive cross products).
    3. Maximize minimum squared triangle area / hull area using smooth optimization.
    """
    
    def compute_hull_area(pts):
        """Compute convex hull area using shoelace formula on points in convex order."""
        n = len(pts)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += pts[i, 0] * pts[j, 1]
            area -= pts[j, 0] * pts[i, 1]
        return 0.5 * abs(area)
    
    def all_triangle_areas(pts):
        """Compute all triangle areas among triples of points."""
        n = len(pts)
        areas = np.zeros((n, n, n))
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    a = pts[i]
                    b = pts[j]
                    c = pts[k]
                    area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
                    areas[i, j, k] = area
        return areas
    
    def cross_products(pts):
        """Compute cross products for consecutive triples in convex order."""
        n = len(pts)
        crosses = np.zeros(n)
        for i in range(n):
            p1 = pts[i]
            p2 = pts[(i + 1) % n]
            p3 = pts[(i + 2) % n]
            crosses[i] = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
        return crosses
    
    def objective_with_barrier(x, mu):
        """
        Objective: minimize -min_triangle_area^2 / hull_area + barrier term.
        Barrier term: -mu * sum(log(cross_i)) for convexity.
        """
        pts = x.reshape(N, 2)
        
        # Compute cross products for convexity
        crosses = cross_products(pts)
        
        # If any cross product is non-positive, return large penalty
        if np.any(crosses <= 1e-12):
            return 1e10
        
        # Logarithmic barrier for convexity
        barrier = -mu * np.sum(np.log(crosses))
        
        # Compute hull area
        hull_area = compute_hull_area(pts)
        if hull_area < 1e-12:
            return 1e10
        
        # Compute minimum triangle area
        min_area = float('inf')
        for i in range(N):
            for j in range(i + 1, N):
                for k in range(j + 1, N):
                    a = pts[i]
                    b = pts[j]
                    c = pts[k]
                    area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
                    if area < min_area:
                        min_area = area
        
        # We want to maximize min_area / hull_area
        # So minimize -(min_area / hull_area) + barrier
        obj = -(min_area / hull_area) + barrier
        
        return obj
    
    def gradient_with_barrier(x, mu):
        """Compute gradient of objective with barrier using finite differences."""
        eps = 1e-7
        grad = np.zeros_like(x)
        f0 = objective_with_barrier(x, mu)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            f_plus = objective_with_barrier(x_plus, mu)
            grad[i] = (f_plus - f0) / eps
        return grad
    
    # Start from regular 13-gon
    angles_init = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points_init = np.column_stack([np.cos(angles_init), np.sin(angles_init)])
    x0 = points_init.flatten()
    
    # Barrier parameter schedule
    mu0 = 0.1
    mu_factor = 0.5
    max_mu_steps = 15
    
    x_current = x0.copy()
    mu = mu0
    
    for step in range(max_mu_steps):
        # Optimize with current mu
        result = minimize(
            objective_with_barrier,
            x_current,
            args=(mu,),
            method='BFGS',
            jac=gradient_with_barrier,
            options={'maxiter': 200, 'gtol': 1e-8, 'disp': False}
        )
        
        x_current = result.x
        
        # Reduce barrier parameter
        mu *= mu_factor
        
        # If mu is very small, we're close to the unconstrained optimum
        if mu < 1e-4:
            break
    
    optimized_points = x_current.reshape(N, 2)
    
    # Final check: ensure convex position
    crosses = cross_products(optimized_points)
    if np.any(crosses <= 1e-10):
        # Fall back to regular polygon
        optimized_points = points_init
    
    # Normalize: center at origin and scale so convex hull area = 1
    hull_area = compute_hull_area(optimized_points)
    if hull_area > 1e-10:
        scale = 1.0 / np.sqrt(hull_area)
        centroid = optimized_points.mean(axis=0)
        optimized_points = (optimized_points - centroid) * scale
    
    return optimized_points
# EVOLVE-BLOCK-END
