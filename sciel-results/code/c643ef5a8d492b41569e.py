import numpy as np
from scipy.optimize import minimize

N = 11

# Triangle vertices
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3)/2])

# Pre-compute all triangle indices
tri_indices = []
for i in range(N-2):
    for j in range(i+1, N-1):
        for k in range(j+1, N):
            tri_indices.append((i, j, k))
tri_indices = np.array(tri_indices)

def min_area_vectorized(pts):
    """
    Compute minimum triangle area for N points, vectorized.
    pts: (N, 2) array
    """
    # For each triangle (i,j,k), compute cross product
    pi = pts[tri_indices[:, 0]]  # (165, 2)
    pj = pts[tri_indices[:, 1]]  # (165, 2)
    pk = pts[tri_indices[:, 2]]  # (165, 2)
    
    # Cross product: (pj - pi) x (pk - pi)
    v1 = pj - pi  # (165, 2)
    v2 = pk - pi  # (165, 2)
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]  # (165,)
    
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def objective_flat(x):
    """
    Objective function for Nelder-Mead: minimize negative min area.
    x: flat array of length 2*N
    """
    pts = x.reshape(N, 2)
    
    # Project points to triangle if needed
    pts = project_points_to_triangle(pts)
    
    area = min_area_vectorized(pts)
    return -area  # Negate because we want to maximize

def project_points_to_triangle(pts):
    """
    Project each point to the triangle if outside.
    Uses barycentric coordinates.
    """
    # Barycentric coordinates for triangle ABC
    # For point P, barycentric coords (a, b, c) such that P = a*A + b*B + c*C
    # and a + b + c = 1
    
    # Vectorized barycentric computation
    # Using the formula with cross products
    
    # Edge vectors
    AB = B - A
    BC = C - B
    CA = A - C
    
    # For each point, compute barycentric coordinates
    # Using the method: 
    # v0 = B - A, v1 = C - A, v2 = P - A
    # d00 = dot(v0, v0), d01 = dot(v0, v1), d11 = dot(v1, v1)
    # d02 = dot(v0, v2), d12 = dot(v1, v2)
    # denom = d00 * d11 - d01 * d01
    # v = (d11 * d02 - d01 * d12) / denom
    # w = (d00 * d12 - d01 * d02) / denom
    # u = 1 - v - w
    
    v0 = B - A  # (2,)
    v1 = C - A  # (2,)
    
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    
    denom = d00 * d11 - d01 * d01
    
    # For all points at once
    v2 = pts - A  # (N, 2)
    
    d02 = np.sum(v0 * v2, axis=1)  # (N,)
    d12 = np.sum(v1 * v2, axis=1)  # (N,)
    
    v = (d11 * d02 - d01 * d12) / denom  # (N,)
    w = (d00 * d12 - d01 * d02) / denom  # (N,)
    u = 1 - v - w  # (N,)
    
    # Check if inside: all barycentric coords >= 0
    inside = (u >= -1e-10) & (v >= -1e-10) & (w >= -1e-10)
    
    # For points outside, project to closest edge
    # We'll handle this by clamping barycentric coords to [0,1] and renormalizing
    # Actually, let's just project to the triangle properly
    
    # For simplicity, let's use a simpler projection:
    # If outside, find the closest edge and project onto it
    
    projected = pts.copy()
    
    for i in range(N):
        if not inside[i]:
            # Point is outside, project to closest edge
            p = pts[i]
            
            # Distance to each edge
            def dist_to_segment(p, a, b):
                ab = b - a
                ap = p - a
                t = np.dot(ap, ab) / np.dot(ab, ab)
                t = max(0, min(1, t))
                proj = a + t * ab
                return np.linalg.norm(p - proj), proj
            
            d1, proj1 = dist_to_segment(p, A, B)
            d2, proj2 = dist_to_segment(p, B, C)
            d3, proj3 = dist_to_segment(p, C, A)
            
            min_d = min(d1, d2, d3)
            if min_d == d1:
                projected[i] = proj1
            elif min_d == d2:
                projected[i] = proj2
            else:
                projected[i] = proj3
    
    return projected

def random_point_in_triangle(rng):
    """Generate a random point inside the triangle using barycentric coordinates."""
    u = rng.random()
    v = rng.random()
    if u + v > 1:
        u = 1 - u
        v = 1 - v
    return u * A + v * B + (1 - u - v) * C

def construct():
    """
    Main function: find 11 points in the unit triangle maximizing minimum triangle area.
    Uses Nelder-Mead optimization with multiple restarts.
    """
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = -np.inf
    
    # Multiple restarts with different initial configurations
    n_restarts = 8
    
    for restart in range(n_restarts):
        # Initialize with random points
        x0 = np.zeros(2 * N)
        for i in range(N):
            p = random_point_in_triangle(rng)
            x0[2*i] = p[0]
            x0[2*i+1] = p[1]
        
        # Run Nelder-Mead
        result = minimize(
            objective_flat,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 5000,
                'xatol': 1e-6,
                'fatol': 1e-8,
                'adaptive': True
            }
        )
        
        # Extract points and project to triangle
        pts = result.x.reshape(N, 2)
        pts = project_points_to_triangle(pts)
        
        score = min_area_vectorized(pts)
        
        if score > best_score:
            best_score = score
            best_points = pts.copy()
        
        # If we found a good solution, we can stop early
        if best_score > 0.015:  # Threshold for good solution
            break
    
    return best_points
