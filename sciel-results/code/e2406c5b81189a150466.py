"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.spatial import ConvexHull

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin before 150s kill
    
    # 1. Generate a high-resolution grid on the sphere for the covering energy
    # We use a Fibonacci spiral for the grid points to ensure uniform distribution.
    # ~20,000 points should be sufficient to capture the covering radius accurately.
    n_grid = 20000
    i = np.arange(n_grid) + 0.5
    z = 1.0 - 2.0 * i / n_grid
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    grid_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Precompute grid norms (should be 1, but safe to normalize)
    grid_norms = np.linalg.norm(grid_pts, axis=1, keepdims=True)
    grid_pts /= grid_norms

    def compute_covering_radius(pts):
        """
        Compute the exact covering radius in degrees.
        Uses ConvexHull to find Delaunay triangulation facets.
        For each facet, find the circumcenter and check if it's inside the facet.
        The covering radius is the max distance from any facet circumcenter to its vertices.
        """
        try:
            hull = ConvexHull(pts)
            max_r_rad = 0.0
            
            # Precompute pairwise distances for faster circumcenter check? 
            # For N=50, iterating over ~100 facets is fast.
            for face in hull.simplices:
                p0, p1, p2 = pts[face[0]], pts[face[1]], pts[face[2]]
                
                # Compute circumcenter of the spherical triangle
                # The circumcenter c satisfies c.p0 = c.p1 = c.p2 = cos(R)
                # Let c = a*p0 + b*p1 + c*p2 (coefficients a,b,c)
                # c.p0 = a + b*cos(p0.p1) + c*cos(p0.p2) = cos(R)
                # c.p1 = a*cos(p1.p0) + b + c*cos(p1.p2) = cos(R)
                # c.p2 = a*cos(p2.p0) + b*cos(p2.p1) + c = cos(R)
                
                # This is a linear system for a, b, c given cos(R).
                # However, there's a simpler geometric way for circumcenter on sphere.
                # The circumcenter is the normalized vector n such that n is orthogonal to (p0-p1) and (p0-p2)?
                # No, that's the centroid of the great circle arc? 
                
                # Let's use the property that the circumcenter is the intersection of the 
                # perpendicular bisectors of the chords p0-p1 and p0-p2.
                # In 3D, the circumcenter lies in the plane of p0, p1, p2.
                # Let normal = cross(p0, p1). 
                # Actually, the circumcenter is the normalized vector v such that 
                # v is in span(p0, p1, p2) and equidistant from p0, p1, p2.
                
                # A robust way: Solve for the vector c in span(p0, p1, p2) 
                # such that |c - p0|^2 = |c - p1|^2 = |c - p2|^2.
                # This implies c . (p0 - p1) = (|p0|^2 - |p1|^2)/2 = 0 (since on unit sphere)
                # So c is orthogonal to p0 - p1 and p0 - p2.
                # Thus c is parallel to cross(p0 - p1, p0 - p2).
                
                v1 = p0 - p1
                v2 = p0 - p2
                c_vec = np.cross(v1, v2)
                
                if np.linalg.norm(c_vec) < 1e-10:
                    # Collinear or coincident points, skip or handle
                    continue
                    
                c_vec /= np.linalg.norm(c_vec)
                
                # Check if c_vec is on the same side as the centroid (to pick the correct circumcenter if ambiguous)
                # Actually, there are two circumcenters on the sphere (antipodal). 
                # The relevant one for the facet is the one "inside" the triangle.
                # We can check the distance to the centroid.
                centroid = (p0 + p1 + p2) / 3.0
                centroid /= np.linalg.norm(centroid)
                
                if np.dot(c_vec, centroid) < 0:
                    c_vec = -c_vec
                    
                # Now c_vec is the circumcenter candidate.
                # Check if it lies within the spherical triangle.
                # A point is inside a spherical triangle if it can be expressed as a 
                # positive linear combination of the vertices (for small triangles) 
                # or if the angles sum to 2*pi.
                # Simpler check: The circumcenter is inside if the distance from circumcenter 
                # to each vertex is less than the distance to the antipode of the vertex? 
                # No.
                
                # Let's check if the circumcenter is in the convex hull of p0, p1, p2 in 3D?
                # No, spherical triangle.
                # A point c is inside spherical triangle (p0, p1, p2) if 
                # c . (p1 x p2) >= 0, c . (p2 x p0) >= 0, c . (p0 x p1) >= 0
                # (assuming consistent orientation).
                
                n1 = np.cross(p1, p2)
                n2 = np.cross(p2, p0)
                n3 = np.cross(p0, p1)
                
                # Ensure normal orientation is consistent with centroid
                if np.dot(centroid, n1) < 0:
                    n1 = -n1
                if np.dot(centroid, n2) < 0:
                    n2 = -n2
                if np.dot(centroid, n3) < 0:
                    n3 = -n3
                    
                if np.dot(c_vec, n1) >= -1e-8 and np.dot(c_vec, n2) >= -1e-8 and np.dot(c_vec, n3) >= -1e-8:
                    # Inside
                    dist = np.arccos(np.clip(np.dot(c_vec, p0), -1.0, 1.0))
                    if dist > max_r_rad:
                        max_r_rad = dist
                        
            return np.degrees(max_r_rad)
        except Exception:
            return 180.0

    def covering_energy(pts):
        """
        Approximate covering radius using grid points.
        Faster than exact hull method, suitable for optimization loop.
        """
        dots = grid_pts @ pts.T
        # For each grid point, find min distance to any config point
        # min_dist = arccos(max_dot)
        max_dots = np.max(dots, axis=1)
        # Clip for numerical stability
        max_dots = np.clip(max_dots, -1.0, 1.0)
        # We want to minimize the MAX angular distance.
        # max_angle = arccos(min_max_dot)
        # To make it differentiable/smooth, we can use a soft-max or just minimize the max directly.
        # Minimize arccos(min_max_dot) is equivalent to maximizing min_max_dot.
        # So objective = -min_max_dot.
        # However, we want to minimize the covering radius.
        # Let's return the max angular distance in radians.
        # To make it smoother for optimization, we can use a large power of (1 - max_dot)
        # or just use the max dot product and maximize it.
        # Let's return -min_max_dot. Minimizing this maximizes the minimum dot product,
        # which minimizes the maximum angle.
        return -np.min(max_dots)

    def energy_and_grad(x):
        """
        Compute energy and gradient for L-BFGS-B.
        x: flattened array of N*3 points.
        """
        pts = x.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        pts = pts / norms
        
        # Energy: -min_i (grid . p_i) for each grid point, then max over grid points.
        # This is not differentiable at the max.
        # Let's use a smooth approximation: 
        # E = -log(sum_i exp(-beta * (grid . p_i))) ... no, that's for min.
        # We want min over i of d(y, p_i). 
        # d(y, p_i) = arccos(y . p_i).
        # Minimizing max_y min_i arccos(y . p_i).
        
        # Let's stick to the grid max for simplicity and use Nelder-Mead or Powell which don't need gradients.
        # Or, use a smooth max approximation.
        # Soft-max of the distances.
        
        dots = grid_pts @ pts.T
        # d(y, p_i) approx proportional to (1 - y.p_i) for small angles? 
        # Actually, let's just use the raw max dot product approach but smooth it.
        # We want to maximize the "worst case" dot product.
        # Let D = min_i (grid . p_i). We want to maximize min(D).
        # This is a max-min problem.
        
        # Alternative: Minimize sum of exp(beta * (1 - max_dot)) 
        # This penalizes points with low max_dot (large distance).
        
        beta = 20.0
        max_dots = np.max(dots, axis=1)
        # 1 - max_dot is proportional to squared distance for small angles.
        # Let's use exp(beta * (1 - max_dot))
        vals = np.exp(beta * (1.0 - max_dots))
        energy = np.sum(vals)
        
        # Gradient
        # dE/dp_j = sum_k dE/d(max_dot_k) * d(max_dot_k)/dp_j
        # max_dot_k = max_i (grid_k . p_i)
        # This is non-differentiable where the argmax changes.
        # We can approximate by using the gradient of the max, which is the gradient of the active term.
        # Let's use a soft-max for the max operation to get a gradient.
        # soft_max_i (grid_k . p_i) = sum_i exp(beta * (grid_k . p_i)) / sum_j exp(beta * (grid_k . p_j))
        
        # Let's re-implement with soft-max for differentiability.
        exp_dots = np.exp(beta * dots)
        sum_exp = np.sum(exp_dots, axis=1, keepdims=True)
        probs = exp_dots / sum_exp  # (n_grid, N)
        
        # soft_max_dot = sum_i probs[k, i] * (grid_k . p_i)
        # E = sum_k exp(beta * (1 - soft_max_dot_k))
        
        # dE/dp_j = sum_k exp(beta * (1 - soft_max_dot_k)) * (-beta) * d(soft_max_dot_k)/dp_j
        # d(soft_max_dot_k)/dp_j = sum_i probs[k, i] * d(grid_k . p_i)/dp_j
        # If i == j, d(grid_k . p_j)/dp_j = grid_k
        # If i != j, 0 (in the soft max derivative wrt the input, but here we are differentiating the max value)
        # Wait, soft_max_dot_k = sum_i probs[k, i] * (grid_k . p_i).
        # The derivative of this wrt p_j involves the derivative of probs and the dot product.
        # This is getting complex.
        
        # Let's just use Powell's method or Nelder-Mead which are derivative-free.
        # Given N=50, 150 dimensions. Powell might be slow.
        # Let's try a simpler energy: Sum of inverse squared distances to nearest neighbor? No, that's Tammes.
        
        # Let's return the energy for a derivative-free optimizer.
        return energy

    # Strategy: Multi-start with Powell's method on the smooth covering energy.
    # Powell is derivative-free and robust for this kind of non-smooth (but smoothed) objective.
    
    best_pts = None
    best_cr = 180.0
    rng = np.random.default_rng(42)
    
    max_restarts = 10
    
    # Initial best: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_cr = compute_covering_radius(fib_pts)
    best_pts = fib_pts
    
    for k in range(max_restarts):
        if time.time() > deadline - 10:
            break
            
        # Random initialization
        init_pts = rng.normal(size=(N, 3))
        init_pts /= np.linalg.norm(init_pts, axis=1, keepdims=True)
        x0 = init_pts.flatten()
        
        # Optimize using Powell
        # We use the smooth energy defined above
        from scipy.optimize import minimize
        
        # Powell can be slow. Let's try Nelder-Mead with a tight tolerance.
        # Or use COBYLA which is good for constrained problems (sphere constraint handled by normalization inside energy).
        
        # Let's use Powell with maxiter limit.
        res = minimize(energy_and_grad, x0, method='Powell', options={'maxiter': 200, 'ftol': 1e-8})
        
        if res.success or res.fun < np.inf:
            pts = res.x.reshape(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            
            # Validate and compute true covering radius
            cr = compute_covering_radius(pts)
            if cr < best_cr:
                best_cr = cr
                best_pts = pts
                # Optional: Refine with a local search step?
                
    # Final refinement: Local search on the best point
    # Try small perturbations
    if time.time() < deadline - 5:
        current_pts = best_pts.copy()
        for _ in range(20):
            if time.time() > deadline - 2:
                break
            # Random perturbation
            delta = rng.normal(size=(N, 3)) * 0.01
            new_pts = current_pts + delta
            new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
            cr_new = compute_covering_radius(new_pts)
            if cr_new < best_cr:
                best_cr = cr_new
                current_pts = new_pts
            else:
                # Backtrack? Or just keep current best
                pass
        best_pts = current_pts

    return best_pts
# EVOLVE-BLOCK-END
