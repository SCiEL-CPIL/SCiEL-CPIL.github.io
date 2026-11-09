# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def construct():
    """
    Minimize the covering radius by maximizing the minimum pairwise angular separation.
    Formulated as: minimize the soft-max of pairwise dot products (proxy for max dot product),
    using projected gradient descent on the sphere with adaptive learning rate and line search.
    Includes random restarts and a final local refinement step targeting the covering radius.
    """
    
    def fibonacci_sphere(n):
        """Generate n points on the unit sphere using Fibonacci spiral."""
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return pts

    def random_sphere(n):
        """Generate n random points on the unit sphere."""
        v = np.random.randn(n, 3)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return v

    def objective_and_grad(X, beta):
        """
        Compute the soft-max objective and its gradient.
        Objective: (1/beta) * log(sum_{i!=j} exp(beta * dot(x_i, x_j)))
        """
        # Compute pairwise dot products
        dots = X @ X.T
        # Mask out diagonal
        np.fill_diagonal(dots, -np.inf)
        
        # For numerical stability, subtract max
        max_dot = np.max(dots)
        exp_dots = np.exp(beta * (dots - max_dot))
        
        # Sum over all pairs (excluding diagonal)
        sum_exp = np.sum(exp_dots)
        
        # Objective
        obj = max_dot + np.log(sum_exp) / beta
        
        # Gradient: for each point k, grad_k = 2 * sum_{j!=k} (exp_dots[k,j] / sum_exp) * x_j
        grad = 2.0 * (exp_dots @ X) / sum_exp
        
        # Project gradient onto tangent space of sphere
        proj_grad = grad - np.sum(grad * X, axis=1, keepdims=True) * X
        
        return obj, proj_grad

    def line_search(X, grad, obj, beta, lr_init=0.1, c=1e-4, rho=0.5, max_iter=20):
        """
        Backtracking line search to find a step size that decreases the objective.
        """
        lr = lr_init
        for _ in range(max_iter):
            X_new = X - lr * grad
            # Project back to unit sphere
            norms = np.linalg.norm(X_new, axis=1, keepdims=True)
            X_new = X_new / norms
            
            obj_new, _ = objective_and_grad(X_new, beta)
            
            # Armijo condition
            if obj_new <= obj - c * lr * np.sum(np.sum(grad * X, axis=1)):
                return X_new, obj_new, lr
            lr *= rho
        # If no good step found, use the smallest step
        X_new = X - lr * grad
        norms = np.linalg.norm(X_new, axis=1, keepdims=True)
        X_new = X_new / norms
        obj_new, _ = objective_and_grad(X_new, beta)
        return X_new, obj_new, lr

    def optimize(X_init, n_iter=1000, beta_start=10.0, beta_end=100.0):
        """
        Run projected gradient descent with increasing beta.
        """
        X = X_init.copy()
        best_obj = np.inf
        best_X = X.copy()
        
        # Schedule beta from beta_start to beta_end
        betas = np.linspace(beta_start, beta_end, n_iter)
        
        for it in range(n_iter):
            beta = betas[it]
            obj, grad = objective_and_grad(X, beta)
            
            if obj < best_obj:
                best_obj = obj
                best_X = X.copy()
                
            X, obj, _ = line_search(X, grad, obj, beta)
            
        return best_X

    def covering_radius(X):
        """
        Calculate the covering radius in degrees.
        This is the maximum angular distance from any point on the sphere to the nearest point in X.
        Approximated by the maximum circumradius of the facets of the convex hull.
        """
        try:
            hull = ConvexHull(X)
            max_circumradius = 0.0
            for simplex in hull.simplices:
                # Get the points defining the facet
                p = X[simplex]
                # Calculate the circumradius of the triangle defined by p
                # For a triangle with sides a, b, c and area A, circumradius R = abc / (4A)
                # But on a sphere, it's the angle subtended by the arc.
                # Alternatively, find the point on the sphere farthest from the facet center.
                # The farthest point is the normalized centroid of the facet vertices? No.
                # The covering radius is the maximum distance to the nearest point.
                # This is equivalent to the maximum radius of a circle (cap) that fits inside a Voronoi cell.
                # For the convex hull, the empty caps are centered at the circumcenters of the facets.
                
                # Let's compute the circumcenter of the triangle on the sphere.
                # If the triangle is acute, the circumcenter is inside.
                # The distance from the circumcenter to the vertices is the radius.
                
                # A simpler robust method: 
                # The covering radius is the max over all points y on sphere of min_i angle(y, x_i).
                # This is equal to the max circumradius of the spherical Voronoi cells.
                # The vertices of the Voronoi cells are the circumcenters of the hull facets.
                
                # Calculate circumcenter of the triangle formed by p[0], p[1], p[2] on the unit sphere.
                # Let a, b, c be the vertices.
                # The circumcenter n is proportional to a*sin(A) + b*sin(B) + c*sin(C) where A, B, C are angles opposite to a, b, c?
                # Actually, the circumcenter of a spherical triangle is the intersection of the perpendicular bisectors.
                # For a triangle on a sphere, the circumcenter is given by the normalized vector:
                # n = (a x b) sin(c_angle) + (b x c) sin(a_angle) + (c x a) sin(b_angle) ? No.
                
                # Let's use the property that the circumcenter is equidistant from the vertices.
                # We can solve for n such that n dot a = n dot b = n dot c.
                # This is a linear system if we fix the norm? No, it's on the sphere.
                # The vector n is orthogonal to (a-b) and (b-c) in the tangent plane? No.
                # n dot (a-b) = 0 and n dot (b-c) = 0.
                # So n is parallel to (a-b) x (b-c).
                # Let's check if this is the circumcenter.
                # The plane containing a, b, c does not pass through the origin generally.
                # The circumcenter of the spherical triangle is the point on the sphere whose projection onto the plane of the triangle is the circumcenter of the planar triangle? No.
                
                # Correct formula for circumcenter of spherical triangle with vertices u, v, w:
                # Let A = u x v, B = v x w, C = w x u.
                # The circumcenter is proportional to A/sin(w) + B/sin(u) + C/sin(v)?
                # Let's use a simpler numerical approach for the radius.
                # The covering radius is the maximum distance to the nearest point.
                # We can sample the sphere densely? Too slow.
                
                # Let's calculate the circumradius of the planar triangle formed by the vectors, then convert?
                # No, the spherical circumradius is what we want.
                
                # Let's find the point n on the sphere that is equidistant from u, v, w.
                # n is in the span of u, v, w.
                # n = alpha u + beta v + gamma w.
                # n dot u = n dot v = n dot w = r.
                # This gives a linear system for alpha, beta, gamma.
                
                M = np.array([
                    [np.dot(u, u), np.dot(v, u), np.dot(w, u)],
                    [np.dot(u, v), np.dot(v, v), np.dot(w, v)],
                    [np.dot(u, w), np.dot(v, w), np.dot(w, w)]
                ])
                # We want n dot u = r, n dot v = r, n dot w = r.
                # [u v w] n = r [1 1 1]^T ? No.
                # n = alpha u + beta v + gamma w
                # n dot u = alpha + beta (v dot u) + gamma (w dot u) = r
                # n dot v = alpha (u dot v) + beta + gamma (w dot v) = r
                # n dot w = alpha (u dot w) + beta (v dot w) + gamma = r
                
                # Let D be the Gram matrix of u, v, w.
                # D [alpha, beta, gamma]^T = r [1, 1, 1]^T
                # [alpha, beta, gamma]^T = r D^{-1} [1, 1, 1]^T
                # We need to find r such that ||n|| = 1.
                # n = r D^{-1} [1, 1, 1]^T
                # ||n||^2 = r^2 [1, 1, 1]^T D^{-1} D D^{-1} [1, 1, 1]^T = r^2 [1, 1, 1]^T D^{-1} [1, 1, 1]^T = 1
                # So r = 1 / sqrt([1, 1, 1]^T D^{-1} [1, 1, 1]^T)
                
                u, v, w = p[0], p[1], p[2]
                D = np.array([
                    [np.dot(u, u), np.dot(v, u), np.dot(w, u)],
                    [np.dot(u, v), np.dot(v, v), np.dot(w, v)],
                    [np.dot(u, w), np.dot(v, w), np.dot(w, w)]
                ])
                try:
                    D_inv = np.linalg.inv(D)
                    ones = np.ones(3)
                    val = ones @ D_inv @ ones
                    if val <= 0:
                        continue
                    r = 1.0 / np.sqrt(val)
                    # The angular distance is arccos(r)
                    angle = np.arccos(np.clip(r, -1.0, 1.0))
                    max_circumradius = max(max_circumradius, angle)
                except np.linalg.LinAlgError:
                    continue
            
            return np.degrees(max_circumradius)
        except Exception:
            return 180.0

    # Optimization Loop
    best_X = None
    best_radius = np.inf
    
    # Run multiple restarts
    n_restarts = 5
    for i in range(n_restarts):
        if i == 0:
            X_init = fibonacci_sphere(N)
        else:
            X_init = random_sphere(N)
        
        X_opt = optimize(X_init, n_iter=500)
        
        # Calculate covering radius
        rad = covering_radius(X_opt)
        
        if rad < best_radius:
            best_radius = rad
            best_X = X_opt
            
    return best_X

# EVOLVE-BLOCK-END
