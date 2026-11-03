import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def compute_covering_radius_and_facets(points):
    """
    Computes the covering radius and returns the radii and indices for each facet.
    points: (N, 3) array of unit vectors.
    Returns:
        max_radius: float
        radii: (M,) array of circumradii for each facet
        simplices: (M, 3) array of vertex indices for each facet
    """
    try:
        hull = ConvexHull(points)
        simplices = hull.simplices
    except Exception:
        # Fallback if hull fails (shouldn't happen for random points)
        return np.pi, np.array([]), np.array([])
    
    radii = np.zeros(len(simplices))
    
    for idx, simplex in enumerate(simplices):
        i, j, k = simplex
        a = points[i]
        b = points[j]
        c = points[k]
        
        # Normal vector to the plane of the triangle
        n = np.cross(b - a, c - a)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            # Degenerate triangle, radius is effectively pi/2 or undefined
            radii[idx] = np.pi / 2
            continue
            
        u = n / norm_n
        
        # Ensure u points towards the centroid (inside the spherical triangle)
        centroid = (a + b + c) / 3.0
        if np.dot(u, centroid) < 0:
            u = -u
            
        # Circumradius is the angle between u and any vertex
        cos_theta = np.dot(u, a)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        radii[idx] = np.arccos(cos_theta)
        
    max_radius = np.max(radii) if len(radii) > 0 else np.pi
    return max_radius, radii, simplices

def construct():
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    best_points = None
    best_radius = np.pi

    # New Approach: Repulsive Force Minimization (Lennard-Jones style)
    # Energy E = sum_{i<j} 1 / (1 - p_i . p_j)^2
    # This encourages uniform distribution on the sphere.
    
    for restart in range(10):
        if time.time() - start_time > time_limit:
            break
            
        np.random.seed(100 + restart)
        points = np.random.randn(N, 3)
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        current_radius = np.pi
        no_improve_count = 0
        
        # Adaptive learning rate
        alpha = 0.01
        min_alpha = 1e-6
        
        for step in range(3000):
            if time.time() - start_time > time_limit:
                break
                
            # Compute repulsive forces
            # Dots[i, j] = points[i] . points[j]
            dots = points @ points.T
            np.fill_diagonal(dots, 1.0)
            
            # Force term: 2 * (1 - dots)^-3 * points[j]
            # We want the gradient of the energy.
            # E = sum_{i!=j} 0.5 * (1 - dots[i,j])^-2  (factor 0.5 to simplify sum over i,j vs i<j)
            # dE/dp_i = sum_{j!=i} (1 - dots[i,j])^-3 * (-p_j) * (-1) ?? 
            # Let u = p_i . p_j. d(1-u)^-2 / du = 2(1-u)^-3.
            # d u / d p_i = -p_j.
            # dE/dp_i = sum 2(1-u)^-3 (-p_j) = -2 sum (1-u)^-3 p_j.
            # Gradient Descent: p_i <- p_i - alpha * dE/dp_i = p_i + 2 alpha sum (1-u)^-3 p_j.
            
            # Calculate weights W[i, j] = (1 - dots[i, j])^-3
            # Avoid division by zero if points coincide
            diff = 1.0 - dots
            diff[diff < 1e-6] = 1e-6
            weights = 1.0 / (diff ** 3)
            np.fill_diagonal(weights, 0.0)
            
            # Force on i is sum_j weights[i, j] * points[j]
            # F = weights @ points
            forces = weights @ points
            forces *= 2.0 # From the derivative
            
            # Project forces to tangent plane to keep on sphere
            # Tangent vector v = F - (F . p) p
            dot_forces = np.sum(forces * points, axis=1, keepdims=True)
            tangent_forces = forces - dot_forces * points
            
            # Update
            # Gradient points towards neighbors (increases energy if we move towards them? No, energy is high when close. Moving towards them increases closeness, which increases energy? 
            # Wait.
            # If p_i and p_j are close, 1-dot is small, weight is large.
            # Force vector F_i = sum w_ij p_j.
            # If p_j is close to p_i, p_j is roughly in direction of p_i.
            # So F_i points roughly in direction of p_i.
            # This means the gradient points "outward" from the center?
            # No, p_j is a unit vector.
            # Let's test 1D analogy.
            # Energy (1-x)^-2. Derivative 2(1-x)^-3. Positive.
            # Gradient is positive. We want to decrease energy, so move x in negative direction.
            # Here, "x" is the dot product.
            # Increasing dot product decreases (1-dot), increases energy.
            # So gradient of energy w.r.t dot product is positive.
            # Gradient w.r.t p_i: dE/du * du/dp_i = Positive * (-p_j) = Negative * p_j.
            # So Gradient vector points in direction of -p_j.
            # We want to move in direction -Gradient = +p_j.
            # So we move TOWARDS p_j?
            # This implies the energy is a potential well that attracts points?
            # No, $1/(1-u)^2$ goes to infinity as $u \to 1$.
            # So energy is HIGH when points are close.
            # We want to move to LOWER energy.
            # So we should move AWAY from close points.
            # Let's re-calculate derivative carefully.
            # $f(u) = (1-u)^{-2}$.
            # $f'(u) = -2(1-u)^{-3} \cdot (-1) = 2(1-u)^{-3}$.
            # This is positive.
            # $E = \sum f(u_{ij})$.
            # $\nabla_{p_i} E = \sum \nabla_{p_i} f(u_{ij}) = \sum f'(u_{ij}) \nabla_{p_i} u_{ij}$.
            # $u_{ij} = p_i \cdot p_j$.
            # $\nabla_{p_i} u_{ij} = p_j$.
            # So $\nabla_{p_i} E = \sum 2(1-u_{ij})^{-3} p_j$.
            # This vector points in the direction of the weighted sum of neighbors.
            # If neighbors are close, this vector points towards them.
            # Gradient points in direction of INCREASING energy.
            # So moving in direction of Gradient increases energy.
            # We want to DECREASE energy.
            # So we must move in direction $-\nabla E$.
            # $-\nabla E = - \sum 2(1-u_{ij})^{-3} p_j$.
            # This points AWAY from the neighbors.
            # So `points -= alpha * forces` is CORRECT.
            points -= alpha * tangent_forces
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            
            # Evaluate covering radius every 10 steps to save time
            if step % 10 == 0:
                max_radius, radii, simplices = compute_covering_radius_and_facets(points)
                
                if max_radius < best_radius:
                    best_radius = max_radius
                    best_points = points.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
                if no_improve_count > 50:
                    alpha *= 0.9
                    if alpha < min_alpha:
                        break
            else:
                # If not evaluating, we can't update no_improve_count accurately for the radius,
                # but we can still decay alpha slowly or keep it.
                pass

            # Random perturbation to escape local minima
            if step % 100 == 0 and alpha > min_alpha * 10:
                points += np.random.randn(N, 3) * 0.01
                points /= np.linalg.norm(points, axis=1, keepdims=True)
                
            if alpha < min_alpha:
                break
                
        # End of restart loop
        if time.time() - start_time > time_limit:
            break

    if best_points is None:
        # Fallback to Fibonacci if optimization failed completely
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
    return best_points
