import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Ensure points are on unit sphere
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    best_points = points.copy()
    best_radius = np.inf
    
    # Helper to compute covering radius from convex hull
    def compute_covering_radius(pts):
        try:
            hull = ConvexHull(pts)
            facets = hull.simplices
            max_r = 0.0
            for f in facets:
                a, b, c = pts[f[0]], pts[f[1]], pts[f[2]]
                # Circumcenter of triangle on unit sphere
                # Cross product gives normal
                n = np.cross(b - a, c - a)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-10:
                    continue
                n /= norm_n
                # The circumcenter m is along n? No, the circumcenter of a triangle on a sphere is the intersection of the sphere with the line passing through the origin and the circumcenter of the planar triangle?
                # Actually, for a triangle on a unit sphere, the circumcenter is the normalized vector sum of the vertices if the triangle is acute? No.
                # The circumcenter $m$ satisfies $m \cdot a = m \cdot b = m \cdot c = \cos(R)$.
                # $m$ is in the plane spanned by a, b, c? No, $m$ is on the sphere.
                # The vector $v = a \times b + b \times c + c \times a$ points to the circumcenter?
                # Let's use the formula: $m \propto a \times b + b \times c + c \times a$ is not generally correct for arbitrary triangles.
                # Better: Solve linear system for $m$.
                # $m \cdot a = m \cdot b = m \cdot c = \cos R$
                # And $|m| = 1$.
                # This is equivalent to finding the point on the sphere equidistant from a, b, c.
                # The circumcenter of the spherical triangle.
                # Direction is given by the normalized sum of the vertices? No.
                # Let's use the property that the circumcenter is perpendicular to the edges?
                # Simplest: The circumcenter $m$ is parallel to the vector $a \sin(2A) + b \sin(2B) + c \sin(2C)$? Too complex.
                
                # Alternative: The circumradius $R$ is the angle $\arccos(m \cdot a)$.
                # We can compute $R$ directly from the side lengths of the triangle.
                # Sides: $c = \arccos(a \cdot b)$, $b = \arccos(a \cdot c)$, $a = \arccos(b \cdot c)$.
                # Spherical Law of Cosines: $\cos a = \cos b \cos c + \sin b \sin c \cos A$.
                # This gives angles.
                # Circumradius formula for spherical triangle:
                # $\sin R = \frac{\sin A \sin B \sin C}{\sin a \sin b \sin C}$? No.
                # $\cos R = \frac{\cos a \cos b \cos c + 2 \cos a \cos b \cos c}{\dots}$?
                # Formula: $\cos R = \frac{1 + \cos a + \cos b + \cos c}{2 \sqrt{(1+\cos a+\cos c-\cos b)(1+\cos b+\cos c-\cos a)(1+\cos a+\cos b-\cos c)}}$? 
                # Actually, there is a simpler vector formula.
                # The circumcenter $m$ is the normalized vector $a \times b + b \times c + c \times a$ ONLY if the triangle is right-angled? No.
                
                # Let's just compute the distance from the origin to the plane? No.
                # Let's use the fact that $m$ is the solution to $M m = \cos R \mathbf{1}$ where rows are a, b, c.
                # This is underdetermined.
                # However, $m$ is orthogonal to $a \times b$? No.
                # $m$ lies in the plane defined by a, b, c? Yes, the circumcenter of a spherical triangle lies in the plane of the triangle? No, the plane of the triangle is the one passing through a,b,c and the origin. Yes, it is.
                # So $m = \alpha a + \beta b + \gamma c$.
                # And $m \cdot a = m \cdot b = m \cdot c$.
                # Let $G$ be the Gram matrix of a,b,c. $G_{ij} = p_i \cdot p_j$.
                # $G \mathbf{x} = \cos R \mathbf{1}$.
                # $\mathbf{x} = G^{-1} \cos R \mathbf{1}$.
                # $m = P \mathbf{x}$.
                # $|m|^2 = \mathbf{x}^T G \mathbf{x} = \cos R \mathbf{1}^T G^{-1} G G^{-1} \mathbf{1} \cos R = \cos^2 R \mathbf{1}^T G^{-1} \mathbf{1}$.
                # So $\cos R = \frac{1}{\sqrt{\mathbf{1}^T G^{-1} \mathbf{1}}}$.
                
                G = np.array([
                    [a @ a, a @ b, a @ c],
                    [b @ a, b @ b, b @ c],
                    [c @ a, c @ b, c @ c]
                ])
                # G is 3x3.
                try:
                    G_inv = np.linalg.inv(G)
                    ones = np.ones(3)
                    val = np.dot(ones, G_inv.dot(ones))
                    if val <= 0:
                        continue
                    cos_R = 1.0 / np.sqrt(val)
                    # Clip for numerical stability
                    cos_R = np.clip(cos_R, -1.0, 1.0)
                    R = np.arccos(cos_R)
                    if R > max_r:
                        max_r = R
                except np.linalg.LinAlgError:
                    pass
            return np.degrees(max_r)
        except Exception:
            return np.inf

    # Initial radius
    best_radius = compute_covering_radius(points)
    
    # Optimization Loop
    step_size = 0.01
    for iteration in range(1000):
        if time.time() > deadline:
            break
            
        # Compute convex hull
        try:
            hull = ConvexHull(points)
            facets = hull.simplices
        except Exception:
            break
            
        # Compute circumradii and identify worst facets
        # We want to move points to reduce the max radius.
        # Gradient approximation: For each point, accumulate a "force" from adjacent facets.
        # Force direction: from point to circumcenter of the facet.
        # Magnitude: proportional to how much that facet's radius exceeds the average or max.
        
        forces = np.zeros_like(points)
        radii = np.zeros(len(facets))
        
        for idx, f in enumerate(facets):
            a, b, c = points[f[0]], points[f[1]], points[f[2]]
            G = np.array([
                [a @ a, a @ b, a @ c],
                [b @ a, b @ b, b @ c],
                [c @ a, c @ b, c @ c]
            ])
            try:
                G_inv = np.linalg.inv(G)
                ones = np.ones(3)
                val = np.dot(ones, G_inv.dot(ones))
                if val <= 0:
                    radii[idx] = 0.0
                    continue
                cos_R = 1.0 / np.sqrt(val)
                cos_R = np.clip(cos_R, -1.0, 1.0)
                R = np.arccos(cos_R)
                radii[idx] = R
                
                # Compute circumcenter direction
                # m = P @ (G_inv @ ones * cos_R)
                x = G_inv.dot(ones) * cos_R
                m = a * x[0] + b * x[1] + c * x[2]
                m_norm = np.linalg.norm(m)
                if m_norm < 1e-10:
                    continue
                m /= m_norm
                
                # Apply force to vertices a, b, c
                # Move towards m to reduce R?
                # If we move a towards m, the distance a-m decreases.
                # But it might increase distance to other facets.
                # Weight by R to focus on large facets.
                weight = R ** 2 # Emphasize large radii
                
                forces[f[0]] += (m - a) * weight
                forces[f[1]] += (m - b) * weight
                forces[f[2]] += (m - c) * weight
                
            except np.linalg.LinAlgError:
                continue
        
        # Normalize forces to be tangent to sphere
        for i in range(N):
            p = points[i]
            f = forces[i]
            # Project f onto tangent space: f - (f . p) p
            f_tangent = f - np.dot(f, p) * p
            forces[i] = f_tangent
            
        # Update points
        points += step_size * forces
        # Normalize
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Check improvement
        current_radius = compute_covering_radius(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
        else:
            # Revert if worse, or reduce step size
            points = best_points.copy()
            step_size *= 0.95
            if step_size < 1e-6:
                break

    return best_points
