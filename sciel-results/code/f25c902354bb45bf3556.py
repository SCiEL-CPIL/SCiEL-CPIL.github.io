import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit

    def compute_covering_radius_and_grad(pts):
        """
        Computes the exact covering radius and the gradient of the max radius.
        Returns: (max_rad, gradient_array)
        """
        try:
            hull = ConvexHull(pts)
            max_rad = 0.0
            active_facets = []
            
            # First pass: find max radius and identify active facets
            for simplex in hull.simplices:
                v1, v2, v3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
                
                normal = np.cross(v2 - v1, v3 - v1)
                norm = np.linalg.norm(normal)
                if norm < 1e-10:
                    continue
                c_s = normal / norm
                
                # Check if circumcenter is inside the spherical triangle
                # If the triangle is acute, circumcenter is inside.
                # If obtuse, the max distance is to the longest edge midpoint.
                
                d1 = np.arccos(np.clip(np.dot(c_s, v1), -1.0, 1.0))
                d2 = np.arccos(np.clip(np.dot(c_s, v2), -1.0, 1.0))
                d3 = np.arccos(np.clip(np.dot(c_s, v3), -1.0, 1.0))
                max_d_circ = max(d1, d2, d3)
                
                # Edge midpoints
                e12 = v1 + v2
                e12_norm = np.linalg.norm(e12)
                if e12_norm > 1e-10:
                    m12 = e12 / e12_norm
                    d_e12 = np.arccos(np.clip(np.dot(m12, v1), -1.0, 1.0))
                else:
                    d_e12 = 0.0
                    
                e23 = v2 + v3
                e23_norm = np.linalg.norm(e23)
                if e23_norm > 1e-10:
                    m23 = e23 / e23_norm
                    d_e23 = np.arccos(np.clip(np.dot(m23, v2), -1.0, 1.0))
                else:
                    d_e23 = 0.0

                e31 = v3 + v1
                e31_norm = np.linalg.norm(e31)
                if e31_norm > 1e-10:
                    m31 = e31 / e31_norm
                    d_e31 = np.arccos(np.clip(np.dot(m31, v3), -1.0, 1.0))
                else:
                    d_e31 = 0.0

                max_d_edge = max(d_e12, d_e23, d_e31)

                if max_d_circ <= np.pi / 2.0:
                    local_rad = max_d_circ
                    center = c_s
                    # Gradient contribution for circumcenter case
                    # We want to move vertices towards the center to reduce the radius.
                    # The "gradient" of the radius w.r.t vertex positions on the sphere
                    # is roughly the direction from the vertex to the center projected onto the tangent space.
                    # Actually, if we move v_i towards c_s, the angle decreases.
                    # Let's define the force as (c_s - v_i) projected to tangent.
                    g1 = c_s - v1
                    g2 = c_s - v2
                    g3 = c_s - v3
                else:
                    local_rad = max_d_edge
                    if d_e12 == max_d_edge:
                        center = m12
                        g1 = m12 - v1
                        g2 = m12 - v2
                        g3 = np.zeros(3)
                    elif d_e23 == max_d_edge:
                        center = m23
                        g1 = np.zeros(3)
                        g2 = m23 - v2
                        g3 = m23 - v3
                    else:
                        center = m31
                        g1 = m31 - v1
                        g2 = np.zeros(3)
                        g3 = m31 - v3

                if local_rad > max_rad:
                    max_rad = local_rad
                    active_facets = [(simplex, local_rad, g1, g2, g3, center)]
                elif abs(local_rad - max_rad) < 1e-4:
                    # Include near-max facets to aggregate gradient
                    active_facets.append((simplex, local_rad, g1, g2, g3, center))

            if not active_facets:
                return 0.0, np.zeros((N, 3))

            # Aggregate gradients
            grad = np.zeros((N, 3))
            for simplex, rad, g1, g2, g3, center in active_facets:
                # Weight by how close to max? Or just sum?
                # Summing helps smooth out the landscape.
                grad[simplex[0]] += g1
                grad[simplex[1]] += g2
                grad[simplex[2]] += g3
            
            # Project gradients onto tangent planes
            for i in range(N):
                dot_prod = np.dot(grad[i], pts[i])
                grad[i] -= dot_prod * pts[i]
                
            return max_rad, grad

        except Exception:
            return np.inf, np.zeros((N, 3))

    # Initialization: Fibonacci Sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_score = np.inf

    # Optimization Loop with Multiple Restarts
    num_restarts = 5
    steps_per_restart = 300
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit - 5.0:
            break
            
        if restart == 0:
            x = points.copy()
        else:
            # Perturbed Fibonacci or Random
            if restart % 2 == 0:
                x = points.copy()
                x += np.random.normal(0, 0.1, x.shape)
            else:
                x = np.random.randn(N, 3)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        
        current_rad, grad = compute_covering_radius_and_grad(x)
        
        # Adaptive learning rate
        lr = 0.1
        patience = 0
        best_local_rad = current_rad
        best_local_x = x.copy()
        
        for step in range(steps_per_restart):
            if time.time() - start_time > time_limit - 5.0:
                break
                
            # Line search / Step adjustment
            # Try a step in the direction of -grad (to reduce radius)
            # Note: grad points towards the center of the worst facet.
            # Moving vertices towards the center reduces the local radius.
            # So we move in direction of grad?
            # Let's check: if we move v towards c, angle(v, c) decreases.
            # Yes, moving in direction of (c - v) reduces distance to c.
            # So update: x_new = x + lr * grad_projected
            
            x_new = x + lr * grad
            
            # Project to sphere
            x_new /= np.linalg.norm(x_new, axis=1, keepdims=True)
            
            new_rad, new_grad = compute_covering_radius_and_grad(x_new)
            
            if new_rad < current_rad:
                x = x_new
                current_rad = new_rad
                grad = new_grad
                patience = 0
                if current_rad < best_local_rad:
                    best_local_rad = current_rad
                    best_local_x = x.copy()
            else:
                patience += 1
                lr *= 0.8
                if patience > 10:
                    break
        
        if best_local_rad < best_score:
            best_score = best_local_rad
            best_points = best_local_x.copy()

    # Final Polish: Very small steps on the best configuration
    if time.time() - start_time < time_limit - 2.0:
        x = best_points.copy()
        current_rad, grad = compute_covering_radius_and_grad(x)
        lr = 0.01
        for step in range(200):
            if time.time() - start_time > time_limit - 2.0:
                break
            x_new = x + lr * grad
            x_new /= np.linalg.norm(x_new, axis=1, keepdims=True)
            new_rad, new_grad = compute_covering_radius_and_grad(x_new)
            if new_rad < current_rad:
                x = x_new
                current_rad = new_rad
                grad = new_grad
                if current_rad < best_score:
                    best_score = current_rad
                    best_points = x.copy()
            else:
                lr *= 0.9
                if lr < 1e-6:
                    break

    return best_points

# EVOLVE-BLOCK-END
