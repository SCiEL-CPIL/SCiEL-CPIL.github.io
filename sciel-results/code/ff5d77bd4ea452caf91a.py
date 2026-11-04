import numpy as np
from scipy.spatial import ConvexHull

N = 130

def construct():
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    P = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Optimization parameters
    iterations = 1000
    step_size = 0.1
    momentum = 0.5
    velocity = np.zeros_like(P)
    
    for it in range(iterations):
        # Compute Convex Hull
        try:
            hull = ConvexHull(P)
            faces = hull.simplices
        except Exception:
            # If hull fails (e.g., points coplanar or inside hemisphere), add small noise
            P += np.random.normal(0, 0.01, P.shape)
            P /= np.linalg.norm(P, axis=1, keepdims=True)
            continue
        
        # Map each point to its incident triangles
        point_to_triangles = [[] for _ in range(N)]
        for t_idx, face in enumerate(faces):
            for p_idx in face:
                point_to_triangles[p_idx].append(t_idx)
        
        # Precompute circumcenters and radii for all faces
        circumcenters = np.zeros((len(faces), 3))
        radii = np.zeros(len(faces))
        
        for f_idx, face in enumerate(faces):
            p1, p2, p3 = P[face[0]], P[face[1]], P[face[2]]
            
            # Compute angles
            def ang_dist(u, v):
                d = np.clip(np.dot(u, v), -1.0, 1.0)
                return np.arccos(d)
                
            a = ang_dist(p2, p3)
            b = ang_dist(p1, p3)
            c = ang_dist(p1, p2)
            
            sin_b = np.sin(b)
            sin_c = np.sin(c)
            sin_a = np.sin(a)
            
            if sin_b * sin_c < 1e-9 or sin_a * sin_c < 1e-9 or sin_a * sin_b < 1e-9:
                circumcenters[f_idx] = p1
                radii[f_idx] = 0.0
                continue
                
            cos_a = np.cos(a)
            cos_b = np.cos(b)
            cos_c = np.cos(c)
            
            cos_alpha = (cos_a - cos_b * cos_c) / (sin_b * sin_c)
            cos_beta = (cos_b - cos_a * cos_c) / (sin_a * sin_c)
            cos_gamma = (cos_c - cos_a * cos_b) / (sin_a * sin_b)
            
            cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
            cos_beta = np.clip(cos_beta, -1.0, 1.0)
            cos_gamma = np.clip(cos_gamma, -1.0, 1.0)
            
            alpha = np.arccos(cos_alpha)
            beta = np.arccos(cos_beta)
            gamma = np.arccos(cos_gamma)
            
            s2a = np.sin(2 * alpha)
            s2b = np.sin(2 * beta)
            s2g = np.sin(2 * gamma)
            
            C_vec = p1 * s2a + p2 * s2b + p3 * s2g
            C_norm = np.linalg.norm(C_vec)
            
            if C_norm < 1e-9:
                circumcenters[f_idx] = p1
                radii[f_idx] = 0.0
                continue
                
            C = C_vec / C_norm
            circumcenters[f_idx] = C
            
            # Radius is distance from C to p1
            dot_cp1 = np.clip(np.dot(C, p1), -1.0, 1.0)
            R = np.arccos(dot_cp1)
            radii[f_idx] = R
            
        # For each point, find the triangle with max radius
        targets = np.zeros_like(P)
        for p_idx in range(N):
            max_r = -1
            best_c = P[p_idx] # Default to itself
            for t_idx in point_to_triangles[p_idx]:
                if radii[t_idx] > max_r:
                    max_r = radii[t_idx]
                    best_c = circumcenters[t_idx]
            targets[p_idx] = best_c
            
        # Update positions using momentum
        forces = targets - P
        norms = np.linalg.norm(forces, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        forces = forces / norms
        
        velocity = momentum * velocity + (1 - momentum) * forces
        P = P + step_size * velocity
        
        # Normalize
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        
        # Decay step size
        step_size *= 0.995
        
    return P
