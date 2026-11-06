"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    def get_delaunay_and_radii(points):
        try:
            tri = Delaunay(points)
            faces = tri.simplices
        except Exception:
            return None, None, None
            
        radii = np.zeros(len(faces))
        circumcenters = np.zeros((len(faces), 3))
        
        for i, face in enumerate(faces):
            a = points[face[0]]
            b = points[face[1]]
            c = points[face[2]]
            
            # Circumcenter
            m = np.cross(a - b, b - c)
            norm_m = np.linalg.norm(m)
            if norm_m < 1e-10:
                radii[i] = np.pi
                continue
            m /= norm_m
            
            # Orientation
            centroid = (a + b + c) / 3.0
            if np.dot(m, centroid) < 0:
                m = -m
                
            circumcenters[i] = m
            dist = np.arccos(np.clip(np.dot(a, m), -1.0, 1.0))
            radii[i] = dist
            
        return faces, radii, circumcenters

    def lloyd_step(points, faces, circumcenters):
        # Map point index to list of face indices
        point_faces = [[] for _ in range(N)]
        for i, face in enumerate(faces):
            for idx in face:
                point_faces[idx].append(i)
        
        new_points = np.zeros_like(points)
        for i in range(N):
            if not point_faces[i]:
                new_points[i] = points[i]
                continue
            
            sum_vec = np.zeros(3)
            for f_idx in point_faces[i]:
                sum_vec += circumcenters[f_idx]
            
            norm_sum = np.linalg.norm(sum_vec)
            if norm_sum < 1e-10:
                new_points[i] = points[i]
            else:
                new_points[i] = sum_vec / norm_sum
                
        return new_points

    best_points = None
    best_radius = float('inf')
    
    # Multiple restarts
    max_restarts = 15
    for restart in range(max_restarts):
        if time.time() > deadline - 15.0:
            break
            
        # Initialize with Fibonacci spiral + noise
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        # Random perturbation
        noise_scale = 0.05
        points += np.random.normal(0, noise_scale, (N, 3))
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Initial radius
        faces, radii, circumcenters = get_delaunay_and_radii(points)
        if faces is None:
            continue
            
        current_radius = np.max(radii)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
            
        # Lloyd Iterations
        for iter in range(300):
            if time.time() > deadline - 5.0:
                break
                
            new_points = lloyd_step(points, faces, circumcenters)
            
            # Try full step
            new_faces, new_radii, new_cc = get_delaunay_and_radii(new_points)
            if new_faces is None:
                break
                
            new_radius = np.max(new_radii)
            
            if new_radius < current_radius:
                points = new_points
                current_radius = new_radius
                faces = new_faces
                radii = new_radii
                circumcenters = new_cc
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_points = points.copy()
            else:
                # Try half step
                mid_points = (points + new_points) / 2.0
                mid_points /= np.linalg.norm(mid_points, axis=1, keepdims=True)
                mid_faces, mid_radii, mid_cc = get_delaunay_and_radii(mid_points)
                if mid_faces is None:
                    break
                mid_radius = np.max(mid_radii)
                
                if mid_radius < current_radius:
                    points = mid_points
                    current_radius = mid_radius
                    faces = mid_faces
                    radii = mid_radii
                    circumcenters = mid_cc
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_points = points.copy()
                else:
                    # Stagnation, break
                    break

    return best_points
# EVOLVE-BLOCK-END
