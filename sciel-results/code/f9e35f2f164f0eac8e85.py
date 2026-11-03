"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
import time

def construct():
    """
    Constructs a configuration of 50 points on the unit sphere minimizing the covering radius.
    Uses Spherical Lloyd's Algorithm (Centroidal Voronoi Tessellation) starting from a Fibonacci spiral.
    """
    # Deadline guard: 120s limit, aim for 110s
    start_time = time.time()
    deadline = start_time + 110.0

    # 1. Initialization: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    P = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize to be safe
    P = P / np.linalg.norm(P, axis=1, keepdims=True)

    best_P = P.copy()
    best_radius = np.inf

    def compute_covering_radius(points):
        # Approximate covering radius by checking distances to a dense grid of test points
        # Or better, use the Voronoi vertices (circumcenters) which are the critical points
        # The covering radius is the max over Voronoi vertices of the distance to the nearest site.
        # For N=50, we can compute the Delaunay triangulation and check circumcenters.
        try:
            tri = Delaunay(points)
            # Get circumcenters for all triangles
            # The circumcenter of a triangle (a,b,c) on the sphere is normalize((a-b) x (a-c))
            # We need to ensure we pick the correct sign (the one in the Voronoi cell).
            # The Voronoi vertex is the point on the sphere equidistant to the 3 sites.
            # It lies in the direction of the normal to the plane of the triangle.
            
            # Efficiently compute normals for all triangles
            # tri.simplices is (M, 3)
            simplices = tri.simplices
            p_i = points[simplices[:, 0]]
            p_j = points[simplices[:, 1]]
            p_k = points[simplices[:, 2]]
            
            # Normal vectors
            v1 = p_i - p_j
            v2 = p_i - p_k
            normals = np.cross(v1, v2)
            
            # Normalize normals to get points on the sphere
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            # Handle zero norms (degenerate triangles)
            norms[norms == 0] = 1.0
            circumcenters = normals / norms
            
            # We need to select the correct circumcenter.
            # The Voronoi vertex must be "inside" the spherical triangle.
            # This is true if the dot product of the circumcenter with the centroid of the triangle is positive.
            centroids = (p_i + p_j + p_k) / 3.0
            dots = np.sum(circumcenters * centroids, axis=1)
            # Flip signs where dot is negative
            circumcenters[dots < 0] *= -1
            
            # Now, for each circumcenter, find the distance to the nearest point in P
            # We can do this by checking distances to the 3 defining points, as they are the nearest for that cell's vertex
            # dist = arccos(circumcenter . p_i)
            # Since circumcenter is equidistant to p_i, p_j, p_k, we just need one dot product.
            dists = np.arccos(np.clip(np.sum(circumcenters * p_i, axis=1), -1.0, 1.0))
            
            # The covering radius is the max of these distances
            return np.max(dists) * 180.0 / np.pi
        except Exception:
            return np.inf

    # Initial radius
    current_radius = compute_covering_radius(P)
    if current_radius < best_radius:
        best_radius = current_radius
        best_P = P.copy()

    # Lloyd's iterations
    max_iter = 500
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        try:
            tri = Delaunay(P)
            simplices = tri.simplices
            
            # For each point, accumulate the sum of adjacent circumcenters
            # We need a mapping from point index to list of triangle indices
            # Or simply iterate over triangles and add to the 3 vertices
            new_P = np.zeros_like(P)
            counts = np.zeros(N)
            
            p_i = P[simplices[:, 0]]
            p_j = P[simplices[:, 1]]
            p_k = P[simplices[:, 2]]
            
            v1 = p_i - p_j
            v2 = p_i - p_k
            normals = np.cross(v1, v2)
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            circumcenters = normals / norms
            
            centroids = (p_i + p_j + p_k) / 3.0
            dots = np.sum(circumcenters * centroids, axis=1)
            circumcenters[dots < 0] *= -1
            
            # Accumulate
            idx_i = simplices[:, 0]
            idx_j = simplices[:, 1]
            idx_k = simplices[:, 2]
            
            np.add.at(new_P, idx_i, circumcenters)
            np.add.at(new_P, idx_j, circumcenters)
            np.add.at(new_P, idx_k, circumcenters)
            
            np.add.at(counts, idx_i, 1)
            np.add.at(counts, idx_j, 1)
            np.add.at(counts, idx_k, 1)
            
            # Average and normalize
            # Avoid division by zero
            counts[counts == 0] = 1.0
            new_P = new_P / counts[:, np.newaxis]
            new_P = new_P / np.linalg.norm(new_P, axis=1, keepdims=True)
            
            # Check convergence
            diff = np.linalg.norm(new_P - P)
            P = new_P
            
            # Evaluate radius
            current_radius = compute_covering_radius(P)
            if current_radius < best_radius:
                best_radius = current_radius
                best_P = P.copy()
                
            # If diff is very small, we might be converged
            if diff < 1e-6:
                # Try to escape local minima with a small perturbation
                # Only if we have time
                if time.time() < deadline - 5.0:
                    P = P + 0.01 * np.random.randn(N, 3)
                    P = P / np.linalg.norm(P, axis=1, keepdims=True)
                else:
                    break
                    
        except Exception:
            break

    return best_P
# EVOLVE-BLOCK-END
