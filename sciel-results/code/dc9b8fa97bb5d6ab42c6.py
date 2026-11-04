import numpy as np
import time

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
    pi = pts[tri_indices[:, 0]]  # (165, 2)
    pj = pts[tri_indices[:, 1]]  # (165, 2)
    pk = pts[tri_indices[:, 2]]  # (165, 2)
    
    v1 = pj - pi  # (165, 2)
    v2 = pk - pi  # (165, 2)
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]  # (165,)
    
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def project_points_to_triangle(pts):
    """
    Project each point to the triangle if outside.
    Uses barycentric coordinates.
    """
    v0 = B - A  # (2,)
    v1 = C - A  # (2,)
    
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    
    denom = d00 * d11 - d01 * d01
    
    v2 = pts - A  # (N, 2)
    
    d02 = np.sum(v0 * v2, axis=1)  # (N,)
    d12 = np.sum(v1 * v2, axis=1)  # (N,)
    
    v = (d11 * d02 - d01 * d12) / denom  # (N,)
    w = (d00 * d12 - d01 * d02) / denom  # (N,)
    u = 1 - v - w  # (N,)
    
    # Check if inside: all barycentric coords >= 0
    inside = (u >= -1e-10) & (v >= -1e-10) & (w >= -1e-10)
    
    projected = pts.copy()
    
    for i in range(N):
        if not inside[i]:
            p = pts[i]
            
            def dist_to_segment(p, a, b):
                ab = b - a
                ap = p - a
                t = np.dot(ap, ab) / np.dot(ab, ab)
                t = max(0.0, min(1.0, t))
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
    Uses Simulated Annealing with multiple restarts and local polishing.
    """
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = -np.inf
    
    # Parameters for Simulated Annealing
    T0 = 0.01  # Initial temperature
    Tend = 1e-6 # Final temperature
    alpha = 0.995 # Cooling rate
    n_restarts = 10
    
    # Time limit management
    start_time = time.time()
    max_time = 110.0 # Leave 10s buffer
    
    for restart in range(n_restarts):
        if time.time() - start_time > max_time:
            break
            
        # Initialize with random points
        current_pts = np.zeros((N, 2))
        for i in range(N):
            current_pts[i] = random_point_in_triangle(rng)
        
        current_score = min_area_vectorized(current_pts)
        
        # Simulated Annealing Loop
        T = T0
        while T > Tend:
            if time.time() - start_time > max_time:
                break
            
            # Generate a new candidate by perturbing 1 or 2 random points
            new_pts = current_pts.copy()
            n_perturb = rng.integers(1, 3) # 1 or 2 points
            idxs = rng.choice(N, size=n_perturb, replace=False)
            
            # Perturbation size scales with temperature
            step_size = 0.05 * (T / T0) ** 0.5
            for idx in idxs:
                noise = rng.normal(0, step_size, size=2)
                new_pts[idx] += noise
            
            # Project to triangle
            new_pts = project_points_to_triangle(new_pts)
            
            new_score = min_area_vectorized(new_pts)
            
            # Acceptance criterion
            delta = new_score - current_score
            if delta > 0 or rng.random() < np.exp(delta / T):
                current_pts = new_pts
                current_score = new_score
            
            T *= alpha
        
        # Local Polishing (Hill Climbing)
        # Try small random steps that only accept improvements
        polish_steps = 200
        step_size = 0.005
        for _ in range(polish_steps):
            if time.time() - start_time > max_time:
                break
            new_pts = current_pts.copy()
            idx = rng.integers(0, N)
            noise = rng.normal(0, step_size, size=2)
            new_pts[idx] += noise
            new_pts = project_points_to_triangle(new_pts)
            new_score = min_area_vectorized(new_pts)
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
            else:
                # Shrink step size if no improvement
                step_size *= 0.95
        
        if current_score > best_score:
            best_score = current_score
            best_points = current_pts.copy()
            
        # If we found a very good solution, we can stop early
        if best_score > 0.016:
            break

    return best_points
