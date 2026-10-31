import numpy as np
import time
import scipy.spatial as spat

N = 13

def calculate_score(points):
    """
    Calculate the minimum triangle area divided by the convex hull area.
    Returns 0 if points are collinear or hull area is 0.
    """
    if len(points) < 3:
        return 0.0
    
    # Compute convex hull area
    try:
        hull = spat.ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0

    pts = points
    n = len(pts)
    
    # Vectorized triangle area calculation
    # Indices for all unique triples (i, j, k) with i < j < k
    # N=13, C(13,3) = 286 triangles
    # Pre-computing indices is faster than generating them every time if we cache, 
    # but 286 is small. Let's generate once outside if possible, but for simplicity 
    # inside with numpy broadcasting is fast enough for 13 points.
    
    # Efficient way: 
    # Create arrays of indices
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    v1 = pts[j_flat] - pts[i_flat] 
    v2 = pts[k_flat] - pts[i_flat]
    
    # Determinant in 2D
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(det)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def find_bottleneck(points):
    """
    Returns the indices (i, j, k) of the triangle with the minimum area,
    and the area itself.
    """
    pts = points
    n = len(pts)
    
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    v1 = pts[j_flat] - pts[i_flat] 
    v2 = pts[k_flat] - pts[i_flat]
    
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(det)
    
    min_idx = np.argmin(areas)
    return i_flat[min_idx], j_flat[min_idx], k_flat[min_idx], areas[min_idx]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    best_score = -1.0
    best_points = None
    
    # Initialize with a good structured seed: 
    # A regular polygon is a strong starting point for max-min problems.
    # We can also use a "golden angle" spiral or grid, but regular polygon is symmetric.
    
    # Let's try multiple initial configurations to avoid local minima.
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    initial_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # 2. Perturbed regular 13-gon
    np.random.seed(42)
    noise = np.random.normal(0, 0.1, (N, 2))
    initial_pts2 = initial_pts + noise
    
    # 3. Fibonacci lattice (better spacing)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    fib_pts = np.zeros((N, 2))
    for i in range(N):
        r = np.sqrt((i + 0.5) / N)
        theta = golden_angle * i
        fib_pts[i] = [r * np.cos(theta), r * np.sin(theta)]
    # Normalize to unit area roughly? 
    # The score is invariant to scaling, so we just need to ensure they are spread out.
    # Let's scale to fit in a unit circle or square.
    fib_pts /= np.max(np.linalg.norm(fib_pts, axis=1))
    
    candidates = [initial_pts, initial_pts2, fib_pts]
    
    # Pre-compute indices for bottleneck finding to save time in loop
    # We'll just call find_bottleneck which does it. 
    # To optimize further, we could cache the index arrays, but N=13 is small.
    
    for init in candidates:
        if time.time() > deadline - 5.0:
            break
            
        current_pts = init.copy()
        current_score = calculate_score(current_pts)
        
        if current_score > best_score:
            best_score = current_score
            best_points = current_pts.copy()
        
        # Local Search Loop
        step_size = 0.05
        no_improve_count = 0
        max_no_improve = 100
        
        while time.time() < deadline - 10.0:
            if no_improve_count > max_no_improve:
                break
                
            # Identify the bottleneck triangle
            i, j, k, min_area = find_bottleneck(current_pts)
            
            # Strategy: Move the vertices of the bottleneck triangle apart.
            # To increase the area of triangle (i, j, k), we can move 
            # i away from the line segment jk, or move j and k apart.
            
            # Simple heuristic: Move each vertex in the direction away from the centroid of the other two.
            # This tends to expand the triangle.
            
            p_i = current_pts[i]
            p_j = current_pts[j]
            p_k = current_pts[k]
            
            # Centroid of the other two
            centroid_jk = (p_j + p_k) / 2.0
            centroid_ik = (p_i + p_k) / 2.0
            centroid_ij = (p_i + p_j) / 2.0
            
            # Direction vectors
            dir_i = p_i - centroid_jk
            dir_j = p_j - centroid_ik
            dir_k = p_k - centroid_ij
            
            # Normalize directions
            norm_i = np.linalg.norm(dir_i)
            norm_j = np.linalg.norm(dir_j)
            norm_k = np.linalg.norm(dir_k)
            
            if norm_i > 1e-6: dir_i /= norm_i
            if norm_j > 1e-6: dir_j /= norm_j
            if norm_k > 1e-6: dir_k /= norm_k
            
            # Try moving all three
            trial_pts = current_pts.copy()
            trial_pts[i] += step_size * dir_i
            trial_pts[j] += step_size * dir_j
            trial_pts[k] += step_size * dir_k
            
            trial_score = calculate_score(trial_pts)
            
            if trial_score > current_score:
                current_pts = trial_pts
                current_score = trial_score
                no_improve_count = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_pts.copy()
                    
                # Optionally reduce step size slightly as we converge? 
                # Or keep it constant to escape local minima? 
                # Let's keep it constant but maybe decrease if stuck.
            else:
                no_improve_count += 1
                # If stuck, try a smaller step or a different move (e.g., random perturbation)
                if no_improve_count % 10 == 0:
                    # Random perturbation to escape local minimum
                    np.random.seed(int(time.time() * 1000) % 10000)
                    perturb = np.random.normal(0, step_size / 2, (N, 2))
                    trial_pts = current_pts + perturb
                    trial_score = calculate_score(trial_pts)
                    if trial_score > current_score:
                        current_pts = trial_pts
                        current_score = trial_score
                        no_improve_count = 0
                        if current_score > best_score:
                            best_score = current_score
                            best_points = current_pts.copy()
                    else:
                        # Shrink step size
                        step_size *= 0.9
                        if step_size < 1e-4:
                            break

    if best_points is None:
        return np.random.rand(N, 2)
        
    return best_points
