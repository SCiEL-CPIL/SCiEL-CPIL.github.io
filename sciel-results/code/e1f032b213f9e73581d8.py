# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min / d_max)^2.
    Uses a multi-start local optimization with random initializations.
    """
    n = 14
    d = 3
    deadline = time.time() + 600.0 - 30.0  # safety margin
    
    best_score = -1.0
    best_points = None
    
    # Generate a set of candidate initial configurations
    # We'll use:
    # 1. Random Gaussian
    # 2. Points on a sphere (Fibonacci sphere)
    # 3. Subset of dodecahedron vertices
    # 4. Subset of icosahedron vertices (only 12, so add 2 more)
    # 5. Random points on sphere with various seeds
    
    candidates = []
    
    # 1. Random Gaussian with different seeds
    for seed in range(20):
        rng = np.random.RandomState(seed)
        pts = rng.randn(n, d)
        candidates.append(pts)
    
    # 2. Fibonacci sphere
    def fibonacci_sphere(n_points):
        points = []
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(n_points):
            y = 1 - (i / (n_points - 1)) * 2
            radius = np.sqrt(1 - y**2)
            theta = golden_angle * i
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)
    
    fib_pts = fibonacci_sphere(n)
    candidates.append(fib_pts)
    candidates.append(fib_pts * 0.5)  # scaled version
    
    # 3. Dodecahedron vertices (20 vertices, pick 14)
    phi = (1 + np.sqrt(5)) / 2.0
    dodecahedron_vertices = []
    # (±1, ±1, ±1)
    for a in [-1, 1]:
        for b in [-1, 1]:
            for c in [-1, 1]:
                dodecahedron_vertices.append([a, b, c])
    # (0, ±1/phi, ±phi)
    for a in [-1/phi, 1/phi]:
        for b in [-phi, phi]:
            dodecahedron_vertices.append([0, a, b])
    # (±phi, 0, ±1/phi)
    for a in [-phi, phi]:
        for b in [-1/phi, 1/phi]:
            dodecahedron_vertices.append([a, 0, b])
    # (±1/phi, ±phi, 0)
    for a in [-1/phi, 1/phi]:
        for b in [-phi, phi]:
            dodecahedron_vertices.append([a, b, 0])
    
    dodecahedron_vertices = np.array(dodecahedron_vertices)
    # Normalize to unit sphere
    norms = np.linalg.norm(dodecahedron_vertices, axis=1, keepdims=True)
    dodecahedron_vertices = dodecahedron_vertices / norms
    
    # Pick 14 well-spread vertices from dodecahedron
    # Use a greedy approach: start with random vertex, then add farthest point iteratively
    def greedy_subset(vertices, k, seed=0):
        rng = np.random.RandomState(seed)
        n_verts = len(vertices)
        selected = [rng.randint(n_verts)]
        for _ in range(k - 1):
            min_dists = np.min(pdist(vertices[selected] if len(selected) > 1 else vertices[selected][None], axis=1), axis=0)
            # Actually, let's compute min distance from each unselected vertex to selected set
            selected_arr = vertices[selected]
            min_dists = np.full(n_verts, np.inf)
            for i in range(n_verts):
                if i not in selected:
                    dists = np.linalg.norm(vertices[i] - selected_arr, axis=1)
                    min_dists[i] = np.min(dists)
            # Pick the one with max min_dist
            unselected = [i for i in range(n_verts) if i not in selected]
            best_idx = max(unselected, key=lambda i: min_dists[i])
            selected.append(best_idx)
        return vertices[selected]
    
    for seed in range(5):
        subset = greedy_subset(dodecahedron_vertices, n, seed)
        candidates.append(subset)
    
    # 4. Random points on sphere
    for seed in range(20):
        rng = np.random.RandomState(seed + 1000)
        pts = rng.randn(n, d)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        candidates.append(pts)
    
    # 5. Icosahedron (12 vertices) + 2 more points
    phi = (1 + np.sqrt(5)) / 2.0
    ico_vertices = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    norms = np.linalg.norm(ico_vertices, axis=1, keepdims=True)
    ico_vertices = ico_vertices / norms
    
    # Add 2 points: use opposite of centroid or other strategic points
    # Let's add points at (0,0,0) perturbed, or better, use the 2 vertices from dodecahedron that are farthest
    for seed in range(5):
        rng = np.random.RandomState(seed + 2000)
        # Pick 2 additional points from dodecahedron that are far from icosahedron
        all_candidates = np.vstack([ico_vertices, dodecahedron_vertices])
        # Greedily pick 14 from all_candidates
        subset = greedy_subset(all_candidates, n, seed)
        candidates.append(subset)
    
    # Local optimization function
    def optimize_points(points, max_iter=5000, lr=0.01, deadline=None):
        points = points.copy()
        best_local_score = -1.0
        best_local_points = points.copy()
        
        for it in range(max_iter):
            if deadline is not None and time.time() > deadline:
                break
            
            # Compute all pairwise distances
            dists = pdist(points)
            d_min = np.min(dists)
            d_max = np.max(dists)
            
            if d_max < 1e-10:
                score = 0.0
            else:
                score = (d_min / d_max) ** 2
            
            if score > best_local_score:
                best_local_score = score
                best_local_points = points.copy()
            
            # Compute gradient of (d_min/d_max)^2
            # This is complex, so use a simple approach:
            # Move points to increase min distance and decrease max distance
            
            # For each point, compute gradient contribution
            # Simplified: use finite differences or analytical gradient
            
            # Analytical gradient approach:
            # Let f = (d_min/d_max)^2
            # We want to increase d_min and decrease d_max
            # For the pair with min distance, push them apart
            # For the pair with max distance, pull them together
            
            min_idx = np.argmin(dists)
            max_idx = np.argmax(dists)
            
            # Convert pair index to point indices
            n_pts = len(points)
            def pair_to_indices(pair_idx):
                i = 0
                while True:
                    j = i + 1
                    for j in range(i + 1, n_pts):
                        if pair_idx == 0:
                            return i, j
                        pair_idx -= 1
                    i += 1
            
            # More efficient conversion
            def pair_to_indices_fast(pair_idx, n):
                # Triangular number inverse
                # pair_idx = i*n - i*(i+1)/2 + (j - i - 1) where i < j
                # Approximate: i ≈ sqrt(2*pair_idx)
                i = int(np.floor((np.sqrt(8 * pair_idx + 1) - 1) / 2))
                # Check and adjust
                while True:
                    start = i * n - i * (i + 1) // 2
                    end = start + (n - i - 1)
                    if start <= pair_idx < end:
                        j = i + 1 + (pair_idx - start)
                        return i, j
                    i += 1
            
            min_i, min_j = pair_to_indices_fast(min_idx, n_pts)
            max_i, max_j = pair_to_indices_fast(max_idx, n_pts)
            
            # Gradient direction for min pair: push apart
            grad = np.zeros_like(points)
            dir_min = points[min_i] - points[min_j]
            norm_min = np.linalg.norm(dir_min)
            if norm_min > 1e-10:
                dir_min = dir_min / norm_min
                grad[min_i] += dir_min
                grad[min_j] -= dir_min
            
            # Gradient direction for max pair: pull together
            dir_max = points[max_i] - points[max_j]
            norm_max = np.linalg.norm(dir_max)
            if norm_max > 1e-10:
                dir_max = dir_max / norm_max
                grad[max_i] -= dir_max
                grad[max_j] += dir_max
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-10:
                grad = grad / grad_norm
            
            # Update
            points = points + lr * grad
            
            # Also try small random perturbations to escape local minima
            if it % 100 == 0:
                perturbation = np.random.randn(n_pts, d) * 0.01
                test_points = points + perturbation
                test_dists = pdist(test_points)
                test_d_min = np.min(test_dists)
                test_d_max = np.max(test_dists)
                if test_d_max > 1e-10:
                    test_score = (test_d_min / test_d_max) ** 2
                    if test_score > score:
                        points = test_points
        
        return best_local_points, best_local_score
    
    # Optimize each candidate
    for candidate in candidates:
        if time.time() > deadline:
            break
        
        optimized, score = optimize_points(candidate, max_iter=3000, lr=0.05, deadline=deadline)
        if score > best_score:
            best_score = score
            best_points = optimized
    
    # If we haven't found good points, return a reasonable default
    if best_points is None:
        # Return Fibonacci sphere
        return fibonacci_sphere(n)
    
    # Ensure finite values
    if not np.all(np.isfinite(best_points)):
        return fibonacci_sphere(n)
    
    return best_points

# EVOLVE-BLOCK-END
