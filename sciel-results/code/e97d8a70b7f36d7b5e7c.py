# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    N = 50
    
    def compute_score(centers):
        """Compute the packing radius."""
        # Boundary constraint
        dists_origin = np.linalg.norm(centers, axis=1)
        if np.any(dists_origin > 1.0):
            return -1e6
        min_boundary = 1.0 - np.max(dists_origin)
        
        # Pairwise distances
        # Vectorized distance calculation
        # diff shape (N, N, 2)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        return min(min_boundary, min_pair)

    def get_hexagonal_initial(spacing):
        """Generate hexagonal lattice points and select N closest to origin."""
        # Generate a grid of hexagonal points
        # Hexagonal spacing: dx = spacing, dy = spacing * sqrt(3)/2
        # Offset odd rows by spacing/2
        k = int(1.0 / spacing) + 3
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = i * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if j % 2 != 0:
                    x += spacing / 2.0
                points.append([x, y])
        points = np.array(points)
        
        # Filter points inside unit circle (with margin for radius)
        # We want to pick points that can fit a circle of radius r.
        # Initial r estimate is spacing/2.
        # Let's just pick points inside 1.0 and let optimization handle the radius.
        # But we need them to be non-overlapping initially.
        # If we pick from lattice, they are non-overlapping if spacing >= 2*r.
        
        dists_origin = np.linalg.norm(points, axis=1)
        # Sort by distance to origin
        idx = np.argsort(dists_origin)
        points = points[idx]
        
        # Take the first N points
        # Ensure they are within the unit circle
        # The furthest one might be > 1.0, but we can push it in during optimization.
        # However, for a valid initial state, let's try to keep them inside.
        # If the N-th point is outside, we might need a smaller spacing.
        # But for now, return the N closest.
        return points[:N].copy()

    best_centers = None
    best_score = -np.inf
    
    # Try a few different initial spacings
    base_spacing = 2.0 / (np.sqrt(N) + 1.0)
    
    for trial in range(20):
        if time.time() > deadline:
            break
            
        # Vary the spacing slightly
        spacing = base_spacing * (1.0 + 0.1 * np.random.randn())
        if spacing < 0.1:
            spacing = 0.1
            
        centers = get_hexagonal_initial(spacing)
        
        # Add small random jitter to break symmetry
        centers += np.random.randn(N, 2) * 0.01
        
        current_score = compute_score(centers)
        
        # Local search
        step_size = 0.05
        no_improve_count = 0
        max_no_improve = 2000
        
        while time.time() < deadline - 5.0 and no_improve_count < max_no_improve:
            # Pick a random circle
            i = np.random.randint(N)
            
            # Random direction and magnitude
            angle = np.random.uniform(0, 2 * np.pi)
            mag = np.random.uniform(0, step_size)
            dx = mag * np.cos(angle)
            dy = mag * np.sin(angle)
            
            new_centers = centers.copy()
            new_centers[i, 0] += dx
            new_centers[i, 1] += dy
            
            new_score = compute_score(new_centers)
            
            if new_score > current_score:
                centers = new_centers
                current_score = new_score
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            # Occasionally decrease step size
            if no_improve_count > 100:
                step_size *= 0.9
                no_improve_count = 0
                
        if current_score > best_score:
            best_score = current_score
            best_centers = centers.copy()
            
        # Print progress (optional, but good for debugging if visible)
        # print(f"Trial {trial}: Score {best_score:.6f}")

    if best_centers is None:
        # Fallback
        return get_hexagonal_initial(base_spacing)
        
    return best_centers
# EVOLVE-BLOCK-END
