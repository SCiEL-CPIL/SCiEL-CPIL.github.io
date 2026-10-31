import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 100.0  # Safety margin
    
    # Minimum distance between centers of non-overlapping unit hexagons
    min_dist = np.sqrt(3)
    
    # Initialize positions
    # Start with a 7-point hex cluster and 4 random points
    pos = np.zeros((N, 2))
    # 7-point cluster
    # Center
    pos[0] = [0.0, 0.0]
    # 6 neighbors
    for i in range(1, 7):
        angle = i * np.pi / 3
        pos[i] = [min_dist * np.cos(angle), min_dist * np.sin(angle)]
    # 4 additional points, placed roughly in the second ring
    for i in range(7, N):
        angle = (i - 6) * np.pi / 3 + np.pi / 6
        pos[i] = [2.0 * min_dist * np.cos(angle), 2.0 * min_dist * np.sin(angle)]
        
    # Optimization loop
    best_S = float('inf')
    best_pos = pos.copy()
    
    # Parameters
    learning_rate = 0.05
    overlap_penalty = 1.0
    
    while time.time() - start_time < deadline:
        # Calculate bounding hexagon side length L for centers
        # S = L + 1
        # L = max( max|x|, 2/sqrt(3)*max|y|, 1/sqrt(3)*max|x+sqrt(3)y|, 1/sqrt(3)*max|x-sqrt(3)y| )
        
        x = pos[:, 0]
        y = pos[:, 1]
        
        L1 = np.max(np.abs(x))
        L2 = (2.0 / np.sqrt(3)) * np.max(np.abs(y))
        L3 = (1.0 / np.sqrt(3)) * np.max(np.abs(x + np.sqrt(3) * y))
        L4 = (1.0 / np.sqrt(3)) * np.max(np.abs(x - np.sqrt(3) * y))
        
        L = max(L1, L2, L3, L4)
        S = L + 1.0
        
        if S < best_S:
            best_S = S
            best_pos = pos.copy()
            
        # Check for overlaps
        # Calculate pairwise distances
        # dist[i, j] = ||pos[i] - pos[j]||
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        # Find overlapping pairs
        # We want to push them apart if dist < min_dist
        # Force calculation
        force = np.zeros_like(pos)
        
        for i in range(N):
            for j in range(i + 1, N):
                if dist[i, j] < min_dist and dist[i, j] > 1e-6:
                    # Push apart
                    # Vector from i to j
                    vec = pos[j] - pos[i]
                    # Normalize
                    vec_norm = vec / dist[i, j]
                    # Magnitude of push
                    push = (min_dist - dist[i, j]) * overlap_penalty
                    force[i] -= vec_norm * push
                    force[j] += vec_norm * push
                elif dist[i, j] < 1e-6:
                    # If exactly same, push randomly
                    force[i] += np.random.normal(size=2) * 0.1
                    force[j] -= np.random.normal(size=2) * 0.1
        
        # Apply force
        pos += learning_rate * force
        
        # Scale down to reduce S
        # Calculate new S after force application? 
        # Simpler: Just scale the whole configuration slightly
        # If we scale by k, L becomes k*L.
        # We want to reduce L.
        # But scaling might cause overlaps.
        # Let's try scaling by 0.99
        pos *= 0.995
        
        # Add small random perturbation to escape local minima
        pos += np.random.normal(0, 0.01, size=pos.shape)
        
    # Final calculation of S for best_pos
    x = best_pos[:, 0]
    y = best_pos[:, 1]
    
    L1 = np.max(np.abs(x))
    L2 = (2.0 / np.sqrt(3)) * np.max(np.abs(y))
    L3 = (1.0 / np.sqrt(3)) * np.max(np.abs(x + np.sqrt(3) * y))
    L4 = (1.0 / np.sqrt(3)) * np.max(np.abs(x - np.sqrt(3) * y))
    
    L = max(L1, L2, L3, L4)
    S = L + 1.0
    
    # Construct inner list
    # Format: (x, y, angle_degrees)
    # All angles 0.0
    inner = [[float(p[0]), float(p[1]), 0.0] for p in best_pos]
    
    # Outer hexagon center
    # The bounding hexagon is centered at (0,0) in our coordinate system
    # So the outer hexagon center is (0,0)
    center = (0.0, 0.0)
    side = float(S)
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
