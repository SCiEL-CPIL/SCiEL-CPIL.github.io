import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Initialize centers in a hexagonal lattice pattern
    centers = []
    centers.append([0.0, 0.0])
    for i in range(6):
        angle = i * np.pi / 3
        centers.append([2.0 * np.cos(angle), 2.0 * np.sin(angle)])
    # Add 4 more circles
    for i in range(4):
        angle = i * np.pi / 2
        centers.append([4.0 * np.cos(angle), 4.0 * np.sin(angle)])
    
    centers = np.array(centers, dtype=float)
    
    # Force-directed optimization
    repulsion_strength = 10.0
    attraction_strength = 0.5
    damping = 0.5
    max_iter = 2000
    tol = 1e-8
    
    for it in range(max_iter):
        forces = np.zeros_like(centers)
        
        # Repulsive forces between overlapping circles
        for i in range(N):
            for j in range(i + 1, N):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-6:
                    diff = np.random.randn(2) * 0.01
                    dist = np.linalg.norm(diff)
                if dist < 2.0:
                    overlap = 2.0 - dist
                    force = (diff / dist) * overlap * repulsion_strength
                    forces[i] += force
                    forces[j] -= force
        
        # Attractive force towards origin
        for i in range(N):
            forces[i] -= centers[i] * attraction_strength
        
        # Update positions
        centers += forces * damping
        
        # Check convergence
        if np.max(np.abs(forces)) < tol:
            break
    
    # Compute the required outer side length
    max_center_dist = np.max(np.linalg.norm(centers, axis=1))
    side_length = max_center_dist + 1.0
    
    # Construct inner hexagons
    inner = []
    for c in centers:
        inner.append([c[0], c[1], 0.0])
    
    return inner, (0.0, 0.0), side_length, 0.0
# EVOLVE-BLOCK-END
