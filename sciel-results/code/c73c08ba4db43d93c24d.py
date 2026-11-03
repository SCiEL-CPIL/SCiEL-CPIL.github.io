"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    np.random.seed(42)
    
    # Initialize positions randomly
    # Using a jittered grid to ensure good initial spread
    side = int(np.ceil(np.sqrt(N)))
    positions = np.zeros((N, 2))
    for i in range(N):
        row = i // side
        col = i % side
        # Center of cell + random jitter
        positions[i, 0] = (col + 0.5) / side + np.random.uniform(-0.2, 0.2) / side
        positions[i, 1] = (row + 0.5) / side + np.random.uniform(-0.2, 0.2) / side
        # Clip to ensure inside
        positions[i] = np.clip(positions[i], 0.1, 0.9)
        
    radii = np.full(N, 1e-4)
    
    # Parameters for Grow and Relax
    dr = 0.001 # Initial radius increment
    step_size = 0.05 # Position update step size
    max_iter = 10000
    min_step = 1e-5
    current_step = step_size
    
    best_sum = np.sum(radii)
    best_pos = positions.copy()
    best_rad = radii.copy()
    
    for iteration in range(max_iter):
        # 1. Grow radii
        radii += dr
        
        # 2. Relax positions to resolve conflicts
        for inner_iter in range(100):
            forces = np.zeros((N, 2))
            valid = True
            
            # Boundary forces
            for i in range(N):
                if positions[i, 0] < radii[i]:
                    forces[i, 0] += (radii[i] - positions[i, 0])
                elif positions[i, 0] > 1 - radii[i]:
                    forces[i, 0] -= (positions[i, 0] - (1 - radii[i]))
                    
                if positions[i, 1] < radii[i]:
                    forces[i, 1] += (radii[i] - positions[i, 1])
                elif positions[i, 1] > 1 - radii[i]:
                    forces[i, 1] -= (positions[i, 1] - (1 - radii[i]))
                
                # Check if strictly inside (with small epsilon)
                if positions[i, 0] - radii[i] < -1e-9 or positions[i, 0] + radii[i] > 1 + 1e-9:
                    valid = False
                if positions[i, 1] - radii[i] < -1e-9 or positions[i, 1] + radii[i] > 1 + 1e-9:
                    valid = False
            
            # Overlap forces
            for i in range(N):
                for j in range(i+1, N):
                    delta = positions[i] - positions[j]
                    dist_sq = np.dot(delta, delta)
                    min_dist = radii[i] + radii[j]
                    if dist_sq < min_dist * min_dist:
                        dist = np.sqrt(dist_sq)
                        if dist < 1e-9:
                            # Random nudge if centers coincide
                            delta = np.array([np.random.uniform(-1,1), np.random.uniform(-1,1)])
                            dist = np.linalg.norm(delta)
                        overlap = min_dist - dist
                        force_mag = overlap
                        forces[i] += (delta / dist) * force_mag
                        forces[j] -= (delta / dist) * force_mag
                        valid = False
            
            if valid:
                break
            
            # Update positions
            positions += forces * current_step
            
            # Clip to square (safety net, forces should handle it)
            positions = np.clip(positions, 0, 1)
            
        # If we couldn't resolve conflicts even after relaxation, shrink step or revert?
        # In this simple version, we just accept the state and continue.
        # To improve, we can check if sum of radii increased.
        
        current_sum = np.sum(radii)
        if current_sum > best_sum:
            best_sum = current_sum
            best_pos = positions.copy()
            best_rad = radii.copy()
            
        # Adapt step size
        if iteration % 100 == 0:
            # Reduce growth rate slightly to allow finer packing
            dr *= 0.98
            # Reduce step size if oscillating? Hard to detect simply. 
            # Let's just decay step size slowly.
            current_step *= 0.995
            if current_step < min_step:
                current_step = min_step
                
    # Final check and return
    # Ensure validity
    def check_valid(pos, rad):
        if np.any(pos[:, 0] - rad < -1e-9) or np.any(pos[:, 0] + rad > 1 + 1e-9):
            return False
        if np.any(pos[:, 1] - rad < -1e-9) or np.any(pos[:, 1] + rad > 1 + 1e-9):
            return False
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(pos[i] - pos[j])
                if dist < rad[i] + rad[j] - 1e-9:
                    return False
        return True

    if check_valid(best_pos, best_rad):
        return np.column_stack([best_pos, best_rad])
    
    # Fallback if invalid (should not happen often with this method)
    # Return a simple grid
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            cells.append([ (i+0.5)/side, (j+0.5)/side, 0.5/side * 0.9 ])
    return np.array(cells[:N])

# EVOLVE-BLOCK-END
