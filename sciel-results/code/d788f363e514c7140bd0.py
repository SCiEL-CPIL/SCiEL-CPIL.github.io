"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Initialize: 5x10 grid (fits 50 exactly)
    # 5 columns, 10 rows
    cols = 5
    rows = 10
    x_coords = np.linspace(-0.5 + 0.1, 0.5 - 0.1, cols)
    y_coords = np.linspace(-0.5 + 0.05, 0.5 - 0.05, rows)
    positions = np.array([(x, y) for y in y_coords for x in x_coords])
    positions = positions[:N]
    
    # Start with a small radius and grow
    r = 0.05
    best_positions = positions.copy()
    best_r = r
    
    # Hard-sphere relaxation: increase r, push overlapping circles apart
    max_iter_total = 50000
    iter_count = 0
    
    while time.time() < deadline and iter_count < max_iter_total:
        # Try to increase r slightly
        r_new = r * 1.005 + 1e-6
        # Relax positions to eliminate overlaps at radius r_new
        converged = False
        for inner_iter in range(500):
            if time.time() > deadline - 5.0:
                break
            max_overlap = 0.0
            # For each pair, if overlapping, push apart
            for i in range(N):
                for j in range(i+1, N):
                    d = positions[i] - positions[j]
                    dist = np.linalg.norm(d)
                    if dist < 2 * r_new and dist > 1e-12:
                        # Push apart
                        push = (2 * r_new - dist) / (2 * dist)
                        positions[i] += push * d
                        positions[j] -= push * d
                        max_overlap = max(max_overlap, 2 * r_new - dist)
                    elif dist <= 1e-12:
                        # Coincident, push randomly
                        positions[i][0] += 1e-4
                        positions[j][0] -= 1e-4
                        max_overlap = max(max_overlap, 2 * r_new)
            # Keep inside box
            for i in range(N):
                for k in range(2):
                    if positions[i][k] < -0.5 + r_new:
                        positions[i][k] = -0.5 + r_new
                    elif positions[i][k] > 0.5 - r_new:
                        positions[i][k] = 0.5 - r_new
            iter_count += 1
            if max_overlap < 1e-8:
                converged = True
                break
        
        if converged:
            r = r_new
            # Update best
            best_r = r
            best_positions = positions.copy()
        else:
            # Could not converge at r_new, revert and stop growing
            # Try a smaller increment
            r_new = r * 1.001 + 1e-7
            converged = False
            for inner_iter in range(1000):
                if time.time() > deadline - 5.0:
                    break
                max_overlap = 0.0
                for i in range(N):
                    for j in range(i+1, N):
                        d = positions[i] - positions[j]
                        dist = np.linalg.norm(d)
                        if dist < 2 * r_new and dist > 1e-12:
                            push = (2 * r_new - dist) / (2 * dist)
                            positions[i] += push * d
                            positions[j] -= push * d
                            max_overlap = max(max_overlap, 2 * r_new - dist)
                        elif dist <= 1e-12:
                            positions[i][0] += 1e-4
                            positions[j][0] -= 1e-4
                            max_overlap = max(max_overlap, 2 * r_new)
                for i in range(N):
                    for k in range(2):
                        if positions[i][k] < -0.5 + r_new:
                            positions[i][k] = -0.5 + r_new
                        elif positions[i][k] > 0.5 - r_new:
                            positions[i][k] = 0.5 - r_new
                iter_count += 1
                if max_overlap < 1e-8:
                    converged = True
                    break
            if converged:
                r = r_new
                best_r = r
                best_positions = positions.copy()
            else:
                break
    
    # Final local polish: for each circle, try to move it to increase min distance
    # Use a simple hill-climbing: for each circle, try small random perturbations
    for polish_iter in range(200):
        if time.time() > deadline - 5.0:
            break
        improved = False
        for i in range(N):
            if time.time() > deadline - 5.0:
                break
            # Compute current min distance for circle i
            min_dist_i = np.inf
            for j in range(N):
                if i != j:
                    dist = np.linalg.norm(positions[i] - positions[j])
                    min_dist_i = min(min_dist_i, dist)
            # Also boundary distance
            for k in range(2):
                min_dist_i = min(min_dist_i, 0.5 - abs(positions[i][k]))
            
            # Try small perturbations
            for _ in range(50):
                if time.time() > deadline - 5.0:
                    break
                new_pos = positions[i].copy()
                new_pos += np.random.normal(0, 0.005, 2)
                # Check boundary
                valid = True
                for k in range(2):
                    if new_pos[k] < -0.5 or new_pos[k] > 0.5:
                        valid = False
                        break
                if not valid:
                    continue
                # Check min distance from new position
                new_min_dist = np.inf
                for j in range(N):
                    if i != j:
                        dist = np.linalg.norm(new_pos - positions[j])
                        new_min_dist = min(new_min_dist, dist)
                for k in range(2):
                    new_min_dist = min(new_min_dist, 0.5 - abs(new_pos[k]))
                if new_min_dist > min_dist_i * 1.001:
                    positions[i] = new_pos
                    improved = True
                    break
        if not improved and polish_iter > 50:
            break
    
    # Compute final radius
    min_pair_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(positions[i] - positions[j])
            min_pair_dist = min(min_pair_dist, d)
    min_boundary = np.inf
    for i in range(N):
        for k in range(2):
            min_boundary = min(min_boundary, 0.5 - abs(positions[i][k]))
    final_r = min(min_pair_dist / 2, min_boundary)
    
    # Return the best positions (use final positions with computed radius)
    return positions
# EVOLVE-BLOCK-END
