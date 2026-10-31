"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 145s deadline, 5s safety margin
    
    # Perimeter 4 means 2*(w+h) = 4, so w+h = 2.
    # We can parameterize by w, with h = 2 - w.
    # To maximize sum of radii, we want to find a good rectangle aspect ratio.
    # Let's try a few aspect ratios and pick the best one.
    
    best_value = 0.0
    best_result = None
    
    # Try different aspect ratios
    # w from 0.5 to 1.5 (h = 2-w from 1.5 to 0.5)
    aspect_ratios = np.linspace(0.5, 1.5, 11)
    
    for w in aspect_ratios:
        if time.time() > deadline - 5:
            break
        h = 2.0 - w
        
        # For a rectangle of width w and height h, pack N circles.
        # Use a grid-based approach with LP refinement.
        # 
        # Strategy: Use a grid of potential positions, then solve an LP to
        # maximize sum of radii subject to non-overlap constraints.
        #
        # Simpler approach: Use a fixed grid pattern and optimize the radius.
        # For a regular grid of k x m circles in rectangle w x h:
        #   spacing_x = w / k, spacing_y = h / m
        #   radius = min(spacing_x, spacing_y) / 2
        #   sum of radii = N * radius
        #
        # But we can do better with optimization.
        
        # Let's use a simple iterative approach:
        # 1. Initialize with a grid
        # 2. Optimize positions and radii using gradient ascent or LP
        
        # For simplicity and robustness, let's use a grid-based initialization
        # and then a simple local optimization.
        
        # Grid initialization
        # Try different grid configurations
        best_for_aspect = 0.0
        best_for_aspect_result = None
        
        for k in range(1, 7):  # columns
            for m in range(1, 7):  # rows
                if k * m < N:
                    continue
                if time.time() > deadline - 10:
                    break
                
                # Place N circles in a k x m grid
                # The first N positions in the grid
                positions = []
                radii = []
                
                spacing_x = w / k
                spacing_y = h / m
                base_radius = min(spacing_x, spacing_y) / 2.0
                
                for i in range(k):
                    for j in range(m):
                        if len(positions) >= N:
                            break
                        x = (i + 0.5) * spacing_x
                        y = (j + 0.5) * spacing_y
                        positions.append([x, y])
                        radii.append(base_radius)
                
                if len(positions) < N:
                    continue
                
                positions = np.array(positions[:N])
                radii = np.array(radii[:N])
                
                # Check validity: circles must be inside rectangle and non-overlapping
                valid = True
                for i in range(N):
                    if positions[i][0] - radii[i] < -1e-9 or positions[i][0] + radii[i] > w + 1e-9:
                        valid = False
                        break
                    if positions[i][1] - radii[i] < -1e-9 or positions[i][1] + radii[i] > h + 1e-9:
                        valid = False
                        break
                    for j in range(i + 1, N):
                        dist = np.linalg.norm(positions[i] - positions[j])
                        if dist < radii[i] + radii[j] - 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                value = np.sum(radii)
                if value > best_for_aspect:
                    best_for_aspect = value
                    best_for_aspect_result = np.column_stack([positions, radii])
        
        if best_for_aspect_result is not None and best_for_aspect > best_value:
            best_value = best_for_aspect
            best_result = best_for_aspect_result
    
    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result

# EVOLVE-BLOCK-END
