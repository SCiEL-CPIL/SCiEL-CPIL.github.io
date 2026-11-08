import numpy as np
import time

N = 21

def construct():
    start_time = time.time()
    deadline = 140  # safety margin
    
    # Initial configuration: 5x5 grid, use 21 circles
    # Rectangle: 1x1 (perimeter 4)
    W = 1.0
    H = 1.0
    
    # 5x5 grid positions
    side = 5
    positions = []
    for i in range(side):
        for j in range(side):
            positions.append([(i + 0.5) / side, (j + 0.5) / side])
    positions = np.array(positions[:N])
    r = np.full(N, 0.1 - 1e-3)
    
    best_value = np.sum(r)
    best_W, best_H = W, H
    best_pos = positions.copy()
    best_r = r.copy()
    
    # Optimization using iterative adjustment
    # Use a simple gradient-free approach: perturb and accept if better
    
    def evaluate(pos, r, W, H):
        """Return (value, valid) where value is sum of radii if valid, else -inf"""
        # Check bounds
        if np.any(pos[:, 0] - r < -1e-9) or np.any(pos[:, 0] + r > W + 1e-9):
            return -np.inf, False
        if np.any(pos[:, 1] - r < -1e-9) or np.any(pos[:, 1] + r > H + 1e-9):
            return -np.inf, False
        # Check overlaps
        n = len(r)
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt((pos[i,0]-pos[j,0])**2 + (pos[i,1]-pos[j,1])**2)
                if dist < r[i] + r[j] - 1e-9:
                    return -np.inf, False
        return np.sum(r), True
    
    # Try to improve by adjusting radii and positions
    # Use a simple local search
    
    current_pos = positions.copy()
    current_r = r.copy()
    current_W = W
    current_H = H
    current_value = np.sum(current_r)
    
    # First, try to maximize radii for fixed positions and rectangle
    # Then try to adjust positions and rectangle
    
    # Step 1: For fixed positions, find max uniform radius
    # The max radius is limited by:
    # - Distance to walls: min over all i of min(x_i, W-x_i, y_i, H-y_i)
    # - Distance between circles: min over all i<j of dist(i,j)/2
    
    def max_uniform_radius(pos, W, H):
        n = len(pos)
        max_r = np.inf
        # Wall constraints
        for i in range(n):
            max_r = min(max_r, pos[i,0], W - pos[i,0], pos[i,1], H - pos[i,1])
        # Circle-circle constraints
        for i in range(n):
            for j in range(i+1, n):
                d = np.sqrt((pos[i,0]-pos[j,0])**2 + (pos[i,1]-pos[j,1])**2)
                max_r = min(max_r, d/2)
        return max_r
    
    # Try different rectangle aspect ratios
    # W + H = 2, so H = 2 - W
    best_overall_value = 0
    best_overall_pos = None
    best_overall_r = None
    best_overall_W = None
    best_overall_H = None
    
    # Try a range of aspect ratios
    for W_frac in np.linspace(0.5, 1.5, 21):  # W from 0.5 to 1.5, H from 1.5 to 0.5
        W_test = W_frac
        H_test = 2 - W_frac
        if W_test <= 0 or H_test <= 0:
            continue
        
        # Scale positions to fit in this rectangle
        # Keep relative positions but scale
        pos_test = current_pos.copy()
        pos_test[:, 0] *= W_test
        pos_test[:, 1] *= H_test
        
        # Find max uniform radius
        r_max = max_uniform_radius(pos_test, W_test, H_test)
        if r_max > 0:
            value = N * r_max
            if value > best_overall_value:
                best_overall_value = value
                best_overall_pos = pos_test.copy()
                best_overall_r = np.full(N, r_max)
                best_overall_W = W_test
                best_overall_H = H_test
    
    if best_overall_pos is not None:
        best_value = best_overall_value
        best_pos = best_overall_pos
        best_r = best_overall_r
        best_W = best_overall_W
        best_H = best_overall_H
    
    # Now try local optimization
    # Use a simple iterative method: nudge circles to increase radius
    
    if time.time() < deadline:
        # Local optimization: try to increase radius by moving circles
        current_pos = best_pos.copy()
        current_r = best_r.copy()
        current_W = best_W
        current_H = best_H
        
        # Try to increase all radii slightly and see if we can adjust positions
        # Use a simple gradient-free approach
        
        for iteration in range(500):
            if time.time() > deadline:
                break
            
            # Try to increase radius by a small amount
            dr = 1e-4
            new_r = current_r + dr
            
            # Check if this is feasible with current positions
            feasible = True
            for i in range(N):
                if current_pos[i,0] - new_r[i] < 0 or current_pos[i,0] + new_r[i] > current_W:
                    feasible = False
                    break
                if current_pos[i,1] - new_r[i] < 0 or current_pos[i,1] + new_r[i] > current_H:
                    feasible = False
                    break
            if feasible:
                for i in range(N):
                    for j in range(i+1, N):
                        d = np.sqrt((current_pos[i,0]-current_pos[j,0])**2 + (current_pos[i,1]-current_pos[j,1])**2)
                        if d < new_r[i] + new_r[j] - 1e-10:
                            feasible = False
                            break
                    if not feasible:
                        break
            
            if feasible:
                current_r = new_r
            else:
                # Try to move circles to make room
                # Find the most constrained pair
                # This is complex, so just break
                break
        
        value = np.sum(current_r)
        if value > best_value:
            best_value = value
            best_pos = current_pos
            best_r = current_r
    
    # Build result
    result = np.zeros((N, 3))
    result[:, 0] = best_pos[:, 0]
    result[:, 1] = best_pos[:, 1]
    result[:, 2] = best_r
    
    return result
