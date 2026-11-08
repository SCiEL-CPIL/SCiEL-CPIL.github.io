import numpy as np
import time

N = 21

def construct():
    def relax(positions, r, steps=200):
        """
        Resolve collisions for circles of radius r.
        Returns max displacement in the last step.
        """
        max_disp = 0.0
        for _ in range(steps):
            disp = 0.0
            # Pairwise repulsion
            for i in range(N):
                for j in range(i + 1, N):
                    dx = positions[j, 0] - positions[i, 0]
                    dy = positions[j, 1] - positions[i, 1]
                    dist_sq = dx * dx + dy * dy
                    min_dist = 2 * r
                    if dist_sq < min_dist * min_dist:
                        dist = np.sqrt(dist_sq)
                        if dist < 1e-9:
                            dx = np.random.uniform(-0.01, 0.01)
                            dy = np.random.uniform(-0.01, 0.01)
                            dist = np.sqrt(dx * dx + dy * dy)
                        push = (min_dist - dist) / 2.0
                        nx = dx / dist
                        ny = dy / dist
                        positions[i, 0] -= nx * push
                        positions[i, 1] -= ny * push
                        positions[j, 0] += nx * push
                        positions[j, 1] += ny * push
                        disp += push
            
            # Boundary constraints
            for i in range(N):
                if positions[i, 0] < r:
                    positions[i, 0] = r
                    disp += 0.01
                elif positions[i, 0] > 1 - r:
                    positions[i, 0] = 1 - r
                    disp += 0.01
                if positions[i, 1] < r:
                    positions[i, 1] = r
                    disp += 0.01
                elif positions[i, 1] > 1 - r:
                    positions[i, 1] = 1 - r
                    disp += 0.01
            
            max_disp = max(max_disp, disp)
            if disp < 1e-6:
                break
        return max_disp

    best_score = 0.0
    best_positions = None
    best_r = 0.0
    
    start_time = time.time()
    seed = 42
    
    while time.time() - start_time < 100:
        seed += 1
        np.random.seed(seed)
        
        # Initialize in a grid
        grid_pos = []
        side = 5
        for i in range(side):
            for j in range(side):
                grid_pos.append([(i + 0.5) / side, (j + 0.5) / side])
        
        np.random.shuffle(grid_pos)
        positions = np.array(grid_pos[:N])
        positions += np.random.uniform(-0.01, 0.01, (N, 2))
        
        r = 0.01
        dr = 0.002
        max_r = 0.5
        stuck_counter = 0
        
        while r < max_r and time.time() - start_time < 95:
            disp = relax(positions, r, steps=200)
            if disp < 1e-5:
                r += dr
                stuck_counter = 0
            else:
                if stuck_counter > 50:
                    # Add noise to escape local minimum
                    positions += np.random.uniform(-0.05, 0.05, (N, 2))
                    positions[:, 0] = np.clip(positions[:, 0], r, 1 - r)
                    positions[:, 1] = np.clip(positions[:, 1], r, 1 - r)
                    stuck_counter = 0
                else:
                    stuck_counter += 1
                    relax(positions, r, steps=500)
        
        # Final validation
        valid = True
        for i in range(N):
            if positions[i, 0] < r - 1e-5 or positions[i, 0] > 1 - r + 1e-5:
                valid = False
            if positions[i, 1] < r - 1e-5 or positions[i, 1] > 1 - r + 1e-5:
                valid = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < 2 * r - 1e-5:
                    valid = False
        
        if valid:
            score = N * r
            if score > best_score:
                best_score = score
                best_positions = positions.copy()
                best_r = r

    if best_positions is None:
        # Fallback
        side = 5
        cells = []
        count = 0
        for i in range(side):
            for j in range(side):
                if count < N:
                    cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.1])
                    count += 1
        return np.array(cells)
    
    result = []
    for i in range(N):
        result.append([best_positions[i, 0], best_positions[i, 1], best_r])
    
    return np.array(result)
