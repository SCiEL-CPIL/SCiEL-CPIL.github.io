import numpy as np
import random
import math
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Vertices of a unit regular hexagon centered at origin, angle 0
    # Vertices at angles 0, 60, 120, 180, 240, 300
    hex_verts = np.array([
        [math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] 
        for i in range(6)
    ])
    
    # Normals to the sides of the outer hexagon (which is also aligned)
    # The sides are perpendicular to vectors at angles 30, 90, 150, 210, 270, 330
    outer_normals = np.array([
        [math.cos(math.pi/6 + i * math.pi / 3), math.sin(math.pi/6 + i * math.pi / 3)] 
        for i in range(6)
    ])
    
    # Precompute min and max projections of unit hexagon onto outer normals
    proj_matrix = hex_verts @ outer_normals.T  # 6x6
    max_proj_inner = np.max(proj_matrix, axis=0)  # (6,)
    min_proj_inner = np.min(proj_matrix, axis=0)  # (6,)
    
    # Helper to compute minimal enclosing hexagon side length for given positions
    def compute_S(pos):
        # pos is Nx2 array
        # For each hexagon i, and each normal k:
        # constraint: pos[i] @ n[k] + max_proj_inner[k] <= S * sqrt(3) / 2
        # S >= (pos[i] @ n[k] + max_proj_inner[k]) * 2 / sqrt(3)
        projs = pos @ outer_normals.T  # Nx6
        projs += max_proj_inner  # Nx6
        max_val = np.max(projs)
        S_req = max_val * 2.0 / math.sqrt(3)
        return S_req
    
    # Helper to check if two hexagons overlap and return penetration depth
    def check_overlap(pos_i, pos_j):
        # Use SAT with 6 axes
        dist_sq = np.sum((pos_i - pos_j)**2)
        if dist_sq > 4.0:  # Max distance for overlap is 2 (diagonal)
            return 0.0
        
        min_penetration = float('inf')
        
        for k in range(6):
            n = outer_normals[k]
            min_p_i = pos_i @ n + min_proj_inner[k]
            max_p_i = pos_i @ n + max_proj_inner[k]
            min_p_j = pos_j @ n + min_proj_inner[k]
            max_p_j = pos_j @ n + max_proj_inner[k]
            
            if max_p_i < min_p_j or max_p_j < min_p_i:
                return 0.0  # Separated
            
            pen = min(max_p_i, max_p_j) - max(min_p_i, min_p_j)
            if pen < min_penetration:
                min_penetration = pen
        
        return min_penetration
    
    # Helper to compute total overlap penalty
    def compute_penalty(pos):
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                pen = check_overlap(pos[i], pos[j])
                if pen > 0:
                    penalty += pen
        return penalty
    
    # Helper to compute total cost
    def compute_cost(pos):
        S = compute_S(pos)
        penalty = compute_penalty(pos)
        return S + penalty * 100.0  # High penalty to avoid overlaps
    
    # Generate initial positions: honeycomb cluster
    def generate_initial_positions():
        positions = []
        # 4x3 rectangular patch of hex grid
        for i in range(4):
            for j in range(3):
                x = i * 1.5 + j * 0.5
                y = j * math.sqrt(3) / 2
                positions.append([x, y])
        
        centers = np.array(positions)
        mean_c = np.mean(centers, axis=0)
        centers = centers - mean_c
        return centers
    
    # Simulated Annealing
    def simulated_annealing(initial_pos, max_time=110.0):
        T = 5.0
        cooling = 0.9995
        min_T = 0.001
        step_scale = 0.3
        
        current_pos = initial_pos.copy()
        current_cost = compute_cost(current_pos)
        best_pos = current_pos.copy()
        best_cost = current_cost
        
        start_time = time.time()
        
        while T > min_T:
            if time.time() - start_time > max_time:
                break
            
            # Pick random hexagon to move
            idx = random.randint(0, N - 1)
            old_pos = current_pos[idx].copy()
            
            # Move by random small vector
            step_size = T * step_scale
            dx = random.uniform(-step_size, step_size)
            dy = random.uniform(-step_size, step_size)
            
            current_pos[idx][0] += dx
            current_pos[idx][1] += dy
            
            new_cost = compute_cost(current_pos)
            
            # Acceptance probability
            if new_cost < current_cost:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_pos = current_pos.copy()
            else:
                delta = new_cost - current_cost
                if delta > 0:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        current_cost = new_cost
                    else:
                        current_pos[idx] = old_pos
            
            T *= cooling
        
        return best_pos, best_cost
    
    # Multi-start local search
    best_positions = None
    best_S = float('inf')
    
    num_starts = 5
    for start in range(num_starts):
        if start == 0:
            init_pos = generate_initial_positions()
        else:
            # Perturb initial positions
            init_pos = generate_initial_positions()
            for i in range(N):
                init_pos[i][0] += random.uniform(-0.5, 0.5)
                init_pos[i][1] += random.uniform(-0.5, 0.5)
        
        pos, cost = simulated_annealing(init_pos, max_time=20.0)
        
        # Verify no overlaps in final solution
        has_overlap = False
        for i in range(N):
            for j in range(i + 1, N):
                if check_overlap(pos[i], pos[j]) > 0:
                    has_overlap = True
                    break
            if has_overlap:
                break
        
        if not has_overlap:
            S = compute_S(pos)
            if S < best_S:
                best_S = S
                best_positions = pos.copy()
    
    # If no valid solution found, use the one with lowest cost
    if best_positions is None:
        pos, cost = simulated_annealing(generate_initial_positions(), max_time=110.0)
        best_positions = pos
        best_S = compute_S(pos)
    
    inner = []
    for p in best_positions:
        inner.append([float(p[0]), float(p[1]), 0.0])
    
    return inner, (0.0, 0.0), float(best_S), 0.0

# EVOLVE-BLOCK-END
