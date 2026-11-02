import numpy as np
import time
import math
import random

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def required_outer_side(positions):
        """
        Compute the exact side length of the smallest regular hexagon (angle 0, centered at origin)
        that contains all unit hexagons at given positions.
        """
        verts = np.zeros((N, 6, 2))
        for i in range(N):
            verts[i] = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, 0.0)
        verts_flat = verts.reshape(-1, 2)
        
        x = verts_flat[:, 0]
        y = verts_flat[:, 1]
        sqrt3 = np.sqrt(3.0)
        
        # The support function of a regular hexagon of side S centered at origin 
        # in directions 30, 90, 150, 210, 270, 330 degrees is S * sqrt(3) / 2.
        # We need max_k (v . n_k) <= S * sqrt(3) / 2 for all vertices v.
        # Thus S >= 2/sqrt(3) * max_k (v . n_k).
        
        # Normals n_k for hexagon with flat top/bottom (vertices at 0, 60, ...)
        # Vertices are at angles 0, 60, 120...
        # Normals are at 30, 90, 150...
        # n_0 = (cos(30), sin(30)) = (sqrt(3)/2, 1/2)
        # n_1 = (cos(90), sin(90)) = (0, 1)
        # n_2 = (cos(150), sin(150)) = (-sqrt(3)/2, 1/2)
        # n_3 = (cos(210), sin(210)) = (-sqrt(3)/2, -1/2)
        # n_4 = (cos(270), sin(270)) = (0, -1)
        # n_5 = (cos(330), sin(330)) = (sqrt(3)/2, -1/2)
        
        L1 = x * (sqrt3/2) + y * (1/2)
        L2 = y
        L3 = x * (-sqrt3/2) + y * (1/2)
        L4 = x * (-sqrt3/2) + y * (-1/2)
        L5 = -y
        L6 = x * (sqrt3/2) + y * (-1/2)
        
        all_L = np.stack([L1, L2, L3, L4, L5, L6], axis=1)
        max_proj = np.max(all_L)
        return (2.0 / sqrt3) * max_proj

    def check_validity(positions, tol=1e-4):
        """
        Check if hexagons are disjoint.
        Returns True if valid, False otherwise.
        Uses SAT (Separating Axis Theorem) for hexagon-hexagon collision.
        """
        verts_list = [hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, 0.0) for i in range(N)]
        
        for i in range(N):
            for j in range(i + 1, N):
                v1 = verts_list[i]
                v2 = verts_list[j]
                
                # Quick bounding box check
                if (np.max(v1[:,0]) + tol < np.min(v2[:,0]) or 
                    np.max(v2[:,0]) + tol < np.min(v1[:,0]) or
                    np.max(v1[:,1]) + tol < np.min(v2[:,1]) or
                    np.max(v2[:,1]) + tol < np.min(v1[:,1])):
                    continue

                # SAT check
                # Normals for hexagon 1 and 2 (all same orientation)
                # We only need to check 6 unique normal directions for regular hexagons
                # Actually, since they can rotate, we should check normals of both.
                # But in our current implementation, we fix angle to 0.
                # If we allowed rotation, we'd need 12 checks.
                # For now, fixed angle 0 means 6 unique normals.
                
                # Generate normals for hexagon 1
                for k in range(6):
                    edge = v1[(k+1)%6] - v1[k]
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm < 1e-10: continue
                    normal /= norm
                    
                    proj1 = np.dot(v1, normal)
                    proj2 = np.dot(v2, normal)
                    
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    
                    # Check for separation
                    if max1 + tol < min2 or max2 + tol < min1:
                        break
                else:
                    # No separating axis found among hexagon 1's normals
                    # Check hexagon 2's normals (redundant if same orientation, but safe)
                    # Actually, for same orientation, normals are identical sets.
                    # So if no separation found, they overlap.
                    return False
        return True

    def simulated_annealing(initial_positions, time_budget=5.0):
        """
        Run Simulated Annealing to optimize positions.
        """
        current_pos = initial_positions.copy()
        current_side = required_outer_side(current_pos)
        
        # Ensure we start with a valid config or a very large side
        # If invalid, we should probably not use this start, or fix it.
        # For SA, we typically start valid.
        if not check_validity(current_pos):
            # Try to spread out
            current_pos = initial_positions * 2.0
            current_side = required_outer_side(current_pos)
            if not check_validity(current_pos):
                current_pos = initial_positions * 3.0
                current_side = required_outer_side(current_pos)
                if not check_validity(current_pos):
                    return current_pos, current_side

        best_pos = current_pos.copy()
        best_side = current_side
        
        T = 1.0  # Initial temperature
        cooling_rate = 0.9995
        
        end_time = time.time() + time_budget
        
        while time.time() < end_time:
            if T < 1e-5:
                break
                
            # Generate a new candidate
            # Move one or a few random hexagons
            num_moves = random.randint(1, 3)
            indices = random.sample(range(N), num_moves)
            
            new_pos = current_pos.copy()
            for idx in indices:
                # Random displacement
                # Step size scales with temperature
                step_size = T * 0.5
                dx = random.uniform(-step_size, step_size)
                dy = random.uniform(-step_size, step_size)
                new_pos[idx, 0] += dx
                new_pos[idx, 1] += dy
            
            # Check validity
            if not check_validity(new_pos):
                # Invalid move: reject with high probability, or accept if very hot
                # Standard SA usually rejects invalid states in constrained packing,
                # or uses a penalty. Here we use a hard constraint rejection to keep search in valid space.
                # This makes it a "constrained SA".
                # If we reject all invalids, we might get stuck. 
                # Alternative: accept invalids with penalty in objective.
                # Let's stick to hard constraint for simplicity and robustness of "valid" result.
                # But if we are stuck, we might need to allow temporary invalidity.
                # Let's try a hybrid: if T is high, allow invalidity with penalty.
                # Actually, let's just reject invalids to ensure we always have a valid best.
                # To escape local minima, we rely on the random steps being large enough initially.
                continue

            new_side = required_outer_side(new_pos)
            delta = new_side - current_side
            
            if delta < 0:
                # Improvement
                current_pos = new_pos
                current_side = new_side
                if new_side < best_side:
                    best_pos = new_pos.copy()
                    best_side = new_side
            else:
                # Worse
                # Accept with probability exp(-delta / T)
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_pos = new_pos
                    current_side = new_side
            
            T *= cooling_rate
            
            # Occasionally restart or jitter to escape local minima
            if random.random() < 0.01:
                # Large random jump
                jump_size = T * 2.0
                for i in range(N):
                    new_pos_jump = current_pos.copy()
                    new_pos_jump[i, 0] += random.uniform(-jump_size, jump_size)
                    new_pos_jump[i, 1] += random.uniform(-jump_size, jump_size)
                    if check_validity(new_pos_jump):
                        jump_side = required_outer_side(new_pos_jump)
                        if jump_side < current_side or random.random() < math.exp((jump_side - current_side) / T):
                            current_pos = new_pos_jump
                            current_side = jump_side
                            if current_side < best_side:
                                best_pos = current_pos.copy()
                                best_side = current_side

        return best_pos, best_side

    # Generate diverse initial configurations
    configs = []
    
    # 1. Hexagonal lattice (compact)
    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    init1 = []
    for q, r in coords:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        init1.append([x, y])
    configs.append(np.array(init1))
    
    # 2. Circular arrangement
    init2 = np.zeros((N, 2))
    for i in range(N):
        angle = 2 * np.pi * i / N
        r = 2.5
        init2[i, 0] = r * np.cos(angle)
        init2[i, 1] = r * np.sin(angle)
    configs.append(init2)
    
    # 3. Random seeds
    np.random.seed(42)
    init3 = np.random.uniform(-5, 5, (N, 2))
    configs.append(init3)
    
    np.random.seed(123)
    init4 = np.random.uniform(-5, 5, (N, 2))
    configs.append(init4)
    
    # 4. Perturbed Lattice
    init5 = init1 + np.random.uniform(-0.5, 0.5, (N, 2))
    configs.append(init5)

    # 5. Another random
    np.random.seed(999)
    init6 = np.random.uniform(-5, 5, (N, 2))
    configs.append(init6)

    best_positions = None
    best_side = float('inf')
    
    # Allocate time to each restart
    num_restarts = 5
    time_per_restart = (deadline - time.time()) / num_restarts
    if time_per_restart < 1.0:
        time_per_restart = 1.0
        num_restarts = 1

    for i, init in enumerate(configs):
        if time.time() > deadline:
            break
        
        # Ensure initial config is valid; if not, spread out
        verts_list = [hexagon_vertices(init[i, 0], init[i, 1], 1.0, 0.0) for i in range(N)]
        valid = True
        for a in range(N):
            for b in range(a + 1, N):
                v1 = verts_list[a]
                v2 = verts_list[b]
                # Simple bounding box check for speed in init validation
                if (np.max(v1[:,0]) < np.min(v2[:,0]) - 1e-4 or 
                    np.max(v2[:,0]) < np.min(v1[:,0]) - 1e-4 or
                    np.max(v1[:,1]) < np.min(v2[:,1]) - 1e-4 or
                    np.max(v2[:,1]) < np.min(v1[:,1]) - 1e-4):
                    continue
                # SAT
                for k in range(6):
                    edge = v1[(k+1)%6] - v1[k]
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm < 1e-10: continue
                    normal /= norm
                    proj1 = np.dot(v1, normal)
                    proj2 = np.dot(v2, normal)
                    if np.max(proj1) + 1e-4 < np.min(proj2) or np.max(proj2) + 1e-4 < np.min(proj1):
                        break
                else:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # Try to fix by scaling
            fixed = False
            for scale in [1.5, 2.0, 2.5, 3.0]:
                scaled = init * scale
                if check_validity(scaled):
                    init = scaled
                    fixed = True
                    break
            if not fixed:
                continue

        # Run optimization
        pos, side = simulated_annealing(init, time_budget=time_per_restart)
        if pos is not None and side < best_side:
            best_side = side
            best_positions = pos

    # Fallback
    if best_positions is None:
        # Use the first valid config
        for init in configs:
            if check_validity(init):
                best_positions = init
                best_side = required_outer_side(init)
                break
        
        if best_positions is None:
            # Last resort: spread out linearly
            best_positions = np.array([[i * 3.0, 0.0] for i in range(N)])
            best_side = required_outer_side(best_positions)
    
    # Calculate final exact side
    final_side = required_outer_side(best_positions)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), 0.0])
        
    return inner, (0.0, 0.0), float(final_side), 0.0

# EVOLVE-BLOCK-END
