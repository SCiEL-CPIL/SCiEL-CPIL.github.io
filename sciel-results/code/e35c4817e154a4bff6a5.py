"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import random
import math


def construct():
    """
    Pack 26 circles in the unit square [0,1]^2 with disjoint interiors,
    maximizing the sum of their radii.
    
    Uses a force-directed relaxation method:
    - Circles repel each other via spring forces proportional to overlap depth
    - Circles are attracted to the nearest available empty space
    - Discrete event-driven simulation that minimizes potential energy
    - Multiple restarts with different initial configurations
    """
    
    best_solution = None
    best_sum = -1.0
    
    n_restarts = 15
    max_iterations = 5000
    damping = 0.95
    dt = 0.01
    
    for restart in range(n_restarts):
        # Initialize positions using different strategies
        if restart == 0:
            # Grid initialization
            side = int(math.ceil(math.sqrt(N)))
            positions = []
            for i in range(side):
                for j in range(side):
                    if len(positions) < N:
                        x = (i + 0.5) / side
                        y = (j + 0.5) / side
                        positions.append([x, y])
        elif restart == 1:
            # Hexagonal grid
            positions = []
            rows = 5
            cols = 6
            for row in range(rows):
                for col in range(cols):
                    if len(positions) >= N:
                        break
                    x = (col + 0.5) / cols
                    y = (row + 0.5) / rows
                    if row % 2 == 1:
                        x += 0.5 / cols
                    x = min(0.95, max(0.05, x))
                    positions.append([x, y])
        else:
            # Jittered grid with random perturbation
            side = int(math.ceil(math.sqrt(N)))
            positions = []
            for i in range(side):
                for j in range(side):
                    if len(positions) < N:
                        x = (i + 0.5) / side + random.uniform(-0.15, 0.15) / side
                        y = (j + 0.5) / side + random.uniform(-0.15, 0.15) / side
                        x = max(0.05, min(0.95, x))
                        y = max(0.05, min(0.95, y))
                        positions.append([x, y])
        
        positions = np.array(positions, dtype=np.float64)
        velocities = np.zeros_like(positions)
        
        # Start with small equal radii
        r_min = 0.01
        r_max = 0.15
        radii = np.full(N, r_min + 0.3 * (r_max - r_min))
        
        # Force-directed relaxation
        for iteration in range(max_iterations):
            # Compute forces
            forces = np.zeros_like(positions)
            
            # Pairwise repulsion forces
            for i in range(N):
                for j in range(i + 1, N):
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    dist_sq = dx * dx + dy * dy
                    dist = math.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-6
                    
                    min_dist = radii[i] + radii[j]
                    
                    if dist < min_dist:
                        # Overlap: repulsive force proportional to overlap depth
                        overlap = min_dist - dist
                        # Spring constant
                        k = 10.0
                        force_mag = k * overlap
                        # Unit vector from j to i
                        if dist > 1e-12:
                            ux = dx / dist
                            uy = dy / dist
                        else:
                            # Random direction if exactly overlapping
                            angle = random.uniform(0, 2 * math.pi)
                            ux = math.cos(angle)
                            uy = math.sin(angle)
                        
                        forces[i, 0] += force_mag * ux
                        forces[i, 1] += force_mag * uy
                        forces[j, 0] -= force_mag * ux
                        forces[j, 1] -= force_mag * uy
            
            # Boundary forces: push circles back inside
            for i in range(N):
                r = radii[i]
                # Left boundary
                if positions[i, 0] - r < 0:
                    forces[i, 0] += 5.0 * (r - positions[i, 0])
                # Right boundary
                if positions[i, 0] + r > 1:
                    forces[i, 0] -= 5.0 * (positions[i, 0] + r - 1)
                # Bottom boundary
                if positions[i, 1] - r < 0:
                    forces[i, 1] += 5.0 * (r - positions[i, 1])
                # Top boundary
                if positions[i, 1] + r > 1:
                    forces[i, 1] -= 5.0 * (positions[i, 1] + r - 1)
            
            # Update velocities and positions
            velocities = damping * velocities + forces * dt
            positions += velocities * dt
            
            # Clamp positions to stay within bounds (with margin for radius)
            for i in range(N):
                r = radii[i]
                positions[i, 0] = max(r, min(1 - r, positions[i, 0]))
                positions[i, 1] = max(r, min(1 - r, positions[i, 1]))
            
            # Gradually increase radii as packing stabilizes
            if iteration > 100:
                # Increase radii slightly each iteration
                growth_rate = 0.0001
                for i in range(N):
                    # Check if circle can grow
                    max_r = min(positions[i, 0], 1 - positions[i, 0],
                               positions[i, 1], 1 - positions[i, 1])
                    # Check against other circles
                    for j in range(N):
                        if i == j:
                            continue
                        dx = positions[i, 0] - positions[j, 0]
                        dy = positions[i, 1] - positions[j, 1]
                        dist = math.sqrt(dx * dx + dy * dy)
                        available = (dist - radii[j]) / 2
                        max_r = min(max_r, available)
                    
                    if radii[i] < max_r - 1e-6:
                        radii[i] = min(max_r, radii[i] + growth_rate)
            
            # Check for convergence
            if iteration > 100:
                force_mag = np.max(np.abs(forces))
                if force_mag < 1e-4:
                    break
        
        # Validate final solution
        valid = True
        for i in range(N):
            if radii[i] < 0:
                valid = False
                break
            if positions[i, 0] - radii[i] < -1e-6 or positions[i, 0] + radii[i] > 1 + 1e-6:
                valid = False
                break
            if positions[i, 1] - radii[i] < -1e-6 or positions[i, 1] + radii[i] > 1 + 1e-6:
                valid = False
                break
        
        if valid:
            for i in range(N):
                for j in range(i + 1, N):
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    dist_sq = dx * dx + dy * dy
                    min_dist = radii[i] + radii[j]
                    if dist_sq < min_dist * min_dist - 1e-6:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            # Clamp radii to be safely inside
            for i in range(N):
                radii[i] = min(radii[i], positions[i, 0], 1 - positions[i, 0],
                              positions[i, 1], 1 - positions[i, 1])
            
            # Final validation
            valid2 = True
            for i in range(N):
                if radii[i] < 0:
                    valid2 = False
                    break
                if positions[i, 0] - radii[i] < -1e-7 or positions[i, 0] + radii[i] > 1 + 1e-7:
                    valid2 = False
                    break
                if positions[i, 1] - radii[i] < -1e-7 or positions[i, 1] + radii[i] > 1 + 1e-7:
                    valid2 = False
                    break
            
            if valid2:
                for i in range(N):
                    for j in range(i + 1, N):
                        dx = positions[i, 0] - positions[j, 0]
                        dy = positions[i, 1] - positions[j, 1]
                        dist_sq = dx * dx + dy * dy
                        min_dist = radii[i] + radii[j]
                        if dist_sq < min_dist * min_dist - 1e-7:
                            valid2 = False
                            break
                    if not valid2:
                        break
                
                if valid2:
                    total = np.sum(radii)
                    if total > best_sum:
                        best_sum = total
                        best_solution = np.column_stack([positions[:, 0], positions[:, 1], radii])
    
    # If no valid solution found, fall back to the simple grid
    if best_solution is None:
        side = int(math.ceil(math.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_solution

# EVOLVE-BLOCK-END
