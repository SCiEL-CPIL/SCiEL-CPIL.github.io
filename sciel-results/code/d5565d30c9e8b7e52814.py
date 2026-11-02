"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START
def construct():
    """
    Pack N=26 circles in [0,1]^2, disjoint interiors, maximize sum of radii.
    
    Strategy:
    1. Generate a good initial guess using a structured placement (e.g., hexagonal or grid-based).
    2. Use a simple local optimization (gradient-free, e.g., Nelder-Mead or a custom repulsion-based 
       iterative method) to refine the configuration.
    3. Use multiple restarts with random perturbations to escape local minima.
    4. Keep the best feasible solution found.
    
    Time budget: ~100s to leave safety margin.
    """
    start_time = time.time()
    deadline = start_time + 110  # 110s out of 120s
    
    best_sum = -np.inf
    best_config = None
    
    # Initial structured guess: place circles in a roughly hexagonal pattern
    def initial_guess(n, seed):
        rng = np.random.default_rng(seed)
        # Start with a grid-like arrangement, then jitter
        side = int(np.ceil(np.sqrt(n)))
        cols = side
        rows = int(np.ceil(n / side))
        
        # Create a grid of positions
        xs = np.linspace(0.5 / cols, 1 - 0.5 / cols, cols)
        ys = np.linspace(0.5 / rows, 1 - 0.5 / rows, rows)
        
        positions = []
        radii = []
        for i in range(cols):
            for j in range(rows):
                if len(positions) >= n:
                    break
                x = xs[i] + rng.uniform(-0.02, 0.02)
                y = ys[j] + rng.uniform(-0.02, 0.02)
                # Initial radius estimate
                r = 0.5 / max(cols, rows) * 0.8
                positions.append((x, y))
                radii.append(r)
        
        # If we don't have enough, add more random positions
        while len(positions) < n:
            x = rng.uniform(0.1, 0.9)
            y = rng.uniform(0.1, 0.9)
            r = 0.03
            positions.append((x, y))
            radii.append(r)
        
        return np.array(positions[:n]), np.array(radii[:n])
    
    def check_feasibility(positions, radii):
        """Check if all circles are inside [0,1]^2 and don't overlap."""
        n = len(positions)
        # Boundary check
        for i in range(n):
            x, y = positions[i]
            r = radii[i]
            if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
                return False
        # Overlap check
        for i in range(n):
            for j in range(i + 1, n):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < radii[i] + radii[j] - 1e-9:
                    return False
        return True
    
    def optimize(positions, radii, max_iter=5000, time_limit=None):
        """
        Simple iterative optimization using repulsion/attraction forces.
        """
        n = len(positions)
        positions = positions.copy()
        radii = radii.copy()
        
        # Use a simple gradient-free approach: adjust positions and radii iteratively
        lr_pos = 0.005
        lr_rad = 0.001
        
        for it in range(max_iter):
            if time_limit is not None and time.time() > time_limit:
                break
            
            # Compute forces
            forces = np.zeros_like(positions)
            
            # Repulsion between overlapping circles
            for i in range(n):
                for j in range(i + 1, n):
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy) + 1e-12
                    overlap = radii[i] + radii[j] - dist
                    if overlap > 0:
                        # Push apart
                        fx = dx / dist * overlap
                        fy = dy / dist * overlap
                        forces[i, 0] += fx
                        forces[i, 1] += fy
                        forces[j, 0] -= fx
                        forces[j, 1] -= fy
            
            # Boundary repulsion
            for i in range(n):
                # Left wall
                if positions[i, 0] - radii[i] < 0:
                    forces[i, 0] += (radii[i] - positions[i, 0]) * 10
                # Right wall
                if positions[i, 0] + radii[i] > 1:
                    forces[i, 0] -= (positions[i, 0] + radii[i] - 1) * 10
                # Bottom wall
                if positions[i, 1] - radii[i] < 0:
                    forces[i, 1] += (radii[i] - positions[i, 1]) * 10
                # Top wall
                if positions[i, 1] + radii[i] > 1:
                    forces[i, 1] -= (positions[i, 1] + radii[i] - 1) * 10
            
            # Update positions
            positions += lr_pos * forces
            
            # Try to increase radii slightly, but keep feasible
            for i in range(n):
                # Check max possible radius given current position
                max_r = min(positions[i, 0], 1 - positions[i, 0], positions[i, 1], 1 - positions[i, 1])
                # Also check distance to other circles
                for j in range(n):
                    if i == j:
                        continue
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    max_r = min(max_r, (dist - radii[j]) / 2)
                radii[i] = max(0.01, min(radii[i] + lr_rad, max_r))
            
            # Clamp positions to stay in bounds
            for i in range(n):
                positions[i, 0] = np.clip(positions[i, 0], radii[i], 1 - radii[i])
                positions[i, 1] = np.clip(positions[i, 1], radii[i], 1 - radii[i])
        
        return positions, radii
    
    # Multiple restarts
    n_restarts = 20
    rng = np.random.default_rng(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5:
            break
        
        # Generate initial guess
        positions, radii = initial_guess(N, seed=restart * 100 + 42)
        
        # Optimize
        positions, radii = optimize(positions, radii, max_iter=3000, time_limit=deadline - 2)
        
        # Check feasibility
        if check_feasibility(positions, radii):
            s = np.sum(radii)
            if s > best_sum:
                best_sum = s
                best_config = (positions.copy(), radii.copy())
    
    # If no feasible solution found, return a trivial one
    if best_config is None:
        # Return a grid of small circles
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    positions, radii = best_config
    result = np.column_stack([positions, radii])
    return result
# EVOLVE-BLOCK-END
