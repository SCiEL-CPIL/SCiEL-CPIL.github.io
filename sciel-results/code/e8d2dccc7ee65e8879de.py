"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    """
    Pack 32 circles in the unit square maximizing the sum of radii.
    
    Strategy: Position-based dynamics / elastic collision simulation.
    1. Initialize circles in a random or grid-like pattern with small radii.
    2. Iteratively expand radii and resolve collisions by projecting positions.
    3. Use a time budget to stop before the hard limit.
    """
    start_time = time.time()
    time_limit = 100.0  # Leave safety margin before 120s limit (and 150s kill)
    
    # Initialization: Hexagonal packing is a good start for local optimization
    # We'll create a grid of potential positions and pick N, or just use random + relaxation.
    # Let's try a structured initialization: a 6x6 grid (36 cells), pick 32.
    # Actually, a simpler robust approach: Random initialization with many restarts.
    
    best_score = 0.0
    best_circles = None
    
    def score(circles):
        # circles: (N, 3) array of [x, y, r]
        # Check validity:
        # 1. Inside [0,1]^2
        # 2. No overlap
        # If invalid, return -inf (or a very small number)
        # But in PBD, we enforce constraints, so it should always be valid.
        # However, numerical errors might cause slight violations.
        # The evaluator likely checks validity strictly.
        # Let's compute the sum of radii.
        return np.sum(circles[:, 2])

    def is_valid(circles):
        # circles: (N, 3)
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        # Boundary check
        if np.any(x < 0) or np.any(x > 1) or np.any(y < 0) or np.any(y > 1):
            return False
        # Overlap check
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                if dist < r[i] + r[j] - 1e-9:
                    return False
        return True

    def solve_pbd(init_circles, max_iters=2000):
        """
        Position Based Dynamics solver.
        init_circles: (N, 3) array [x, y, r]
        Returns: optimized circles, score
        """
        # Make a copy
        circles = init_circles.copy()
        x = circles[:, 0].copy()
        y = circles[:, 1].copy()
        r = circles[:, 2].copy()
        
        # We want to maximize sum(r).
        # Strategy: 
        # 1. Start with small r.
        # 2. Increase all r by a small delta.
        # 3. Resolve collisions by moving centers.
        # 4. Repeat until no more growth possible or time runs out.
        
        # Initial radius: very small
        r[:] = 0.01
        
        # Growth step
        dr = 0.005
        
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            
            # 1. Grow
            r += dr
            
            # 2. Resolve constraints (iterations for stability)
            # We use a few iterations of constraint projection per growth step
            for _ in range(20):
                # Boundary constraints
                # Push centers inward if they go out of bounds
                # x_i - r_i >= 0 => x_i >= r_i
                # x_i + r_i <= 1 => x_i <= 1 - r_i
                # If x_i < r_i: x_i = r_i
                # If x_i > 1 - r_i: x_i = 1 - r_i
                
                # This is a simple projection. For better results, we should distribute the error.
                # But simple projection is fast.
                
                # Pairwise collisions
                for i in range(N):
                    for j in range(i + 1, N):
                        dx = x[j] - x[i]
                        dy = y[j] - y[i]
                        dist = np.sqrt(dx*dx + dy*dy)
                        min_dist = r[i] + r[j]
                        
                        if dist < min_dist:
                            if dist == 0:
                                # Random push
                                dx = 0.01
                                dy = 0.01
                                dist = np.sqrt(dx*dx + dy*dy)
                            
                            # Overlap amount
                            overlap = min_dist - dist
                            
                            # Normalize direction
                            nx = dx / dist
                            ny = dy / dist
                            
                            # Move apart by half the overlap each
                            # Weight by inverse radius? Or equal?
                            # Equal weight is standard for PBD if masses are equal.
                            # Here, larger circles are "heavier"? 
                            # Let's try equal weight first.
                            shift_x = 0.5 * overlap * nx
                            shift_y = 0.5 * overlap * ny
                            
                            x[i] -= shift_x
                            y[i] -= shift_y
                            x[j] += shift_x
                            y[j] += shift_y
                
                # Boundary projection
                for i in range(N):
                    # Left
                    if x[i] < r[i]:
                        x[i] = r[i]
                    # Right
                    if x[i] > 1.0 - r[i]:
                        x[i] = 1.0 - r[i]
                    # Bottom
                    if y[i] < r[i]:
                        y[i] = r[i]
                    # Top
                    if y[i] > 1.0 - r[i]:
                        y[i] = 1.0 - r[i]
                
                # Check if converged (no movement)
                # Optional: break inner loop if movement is tiny
        
        circles[:, 0] = x
        circles[:, 1] = y
        circles[:, 2] = r
        
        # Final validation and cleanup
        # Ensure strict validity
        if not is_valid(circles):
            # Try to shrink slightly to fix numerical issues
            for i in range(N):
                circles[i, 2] *= 0.999
            if not is_valid(circles):
                return None, 0.0
        
        return circles, score(circles)

    # Multiple restarts with different initializations
    np.random.seed(42)
    
    # Restart 1: Random initialization
    for restart in range(5):
        if time.time() - start_time > time_limit:
            break
        
        # Random positions
        x = np.random.rand(N)
        y = np.random.rand(N)
        r = np.zeros(N)
        init_circles = np.column_stack((x, y, r))
        
        res_circles, res_score = solve_pbd(init_circles, max_iters=1000)
        
        if res_circles is not None:
            if res_score > best_score:
                best_score = res_score
                best_circles = res_circles.copy()
    
    # Restart 2: Grid initialization (Hexagonal-like)
    # Create a grid of 6x6 points
    grid_x = []
    grid_y = []
    side = 6
    for i in range(side):
        for j in range(side):
            grid_x.append((i + 0.5) / side)
            grid_y.append((j + 0.5) / side)
    
    # Pick N points from the grid? 
    # Or just use the first N?
    # Better: Use a jittered grid.
    for restart in range(2):
        if time.time() - start_time > time_limit:
            break
        
        # Jittered grid
        x = np.array(grid_x[:N]) + np.random.rand(N) * 0.1
        y = np.array(grid_y[:N]) + np.random.rand(N) * 0.1
        # Clip to [0,1]
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        r = np.zeros(N)
        init_circles = np.column_stack((x, y, r))
        
        res_circles, res_score = solve_pbd(init_circles, max_iters=1000)
        
        if res_circles is not None:
            if res_score > best_score:
                best_score = res_score
                best_circles = res_circles.copy()

    # If we didn't find a valid solution (shouldn't happen), fallback to seed
    if best_circles is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
    
    return best_circles
# EVOLVE-BLOCK-END
