# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem. Place n squares with pairwise disjoint
    interiors inside the unit square [0,1]^2 so as to MAXIMIZE the sum of their
    side lengths. Here n = 10.
    
    Strategy: Physics-based relaxation (Simulated Annealing) with repulsive forces,
    multiple diverse initializations, and adaptive step sizes.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 585.0  # Safety margin before 600s limit
    
    best_centers = None
    best_sides = None
    best_score = -np.inf
    
    def evaluate(centers, sides):
        """
        Calculate the score (sum of sides) if valid, or a penalized score if invalid.
        """
        score = np.sum(sides)
        tol = 1e-7
        penalty = 0.0
        
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if s <= 0:
                penalty += 1000.0
                continue
            # Bounds penalty
            if cx - s/2 < -tol:
                penalty += 500.0 * (-tol - (cx - s/2))
            if cx + s/2 > 1 + tol:
                penalty += 500.0 * ((cx + s/2) - (1 + tol))
            if cy - s/2 < -tol:
                penalty += 500.0 * (-tol - (cy - s/2))
            if cy + s/2 > 1 + tol:
                penalty += 500.0 * ((cy + s/2) - (1 + tol))
        
        # Overlap penalty
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                dx = abs(centers[i,0] - centers[j,0])
                dy = abs(centers[i,1] - centers[j,1])
                if dx < t and dy < t:
                    overlap_x = t - dx
                    overlap_y = t - dy
                    penalty += 500.0 * overlap_x * overlap_y
        
        return score - penalty

    def is_valid(centers, sides):
        tol = 1e-7
        if np.any(sides <= 0):
            return False
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s/2 < -tol or cx + s/2 > 1 + tol:
                return False
            if cy - s/2 < -tol or cy + s/2 > 1 + tol:
                return False
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                if abs(centers[i,0] - centers[j,0]) < t and abs(centers[i,1] - centers[j,1]) < t:
                    return False
        return True

    def optimize_single_run(centers_init, sides_init, seed):
        random.seed(seed)
        np.random.seed(seed)
        
        centers = centers_init.copy()
        sides = sides_init.copy()
        
        current_score = evaluate(centers, sides)
        
        # Simulated Annealing parameters
        T = 0.08  # Initial temperature
        alpha = 0.997  # Cooling rate
        steps_per_temp = 30
        
        # Adaptive step size
        step_scale = 0.03
        
        while time.time() < deadline:
            for _ in range(steps_per_temp):
                if time.time() > deadline:
                    break
                    
                # Select a random square to move
                i = random.randint(0, n-1)
                
                # Perturbation with adaptive step size
                dx = random.gauss(0, step_scale)
                dy = random.gauss(0, step_scale)
                ds = random.gauss(0, step_scale * 0.5)
                
                new_centers = centers.copy()
                new_sides = sides.copy()
                
                new_centers[i, 0] += dx
                new_centers[i, 1] += dy
                new_sides[i] += ds
                
                # Clamp sides to positive
                if new_sides[i] <= 0:
                    new_sides[i] = 1e-6
                    
                new_score = evaluate(new_centers, new_sides)
                
                # Metropolis criterion
                if new_score > current_score:
                    centers = new_centers
                    sides = new_sides
                    current_score = new_score
                else:
                    delta = current_score - new_score
                    if delta < 0:
                        accept = True
                    else:
                        prob = np.exp(-delta / T)
                        accept = random.random() < prob
                    
                    if accept:
                        centers = new_centers
                        sides = new_sides
                        current_score = new_score
                        
            T *= alpha
            step_scale *= 0.999  # Slowly shrink step size
            
            # If T is too low, stop
            if T < 1e-7:
                break
                
        # Final polishing phase: try small adjustments
        polish_steps = 200
        polish_scale = 0.005
        for _ in range(polish_steps):
            if time.time() > deadline:
                break
            i = random.randint(0, n-1)
            dx = random.gauss(0, polish_scale)
            dy = random.gauss(0, polish_scale)
            ds = random.gauss(0, polish_scale * 0.3)
            
            new_centers = centers.copy()
            new_sides = sides.copy()
            new_centers[i, 0] += dx
            new_centers[i, 1] += dy
            new_sides[i] += ds
            
            if new_sides[i] <= 0:
                new_sides[i] = 1e-6
                
            new_score = evaluate(new_centers, new_sides)
            if new_score > current_score:
                centers = new_centers
                sides = new_sides
                current_score = new_score
        
        # Final validation
        if is_valid(centers, sides):
            return centers, sides, np.sum(sides)
        else:
            # Try to fix by shrinking sides slightly if invalid
            for scale in [0.999, 0.998, 0.995, 0.99, 0.98, 0.95]:
                test_sides = sides * scale
                if is_valid(centers, test_sides):
                    return centers, test_sides, np.sum(test_sides)
            return None, None, -np.inf

    # Define multiple diverse initial configurations
    def make_init_3x3_plus1():
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        side_grid = 1.0 / 3.0
        for i in range(9):
            col = i % 3
            row = i // 3
            centers[i] = [(col + 0.5) * side_grid, (row + 0.5) * side_grid]
            sides[i] = side_grid
        # 10th square: small in top-right corner
        centers[9] = [0.95, 0.95]
        sides[9] = 0.1
        return centers, sides

    def make_init_2x5():
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        side_grid = 1.0 / 5.0
        for i in range(n):
            col = i % 5
            row = i // 5
            centers[i] = [(col + 0.5) * side_grid, (row + 0.5) * side_grid]
            sides[i] = side_grid
        return centers, sides

    def make_init_4x3_partial():
        """4 columns, 3 rows, but only 10 squares placed"""
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        side_grid = 1.0 / 4.0
        for i in range(n):
            col = i % 4
            row = i // 4
            centers[i] = [(col + 0.5) * side_grid, (row + 0.5) * side_grid]
            sides[i] = side_grid
        return centers, sides

    def make_init_random(seed):
        """Random valid initial configuration"""
        random.seed(seed)
        np.random.seed(seed)
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        # Place squares in random non-overlapping positions
        # Start with small squares and grow
        for i in range(n):
            # Try to place square i
            max_attempts = 100
            placed = False
            for attempt in range(max_attempts):
                s = random.uniform(0.05, 0.25)
                cx = random.uniform(s/2, 1 - s/2)
                cy = random.uniform(s/2, 1 - s/2)
                # Check if this overlaps with existing squares
                valid = True
                for j in range(i):
                    t = (s + sides[j]) / 2
                    if abs(cx - centers[j,0]) < t and abs(cy - centers[j,1]) < t:
                        valid = False
                        break
                if valid:
                    centers[i] = [cx, cy]
                    sides[i] = s
                    placed = True
                    break
            if not placed:
                # Fallback: place in a corner
                centers[i] = [0.05, 0.05]
                sides[i] = 0.05
        return centers, sides

    # Run multiple restarts with different initializations
    initializations = [
        ("3x3+1", make_init_3x3_plus1()),
        ("2x5", make_init_2x5()),
        ("4x3", make_init_4x3_partial()),
    ]
    
    # Add random initializations
    for rs in [7, 42, 123, 999]:
        initializations.append((f"random_{rs}", make_init_random(rs)))
    
    # Shuffle to diversify
    random.shuffle(initializations)
    
    for idx, (name, (c_init, s_init)) in enumerate(initializations):
        if time.time() > deadline:
            break
        
        seed = 1000 + idx * 777
        c, s, sc = optimize_single_run(c_init, s_init, seed)
        
        if c is not None and sc > best_score:
            best_centers = c
            best_sides = s
            best_score = sc
            print(f"  {name}: score={sc:.6f}", file=__import__('sys').stderr)

    # If no valid solution found, fallback to grid
    if best_centers is None:
        k = int(np.ceil(np.sqrt(n)))
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
        best_sides = np.full(n, side, dtype=np.float64)

    return best_centers, best_sides


# EVOLVE-BLOCK-END
