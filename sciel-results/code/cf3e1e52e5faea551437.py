# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools

def squares_in_square10() -> tuple:
    """
    Place 10 axis-aligned squares inside the unit square [0,1]^2 to maximize
    the sum of their side lengths, subject to non-overlapping interiors.
    
    Strategy:
    1. Enumerate all permutations of 10 squares (3.6M).
    2. For each permutation, determine a "shelf" packing layout (2 rows of 5).
       - Squares are assigned to slots in a serpentine or grid pattern.
       - Side lengths are determined by the tightest constraint in each row/col.
    3. To handle the complexity, we use a smarter constructive heuristic that
       doesn't require full enumeration of permutations for the initial layout,
       but rather explores a diverse set of "templates" (topologies) directly.
       
    Actually, the current serpentine approach is a subset of this. 
    Let's try a different family: **Simulated Annealing with Continuous Coordinates**.
    - State: (centers, sides)
    - Energy: -sum(sides) + penalty(violations)
    - Moves: 
        1. Move a center slightly.
        2. Change a side length slightly.
        3. Swap two squares.
    - This allows continuous exploration of the configuration space, escaping 
      the discrete "grid" traps of the shelf method.
    """
    n = 10
    tol = 1e-7
    deadline = time.time() + 610  # Leave 20s margin for safety

    def is_feasible(centers, sides):
        """Check if the configuration is valid."""
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if not np.isfinite(cx) or not np.isfinite(cy) or not np.isfinite(s):
                return False
            if s <= 0:
                return False
            if cx - s / 2 < -tol or cx + s / 2 > 1 + tol:
                return False
            if cy - s / 2 < -tol or cy + s / 2 > 1 + tol:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                if abs(centers[i][0] - centers[j][0]) < t and abs(centers[i][1] - centers[j][1]) < t:
                    return False
        return True

    def penalty(centers, sides):
        """Calculate the penalty for constraint violations."""
        pen = 0.0
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if s <= 0:
                return 1e10
            # Boundary violation
            if cx - s / 2 < -tol:
                pen += (-tol - (cx - s / 2)) ** 2
            if cx + s / 2 > 1 + tol:
                pen += ((cx + s / 2) - (1 + tol)) ** 2
            if cy - s / 2 < -tol:
                pen += (-tol - (cy - s / 2)) ** 2
            if cy + s / 2 > 1 + tol:
                pen += ((cy + s / 2) - (1 + tol)) ** 2
        
        # Overlap violation
        for i in range(n):
            for j in range(i + 1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                dx = abs(centers[i][0] - centers[j][0])
                dy = abs(centers[i][1] - centers[j][1])
                if dx < t and dy < t:
                    # Penalty proportional to depth of overlap
                    pen += (t - dx) ** 2 + (t - dy) ** 2
        
        return pen

    def objective(centers, sides):
        """Objective function: -sum(sides) + penalty."""
        return -np.sum(sides) + penalty(centers, sides)

    def simulated_annealing(centers, sides, rng, max_iters=5000, T_start=0.1, T_end=1e-5):
        """
        Run Simulated Annealing to optimize the configuration.
        """
        best_centers = centers.copy()
        best_sides = sides.copy()
        best_score = np.sum(sides)
        
        if not is_feasible(best_centers, best_sides):
             # Try to repair by shrinking
             best_sides *= 0.5
             if not is_feasible(best_centers, best_sides):
                 best_sides *= 0.5
                 if not is_feasible(best_centers, best_sides):
                     return best_centers, best_sides, -np.inf
             best_score = np.sum(best_sides)
        
        current_centers = centers.copy()
        current_sides = sides.copy()
        current_obj = objective(current_centers, current_sides)
        
        T = T_start
        cooling_rate = (T_end / T_start) ** (1.0 / max_iters)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            # Generate a move
            move_type = rng.integers(0, 3)
            
            if move_type == 0:
                # Move a center
                i = rng.integers(0, n)
                d = rng.integers(0, 2)
                step = rng.normal(0, 0.05)
                new_centers = current_centers.copy()
                new_centers[i, d] += step
                new_sides = current_sides.copy()
            elif move_type == 1:
                # Change a side
                i = rng.integers(0, n)
                factor = 1.0 + rng.normal(0, 0.05)
                factor = np.clip(factor, 0.5, 2.0)
                new_sides = current_sides.copy()
                new_sides[i] *= factor
                new_sides[i] = np.clip(new_sides[i], 0.01, 1.0)
                new_centers = current_centers.copy()
            else:
                # Swap two squares (positions and sizes)
                i, j = rng.choice(n, 2, replace=False)
                new_centers = current_centers.copy()
                new_sides = current_sides.copy()
                new_centers[i], new_centers[j] = current_centers[j], current_centers[i]
                new_sides[i], new_sides[j] = current_sides[j], current_sides[i]

            # Ensure boundaries are respected for new centers
            for i in range(n):
                s = new_sides[i]
                new_centers[i, 0] = np.clip(new_centers[i, 0], s/2 - tol, 1 - s/2 + tol)
                new_centers[i, 1] = np.clip(new_centers[i, 1], s/2 - tol, 1 - s/2 + tol)

            new_obj = objective(new_centers, new_sides)
            
            # Acceptance probability
            delta = new_obj - current_obj
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_centers = new_centers
                current_sides = new_sides
                current_obj = new_obj
                
                # Check if this is a new best feasible solution
                if is_feasible(current_centers, current_sides):
                    score = np.sum(current_sides)
                    if score > best_score:
                        best_score = score
                        best_centers = current_centers.copy()
                        best_sides = current_sides.copy()
            
            T *= cooling_rate
        
        return best_centers, best_sides, best_score

    def grid_init(rng):
        """Generate a random grid-like initial configuration."""
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        
        # Try a few random grid topologies
        # 2x5, 5x2, 3x3+1, etc.
        rows = rng.choice([2, 3, 4, 5])
        cols = int(np.ceil(n / rows))
        
        # Assign squares to grid cells
        for i in range(n):
            r = i // cols
            c = i % cols
            
            # Cell size
            cell_w = 1.0 / cols
            cell_h = 1.0 / rows
            
            # Center in cell
            cx = (c + 0.5) * cell_w
            cy = (r + 0.5) * cell_h
            
            # Add some randomness
            cx += rng.normal(0, cell_w * 0.1)
            cy += rng.normal(0, cell_h * 0.1)
            
            # Clip to boundaries
            s = min(cell_w, cell_h) * 0.9
            s = np.clip(s, 0.01, 1.0)
            
            cx = np.clip(cx, s/2 - tol, 1 - s/2 + tol)
            cy = np.clip(cy, s/2 - tol, 1 - s/2 + tol)
            
            centers[i] = [cx, cy]
            sides[i] = s
            
        return centers, sides

    def random_init(rng):
        """Generate a random initial configuration."""
        centers = rng.uniform(0.1, 0.9, (n, 2))
        sides = rng.uniform(0.1, 0.3, n)
        
        # Clip to boundaries
        for i in range(n):
            s = sides[i]
            centers[i, 0] = np.clip(centers[i, 0], s/2 - tol, 1 - s/2 + tol)
            centers[i, 1] = np.clip(centers[i, 1], s/2 - tol, 1 - s/2 + tol)
            
        return centers, sides

    best_centers = None
    best_sides = None
    best_score = -np.inf

    rng = np.random.default_rng(42)

    # Run multiple restarts with different initializations
    for restart in range(5):
        if time.time() > deadline:
            break
            
        if restart == 0:
            c0, s0 = grid_init(rng)
        elif restart == 1:
            c0, s0 = random_init(rng)
        else:
            # Mix of grid and random
            c0, s0 = grid_init(rng)
            # Perturb
            c0 += rng.normal(0, 0.05, (n, 2))
            s0 *= (1.0 + rng.normal(0, 0.1, n))
            s0 = np.clip(s0, 0.01, 1.0)
            for i in range(n):
                s = s0[i]
                c0[i, 0] = np.clip(c0[i, 0], s/2 - tol, 1 - s/2 + tol)
                c0[i, 1] = np.clip(c0[i, 1], s/2 - tol, 1 - s/2 + tol)

        # Run SA
        c_loc, s_loc, score_loc = simulated_annealing(c0, s0, rng, max_iters=3000)
        
        if score_loc > best_score:
            best_score = score_loc
            best_centers = c_loc
            best_sides = s_loc

    # Fallback
    if best_centers is None:
        centers = np.column_stack((np.linspace(0.1, 0.9, 5).repeat(2), 
                                   np.tile(np.linspace(0.1, 0.9, 2), 5)))
        sides = np.full(n, 0.2)
        return centers, sides

    return best_centers, best_sides

# EVOLVE-BLOCK-END
