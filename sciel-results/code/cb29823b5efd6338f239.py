"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import random
import math

def construct():
    """
    Pack 26 circles in the unit square maximizing sum of radii.
    Uses Simulated Annealing (SA) with position and radius moves.
    """
    n = N
    
    # Helper function to check validity and compute penalty/value
    def get_value_and_validity(circles):
        """
        circles: np.array of shape (n, 3) [x, y, r]
        Returns: (value, is_valid)
        value: sum of radii if valid, else -infinity or large penalty.
        """
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        # Check boundary
        if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9):
            return -np.inf, False
        if np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
            return -np.inf, False
            
        # Check overlaps
        # Vectorized pairwise distance calculation
        # Create meshgrid of indices
        i_idx, j_idx = np.triu_indices(n, k=1)
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dists = np.sqrt(dx*dx + dy*dy)
        min_dists = r[i_idx] + r[j_idx]
        
        if np.any(dists < min_dists - 1e-9):
            return -np.inf, False
            
        return np.sum(r), True

    # Initialization: Create a valid grid-based configuration
    # 5x5 grid = 25 spots, we need 26. 
    # Let's use a 5x5 grid and place the 26th circle in a gap or use a 6x6 grid with smaller circles?
    # Better: Start with a simple valid random configuration or a grid.
    # A 5x5 grid of circles with radius 0.1 fits in [0.1, 0.9]. 
    # Let's generate a 5x5 grid and one extra circle, then relax.
    
    # Start with a 5x5 grid (25 circles) + 1 extra.
    # Actually, let's just start with random valid circles.
    # To ensure a good start, use a grid of 25 circles and try to fit the 26th.
    
    # Grid positions
    grid_side = 5
    step = 1.0 / grid_side
    positions = []
    for i in range(grid_side):
        for j in range(grid_side):
            positions.append([(i + 0.5) * step, (j + 0.5) * step])
    
    # We have 25 positions. Add a 26th at a random location, initially with small radius.
    # Or better, start with 26 random circles that are non-overlapping.
    # Let's use a simple random valid initialization.
    
    circles = np.zeros((n, 3))
    
    # Try to place circles one by one with random radii
    # This is slow. Let's use a grid for the first 25 and random for the 26th?
    # No, let's just use a grid of 26? 5x5 is 25. 6x6 is 36.
    # Let's use a 5x5 grid for 25 circles, and place the 26th in the center? No, center is occupied.
    # Let's just use a random valid configuration generator.
    
    def random_valid_init():
        # Start with small radii
        r = np.full(n, 0.01)
        x = np.zeros(n)
        y = np.zeros(n)
        
        # Place them randomly, ensuring no overlap with existing ones
        # This is O(N^2) per placement, N=26, so 26^2 * 26 ~ 17000 ops, very fast.
        
        # Shuffle order to avoid bias
        order = np.random.permutation(n)
        
        for idx in order:
            placed = False
            attempts = 0
            while not placed and attempts < 1000:
                attempts += 1
                # Try a random position
                rx = np.random.uniform(0, 1)
                ry = np.random.uniform(0, 1)
                
                # Check boundary
                if rx < r[idx] or rx > 1 - r[idx]: continue
                if ry < r[idx] or ry > 1 - r[idx]: continue
                
                # Check overlap with already placed circles
                overlap = False
                for j in range(n):
                    if j == idx: continue
                    # Only check against circles that have been placed? 
                    # Actually, we are building the set. 
                    # Let's just check against all previously placed.
                    # We need to track which are placed.
                    # Simpler: just check against all others in the current array? 
                    # No, others are not placed yet (zeros).
                    # Let's maintain a list of placed circles.
                    pass
                
                # This approach is getting complicated. 
                # Let's just use the 5x5 grid for 25 circles and add the 26th by splitting a cell or placing it carefully.
                
                # Alternative: Start with 26 circles of radius 0.01 scattered randomly, then grow.
                pass
            
            # Fallback: Just place at grid positions for first 25, and (0.5, 0.5) for 26th? 
            # (0.5, 0.5) is center of 5x5 grid, occupied.
            
            # Let's just use a 6x6 grid and take the first 26?
            # 6x6 grid step = 1/6. Center at (i+0.5)/6.
            # Radius max = 0.5/6 = 0.0833.
            # 26 circles in 6x6 grid (36 spots).
            
            pos = []
            for i in range(6):
                for j in range(6):
                    if len(pos) < n:
                        pos.append([(i + 0.5) / 6.0, (j + 0.5) / 6.0])
            
            x = np.array([p[0] for p in pos])
            y = np.array([p[1] for p in pos])
            r = np.full(n, 0.04) # Small initial radius
            
            # Check if this is valid
            # Distances between neighbors in 6x6 grid are 1/6 ~ 0.166.
            # 2*r = 0.08. 0.08 < 0.166. Valid.
            # Boundary: 0.5/6 = 0.0833. r=0.04. Valid.
            
            return np.column_stack([x, y, r])
        
        return None

    circles = random_valid_init()
    if circles is None:
        # Fallback
        circles = np.column_stack([np.full(n, 0.5), np.full(n, 0.5), np.full(n, 0.01)])
        # This is invalid (overlap), but SA will fix it? No, SA starts from valid.
        # Let's ensure the 6x6 init works.
        pass

    # Simulated Annealing
    T = 0.1
    T_min = 0.0001
    cooling = 0.9995
    max_iter = 50000
    current_value, _ = get_value_and_validity(circles)
    
    best_circles = circles.copy()
    best_value = current_value
    
    rng = np.random.RandomState(42)
    
    for it in range(max_iter):
        # Generate a move
        move_type = rng.random()
        
        new_circles = circles.copy()
        
        if move_type < 0.7:
            # Move a circle
            idx = rng.randint(n)
            dx = rng.normal(0, 0.05)
            dy = rng.normal(0, 0.05)
            new_circles[idx, 0] += dx
            new_circles[idx, 1] += dy
        elif move_type < 0.9:
            # Grow a circle
            idx = rng.randint(n)
            # Try to increase radius
            new_r = new_circles[idx, 2] * (1.0 + rng.uniform(0, 0.1))
            # Cap radius
            max_r = min(new_circles[idx, 0], 1 - new_circles[idx, 0], 
                        new_circles[idx, 1], 1 - new_circles[idx, 1])
            new_r = min(new_r, max_r)
            new_circles[idx, 2] = new_r
        else:
            # Shrink a circle (to allow movement or escape local minima)
            idx = rng.randint(n)
            new_circles[idx, 2] *= (1.0 - rng.uniform(0, 0.1))
            # Ensure minimum radius
            new_circles[idx, 2] = max(new_circles[idx, 2], 0.001)

        # Check validity and value
        new_value, is_valid = get_value_and_validity(new_circles)
        
        if not is_valid:
            # Reject invalid moves immediately? 
            # In SA, we usually only accept valid moves or use a penalty.
            # Since we want to maximize sum of radii of *valid* packs, 
            # and invalid packs have "value" -inf, we should reject them 
            # unless we use a penalty landscape. 
            # Given the strict requirement, rejecting invalid moves is safer 
            # and keeps the chain in the valid space.
            continue
            
        delta = new_value - current_value
        
        # Accept if better, or with probability exp(-delta/T) if worse
        # Note: We are maximizing, so delta > 0 is improvement.
        # Standard SA minimizes energy. Here energy = -value.
        # So delta_energy = -delta_value.
        # Accept if delta_value > 0 or with prob exp(delta_value / T).
        
        if delta > 0:
            circles = new_circles
            current_value = new_value
            if current_value > best_value:
                best_value = current_value
                best_circles = circles.copy()
        else:
            # delta < 0
            prob = math.exp(delta / T) # delta is negative, so exp(neg/T) < 1
            if rng.random() < prob:
                circles = new_circles
                current_value = new_value
        
        # Cooling
        T *= cooling
        if T < T_min:
            T = T_min

    # Final cleanup: try to grow all circles greedily
    # Iterate through circles, try to increase radius as much as possible
    for i in range(n):
        # Try to increase radius
        # Max possible radius based on boundaries
        max_r_bound = min(best_circles[i, 0], 1 - best_circles[i, 0], 
                          best_circles[i, 1], 1 - best_circles[i, 1])
        
        # Binary search for max radius that doesn't overlap
        lo = best_circles[i, 2]
        hi = max_r_bound
        
        # If current radius is already max, skip
        if hi <= lo:
            continue
            
        # Check if we can expand
        # We need to check against all other circles
        # Simple linear search for max r
        # This might be slow if done naively for each circle.
        # Let's just do a few steps of greedy expansion.
        
        for _ in range(10):
            new_r = (lo + hi) / 2
            if new_r <= best_circles[i, 2]:
                break
                
            # Check validity with new_r
            temp_circles = best_circles.copy()
            temp_circles[i, 2] = new_r
            
            _, valid = get_value_and_validity(temp_circles)
            if valid:
                lo = new_r
            else:
                hi = new_r
                
        if lo > best_circles[i, 2]:
            best_circles[i, 2] = lo

    return best_circles

# EVOLVE-BLOCK-END
