"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Construct 200 equal, non-overlapping circles in the unit square [-0.5, 0.5]^2
    maximizing the common radius.
    
    Strategy: Start with a hexagonal packing (better than square grid) and use
    gradient-free optimization to improve the packing by maximizing the minimum
    distance between circles (and to boundaries).
    """
    # Start with a hexagonal-like arrangement
    # For 200 circles, we can use a pattern with rows of 14 or 15 circles
    # Let's try to create a better initial guess using a hexagonal lattice
    
    # Generate initial positions using a hexagonal grid pattern
    positions = []
    # Use a hexagonal packing pattern
    # Row spacing: r * sqrt(3) where r is the radius
    # But we don't know r yet, so let's use a reasonable initial guess
    
    # Initial guess: hexagonal lattice
    # For a hexagonal lattice, the distance between adjacent centers is 2r
    # In a hexagonal lattice, horizontal spacing is 2r, vertical spacing is 2r*sqrt(3)/2 = r*sqrt(3)
    # Alternate rows are offset by r
    
    # Let's estimate: if we have ~14-15 columns and ~14-15 rows
    # With hexagonal packing, the effective density is higher
    
    # Start with a square grid as initial guess, then optimize
    m = int(np.ceil(np.sqrt(N)))
    # Create a square grid
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    x0 = grid[:N].copy()
    
    # Define the objective function: we want to maximize the minimum distance
    # between any two circles, and also ensure they're inside the square
    # The radius r = min(min_{i<j} |c_i - c_j|/2, min_i min_k (0.5 - |c_ik|))
    
    def negative_radius(x):
        """Return negative of the achievable radius (to minimize)."""
        x = x.reshape(-1, 2)
        # Check boundaries
        # Distance to each boundary
        dist_to_left = 0.5 - x[:, 0]
        dist_to_right = x[:, 0] + 0.5
        dist_to_bottom = 0.5 - x[:, 1]
        dist_to_top = x[:, 1] + 0.5
        
        # All must be positive for valid packing
        min_boundary = np.min(np.stack([dist_to_left, dist_to_right, dist_to_bottom, dist_to_top], axis=1))
        
        # If any point is outside, return a large penalty
        if min_boundary < 0:
            return 1.0  # Large positive value (we're minimizing negative radius)
        
        # Compute all pairwise distances
        # Use broadcasting for efficiency
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        # Set diagonal to infinity
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # The radius is half the minimum of (min pair dist, min boundary dist)
        r = min(min_pair_dist / 2.0, min_boundary)
        
        return -r
    
    # Use Nelder-Mead or Powell optimization
    # Given the dimension (400), this might be slow, so let's try a few iterations
    # Actually, 400 dimensions is too high for direct optimization.
    
    # Alternative approach: Use a smarter initial configuration
    # Let's try a hexagonal packing as the initial guess
    
    # Hexagonal packing:
    # Row 0: 14 circles
    # Row 1: 15 circles (offset)
    # Row 2: 14 circles
    # etc.
    
    # Let's compute a hexagonal arrangement
    # For 200 circles, let's try different row configurations
    
    # Try to find a good hexagonal arrangement
    # The hexagonal packing has a density of pi/(2*sqrt(3)) ≈ 0.9069
    # Square grid has density pi/4 ≈ 0.7854
    
    # Let's create a hexagonal pattern
    positions = []
    
    # Try different numbers of rows
    best_positions = None
    best_score = -np.inf
    
    for n_rows in range(10, 25):
        # Distribute 200 circles among n_rows
        # Alternate between ceil(200/n_rows) and floor(200/n_rows)
        base = 200 // n_rows
        extra = 200 % n_rows
        
        row_sizes = []
        for i in range(n_rows):
            if i < extra:
                row_sizes.append(base + 1)
            else:
                row_sizes.append(base)
        
        # Create hexagonal positions
        pos = []
        for row_idx, n_in_row in enumerate(row_sizes):
            # Vertical position
            y = -0.5 + (row_idx + 0.5) / n_rows
            
            # Horizontal positions
            if row_idx % 2 == 0:
                # Even row: start at some x
                if n_in_row > 0:
                    xs = np.linspace(-0.5 + 0.5/n_in_row, 0.5 - 0.5/n_in_row, n_in_row)
                else:
                    xs = []
            else:
                # Odd row: offset by half the spacing
                if n_in_row > 0:
                    spacing = 1.0 / n_in_row
                    offset = spacing / 2
                    xs = np.linspace(-0.5 + spacing/2 + offset, 0.5 - spacing/2 + offset, n_in_row)
                    # Make sure they're within bounds
                    xs = np.clip(xs, -0.5, 0.5)
                else:
                    xs = []
            
            for x in xs:
                pos.append([x, y])
        
        if len(pos) == N:
            pos = np.array(pos)
            # Evaluate this configuration
            score = -negative_radius(pos.ravel())
            if score > best_score:
                best_score = score
                best_positions = pos.copy()
    
    if best_positions is not None:
        return best_positions
    
    # Fallback to square grid
    return x0
# EVOLVE-BLOCK-END
