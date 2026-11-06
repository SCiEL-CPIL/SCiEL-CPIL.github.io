"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def design(problem):
    """Design a 32x32 photonic crystal unit cell with four-fold symmetric square rings."""
    size = problem["pixels"]
    
    # Define ring structure using Chebyshev distance from center
    # Center is at (size/2 - 0.5, size/2 - 0.5)
    cx = cy = (size - 1) / 2.0
    
    # Create coordinate grids
    xs = np.arange(size) - cx
    ys = np.arange(size) - cy
    X, Y = np.meshgrid(xs, ys)
    
    # Chebyshev distance (square rings)
    dist = np.maximum(np.abs(X), np.abs(Y))
    
    # Define ring boundaries - we'll use a set of radial positions
    # and assign each ring as dielectric (1) or air (0)
    # Start with a promising configuration based on known photonic crystal designs
    
    # Ring boundaries in pixel units (Chebyshev distance)
    # We'll define rings from center outward
    ring_boundaries = [0, 2, 4, 6, 8, 10, 12, 14, 16]
    
    # Assignment pattern: alternate dielectric and air
    # Index 0: center, 1: first ring, etc.
    # Try a pattern that creates a "hole" in the center surrounded by dielectric
    assignments = [0, 1, 0, 1, 0, 1, 0, 1, 0]
    
    # Build the cell
    cell = np.zeros((size, size), dtype=int)
    
    for i in range(len(ring_boundaries) - 1):
        inner = ring_boundaries[i]
        outer = ring_boundaries[i + 1]
        mask = (dist >= inner) & (dist < outer)
        cell[mask] = assignments[i]
    
    # Handle the outermost region
    outermost = ring_boundaries[-1]
    cell[dist >= outermost] = assignments[-1]
    
    # Convert to nested list
    return cell.tolist()
# EVOLVE-BLOCK-END
