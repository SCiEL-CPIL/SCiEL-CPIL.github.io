"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Place N points at the vertices of a regular N-gon.
    This maximizes the minimum distance between points on the boundary,
    which generally maximizes the minimum triangle area relative to the hull area.
    """
    # Generate angles for N points evenly spaced around the circle
    angles = 2 * np.pi * np.arange(N) / N
    
    # Calculate coordinates
    x = np.cos(angles)
    y = np.sin(angles)
    
    # Stack into (N, 2) array
    points = np.column_stack((x, y))
    
    return points
# EVOLVE-BLOCK-END
