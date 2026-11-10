"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ACO algorithm.
    
    The heuristic favors pairs of items (i, j) that, when placed together,
    leave a small amount of unused space in the bin. This is based on the
    'inverse gap' strategy which encourages tight packing.
    
    Parameters:
    demands: array-like of item sizes.
    capacity: float, the capacity of each bin.
    
    Returns:
    numpy.ndarray: An n x n matrix where entry [i, j] is the attractiveness
                   of placing item j after item i.
    """
    demands = np.asarray(demands, dtype=float)
    n = len(demands)
    
    # Create a matrix of pairwise sums: sum[i, j] = demands[i] + demands[j]
    sums = demands[:, None] + demands[None, :]
    
    # Calculate the gap (unused space) for each pair
    gaps = capacity - sums
    
    # Initialize the heuristic matrix with a very small value for invalid pairs
    eta = np.full((n, n), 1e-9, dtype=float)
    
    # Identify valid pairs where the sum is within capacity
    # We add a small epsilon to avoid division by zero if gap is 0, 
    # but we want gap=0 to be the most attractive.
    # Using gap + 1 ensures that gap=0 -> 1.0, gap=100 -> 1/101.
    valid_mask = sums <= capacity
    
    # Calculate inverse gap for valid pairs
    # We use (gap + 1) to keep values bounded and positive.
    # A smaller gap results in a larger eta.
    eta[valid_mask] = 1.0 / (gaps[valid_mask] + 1.0)
    
    return eta
# EVOLVE-BLOCK-END
