"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def priority(vector, n):
    # Structural change: prioritize vectors with balanced coordinate distributions.
    # Vectors whose coordinates are spread evenly across {0,1,2} create "generic"
    # 3-term APs that are easier to avoid during greedy insertion, allowing more
    # vectors to be accepted. Degenerate vectors (all-same, or two-valued) create
    # structured APs that conflict more readily.
    #
    # Priority = negative coordinate standard deviation (more uniform = higher priority),
    # broken by negative sum (smaller sum = higher priority), then lexicographic.
    coords = list(vector)
    mean = sum(coords) / n
    variance = sum((c - mean) ** 2 for c in coords) / n
    std = math.sqrt(variance)
    total = sum(coords)
    # Higher priority = more uniform coords, then smaller sum
    # Return a scalar: combine std and sum into one number
    # Use large multiplier for std to dominate
    return -std * 1000.0 - total
# EVOLVE-BLOCK-END
