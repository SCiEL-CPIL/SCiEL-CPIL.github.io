# EVOLVE-BLOCK-START
import time

def construct():
    """
    Returns a list of 32 integer pairs [a, b] representing points on a triangular lattice
    a + b*omega (where omega = exp(i*pi/3)) that form an optimal Harborth configuration.
    This configuration has exactly 76 pairs at unit distance, which is the theoretical maximum.
    """
    
    # Deadline guard: Ensure we return within time limits.
    # Since this is a static lookup, it is instant, but we include the structure
    # as per directive.
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit

    # The optimal configuration is a subset of the hexagon of radius 3.
    # The full hexagon of radius 3 has 37 points.
    # We remove 5 specific points to get 32 points while maintaining 76 edges.
    
    # Pre-computed list of the 32 points.
    # This specific set is known to achieve the maximum of 76 edges.
    points = [
        [-3, 0], [-3, 1], [-3, 2], [-3, 3],
        [-2, -1], [-2, 0], [-2, 1], [-2, 2], [-2, 3],
        [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2], [-1, 3],
        [0, -3], [0, -2], [0, -1], [0, 0], [0, 1], [0, 2],
        [1, -3], [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
        [2, -3], [2, -2], [2, -1], [2, 0],
        [3, -3], [3, -2], [3, -1]
    ]
    
    # Note: The list above has exactly 32 points.
    # Verification of count:
    # Row -3: 4 points (b=0..3)
    # Row -2: 5 points (b=-1..3)
    # Row -1: 6 points (b=-2..3)
    # Row 0: 6 points (b=-3..2) -> Wait, let's re-verify the hardcoded list against the previous one.
    
    # Previous remove_set: (3,0), (2,1), (1,2), (0,3), (-1,3)
    # Previous full hexagon points (37):
    # a=-3: b in [0,1,2,3] (4)
    # a=-2: b in [-1,0,1,2,3] (5)
    # a=-1: b in [-2,-1,0,1,2,3] (6)
    # a=0:  b in [-3,-2,-1,0,1,2,3] (7)
    # a=1:  b in [-3,-2,-1,0,1,2] (6) -> (1,3) is not in hex? No, hex radius 3 includes |a|+|b|+|a+b| <= 6?
    # Let's stick to the explicit list from the previous correct solution to avoid errors.
    
    # Re-inserting the exact 32 points from the previous version to ensure correctness.
    points = [
        [-3, 0], [-3, 1], [-3, 2], [-3, 3],
        [-2, -1], [-2, 0], [-2, 1], [-2, 2], [-2, 3],
        [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
        [0, -3], [0, -2], [0, -1], [0, 0], [0, 1], [0, 2],
        [1, -3], [1, -2], [1, -1], [1, 0], [1, 1],
        [2, -3], [2, -2], [2, -1], [2, 0],
        [3, -3], [3, -2], [3, -1]
    ]
    
    # Check if we hit deadline (should not happen)
    if time.time() > deadline:
        return points

    return points
# EVOLVE-BLOCK-END
