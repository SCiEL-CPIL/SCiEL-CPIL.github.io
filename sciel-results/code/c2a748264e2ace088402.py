import time
import sys

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 45.0 # 45s buffer within 60s limit
    
    # Strategy: Use the 6x5 grid which yields a higher multiplicity of small doubled areas
    # compared to the two-line configuration.
    # The seed program (6x5 grid) has a score of 1.0, while the two-line config has 0.6583.
    # We retain the grid structure and add validation/safety checks from the current program.
    
    points = [(x, y) for x in range(6) for y in range(5)]
    
    # Verify distinctness and bounds
    if len(set(points)) != 30:
        # Fallback to a valid configuration if logic fails (should not happen)
        # Using a simple grid that is known to be valid
        return [(i, 0) for i in range(15)] + [(i, 1) for i in range(15)]
        
    for p in points:
        if not (-30 <= p[0] <= 30 and -30 <= p[1] <= 30):
            # Fallback
            return [(i, 0) for i in range(15)] + [(i, 1) for i in range(15)]

    # Check deadline before returning
    if time.time() > deadline:
        return [(i, 0) for i in range(15)] + [(i, 1) for i in range(15)]
    
    return points
