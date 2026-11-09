import numpy as np
import time

def moving_sofa_3d_path() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580
    
    # Try a simpler, more direct path
    # The sofa needs to go from (-3,0,0) with identity to (0,6,7) with (90,90,90)
    # Corridor: {x<=0, 0<=y<=1, 0<=z<=1} U {0<=x<=1, 0<=y<=5, 0<=z<=1} U {0<=x<=1, 5<=y<=6, z>=0}
    
    # Strategy: Move along the corridor, keeping the sofa as compact as possible
    # Phase 1: Move from x=-3 to x=0, y=0.5, z=0.5 (center of first corridor)
    # Phase 2: Move from y=0.5 to y=5, x=0.5, z=0.5 (center of second corridor)
    # Phase 3: Move from z=0.5 to z=7, x=0.5, y=5.5 (center of third corridor)
    # Phase 4: Rotate to (90,90,90)
    
    # Let's try a path that stays near the center of the corridor
    path = np.array([
        # Move to center of first corridor segment
        [-2.0, 0.5, 0.5, 0, 0, 0],
        [-1.0, 0.5, 0.5, 0, 0, 0],
        [0.0, 0.5, 0.5, 0, 0, 0],
        # Move to center of second corridor segment
        [0.5, 0.5, 0.5, 0, 0, 0],
        [0.5, 2.5, 0.5, 0, 0, 0],
        [0.5, 4.5, 0.5, 0, 0, 0],
        # Move to center of third corridor segment
        [0.5, 5.5, 0.5, 0, 0, 0],
        [0.5, 5.5, 2.5, 0, 0, 0],
        [0.5, 5.5, 4.5, 0, 0, 0],
        [0.5, 5.5, 6.5, 0, 0, 0],
        # Rotate to final orientation
        [0.5, 5.5, 6.5, 45, 0, 0],
        [0.5, 5.5, 6.5, 90, 0, 0],
        [0.5, 5.5, 6.5, 90, 45, 0],
        [0.5, 5.5, 6.5, 90, 90, 0],
        [0.5, 5.5, 6.5, 90, 90, 45],
        [0.5, 5.5, 6.5, 90, 90, 90],
    ], dtype=np.float64)
    
    return path
