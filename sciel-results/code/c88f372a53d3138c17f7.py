# EVOLVE-BLOCK-START
import numpy as np


def moving_sofa_3d_path() -> np.ndarray:
    """
    3D moving sofa path through L-shaped corridor.
    Uses a parameterized smooth path with fewer waypoints,
    letting the evaluator's densification handle interpolation.
    """
    # Phase 1: Start in left arm (x<0, y in [0,1], z in [0,1])
    # Rotate to align with the corridor turn
    # Phase 2: Move through bottom of L (x in [0,1], y in [0,5], z in [0,1])
    # Phase 3: Turn up (x in [0,1], y in [5,6], z >= 0)
    # Phase 4: End at (0, 6, 7) with (90, 90, 90)
    
    # Define key waypoints - the evaluator will interpolate between these
    # with max 0.05 translation and 2 degree angle changes per step
    waypoints = [
        # Start: near origin in left arm, identity orientation
        [0.0, 0.5, 0.5, 0.0, 0.0, 0.0],
        # Move toward the corner
        [0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
        # At the corner, begin rotating
        [1.0, 0.5, 0.5, 0.0, 0.0, 0.0],
        # Start turning (yaw rotation)
        [1.0, 1.0, 0.5, 45.0, 0.0, 0.0],
        [1.0, 2.5, 0.5, 90.0, 0.0, 0.0],
        # Now moving along y-axis with 90 deg yaw
        [1.0, 4.0, 0.5, 90.0, 0.0, 0.0],
        # Approach the second turn
        [1.0, 5.0, 0.5, 90.0, 0.0, 0.0],
        # Begin pitch rotation
        [1.0, 5.5, 0.5, 90.0, 45.0, 0.0],
        [1.0, 5.5, 1.5, 90.0, 90.0, 0.0],
        # Now moving up in z
        [1.0, 5.5, 3.0, 90.0, 90.0, 0.0],
        [1.0, 5.5, 5.0, 90.0, 90.0, 0.0],
        # Begin roll rotation
        [1.0, 5.5, 6.0, 90.0, 90.0, 45.0],
        [0.5, 5.5, 6.5, 90.0, 90.0, 90.0],
        # Move toward end
        [0.0, 5.5, 7.0, 90.0, 90.0, 90.0],
        # End near target
        [0.0, 6.0, 7.0, 90.0, 90.0, 90.0],
    ]
    
    path = np.array(waypoints, dtype=np.float64)
    return path


# EVOLVE-BLOCK-END
