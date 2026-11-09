import numpy as np

def moving_sofa_3d_path() -> np.ndarray:
    # Key poses
    # Start is prepended: (-3, 0, 0, 0, 0, 0)
    # End is appended: (0, 6, 7, 90, 90, 90)
    
    path = [
        # Move to center of first leg
        [-0.5, 0.5, 0.5, 0, 0, 0],
        
        # Approach corner 1
        [0.0, 0.5, 0.5, 0, 0, 0],
        
        # Rotate in corner 1 (Center 0.5, 0.5, 0.5)
        [0.5, 0.5, 0.5, 45, 0, 0],
        [0.5, 0.5, 0.5, 90, 0, 0],
        
        # Move along second leg
        [0.5, 1.0, 0.5, 90, 0, 0],
        [0.5, 2.5, 0.5, 90, 0, 0],
        [0.5, 4.0, 0.5, 90, 0, 0],
        [0.5, 5.0, 0.5, 90, 0, 0],
        
        # Approach corner 2
        [0.5, 5.5, 0.5, 90, 0, 0],
        
        # Rotate in corner 2 (Center 0.5, 5.5, 0.5)
        # Need to go from (90, 0, 0) to (90, 90, 90)
        # Intermediate: (90, 45, 45)
        [0.5, 5.5, 0.5, 90, 45, 45],
        [0.5, 5.5, 0.5, 90, 90, 90],
        
        # Move along third leg
        [0.5, 5.5, 1.0, 90, 90, 90],
        [0.5, 5.5, 2.5, 90, 90, 90],
        [0.5, 5.5, 4.0, 90, 90, 90],
        [0.5, 5.5, 5.5, 90, 90, 90],
        
        # Approach end
        [0.5, 5.5, 6.5, 90, 90, 90],
        [0.0, 6.0, 7.0, 90, 90, 90] # This will be interpolated with the appended end pose
    ]
    
    return np.array(path, dtype=np.float64)
