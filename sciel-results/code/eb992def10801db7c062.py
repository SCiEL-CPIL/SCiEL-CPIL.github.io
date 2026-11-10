import numpy as np


def moving_sofa_3d_path() -> np.ndarray:
    """
    Generate a path for the 3D moving sofa problem.
    
    Strategy: Use dense keyframes with separated translation and rotation phases
    to minimize the bounding box of the sofa during critical corner transitions.
    The sofa is kept centered in the corridor cross-sections (y=0.5, z=0.5) where possible.
    """
    keyframes = []
    
    # Helper to add a sequence of poses
    def add_seq(start, end, n, start_angles, end_angles):
        if n <= 0:
            return
        for i in range(n):
            t = i / (n - 1) if n > 1 else 0.0
            # Linear interpolation for position
            x = start[0] + (end[0] - start[0]) * t
            y = start[1] + (end[1] - start[1]) * t
            z = start[2] + (end[2] - start[2]) * t
            # Linear interpolation for angles
            yaw = start_angles[0] + (end_angles[0] - start_angles[0]) * t
            pitch = start_angles[1] + (end_angles[1] - start_angles[1]) * t
            roll = start_angles[2] + (end_angles[2] - start_angles[2]) * t
            keyframes.append([x, y, z, yaw, pitch, roll])

    # 1. Start at (-3, 0.5, 0.5), Yaw=0
    # Move X from -3 to -0.5
    add_seq((-3.0, 0.5, 0.5), (-0.5, 0.5, 0.5), 20, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    
    # 2. Move X from -0.5 to 0.5 (straddling the corner)
    # Keep Yaw=0 to minimize Y extent (sofa is aligned with X)
    add_seq((-0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 20, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    
    # 3. Rotate Yaw from 0 to 90 at x=0.5, y=0.5, z=0.5
    # This is the critical step. The sofa rotates in the corner.
    # The bounding box of the rotation arc must stay within the corridor union.
    # With y=0.5, z=0.5, the sofa is centered.
    add_seq((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 30, [0.0, 0.0, 0.0], [90.0, 0.0, 0.0])
    
    # 4. Move Y from 0.5 to 5.0 in Region 2
    # Yaw=90, so sofa is aligned with Y.
    add_seq((0.5, 0.5, 0.5), (0.5, 5.0, 0.5), 40, [90.0, 0.0, 0.0], [90.0, 0.0, 0.0])
    
    # 5. Move Y from 5.0 to 5.5 (center of Region 3 y-range 5-6)
    add_seq((0.5, 5.0, 0.5), (0.5, 5.5, 0.5), 10, [90.0, 0.0, 0.0], [90.0, 0.0, 0.0])
    
    # 6. Rotate Pitch from 0 to 90 at x=0.5, y=5.5, z=0.5
    # Aligns sofa with Z axis.
    add_seq((0.5, 5.5, 0.5), (0.5, 5.5, 0.5), 30, [90.0, 0.0, 0.0], [90.0, 90.0, 0.0])
    
    # 7. Move Z from 0.5 to 7.0 in Region 3
    # Pitch=90, Yaw=90.
    add_seq((0.5, 5.5, 0.5), (0.5, 5.5, 7.0), 60, [90.0, 90.0, 0.0], [90.0, 90.0, 0.0])
    
    # 8. Move to final position x=0, y=6, z=7
    # Adjust x and y to match end pose.
    add_seq((0.5, 5.5, 7.0), (0.0, 6.0, 7.0), 20, [90.0, 90.0, 0.0], [90.0, 90.0, 90.0])
    
    # Ensure the last pose is exactly the target if not already
    if keyframes[-1][0] != 0.0 or keyframes[-1][1] != 6.0 or keyframes[-1][2] != 7.0:
        keyframes.append([0.0, 6.0, 7.0, 90.0, 90.0, 90.0])
        
    path = np.array(keyframes, dtype=np.float64)
    return path
