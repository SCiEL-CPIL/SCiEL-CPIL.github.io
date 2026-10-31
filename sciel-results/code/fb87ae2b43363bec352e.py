<<<<<<< SEARCH
    # Outer hexagon: center, side length, rotation
    # The rotation of the outer hexagon: we need to determine which angle gave the min
    # For simplicity, use 0 rotation and adjust center
    # Actually, let's recompute with the optimal rotation
    # For now, return with 0 rotation and the computed side
    
    return inner, (center[0], center[1]), side, 0.0
=======
    # Outer hexagon: center, side length, rotation
    # Convert angle from radians to degrees
    angle_deg = np.degrees(angle)
    
    return inner, (center[0], center[1]), side, angle_deg
>>>>>>> REPLACE
