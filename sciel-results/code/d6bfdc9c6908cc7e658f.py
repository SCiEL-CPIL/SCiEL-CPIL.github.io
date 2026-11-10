import numpy as np

def hexagon_packing_11():
    n = 11
    # Unit hexagons
    # Center
    centers = [
        [0, 0],
        # Layer 1
        [np.sqrt(3), 0],
        [np.sqrt(3)/2, 1.5],
        [-np.sqrt(3)/2, 1.5],
        [-np.sqrt(3), 0],
        [-np.sqrt(3)/2, -1.5],
        [np.sqrt(3)/2, -1.5],
        # Layer 2 (4 gaps)
        # Angles 30, 150, 210, 330
        [3 * np.cos(np.pi/6), 3 * np.sin(np.pi/6)],
        [3 * np.cos(5*np.pi/6), 3 * np.sin(5*np.pi/6)],
        [3 * np.cos(7*np.pi/6), 3 * np.sin(7*np.pi/6)],
        [3 * np.cos(11*np.pi/6), 3 * np.sin(11*np.pi/6)],
    ]
    
    inner_hex_data = np.array([[c[0], c[1], 0] for c in centers])
    
    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = 4.0
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length
