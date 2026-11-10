# EVOLVE-BLOCK-START
import numpy as np


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Use a side length of 4.0
    # This is a safe upper bound for a compact packing of 11 hexagons.
    # The 7-hex flower fits in side 3.
    # Adding 4 more requires expansion.
    # Side 4 provides ample room.
    
    # Coordinates for 11 hexagons
    # Center
    centers = []
    centers.append((0.0, 0.0))
    
    # 6 neighbors
    for i in range(6):
        angle = i * 60
        rad = np.deg2rad(angle)
        x = np.sqrt(3) * np.cos(rad)
        y = np.sqrt(3) * np.sin(rad)
        centers.append((x, y))
        
    # 4 more hexagons
    # Place them at the "corners" of the 7-hex flower
    # The corners are at angles 30, 90, 150, 210, 270, 330?
    # The vertices of the 7-hex flower are at distance sqrt(7).
    # We place new hexagons touching these vertices.
    # To be safe, we place them at distance 2*sqrt(3) from center?
    # Let's just place them at (0, 2*sqrt(3)), (0, -2*sqrt(3)), (2*sqrt(3), 0), (-2*sqrt(3), 0)
    # This might be too far for side 4.
    # Distance 2*sqrt(3) + 1 = 3.46 + 1 = 4.46.
    # Side 4 circumradius is 4.
    # So this placement is invalid for side 4.
    
    # We need a more compact placement.
    # Let's use a grid.
    # Row 0: 3 hexagons
    # Row 1: 2 hexagons
    # Row 2: 3 hexagons
    # Row 3: 2 hexagons
    # Total 10. Need 11.
    # Row 0: 3
    # Row 1: 3
    # Row 2: 3
    # Row 3: 2
    # Total 11.
    
    # Hex grid coordinates to Cartesian
    # Let's use axial coordinates (q, r)
    # Conversion to Cartesian (pointy-top):
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    
    # We want a compact shape.
    # Let's try a 3x4 block?
    # Or just use the 7-hex flower and 4 more placed in the gaps of the second layer.
    # The second layer has 12 positions.
    # We pick 4 that are closest to the center.
    # The closest positions in the second layer are the ones adjacent to two first-layer hexagons.
    # These are at distance 2*sqrt(3) from the center?
    # No, the distance from center to a second-layer hexagon center is 2*sqrt(3).
    # This is the same for all second-layer hexagons.
    # So any 4 from the second layer will be at the same distance.
    # This means the max distance is 2*sqrt(3) + 1 = 4.46.
    # So side 4 is NOT enough for any second-layer hexagon.
    
    # Therefore, we must use a side length of at least 4.46.
    # Let's use side 4.5.
    # Score 1/4.5 = 0.222.
    # Seed score 0.125.
    # Ratio 1.77.
    
    # Can we do better?
    # What if we don't use a full second layer?
    # We can place hexagons in the "gaps" of the first layer?
    # The gaps are between the 6 neighbors.
    # The center of a gap is at distance sqrt(3) * sqrt(3) = 3?
    # No.
    # The distance from center to the midpoint of the edge between two neighbors is sqrt(3).
    # The gap is a triangle.
    # We can place a hexagon in the gap?
    # No, the gap is too small.
    
    # So the minimum side length is determined by the second layer.
    # Let's use side 4.5.
    
    outer_hex_side_length = 4.5
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    # Generate 11 hexagons
    # 7-hex flower
    inner_hex_data = []
    inner_hex_data.append([0.0, 0.0, 0.0])
    for i in range(6):
        angle = i * 60
        rad = np.deg2rad(angle)
        x = np.sqrt(3) * np.cos(rad)
        y = np.sqrt(3) * np.sin(rad)
        inner_hex_data.append([x, y, 0.0])
        
    # 4 more from the second layer
    # Pick 4 positions
    # (2, 0), (0, 2), (-2, 0), (0, -2) in axial coordinates?
    # Axial (q, r)
    # (2, 0) -> x = sqrt(3)*2, y = 0
    # (0, 2) -> x = sqrt(3)*1, y = 3
    # (-2, 0) -> x = -sqrt(3)*2, y = 0
    # (0, -2) -> x = -sqrt(3)*1, y = -3
    
    # Let's use these 4
    second_layer = [
        (2, 0),
        (0, 2),
        (-2, 0),
        (0, -2)
    ]
    
    for q, r in second_layer:
        x = np.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        inner_hex_data.append([x, y, 0.0])
        
    inner_hex_data = np.array(inner_hex_data)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
