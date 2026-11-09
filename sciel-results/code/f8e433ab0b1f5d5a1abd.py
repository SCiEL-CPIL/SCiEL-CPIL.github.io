import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Generate a 3x4 block of hexagons in a hexagonal lattice
    # Hex lattice coordinates:
    # x = i * sqrt(3) + j * sqrt(3)/2
    # y = j * 1.5
    
    sqrt3 = np.sqrt(3)
    centers = []
    
    # 3 rows (j = -1, 0, 1) and 4 columns (i = -1, 0, 1, 2)
    # We will center the resulting shape afterwards.
    for j in range(-1, 2):
        for i in range(-1, 3):
            x = i * sqrt3 + j * sqrt3 / 2.0
            y = j * 1.5
            centers.append([x, y])
            
    centers = np.array(centers)
    
    # Center the block
    centroid = np.mean(centers, axis=0)
    centers = centers - centroid
    
    inner = []
    for x, y in centers:
        inner.append([float(x), float(y), 0.0])
        
    # Outer hexagon side length 5.0
    # Center (0.0, 0.0)
    # Angle 0.0 (pointy top, aligned with the lattice such that flat sides are not parallel to axes? 
    # The lattice has horizontal rows. The hexagons are "flat-top" if angle=0?
    # Standard hexagon definition:
    # If angle=0, vertices are usually at 0, 60, etc.
    # In our lattice derivation, we assumed "flat-top" hexagons?
    # Let's check the orientation.
    # In the lattice formula x = i*sqrt(3) + j*sqrt(3)/2, the hexagons are "flat-top" (width sqrt(3), height 2).
    # Wait, if width is sqrt(3) and height is 2, the vertices are at top and bottom.
    # This corresponds to a "pointy-top" hexagon.
    # A "pointy-top" hexagon has vertices at 90, 30, -30, -90, -150, 150 degrees.
    # In standard math coordinates, this is often achieved by rotating a "flat-top" hexagon by 30 degrees.
    # If the interface expects angle_degrees, we need to specify the rotation of the inner hexagons.
    # If we set angle=0, what is the orientation?
    # Usually, angle=0 means a vertex is at 0 degrees (right side).
    # If vertex is at 0, the hexagon is "pointy-side" (flat top/bottom? No, flat left/right? No).
    # Vertex at 0: (1,0). Next vertex at 60: (0.5, sqrt(3)/2).
    # This is a "pointy-side" hexagon? No, it has a point on the right.
    # This orientation has width 2 (x-axis) and height sqrt(3) (y-axis).
    # My lattice assumed width sqrt(3) and height 2.
    # So I need to rotate the inner hexagons by 30 degrees to match the lattice spacing?
    # Or adjust the lattice?
    
    # Let's stick to the lattice geometry which is robust.
    # The lattice of "pointy-top" hexagons (vertices at 90, 30...) has:
    # Horizontal distance between centers in same row: sqrt(3).
    # Vertical distance between rows: 1.5.
    # Offset in x for next row: sqrt(3)/2.
    # This matches my formula.
    # So the inner hexagons must be "pointy-top".
    # A "pointy-top" hexagon has a vertex at 90 degrees.
    # If angle=0 means vertex at 0 degrees, then angle=90 means vertex at 90 degrees.
    # So we should set angle_degrees = 90.0.
    
    # Let's double check the outer hexagon orientation.
    # The outer hexagon side is 5.
    # If we set outer angle to 0, it has vertex at 0.
    # If we set outer angle to 90, it has vertex at 90 (pointy top).
    # My fit calculation assumed the outer hexagon was "pointy-top" (vertices at 90, 270).
    # So outer angle should be 90.0.
    
    # Wait, the task says "outer hexagon center (x, y), side length and rotation".
    # The score is the outer side length.
    # The rotation of the outer hexagon does not change its side length, but it must contain the inner ones.
    # So we must align the outer hexagon to contain the inner cluster.
    
    return inner, (0.0, 0.0), 5.0, 90.0
# EVOLVE-BLOCK-END
