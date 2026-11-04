import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # 12 unit hexagons arranged in 3 rows of 4, flat-to-flat (0-degree rotation).
    # Row y-coordinates: -sqrt(3), 0, +sqrt(3)
    # Within each row, x-coordinates spaced by 2 (vertex-to-vertex of unit hex).
    # Alternating rows offset by 1 to nestle hexagons.
    #
    # Container: regular hexagon centered at origin, side length 4, flat-to-flat
    # (vertices at angles 0,60,120,180,240,300 from center).
    #
    # Verify: outer hexagon vertices at (4,0), (2, 2*sqrt(3)), (-2, 2*sqrt(3)),
    # (-4,0), (-2,-2*sqrt(3)), (2,-2*sqrt(3)).
    #
    # Row at y=0: hexagons at x = -3, -1, 1, 3
    #   Leftmost vertex: x=-3-1=-4, y=0 -> on boundary (4,0) rotated: (-4,0) OK
    #   Rightmost vertex: x=3+1=4, y=0 -> (4,0) OK
    #   Top vertex: x=3, y=0+sqrt(3)/2*... wait, unit hex vertices.
    #
    # Unit hexagon centered at (cx,cy) with 0-deg rotation (flat-to-flat):
    #   vertices at angles 0,60,120,180,240,300:
    #   (cx+1, cy), (cx+0.5, cy+sqrt(3)/2), (cx-0.5, cy+sqrt(3)/2),
    #   (cx-1, cy), (cx-0.5, cy-sqrt(3)/2), (cx+0.5, cy-sqrt(3)/2)
    #
    # Row y=0, x in {-3,-1,1,3}:
    #   x=3: vertices (4,0),(3.5,0.866),(2.5,0.866),(2,0),(2.5,-0.866),(3.5,-0.866)
    #   x=-3: vertices (-2,0),(-2.5,0.866),(-3.5,0.866),(-4,0),(-3.5,-0.866),(-2.5,-0.866)
    #   x=-1: vertices (0,0),(-0.5,0.866),(-1.5,0.866),(-2,0),(-1.5,-0.866),(-0.5,-0.866)
    #   x=1: vertices (2,0),(1.5,0.866),(0.5,0.866),(0,0),(0.5,-0.866),(1.5,-0.866)
    #
    # Row y=sqrt(3), x in {-2,0,2} (offset by 1, so 3 hexagons? No, we need 4 per row for 12 total)
    # Wait: 3 rows of 4 = 12. Let me reconsider.
    #
    # Actually for 12 hexagons in a hexagonal container, a common optimal is:
    # 4 in middle row, 4 in top row, 4 in bottom row, but offset.
    #
    # Let me use: rows at y = -sqrt(3), 0, sqrt(3)
    #   Row y=0: x = -3, -1, 1, 3  (4 hexagons)
    #   Row y=sqrt(3): x = -2, 0, 2  (3 hexagons) -- but that's only 11
    #
    # Need 12. Let me try 4+4+4:
    #   Row y=0: x = -3, -1, 1, 3
    #   Row y=sqrt(3): x = -2, 0, 2, 4? No, too wide.
    #
    # Alternative: 2-3-4-3-2 = 14, too many. 1-3-5-3-1=13. 2-4-4-2=12!
    #   Row y=-2*sqrt(3): x = -1, 1  (2 hexagons)
    #   Row y=-sqrt(3): x = -2, 0, 2, 4? 
    #
    # Let me just use a simple 3x4 grid with proper hexagonal spacing.
    # Rows at y = 0, sqrt(3), 2*sqrt(3) -- no, let me center them.
    #
    # Simplest: 3 rows of 4, rows at y = -sqrt(3), 0, sqrt(3)
    #   Row y=0: x = -3, -1, 1, 3
    #   Row y=sqrt(3): x = -2, 0, 2, 4 -- but x=4 is too far right
    #
    # The issue: with offset rows, the top/bottom rows are narrower.
    # For 12 = 4+4+4, all rows have 4 hexagons, so no offset needed for count,
    # but geometrically offset rows nestle better.
    #
    # Let me just do a straight 3x4 grid (no offset) and check if it fits in s=4.
    #   Row y=0: x = -3, -1, 1, 3
    #   Row y=sqrt(3): x = -3, -1, 1, 3
    #   Row y=-sqrt(3): x = -3, -1, 1, 3
    #
    # Check top-right corner: hex at (3, sqrt(3)):
    #   vertex (4, sqrt(3)), (3.5, sqrt(3)+0.866), (2.5, sqrt(3)+0.866), ...
    #   sqrt(3) ≈ 1.732, so top vertex y ≈ 2.598
    #   Outer hex at angle 60: vertex (2, 2*sqrt(3)) ≈ (2, 3.464)
    #   The hex vertex (3.5, 2.598) -- is it inside?
    #   Outer hex edge from (4,0) to (2, 3.464): line x + y/sqrt(3) = 4
    #   At (3.5, 2.598): 3.5 + 2.598/1.732 = 3.5 + 1.5 = 5.0 > 4. OUTSIDE!
    #
    # So straight grid doesn't fit in s=4. Need offset rows or smaller s.
    #
    # With offset: top row shifted by 1, so x = -2, 0, 2, 4 -- still x=4 is bad.
    # Top row x = -2, 0, 2 only (3 hexagons), but then total is 4+3+4=11.
    #
    # For 12, try 2-4-4-2 = 12:
    #   Row y=-1.5*sqrt(3): x = -1, 1  (2)
    #   Row y=-0.5*sqrt(3): x = -2, 0, 2, 4? 
    #
    # This is getting complex. Let me just use a known good configuration:
    # Place hexagons in a hexagonal lattice pattern that fits in s=4.
    #
    # Actually, the minimum side length for 12 unit hexagons in a hexagon is known.
    # Let me just use a reasonable packing and let the evaluator score it.
    #
    # Simple approach: 4 columns of 3, hexagonal lattice.
    sqrt3 = np.sqrt(3)
    inner = []
    # 3 rows, 4 columns, hexagonal offset
    for row in range(3):
        y = (row - 1) * sqrt3  # -sqrt3, 0, sqrt3
        for col in range(4):
            x = (col - 1.5) * 2.0  # -3, -1, 1, 3
            if row % 2 == 1:  # middle row offset? No, let's not offset
                x += 0.0
            inner.append([x, y, 0.0])
    
    # This gives 12 hexagons. Container side length:
    # The bounding hexagon needs to contain all vertices.
    # Max extent: x from -4 to 4, y from -sqrt3-sqrt3/2 to sqrt3+sqrt3/2 = -1.5*sqrt3 to 1.5*sqrt3
    # For a flat-to-flat hexagon of side s, the half-width is s and half-height is s*sqrt3/2.
    # Need s >= 4 (from x extent) and s*sqrt3/2 >= 1.5*sqrt3 => s >= 3.
    # So s=4 should work if the corners fit. Let me verify more carefully.
    
    # Outer hexagon vertices (flat-to-flat, side s=4):
    # (4,0), (2, 2*sqrt3), (-2, 2*sqrt3), (-4,0), (-2,-2*sqrt3), (2,-2*sqrt3)
    # Edges:
    #   Right: x = 4 (from (4,0) to... no, flat-to-flat has vertices at those points)
    #   Actually flat-to-flat means flats are horizontal. Vertices at angles 0,60,...
    #   The rightmost point is (4,0). Top-right edge from (4,0) to (2, 2*sqrt3).
    #   Line: x - 2 = (4-2)/(0-2*sqrt3) * (y - 2*sqrt3) => x-2 = -1/sqrt3*(y-2*sqrt3)
    #   => x + y/sqrt3 = 4
    #
    # Check hex at (3, sqrt3): top-right vertex (3.5, sqrt3+sqrt3/2) = (3.5, 1.5*sqrt3)
    #   3.5 + 1.5*sqrt3/sqrt3 = 3.5 + 1.5 = 5.0 > 4. OUTSIDE.
    #
    # So s=4 is too small. Need s such that 3.5 + 1.5 = 5.0 <= s, so s >= 5?
    # But that seems large. Let me reconsider the layout.
    
    # Better: use offset rows so top/bottom rows are narrower.
    # Row y=0: x = -3, -1, 1, 3  (4 hexagons)
    # Row y=sqrt3: x = -2, 0, 2  (3 hexagons)  -- offset by 1
    # Row y=-sqrt3: x = -2, 0, 2  (3 hexagons)
    # Total: 10, not 12.
    
    # For 12: 4 + 4 + 4 with no offset, but then s must be larger.
    # Or 2 + 4 + 4 + 2 = 12:
    #   Row y=-1.5*sqrt3: x = -1, 1
    #   Row y=-0.5*sqrt3: x = -2, 0, 2, 4? 
    #
    # Let me just go with a compact arrangement and set s=5 to be safe,
    # then the evaluator will score it. We can optimize later.
    
    # Actually, let me use a cleaner hexagonal packing:
    # 12 hexagons in a hexagonal cluster of "radius" 2 (center + 6 + 9 = 16, too many)
    # Radius 1: 1 + 6 = 7. Radius 2: 7 + 12 = 19. 
    # 12 is between. Let me just do a 3x4 grid with s=5.
    
    inner = []
    for row in range(3):
        y = (row - 1) * sqrt3
        for col in range(4):
            x = (col - 1.5) * 2.0
            inner.append([x, y, 0.0])
    
    return inner, (0.0, 0.0), 5.0, 0.0
# EVOLVE-BLOCK-END
