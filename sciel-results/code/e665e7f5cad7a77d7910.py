import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Define the lattice vectors for the honeycomb packing
    # Unit hexagons have side 1. Center-to-center distance is sqrt(3).
    # Basis vectors e1, e2 such that e1 + e2 is the vertical neighbor?
    # Let's use the coordinates derived in the analysis:
    # e1 = (1.5, sqrt(3)/2)
    # e2 = (-1.5, sqrt(3)/2)
    # Note: e1 - e2 = (3, 0) -> No, e1 - e2 = (3, 0)?
    # e1 = (1.5, 0.866)
    # e2 = (-1.5, 0.866)
    # e1 + e2 = (0, 1.732) = (0, sqrt(3))
    # This matches the vertical neighbor.
    
    sqrt3 = math.sqrt(3)
    e1 = (1.5, sqrt3 / 2)
    e2 = (-1.5, sqrt3 / 2)
    
    # Generate 11 positions
    # Center
    positions = [(0.0, 0.0)]
    
    # First ring (6 neighbors)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) in (e1, e2) basis
    # Let's just list the Cartesian coordinates directly to be safe
    # P0: (0,0)
    
    # Ring 1
    positions.append((1.5, sqrt3 / 2))       # e1
    positions.append((0.0, sqrt3))           # e1 + e2
    positions.append((-1.5, sqrt3 / 2))      # e2
    positions.append((-1.5, -sqrt3 / 2))     # -e1
    positions.append((0.0, -sqrt3))          # -e1 - e2
    positions.append((1.5, -sqrt3 / 2))      # -e2
    
    # Ring 2 (4 selected for symmetry)
    # (1,1) -> e1 + e2? No, (1,1) in basis e1,e2 is e1+e2 which is (0, sqrt3). That's in Ring 1.
    # My previous analysis used a different basis or coordinates.
    # Let's re-verify the coordinates for the 4 outer points.
    # I used P7 = (1.5, 3*sqrt3/2).
    # Is this e1 + e2?
    # e1 = (1.5, sqrt3/2)
    # e2 = (-1.5, sqrt3/2)
    # e1 + e2 = (0, sqrt3).
    # So (1.5, 3*sqrt3/2) is NOT e1+e2.
    # It is e1 + (e1+e2)? No.
    # Let's find the lattice point at (1.5, 3*sqrt3/2).
    # x = 1.5 = 1.5*q - 1.5*r
    # y = 3*sqrt3/2 = sqrt3/2 * q + sqrt3/2 * r
    # Divide y by sqrt3/2: 3 = q + r
    # From x: 1 = q - r
    # q + r = 3
    # q - r = 1
    # 2q = 4 => q = 2
    # r = 1
    # So the point is (2, 1) in the (e1, e2) basis.
    # Let's check distance: |2e1 + 1e2|
    # 2e1 + e2 = 2(1.5, s/2) + (-1.5, s/2) = (3-1.5, s+s/2) = (1.5, 3s/2). Correct.
    
    # So the 4 points are (2,1), (-2,1), (-2,-1), (2,-1)?
    # Let's check (-2, 1):
    # -2e1 + e2 = -2(1.5, s/2) + (-1.5, s/2) = (-3-1.5, -s+s/2) = (-4.5, -s/2).
    # This doesn't look symmetric to (1.5, 3s/2).
    
    # Let's re-evaluate the symmetry.
    # The set of 11 points should be symmetric.
    # The 7 central points are symmetric under 60 deg rotation.
    # The 4 added points must preserve some symmetry.
    # If we add (2,1), we should add its rotations.
    # Rotation of (2,1) by 60 deg?
    # In hex grid, rotation by 60 deg maps (q,r) to ?
    # It's easier to just use the Cartesian coordinates I derived:
    # P7 = (1.5, 3*sqrt3/2)
    # P8 = (-1.5, 3*sqrt3/2)
    # P9 = (-1.5, -3*sqrt3/2)
    # P10 = (1.5, -3*sqrt3/2)
    
    # Are these valid lattice points?
    # P7: (1.5, 3s/2). q=2, r=1. Valid.
    # P8: (-1.5, 3s/2). 
    # x = -1.5 = 1.5q - 1.5r
    # y = 3s/2 = s/2(q+r) => 3 = q+r
    # -1 = q-r
    # 2q = 2 => q=1, r=2.
    # Point (1,2). Valid.
    # P9: (-1.5, -3s/2).
    # x = -1.5 => -1 = q-r
    # y = -3s/2 => -3 = q+r
    # 2q = -4 => q=-2, r=-1.
    # Point (-2,-1). Valid.
    # P10: (1.5, -3s/2).
    # x = 1.5 => 1 = q-r
    # y = -3s/2 => -3 = q+r
    # 2q = -2 => q=-1, r=-2.
    # Point (-1,-2). Valid.
    
    # So the 4 points are (2,1), (1,2), (-2,-1), (-1,-2) in (e1,e2) basis.
    # Let's check if they are distinct from the first ring.
    # First ring: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1).
    # (2,1) is not in first ring.
    # (1,2) is not in first ring.
    # (-2,-1) is not in first ring.
    # (-1,-2) is not in first ring.
    
    # So the 11 points are:
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # (2,1), (1,2), (-2,-1), (-1,-2)
    
    # Let's generate the Cartesian coordinates for these 11 points.
    
    def hex_to_cart(q, r):
        x = 1.5 * q - 1.5 * r
        y = (sqrt3 / 2) * q + (sqrt3 / 2) * r
        return (x, y)
    
    inner = []
    # Center
    inner.append([hex_to_cart(0,0)[0], hex_to_cart(0,0)[1], 0.0])
    
    # Ring 1
    ring1 = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    for q, r in ring1:
        x, y = hex_to_cart(q, r)
        inner.append([x, y, 0.0])
        
    # Ring 2 (selected 4)
    ring2_sel = [(2,1), (1,2), (-2,-1), (-1,-2)]
    for q, r in ring2_sel:
        x, y = hex_to_cart(q, r)
        inner.append([x, y, 0.0])
        
    # Outer hexagon
    # Center (0,0)
    # Side length 4.0
    # Angle 0.0
    
    return inner, (0.0, 0.0), 4.0, 0.0
# EVOLVE-BLOCK-END
