import math

N = 11

def construct():
    # Hexagonal lattice vectors
    # u = (sqrt(3), 0)
    # v = (sqrt(3)/2, 1.5)
    
    sqrt3 = math.sqrt(3)
    u = (sqrt3, 0.0)
    v = (sqrt3 / 2.0, 1.5)
    
    # Generate 7 core hexagons (center + 6 neighbors)
    # Lattice coords (i, j)
    core_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1),
        (-1, 0), (0, -1), (1, -1)
    ]
    
    # Add 4 extra hexagons at "corners" of the 2nd ring
    # These correspond to (1, 1), (1, -1), (-1, 1), (-1, -1) in a specific basis?
    # Let's verify the positions.
    # (1, 1) -> 1*u + 1*v = (sqrt3 + sqrt3/2, 1.5) = (1.5*sqrt3, 1.5)
    # (1, -1) -> 1*u - 1*v = (sqrt3 - sqrt3/2, -1.5) = (0.5*sqrt3, -1.5) -> Wait.
    # Lattice basis:
    # P(i,j) = i*u + j*v
    # Neighbors of (0,0):
    # (1,0) -> (sqrt3, 0)
    # (0,1) -> (sqrt3/2, 1.5)
    # (-1,1) -> -u + v = (-sqrt3 + sqrt3/2, 1.5) = (-sqrt3/2, 1.5)
    # (-1,0) -> (-sqrt3, 0)
    # (0,-1) -> (-sqrt3/2, -1.5)
    # (1,-1) -> u - v = (sqrt3 - sqrt3/2, -1.5) = (sqrt3/2, -1.5)
    
    # The "corners" of the 7-cluster are the neighbors themselves.
    # The 2nd ring "corners" are at distance 2 in lattice steps?
    # Positions like (1, 1), (2, 0), etc.
    # (1, 1) is adjacent to (1,0) and (0,1).
    # (1, 1) -> (1.5*sqrt3, 1.5)
    # (1, -1) -> (0.5*sqrt3, -1.5) -- This is adjacent to (1,0) and (0,-1)?
    # (1,0) is (sqrt3, 0). (0,-1) is (-sqrt3/2, -1.5).
    # (1,-1) is (sqrt3/2, -1.5).
    # Distance (1,0) to (1,-1): sqrt((sqrt3-sqrt3/2)^2 + (0 - (-1.5))^2) = sqrt((sqrt3/2)^2 + 1.5^2) = sqrt(0.75 + 2.25) = sqrt(3). Correct.
    
    # So the 4 "corner" positions to add are:
    # (1, 1) -> (1.5*sqrt3, 1.5)
    # (1, -1) -> (0.5*sqrt3, -1.5)  <-- Wait, is this a corner?
    # Let's check symmetry.
    # The 7 cluster is symmetric.
    # The 2nd ring has 12 positions.
    # 6 are "side" extensions: (2,0), (0,2), (-2,2), (-2,0), (0,-2), (2,-2).
    # 6 are "corner" extensions: (1,1), (1,-1), (-1,1), (-1,-1), (2,1), (-2,1)...
    # Actually, the 6 "corner" positions are (1,1), (1,-1), (-1,1), (-1,-1) and...
    # Let's list all 12:
    # (2,0), (1,1), (0,2), (-1,1), (-2,0), (-1,-1), (0,-2), (1,-1), (2,-1)... no.
    # Standard hex grid neighbors of (1,0) are (2,0), (1,1), (0,1), (0,-1), (1,-1), (2,-1).
    # (0,1) and (0,-1) are in the core.
    # So new neighbors of (1,0) are (2,0), (1,1), (1,-1), (2,-1).
    # (2,0) is a side extension.
    # (1,1) and (1,-1) are corner extensions?
    # (1,1) is at (1.5*sqrt3, 1.5).
    # (1,-1) is at (0.5*sqrt3, -1.5).
    # These are not symmetric with each other relative to the x-axis.
    # (1,1) is in Q1. (1,-1) is in Q4.
    # To get 4 symmetric ones, we can pick:
    # (1, 1) -> Q1
    # (-1, 1) -> Q2 (from neighbor of (-1,0) or (0,1))
    # (-1, -1) -> Q3
    # (1, -1) -> Q4
    
    # Let's verify (-1, 1).
    # -1*u + 1*v = (-sqrt3 + sqrt3/2, 1.5) = (-sqrt3/2, 1.5).
    # This is the same as (0,1) which is in the core!
    # Wait.
    # Core has (0,1) at (-sqrt3/2, 1.5).
    # So (-1,1) is NOT a new position.
    
    # Let's re-list the 2nd ring positions carefully.
    # Core: (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1).
    # Note: (-1,1) is in core. (1,-1) is in core.
    # So the "corner" positions of the 2nd ring are NOT (1,1) etc?
    # (1,1) is adjacent to (1,0) and (0,1). Both in core.
    # So (1,1) is a valid new position.
    # (1,-1) is adjacent to (1,0) and (0,-1). Both in core.
    # So (1,-1) is a valid new position.
    # (-1,1) is in core.
    # (-1,-1) is adjacent to (-1,0) and (0,-1). Both in core.
    # So (-1,-1) is a valid new position.
    # (2,0) is adjacent to (1,0) and (1,-1)? No, (1,0) and (1,1)?
    # (2,0) is adjacent to (1,0) and (1,-1)?
    # (1,-1) is in core.
    # So (2,0) is a valid new position.
    
    # The 12 new positions are:
    # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (1,-1)... wait.
    # (1,-1) is in core.
    # Let's just generate all lattice points with distance <= 2 and exclude core.
    
    lattice_points = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            # Check if distance is small
            # Just check if it's in the 2nd ring
            # A point is in 2nd ring if it's not in core and adjacent to core.
            # Simpler: list all (i,j) with max(|i|, |j|, |i-j|) <= 2?
            # Hex distance: (|i| + |j| + |i-j|) / 2
            # Core has dist <= 1.
            # We want dist == 2.
            dist = (abs(i) + abs(j) + abs(i-j)) / 2.0
            if dist == 2.0:
                x = i * u[0] + j * v[0]
                y = i * u[1] + j * v[1]
                lattice_points.append((x, y, i, j))
                
    # We have 12 points.
    # We need to pick 4.
    # To maintain symmetry and minimize S, we should pick the ones that are "corner-like".
    # The "side-like" ones are (2,0), (0,2), etc. (6 of them).
    # The "corner-like" ones are the other 6.
    # Let's identify them by angle.
    # (2,0) -> (2*sqrt3, 0). Angle 0.
    # (1,1) -> (1.5*sqrt3, 1.5). Angle 30.
    # (0,2) -> (sqrt3, 3). Angle 60.
    # (-1,2) -> (-sqrt3/2, 3). Angle 120.
    # (-2,1) -> (-1.5*sqrt3, 1.5). Angle 150.
    # (-2,0) -> (-2*sqrt3, 0). Angle 180.
    # (-1,-1) -> (-1.5*sqrt3, -1.5). Angle 210.
    # (0,-2) -> (-sqrt3, -3). Angle 240.
    # (1,-2) -> (sqrt3/2, -3). Angle 300.
    # (2,-1) -> (1.5*sqrt3, -1.5). Angle 330.
    # (1,-1) is in core.
    # (-1,1) is in core.
    
    # The 6 "corner" positions are at angles 30, 90 (no, 0,2 is 60), 120, 150, 210, 240, 300, 330.
    # Wait, (0,2) is at angle 60.
    # (1,1) is at angle 30.
    # So the 12 points are at 0, 30, 60, 120, 150, 180, 210, 240, 300, 330...
    # Missing 90, 270.
    # So we have 2 points per 60 degree sector?
    # Sector 0-60: (2,0) at 0, (1,1) at 30.
    # Sector 60-120: (0,2) at 60, (-1,2) at 120? No, (-1,2) is 120.
    # So in each 60 deg sector, there are 2 points: one at 0, one at 30 (relative to sector start).
    # To minimize the bounding hexagon, we should pick the points that are "closer" to the vertices of the outer hexagon?
    # If outer hex vertices are at 0, 60, 120...
    # The points at 0, 60, 120... are "side" extensions.
    # The points at 30, 90 (none), 150... are "corner" extensions.
    # Previously I calculated that "corner" extensions (like 1,1) require S=3.96.
    # "Side" extensions (like 2,0) require S=4.46.
    # So we should pick the "corner" extensions.
    # There are 6 corner extensions: (1,1), (-1,2), (-2,1), (-1,-1), (1,-2), (2,-1).
    # Wait, (-1,-1) was listed as valid.
    # Let's check (-1,-1) angle.
    # x = -1.5*sqrt3, y = -1.5. Angle 210.
    # Yes.
    # So we pick 4 from these 6.
    # Any 4 will have similar worst-case S.
    # Let's pick (1,1), (-1,2), (-2,1), (-1,-1).
    
    # Actually, just picking any 4 from the 12 that are not the "side" ones (0, 60, 120, 180, 240, 300) is best.
    # The "side" ones are (2,0), (0,2), (-2,2)?? No.
    # Side ones: (2,0), (0,2), (-2,0), (0,-2), (2,-2)??
    # (2,-2) -> (2*sqrt3 - 2*sqrt3/2, -3) = (sqrt3, -3). Angle 300.
    # So Side ones are at 0, 60, 120, 180, 240, 300.
    # Corner ones are at 30, 90 (none), 150, 210, 270 (none), 330.
    # Wait, (-1,2) is at 120?
    # x = -sqrt3/2, y = 3. Angle 100?
    # atan2(3, -0.866) = 106 deg.
    # My angle approximation was rough.
    
    # Regardless, the 6 "side" ones are further out in the vertex direction.
    # The 6 "corner" ones are further out in the side normal direction.
    # Since the outer hex is constrained by the side normal (apothem) more tightly relative to the vertex for these positions?
    # No, S is determined by max projection.
    # For side ones, projection on vertex direction is large.
    # For corner ones, projection on side normal direction is large.
    # We calculated Side -> S=4.46, Corner -> S=3.96.
    # So we pick Corner ones.
    
    # We need 4.
    # Let's pick (1,1), (-2,1), (-1,-1), (2,-1).
    
    selected_coords = [
        (1, 1),
        (-2, 1),
        (-1, -1),
        (2, -1)
    ]
    
    all_coords = core_coords + selected_coords
    
    inner = []
    for i, j in all_coords:
        x = i * u[0] + j * v[0]
        y = i * u[1] + j * v[1]
        inner.append([x, y, 0.0])
        
    # Calculate S
    # S = 2*sqrt(3) + 0.5
    S = 2.0 * sqrt3 + 0.5
    
    return inner, (0.0, 0.0), S, 0.0

# EVOLVE-BLOCK-END
