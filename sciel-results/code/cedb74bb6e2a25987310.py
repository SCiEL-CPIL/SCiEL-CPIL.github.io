"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Hexagonal grid packing: unit hexagons on honeycomb lattice
    # Center hexagon at (0,0), 6 neighbors at distance 2, 
    # plus additional hexagons to reach 11 total
    
    # For N=11, we can use:
    # - 1 center
    # - 6 at ring 1 (distance 2)
    # - 4 from ring 2 (selected to minimize enclosing hexagon)
    
    # Axial coordinate system for hexagons
    # Center: (0,0)
    # Ring 1: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # These are 6 hexagons at distance 2 from center
    
    # Ring 2 candidates: (2,0), (2,-1), (1,-2), (-1,-1), (-2,1), (-2,2), etc.
    # We need 4 more to reach 11 total
    
    # Convert axial to Cartesian for pointy-top hexagons
    def axial_to_cartesian(q, r):
        x = math.sqrt(3) * (q + r/2)
        y = 1.5 * r
        return (x, y)
    
    # Build the 11 positions
    positions_axial = [(0,0)]  # center
    # Ring 1: 6 hexagons
    ring1 = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    positions_axial.extend(ring1)
    
    # Ring 2: pick 4 hexagons that minimize the enclosing hexagon
    # Try all combinations of 4 from ring 2 and compute required side length
    ring2_all = [(2,0), (2,-1), (1,-2), (-1,-1), (-2,1), (-2,2), (-1,2), (0,2), (1,1), (0,-2), (-1,-2), (-2,-1)]
    # Actually ring 2 has 12 hexagons: (q,r) where |q|+|r|+|q+r|=2
    
    # Generate proper ring 2
    ring2_proper = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if abs(q) + abs(r) + abs(s) == 2:
                ring2_proper.append((q, r))
    
    # For now, use a simple approach: place remaining 4 at (2,0), (0,2), (-2,2), (-2,0) 
    # or similar symmetric configuration
    
    # Actually, let's just use the first 4 from ring 2 for simplicity
    positions_axial.extend(ring2_proper[:4])
    
    # Convert to Cartesian
    inner = []
    for q, r in positions_axial:
        x, y = axial_to_cartesian(q, r)
        inner.append([x, y, 0.0])
    
    # Compute the minimal enclosing hexagon
    # For a regular hexagon with side S, center at (cx, cy), rotation theta:
    # The apothem is S*sqrt(3)/2
    # Each point must satisfy: projection onto each of 6 normals <= apothem + A_inner
    
    # A_inner (apothem of unit hexagon) = sqrt(3)/2
    
    # Try different rotations and find minimal S
    best_S = float('inf')
    best_theta = 0.0
    
    points = [(p[0], p[1]) for p in inner]
    
    # For pointy-top hexagon (vertex up), normals are at angles 30, 90, 150, 210, 270, 330 degrees
    # For flat-top hexagon (side up), normals are at 0, 60, 120, 180, 240, 300 degrees
    
    # Let's use flat-top orientation (vertex at right)
    # Normals at 0, 60, 120, 180, 240, 300 degrees
    
    # The required apothem A must satisfy for all points and all normals:
    # dot(point - center, normal) + A_inner <= A
    # So A >= max over all points and normals of (dot(point - center, normal) + A_inner)
    
    A_inner = math.sqrt(3) / 2
    
    # Try center at (0,0) and various rotations
    for theta_deg in range(0, 360, 10):
        theta = math.radians(theta_deg)
        # Normals for flat-top hexagon rotated by theta
        base_normals = [math.radians(0), math.radians(60), math.radians(120), 
                        math.radians(180), math.radians(240), math.radians(300)]
        normals = [b + theta for b in base_normals]
        
        max_proj = 0
        for px, py in points:
            for n in normals:
                proj = px * math.cos(n) + py * math.sin(n)
                if proj > max_proj:
                    max_proj = proj
        
        A_required = max_proj + A_inner
        S_required = A_required * 2 / math.sqrt(3)
        
        if S_required < best_S:
            best_S = S_required
            best_theta = theta_deg
    
    # Round up to ensure validity
    side = math.ceil(best_S * 1000) / 1000.0
    
    return inner, (0.0, 0.0), side, best_theta
# EVOLVE-BLOCK-END
