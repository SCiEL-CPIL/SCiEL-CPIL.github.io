import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # We use a hexagonal lattice packing.
    # Inner hexagons are aligned with the outer hexagon (angle 0).
    # To ensure containment, the centers must be at least the apothem of the unit hexagon
    # (sqrt(3)/2) away from the sides of the outer hexagon.
    # This is equivalent to packing the centers in a smaller hexagon of side S - 1.
    # Let K = S - 1. The apothem of the center-hexagon is K * sqrt(3) / 2.
    # We need to fit 12 points with minimum distance sqrt(3) in this hexagon.
    # The triangular lattice points at distance 0, sqrt(3), and 3 from the origin are:
    # 1 at dist 0
    # 6 at dist sqrt(3)
    # 6 at dist 3
    # To fit the points at dist 3, we need K * sqrt(3) / 2 >= 3 => K >= 2 * sqrt(3).
    # So S = K + 1 >= 1 + 2 * sqrt(3) approx 4.464.
    
    side = 1.0 + 2.0 * np.sqrt(3.0)
    
    # Generate centers on a triangular lattice
    # v1 = (sqrt(3), 0), v2 = (sqrt(3)/2, 1.5)
    v1 = np.array([np.sqrt(3.0), 0.0])
    v2 = np.array([np.sqrt(3.0)/2.0, 1.5])
    
    centers = []
    # Add center
    centers.append([0.0, 0.0])
    
    # Add 6 neighbors (ring 1)
    for i in range(6):
        angle = np.deg2rad(60 * i)
        c = np.array([np.sqrt(3.0) * np.cos(angle), np.sqrt(3.0) * np.sin(angle)])
        centers.append(c.tolist())
        
    # Add 5 points from ring 2 (dist 3)
    # Ring 2 points are at angles 0, 60, 120, 180, 240, 300 but further out?
    # Actually, the 6 points at dist 3 are at axial coords (2,0), (1,1), (0,2), (-1,1), (-2,0), (-1,-1) etc?
    # Let's just generate a larger grid and pick the 12 closest to origin that satisfy min dist.
    # Simpler: Just use the 1 center, 6 ring 1, and 5 from the 6 "corners" of the hexagon of side 2 in lattice units.
    # The 6 points at cartesian distance 3 are:
    ring2 = []
    for i in range(6):
        angle = np.deg2rad(60 * i)
        c = np.array([3.0 * np.cos(angle), 3.0 * np.sin(angle)])
        ring2.append(c.tolist())
        
    # We need 5 more. Take the first 5 from ring2.
    centers.extend(ring2[:5])
    
    inner = []
    for c in centers:
        inner.append([c[0], c[1], 0.0])
        
    return inner, (0.0, 0.0), side, 0.0
# EVOLVE-BLOCK-END
