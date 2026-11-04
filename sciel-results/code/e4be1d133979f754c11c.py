"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    """
    Place 12 unit regular hexagons inside a regular hexagon, minimizing outer side length.
    
    Strategy: Use a symmetric hexagonal cluster arrangement.
    - 1 hexagon at center.
    - 6 hexagons in the first ring (hexagonal lattice neighbors).
    - 5 hexagons from the second ring, chosen to minimize the bounding hexagon side length.
    
    A full second ring would require L=5. By omitting one hexagon from the second ring
    (specifically the one on the positive x-axis), we can reduce the required outer side length.
    The remaining second-ring hexagons are at a radial distance of 2*sqrt(3) ~ 3.464.
    The vertex of a hexagon at distance d from center (aligned with lattice) extends to d + 1.
    So max extent is ~4.464. However, due to the shape of the hexagon, the bounding box
    in the hexagonal metric (Chebyshev distance in axial coords) determines the outer side.
    
    For a hexagon at axial coordinate (q, r), the required outer hexagon side L is
    max(|q|, |r|, |q+r|) + 1.
    
    Lattice coordinates for the chosen packing:
    Center: (0, 0)
    Ring 1: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    Ring 2 (full): (2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-1), (-2,0), (-2,1), (-2,2), (-1,2), (0,2), (1,1)
    
    We need 5 more from Ring 2. Total 12.
    Let's pick 5 from Ring 2 that are symmetric.
    Full Ring 2 has 12 hexes. We need to remove 7.
    Actually, it's easier to think in terms of the bounding box.
    The max L for any hex at (q,r) is max(|q|,|r|,|q+r|) + 1.
    For Ring 2, q,r values are in {-2,-1,0,1,2} with |q+r|<=2 and not both 0.
    Max of max(|q|,|r|,|q+r|) for Ring 2 is 2.
    So any single Ring 2 hex requires L >= 2+1 = 3.
    Wait, the center is (0,0) -> L>=1.
    Ring 1: max val 1 -> L>=2.
    Ring 2: max val 2 -> L>=3.
    
    So if we use ANY hex from Ring 2, the outer side must be at least 3.
    Can we do L=3?
    If L=3, the outer hexagon covers all lattice points with max(|q|,|r|,|q+r|) <= 2.
    This is exactly the set of hexagons at distance <= 2 from center (the k=2 cluster).
    The k=2 cluster has 1 + 6 + 12 = 19 hexagons.
    We need to place 12 of them. Since all 19 fit inside L=3, we can just pick any 12 of them.
    To be safe and valid, we pick the first 12 in a deterministic order.
    The outer side length L=3 is valid.
    """
    # Generate all hexagons in the k=2 cluster (L=3 capacity)
    hexes = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                hexes.append((q, r))
    
    # We need 12 hexagons. The cluster has 19.
    # We just take the first 12.
    # To ensure symmetry and avoid edge cases, let's take the center, ring 1, and part of ring 2.
    # Center: (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # That's 7. We need 5 more from Ring 2.
    # Ring 2: (2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-2), (-2,-1), (-2,0), (-2,1), (-2,2), (-1,2), (0,2)
    # Let's take 5 from Ring 2.
    
    selected = [(0,0)]
    ring1 = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    selected.extend(ring1)
    ring2 = [(2,0), (2,-1), (2,-2), (1,-2), (0,-2)]
    selected.extend(ring2)
    
    # Convert axial coordinates to Cartesian for output
    # Axial to Cartesian:
    # x = 1.5 * q
    # y = sqrt(3) * (r + 0.5 * q)
    inner = []
    for q, r in selected:
        x = 1.5 * q
        y = math.sqrt(3) * (r + 0.5 * q)
        angle = 0.0
        inner.append([x, y, angle])
    
    center = (0.0, 0.0)
    side = 3.0
    outer_angle = 0.0
    
    return inner, center, side, outer_angle
# EVOLVE-BLOCK-END
