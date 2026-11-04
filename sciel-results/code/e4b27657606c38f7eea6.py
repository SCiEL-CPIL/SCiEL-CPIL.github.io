import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # The theoretical minimum side length for packing N unit hexagons into a larger hexagon
    # is determined by area: S >= sqrt(N). For N=12, S >= sqrt(12) ~ 3.46? 
    # Wait, area of hex is (3*sqrt(3)/2)*S^2.
    # 12 * (3*sqrt(3)/2)*1^2 <= (3*sqrt(3)/2)*S^2 => 12 <= S^2 => S >= sqrt(12) ~ 3.46.
    # However, geometric constraints are stricter.
    # A hexagon of side S can tile S^2 unit hexagons? No, 3S(S-1)+1.
    # For S=4, count = 3*4*3+1 = 37.
    # For S=3, count = 3*3*2+1 = 19.
    # For S=2, count = 3*2*1+1 = 7.
    # So S=3 can fit 19 hexagons. We need 12.
    # So S=3 is sufficient in terms of count.
    # Can S be smaller than 3?
    # If S < 3, say S=2.9. Area bound allows S>=3.46? No, S>=sqrt(12)=3.46.
    # Wait, my previous area calculation:
    # Area(1 unit) = 2.598. Area(12) = 31.17.
    # Area(S) = 2.598 * S^2.
    # 2.598 * S^2 >= 31.17 => S^2 >= 12 => S >= 3.464.
    # So S must be at least 3.464.
    # Since S=3 only fits 19 hexagons in a *tiling*, but can we pack 12 in S=3.46?
    # The tiling argument shows S=3 fits 19, so S=3 is a valid upper bound.
    # But S=3 < 3.464? No, 3 < 3.464.
    # Contradiction?
    # Area of S=3 hex: 2.598 * 9 = 23.38.
    # Area of 12 unit hexes: 31.17.
    # 23.38 < 31.17. So 12 unit hexes CANNOT fit in a side-3 hexagon.
    # My tiling count formula: 3S(S-1)+1.
    # S=3 -> 19 hexes.
    # Area of 19 unit hexes = 19 * 2.598 = 49.36.
    # Area of S=3 hex = 23.38.
    # This implies my tiling formula or area formula is wrong.
    
    # Let's re-evaluate Area of Hexagon.
    # Side a. Area = (3*sqrt(3)/2) * a^2.
    # Unit hex (a=1): Area = 2.598.
    # S=3 hex: Area = 2.598 * 9 = 23.38.
    # How many unit hexes fit in S=3?
    # Visualizing a hex of side 3.
    # Rows: 3, 4, 5, 4, 3? No.
    # Side 1: 1 hex.
    # Side 2: 7 hexes (1 center, 6 around).
    # Side 3: 19 hexes.
    # Let's sum areas: 19 * 2.598 = 49.36.
    # This is double the area of the container (23.38).
    # Why?
    # Ah, the "side length" of the tiling hexagons vs the container.
    # If we tile a hex of side N with unit hexes, the unit hexes are smaller?
    # No, if we place unit hexes (side 1) in a hex of side N (side N),
    # The distance from center to vertex of container is N.
    # The distance from center to vertex of unit hex is 1.
    # In a tiling, the unit hexes are arranged such that their centers are at distance sqrt(3) apart.
    # The number of hexes in a hex of side N (in grid steps) is 3N(N-1)+1.
    # But what is the physical side length of the container that fits this grid?
    # If the grid has "radius" N-1 steps, the physical extent is larger.
    # Let's check S=1 (1 hex). Fits 1 unit hex. Side length 1?
    # If we place 1 unit hex in a container of side 1, it fits perfectly.
    # Area(1) = 2.598. Area(1) = 2.598. OK.
    # S=2 (7 hexes).
    # If container side is 2, Area = 2.598 * 4 = 10.39.
    # 7 unit hexes Area = 7 * 2.598 = 18.18.
    # 18.18 > 10.39. So 7 unit hexes do NOT fit in a side-2 hexagon.
    # So the tiling formula 3S(S-1)+1 applies to a container of side S *in grid units*,
    # but the physical side length of that container is NOT S.
    
    # Let's find the physical side length $L$ that fits $K$ unit hexes in a hexagonal cluster.
    # A cluster of radius $R$ (in grid steps, i.e., max distance from center is R) has $3R(R+1)+1$ hexes?
    # R=0: 1 hex.
    # R=1: 7 hexes.
    # R=2: 19 hexes.
    # R=3: 37 hexes.
    # We need 12 hexes.
    # R=1 gives 7. R=2 gives 19.
    # So we can fit 12 hexes in a cluster of radius R=2 (since 12 < 19).
    # What is the physical side length of the container that fits this cluster?
    # The cluster of radius R=2 extends to distance $2 \times \sqrt{3}$ from center along the grid axes?
    # Distance between centers is $\sqrt{3}$.
    # Furthest center is at distance $2\sqrt{3}$.
    # The hexagon itself has circumradius 1.
    # So the furthest point of the packing is $2\sqrt{3} + 1 \approx 3.46 + 1 = 4.46$.
    # The container is a regular hexagon.
    # If we align the container with the packing (angle 0), the container must have a circumradius $R_{out} \ge 4.46$.
    # For a regular hexagon, Side $S = R_{out}$.
    # So $S \ge 4.46$.
    
    # Can we do better with rotation or different packing?
    # Area bound: $S \ge \sqrt{12} \approx 3.46$.
    # Current estimate 4.46.
    
    # Let's try to pack 12 hexes more tightly.
    # Maybe a rectangular arrangement?
    # Or just accept the hexagonal cluster.
    # Is there a packing with S < 4.46?
    # Consider a "staircase" or other shapes.
    # However, for N=12, the hexagonal cluster of R=2 is very compact.
    # Let's calculate the required S for the R=2 cluster exactly.
    # The bounding hexagon of the R=2 cluster (aligned) has side length?
    # The cluster fits in a hexagon of grid-side 3?
    # Grid side $G$ contains $3G(G-1)+1$ hexes? No, that formula was for the number of hexes in a hex of side G.
    # If G=3, count=19.
    # The physical side length of a container that fits a grid-hex of side G is $L = G \times \sqrt{3}$?
    # Let's check G=1. 1 hex. Physical side 1?
    # If we put 1 unit hex in a container, min side is 1.
    # Formula $L = G \sqrt{3}$? $1 \times 1.732 = 1.732$. Too big.
    # Actually, if we place 1 unit hex in a container, the container side must be at least 1 (if aligned).
    # If we place 7 hexes (G=2 grid), the physical extent is larger.
    # Distance from center to vertex of outer ring hexes:
    # Center to outer ring center: $\sqrt{3}$.
    # Outer ring hex vertex: $1$.
    # Total distance: $\sqrt{3} + 1 \approx 2.732$.
    # So container circumradius must be $\ge 2.732$.
    # Side $S \ge 2.732$.
    # Area check: $S=2.732 \implies S^2 \approx 7.46$. Area $\approx 19.4$.
    # 7 hexes Area = 18.18. Fits.
    
    # For 12 hexes, we need a shape that fits 12.
    # The R=2 cluster (19 hexes) has max distance $\sqrt{3} \times 2 + 1 = 2\sqrt{3} + 1 \approx 4.464$.
    # So $S \ge 4.464$.
    
    # Can we pack 12 hexes in a smaller container?
    # Try a 3x4 rectangular-ish block.
    # Or a "brick wall".
    # Let's try to minimize the bounding hexagon side.
    # The optimal packing for small N is often a hexagonal cluster or a distorted one.
    # For N=12, the hexagonal cluster of R=1 (7 hexes) + 5 more.
    # Adding 5 to the R=1 ring creates an asymmetric shape.
    # The max distance might be less than the full R=2 ring.
    # In the R=2 ring, all 12 positions are at distance $2\sqrt{3}$ from center.
    # If we pick only 5 of them, we might reduce the max distance?
    # No, the distance is determined by the grid position. All R=2 positions are at the same radial distance in the hex grid metric, but in Euclidean space, they are on a circle of radius $2\sqrt{3}$ (approx).
    # Actually, the vertices of the hex grid at distance 2 form a hexagon.
    # The Euclidean distance is constant for all 12 vertices of the "hexagon" of radius 2?
    # Yes, by symmetry.
    # So any hex at R=2 adds the same "extremeness".
    # So the bounding box is determined by the presence of ANY R=2 hex.
    # To minimize S, we should avoid R=2 hexes if possible?
    # But we have 7 in R<=1. We need 5 more.
    # We MUST use R=2 hexes.
    # So the max distance is fixed at $2\sqrt{3} + 1$.
    # Thus $S \ge 2\sqrt{3} + 1 \approx 4.464$.
    
    # Is it possible to rotate the inner hexes to reduce the bounding box?
    # If we rotate the inner hexes, their "radius" (max distance from center to vertex) remains 1.
    # The center positions are fixed by the packing.
    # So the bounding box of the union is determined by the centers + 1.
    # So the minimum S is determined by the minimum enclosing hexagon of the set of centers, expanded by 1.
    # The minimum enclosing hexagon of the 12 centers (7 in R1, 5 in R2) will have a circumradius equal to the max distance of the centers from the centroid (if we center it optimally) + 1.
    # If we center the 12 hexes at the origin, the max distance is $2\sqrt{3} \approx 3.46$.
    # So S must be at least $3.46 + 1 = 4.46$.
    
    # Can we shift the cluster to reduce the max distance from the container center?
    # The container is a hexagon. We can place the cluster anywhere inside.
    # To minimize the container size, we should place the cluster such that its "bounding hexagon" is as small as possible.
    # The bounding hexagon of the 12 centers (aligned with grid) has "grid radius" 2.
    # The physical side length of the minimal enclosing hexagon for a grid-hex of radius 2 is?
    # It is $2\sqrt{3} + 1$?
    # Let's check the distance from the center of the grid-hex to the furthest vertex of the furthest unit hex.
    # Center of grid-hex is at a unit hex center.
    # Furthest unit hex center is at distance $2\sqrt{3}$.
    # Furthest vertex of that hex is at distance $2\sqrt{3} + 1$.
    # So the circumradius of the container must be $\ge 2\sqrt{3} + 1$.
    # Side $S = 2\sqrt{3} + 1 \approx 4.464$.
    
    # Can we do better with a non-hexagonal cluster?
    # E.g. a 2x6 rectangle?
    # Centers of a 2x6 rectangle of hexes.
    # Width: 5 steps?
    # This will likely be longer.
    
    # So 4.464 is a strong candidate.
    # Let's verify if we can fit 12 hexes in S=4.464.
    # Yes, by placing the R=2 cluster (19 hexes) and removing 7.
    # The remaining 12 are within the bounding box.
    
    # Is there a packing with S < 4.464?
    # Area bound is 3.46.
    # If we rotate the inner hexes by 30 degrees, their "width" changes.
    # Inradius becomes 1, Circumradius becomes $2/\sqrt{3} \approx 1.15$.
    # This increases the extent. So angle 0 is best.
    
    # So the optimal S is likely $2\sqrt{3} + 1$.
    
    import math
    S = 2 * math.sqrt(3) + 1
    
    # Construct 12 hexes
    # 7 in R<=1, 5 in R=2
    hexes = []
    # R=0
    hexes.append((0, 0))
    # R=1
    hexes.append((1, 0))
    hexes.append((0, 1))
    hexes.append((-1, 1))
    hexes.append((-1, 0))
    hexes.append((0, -1))
    hexes.append((1, -1))
    # R=2 (pick 5)
    hexes.append((2, 0))
    hexes.append((1, 1))
    hexes.append((0, 2))
    hexes.append((-1, 2))
    hexes.append((-2, 1))
    
    inner = []
    for q, r in hexes:
        x = 1.5 * q
        y = math.sqrt(3) * (r + 0.5 * q)
        inner.append([x, y, 0.0])
        
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
