import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Generate the 12 center positions
    # 7 central points (hexagon of side 1 in lattice units)
    centers = []
    # (0,0)
    centers.append((0.0, 0.0))
    
    # 6 neighbors at distance sqrt(3)
    # Lattice vectors u=(sqrt(3),0), v=(sqrt(3)/2, 3/2)
    # Neighbors are at i*u + j*v where i,j in {-1,0,1} and not both 0, and |i|+|j|+|i-j| <= 2?
    # Actually just the 6 closest.
    for i, j in [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]:
        x = i * math.sqrt(3) + j * (math.sqrt(3) / 2)
        y = i * 0 + j * (3 / 2)
        centers.append((x, y))
        
    # 5 more points from the "edge" positions of the second ring
    # The 6 edge positions are at lattice coords (1,1), (2,-1)? No.
    # Let's use the vector calculation for the 6 "edge" midpoints of the hexagonal cell.
    # These correspond to lattice points (1,1), (1,-2)? 
    # Let's just generate the 12 points of the 2nd ring and pick the ones with lower projection.
    # The 12 points of the 2nd ring (distance 3 from origin) are:
    # (2,0), (1,1), (-1,2), (-2,2), (-2,0), (-1,-1), (1,-2), (2,-2), (2,0)... wait.
    # Lattice points (i,j) with i^2+j^2+ij = 3.
    # (1,1): 1+1+1=3. x=1.5sqrt(3), y=1.5.
    # (1,-2): 1+4-2=3. x=sqrt(3)-sqrt(3)=0, y=-3.
    # (2,-1): 4+1-2=3. x=2sqrt(3)-sqrt(3)/2=1.5sqrt(3), y=-1.5.
    # (-1,2): -1+4-2=1? No. -1+4-(-1*2)=1+4+2=7? No.
    # Formula: dist^2 = 3(i^2+j^2+ij).
    # We want dist=3, so i^2+j^2+ij=3.
    # Solutions:
    # (1,1) -> 3
    # (1,-2) -> 1+4-2=3
    # (-1,2) -> 1+4-(-2)=7 (No)
    # (-1,-1) -> 1+1+1=3
    # (-2,1) -> 4+1-2=3
    # (2,-1) -> 4+1-2=3
    # (2,1) -> 4+1+2=7 (No)
    # (-2,-1) -> 4+1+2=7 (No)
    # (0, sqrt(3))? No integer.
    
    # The 6 solutions for i^2+j^2+ij=3 are:
    # (1,1), (1,-2), (-1,-1), (-1,2)? Let's check (-1,2): 1+4-2=3. Yes.
    # (-2,1): 4+1-2=3. Yes.
    # (2,-1): 4+1-2=3. Yes.
    
    # So the 6 points are:
    edge_coords = [(1, 1), (1, -2), (-1, -1), (-1, 2), (-2, 1), (2, -1)]
    
    # We need 5 of these. We pick the first 5.
    for i, j in edge_coords[:5]:
        x = i * math.sqrt(3) + j * (math.sqrt(3) / 2)
        y = i * 0 + j * (3 / 2)
        centers.append((x, y))
        
    # Now we have 12 centers.
    # Calculate the required side length L' of the safe region (hexagon containing centers).
    # The safe region is a regular hexagon centered at origin.
    # We need to find the minimum L' such that all centers are inside.
    # This is equivalent to finding the maximum projection of any center onto the 6 normal vectors.
    max_proj = 0.0
    for angle_deg in range(0, 360, 60):
        angle_rad = math.radians(angle_deg)
        nx = math.cos(angle_rad)
        ny = math.sin(angle_rad)
        for x, y in centers:
            proj = x * nx + y * ny
            if proj > max_proj:
                max_proj = proj
                
    # max_proj is the required apothem a'
    a_prime = max_proj
    # Side length L' = 2 * a' / sqrt(3)
    L_prime = 2 * a_prime / math.sqrt(3)
    
    # Total side length L = L' + sqrt(3)
    L = L_prime + math.sqrt(3)
    
    # Construct the output
    inner = []
    for x, y in centers:
        inner.append([x, y, 0.0])
        
    return inner, (0.0, 0.0), L, 0.0
# EVOLVE-BLOCK-END
