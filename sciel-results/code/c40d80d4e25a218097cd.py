import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Calculate minimal integer side length S such that 3*S^2 - 3*S + 1 >= N
    # 3S^2 - 3S + 1 - N >= 0
    # S = (3 + sqrt(9 - 12(1-N))) / 6 = (3 + sqrt(12N - 3)) / 6
    disc = 12 * N - 3
    if disc < 0:
        S = 1
    else:
        S = math.ceil((3 + math.sqrt(disc)) / 6.0)
    
    # Ensure S is at least 1
    if S < 1:
        S = 1
        
    # Generate coordinates for a hexagonal lattice of side S
    # Using axial coordinates (q, r) where:
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    # Condition for being inside hexagon of side S:
    # |q| <= S-1, |r| <= S-1, |q+r| <= S-1
    
    inner = []
    sqrt3 = math.sqrt(3)
    
    # Iterate through all valid (q, r)
    # r ranges from -(S-1) to S-1
    for r in range(-(S - 1), S):
        # q ranges such that |q| <= S-1 and |q+r| <= S-1
        # => - (S-1) <= q <= S-1
        # => - (S-1) - r <= q <= (S-1) - r
        q_min = max(-(S - 1), -(S - 1) - r)
        q_max = min(S - 1, (S - 1) - r)
        
        for q in range(q_min, q_max + 1):
            x = sqrt3 * (q + r / 2.0)
            y = 1.5 * r
            # Angle 0 means aligned with the outer hexagon
            inner.append([x, y, 0.0])
            
            # Stop if we have collected N hexagons
            if len(inner) >= N:
                break
        if len(inner) >= N:
            break
            
    # Trim to exactly N if we collected more (though the loop breaks)
    inner = inner[:N]
    
    # Outer hexagon parameters
    center = (0.0, 0.0)
    side = float(S)
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
