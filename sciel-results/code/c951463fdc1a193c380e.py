import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Generate centers for 11 hexagons
    # 1 center
    centers = [(0.0, 0.0)]
    
    # 6 neighbors at distance sqrt(3)
    for i in range(6):
        angle = math.radians(i * 60)
        x = math.sqrt(3) * math.cos(angle)
        y = math.sqrt(3) * math.sin(angle)
        centers.append((x, y))
        
    # 4 more at distance 2*sqrt(3)
    # Pick 0, 60, 120, 180 to keep it somewhat compact
    for i in range(4):
        angle = math.radians(i * 60)
        x = 2 * math.sqrt(3) * math.cos(angle)
        y = 2 * math.sqrt(3) * math.sin(angle)
        centers.append((x, y))
        
    # Ensure we have exactly N centers
    # If N > 11, we'd need more logic, but N is fixed at 11 in the prompt context?
    # The prompt says N=11 in the seed. I will assume N is fixed or I should handle general N?
    # The seed has N=11 hardcoded. I will stick to N=11 logic but make it robust if N changes slightly?
    # The task says "Place disjoint regular hexagons ... of n unit hexagons".
    # The seed defines N=11. I will assume N is 11.
    
    inner = []
    all_vertices = []
    
    for cx, cy in centers:
        # Angle 0 for inner hexagons
        inner.append([cx, cy, 0.0])
        
        # Generate vertices of this hexagon
        for j in range(6):
            angle = math.radians(j * 60)
            vx = cx + math.cos(angle)
            vy = cy + math.sin(angle)
            all_vertices.append((vx, vy))
            
    # Calculate minimal side length S for outer hexagon (aligned with inner, angle 0)
    # Conditions: |x| <= S, |x + y/sqrt(3)| <= S, |x - y/sqrt(3)| <= S
    inv_sqrt3 = 1.0 / math.sqrt(3)
    max_s = 0.0
    for vx, vy in all_vertices:
        s1 = abs(vx)
        s2 = abs(vx + vy * inv_sqrt3)
        s3 = abs(vx - vy * inv_sqrt3)
        local_max = max(s1, s2, s3)
        if local_max > max_s:
            max_s = local_max
            
    # Add a tiny epsilon to ensure containment due to floating point
    side_length = max_s + 1e-6
    
    return inner, (0.0, 0.0), side_length, 0.0
# EVOLVE-BLOCK-END
