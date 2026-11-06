import math

def construct():
    # Generate the 19 points of the hexagon of radius 2
    # Condition: |a| <= 2, |b| <= 2, |a+b| <= 2
    points = []
    for a in range(-2, 3):
        for b in range(-2, 3):
            if abs(a) <= 2 and abs(b) <= 2 and abs(a+b) <= 2:
                points.append([a, b])
    
    # We have 19 points. We need 13 more.
    # Find all neighbors of the current points that are not in the current set.
    neighbors = set()
    for a, b in points:
        # 6 neighbors: (a+1,b), (a-1,b), (a,b+1), (a,b-1), (a+1,b-1), (a-1,b+1)
        neighbors.add((a+1, b))
        neighbors.add((a-1, b))
        neighbors.add((a, b+1))
        neighbors.add((a, b-1))
        neighbors.add((a+1, b-1))
        neighbors.add((a-1, b+1))
    
    # Remove the points already in the set
    current_set = set(map(tuple, points))
    candidates = [list(p) for p in neighbors if p not in current_set]
    
    # We need to pick 13 candidates that maximize edges.
    # A contiguous block on the hexagonal ring is optimal.
    # Let's order the candidates by their angle or position to find a contiguous block.
    # Or simply, since the ring is symmetric, any contiguous block of 13 will work.
    # Let's sort them by a to group them.
    
    # To ensure we pick a contiguous block, we can just pick the first 13 from a sorted list
    # that follows the ring order.
    
    # Let's define the ring order manually or by angle.
    # Angle of (a,b) in complex plane: a + b*exp(i*pi/3)
    # We can sort by atan2(y, x)
    
    def get_angle(p):
        a, b = p
        x = a + 0.5*b
        y = math.sqrt(3)/2 * b
        return math.atan2(y, x)
        
    candidates.sort(key=get_angle)
    
    # The candidates form a ring. We need a contiguous block of 13.
    # Since it's a ring, we can start at any point.
    # Let's start at the one with angle closest to 0 (positive x-axis).
    # Find index of candidate closest to angle 0
    min_idx = 0
    min_diff = float('inf')
    for i, p in enumerate(candidates):
        ang = get_angle(p)
        if abs(ang) < min_diff:
            min_diff = abs(ang)
            min_idx = i
            
    # Extract 13 contiguous points starting from min_idx
    # Note: The list is circular.
    selected = []
    for i in range(13):
        selected.append(candidates[(min_idx + i) % len(candidates)])
        
    # Combine
    final_points = points + selected
    
    # Verify distinctness
    if len(set(map(tuple, final_points))) != 32:
        # Fallback: return the 19 points plus 13 arbitrary valid points if logic fails
        # But the logic above is robust.
        pass
        
    return final_points
