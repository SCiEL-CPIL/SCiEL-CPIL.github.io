import time
from collections import Counter
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin
    
    # Seed configuration: 15 points on y=0, 15 on y=1
    # This gives doubled area 1 for 15*15=225 triangles
    seed_points = [(i, 0) for i in range(15)] + [(i, 1) for i in range(15)]
    
    def evaluate(points):
        """Compute the maximum multiplicity of any positive doubled area."""
        counts = Counter()
        for i in range(len(points)):
            x1, y1 = points[i]
            for j in range(i+1, len(points)):
                x2, y2 = points[j]
                for k in range(j+1, len(points)):
                    x3, y3 = points[k]
                    # Doubled area = |det|
                    area2 = abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
                    if area2 > 0:
                        counts[area2] += 1
        if not counts:
            return 0
        return max(counts.values())
    
    # Evaluate seed
    best_points = seed_points
    best_score = evaluate(seed_points)
    # print(f"Seed score: {best_score}")
    
    # Try to improve by moving points
    # Strategy: Try to increase multiplicity of a specific area
    # The key insight: if we have points on two parallel lines, the doubled area
    # for a triangle with 2 points on one line and 1 on the other is proportional
    # to the distance between the lines times the base length.
    
    # Let's try different configurations:
    # 1. Points on two parallel lines at distance d apart
    #    If we have a points on line 1 and b on line 2, with a+b=30
    #    Triangles with 2 on line 1, 1 on line 2: area = d * |x_i - x_j| / 2
    #    Doubled area = d * |x_i - x_j|
    #    To maximize multiplicity, we want many pairs with the same |x_i - x_j|
    
    # For consecutive integers, the number of pairs with difference k is (n-k)
    # So if we have n consecutive integers, the most common difference is 1,
    # with (n-1) pairs.
    
    # With a points on line 1 and b on line 2:
    # Triangles with 2 on line 1, 1 on line 2: for each pair on line 1 with diff d,
    # and each point on line 2, we get doubled area = D * d where D is line distance
    # Count for area D*d = (a-d) * b
    # Triangles with 1 on line 1, 2 on line 2: count for area D*d = a * (b-d)
    # Total for area D*d = (a-d)*b + a*(b-d) = a*b - a*d + b*a - b*d = 2ab - d(a+b) = 2ab - 30d
    
    # To maximize, we want small d. d=1 gives 2ab - 30
    # Maximize 2ab subject to a+b=30: a=b=15, gives 2*225-30 = 420
    
    # But we can also have triangles with all 3 on the same line (area 0, excluded)
    # So the maximum multiplicity for area D is 2ab - 30D when D is the line distance
    
    # Wait, that's not quite right. Let me reconsider.
    # If lines are at y=0 and y=D, with a points on y=0 at x=0..a-1 and b points on y=D at x=0..b-1
    # Triangle with points (x1,0), (x2,0), (x3,D): doubled area = D*|x1-x2|
    # Triangle with points (x1,0), (x2,D), (x3,D): doubled area = D*|x2-x3|
    
    # So for doubled area = D*k:
    # From first type: (a-k) * b  [pairs on line 1 with diff k, times b points on line 2]
    # From second type: a * (b-k)  [a points on line 1, times pairs on line 2 with diff k]
    # Total: (a-k)*b + a*(b-k) = ab - bk + ab - ak = 2ab - k(a+b) = 2ab - 30k
    
    # So for k=1: 2ab - 30
    # Maximized at a=b=15: 450 - 30 = 420
    
    # Can we do better with non-consecutive or 3 lines?
    
    # Let's try 3 parallel lines. If we have a,b,c points on 3 lines at y=0, y=D, y=2D
    # Triangles with 2 on one line, 1 on another:
    # Lines 0 and D: doubled area = D * diff, count = (a-diff)*b + a*(b-diff)
    # Lines 0 and 2D: doubled area = 2D * diff, count = (a-diff)*c + a*(c-diff)
    # Lines D and 2D: doubled area = D * diff, count = (b-diff)*c + b*(c-diff)
    
    # So area D*1 gets contributions from lines 0-D and D-2D:
    # (a-1)*b + a*(b-1) + (b-1)*c + b*(c-1) = 2ab - a - b + 2bc - b - c = 2ab + 2bc - a - 2b - c
    # = 2b(a+c) - a - 2b - c
    
    # With a+b+c=30, let a=c (symmetric): 2b(2a) - a - 2b - a = 4ab - 2a - 2b
    # = 2(2ab - a - b) with a+b=30-b... this gets complex.
    
    # Let's just try a=b=c=10:
    # Area D: 2*10*10 + 2*10*10 - 10 - 2*10 - 10 = 200+200-10-20-10 = 360
    # Less than 420.
    
    # What about a=12, b=6, c=12?
    # Area D: 2*12*12 + 2*6*12 - 12 - 2*6 - 12 = 288+144-12-12-12 = 396
    # Still less.
    
    # What about making the points non-consecutive to create more pairs with the same difference?
    # If we use points at 0,1,2,...,n-1, the number of pairs with difference k is n-k.
    # If we use a different set, we might get more pairs with difference 1.
    # But the maximum number of pairs with difference 1 in a set of n integers is n-1
    # (achieved by consecutive integers).
    
    # So 420 seems to be the upper bound for 2 parallel lines.
    
    # Let's verify the seed gives 420.
    # Actually, let's just run a local search to see if we can beat 420.
    
    # Try swapping a point
    import random
    random.seed(42)
    
    for iteration in range(2000):
        if time.time() > deadline:
            break
        
        # Random mutation: move one point
        new_points = list(best_points)
        idx = random.randint(0, 29)
        old_x, old_y = new_points[idx]
        new_x = random.randint(-30, 30)
        new_y = random.randint(-30, 30)
        new_points[idx] = (new_x, new_y)
        
        # Check distinctness
        if len(set(new_points)) != 30:
            continue
        
        # Check bounds
        if any(not (-30 <= px <= 30 and -30 <= py <= 30) for px, py in new_points):
            continue
        
        score = evaluate(new_points)
        if score > best_score:
            best_score = score
            best_points = new_points
            # print(f"Improved to {best_score} at iteration {iteration}")
    
    # Try more targeted search: try to increase multiplicity of area 1
    # by ensuring many triangles have doubled area exactly 1
    
    # The seed already gives 420 for area 1.
    # Let's try to see if we can get any area to have more than 420.
    
    # Try configurations with points on a grid
    for w in range(3, 11):
        for h in range(3, 11):
            if w * h < 30:
                continue
            if w * h > 30:
                # Take first 30 points of w x h grid
                pts = []
                for x in range(w):
                    for y in range(h):
                        pts.append((x, y))
                        if len(pts) == 30:
                            break
                    if len(pts) == 30:
                        break
            else:
                pts = [(x, y) for x in range(w) for y in range(h)]
            
            if len(pts) != 30:
                continue
            if len(set(pts)) != 30:
                continue
            if any(not (-30 <= px <= 30 and -30 <= py <= 30) for px, py in pts):
                continue
            
            score = evaluate(pts)
            if score > best_score:
                best_score = score
                best_points = pts
    
    return best_points
