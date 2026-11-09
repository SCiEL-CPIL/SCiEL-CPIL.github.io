import random
import itertools
import time
from math import gcd

def count_exact_4_lines(points):
    """
    Counts the number of lines that contain exactly 4 of the given points.
    Returns 0 if any line contains 5 or more points.
    """
    n = len(points)
    if n != 16:
        return 0
    
    # Check for duplicates
    if len(set(map(tuple, points))) != 16:
        return 0

    lines = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i+1, n):
            x2, y2 = points[j]
            # a*x + b*y + c = 0
            a = y2 - y1
            b = -(x2 - x1)
            c = x1 * y2 - x2 * y1
            
            # Normalize
            g = gcd(gcd(abs(a), abs(b)), abs(c))
            if g == 0:
                continue # Should not happen for distinct points
            a //= g
            b //= g
            c //= g
            
            # Ensure canonical sign
            if a < 0:
                a = -a
                b = -b
                c = -c
            elif a == 0:
                if b < 0:
                    b = -b
                    c = -c
                elif b == 0:
                    if c < 0:
                        c = -c
                        
            key = (a, b, c)
            if key not in lines:
                lines[key] = 0
            lines[key] += 1
            
    count = 0
    for key, cnt in lines.items():
        # cnt is the number of pairs on the line.
        # If k points are on the line, there are k*(k-1)/2 pairs.
        # We need to find k from cnt.
        # k=2 -> 1 pair
        # k=3 -> 3 pairs
        # k=4 -> 6 pairs
        # k=5 -> 10 pairs
        # k=6 -> 15 pairs
        # ...
        # We can check if cnt corresponds to a valid k.
        # If cnt is not of the form k*(k-1)/2, something is wrong? 
        # No, wait. The dictionary counts pairs. 
        # If a line has 4 points, it contributes 6 pairs.
        # If a line has 3 points, it contributes 3 pairs.
        # If a line has 2 points, it contributes 1 pair.
        # We need to identify the number of points k on the line.
        
        # Actually, the previous method using sets was clearer for determining k.
        # Let's revert to the set method but keep it tight.
        pass
        
    # The pair counting method requires solving k*(k-1)/2 = cnt.
    # This is possible but tricky if multiple lines share pairs? No, each pair belongs to exactly one line.
    # So for each line key, cnt is exactly the number of pairs among the points on that line.
    # So we can recover k.
    
    count = 0
    for key, cnt in lines.items():
        # Find k such that k*(k-1)/2 == cnt
        # k=1 -> 0 (impossible for a line defined by 2 pts)
        # k=2 -> 1
        # k=3 -> 3
        # k=4 -> 6
        # k=5 -> 10
        # k=6 -> 15
        # k=7 -> 21
        # Max k is 16 -> 120 pairs.
        
        # We can just check:
        if cnt == 1:
            k = 2
        elif cnt == 3:
            k = 3
        elif cnt == 6:
            k = 4
        elif cnt == 10:
            k = 5
        elif cnt == 15:
            k = 6
        elif cnt == 21:
            k = 7
        elif cnt == 28:
            k = 8
        elif cnt == 36:
            k = 9
        elif cnt == 45:
            k = 10
        elif cnt == 55:
            k = 11
        elif cnt == 66:
            k = 12
        elif cnt == 78:
            k = 13
        elif cnt == 91:
            k = 14
        elif cnt == 105:
            k = 15
        elif cnt == 120:
            k = 16
        else:
            # This should not happen if the logic is correct
            # A line defined by points must have pairs that sum to this.
            # Wait, if 4 points are collinear, the 6 pairs all map to the same key.
            # So cnt will be 6.
            # If 3 points, cnt is 3.
            # So this mapping is valid.
            continue

        if k > 4:
            return 0
        if k == 4:
            count += 1
            
    return count

def construct():
    # New Approach: Simulated Annealing
    # Grid size: -20 to 20
    min_coord = -20
    max_coord = 20
    
    # Initial configuration: Random
    random.seed(42)
    
    def random_points():
        pts = []
        while len(pts) < 16:
            x = random.randint(min_coord, max_coord)
            y = random.randint(min_coord, max_coord)
            p = [x, y]
            if p not in pts:
                pts.append(p)
        return pts
    
    best_points = random_points()
    best_score = count_exact_4_lines(best_points)
    
    # Simulated Annealing Parameters
    start_temp = 10.0
    end_temp = 0.1
    num_iterations = 50000
    start_time = time.time()
    time_limit = 100 # Leave some buffer for 120s limit
    
    for i in range(num_iterations):
        if time.time() - start_time > time_limit:
            break
            
        # Cooling schedule
        temp = start_temp * (end_temp / start_temp) ** (i / num_iterations)
        
        # Move: Pick a random point and move it to a random location
        idx = random.randint(0, 15)
        old_p = best_points[idx]
        
        new_x = random.randint(min_coord, max_coord)
        new_y = random.randint(min_coord, max_coord)
        new_p = [new_x, new_y]
        
        # Check if new point is already in the set (excluding the one being moved)
        if new_p in best_points and new_p != old_p:
            continue
            
        best_points[idx] = new_p
        new_score = count_exact_4_lines(best_points)
        
        if new_score > best_score:
            best_score = new_score
        else:
            # Accept with probability exp((new_score - best_score) / temp)
            # Since we are maximizing, delta = new - best
            delta = new_score - best_score
            if delta < 0:
                prob = 2.718281828 ** (delta / temp)
                if random.random() > prob:
                    # Reject
                    best_points[idx] = old_p
                else:
                    # Accept (worse solution)
                    best_score = new_score
            else:
                # Better or equal
                best_score = new_score
                
    # Final check
    final_score = count_exact_4_lines(best_points)
    if final_score == 0:
        # Fallback to a known valid configuration if needed
        # The 4x4 grid is valid
        return [[x, y] for x in range(4) for y in range(4)]
        
    return best_points
