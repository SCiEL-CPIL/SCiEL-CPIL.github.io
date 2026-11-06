import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin for 120s limit

    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def get_lines(points):
        """
        Returns a dict mapping normalized line keys to sets of point indices.
        Also returns a boolean if any line has >= 4 points.
        """
        n = len(points)
        lines = {}
        has_4 = False
        
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                a = y1 - y2
                b = x2 - x1
                c = x1 * y2 - x2 * y1
                
                g = gcd(gcd(abs(a), abs(b)), abs(c))
                if g > 0:
                    a //= g
                    b //= g
                    c //= g
                else:
                    # If a,b,c are all 0, it's not a valid line (points are same), 
                    # but points are distinct, so this shouldn't happen unless logic error.
                    continue
                
                # Normalize sign
                if a < 0 or (a == 0 and b < 0):
                    a = -a
                    b = -b
                    c = -c
                
                key = (a, b, c)
                if key not in lines:
                    lines[key] = set()
                lines[key].add(i)
                lines[key].add(j)
                
                if len(lines[key]) >= 4:
                    has_4 = True
        
        return lines, has_4

    def count_triples(lines):
        """
        Counts total number of collinear triples.
        A line with k points contributes C(k, 3) triples.
        """
        total = 0
        for pts in lines.values():
            k = len(pts)
            if k >= 3:
                total += k * (k - 1) * (k - 2) // 6
        return total

    # Initialization
    # Start with a grid-like structure or random
    points = []
    seen = set()
    while len(points) < 20:
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        if (x, y) not in seen:
            seen.add((x, y))
            points.append([x, y])
    
    # Initial check
    lines, has_4 = get_lines(points)
    if has_4:
        # Perturb until valid
        while has_4:
            idx = random.randint(0, 19)
            points[idx] = [random.randint(-100, 100), random.randint(-100, 100)]
            if (points[idx][0], points[idx][1]) in seen:
                 # This simple check is flawed if we don't update seen properly, 
                 # but for SA, collisions are rare and we can just retry or allow dupes? 
                 # Task requires distinct points. 
                 # Let's just regenerate until unique.
                 while (points[idx][0], points[idx][1]) in seen:
                     points[idx] = [random.randint(-100, 100), random.randint(-100, 100)]
            lines, has_4 = get_lines(points)
            
    current_triples = count_triples(lines)
    best_triples = current_triples
    best_points = points[:]
    
    T = 10.0
    T_min = 0.1
    cooling = 0.999
    
    while time.time() < deadline:
        idx = random.randint(0, 19)
        old_x, old_y = points[idx]
        
        # Random move
        points[idx] = [random.randint(-100, 100), random.randint(-100, 100)]
        
        # Ensure distinctness (simple check)
        # In SA, we can just reject if duplicate
        if (points[idx][0], points[idx][1]) in seen:
             # Remove old, add new? No, just revert if duplicate to keep 'seen' consistent
             # Actually, maintaining 'seen' dynamically is tricky with reverts.
             # Let's just check against all other points. O(N).
             is_dup = False
             for k in range(20):
                 if k != idx and points[k] == points[idx]:
                     is_dup = True
                     break
             if is_dup:
                 points[idx] = [old_x, old_y]
                 continue
                 
        lines, has_4 = get_lines(points)
        
        if has_4:
            # Penalty for invalid
            new_triples = -1000
        else:
            new_triples = count_triples(lines)
            
        delta = new_triples - current_triples
        
        if delta > 0 or random.random() < math.exp(delta / T):
            current_triples = new_triples
            if new_triples > best_triples:
                best_triples = new_triples
                best_points = points[:]
        else:
            points[idx] = [old_x, old_y]
            
        T *= cooling
        if T < T_min:
            T = T_min
            
    return best_points
