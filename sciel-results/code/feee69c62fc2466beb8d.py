import random
import time
import math

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 115.0  # 115 seconds to be safe
    
    n = 20
    min_coord = -100
    max_coord = 100
    
    def is_collinear(p1, p2, p3):
        # (x2-x1)*(y3-y1) == (y2-y1)*(x3-x1)
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p2[1] - p1[1]) * (p3[0] - p1[0])

    def count_triples(points):
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if is_collinear(points[i], points[j], points[k]):
                        count += 1
        return count

    def is_valid_config(points):
        # Check distinctness
        seen = set()
        for p in points:
            if p in seen:
                return False
            seen.add(p)
        
        # Check no 4 collinear
        # For each pair, count points on the line
        # Optimization: Just check if any line has >= 4 points
        # Since n=20, O(n^2) lines, for each line O(n) check is O(n^3) = 8000.
        # Or use a set of lines represented by (slope, intercept) or normalized vector.
        # Normalized line representation:
        # A line is defined by (A, B, C) such that Ax + By + C = 0, with gcd(A,B,C)=1 and A>0 or A==0,B>0.
        line_counts = {}
        for i in range(n):
            for j in range(i + 1, n):
                p1 = points[i]
                p2 = points[j]
                # Calculate line coefficients
                A = p1[1] - p2[1]
                B = p2[0] - p1[0]
                C = p1[0] * p2[1] - p2[0] * p1[1]
                
                # Normalize
                if A == 0 and B == 0:
                    continue # Should not happen if distinct
                if A < 0 or (A == 0 and B < 0):
                    A = -A
                    B = -B
                    C = -C
                g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
                if g != 0:
                    A //= g
                    B //= g
                    C //= g
                key = (A, B, C)
                if key not in line_counts:
                    line_counts[key] = 0
                line_counts[key] += 1
        
        for count in line_counts.values():
            if count >= 4:
                return False
        return True

    def get_new_position(points, exclude_idx):
        # Try to find a valid position for point at exclude_idx
        # Random search
        for _ in range(1000):
            x = random.randint(min_coord, max_coord)
            y = random.randint(min_coord, max_coord)
            new_p = [x, y]
            
            # Check distinctness
            valid = True
            for i in range(n):
                if i == exclude_idx:
                    continue
                if points[i] == new_p:
                    valid = False
                    break
            if not valid:
                continue
            
            # Check no 4 collinear
            # Create a temporary list
            temp_points = points[:]
            temp_points[exclude_idx] = new_p
            
            # Quick check: count points on lines formed by new_p
            # If any line through new_p has >= 3 other points, invalid.
            # There are n-1 = 19 other points.
            # Lines through new_p and each other point.
            # If a line has k other points, total is k+1.
            # We need k+1 < 4 => k < 3 => k <= 2.
            # So if any line through new_p has >= 3 other points, it's invalid.
            
            line_counts = {}
            for i in range(n):
                if i == exclude_idx:
                    continue
                p1 = new_p
                p2 = temp_points[i]
                A = p1[1] - p2[1]
                B = p2[0] - p1[0]
                C = p1[0] * p2[1] - p2[0] * p1[1]
                
                if A < 0 or (A == 0 and B < 0):
                    A = -A
                    B = -B
                    C = -C
                g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
                if g != 0:
                    A //= g
                    B //= g
                    C //= g
                key = (A, B, C)
                if key not in line_counts:
                    line_counts[key] = 0
                line_counts[key] += 1
            
            valid = True
            for count in line_counts.values():
                if count >= 3:
                    valid = False
                    break
            
            if valid:
                return new_p
        return None

    # Initialization
    # Start with a random valid configuration
    current_points = []
    attempts = 0
    while len(current_points) < n and attempts < 10000:
        x = random.randint(min_coord, max_coord)
        y = random.randint(min_coord, max_coord)
        p = [x, y]
        if p not in current_points:
            # Check if adding p creates 4 collinear
            valid = True
            # Check lines with existing points
            # If p is on a line with 3 existing points, invalid.
            line_counts = {}
            for i in range(len(current_points)):
                p1 = p
                p2 = current_points[i]
                A = p1[1] - p2[1]
                B = p2[0] - p1[0]
                C = p1[0] * p2[1] - p2[0] * p1[1]
                
                if A < 0 or (A == 0 and B < 0):
                    A = -A
                    B = -B
                    C = -C
                g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
                if g != 0:
                    A //= g
                    B //= g
                    C //= g
                key = (A, B, C)
                if key not in line_counts:
                    line_counts[key] = 0
                line_counts[key] += 1
            
            for count in line_counts.values():
                if count >= 3:
                    valid = False
                    break
            
            if valid:
                current_points.append(p)
            else:
                pass # Just retry
        attempts += 1
    
    if len(current_points) < n:
        # Fallback to parabola if random init fails (unlikely)
        return [[x, x * x] for x in range(-10, 10)]

    current_triples = count_triples(current_points)
    best_points = current_points[:]
    best_triples = current_triples
    
    T = 50.0  # Initial temperature
    T_min = 1.0
    cooling = 0.999
    
    while time.time() < deadline:
        if T < T_min:
            T = T_min
            
        i = random.randint(0, n - 1)
        old_p = current_points[i]
        
        # Generate new position
        # Try small perturbation first, then random
        if random.random() < 0.5:
            new_p = None
            for _ in range(10):
                dx = random.randint(-5, 5)
                dy = random.randint(-5, 5)
                nx = old_p[0] + dx
                ny = old_p[1] + dy
                if min_coord <= nx <= max_coord and min_coord <= ny <= max_coord:
                    candidate = [nx, ny]
                    if candidate == old_p:
                        continue
                    # Check distinctness
                    valid = True
                    for j in range(n):
                        if j != i and current_points[j] == candidate:
                            valid = False
                            break
                    if not valid:
                        continue
                    # Check 4 collinear
                    # Use the logic from get_new_position
                    line_counts = {}
                    for j in range(n):
                        if j == i:
                            continue
                        p1 = candidate
                        p2 = current_points[j]
                        A = p1[1] - p2[1]
                        B = p2[0] - p1[0]
                        C = p1[0] * p2[1] - p2[0] * p1[1]
                        
                        if A < 0 or (A == 0 and B < 0):
                            A = -A
                            B = -B
                            C = -C
                        g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
                        if g != 0:
                            A //= g
                            B //= g
                            C //= g
                        key = (A, B, C)
                        if key not in line_counts:
                            line_counts[key] = 0
                        line_counts[key] += 1
                    
                    valid = True
                    for count in line_counts.values():
                        if count >= 3:
                            valid = False
                            break
                    if valid:
                        new_p = candidate
                        break
        else:
            new_p = get_new_position(current_points, i)
        
        if new_p is None:
            continue
        
        # Calculate new triples
        # We can optimize by only counting triples involving i
        # But n=20, count_triples is fast enough.
        temp_points = current_points[:]
        temp_points[i] = new_p
        new_triples = count_triples(temp_points)
        
        delta = new_triples - current_triples
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_points[i] = new_p
            current_triples = new_triples
            if current_triples < best_triples:
                best_triples = current_triples
                best_points = current_points[:]
        
        T *= cooling
        
    return best_points
