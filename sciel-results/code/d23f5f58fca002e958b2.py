import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def are_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p2[1] - p1[1]) * (p3[0] - p1[0])

    def line_key(p1, p2):
        # Return a normalized representation of the line
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        # Normalize direction vector
        if dx == 0 and dy == 0:
            return None # Should not happen for distinct points
            
        # Use gcd to normalize
        import math
        g = math.gcd(abs(dx), abs(dy))
        dx //= g
        dy //= g
        
        # Ensure consistent sign
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
            
        # The line is defined by direction (dx, dy) and a point on it.
        # To make it unique regardless of which point is p1 or p2, 
        # we can use the cross product or intercept.
        # Better: Use the equation Ax + By = C.
        # A = dy, B = -dx, C = -(dy*x1 - dx*y1)
        A = dy
        B = -dx
        C = -(dy * p1[0] - dx * p1[1])
        
        # Normalize A, B, C by gcd
        g2 = math.gcd(abs(A), abs(B))
        if g2 != 0:
            A //= g2
            B //= g2
        # C doesn't need to be divided by g2 if we are just using it as a key?
        # Actually, if we divide A and B, we should divide C by the same factor used for A,B?
        # No, the equation A x + B y = C is unique if A,B are normalized.
        # But C must be an integer.
        # Let's just use the tuple (A, B, C) as the key.
        return (A, B, C)

    def has_four_collinear(points):
        line_counts = {}
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                key = line_key(points[i], points[j])
                if key:
                    line_counts[key] = line_counts.get(key, 0) + 1
                    if line_counts[key] >= 2:
                        # This line has at least 2 pairs.
                        # If a line has k points, it has k*(k-1)/2 pairs.
                        # We need to know if k >= 4.
                        # 4 points -> 6 pairs.
                        # 3 points -> 3 pairs.
                        # So if count >= 6, we definitely have 4 points.
                        # But wait, the count is the number of PAIRS on that line.
                        # If count is 3, it could be 3 points (3 pairs) or ... no, 3 points is the only way to get 3 pairs from distinct points on a line?
                        # No. 3 points form 3 pairs. 4 points form 6 pairs.
                        # So if line_counts[key] >= 6, we have 4 collinear points.
                        # However, we can check earlier.
                        # If we see the 3rd pair, it could be 3 points.
                        # If we see the 4th pair, it must be 4 points?
                        # Points: 1,2,3. Pairs: (1,2), (1,3), (2,3). Count=3.
                        # Points: 1,2,3,4. Pairs: 6.
                        # So if count reaches 4, it must be 4 points?
                        # No, 4 points have 6 pairs.
                        # Is it possible to have 4 pairs without 4 points?
                        # No, the number of pairs is strictly determined by the number of points k: k(k-1)/2.
                        # k=1 -> 0
                        # k=2 -> 1
                        # k=3 -> 3
                        # k=4 -> 6
                        # So if count >= 4, it must be k>=4?
                        # No, count=4 is impossible for integer k.
                        # Count must be 0, 1, 3, 6, 10...
                        # So if count >= 4, it must be 6 or more, which implies k>=4.
                        # So we can just check if count >= 4.
                        return True
        return False

    def count_triples(points):
        count = 0
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        count += 1
        return count

    # Generate pool
    pool = []
    for _ in range(2000):
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        pool.append([x, y])
    unique_pool = list(set(map(tuple, pool)))
    
    best_set = None
    best_triple_count = -1
    
    while time.time() < deadline:
        current_set = random.sample(unique_pool, 20)
        
        if has_four_collinear(current_set):
            continue
            
        current_triples = count_triples(current_set)
        
        if current_triples > best_triple_count:
            best_triple_count = current_triples
            best_set = [list(p) for p in current_set]
            
        no_improve_count = 0
        while time.time() < deadline and no_improve_count < 100:
            idx = random.randint(0, 19)
            old_point = current_set[idx]
            new_point = random.choice(unique_pool)
            
            if new_point == old_point:
                continue
                
            new_set = current_set.copy()
            new_set[idx] = new_point
            
            if has_four_collinear(new_set):
                continue
                
            new_triples = count_triples(new_set)
            
            if new_triples > current_triples:
                current_set = new_set
                current_triples = new_triples
                no_improve_count = 0
                
                if current_triples > best_triple_count:
                    best_triple_count = current_triples
                    best_set = [list(p) for p in current_set]
            else:
                no_improve_count += 1
                
    if best_set:
        return best_set
    else:
        return [[x, x * x] for x in range(-10, 10)]
