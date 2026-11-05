import random
import math
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit

    def is_perfect_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n

    def is_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        area2 = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
        return area2 == 0

    def is_concyclic(p1, p2, p3, p4):
        rows = []
        for p in (p1, p2, p3, p4):
            x, y = p
            rows.append([x*x + y*y, x, y, 1])
        def det4(m):
            return (m[0][0]*(m[1][1]*m[2][2]*m[3][3] + m[1][2]*m[2][3]*m[3][1] + m[1][3]*m[2][1]*m[3][2] - 
                             m[1][1]*m[2][3]*m[3][2] - m[1][2]*m[2][1]*m[3][3] - m[1][3]*m[2][2]*m[3][1]) -
                    m[0][1]*(m[1][0]*m[2][2]*m[3][3] + m[1][2]*m[2][3]*m[3][0] + m[1][3]*m[2][0]*m[3][2] - 
                             m[1][0]*m[2][3]*m[3][2] - m[1][2]*m[2][0]*m[3][3] - m[1][3]*m[2][2]*m[3][0]) +
                    m[0][2]*(m[1][0]*m[2][1]*m[3][3] + m[1][1]*m[2][3]*m[3][0] + m[1][3]*m[2][0]*m[3][1] - 
                             m[1][0]*m[2][3]*m[3][1] - m[1][1]*m[2][0]*m[3][3] - m[1][3]*m[2][1]*m[3][0]) -
                    m[0][3]*(m[1][0]*m[2][1]*m[3][2] + m[1][1]*m[2][2]*m[3][0] + m[1][2]*m[2][0]*m[3][1] - 
                             m[1][0]*m[2][2]*m[3][1] - m[1][1]*m[2][0]*m[3][2] - m[1][2]*m[2][1]*m[3][0]))
        return det4(rows) == 0

    def is_valid_set(points):
        n = len(points)
        if len(set(points)) != n:
            return False
        for i in range(n):
            for j in range(i+1, n):
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                d2 = dx*dx + dy*dy
                if not is_perfect_square(d2):
                    return False
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if is_collinear(points[i], points[j], points[k]):
                        return False
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if is_concyclic(points[i], points[j], points[k], points[l]):
                            return False
        return True

    def fitness(points):
        """Count valid pairs minus penalties for collinearities and concyclicities."""
        n = len(points)
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                d2 = dx*dx + dy*dy
                if is_perfect_square(d2):
                    count += 1
        # Penalize collinear triples
        penalty = 0
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if is_collinear(points[i], points[j], points[k]):
                        penalty += 50
        # Penalize concyclic quadruples
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if is_concyclic(points[i], points[j], points[k], points[l]):
                            penalty += 100
        return count - penalty

    # Generate Pythagorean triple candidates
    def generate_pythagorean_candidates(limit=200):
        """Generate (x, y) pairs where x^2 + y^2 is a perfect square."""
        candidates = set()
        # Standard Pythagorean triples: (m^2-n^2, 2mn) and (2mn, m^2-n^2)
        for m in range(2, 30):
            for n in range(1, m):
                if (m - n) % 2 == 0 and math.gcd(m, n) != 1:
                    continue
                a = m*m - n*n
                b = 2*m*n
                c = m*m + n*n
                if a <= limit and b <= limit:
                    # Add all sign variations
                    for sx in [1, -1]:
                        for sy in [1, -1]:
                            candidates.add((sx*a, sy*b))
                            candidates.add((sx*b, sy*a))
        # Also add scaled versions
        for base in list(candidates):
            for scale in range(2, 5):
                new_pt = (base[0]*scale, base[1]*scale)
                if abs(new_pt[0]) <= limit*3 and abs(new_pt[1]) <= limit*3:
                    candidates.add(new_pt)
        return list(candidates)

    # Seed solution (5 points) - verified valid
    seed_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    best_solution = seed_points
    best_fitness = fitness(seed_points)
    
    # Generate candidate pool
    candidates = generate_pythagorean_candidates(150)
    
    # Strategy 1: Greedy construction from origin
    def greedy_construct(target_size, candidates, max_time=10.0):
        """Greedily add points that maintain validity."""
        points = [(0, 0)]
        remaining = [c for c in candidates if c != (0, 0)]
        random.shuffle(remaining)
        
        start = time.time()
        while len(points) < target_size and time.time() - start < max_time:
            added = False
            for i, cand in enumerate(remaining):
                new_points = points + [cand]
                if len(set(new_points)) != len(new_points):
                    continue
                # Check all pairwise distances
                valid = True
                for j in range(len(new_points)):
                    for k in range(j+1, len(new_points)):
                        dx = new_points[j][0] - new_points[k][0]
                        dy = new_points[j][1] - new_points[k][1]
                        d2 = dx*dx + dy*dy
                        if not is_perfect_square(d2):
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                # Check collinearity
                n = len(new_points)
                for a in range(n):
                    for b in range(a+1, n):
                        for c in range(b+1, n):
                            if is_collinear(new_points[a], new_points[b], new_points[c]):
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                # Check concyclicity
                for a in range(n):
                    for b in range(a+1, n):
                        for c in range(b+1, n):
                            for d in range(c+1, n):
                                if is_concyclic(new_points[a], new_points[b], new_points[c], new_points[d]):
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    points.append(cand)
                    remaining.pop(i)
                    added = True
                    break
            if not added:
                break
        return points

    # Try greedy construction
    if time.time() < deadline - 30.0:
        greedy_result = greedy_construct(6, candidates, max_time=20.0)
        if len(greedy_result) >= 6:
            if is_valid_set(greedy_result):
                best_solution = greedy_result
                best_fitness = fitness(greedy_result)
    
    # Strategy 2: Hill climbing starting from seed + random candidates
    if time.time() < deadline - 10.0:
        max_restarts = 30
        for r in range(max_restarts):
            if time.time() > deadline - 5.0:
                break
            
            # Initialize with seed + 1 random candidate
            base = seed_points[:]
            extra = random.choice(candidates)
            if extra not in base:
                points = base + [extra]
            else:
                points = base + [random.choice(candidates)]
            
            # Ensure unique
            while len(set(points)) < 6:
                idx = random.randint(0, 5)
                points[idx] = random.choice(candidates)
            
            current_fitness = fitness(points)
            
            # Hill climbing with adaptive steps
            improved = True
            steps = 0
            while improved and steps < 500:
                if time.time() > deadline - 3.0:
                    break
                improved = False
                steps += 1
                idx = random.randint(0, 5)
                old_x, old_y = points[idx]
                
                # Try replacing with a random candidate
                new_pt = random.choice(candidates)
                points[idx] = new_pt
                new_fitness = fitness(points)
                
                if new_fitness > current_fitness:
                    current_fitness = new_fitness
                    improved = True
                else:
                    points[idx] = (old_x, old_y)
                
                if current_fitness >= 15:
                    if is_valid_set(points):
                        if len(points) > len(best_solution) or (len(points) == len(best_solution) and current_fitness > best_fitness):
                            best_solution = points[:]
                            best_fitness = current_fitness
                        break
            
            if best_fitness >= 15 and len(best_solution) >= 6:
                break
    
    # Strategy 3: Local search around best solution found
    if len(best_solution) >= 5 and time.time() < deadline - 5.0:
        # Try to extend by one point
        for cand in candidates:
            if cand in best_solution:
                continue
            new_points = best_solution + [cand]
            if len(set(new_points)) != len(new_points):
                continue
            if is_valid_set(new_points):
                best_solution = new_points
                best_fitness = fitness(new_points)
                break
    
    # Final verification and fallback
    if not is_valid_set(best_solution):
        best_solution = seed_points
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_solution]}
# EVOLVE-BLOCK-END
