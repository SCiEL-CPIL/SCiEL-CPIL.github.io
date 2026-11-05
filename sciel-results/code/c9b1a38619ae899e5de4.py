import random
import math
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    def is_perfect_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n

    def is_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0

    def is_concyclic(p1, p2, p3, p4):
        # Determinant of:
        # | x^2+y^2  x  y  1 |
        # | x^2+y^2  x  y  1 |
        # | x^2+y^2  x  y  1 |
        # | x^2+y^2  x  y  1 |
        # Using expansion for 4x4
        def det4(m):
            return (m[0][0]*(m[1][1]*m[2][2]*m[3][3] + m[1][2]*m[2][3]*m[3][1] + m[1][3]*m[2][1]*m[3][2] - 
                             m[1][1]*m[2][3]*m[3][2] - m[1][2]*m[2][1]*m[3][3] - m[1][3]*m[2][2]*m[3][1]) -
                    m[0][1]*(m[1][0]*m[2][2]*m[3][3] + m[1][2]*m[2][3]*m[3][0] + m[1][3]*m[2][0]*m[3][2] - 
                             m[1][0]*m[2][3]*m[3][2] - m[1][2]*m[2][0]*m[3][3] - m[1][3]*m[2][2]*m[3][0]) +
                    m[0][2]*(m[1][0]*m[2][1]*m[3][3] + m[1][1]*m[2][3]*m[3][0] + m[1][3]*m[2][0]*m[3][1] - 
                             m[1][0]*m[2][3]*m[3][1] - m[1][1]*m[2][0]*m[3][3] - m[1][3]*m[2][1]*m[3][0]) -
                    m[0][3]*(m[1][0]*m[2][1]*m[3][2] + m[1][1]*m[2][2]*m[3][0] + m[1][2]*m[2][0]*m[3][1] - 
                             m[1][0]*m[2][2]*m[3][1] - m[1][1]*m[2][0]*m[3][2] - m[1][2]*m[2][1]*m[3][0]))
        
        rows = []
        for p in (p1, p2, p3, p4):
            x, y = p
            rows.append([x*x + y*y, x, y, 1])
        return det4(rows) == 0

    def count_rational_pairs(points, D=1):
        count = 0
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                d2 = dx*dx + D*dy*dy
                if is_perfect_square(d2):
                    count += 1
        return count

    def check_constraints(points, D=1):
        n = len(points)
        if len(set(points)) != n:
            return False
        # Check distances
        for i in range(n):
            for j in range(i+1, n):
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                d2 = dx*dx + D*dy*dy
                if not is_perfect_square(d2):
                    return False
        # Check collinear
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if is_collinear(points[i], points[j], points[k]):
                        return False
        # Check concyclic
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if is_concyclic(points[i], points[j], points[k], points[l]):
                            return False
        return True

    def fitness(points, D=1):
        n = len(points)
        if len(set(points)) != n:
            return -1000
        rat_pairs = count_rational_pairs(points, D)
        # Penalty for collinear
        collinear_count = 0
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if is_collinear(points[i], points[j], points[k]):
                        collinear_count += 1
        # Penalty for concyclic (only if many rational pairs to save time)
        concyclic_count = 0
        if rat_pairs >= n*(n-1)//2 - 2: # Close to full
             for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        for l in range(k+1, n):
                            if is_concyclic(points[i], points[j], points[k], points[l]):
                                concyclic_count += 1
        return rat_pairs * 10 - collinear_count * 50 - concyclic_count * 100

    # Known 5-point seed
    seed_5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best_points = seed_5
    best_score = 5 # We know this is valid
    
    D = 1
    target_size = 6
    
    # Try to find 6 points
    # Strategy: Simulated Annealing on 6 points
    # Initialize with 5 seed points + 1 random point
    max_restarts = 20
    
    for r in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
            
        if r == 0:
            points = seed_5 + [(random.randint(-100, 100), random.randint(-100, 100))]
        else:
            # Random initialization
            box = 150
            points = [(random.randint(-box, box), random.randint(-box, box)) for _ in range(6)]
            # Ensure distinct
            while len(set(points)) < 6:
                idx = random.randint(0, 5)
                points[idx] = (random.randint(-box, box), random.randint(-box, box))
        
        current_fitness = fitness(points, D)
        best_local_fitness = current_fitness
        best_local_points = list(points)
        
        temp = 50.0
        cooling = 0.99
        
        for step in range(1000):
            if time.time() > deadline - 5.0:
                break
            
            idx = random.randint(0, 5)
            old_x, old_y = points[idx]
            
            # Random move
            # Mix of small and large steps
            if random.random() < 0.1:
                step_size = random.randint(10, 50)
            else:
                step_size = random.randint(1, 10)
            
            dx = random.randint(-step_size, step_size)
            dy = random.randint(-step_size, step_size)
            
            new_x = old_x + dx
            new_y = old_y + dy
            
            points[idx] = (new_x, new_y)
            new_fitness = fitness(points, D)
            
            delta = new_fitness - current_fitness
            
            if delta > 0 or random.random() < math.exp(delta / max(temp, 0.1)):
                current_fitness = new_fitness
                if new_fitness > best_local_fitness:
                    best_local_fitness = new_fitness
                    best_local_points = list(points)
                    
                # Check if we found a valid 6-point set
                # A valid 6-point set has 15 rational pairs, 0 collinear, 0 concyclic
                # Max possible fitness for 6 points: 15*10 - 0 - 0 = 150
                if best_local_fitness >= 150:
                    if check_constraints(best_local_points, D):
                        best_points = best_local_points
                        best_score = 6
                        # Try to find 7?
                        # For now, return 6. 7 is harder.
                        return {'D': D, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
            else:
                points[idx] = (old_x, old_y)
            
            temp *= cooling
        
        # Update global best
        if best_local_fitness > fitness(best_points, D):
             # Verify validity before updating global best to ensure we don't return invalid
             if check_constraints(best_local_points, D):
                 if len(best_local_points) > len(best_points):
                     best_points = best_local_points
                     best_score = len(best_points)

    # If we found 6, return it.
    if best_score >= 6:
        return {'D': D, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
    
    # Fallback to 5
    return {'D': D, 'points': [[[x, 1], [y, 1]] for x, y in seed_5]}
# EVOLVE-BLOCK-END
