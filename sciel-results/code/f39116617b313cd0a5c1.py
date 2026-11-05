import time
import random
import itertools
from math import comb

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer for safety
    
    # Helper to check collinearity of 3 points
    def is_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])

    # Helper to calculate the number of collinear triples
    def count_triples(points):
        count = 0
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if is_collinear(points[i], points[j], points[k]):
                        count += 1
        return count

    # Helper to check if any 4 points are collinear
    def has_four_collinear(points):
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        if is_collinear(points[i], points[j], points[k]) and is_collinear(points[i], points[j], points[l]):
                            return True
        return False

    # Initial random configuration
    def random_point():
        return [random.randint(-100, 100), random.randint(-100, 100)]

    # Generate initial 20 distinct points with no 4 collinear
    points = []
    attempts = 0
    while len(points) < 20:
        p = random_point()
        # Check distinctness
        if p in points:
            continue
        # Check no 4 collinear with existing points
        valid = True
        # To be efficient, we can just check if adding p creates a 4-collinear set
        # This requires checking lines formed by pairs of existing points
        # But simpler: just check all combinations of 3 existing + p
        # Since n is small (<=19), this is fast
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                for k in range(j + 1, len(points)):
                    if is_collinear(points[i], points[j], p) and is_collinear(points[i], points[k], p):
                         # This logic is slightly flawed for checking 4 collinear.
                         # 4 collinear means 4 points on a line.
                         # If p is on line AB and line AC, then A, B, C, p are collinear?
                         # If A, B, C are distinct, line AB is line AC. So yes.
                         # But what if A, B, C are not collinear? Then p can't be on both lines unless A=B or A=C.
                         # So this check is valid: if p lies on a line defined by 2 existing points, 
                         # and another existing point also lies on that line, we have 4.
                         # Actually, simpler: just check all triples of existing points + p.
                         # If any 3 existing points are collinear with p, we have a 4-collinear set.
                         valid = False
                         break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            points.append(p)
        attempts += 1
        if attempts > 10000:
            # Fallback: restart if stuck
            points = []
            attempts = 0

    current_points = points
    current_triples = count_triples(current_points)
    best_points = current_points
    best_triples = current_triples

    # Simulated Annealing
    temperature = 10.0
    cooling = 0.99
    
    while time.time() < deadline:
        # Perturbation: move one point
        idx = random.randint(0, 19)
        old_p = current_points[idx]
        new_p = [random.randint(-100, 100), random.randint(-100, 100)]
        
        # Ensure distinctness
        if new_p in current_points:
            continue
            
        # Check validity (no 4 collinear)
        # Remove old point, add new point, check
        test_points = current_points.copy()
        test_points[idx] = new_p
        
        if has_four_collinear(test_points):
            continue
            
        new_triples = count_triples(test_points)
        
        # Acceptance criterion
        if new_triples < current_triples:
            current_points = test_points
            current_triples = new_triples
            if new_triples < best_triples:
                best_points = current_points
                best_triples = new_triples
        else:
            # Accept with probability exp(-delta / T)
            delta = new_triples - current_triples
            if delta == 0:
                # Accept equal moves to allow exploration
                current_points = test_points
                current_triples = new_triples
            else:
                prob = pow(2.718281828, -delta / temperature)
                if random.random() < prob:
                    current_points = test_points
                    current_triples = new_triples
        
        temperature *= cooling
        if temperature < 0.1:
            temperature = 1.0 # Restart annealing if it gets too cold to escape local minima
            
    return best_points
# EVOLVE-BLOCK-END
