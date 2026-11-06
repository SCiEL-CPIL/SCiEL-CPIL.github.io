"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def construct():
    """
    Uses Local Search (Hill Climbing) to maximize the number of 3-collinear triples
    while maintaining the constraint that no 4 points are collinear.
    Maximizing 3-collinear triples is a strong proxy for minimizing the size of the
    largest general position subset (no 3 collinear).
    """
    random.seed(42)
    N = 20
    LIMIT = 100
    
    def is_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p2[1] - p1[1]) * (p3[0] - p1[0])
        
    def is_valid_config(points):
        """Check if no 4 points are collinear."""
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                # Find all points collinear with points[i] and points[j]
                collinear_count = 2 # i and j
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if is_collinear(points[i], points[j], points[k]):
                        collinear_count += 1
                if collinear_count >= 4:
                    return False
        return True
        
    def count_3collinear_triples(points):
        """Count the number of distinct triples of collinear points."""
        n = len(points)
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if is_collinear(points[i], points[j], points[k]):
                        count += 1
        return count

    # 1. Initialization: Generate a valid random configuration
    # We use a grid-like structure with noise to ensure validity initially
    # Or simply random with retry until valid
    def generate_initial():
        while True:
            pts = []
            used = set()
            for _ in range(N):
                while True:
                    x = random.randint(-LIMIT, LIMIT)
                    y = random.randint(-LIMIT, LIMIT)
                    if (x, y) not in used:
                        used.add((x, y))
                        break
                pts.append([x, y])
            if is_valid_config(pts):
                return pts

    points = generate_initial()
    current_triples = count_3collinear_triples(points)
    
    # 2. Local Search
    # We will try to swap a point with a new random point if it increases the triple count
    # and maintains validity.
    # To speed up, we can approximate the change in triples, but for N=20,
    # exact recounting is fast enough if we limit iterations.
    
    start_time = time.time()
    timeout = 110 # Leave some time for the evaluator
    
    iterations = 0
    max_iterations = 500000 # Safety cap
    
    while time.time() - start_time < timeout and iterations < max_iterations:
        iterations += 1
        
        # Pick a random index to move
        idx = random.randint(0, N - 1)
        old_point = points[idx]
        
        # Generate a new candidate point
        # Bias towards existing points' neighborhoods to find clusters? 
        # Or just random? Random is safer for exploration.
        new_point = [random.randint(-LIMIT, LIMIT), random.randint(-LIMIT, LIMIT)]
        
        # Ensure distinctness
        if new_point == old_point:
            continue
        if any(p == new_point for p in points if p != old_point):
            continue
            
        # Tentatively apply
        points[idx] = new_point
        
        # Check validity (No 4 collinear)
        if not is_valid_config(points):
            points[idx] = old_point
            continue
            
        # Calculate new triple count
        new_triples = count_3collinear_triples(points)
        
        # Accept if better
        if new_triples > current_triples:
            current_triples = new_triples
            # Keep the new point
        else:
            # Revert
            points[idx] = old_point
            
    return points
# EVOLVE-BLOCK-END
