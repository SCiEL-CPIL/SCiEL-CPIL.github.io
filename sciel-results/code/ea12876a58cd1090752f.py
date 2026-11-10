"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    # Start with the 10x3 grid
    points = [(x, y) for x in range(10) for y in range(3)]
    
    MIN_COORD = -30
    MAX_COORD = 30
    N = 30
    
    def calculate_max_multiplicity(pts):
        # Calculate doubled areas for all triples
        counts = {}
        for i, j, k in combinations(range(N), 3):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            x3, y3 = pts[k]
            area = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
            if area > 0:
                counts[area] = counts.get(area, 0) + 1
        
        if not counts:
            return 0
        return max(counts.values())

    current_score = calculate_max_multiplicity(points)
    best_points = points.copy()
    best_score = current_score
    
    # Local Search with Simulated Annealing-like acceptance
    # Simple Hill Climbing first
    iterations = 15000
    random.seed(42)
    
    for _ in range(iterations):
        i = random.randint(0, N - 1)
        old_pt = points[i]
        
        # Generate a new random point
        # To increase chance of improvement, maybe try nearby points first?
        # Let's mix: 50% random anywhere, 50% nearby
        if random.random() < 0.5:
            new_x = random.randint(MIN_COORD, MAX_COORD)
            new_y = random.randint(MIN_COORD, MAX_COORD)
        else:
            # Nearby move
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
            new_x = old_pt[0] + dx
            new_y = old_pt[1] + dy
            # Clamp to bounds
            if new_x < MIN_COORD: new_x = MIN_COORD
            if new_x > MAX_COORD: new_x = MAX_COORD
            if new_y < MIN_COORD: new_y = MIN_COORD
            if new_y > MAX_COORD: new_y = MAX_COORD
            
        # Check collision
        collision = False
        for j in range(N):
            if j == i:
                continue
            if points[j] == (new_x, new_y):
                collision = True
                break
                
        if collision:
            continue
            
        points[i] = (new_x, new_y)
        new_score = calculate_max_multiplicity(points)
        
        if new_score >= current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = points.copy()
        else:
            points[i] = old_pt
            
    return best_points
# EVOLVE-BLOCK-END
