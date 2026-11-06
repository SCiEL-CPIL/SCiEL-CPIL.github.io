import random
import time
import itertools

def count_triples(points):
    """Count total number of collinear triples. Assumes no 4 collinear."""
    n = len(points)
    triples = 0
    # Use a set of normalized line tuples to avoid double counting?
    # Actually, iterating over pairs and counting is O(N^3) which is fine for N=20.
    # But we can be smarter.
    # For each point i, count how many pairs (j, k) with j,k != i are collinear with i.
    # This is still O(N^3).
    
    # Better: Iterate over all pairs (i, j). Determine the line. Count points on it.
    # To avoid double counting, only consider pairs where i < j.
    # But a line with k points is counted k*(k-1)/2 times.
    # We want sum of C(k, 3).
    # If we count lines by their unique representation, we can sum C(k,3).
    
    lines = {}
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            # Normalize direction
            g = __import__('math').gcd(abs(dx), abs(dy))
            if g == 0: continue
            dx //= g
            dy //= g
            
            # Ensure unique representation
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            
            # Line equation: ax + by = c
            # Normal vector is (dy, -dx)
            a = dy
            b = -dx
            c = a * x1 + b * y1
            
            # Normalize c? No, just use (a, b, c) as key.
            # Since (a,b) is normalized, c is unique for the line.
            key = (a, b, c)
            
            if key not in lines:
                lines[key] = [i, j]
            else:
                lines[key].append(i)
                lines[key].append(j) # This is inefficient, better to just count unique points
    
    # The above logic is flawed for counting unique points.
    # Let's just do O(N^3) count which is 20^3 = 8000 ops. Very fast.
    
    triples = 0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                # Cross product
                if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0:
                    triples += 1
    return triples

def is_valid(points):
    """Check if no 4 points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            count = 2
            for k in range(n):
                if k == i or k == j: continue
                x3, y3 = points[k]
                if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0:
                    count += 1
            if count >= 4:
                return False
    return True

def max_no3_collinear_subset(points):
    """
    Estimate or calculate the largest subset with no 3 collinear.
    For N=20, we can use a greedy approach or simple backtracking.
    Greedy: Sort points by number of collinear pairs with others?
    Actually, the evaluator does this exactly. We just need a good proxy for search.
    The proxy "maximize triples" is good.
    """
    pass

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    best_points = None
    best_triples = -1
    
    # Initial random configuration
    # Try to find a valid starting point
    attempts = 0
    while True:
        pts = [[random.randint(-100, 100), random.randint(-100, 100)] for _ in range(20)]
        # Ensure distinct
        if len(set(map(tuple, pts))) == 20:
            if is_valid(pts):
                best_points = pts
                best_triples = count_triples(pts)
                break
        attempts += 1
        if attempts > 1000:
             # Fallback to parabola if random fails (unlikely)
             best_points = [[x, x * x] for x in range(-10, 10)]
             break

    # Local Search
    # We want to maximize triples.
    # Move one point at a time to a random valid location.
    
    current_points = best_points
    current_triples = best_triples
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 19)
        old_x, old_y = current_points[idx]
        
        # Try a few random moves
        improved = False
        for _ in range(50):
            new_x = random.randint(-100, 100)
            new_y = random.randint(-100, 100)
            
            # Check distinctness
            if any(p[0] == new_x and p[1] == new_y for i, p in enumerate(current_points) if i != idx):
                continue
                
            # Create new configuration
            new_points = current_points.copy()
            new_points[idx] = [new_x, new_y]
            
            if not is_valid(new_points):
                continue
                
            new_triples = count_triples(new_points)
            
            if new_triples > current_triples:
                current_points = new_points
                current_triples = new_triples
                improved = True
                if current_triples > best_triples:
                    best_points = current_points.copy()
                    best_triples = current_triples
                break # Restart move search for this point? Or continue?
                # Breaking allows us to pick a new point, which is good for diversity.
        
        if not improved:
            # If no improvement in this batch, maybe try a larger jump or restart?
            # For now, just continue.
            pass
            
    return best_points

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    best_points = None
    best_triples = -1
    
    # Initial random configuration
    attempts = 0
    while True:
        pts = [[random.randint(-100, 100), random.randint(-100, 100)] for _ in range(20)]
        if len(set(map(tuple, pts))) == 20:
            if is_valid(pts):
                best_points = pts
                best_triples = count_triples(pts)
                break
        attempts += 1
        if attempts > 1000:
             best_points = [[x, x * x] for x in range(-10, 10)]
             best_triples = count_triples(best_points)
             break

    current_points = best_points
    current_triples = best_triples
    
    # Simulated Annealing / Hill Climbing
    # We maximize triples.
    
    while time.time() < deadline:
        idx = random.randint(0, 19)
        old_pt = current_points[idx]
        
        # Try multiple random moves
        for _ in range(20):
            new_x = random.randint(-100, 100)
            new_y = random.randint(-100, 100)
            
            if any(p[0] == new_x and p[1] == new_y for i, p in enumerate(current_points) if i != idx):
                continue
                
            new_points = current_points.copy()
            new_points[idx] = [new_x, new_y]
            
            if not is_valid(new_points):
                continue
                
            new_triples = count_triples(new_points)
            
            if new_triples >= current_triples:
                current_points = new_points
                current_triples = new_triples
                if current_triples > best_triples:
                    best_points = current_points.copy()
                    best_triples = current_triples
                break # Move accepted, pick next point
                
    return best_points
# EVOLVE-BLOCK-END
