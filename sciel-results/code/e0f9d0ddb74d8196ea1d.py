import numpy as np
import time
import random
from scipy.optimize import milp, LinearConstraint, Bounds
from itertools import combinations

# Constants
N = 20
BOX = 100
DEADLINE = 150.0
START_TIME = time.time()

def check_four_collinear(points):
    """
    Returns True if any 4 points are collinear.
    points: list of (x, y) tuples
    """
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            # Normalize direction? No, just check cross product with others
            for k in range(j+1, n):
                x3, y3 = points[k]
                # Check if (x3,y3) is on line through (x1,y1) and (x2,y2)
                # (x3-x1)*dy - (y3-y1)*dx == 0
                if (x3 - x1) * dy - (y3 - y1) * dx != 0:
                    continue
                # Found 3 collinear: i, j, k
                # Check if any other point l is on this line
                for l in range(k+1, n):
                    x4, y4 = points[l]
                    if (x4 - x1) * dy - (y4 - y1) * dx == 0:
                        return True
    return False

def get_collinear_triples(points):
    """
    Returns a list of indices (i, j, k) such that points[i], points[j], points[k] are collinear.
    """
    triples = []
    n = len(points)
    # To avoid duplicates, i < j < k
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            for k in range(j+1, n):
                x3, y3 = points[k]
                if (x3 - x1) * dy - (y3 - y1) * dx == 0:
                    triples.append((i, j, k))
    return triples

def compute_score(points):
    """
    Computes the largest subset with no three collinear.
    Returns infinity if invalid (4 collinear).
    """
    if check_four_collinear(points):
        return float('inf')
    
    triples = get_collinear_triples(points)
    n = len(points)
    
    if not triples:
        return n
        
    # ILP: Maximize sum(x_i) subject to x_i + x_j + x_k <= 2 for each triple
    # x_i in {0, 1}
    
    # c: objective coefficients (we want to maximize, milp minimizes, so -1)
    c = np.ones(n) * -1.0
    
    # A_ub, b_ub: constraints
    # Each triple is a row in A_ub
    # If there are no triples, we handled it above.
    A_ub = np.zeros((len(triples), n))
    for idx, (i, j, k) in enumerate(triples):
        A_ub[idx, i] = 1
        A_ub[idx, j] = 1
        A_ub[idx, k] = 1
    
    b_ub = np.ones(len(triples)) * 2
    
    # Constraints: 0 <= x_i <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: 1 for binary
    integrality = np.ones(n)
    
    try:
        res = milp(c=c, constraints=[LinearConstraint(A_ub, -np.inf, b_ub)], bounds=bounds, integrality=integrality)
        if res.status == 0:
            # Maximize sum(x) -> min -sum(x) -> res.fun is -max_sum
            max_size = int(round(-res.fun))
            return max_size
        else:
            # Fallback: simple greedy or random if ILP fails
            # For n=20, we can do a simple backtracking if ILP fails, but ILP should be robust.
            return n # Conservative fallback
    except Exception:
        return n

def construct():
    global START_TIME
    # If global time is not set, set it
    if not hasattr(construct, "start_time"):
        construct.start_time = time.time()
        
    best_score = float('inf')
    best_points = None
    
    # Initial configurations
    candidates = []
    
    # 1. Random
    for _ in range(5):
        pts = [(random.randint(-BOX, BOX), random.randint(-BOX, BOX)) for _ in range(N)]
        # Ensure distinct
        while len(set(pts)) < N:
            pts[random.randrange(N)] = (random.randint(-BOX, BOX), random.randint(-BOX, BOX))
        candidates.append(pts)
        
    # 2. Grid-like perturbation
    # 5x4 grid
    base_grid = []
    for x in range(5):
        for y in range(4):
            base_grid.append((x*20, y*20)) # Spread out
    # Perturb
    for _ in range(5):
        pts = []
        for p in base_grid:
            pts.append((p[0] + random.randint(-5, 5), p[1] + random.randint(-5, 5)))
        # Ensure distinct
        while len(set(pts)) < N:
            i = random.randrange(N)
            pts[i] = (pts[i][0] + random.randint(-10, 10), pts[i][1] + random.randint(-10, 10))
        candidates.append(pts)
        
    # Local Search
    # We will iterate through candidates and try to improve them
    
    for start_pts in candidates:
        current_pts = [list(p) for p in start_pts]
        current_score = compute_score(current_pts)
        
        if current_score < best_score:
            best_score = current_score
            best_points = [list(p) for p in current_pts]
            
        # Local Search Loop
        # Time check
        while time.time() - construct.start_time < DEADLINE - 10:
            improved = False
            # Shuffle points to try different orders
            indices = list(range(N))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() - construct.start_time > DEADLINE - 10:
                    break
                
                old_p = current_pts[i]
                # Try a few random moves
                for _ in range(20):
                    new_p = (random.randint(-BOX, BOX), random.randint(-BOX, BOX))
                    if new_p == old_p:
                        continue
                    # Check validity: no 4 collinear
                    # Create new set
                    new_pts = current_pts.copy()
                    new_pts[i] = list(new_p)
                    
                    # Quick check: is new_p collinear with any 3 from others?
                    # This is O(N^3) which is 8000 ops, fast enough.
                    # But we can just call compute_score which does it.
                    # To save time, do a quick 4-collinear check involving i
                    # If valid, compute score
                    
                    # Optimization: Check 4-collinearity involving i first
                    valid = True
                    others = [p for idx, p in enumerate(current_pts) if idx != i]
                    # Check if new_p is collinear with any 3 in others
                    # There are C(19,3) = 969 triples.
                    # We can just run check_four_collinear on the new set?
                    # It's O(N^4) = 20^4 = 160,000 ops.
                    # 20 moves * 160k = 3.2M ops per point.
                    # 20 points * 3.2M = 64M ops per iteration.
                    # Python is slow. 150s might allow ~100 iterations?
                    # Let's rely on compute_score which includes the check.
                    
                    new_score = compute_score(new_pts)
                    
                    if new_score < current_score:
                        current_pts = new_pts
                        current_score = new_score
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_points = [list(p) for p in current_pts]
                        break # Restart loop with new configuration? Or continue?
                        # Breaking restarts the inner loop for this point, effectively re-evaluating.
                        # It's better to accept and continue to next point to see if further improvements exist.
                        # But changing one point changes validity for others.
                        # Let's just break the move loop and restart the point loop to be safe?
                        # Or just continue. Since it's an improvement, we are in a better state.
                        # Let's continue to next point.
                    
            if not improved:
                # Stuck in local minimum? Try a random restart or larger jump
                # Just break to next candidate
                break
                
    # Final check
    if best_points is None:
        # Fallback to parabola
        return [[x, x * x] for x in range(-10, 10)]
        
    return best_points

# Helper to reset start time if called multiple times?
# The task says "Each evaluation must finish within 120 s".
# It implies construct() is called once per evaluation.
# But to be safe, we can reset the timer inside construct if it's a fresh call?
# No, the deadline guard should be relative to the start of the function.

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    best_score = float('inf')
    best_points = None
    
    candidates = []
    
    # 1. Random
    for _ in range(10):
        pts = [(random.randint(-BOX, BOX), random.randint(-BOX, BOX)) for _ in range(N)]
        while len(set(pts)) < N:
            pts[random.randrange(N)] = (random.randint(-BOX, BOX), random.randint(-BOX, BOX))
        candidates.append(pts)
        
    # 2. Grid-like perturbation (higher chance of collinearities)
    for _ in range(10):
        pts = []
        # Use a smaller grid to encourage collinearity, then perturb
        # 5x4 grid
        for x in range(5):
            for y in range(4):
                # Random perturbation
                px = x * 20 + random.randint(-10, 10)
                py = y * 20 + random.randint(-10, 10)
                pts.append((px, py))
        while len(set(pts)) < N:
            i = random.randrange(N)
            pts[i] = (pts[i][0] + random.randint(-10, 10), pts[i][1] + random.randint(-10, 10))
        candidates.append(pts)
        
    for start_pts in candidates:
        if time.time() > deadline:
            break
            
        current_pts = [list(p) for p in start_pts]
        current_score = compute_score(current_pts)
        
        if current_score < best_score:
            best_score = current_score
            best_points = [list(p) for p in current_pts]
            
        # Local Search
        # Limit iterations to ensure time safety
        for _ in range(50): # Max 50 improvement attempts per candidate
            if time.time() > deadline:
                break
                
            improved = False
            indices = list(range(N))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline:
                    break
                    
                old_p = tuple(current_pts[i])
                # Try moves
                # To speed up, we can try small perturbations first
                moves = []
                for dx in range(-5, 6):
                    for dy in range(-5, 6):
                        if dx == 0 and dy == 0: continue
                        moves.append((old_p[0]+dx, old_p[1]+dy))
                # Add some random jumps
                for _ in range(10):
                    moves.append((random.randint(-BOX, BOX), random.randint(-BOX, BOX)))
                    
                random.shuffle(moves)
                
                for new_p in moves:
                    if new_p == old_p:
                        continue
                    if new_p[0] < -BOX or new_p[0] > BOX or new_p[1] < -BOX or new_p[1] > BOX:
                        continue
                        
                    new_pts = current_pts.copy()
                    new_pts[i] = list(new_p)
                    
                    new_score = compute_score(new_pts)
                    
                    if new_score < current_score:
                        current_pts = new_pts
                        current_score = new_score
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_points = [list(p) for p in current_pts]
                        break # Move accepted, try next point
                        # Breaking here means we don't try other moves for this point immediately.
                        # This is a standard hill climbing step.
                        
            if not improved:
                break
                
    if best_points is None:
        return [[x, x * x] for x in range(-10, 10)]
        
    return best_points
