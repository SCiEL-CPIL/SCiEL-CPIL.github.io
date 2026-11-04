import numpy as np
import time
import itertools

N = 11
SQRT3 = np.sqrt(3)

# Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, SQRT3 / 2.0])
TRI_AREA = 0.5 * SQRT3 / 2.0  # area of the equilateral triangle with side 1

def _barycentric_to_cart(bary):
    """Convert barycentric coords (a,b,c) with a+b+c=1 to cartesian."""
    return bary[0] * A + bary[1] * B + bary[2] * C

def _cart_to_barycentric(pt):
    """Convert cartesian to barycentric coordinates."""
    # Solve: pt = a*A + b*B + c*C, a+b+c=1
    # Using the standard formula for equilateral triangle
    x, y = pt
    c = y / (SQRT3 / 2.0)
    b = x - 0.5 * c
    a = 1.0 - b - c
    return np.array([a, b, c])

def _is_valid_barycentric(bary, eps=1e-9):
    """Check if barycentric coordinates are valid (inside triangle)."""
    return np.all(bary >= -eps) and np.all(bary <= 1.0 + eps)

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the N points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def _min_triangle_area_bary(bary_pts):
    """Compute min triangle area from barycentric coordinate array of shape (N,3)."""
    n = bary_pts.shape[0]
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area in barycentric: determinant of the 3x3 matrix of barycentric coords
                # det([[a1,b1,c1],[a2,b2,c2],[a3,b3,c3]]) * TRI_AREA
                mat = np.array([bary_pts[i], bary_pts[j], bary_pts[k]])
                det = np.linalg.det(mat)
                area = abs(det) * TRI_AREA
                if area < min_area:
                    min_area = area
    return min_area

def _clamp_barycentric(bary):
    """Clamp barycentric coordinates to be valid (non-negative, sum to 1)."""
    b = np.clip(bary, 0.0, 1.0)
    s = b.sum()
    if s < 1e-12:
        return np.array([1.0/3, 1.0/3, 1.0/3])
    return b / s

def _initial_placement():
    """Create an initial symmetric placement of 11 points."""
    # Use a grid-like pattern in barycentric coordinates
    # Place points in a roughly uniform grid
    bary_pts = []
    
    # Corners and edge midpoints
    bary_pts.append(np.array([1.0, 0.0, 0.0]))  # A
    bary_pts.append(np.array([0.0, 1.0, 0.0]))  # B
    bary_pts.append(np.array([0.0, 0.0, 1.0]))  # C
    
    # Edge midpoints
    bary_pts.append(np.array([0.5, 0.5, 0.0]))  # AB mid
    bary_pts.append(np.array([0.5, 0.0, 0.5]))  # AC mid
    bary_pts.append(np.array([0.0, 0.5, 0.5]))  # BC mid
    
    # Interior points - use a small grid
    # Center
    bary_pts.append(np.array([1/3, 1/3, 1/3]))
    
    # Additional interior points
    bary_pts.append(np.array([0.4, 0.3, 0.3]))
    bary_pts.append(np.array([0.3, 0.4, 0.3]))
    bary_pts.append(np.array([0.3, 0.3, 0.4]))
    
    return np.array(bary_pts)

def _gaussian_jitter(bary_pts, sigma):
    """Apply Gaussian jitter in barycentric space, then project back to valid region."""
    n = bary_pts.shape[0]
    new_bary = bary_pts.copy()
    for i in range(n):
        noise = np.random.normal(0, sigma, size=3)
        new_bary[i] = bary_pts[i] + noise
        new_bary[i] = _clamp_barycentric(new_bary[i])
    return new_bary

def _swap_barycentric(bary_pts, i, j):
    """Swap barycentric coordinates of points i and j."""
    new_bary = bary_pts.copy()
    new_bary[i], new_bary[j] = bary_pts[j].copy(), bary_pts[i].copy()
    return new_bary

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit
    
    # Initial placement
    bary_pts = _initial_placement()
    best_bary = bary_pts.copy()
    best_area = _min_triangle_area_bary(bary_pts)
    
    # Simulated annealing parameters
    T_start = 0.01
    T_end = 1e-6
    # We'll adaptively determine the number of steps based on time budget
    
    rng = np.random.RandomState(42)
    
    # Precompute all triangle indices for faster evaluation
    tri_indices = list(itertools.combinations(range(N), 3))
    
    def fast_min_area(bary):
        """Fast computation of min triangle area using precomputed indices."""
        min_a = float('inf')
        for i, j, k in tri_indices:
            mat = np.array([bary[i], bary[j], bary[k]])
            det = mat[0,0]*(mat[1,1]*mat[2,2] - mat[1,2]*mat[2,1]) - \
                  mat[0,1]*(mat[1,0]*mat[2,2] - mat[1,2]*mat[2,0]) + \
                  mat[0,2]*(mat[1,0]*mat[2,1] - mat[1,1]*mat[2,0])
            area = abs(det) * TRI_AREA
            if area < min_a:
                min_a = area
        return min_a
    
    best_area = fast_min_area(bary_pts)
    best_bary = bary_pts.copy()
    
    current_area = best_area
    T = T_start
    
    # Determine cooling schedule based on available time
    # First, estimate how many steps we can do
    # Do a quick timing test
    test_start = time.time()
    for _ in range(100):
        fast_min_area(bary_pts)
    test_elapsed = time.time() - test_start
    avg_step_time = test_elapsed / 100.0
    
    available_time = deadline - time.time()
    max_steps = int(available_time / (avg_step_time * 1.5))  # 1.5x for jitter/swap overhead
    max_steps = max(max_steps, 1000)
    
    # Cooling schedule
    if max_steps > 1:
        cooling_rate = (T_end / T_start) ** (1.0 / max_steps)
    else:
        cooling_rate = 1.0
    
    for step in range(max_steps):
        if time.time() > deadline - 2.0:
            break
        
        # Generate a new candidate
        move_type = rng.random()
        
        if move_type < 0.5:
            # Local jitter
            sigma = 0.02 * (T / T_start + 0.1)
            new_bary = _gaussian_jitter(bary_pts, sigma)
        elif move_type < 0.8:
            # Swap two points
            i, j = rng.choice(N, size=2, replace=False)
            new_bary = _swap_barycentric(bary_pts, i, j)
        else:
            # Larger move: jitter multiple points
            n_jitter = rng.randint(2, 4)
            indices = rng.choice(N, size=n_jitter, replace=False)
            new_bary = bary_pts.copy()
            for idx in indices:
                noise = np.random.normal(0, 0.03, size=3)
                new_bary[idx] = _clamp_barycentric(bary_pts[idx] + noise)
        
        new_area = fast_min_area(new_bary)
        
        # Acceptance criterion
        if new_area < current_area:
            # Always accept improvements
            bary_pts = new_bary
            current_area = new_area
            if new_area > best_area:
                best_area = new_area
                best_bary = new_bary.copy()
        else:
            # Accept worse with probability exp(-delta/T)
            delta = current_area - new_area
            if delta < 1e-12:
                accept = True
            else:
                accept_prob = np.exp(-delta / T)
                accept = rng.random() < accept_prob
            
            if accept:
                bary_pts = new_bary
                current_area = new_area
        
        # Cool down
        T *= cooling_rate
    
    # Final refinement: try to improve best solution with small local moves
    if time.time() < deadline - 1.0:
        bary_pts = best_bary.copy()
        current_area = best_area
        T = 0.001
        cooling_rate = (1e-5 / T) ** (1.0 / 2000)
        
        for step in range(2000):
            if time.time() > deadline - 1.0:
                break
            
            sigma = 0.005 * (T / 0.001 + 0.1)
            new_bary = _gaussian_jitter(bary_pts, sigma)
            new_area = fast_min_area(new_bary)
            
            if new_area > current_area:
                bary_pts = new_bary
                current_area = new_area
                if new_area > best_area:
                    best_area = new_area
                    best_bary = new_bary.copy()
            else:
                delta = current_area - new_area
                if delta > 1e-12:
                    accept_prob = np.exp(-delta / T)
                    if rng.random() < accept_prob:
                        bary_pts = new_bary
                        current_area = new_area
            
            T *= cooling_rate
    
    # Convert best barycentric to cartesian
    points = np.zeros((N, 2))
    for i in range(N):
        points[i] = _barycentric_to_cart(best_bary[i])
    
    return points
