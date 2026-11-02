"""
Heilbronn triangle problem: maximize minimum triangle area for 11 points.
Strategy:
1. Use a vectorized exact minimum area calculation (no smooth approximation).
2. Initialize with a greedy 'iterative closest pair' expansion to get a decent baseline.
3. Use scipy.optimize.dual_annealing for global optimization. This method is robust
   for non-smooth, non-convex problems and handles the 'max-min' landscape better
   than gradient-based methods on smoothed surrogates.
4. The objective is -min_area. We maximize min_area.
5. Points are represented in Cartesian coordinates (x, y).
6. Constraints (points inside triangle) are handled by projecting any point outside
   back to the nearest point inside the triangle within the objective function.
"""
import numpy as np
import time
import random
from scipy.optimize import dual_annealing

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 145.0  # Safety margin before 150s limit

# Triangle vertices
V_A = np.array([0.0, 0.0])
V_B = np.array([1.0, 0.0])
V_C = np.array([0.5, S3H])

# Pre-compute triplet indices for vectorized area calculation
def _get_triplets(n):
    triplets = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triplets.append((i, j, k))
    return np.array(triplets)

TRIPLETS = _get_triplets(N)
I_IDX = TRIPLETS[:, 0]
J_IDX = TRIPLETS[:, 1]
K_IDX = TRIPLETS[:, 2]

def _project_to_triangle(pts):
    """
    Project points inside the triangle defined by (0,0), (1,0), (0.5, sqrt(3)/2).
    pts: (N, 2) array of points.
    Returns: (N, 2) array of projected points.
    """
    x, y = pts[:, 0], pts[:, 1]
    
    # Define the half-planes containing the triangle.
    # 1. y >= 0
    # 2. y <= sqrt(3) * x  (Line from (0,0) to (0.5, S3H): slope = S3H/0.5 = sqrt(3))
    #    Actually, line AC: y = sqrt(3) x. The triangle is below this line?
    #    C is (0.5, sqrt(3)/2). sqrt(3)*0.5 = sqrt(3)/2. Yes.
    #    B is (1,0). 0 <= sqrt(3)*1. Yes.
    #    So y <= sqrt(3) x.
    # 3. y <= -sqrt(3) * (x - 1) = sqrt(3) * (1 - x) (Line from (1,0) to (0.5, S3H))
    #    Slope = (S3H - 0) / (0.5 - 1) = S3H / -0.5 = -sqrt(3).
    #    y - 0 = -sqrt(3)(x - 1) => y = -sqrt(3)x + sqrt(3).
    #    So y <= sqrt(3)(1 - x).
    
    # We need to find the closest point in the intersection of these 3 half-planes.
    # This is a convex feasibility problem. For simplicity and speed in an objective
    # function, we can do a simple iterative projection or just clip/reflect.
    # However, dual_annealing proposes moves. If a move goes outside, the penalty
    # should be high or we project.
    # Projecting to the convex hull is safer.
    
    # Simple approach: check constraints and adjust.
    # If y < 0, set y = 0.
    # If y > sqrt(3)*x, project onto line y=sqrt(3)x.
    # If y > sqrt(3)*(1-x), project onto line y=sqrt(3)*(1-x).
    
    # This simple clipping might not yield the closest point if multiple constraints
    # are violated, but it keeps points inside. For optimization, keeping them inside
    # is the priority. The "closest" point matters less than being inside.
    
    new_pts = pts.copy()
    
    # Constraint 1: y >= 0
    mask1 = new_pts[:, 1] < 0
    new_pts[mask1, 1] = 0
    
    # Constraint 2: y <= sqrt(3) * x
    # Violation if y - sqrt(3)*x > 0
    viol2 = new_pts[:, 1] - np.sqrt(3) * new_pts[:, 0]
    mask2 = viol2 > 0
    if np.any(mask2):
        # Project (x, y) onto y = sqrt(3)x
        # Line: sqrt(3)x - y = 0. Normal: (sqrt(3), -1).
        # d = (sqrt(3)*x - y) / (3 + 1) = viol2 / 4
        # x_proj = x - sqrt(3)*d
        # y_proj = y - (-1)*d = y + d
        d = viol2[mask2] / 4.0
        new_pts[mask2, 0] -= np.sqrt(3) * d
        new_pts[mask2, 1] += d
        
    # Constraint 3: y <= sqrt(3) * (1 - x)
    # Violation if y - sqrt(3)*(1-x) > 0 => y + sqrt(3)*x - sqrt(3) > 0
    viol3 = new_pts[:, 1] + np.sqrt(3) * new_pts[:, 0] - np.sqrt(3)
    mask3 = viol3 > 0
    if np.any(mask3):
        # Project (x, y) onto y = -sqrt(3)x + sqrt(3)
        # Line: sqrt(3)x + y - sqrt(3) = 0. Normal: (sqrt(3), 1).
        # d = viol3 / (3 + 1) = viol3 / 4
        # x_proj = x - sqrt(3)*d
        # y_proj = y - 1*d
        d = viol3[mask3] / 4.0
        new_pts[mask3, 0] -= np.sqrt(3) * d
        new_pts[mask3, 1] -= d
        
    # It's possible that projecting onto one line violates another.
    # A few iterations of projection usually converge for simple convex sets.
    for _ in range(3):
        # Repeat the above logic
        mask1 = new_pts[:, 1] < 0
        new_pts[mask1, 1] = 0
        
        viol2 = new_pts[:, 1] - np.sqrt(3) * new_pts[:, 0]
        mask2 = viol2 > 0
        if np.any(mask2):
            d = viol2[mask2] / 4.0
            new_pts[mask2, 0] -= np.sqrt(3) * d
            new_pts[mask2, 1] += d
            
        viol3 = new_pts[:, 1] + np.sqrt(3) * new_pts[:, 0] - np.sqrt(3)
        mask3 = viol3 > 0
        if np.any(mask3):
            d = viol3[mask3] / 4.0
            new_pts[mask3, 0] -= np.sqrt(3) * d
            new_pts[mask3, 1] -= d
            
    return new_pts

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    p_i = pts[I_IDX]
    p_j = pts[J_IDX]
    p_k = pts[K_IDX]
    
    dx1 = p_j[:, 0] - p_i[:, 0]
    dy1 = p_j[:, 1] - p_i[:, 1]
    dx2 = p_k[:, 0] - p_i[:, 0]
    dy2 = p_k[:, 1] - p_i[:, 1]
    
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _objective(x_flat):
    """
    Objective function for dual_annealing.
    x_flat: (2*N,) array of Cartesian coordinates.
    Returns: -min_area (since dual_annealing minimizes).
    """
    pts = x_flat.reshape(N, 2)
    # Project to ensure validity
    pts = _project_to_triangle(pts)
    
    min_area = _min_triangle_area(pts)
    
    # Add a small penalty if points are very close to boundary?
    # Not strictly necessary if projection works.
    
    return -min_area

def _greedy_init(rng):
    """
    Greedy initialization: Place points one by one to maximize the minimum
    distance to existing points (which correlates with min area).
    """
    pts = np.zeros((N, 2))
    
    # Start with centroid or a corner? Centroid is usually good for packing.
    # Let's start with a random point inside.
    r1 = rng.random()
    r2 = rng.random()
    if r1 + r2 > 1:
        r1 = 1 - r1
        r2 = 1 - r2
    # Barycentric to Cartesian
    x = r1 + 0.5 * r2
    y = S3H * r2
    pts[0] = [x, y]
    
    for i in range(1, N):
        best_pos = None
        best_score = -1
        
        # Sample many random candidates
        for _ in range(200):
            r1 = rng.random()
            r2 = rng.random()
            if r1 + r2 > 1:
                r1 = 1 - r1
                r2 = 1 - r2
            x = r1 + 0.5 * r2
            y = S3H * r2
            cand = np.array([x, y])
            
            # Calculate min distance to existing points
            dists = np.linalg.norm(pts[:i] - cand, axis=1)
            min_dist = np.min(dists)
            
            if min_dist > best_score:
                best_score = min_dist
                best_pos = cand
                
        pts[i] = best_pos
        
    return pts.flatten()

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    # 1. Greedy Initialization
    x0 = _greedy_init(rng)
    # Project x0 to be safe
    x0 = _project_to_triangle(x0.reshape(N, 2)).flatten()
    
    current_best_score = _min_triangle_area(x0.reshape(N, 2))
    best_x = x0.copy()
    
    # 2. Dual Annealing Optimization
    # Bounds for x and y
    # x in [0, 1], y in [0, S3H]
    bounds = [(0.0, 1.0), (0.0, S3H)] * N
    
    # dual_annealing parameters
    # initial_temp: initial temperature
    # accept: acceptance function
    # maxiter: max iterations
    # polish: whether to polish the result with local search
    
    # We have a time limit. We can run dual_annealing with a callback to check time.
    # Or just set maxiter conservatively.
    # dual_annealing is typically fast per iteration but might need many.
    
    def time_check_callback(x, f, accepted):
        if time.time() - start_time > DEADLINE - 10:
            raise TimeoutError("Time limit exceeded")
            
    try:
        result = dual_annealing(
            _objective,
            bounds=bounds,
            x0=x0,
            maxiter=1000,
            initial_temp=1.0,
            restart_temp_ratio=2e-5,
            visit=2.62,
            clip=0.15,
            accept=0.66,
            polish=True,
            callback=time_check_callback
        )
        
        if result.success or result.fun < -current_best_score:
            # result.fun is -min_area
            new_min_area = -result.fun
            if new_min_area > current_best_score:
                current_best_score = new_min_area
                best_x = result.x.copy()
                
    except TimeoutError:
        pass
    except Exception as e:
        # Fallback if dual_annealing fails
        pass
    
    # 3. Local Polish with Nelder-Mead or Powell on the best found
    # dual_annealing with polish=True already does a local search.
    # But we can do a quick Nelder-Mead on the best result to squeeze out more.
    
    if time.time() - start_time < DEADLINE - 5:
        from scipy.optimize import minimize
        
        # Use the best found as start
        x_start = best_x
        
        def nm_obj(x):
            return _objective(x)
            
        try:
            nm_res = minimize(
                nm_obj,
                x_start,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if nm_res.fun < _objective(best_x):
                new_min_area = -nm_res.fun
                if new_min_area > current_best_score:
                    current_best_score = new_min_area
                    best_x = nm_res.x.copy()
        except Exception:
            pass
            
    # 4. Multi-start with a few more random restarts if time permits
    # Since dual_annealing is a global method, it might not need many restarts,
    # but a couple of quick ones can help.
    
    remaining_time = DEADLINE - (time.time() - start_time)
    if remaining_time > 20:
        n_extra_starts = 3
        time_per_start = remaining_time / n_extra_starts
        
        for _ in range(n_extra_starts):
            if time.time() - start_time > DEADLINE - 5:
                break
                
            # Random start
            r1 = rng.random()
            r2 = rng.random()
            if r1 + r2 > 1:
                r1 = 1 - r1
                r2 = 1 - r2
            x = r1 + 0.5 * r2
            y = S3H * r2
            
            # Generate random points
            x_rand = np.zeros(2*N)
            for i in range(N):
                r1 = rng.random()
                r2 = rng.random()
                if r1 + r2 > 1:
                    r1 = 1 - r1
                    r2 = 1 - r2
                x_rand[2*i] = r1 + 0.5 * r2
                x_rand[2*i+1] = S3H * r2
                
            x_rand = _project_to_triangle(x_rand.reshape(N, 2)).flatten()
            
            try:
                # Quick dual_annealing
                # Reduce maxiter for quick runs
                ra_res = dual_annealing(
                    _objective,
                    bounds=bounds,
                    x0=x_rand,
                    maxiter=200,
                    initial_temp=1.0,
                    polish=True
                )
                
                new_min_area = -ra_res.fun
                if new_min_area > current_best_score:
                    current_best_score = new_min_area
                    best_x = ra_res.x.copy()
                    
            except Exception:
                pass

    final_pts = _project_to_triangle(best_x.reshape(N, 2))
    
    return final_pts

if __name__ == "__main__":
    pts = construct()
    print(f"Min area: {_min_triangle_area(pts)}")
