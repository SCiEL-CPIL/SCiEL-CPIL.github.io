import numpy as np
import time
import random
from math import exp, log

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 145.0  # Safety margin before 150s limit

# Pre-compute triplet indices for vectorized area calculation
_triplets = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _triplets.append((i, j, k))
_triplets = np.array(_triplets)
_i_idx = _triplets[:, 0]
_j_idx = _triplets[:, 1]
_k_idx = _triplets[:, 2]

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_k_idx]
    
    dx1 = p_j - p_i
    dx2 = p_k - p_i
    
    cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _soft_min_area(pts, alpha):
    """
    Compute a soft approximation of the minimum area using log-sum-exp.
    LSE(x) = -1/alpha * log(sum(exp(-alpha * x)))
    As alpha -> infinity, this approaches min(x).
    """
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_k_idx]
    
    dx1 = p_j - p_i
    dx2 = p_k - p_i
    
    cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Avoid log(0)
    areas = np.maximum(areas, 1e-12)
    
    # Log-sum-exp trick for stability
    max_a = np.max(areas)
    sum_exp = np.sum(np.exp(-alpha * (areas - max_a)))
    lse = -1.0 / alpha * (log(sum_exp) - alpha * max_a)
    
    return lse

def _check_inside(pts, eps=1e-9):
    """Check if all points are inside or on the boundary of the equilateral triangle."""
    for p in pts:
        x, y = p
        if y < -eps:
            return False
        if y > np.sqrt(3) * x + eps:
            return False
        if y > np.sqrt(3) * (1 - x) + eps:
            return False
        if x < -eps or x > 1 + eps:
            return False
    return True

def _project_to_triangle(pts):
    """Project points that are slightly outside the triangle back inside."""
    pts = np.array(pts, dtype=float)
    for i in range(len(pts)):
        x, y = pts[i]
        if y < 0:
            y = 0
        max_y = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x))
        if y > max_y:
            y = max_y
        if x < 0:
            x = 0
        if x > 1:
            x = 1
        pts[i] = [x, y]
    return pts

def _random_point_in_triangle(rng):
    """Generate a random point inside the equilateral triangle."""
    u = rng.random()
    v = rng.random()
    while u + v > 1:
        u = rng.random()
        v = rng.random()
    x = u + 0.5 * v
    y = S3H * v
    return [x, y]

def _structured_init():
    """
    Structured initialization: Vertices, Midpoints, Center, and some others.
    This provides a good starting point closer to the optimal configuration.
    """
    p_struct = [
        [0.0, 0.0], [1.0, 0.0], [0.5, S3H],
        [0.5, 0.0], [0.75, S3H/2], [0.25, S3H/2],
        [0.5, S3H/3],
        [0.25, 0.0], [0.75, 0.0],
        [0.125, S3H/4], [0.875, S3H/4]
    ]
    return np.array(p_struct)

def _simulated_annealing(pts, rng, time_limit):
    """
    Simulated Annealing with a soft-min objective.
    1. Start with a high temperature and a low alpha (smooth landscape).
    2. Gradually decrease temperature and increase alpha (sharper landscape).
    3. Accept worse moves with a probability based on temperature to escape local minima.
    """
    current_pts = pts.copy()
    current_score = _min_triangle_area(current_pts)
    
    # Initial parameters
    T = 0.01  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.999
    
    # Alpha schedule: start low (smooth) and increase (sharp)
    alpha_start = 50.0
    alpha_end = 5000.0
    
    # Total steps estimate: we will run until time limit or T < T_min
    # We estimate steps based on cooling rate
    # T * cooling_rate^steps = T_min => steps = log(T_min/T) / log(cooling_rate)
    # log(1e-6/0.01) / log(0.999) = log(1e-4) / log(0.999) ≈ -9.2 / -0.001 ≈ 9200 steps
    
    step = 0
    best_pts = current_pts.copy()
    best_score = current_score
    
    # We will use a fixed number of steps per "epoch" and adjust alpha
    # Or just run a loop until time limit.
    
    # To make it faster, we can batch some operations or just keep it simple.
    # 165 triplets * 11 points is small, so vectorized calculation is fast.
    
    while time.time() < time_limit and T > T_min:
        # Determine current alpha based on progress
        # progress = 1 - (T / T_initial)
        # Let's just linearly interpolate alpha from start to end over the total expected steps
        # Or use a fixed schedule. Let's use a schedule based on step count.
        # Total steps ~ 9200.
        progress = min(1.0, step / 9200.0)
        alpha = alpha_start + (alpha_end - alpha_start) * progress
        
        # Generate a random perturbation
        # Move one point randomly
        point_idx = rng.randint(0, N - 1)
        orig_p = current_pts[point_idx].copy()
        
        # Random step size, maybe decreasing with T
        step_size = 0.05 * T  # Scale step with temperature? Or fixed?
        # Let's use a fixed small step size for stability, or scale it.
        # If T is high, we want large jumps. If T is low, small jumps.
        step_size = 0.1 * T + 0.001
        
        angle = rng.uniform(0, 2 * np.pi)
        dx = step_size * np.cos(angle)
        dy = step_size * np.sin(angle)
        
        new_p = orig_p + np.array([dx, dy])
        
        # Project to triangle
        new_p = _project_to_triangle(np.array([new_p]))[0]
        
        # Calculate new score (soft min for acceptance, hard min for tracking)
        test_pts = current_pts.copy()
        test_pts[point_idx] = new_p
        
        # Use soft min for the energy calculation in SA
        new_soft_score = _soft_min_area(test_pts, alpha)
        current_soft_score = _soft_min_area(current_pts, alpha)
        
        delta_energy = new_soft_score - current_soft_score
        
        # Acceptance criterion
        if delta_energy < 0:
            # Better
            current_pts = test_pts
            T *= cooling_rate
        else:
            # Worse, accept with probability exp(-delta_energy / T)
            prob = exp(-delta_energy / T)
            if rng.random() < prob:
                current_pts = test_pts
            # In either case, cool down
            T *= cooling_rate
        
        # Track best hard-min score
        current_hard_score = _min_triangle_area(current_pts)
        if current_hard_score > best_score:
            best_score = current_hard_score
            best_pts = current_pts.copy()
            
        step += 1
        
        # Check time frequently
        if time.time() > time_limit:
            break
            
    return best_pts, best_score

def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize min triangle area.
    Uses Simulated Annealing with Soft-Min Objective in Barycentric-like (Cartesian projected) space.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_score = -1.0
    
    # Initializations
    # 1. Structured
    p_struct = _structured_init()
    
    # 2. Random
    p_rand = np.array([_random_point_in_triangle(rng) for _ in range(N)])
    p_rand = _project_to_triangle(p_rand)
    
    # 3. Greedy-like (maximize min distance)
    p_greedy = p_struct.copy() # Start with structured, then maybe refine? 
    # Let's just use structured and random for SA.
    
    inits = [p_struct, p_rand]
    
    # Run SA on each initialization
    # We have ~145s total.
    # Split time among initializations.
    time_per_init = (DEADLINE - (time.time() - start_time)) / len(inits)
    
    for init_pts in inits:
        if time.time() > DEADLINE - 5:
            break
            
        # Refine this initialization
        # Give it a specific time slice
        end_time_for_this = time.time() + time_per_init
        
        refined_pts, score = _simulated_annealing(init_pts, rng, end_time_for_this)
        
        if score > best_score:
            best_score = score
            best_pts = refined_pts.copy()
            
    # If we have time left, do a final refinement on the best found
    if time.time() < DEADLINE - 10 and best_pts is not None:
        # Run SA again on the best result with a lower starting T or more steps?
        # Or just run it again to see if we find something better.
        # Let's run it again with the remaining time.
        final_pts, final_score = _simulated_annealing(best_pts, rng, DEADLINE - 2)
        if final_score > best_score:
            best_score = final_score
            best_pts = final_pts.copy()

    if best_pts is None:
        best_pts = p_struct
        
    # Final check
    if not _check_inside(best_pts):
        best_pts = _project_to_triangle(best_pts)
        
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_min_triangle_area(pts)}")
