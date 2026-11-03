import time
import numpy as np
from scipy.spatial import ConvexHull
import random

N = 13

# Precompute indices for all triples to speed up score calculation
# There are 286 triples for N=13
TRIPLES = np.array([(i, j, k) for i in range(N - 2) for j in range(i + 1, N - 1) for k in range(j + 1, N)])
NUM_TRIPLES = len(TRIPLES)

def compute_score_and_details(points):
    """
    Compute the score: min triangle area / convex hull area.
    Also returns the indices of the active triangles (those achieving the min).
    Vectorized for speed.
    """
    # Calculate convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0, np.array([], dtype=int)
    except Exception:
        return 0.0, np.array([], dtype=int)

    # Vectorized triangle area calculation
    p1 = points[TRIPLES[:, 0]]
    p2 = points[TRIPLES[:, 1]]
    p3 = points[TRIPLES[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    
    # Cross product in 2D: v1_x * v2_y - v1_y * v2_x
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    if min_area < 1e-10:
        return 0.0, np.array([], dtype=int)
        
    score = min_area / hull_area
    
    # Find active triangles (indices where area is close to min)
    # We use a small tolerance to include triangles that are "nearly" minimal,
    # as the min operator is non-smooth.
    tol = 1e-5
    active_mask = np.abs(areas - min_area) < tol
    active_indices = np.where(active_mask)[0]
    
    return score, active_indices

def generate_initialization(method='jittered_polygon'):
    """
    Generate initial configurations.
    """
    if method == 'jittered_polygon':
        # Start with a regular polygon but perturb angles and radii slightly
        # to break symmetry and allow the optimizer to find a non-symmetric optimum
        theta = np.linspace(0, 2*np.pi, N, endpoint=False)
        # Add small random noise to angles
        theta += np.random.uniform(-0.05, 0.05, N)
        # Add small random noise to radii
        radius = 0.5 + np.random.uniform(-0.05, 0.05, N)
        center = [0.5, 0.5]
        points = np.array([[center[0] + r * np.cos(t), center[1] + r * np.sin(t)] for t, r in zip(theta, radius)])
        return points
    elif method == 'halton':
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r
        points = []
        for i in range(1, N + 1):
            x = halton(i, 2)
            y = halton(i, 3)
            points.append([x, y])
        return np.array(points)
    else:
        return np.random.rand(N, 2)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 5s safety margin

    best_points = None
    best_score = 0.0
    
    # Strategy: 
    # 1. Start with a good initialization (jittered polygon).
    # 2. Use Gradient-Steered Simulated Annealing.
    #    - Maintain a set of "active" triangles.
    #    - For each point, calculate a "force" based on the gradients of the active triangles.
    #    - Move points in the direction of the force, scaled by temperature.
    #    - Accept moves based on Metropolis criterion (or just hill climbing with annealing).
    
    # Initialize
    init_pts = generate_initialization('jittered_polygon')
    current_points = init_pts.copy()
    current_score, active_triples = compute_score_and_details(current_points)
    
    best_points = current_points.copy()
    best_score = current_score
    
    # Parameters
    T = 0.01  # Initial temperature (step size scale)
    cooling_rate = 0.995
    min_T = 1e-7
    
    # Time budget per iteration is roughly 1ms if optimized.
    # We can do ~50k-100k iterations in 115s.
    
    # Pre-allocate arrays for speed
    # Gradients for each point: (N, 2)
    
    while time.time() < deadline:
        # Check time frequently in inner loops if N is small, but N=13 is small.
        
        # Update active triples periodically or if score changes significantly?
        # For simplicity and robustness, we recompute active triples every K iterations
        # or if the score drops (rejection). But computing active triples is cheap (vectorized).
        
        # Let's recompute active triples every 100 iterations to save overhead
        # actually, computing areas is the main cost. We already did it in compute_score.
        # To save time, we will just use the `active_triples` from the last full score calculation.
        # We will only update `active_triples` if we accept a move.
        
        # Pick a random point to move
        idx = random.randint(0, N - 1)
        
        # Calculate gradient approximation for point idx
        # We sum the contributions from active triangles that include idx.
        grad = np.zeros(2)
        
        # Find which active triples include idx
        # active_triples are indices into TRIPLES array.
        # TRIPLES[t] = (i, j, k)
        # We need to check if idx is i, j, or k.
        
        # Vectorized check:
        # This is a bit slow if done naively every step. 
        # Optimization: Pre-compute for each point, which triples it belongs to.
        
        # Let's pre-compute point_to_triples mapping outside the loop? 
        # No, let's just iterate over active_triples. 
        # If active_triples is small (e.g., < 50), this is fast.
        
        for t_idx in active_triples:
            i, j, k = TRIPLES[t_idx]
            if idx != i and idx != j and idx != k:
                continue
            
            p1 = current_points[i]
            p2 = current_points[j]
            p3 = current_points[k]
            
            # Area = 0.5 * |cross(v1, v2)|
            # Let v1 = p2 - p1, v2 = p3 - p1
            # If idx == i:
            #   dA/dp1 = -0.5 * cross(v1, v2) ... derivative of |x| is sign(x)
            # Actually, let's use the formula:
            # Area = 0.5 * | (x2-x1)(y3-y1) - (y2-y1)(x3-x1) |
            # Let C = (x2-x1)(y3-y1) - (y2-y1)(x3-x1)
            # Area = 0.5 * |C|
            # dArea/dp = 0.5 * sign(C) * dC/dp
            
            v1 = p2 - p1
            v2 = p3 - p1
            C = v1[0]*v2[1] - v1[1]*v2[0]
            
            if C == 0:
                continue
                
            sign_C = 1.0 if C > 0 else -1.0
            
            # Gradient of C with respect to points
            # C = (x2-x1)(y3-y1) - (y2-y1)(x3-x1)
            
            if idx == i:
                # dC/dx1 = -1*(y3-y1) - (y2-y1)*(-1) = y1-y3 + y2-y1 = y2-y3
                # dC/dy1 = -(x2-x1) - (y2-y1)*(-1) ... wait.
                # C = x2*y3 - x2*y1 - x1*y3 + x1*y1 - y2*x3 + y2*x1 + y1*x3 - y1*x1
                # dC/dx1 = -y3 + y2 = y2 - y3
                # dC/dy1 = -x2 + x3 = x3 - x2
                dCx = p2[0] - p3[0] # y2-y3 is wrong dim, careful.
                # C = (x2-x1)(y3-y1) - (y2-y1)(x3-x1)
                # dC/dx1 = -(y3-y1) - (y2-y1)(-1) = -y3 + y1 + y2 - y1 = y2 - y3
                # dC/dy1 = -(x2-x1) - (y2-y1)(-1) = -x2 + x1 + y2 - y1 ... No.
                # Product rule:
                # d/dx1 [(x2-x1)(y3-y1)] = (-1)(y3-y1) + (x2-x1)(0) = y1 - y3
                # d/dx1 [(y2-y1)(x3-x1)] = (0)(x3-x1) + (y2-y1)(-1) = y1 - y2
                # dC/dx1 = (y1 - y3) - (y1 - y2) = y2 - y3
                # d/dy1 [(x2-x1)(y3-y1)] = (x2-x1)(-1) + 0 = x1 - x2
                # d/dy1 [(y2-y1)(x3-x1)] = (-1)(x3-x1) + 0 = x1 - x3
                # dC/dy1 = (x1 - x2) - (x1 - x3) = x3 - x2
                
                dC_dx1 = p2[1] - p3[1]
                dC_dy1 = p3[0] - p2[0]
                grad[0] += 0.5 * sign_C * dC_dx1
                grad[1] += 0.5 * sign_C * dC_dy1
                
            elif idx == j:
                # d/dx2 [(x2-x1)(y3-y1)] = (y3-y1)
                # d/dx2 [(y2-y1)(x3-x1)] = 0
                # dC/dx2 = y3 - y1
                # d/dy2 [(x2-x1)(y3-y1)] = 0
                # d/dy2 [(y2-y1)(x3-x1)] = (x3-x1)
                # dC/dy2 = -(x3-x1) = x1 - x3
                
                dC_dx2 = p3[1] - p1[1]
                dC_dy2 = p1[0] - p3[0]
                grad[0] += 0.5 * sign_C * dC_dx2
                grad[1] += 0.5 * sign_C * dC_dy2
                
            elif idx == k:
                # d/dx3 [(x2-x1)(y3-y1)] = 0
                # d/dx3 [(y2-y1)(x3-x1)] = (y2-y1)
                # dC/dx3 = -(y2-y1) = y1 - y2
                # d/dy3 [(x2-x1)(y3-y1)] = (x2-x1)
                # d/dy3 [(y2-y1)(x3-x1)] = 0
                # dC/dy3 = x2 - x1
                
                dC_dx3 = p1[1] - p2[1]
                dC_dy3 = p2[0] - p1[0]
                grad[0] += 0.5 * sign_C * dC_dx3
                grad[1] += 0.5 * sign_C * dC_dy3

        # Normalize gradient to prevent huge steps
        norm = np.linalg.norm(grad)
        if norm > 1e-12:
            grad = grad / norm
        
        # Proposed move: move in direction of gradient (increase area)
        # We want to MAXIMIZE min area. 
        # The gradient calculated is for the SUM of areas of active triangles.
        # Increasing the area of the smallest triangles increases the min.
        # So we move in the direction of the gradient.
        
        step_size = T
        new_points = current_points.copy()
        new_points[idx] = current_points[idx] + grad * step_size
        
        # Evaluate new score
        # To save time, we can do a quick check? No, compute_score is fast enough.
        new_score, new_active_triples = compute_score_and_details(new_points)
        
        # Metropolis Acceptance
        if new_score >= current_score:
            current_points = new_points
            current_score = new_score
            active_triples = new_active_triples
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
            
            # Cool down
            T *= cooling_rate
        else:
            # Accept with probability exp(-(delta)/T)? 
            # Here T is step size, not temperature for probability.
            # For annealing, we should have a separate temperature parameter.
            # Let's use simple hill climbing with random restarts if stuck.
            # Or, we can just not accept and reduce T?
            # If we don't accept, we might get stuck.
            # Let's try a small random perturbation instead of gradient if rejected often?
            # No, just reduce T and try again.
            T *= cooling_rate
            # If T gets too small, we might stagnate.
            # Let's keep T bounded.
            if T < min_T:
                T = min_T

    # Final check
    if best_points is None:
        best_points = init_pts

    return best_points

# Keep the original function name and signature
# The task says: "Keep the function names and signatures of the seed program."
# The seed program had: construct()
# It also had helpers like compute_score, generate_initialization, local_search_adaptive.
# I am rewriting from scratch, so I can define my own helpers, but construct() must exist.

if __name__ == "__main__":
    # For local testing
    pts = construct()
    score, _ = compute_score_and_details(pts)
    print(f"Score: {score}")
