import math
import numpy as np
import time
import random

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Initial configuration: Hexagonal grid
    # Center at (0,0)
    # Ring 0: (0,0)
    # Ring 1: 6 neighbors
    # Ring 2: 12 neighbors, pick 4 of them to make 11 total
    # Axial coords:
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)
    # Let's pick a symmetric subset of Ring 2
    axial_init = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (1, -2)
    ]
    
    def axial_to_cartesian(q, r):
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return np.array([x, y])

    P = np.array([axial_to_cartesian(q, r) for q, r in axial_init])
    
    # Randomize slightly to break symmetry
    P += np.random.normal(0, 0.01, P.shape)

    def calc_S(P):
        # P is (N, 2)
        # Minimize max_k W_k(theta) / sqrt(3)
        # W_k(theta) = max_i (n_k . p_i) - min_i (n_k . p_i)
        # n_k angles: theta + k * pi/3
        best_S = float('inf')
        best_theta = 0.0
        
        # Scan theta in [0, pi/6]
        # Step size 1 degree is ~0.017 rad. 60 steps is fine.
        for i in range(61):
            theta = i * (math.pi / 6) / 60.0
            max_W = 0.0
            for k in range(6):
                ang = theta + k * math.pi / 3.0
                nx = math.cos(ang)
                ny = math.sin(ang)
                projs = P[:, 0] * nx + P[:, 1] * ny
                W = np.max(projs) - np.min(projs)
                if W > max_W:
                    max_W = W
            S = max_W / math.sqrt(3)
            if S < best_S:
                best_S = S
                best_theta = theta
        
        return best_S, best_theta

    def calc_penalty(P):
        # Penalty for overlaps (distance < sqrt(3))
        pen = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(P[i] - P[j])
                if d < math.sqrt(3):
                    pen += (math.sqrt(3) - d) ** 2
        return pen

    def objective(P):
        S, _ = calc_S(P)
        pen = calc_penalty(P)
        return S + 1000.0 * pen

    # Simulated Annealing
    current_P = P.copy()
    current_S, current_theta = calc_S(current_P)
    current_pen = calc_penalty(current_P)
    current_cost = current_S + 1000.0 * current_pen
    
    best_P = current_P.copy()
    best_S = current_S
    best_theta = current_theta
    best_pen = current_pen
    best_cost = current_cost

    T = 0.5
    cooling = 0.999
    steps = 20000
    
    for step in range(steps):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        # Propose new position
        new_P = current_P.copy()
        idx = random.randint(0, N - 1)
        # Random move
        move = np.random.normal(0, 0.1, 2)
        new_P[idx] += move
        
        new_S, new_theta = calc_S(new_P)
        new_pen = calc_penalty(new_P)
        new_cost = new_S + 1000.0 * new_pen
        
        delta = new_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_P = new_P
            current_S = new_S
            current_theta = new_theta
            current_pen = new_pen
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_P = current_P.copy()
                best_S = current_S
                best_theta = current_theta
                best_pen = current_pen
                best_cost = current_cost
        
        T *= cooling
        if T < 0.001:
            break

    # Final refinement: Local search on best_P
    # Try small perturbations on each point
    for _ in range(500):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        improved = False
        for idx in range(N):
            for dx in [-0.01, 0, 0.01]:
                for dy in [-0.01, 0, 0.01]:
                    if dx == 0 and dy == 0: continue
                    trial_P = best_P.copy()
                    trial_P[idx, 0] += dx
                    trial_P[idx, 1] += dy
                    trial_S, trial_theta = calc_S(trial_P)
                    trial_pen = calc_penalty(trial_P)
                    trial_cost = trial_S + 1000.0 * trial_pen
                    if trial_cost < best_cost:
                        best_P = trial_P
                        best_S = trial_S
                        best_theta = trial_theta
                        best_pen = trial_pen
                        best_cost = trial_cost
                        improved = True
        if not improved:
            break

    # Calculate final center and rotation
    # The calc_S function finds the optimal theta for the given P.
    # But we also need the center (cx, cy) of the enclosing hexagon.
    # For the optimal theta, the center is the midpoint of the projections?
    # No, the center must satisfy the constraints for all k.
    # The optimal center for a fixed theta is the intersection of the strips.
    # Since we minimized max W_k, the strips are tight.
    # The center can be found by averaging the midpoints of the min/max projections for the "active" constraints?
    # Simpler: The center of the bounding box in the rotated frame.
    # Let's find the center that minimizes the max distance for the best_theta.
    
    theta = best_theta
    min_proj = np.zeros(6)
    max_proj = np.zeros(6)
    for k in range(6):
        ang = theta + k * math.pi / 3.0
        nx = math.cos(ang)
        ny = math.sin(ang)
        projs = best_P[:, 0] * nx + best_P[:, 1] * ny
        min_proj[k] = np.min(projs)
        max_proj[k] = np.max(projs)
        
    # The center c must be such that n_k . c is in [min_proj[k] + A, max_proj[k] - A] ?
    # Actually, the width W_k = max_proj[k] - min_proj[k].
    # A = max(W_k) / 2.
    # The center c is the point that minimizes the max deviation.
    # For a hexagon, if the widths are balanced, the center is the centroid of the "bounding" points.
    # Let's just use the centroid of the points as an approximation, or solve for c.
    # Solve for c: minimize max_k | n_k . c - mid_k | where mid_k = (min_proj[k] + max_proj[k])/2?
    # No, the constraint is that the hexagon contains the points.
    # The hexagon is defined by c and theta.
    # The "tightest" fit means the hexagon sides touch the points.
    # Let's find c by minimizing the sum of squared distances to the "midline" of the projections?
    # Actually, for a given theta, the optimal c is the center of the minimal enclosing circle of the projected intervals?
    # No.
    # Let's use the fact that the optimal c is the centroid of the points if the shape is centrally symmetric?
    # For 11 points, it might not be.
    # Let's solve for c using linear programming or simple averaging of the "active" constraints.
    # Active constraints are those where W_k is close to max_W.
    
    max_W = max(max_proj[k] - min_proj[k] for k in range(6))
    A = max_W / 2.0
    
    # The center c must satisfy:
    # min_proj[k] + A <= n_k . c <= max_proj[k] - A
    # Let's find c that satisfies this.
    # If we assume the hexagon is centered at the centroid of the points, it might not be optimal.
    # Let's try to find c by averaging the midpoints of the active strips.
    active_midpoints = []
    for k in range(6):
        W_k = max_proj[k] - min_proj[k]
        if W_k > max_W - 1e-6:
            mid = (min_proj[k] + max_proj[k]) / 2.0
            # n_k . c = mid
            active_midpoints.append((math.cos(theta + k * math.pi / 3.0), math.sin(theta + k * math.pi / 3.0), mid))
            
    if len(active_midpoints) >= 2:
        # Solve for c using least squares
        A_mat = np.array([[nx, ny] for nx, ny, _ in active_midpoints])
        b_vec = np.array([mid for _, _, mid in active_midpoints])
        c_vec, residuals, rank, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        cx, cy = c_vec
    else:
        cx, cy = np.mean(best_P, axis=0)
        
    # Recalculate S with this center to be sure
    # S = (D + APOTHEM) / APOTHEM where D is max projection from center?
    # No, S is determined by the width.
    # The side length is max_W / sqrt(3).
    
    side = best_S
    center = [float(cx), float(cy)]
    angle_deg = math.degrees(theta)
    
    inner = []
    for x, y in best_P:
        inner.append([float(x), float(y), float(angle_deg)])
        
    return inner, center, side, angle_deg

