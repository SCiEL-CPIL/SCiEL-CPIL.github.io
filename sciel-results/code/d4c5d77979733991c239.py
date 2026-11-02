"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time
import random

N = 21
TIME_LIMIT = 115.0  # Safety margin before 120s kill

def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time

    # State: x[N], y[N], r[N], W
    # H = 2 - W
    # Objective: Maximize sum(r) - lambda * Penalty
    # Penalty = sum(overlap^2) + sum(boundary_violation^2)
    
    def calculate_penalty(x, y, r, W):
        H = 2.0 - W
        if W <= 0 or H <= 0:
            return 1e10, -1e10  # Invalid state
        
        pen = 0.0
        # Boundary penalties
        for i in range(N):
            # Left
            if x[i] < r[i]:
                pen += (r[i] - x[i])**2
            # Right
            if x[i] > W - r[i]:
                pen += (x[i] - (W - r[i]))**2
            # Bottom
            if y[i] < r[i]:
                pen += (r[i] - y[i])**2
            # Top
            if y[i] > H - r[i]:
                pen += (y[i] - (H - r[i]))**2
        
        # Pairwise overlap penalties
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                min_dist = r[i] + r[j]
                if dist_sq < min_dist*min_dist:
                    dist = np.sqrt(dist_sq)
                    if dist < 1e-9:
                        pen += min_dist*min_dist
                    else:
                        pen += (min_dist - dist)**2
        
        sum_r = np.sum(r)
        return pen, sum_r

    def objective(params, lambda_pen=10.0):
        x = params[0:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        W = params[3*N]
        pen, sum_r = calculate_penalty(x, y, r, W)
        return -(sum_r - lambda_pen * pen)  # Minimize negative score

    def initial_hex_guess(W_init):
        H_init = 2.0 - W_init
        if H_init <= 0:
            H_init = 1.0
            W_init = 1.0
        
        # Estimate hex packing
        # Area per circle approx pi * r^2. Hex packing density ~ 0.9069
        # Total area = W*H. N * pi * r^2 * 0.9069 <= W*H
        # r_max_est = sqrt(W*H / (N * pi * 0.9069))
        
        # Let's just try a grid of hex centers
        # Vertical spacing: r * sqrt(3)
        # Horizontal spacing: 2 * r
        # We need to fit N circles.
        
        # Try a range of potential radii to find a good fit
        best_params = None
        best_score = -np.inf
        
        # Estimate average radius
        r_est = np.sqrt(W_init * H_init / (N * np.pi * 1.1)) # Conservative
        
        # Try a few aspect ratios and initial radii
        for r_try in [r_est * 0.8, r_est, r_est * 1.2]:
            # Determine rows and cols
            # Hex lattice:
            # Row 0: 0, 2r, 4r...
            # Row 1: r, 3r, 5r... (shifted by r)
            # Vertical step: r * sqrt(3)
            
            # Max rows
            max_rows = int(H_init / (r_try * np.sqrt(3))) + 2
            max_cols = int(W_init / (2 * r_try)) + 2
            
            # Generate centers
            centers = []
            for row in range(max_rows):
                y_c = (row + 0.5) * r_try * np.sqrt(3)
                if y_c + r_try > H_init:
                    continue
                for col in range(max_cols):
                    if row % 2 == 0:
                        x_c = (col + 0.5) * 2 * r_try
                    else:
                        x_c = (col + 1.0) * 2 * r_try
                    if x_c + r_try > W_init:
                        continue
                    if x_c - r_try < 0:
                        continue
                    centers.append((x_c, y_c))
            
            if len(centers) < N:
                continue
            
            # Take first N centers? Or random N?
            # Better to take a contiguous block or random selection to avoid bias
            # Let's take a random subset of size N from available centers
            # But we need to ensure they are "packed" well.
            # Actually, just taking the first N in row-major order is fine for init.
            
            selected = centers[:N]
            x = np.array([p[0] for p in selected])
            y = np.array([p[1] for p in selected])
            r = np.full(N, r_try)
            
            # Adjust W to fit? No, W is fixed in this loop for init.
            # We will optimize W later.
            
            params = np.concatenate([x, y, r, [W_init]])
            
            # Check validity roughly
            pen, sum_r = calculate_penalty(x, y, r, W_init)
            score = sum_r - 10.0 * pen
            if score > best_score:
                best_score = score
                best_params = params
        
        if best_params is None:
            # Fallback to grid
            side = int(np.ceil(np.sqrt(N)))
            cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.01) for i in range(side) for j in range(side)]
            x = np.array([c[0] for c in cells[:N]])
            y = np.array([c[1] for c in cells[:N]])
            r = np.array([c[2] for c in cells[:N]])
            best_params = np.concatenate([x, y, r, [1.0]])
            
        return best_params

    # Simulated Annealing
    def simulated_annealing(params0, time_limit):
        params = params0.copy()
        T = 0.1  # Initial temperature
        T_min = 1e-6
        cooling = 0.995
        
        current_pen, current_sum_r = calculate_penalty(params[0:N], params[N:2*N], params[2*N:3*N], params[3*N])
        current_score = current_sum_r - 10.0 * current_pen
        
        best_params = params.copy()
        best_score = current_score
        
        start_sa = time.time()
        
        while time.time() - start_sa < time_limit:
            if elapsed() > TIME_LIMIT - 5:
                break
            
            # Propose move
            new_params = params.copy()
            
            # Randomly pick a circle to perturb
            idx = np.random.randint(N)
            
            # Perturbation size depends on T
            # Scale with current radius or average radius
            avg_r = np.mean(params[2*N:3*N])
            scale = max(avg_r * 0.5, 0.01) * T * 10 
            
            # Perturb x, y
            new_params[0+idx] += np.random.normal(0, scale)
            new_params[N+idx] += np.random.normal(0, scale)
            
            # Perturb r
            new_params[2*N+idx] += np.random.normal(0, scale * 0.5)
            # Ensure r > 0
            if new_params[2*N+idx] < 0.01:
                new_params[2*N+idx] = 0.01
                
            # Occasionally perturb W
            if np.random.random() < 0.1:
                new_params[3*N] += np.random.normal(0, 0.05 * T * 10)
                # Clamp W to (0.1, 1.9) to keep H positive
                new_params[3*N] = np.clip(new_params[3*N], 0.1, 1.9)
            
            # Evaluate new state
            new_x = new_params[0:N]
            new_y = new_params[N:2*N]
            new_r = new_params[2*N:3*N]
            new_W = new_params[3*N]
            
            new_pen, new_sum_r = calculate_penalty(new_x, new_y, new_r, new_W)
            new_score = new_sum_r - 10.0 * new_pen
            
            delta_score = new_score - current_score
            
            if delta_score > 0 or np.random.random() < np.exp(delta_score / T):
                params = new_params
                current_score = new_score
                current_pen = new_pen
                current_sum_r = new_sum_r
                
                if current_score > best_score:
                    best_score = current_score
                    best_params = params.copy()
            
            # Cool down
            T *= cooling
            if T < T_min:
                T = T_min
        
        return best_params

    # Run SA for a few different initializations
    best_overall_params = None
    best_overall_score = -np.inf
    
    # Try a few initial aspect ratios
    for W_init in [0.8, 1.0, 1.2, 1.4]:
        if elapsed() > TIME_LIMIT - 20:
            break
            
        params0 = initial_hex_guess(W_init)
        
        # Run SA
        # Give SA about 10-15 seconds per run
        sa_time_limit = min(15.0, TIME_LIMIT - 10 - elapsed())
        if sa_time_limit < 1.0:
            break
            
        sa_result = simulated_annealing(params0, sa_time_limit)
        
        # Evaluate SA result
        x = sa_result[0:N]
        y = sa_result[N:2*N]
        r = sa_result[2*N:3*N]
        W = sa_result[3*N]
        
        pen, sum_r = calculate_penalty(x, y, r, W)
        score = sum_r - 10.0 * pen
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_params = sa_result.copy()
            
    if best_overall_params is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    # Polish with L-BFGS-B
    # We need a smooth objective for L-BFGS.
    # The penalty function is C1 continuous (squared terms), so it's smooth enough.
    
    def smooth_objective(params):
        x = params[0:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        W = params[3*N]
        
        H = 2.0 - W
        
        # Soft penalty
        pen = 0.0
        # Boundaries
        for i in range(N):
            # Use softplus-like or just squared
            if x[i] < r[i]:
                pen += (r[i] - x[i])**2
            if x[i] > W - r[i]:
                pen += (x[i] - (W - r[i]))**2
            if y[i] < r[i]:
                pen += (r[i] - y[i])**2
            if y[i] > H - r[i]:
                pen += (y[i] - (H - r[i]))**2
        
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                min_dist = r[i] + r[j]
                if dist_sq < min_dist*min_dist:
                    dist = np.sqrt(dist_sq)
                    if dist < 1e-9:
                        pen += min_dist*min_dist
                    else:
                        pen += (min_dist - dist)**2
                        
        sum_r = np.sum(r)
        # We want to maximize sum_r - lambda * pen
        # Minimize: -(sum_r - lambda * pen)
        return -(sum_r - 10.0 * pen)
        
    if elapsed() < TIME_LIMIT - 5:
        try:
            res = minimize(smooth_objective, best_overall_params, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            
            # Check if result is better
            x = res.x[0:N]
            y = res.x[N:2*N]
            r = res.x[2*N:3*N]
            W = res.x[3*N]
            
            # Ensure validity
            valid = True
            H = 2.0 - W
            if W <= 0 or H <= 0:
                valid = False
            else:
                for i in range(N):
                    if x[i] - r[i] < -1e-6 or x[i] + r[i] > W + 1e-6 or y[i] - r[i] < -1e-6 or y[i] + r[i] > H + 1e-6:
                        valid = False
                        break
                    if r[i] <= 0:
                        valid = False
                        break
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            dx = x[i] - x[j]
                            dy = y[i] - y[j]
                            dist = np.sqrt(dx*dx + dy*dy)
                            if dist < r[i] + r[j] - 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
            
            if valid:
                # Calculate actual value
                val = np.sum(r)
                # Compare with best_overall_score? 
                # best_overall_score was sum_r - 10*pen. 
                # If pen is 0, score = sum_r.
                # So if valid, score = val.
                if val > best_overall_score:
                    best_overall_params = res.x.copy()
                    best_overall_score = val
        except Exception:
            pass
            
    # Final cleanup: ensure strict non-overlap and boundaries
    x = best_overall_params[0:N]
    y = best_overall_params[N:2*N]
    r = best_overall_params[2*N:3*N]
    W = best_overall_params[3*N]
    H = 2.0 - W
    
    # Shrink radii slightly to ensure disjoint interiors
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < r[i] + r[j]:
                shrink = (r[i] + r[j] - dist) / 2 + 1e-9
                r[i] -= shrink
                r[j] -= shrink
        # Clamp to boundaries
        r[i] = min(r[i], x[i], W - x[i], y[i], H - y[i])
        r[i] = max(r[i], 1e-6)
        
    result = np.column_stack([x, y, r])
    return result

if __name__ == "__main__":
    construct()
