import numpy as np
import time

N = 21

def construct():
    start_time = time.time()
    deadline = 145  # Safety margin from 150s limit

    best_score = 0
    best_solution = None

    def elapsed():
        return time.time() - start_time

    def is_valid(x, y, r, W, H, tol=1e-9):
        """Check if circles are all within [0,W]x[0,H] and non-overlapping."""
        # Boundary
        if np.any(x < -tol) or np.any(y < -tol):
            return False
        if np.any(x + r > W + tol) or np.any(y + r > H + tol):
            return False
        # Overlap
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < r[i] + r[j] - tol:
                    return False
        return True

    def sum_radii(r):
        return np.sum(r)

    def perimeter(W, H):
        return 2 * (W + H)

    def make_initial(W, H, rng):
        """Create an initial feasible configuration."""
        # Start with small circles in a grid pattern
        n_grid = 5
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            row = i // n_grid
            col = i % n_grid
            x[i] = (col + 0.5) * W / n_grid
            y[i] = (row + 0.5) * H / n_grid
            r[i] = min(W, H) / n_grid * 0.3
        
        # Ensure feasibility by shrinking if needed
        for _ in range(100):
            if is_valid(x, y, r, W, H):
                break
            r *= 0.95
        
        return x, y, r

    def simulated_annealing(x0, y0, r0, W, H, max_iter, rng):
        """Run simulated annealing to maximize sum of radii."""
        x = x0.copy()
        y = y0.copy()
        r = r0.copy()
        
        T = 0.1
        T_min = 1e-6
        cooling = 0.9995
        
        best_local_score = sum_radii(r)
        best_local_x, best_local_y, best_local_r = x.copy(), y.copy(), r.copy()
        
        for it in range(max_iter):
            if elapsed() > deadline - 5:
                break
            
            # Choose a random circle to perturb
            i = rng.integers(0, N)
            
            # Decide what to perturb: position or radius
            if rng.random() < 0.7:
                # Perturb position
                dx = rng.normal(0, 0.01)
                dy = rng.normal(0, 0.01)
                x_new = x.copy()
                y_new = y.copy()
                x_new[i] += dx
                y_new[i] += dy
                # Keep in bounds
                x_new[i] = np.clip(x_new[i], 0, W)
                y_new[i] = np.clip(y_new[i], 0, H)
                r_new = r.copy()
            else:
                # Perturb radius
                dr = rng.normal(0, 0.005)
                r_new = r.copy()
                r_new[i] = max(0.001, r_new[i] + dr)
                x_new = x.copy()
                y_new = y.copy()
            
            # Check feasibility of new state
            if is_valid(x_new, y_new, r_new, W, H):
                score_new = sum_radii(r_new)
                score_old = sum_radii(r)
                delta = score_new - score_old
                
                if delta > 0 or rng.random() < np.exp(delta / T):
                    x, y, r = x_new, y_new, r_new
                    if score_new > best_local_score:
                        best_local_score = score_new
                        best_local_x, best_local_y, best_local_r = x.copy(), y.copy(), r.copy()
            else:
                # Try to repair: shrink the moved circle
                for shrink in [0.9, 0.8, 0.7, 0.6, 0.5]:
                    r_try = r_new.copy()
                    r_try[i] *= shrink
                    if is_valid(x_new, y_new, r_try, W, H):
                        score_new = sum_radii(r_try)
                        score_old = sum_radii(r)
                        delta = score_new - score_old
                        if delta > 0 or rng.random() < np.exp(delta / T):
                            x, y, r = x_new, y_new, r_try
                            if score_new > best_local_score:
                                best_local_score = score_new
                                best_local_x, best_local_y, best_local_r = x.copy(), y.copy(), r.copy()
                        break
            
            T = max(T_min, T * cooling)
        
        return best_local_x, best_local_y, best_local_r, best_local_score

    # Try different aspect ratios (W * H = 1 since perimeter = 4 => 2(W+H)=4 => W+H=2)
    # So H = 2 - W, and W in (0, 2)
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5]
    
    rng = np.random.default_rng(42)
    
    for W in aspect_ratios:
        if elapsed() > deadline - 10:
            break
        H = 2.0 - W
        if H <= 0.1:
            continue
        
        # Multiple restarts per aspect ratio
        n_restarts = 3
        for restart in range(n_restarts):
            if elapsed() > deadline - 10:
                break
            
            x0, y0, r0 = make_initial(W, H, rng)
            max_iter = 5000
            x_best, y_best, r_best, score = simulated_annealing(x0, y0, r0, W, H, max_iter, rng)
            
            # Validate and scale back to perimeter-4 rectangle
            # The rectangle is W x H with 2(W+H) = 4, so it's already perimeter 4
            if score > best_score:
                best_score = score
                best_solution = np.column_stack((x_best, y_best, r_best))
            
            # If we have a good solution, try to refine it
            if best_score > 1.0 and elapsed() < deadline - 30:
                # Do another round of SA starting from best
                x_ref, y_ref, r_ref, score_ref = simulated_annealing(
                    x_best, y_best, r_best, W, H, 3000, rng
                )
                if score_ref > best_score:
                    best_score = score_ref
                    best_solution = np.column_stack((x_ref, y_ref, r_ref))
    
    # Final refinement on the best solution found
    if best_solution is not None and elapsed() < deadline - 20:
        x_best = best_solution[:, 0]
        y_best = best_solution[:, 1]
        r_best = best_solution[:, 2]
        
        # Determine which aspect ratio was used by checking bounding box
        W_est = max(x_best + r_best)
        H_est = max(y_best + r_best)
        # Snap to nearest valid aspect ratio
        for W in aspect_ratios:
            H = 2.0 - W
            if abs(W - W_est) < 0.1 and abs(H - H_est) < 0.1:
                break
        
        x_ref, y_ref, r_ref, score_ref = simulated_annealing(
            x_best, y_best, r_best, W, 2.0 - W, 5000, rng
        )
        if score_ref > best_score:
            best_score = score_ref
            best_solution = np.column_stack((x_ref, y_ref, r_ref))
    
    # Fallback: simple grid in 1x1 square
    if best_solution is None or best_score < 0.5:
        n_grid = 5
        x_list, y_list, r_list = [], [], []
        for i in range(N):
            row = i // n_grid
            col = i % n_grid
            x_list.append((col + 0.5) / n_grid)
            y_list.append((row + 0.5) / n_grid)
            r_list.append(0.5 / n_grid * 0.9)
        best_solution = np.column_stack((x_list, y_list, r_list))
    
    return best_solution
