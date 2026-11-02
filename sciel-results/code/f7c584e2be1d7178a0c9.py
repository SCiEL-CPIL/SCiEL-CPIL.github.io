"""Circle packing in perimeter-4 rectangle: maximize sum of radii for 21 disjoint circles."""
import numpy as np
import time
import random

N = 21

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def elapsed():
        return time.time() - start_time
    
    def validate(w, circles):
        """Return sum of radii if valid, else 0.0"""
        h = 2.0 - w
        if w <= 0 or h <= 0:
            return 0.0
        n = len(circles)
        total_r = 0.0
        for i in range(n):
            xi, yi, ri = circles[i]
            if ri <= 1e-12:
                return 0.0
            if xi - ri < -1e-9 or xi + ri > w + 1e-9:
                return 0.0
            if yi - ri < -1e-9 or yi + ri > h + 1e-9:
                return 0.0
            total_r += ri
        for i in range(n):
            for j in range(i+1, n):
                dx = circles[i][0] - circles[j][0]
                dy = circles[i][1] - circles[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < circles[i][2] + circles[j][2] - 1e-9:
                    return 0.0
        return total_r
    
    def penalty_objective(params):
        """Penalty-based objective for optimization. Returns negative score (to minimize)."""
        w = params[0]
        h = 2.0 - w
        if w <= 0.05 or h <= 0.05:
            return 1e10
        
        circles = params[1:].reshape(N, 3)
        penalty = 0.0
        total_r = 0.0
        
        for i in range(N):
            xi, yi, ri = circles[i]
            if ri <= 0:
                penalty += 1000.0
                continue
            total_r += ri
            # Boundary violations
            if xi - ri < 0:
                penalty += 500.0 * (xi - ri)
            if xi + ri > w:
                penalty += 500.0 * (xi + ri - w)
            if yi - ri < 0:
                penalty += 500.0 * (yi - ri)
            if yi + ri > h:
                penalty += 500.0 * (yi + ri - h)
        
        # Overlap violations
        for i in range(N):
            for j in range(i+1, N):
                dx = circles[i][0] - circles[j][0]
                dy = circles[i][1] - circles[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                overlap = circles[i][2] + circles[j][2] - dist
                if overlap > 0:
                    penalty += 500.0 * overlap
        
        return penalty - total_r
    
    def init_circles(w, h):
        """Initialize circles in a grid-like pattern within the rectangle."""
        # Try different grid layouts
        best_circles = None
        best_pen = 1e10
        
        for rows in range(3, 8):
            cols = int(np.ceil(N / rows))
            if rows * cols < N:
                continue
            
            circles = []
            for i in range(rows):
                for j in range(cols):
                    if len(circles) >= N:
                        break
                    cx = (j + 0.5) * w / cols
                    cy = (i + 0.5) * h / rows
                    r = min(w/cols, h/rows) * 0.45
                    circles.append([cx, cy, r])
            
            params = np.array([w] + [x for c in circles for x in c])
            pen = penalty_objective(params)
            if pen < best_pen:
                best_pen = pen
                best_circles = circles
        
        if best_circles is None:
            # Fallback: simple grid
            side = int(np.ceil(np.sqrt(N)))
            best_circles = []
            for i in range(side):
                for j in range(side):
                    if len(best_circles) >= N:
                        break
                    cx = (j + 0.5) * w / side
                    cy = (i + 0.5) * h / side
                    r = min(w, h) / side * 0.45
                    best_circles.append([cx, cy, r])
        
        return best_circles
    
    def relax_positions(w, circles, max_iters=2000):
        """Relax positions to remove overlaps using gradient descent."""
        h = 2.0 - w
        circles = [list(c) for c in circles]
        lr = 0.01
        
        for it in range(max_iters):
            if elapsed() > deadline - 10:
                break
            
            # Compute forces
            forces = np.zeros((N, 2))
            overlap_count = 0
            
            for i in range(N):
                xi, yi, ri = circles[i]
                xj, yj, rj = [0, 0, 0]
                
                # Boundary forces
                if xi - ri < 0:
                    forces[i, 0] += (ri - xi) * 10.0
                if xi + ri > w:
                    forces[i, 0] -= (xi + ri - w) * 10.0
                if yi - ri < 0:
                    forces[i, 1] += (ri - yi) * 10.0
                if yi + ri > h:
                    forces[i, 1] -= (yi + ri - h) * 10.0
                
                # Overlap forces
                for j in range(i+1, N):
                    dx = xi - circles[j][0]
                    dy = yi - circles[j][1]
                    dist = np.sqrt(dx*dx + dy*dy) + 1e-10
                    overlap = ri + circles[j][2] - dist
                    if overlap > 0:
                        overlap_count += 1
                        f = overlap * 5.0 / dist
                        forces[i, 0] += dx * f
                        forces[i, 1] += dy * f
                        forces[j, 0] -= dx * f
                        forces[j, 1] -= dy * f
            
            if overlap_count == 0:
                break
            
            # Update positions
            for i in range(N):
                circles[i][0] += lr * forces[i, 0]
                circles[i][1] += lr * forces[i, 1]
                # Clamp to valid range
                circles[i][0] = max(circles[i][2], min(w - circles[i][2], circles[i][0]))
                circles[i][1] = max(circles[i][2], min(h - circles[i][2], circles[i][1]))
            
            lr *= 0.995
        
        return circles
    
    def maximize_radii(w, circles, max_iters=500):
        """Try to increase radii while maintaining validity."""
        h = 2.0 - w
        circles = [list(c) for c in circles]
        lr = 0.005
        best = [list(c) for c in circles]
        best_score = validate(w, best)
        
        for it in range(max_iters):
            if elapsed() > deadline - 5:
                break
            
            # Compute radius increase forces
            max_inc = np.zeros(N)
            
            for i in range(N):
                xi, yi, ri = circles[i]
                # Max increase from boundaries
                inc_b = min(
                    w - xi - ri,  # right
                    xi - ri,      # left
                    h - yi - ri,  # top
                    yi - ri       # bottom
                )
                
                # Max increase from other circles
                inc_c = inc_b
                for j in range(N):
                    if i == j:
                        continue
                    dx = xi - circles[j][0]
                    dy = yi - circles[j][1]
                    dist = np.sqrt(dx*dx + dy*dy) + 1e-10
                    # Distance to j's boundary
                    inc_c = min(inc_c, (dist - ri - circles[j][2]) / 2.0)
                
                max_inc[i] = max(0.0, inc_c)
            
            # Try to increase all radii
            success = True
            trial = [list(c) for c in circles]
            for i in range(N):
                if max_inc[i] > 1e-8:
                    trial[i][2] += lr * max_inc[i]
            
            # Check validity
            trial_score = validate(w, trial)
            if trial_score > best_score - 1e-6:
                circles = trial
                best = [list(c) for c in circles]
                best_score = trial_score
            else:
                # Try smaller step
                for i in range(N):
                    circles[i][2] -= lr * max_inc[i] * 0.5
        
        return best
    
    # Main optimization loop
    best_score = 0.0
    best_result = None
    
    # Try several aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 2.0, 0.4]
    random.shuffle(aspect_ratios)
    
    for ar in aspect_ratios:
        if elapsed() > deadline - 15:
            break
        
        w = ar / (1 + ar) * 2.0
        h = 2.0 - w
        
        # Initialize
        circles = init_circles(w, h)
        
        # Relax positions
        circles = relax_positions(w, circles)
        
        # Check validity
        score = validate(w, circles)
        if score > best_score:
            best_score = score
            best_result = (w, [list(c) for c in circles])
        
        # Maximize radii
        if elapsed() < deadline - 10:
            circles_opt = maximize_radii(w, circles)
            score_opt = validate(w, circles_opt)
            if score_opt > best_score:
                best_score = score_opt
                best_result = (w, [list(c) for c in circles_opt])
        
        # Local search: perturb and re-optimize
        if elapsed() < deadline - 20:
            for perturb_iter in range(5):
                if elapsed() > deadline - 15:
                    break
                
                # Random perturbation
                trial = [list(c) for c in circles]
                for _ in range(random.randint(1, 3)):
                    idx = random.randint(0, N-1)
                    trial[idx][0] += random.gauss(0, 0.03)
                    trial[idx][1] += random.gauss(0, 0.03)
                    trial[idx][2] *= random.uniform(0.9, 1.1)
                    trial[idx][2] = max(0.01, min(trial[idx][2], 0.5))
                
                # Relax
                trial = relax_positions(w, trial)
                score_t = validate(w, trial)
                
                if score_t > best_score - 0.01:
                    circles = trial
                    if score_t > best_score:
                        best_score = score_t
                        best_result = (w, [list(c) for c in circles])
                    
                    # Maximize radii
                    if elapsed() < deadline - 12:
                        circles_opt = maximize_radii(w, circles)
                        score_opt = validate(w, circles_opt)
                        if score_opt > best_score:
                            best_score = score_opt
                            best_result = (w, [list(c) for c in circles_opt])
                            circles = circles_opt
    
    # Final polish
    if best_result is not None and elapsed() < deadline - 5:
        w, circles = best_result
        circles_opt = maximize_radii(w, circles)
        score_final = validate(w, circles_opt)
        if score_final > best_score:
            best_score = score_final
            best_result = (w, circles_opt)
    
    # Build output
    if best_result is None:
        # Fallback: grid in square
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cx = (j + 0.5) / side
                cy = (i + 0.5) / side
                r = 0.5 / side * 0.95
                cells.append([cx, cy, r])
        return np.array(cells[:N])
    
    w, circles = best_result
    result = np.array([[c[0], c[1], c[2]] for c in circles])
    return result

# EVOLVE-BLOCK-END
