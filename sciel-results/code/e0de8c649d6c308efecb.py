"""Circle packing in perimeter-4 rectangle: Simulated Annealing with Rigid Body Moves."""
import numpy as np
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def get_overlap(x, y, r, w, h):
        """Calculate total overlap penalty (squared) and boundary violations."""
        # Boundary violations
        # We want x >= r, x <= w-r, etc.
        # Penalty is square of violation
        pen_x_left = np.sum(np.maximum(r - x, 0)**2)
        pen_x_right = np.sum(np.maximum(x + r - w, 0)**2)
        pen_y_bottom = np.sum(np.maximum(r - y, 0)**2)
        pen_y_top = np.sum(np.maximum(y + r - h, 0)**2)
        
        # Pairwise overlaps
        # d^2 < (r_i + r_j)^2
        # We use (r_i + r_j)^2 - d^2 if positive
        iu = np.triu_indices(N, k=1)
        dx = x[iu[0]] - x[iu[1]]
        dy = y[iu[0]] - y[iu[1]]
        d2 = dx*dx + dy*dy
        r_sum = r[iu[0]] + r[iu[1]]
        overlap = np.maximum(r_sum**2 - d2, 0)
        
        total_penalty = pen_x_left + pen_x_right + pen_y_bottom + pen_y_top + np.sum(overlap)
        return total_penalty

    def objective(x, y, r, w):
        """Objective function to minimize: -sum(r) + penalty."""
        h = 2.0 - w
        pen = get_overlap(x, y, r, w, h)
        return -np.sum(r) + pen * 10.0  # Weight penalty heavily

    def random_init(w):
        h = 2.0 - w
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        # Start with small circles to avoid massive initial overlap
        for i in range(N):
            r[i] = np.random.uniform(0.02, 0.05)
            x[i] = np.random.uniform(r[i], w - r[i])
            y[i] = np.random.uniform(r[i], h - r[i])
        return x, y, r

    best_score = 0.0
    best_state = None

    # Multiple restarts with different aspect ratios and temperatures
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2, 0.75, 1.25, 0.7]
    
    for w_init in aspect_ratios:
        if time.time() > deadline - 15:
            break
            
        h_init = 2.0 - w_init
        if h_init < 0.3:
            continue

        x, y, r = random_init(w_init)
        w = w_init
        
        # Initial penalty check
        # If initial penalty is too high, shrink radii
        pen = get_overlap(x, y, r, w, h_init)
        if pen > 0.1:
             # Shrink radii to reduce overlap
             scale = 0.5
             r = r * scale
             # Repeat if necessary
             for _ in range(10):
                 pen = get_overlap(x, y, r, w, h_init)
                 if pen < 0.01:
                     break
                 r = r * 0.9

        T = 0.05
        T_min = 0.001
        alpha = 0.995
        max_iter = 5000
        
        current_cost = objective(x, y, r, w)
        best_cost = current_cost
        best_x, best_y, best_r, best_w = x.copy(), y.copy(), r.copy(), w

        for i in range(max_iter):
            if time.time() > deadline - 10:
                break
                
            # Generate move
            move_type = np.random.random()
            new_x, new_y, new_r, new_w = x.copy(), y.copy(), r.copy(), w

            if move_type < 0.4:
                # Move single circle
                idx = np.random.randint(N)
                # Translate
                delta = np.random.uniform(-0.05, 0.05)
                new_x[idx] += delta
                delta = np.random.uniform(-0.05, 0.05)
                new_y[idx] += delta
                # Resize
                new_r[idx] = np.random.uniform(0.01, 0.3)
                
            elif move_type < 0.7:
                # Move pair (Rigid body rotation/translation)
                i1, i2 = np.random.choice(N, 2, replace=False)
                # Center of mass
                cx = (x[i1] + x[i2]) / 2.0
                cy = (y[i1] + y[i2]) / 2.0
                
                # Rotate
                angle = np.random.uniform(-0.1, 0.1)
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                
                dx1 = x[i1] - cx
                dy1 = y[i1] - cy
                dx2 = x[i2] - cx
                dy2 = y[i2] - cy
                
                new_x[i1] = cx + dx1*cos_a - dy1*sin_a
                new_y[i1] = cy + dx1*sin_a + dy1*cos_a
                new_x[i2] = cx + dx2*cos_a - dy2*sin_a
                new_y[i2] = cy + dx2*sin_a + dy2*cos_a
                
                # Translate pair
                delta_x = np.random.uniform(-0.02, 0.02)
                delta_y = np.random.uniform(-0.02, 0.02)
                new_x[i1] += delta_x
                new_y[i1] += delta_y
                new_x[i2] += delta_x
                new_y[i2] += delta_y
                
                # Resize pair
                scale = np.random.uniform(0.95, 1.05)
                new_r[i1] = np.clip(new_r[i1] * scale, 0.01, 0.3)
                new_r[i2] = np.clip(new_r[i2] * scale, 0.01, 0.3)

            elif move_type < 0.9:
                # Global scaling of radii
                scale = np.random.uniform(0.98, 1.02)
                new_r = r * scale
                new_r = np.clip(new_r, 0.01, 0.5)
                
            else:
                # Adjust aspect ratio
                new_w = w + np.random.uniform(-0.05, 0.05)
                new_w = np.clip(new_w, 0.3, 1.7)
                # Scale positions to fit new aspect ratio roughly
                if new_w != w:
                    h_old = 2.0 - w
                    h_new = 2.0 - new_w
                    new_x = x * (new_w / w)
                    new_y = y * (h_new / h_old)
                    # Don't scale radii here, let the optimizer handle it, 
                    # but this might cause massive overlaps. 
                    # Better to keep radii same and let penalty guide.
                    pass

            new_cost = objective(new_x, new_y, new_r, new_w)
            
            # Acceptance probability
            delta_cost = new_cost - current_cost
            if delta_cost < 0:
                current_cost = new_cost
                x, y, r, w = new_x, new_y, new_r, new_w
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_x, best_y, best_r, best_w = new_x.copy(), new_y.copy(), new_r.copy(), new_w
            else:
                if np.random.random() < np.exp(-delta_cost / T):
                    current_cost = new_cost
                    x, y, r, w = new_x, new_y, new_r, new_w

            T *= alpha
            if T < T_min:
                T = T_min

        # After SA, try to expand radii (Polish)
        # We have a valid (or near-valid) configuration.
        # We can try to increase all radii by a factor until invalid.
        
        # First, ensure validity by shrinking if necessary
        pen = get_overlap(best_x, best_y, best_r, best_w, 2.0 - best_w)
        if pen > 1e-6:
            # Binary search for scale factor that makes it valid
            lo, hi = 0.5, 1.0
            for _ in range(20):
                mid = (lo + hi) / 2.0
                r_test = best_r * mid
                pen_test = get_overlap(best_x, best_y, r_test, best_w, 2.0 - best_w)
                if pen_test < 1e-8:
                    lo = mid
                else:
                    hi = mid
            best_r = best_r * lo
            
        # Now try to expand
        # We want to maximize sum(r). 
        # Let's try scaling up all radii by a factor k.
        # If valid, keep. If not, binary search max k.
        # But we can also try to expand individual circles.
        
        # Simple expansion polish:
        # Iterate through circles, try to increase radius.
        for _ in range(5): # Passes
            improved = False
            for i in range(N):
                # Try to increase radius of i
                # Max radius is limited by distance to other circles and walls
                # Let's just try increasing by 5% and check validity
                r_test = best_r.copy()
                r_test[i] *= 1.05
                if get_overlap(best_x, best_y, r_test, best_w, 2.0 - best_w) < 1e-8:
                    best_r = r_test
                    improved = True
            if not improved:
                break
        
        # Calculate score
        h_final = 2.0 - best_w
        if get_overlap(best_x, best_y, best_r, best_w, h_final) < 1e-8:
            score = np.sum(best_r)
            if score > best_score:
                best_score = score
                # Store state
                state = np.zeros(3*N + 1)
                state[0::3] = best_x
                state[1::3] = best_y
                state[2::3] = best_r
                state[-1] = best_w
                best_state = state.copy()

    if best_state is None:
        # Fallback: Grid packing
        w = 1.0
        h = 1.0
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        side = 5
        for i in range(N):
            row, col = divmod(i, side)
            x[i] = (col + 0.5) / side
            y[i] = (row + 0.5) / side
            r[i] = 0.09
        state = np.zeros(3*N + 1)
        state[0::3] = x
        state[1::3] = y
        state[2::3] = r
        state[-1] = w
        return state

    return best_state

# Helper for fallback if needed, though best_state should always be set
def _get_fallback():
    w = 1.0
    h = 1.0
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    side = 5
    for i in range(N):
        row, col = divmod(i, side)
        x[i] = (col + 0.5) / side
        y[i] = (row + 0.5) / side
        r[i] = 0.09
    state = np.zeros(3*N + 1)
    state[0::3] = x
    state[1::3] = y
    state[2::3] = r
    state[-1] = w
    return state
