"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    deadline = time.time() + 115.0  # Safety margin before 150s limit
    best_result = None
    best_score = 0.0

    def validate_and_score(x, y, r, w, h, tol=1e-9):
        """Check if circles are valid and return sum of radii."""
        if np.any(r <= 0):
            return 0.0
        for i in range(N):
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return 0.0
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return 0.0
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                if d < r[i] + r[j] - tol:
                    return 0.0
        return float(np.sum(r))

    def generate_initial_layout(w, h, seed_offset=0):
        """Generate a staggered grid initial layout."""
        rng = np.random.RandomState(42 + seed_offset)
        n_cols = max(1, int(np.round(np.sqrt(N * w / h))))
        n_rows = max(1, int(np.ceil(N / n_cols)))
        while n_cols * n_rows < N:
            if n_cols <= n_rows:
                n_cols += 1
            else:
                n_rows += 1
        xs = []
        ys = []
        for i in range(N):
            col = i % n_cols
            row = i // n_cols
            x = (col + 0.5) * w / n_cols
            y = (row + 0.5) * h / n_rows
            if row % 2 == 1:
                x += 0.25 * w / n_cols
                if x > w - w / n_cols * 0.25:
                    x -= 0.5 * w / n_cols
            xs.append(x)
            ys.append(y)
        xs = np.array(xs)
        ys = np.array(ys)
        # Add small random perturbation
        xs += rng.uniform(-0.02, 0.02, N)
        ys += rng.uniform(-0.02, 0.02, N)
        # Clip to stay in bounds
        xs = np.clip(xs, 0.05, w - 0.05)
        ys = np.clip(ys, 0.05, h - 0.05)
        # Compute initial radius
        min_dist = np.inf
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                min_dist = min(min_dist, d)
        for i in range(N):
            min_dist = min(min_dist, xs[i], w - xs[i], ys[i], h - ys[i])
        r0 = 0.35 * min_dist
        radii = np.full(N, r0)
        return xs, ys, radii

    def soft_penalty_objective(vars_, w, h, penalty_weight):
        """Soft-body penalty objective: -sum(r) + penalty for violations."""
        x = vars_[0::3]
        y = vars_[1::3]
        r = vars_[2::3]
        
        # Base objective: maximize sum of radii
        obj = -np.sum(r)
        
        # Wall violation penalty (quadratic)
        wall_pen = 0.0
        for i in range(N):
            if x[i] < r[i]:
                wall_pen += (r[i] - x[i])**2
            if x[i] > w - r[i]:
                wall_pen += (x[i] - (w - r[i]))**2
            if y[i] < r[i]:
                wall_pen += (r[i] - y[i])**2
            if y[i] > h - r[i]:
                wall_pen += (y[i] - (h - r[i]))**2
        
        # Circle-circle overlap penalty (quadratic)
        overlap_pen = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                min_d = r[i] + r[j]
                if d < min_d:
                    overlap_pen += (min_d - d)**2
        
        # Negative radius penalty
        neg_r_pen = 0.0
        for i in range(N):
            if r[i] < 0:
                neg_r_pen += r[i]**2
        
        return obj + penalty_weight * (wall_pen + overlap_pen + neg_r_pen)

    def soft_penalty_grad(vars_, w, h, penalty_weight):
        """Gradient of the soft penalty objective."""
        x = vars_[0::3].copy()
        y = vars_[1::3].copy()
        r = vars_[2::3].copy()
        
        grad = np.zeros(3 * N)
        
        # Base objective gradient: -1 for each radius
        for i in range(N):
            grad[3*i + 2] = -1.0
        
        # Wall violation gradients
        for i in range(N):
            # x[i] < r[i]
            if x[i] < r[i]:
                viol = r[i] - x[i]
                grad[3*i] += -2 * penalty_weight * viol
                grad[3*i + 2] += 2 * penalty_weight * viol
            # x[i] > w - r[i]
            if x[i] > w - r[i]:
                viol = x[i] - (w - r[i])
                grad[3*i] += 2 * penalty_weight * viol
                grad[3*i + 2] += 2 * penalty_weight * viol
            # y[i] < r[i]
            if y[i] < r[i]:
                viol = r[i] - y[i]
                grad[3*i + 1] += -2 * penalty_weight * viol
                grad[3*i + 2] += 2 * penalty_weight * viol
            # y[i] > h - r[i]
            if y[i] > h - r[i]:
                viol = y[i] - (h - r[i])
                grad[3*i + 1] += 2 * penalty_weight * viol
                grad[3*i + 2] += 2 * penalty_weight * viol
        
        # Circle-circle overlap gradients
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                min_d = r[i] + r[j]
                if d < min_d and d > 1e-12:
                    viol = min_d - d
                    # Gradient wrt x[i]
                    grad[3*i] += 2 * penalty_weight * viol * dx / d
                    grad[3*j] += -2 * penalty_weight * viol * dx / d
                    # Gradient wrt y[i]
                    grad[3*i + 1] += 2 * penalty_weight * viol * dy / d
                    grad[3*j + 1] += -2 * penalty_weight * viol * dy / d
                    # Gradient wrt r[i] and r[j]
                    grad[3*i + 2] += 2 * penalty_weight * viol
                    grad[3*j + 2] += 2 * penalty_weight * viol
        
        # Negative radius gradients
        for i in range(N):
            if r[i] < 0:
                grad[3*i + 2] += 2 * penalty_weight * r[i]
        
        return grad

    # Try different aspect ratios with soft-body optimization
    aspect_ratios = [1.0, 1.2, 1.5, 2.0, 0.8, 0.6, 1.3, 1.7, 2.5, 0.5, 1.1, 1.4, 1.8, 0.7, 2.2]
    
    for ar_idx, ar in enumerate(aspect_ratios):
        if time.time() > deadline - 15:
            break

        w = 2.0 / (1.0 + ar)
        h = 2.0 - w

        # Multiple restarts with different initial conditions
        for restart in range(3):
            if time.time() > deadline - 10:
                break

            xs, ys, radii = generate_initial_layout(w, h, seed_offset=ar_idx * 10 + restart)

            x0 = np.zeros(3 * N)
            for i in range(N):
                x0[3*i] = xs[i]
                x0[3*i+1] = ys[i]
                x0[3*i+2] = radii[i]

            # Simulated annealing style: start with low penalty, increase
            penalty_schedule = [1.0, 10.0, 50.0, 200.0, 500.0]
            
            current_vars = x0.copy()
            for pw in penalty_schedule:
                if time.time() > deadline - 8:
                    break
                
                try:
                    result = minimize(
                        soft_penalty_objective, current_vars,
                        jac=soft_penalty_grad,
                        args=(w, h, pw),
                        method='L-BFGS-B',
                        options={'maxiter': 100, 'ftol': 1e-8}
                    )
                    current_vars = result.x
                except Exception:
                    continue

            # Validate final result
            x = current_vars[0::3]
            y = current_vars[1::3]
            r = current_vars[2::3]
            
            # Clip radii to be positive
            r = np.maximum(r, 0.0)
            
            score = validate_and_score(x, y, r, w, h)
            
            if score > best_score:
                best_score = score
                best_result = np.column_stack([x, y, r])

    # Fallback: simple grid in a square of perimeter 4 (w=h=1)
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best_result
# EVOLVE-BLOCK-END
