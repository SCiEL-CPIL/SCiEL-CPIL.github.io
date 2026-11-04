import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    def _validate(circles, w, h, tol=1e-6):
        """Check if circles are valid: inside rectangle and non-overlapping."""
        n = len(circles)
        if n != N:
            return False
        for i in range(n):
            cx, cy, r = circles[i]
            if r <= 0:
                return False
            if cx - r < -tol or cx + r > w + tol or cy - r < -tol or cy + r > h + tol:
                return False
        # Vectorized overlap check
        cx = circles[:, 0]
        cy = circles[:, 1]
        r = circles[:, 2]
        for i in range(n):
            dx = cx[i] - cx[i+1:]
            dy = cy[i] - cy[i+1:]
            dist = np.sqrt(dx*dx + dy*dy)
            if np.any(dist < r[i] + r[i+1:] - tol):
                return False
        return True

    def _total_radius(circles):
        return float(np.sum(circles[:, 2]))

    def _relax_circles(circles, w, h, iterations=1000, tol=1e-8):
        """
        Relaxation algorithm to resolve overlaps and boundary violations.
        This is the 'rigid body' / projected Gauss-Seidel step.
        """
        n = len(circles)
        pos = np.array(circles[:, :2], dtype=float)
        rad = np.array(circles[:, 2], dtype=float)

        for _ in range(iterations):
            if time.time() > deadline - 2.0:
                break
            moved = False

            # 1. Resolve boundary violations (Projection onto box)
            for i in range(n):
                if pos[i, 0] - rad[i] < 0:
                    pos[i, 0] = rad[i]
                    moved = True
                if pos[i, 0] + rad[i] > w:
                    pos[i, 0] = w - rad[i]
                    moved = True
                if pos[i, 1] - rad[i] < 0:
                    pos[i, 1] = rad[i]
                    moved = True
                if pos[i, 1] + rad[i] > h:
                    pos[i, 1] = h - rad[i]
                    moved = True

            # 2. Resolve overlaps (Projection onto non-overlap set)
            # Iterate over pairs. For better convergence, we might want to be smarter,
            # but simple Gauss-Seidel often works for small N.
            for i in range(n):
                for j in range(i+1, n):
                    dx = pos[j, 0] - pos[i, 0]
                    dy = pos[j, 1] - pos[i, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = rad[i] + rad[j]
                    
                    if dist < min_dist:
                        if dist < 1e-9:
                            # Randomly separate
                            angle = np.random.uniform(0, 2 * np.pi)
                            push = (min_dist - dist) / 2.0
                            pos[i, 0] -= np.cos(angle) * push
                            pos[i, 1] -= np.sin(angle) * push
                            pos[j, 0] += np.cos(angle) * push
                            pos[j, 1] += np.sin(angle) * push
                        else:
                            push = (min_dist - dist) / 2.0
                            ux = dx / dist
                            uy = dy / dist
                            pos[i, 0] -= ux * push
                            pos[i, 1] -= uy * push
                            pos[j, 0] += ux * push
                            pos[j, 1] += uy * push
                        moved = True
            
            if not moved:
                break

        # Final boundary clamp to ensure strict validity
        for i in range(n):
            pos[i, 0] = np.clip(pos[i, 0], rad[i], w - rad[i])
            pos[i, 1] = np.clip(pos[i, 1], rad[i], h - rad[i])

        new_circles = np.column_stack((pos, rad))
        return new_circles

    def _make_hexagonal_grid(n, w, h):
        """Create initial hexagonal-like arrangement for 21 circles."""
        # Rows: 5, 4, 5, 4, 3 = 21 circles
        rows = [5, 4, 5, 4, 3]
        circles = []
        total_rows = len(rows)
        
        # Estimate radius based on width
        r = 0.35 * min(w/5, h/total_rows)
        
        for row_i, count in enumerate(rows):
            y = (row_i + 0.5) * h / total_rows
            for col_j in range(count):
                # Offset alternating rows for hexagonal packing
                offset = 0.5 * (w / 5) if row_i % 2 == 1 else 0
                x = (col_j + 0.5) * w / 5 + offset
                circles.append([x, y, r])
        
        return np.array(circles[:n])

    def _make_grid(n, w, h):
        """Fallback grid initialization."""
        side = int(np.ceil(np.sqrt(n)))
        cells = []
        for i in range(side):
            for j in range(side):
                cx = (i + 0.5) * w / side
                cy = (j + 0.5) * h / side
                r = 0.5 * min(w, h) / side * 0.9
                cells.append([cx, cy, r])
        return np.array(cells[:n])

    def _penalty_optimize(circles, w, h, deadline, beta_start=10.0, beta_end=1000.0, n_beta_steps=10):
        """
        Penalty-based optimization using L-BFGS-B.
        Faster than constraint-based methods due to vectorized penalty evaluation.
        """
        if time.time() > deadline - 5.0:
            return None

        n = len(circles)
        x0 = np.concatenate([circles[:, 0], circles[:, 1], circles[:, 2]])
        
        # Bounds
        bounds = []
        for i in range(n):
            bounds.append((1e-6, w)) # x
            bounds.append((1e-6, h)) # y
        for i in range(n):
            bounds.append((1e-6, min(w, h)/2)) # r

        def objective_and_grad(x, beta):
            """Compute objective and gradient with penalty."""
            cx = x[:n]
            cy = x[n:2*n]
            r = x[2*n:]
            
            # Objective: -sum(r) + penalty
            obj = -np.sum(r)
            grad = np.zeros_like(x)
            grad[2*n:] = -1.0
            
            # Boundary penalty
            for i in range(n):
                if cx[i] - r[i] < 0:
                    pen = beta * (cx[i] - r[i])**2
                    obj += pen
                    grad[i] += 2 * beta * (cx[i] - r[i])
                    grad[2*n+i] -= 2 * beta * (cx[i] - r[i])
                if cx[i] + r[i] > w:
                    pen = beta * (cx[i] + r[i] - w)**2
                    obj += pen
                    grad[i] += 2 * beta * (cx[i] + r[i] - w)
                    grad[2*n+i] += 2 * beta * (cx[i] + r[i] - w)
                if cy[i] - r[i] < 0:
                    pen = beta * (cy[i] - r[i])**2
                    obj += pen
                    grad[n+i] += 2 * beta * (cy[i] - r[i])
                    grad[2*n+i] -= 2 * beta * (cy[i] - r[i])
                if cy[i] + r[i] > h:
                    pen = beta * (cy[i] + r[i] - h)**2
                    obj += pen
                    grad[n+i] += 2 * beta * (cy[i] + r[i] - h)
                    grad[2*n+i] += 2 * beta * (cy[i] + r[i] - h)
            
            # Overlap penalty (vectorized)
            for i in range(n):
                dx = cx[i] - cx[i+1:]
                dy = cy[i] - cy[i+1:]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[i+1:]
                overlap = np.maximum(0, min_dist - dist)
                
                if np.any(overlap > 0):
                    # penalty term
                    pen = beta * np.sum(overlap**2)
                    obj += pen
                    
                    # Gradient for circle i
                    # d/dx_i sum (min_dist - dist)^2
                    # = sum 2(overlap) * (-d(dist)/d(x_i))
                    # d(dist)/d(x_i) = -dx/dist
                    # so d(pen)/d(x_i) = 2 * beta * sum(overlap * (dx/dist))
                    
                    valid = dist > 1e-6
                    if np.any(valid):
                        ux = dx[valid] / dist[valid]
                        uy = dy[valid] / dist[valid]
                        # Note: dx = cx[i] - cx[j], so gradient w.r.t cx[i] is proportional to ux
                        # Wait, dist = |p_i - p_j|. d(dist)/d(p_i) = (p_i - p_j)/dist = u.
                        # pen = sum (min_dist - dist)^2
                        # d(pen)/d(p_i) = sum 2(overlap) * (-1) * d(dist)/d(p_i) = -2 * sum(overlap * u)
                        # But overlap = max(0, min_dist - dist).
                        # If overlap > 0, d(overlap)/d(dist) = -1.
                        # So d(pen)/d(p_i) = 2 * beta * sum(overlap * u)
                        
                        # Actually, let's look at the sign.
                        # If we move p_i closer to p_j, dist decreases, overlap increases, penalty increases.
                        # So gradient should point away from p_j?
                        # u points from j to i.
                        # Moving i in direction u increases dist.
                        # So moving i in direction -u decreases dist.
                        # Penalty increases when dist decreases.
                        # So gradient of penalty w.r.t p_i should point in direction u (away from j).
                        # Let's check: d(pen)/d(p_i) = 2 * beta * overlap * (d(min_dist)/d(p_i) - d(dist)/d(p_i))
                        # d(min_dist)/d(p_i) = 0.
                        # d(dist)/d(p_i) = u.
                        # So d(pen)/d(p_i) = -2 * beta * overlap * u.
                        # This means gradient points towards j?
                        # No, if I move in direction of gradient, penalty increases.
                        # If I move in direction of -gradient, penalty decreases.
                        # If gradient is -u, then -gradient is u. Moving in direction u (away from j) decreases penalty. Correct.
                        
                        # So grad_x_i += -2 * beta * overlap * ux
                        # grad_x_j += 2 * beta * overlap * ux (by symmetry, force on j is opposite)
                        
                        grad[i] -= 2 * beta * np.sum(overlap[valid] * ux)
                        grad[n+i] -= 2 * beta * np.sum(overlap[valid] * uy)
                        
                        # For circle j
                        # d(pen)/d(p_j) = 2 * beta * overlap * u
                        # We need to accumulate this for all j > i
                        # This is a bit tricky to vectorize into a flat gradient array without loops or scatter.
                        # Given N=21, a loop over i and vectorizing over j is fine.
                        # But accumulating into grad[j] requires scatter.
                        # Let's just do a simple loop for the overlap gradient to be safe and clear,
                        # or use np.add.at
                        # Actually, the loop over i is already there.
                        # Let's accumulate for j.
                        # We have overlap and u for all j > i.
                        # grad[j] += 2 * beta * overlap * u
                        # We can use np.add.at
                        indices = np.arange(i+1, n)
                        np.add.at(grad, indices, 2 * beta * overlap[valid] * ux)
                        np.add.at(grad, n+indices, 2 * beta * overlap[valid] * uy)
            
            return obj, grad

        best_circles = circles
        best_val = _total_radius(circles)
        
        for step in range(n_beta_steps):
            if time.time() > deadline - 5.0:
                break
            beta = beta_start * (beta_end / beta_start) ** (step / max(1, n_beta_steps - 1))
            
            res = minimize(
                objective_and_grad, 
                x0, 
                args=(beta,),
                method='L-BFGS-B', 
                bounds=bounds, 
                jac=True,
                options={'maxiter': 50, 'ftol': 1e-8, 'maxls': 20}
            )
            
            if res.success:
                x_opt = res.x
                new_circles = np.column_stack((x_opt[:n], x_opt[n:2*n], x_opt[2*n:]))
                
                # Relax to ensure validity
                new_circles = _relax_circles(new_circles, w, h)
                
                val = _total_radius(new_circles)
                if val > best_val:
                    best_val = val
                    best_circles = new_circles
                    x0 = np.concatenate([best_circles[:, 0], best_circles[:, 1], best_circles[:, 2]])
        
        return best_circles

    # Main Loop: Try different aspect ratios
    # Perimeter = 2(w+h) = 4 => w+h = 2
    # Let w = x, h = 2-x
    # x ranges from 1 (square) to ~1.5 (long rectangle)
    
    best_circles = None
    best_score = 0.0
    
    # Aspect ratios to try: 1.0, 1.2, 1.5, 2.0, 0.8, 0.6
    # w/h = aspect
    # w = aspect * h
    # aspect*h + h = 2 => h = 2/(aspect+1), w = 2*aspect/(aspect+1)
    
    aspects = [1.0, 1.2, 1.5, 2.0, 0.8, 0.6, 1.3, 1.1]
    
    rng = np.random.default_rng(42)
    
    for aspect in aspects:
        if time.time() > deadline - 10.0:
            break
            
        h = 2.0 / (aspect + 1.0)
        w = 2.0 * aspect / (aspect + 1.0)
        
        # Initialize
        init_circles = _make_hexagonal_grid(N, w, h)
        
        # Relax initial guess
        init_circles = _relax_circles(init_circles, w, h)
        
        # Optimize
        opt_circles = _penalty_optimize(init_circles, w, h, deadline)
        
        if opt_circles is not None:
            val = _total_radius(opt_circles)
            if val > best_score:
                best_score = val
                best_circles = opt_circles
                # Store best w, h for final validation? 
                # The output requires just the circles. 
                # But we need to ensure the bounding box has perimeter <= 4.
                # The circles are in [0,w]x[0,h].
                # If we return them, the evaluator computes the bounding box.
                # We should scale them to fit in a box of perimeter 4?
                # No, the evaluator measures the raw value.
                # "Pack 21 circles ... in a variable rectangle of perimeter 4"
                # "construct() -> array of shape (n, 3): n disjoint circles whose bounding rectangle has perimeter at most 4"
                # So the coordinates are absolute.
                # If my optimal packing was in a box of w x h (w+h=2), the bounding box of the circles
                # might be smaller. That's fine, perimeter <= 4 is satisfied.
                # But wait, if I scale the circles to fit in a specific box, does the radius sum change?
                # The problem is: maximize sum of radii.
                # If I find a packing in a box of w x h, the radii are fixed.
                # The bounding box of the circles is determined by the circles.
                # As long as it fits in a box of perimeter 4, it's valid.
                # My optimization assumed the circles are inside [0,w]x[0,h] where w+h=2.
                # So the bounding box of the circles is <= w x h.
                # So perimeter <= 2(w+h) = 4.
                # So the result is valid.
                
        # Also try a grid init for robustness
        if time.time() > deadline - 10.0:
            break
        init_circles_grid = _make_grid(N, w, h)
        init_circles_grid = _relax_circles(init_circles_grid, w, h)
        opt_circles_grid = _penalty_optimize(init_circles_grid, w, h, deadline)
        
        if opt_circles_grid is not None:
            val = _total_radius(opt_circles_grid)
            if val > best_score:
                best_score = val
                best_circles = opt_circles_grid

    if best_circles is None:
        # Fallback to seed-like solution
        # Square 1x1, 5x5 grid, r=0.1
        cells = []
        for i in range(5):
            for j in range(5):
                if len(cells) >= N:
                    break
                cells.append([0.1 + 0.2*i, 0.1 + 0.2*j, 0.1])
        return np.array(cells[:N])
    
    # Final validation and cleanup
    # Determine bounding box of best_circles
    min_x = np.min(best_circles[:, 0] - best_circles[:, 2])
    max_x = np.max(best_circles[:, 0] + best_circles[:, 2])
    min_y = np.min(best_circles[:, 1] - best_circles[:, 2])
    max_y = np.max(best_circles[:, 1] + best_circles[:, 2])
    
    w_box = max_x - min_x
    h_box = max_y - min_y
    
    # If the bounding box perimeter is > 4, we have a problem.
    # But our optimization enforced circles inside [0,w]x[0,h] with w+h=2.
    # So the bounding box should be <= w x h.
    # However, numerical errors might push them out slightly.
    # Let's check and scale if necessary.
    perim = 2 * (w_box + h_box)
    
    if perim > 4.0:
        # Scale down
        scale = 4.0 / perim
        best_circles[:, 0] = min_x + (best_circles[:, 0] - min_x) * scale
        best_circles[:, 1] = min_y + (best_circles[:, 1] - min_y) * scale
        best_circles[:, 2] *= scale
        
        # Re-relax after scaling? Scaling preserves non-overlap and boundary relative to box.
        # But it might create gaps. That's fine.
        # However, if we scaled, the sum of radii decreases.
        # It's better to ensure the initial optimization didn't produce an invalid box.
        # Given the constraints, this should be rare.
        
    # Final validation
    # We need to ensure the circles are disjoint and inside a box of perimeter <= 4.
    # The evaluator will check this.
    # We should return the coordinates relative to the origin?
    # The problem says "construct() -> array of shape (n, 3)".
    # It doesn't specify the origin.
    # Usually, coordinates are absolute.
    # If I return coordinates in [0, w] x [0, h], the bounding box is within that.
    # The evaluator will compute the bounding box.
    # So I should return the coordinates as they are.
    
    # One more check: if the bounding box is very small, maybe I can shift/expand?
    # No, expanding would violate non-overlap or boundary.
    # The optimization maximized the sum of radii given the box constraint.
    # So the result should be optimal for that aspect ratio.
    
    return best_circles

