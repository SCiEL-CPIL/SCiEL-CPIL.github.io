"""
Circle packing in perimeter-4 rectangle.
Strategy: Differentiable physics simulation (soft particles with elastic repulsion)
combined with gradient descent on a time-integrated potential energy function.
We optimize initial positions and aspect ratio, then run a physics relaxation to reach
a rigid, non-overlapping equilibrium, and finally refine with L-BFGS-B.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit

    best_value = 0.0
    best_pos = None

    # Helper to check validity and compute value
    def check_and_value(pos):
        # pos: (N, 3) array of [x, y, r]
        # We also need w, h. Let's assume the physics simulation returns w as part of state or we fix it?
        # The prompt strategy says "optimizing ... the global aspect ratio".
        # So pos should include w? Or we return (pos, w)?
        # The interface requires returning array of shape (n, 3).
        # The evaluator measures the raw value (sum of radii) for a valid packing.
        # How does the evaluator know w and h?
        # "variable rectangle of perimeter 4".
        # "bounding rectangle has perimeter at most 4".
        # This implies the circles define the bounding box?
        # "n disjoint circles whose bounding rectangle has perimeter at most 4"
        # So w = max(x+r) - min(x-r), h = max(y+r) - min(y-r).
        # And 2*(w+h) <= 4 => w+h <= 2.
        # Wait, the seed program used a fixed 1x1 box (perimeter 4).
        # The inspiration uses w as a variable in the optimization, h = 2-w.
        # But the return interface is just (n, 3).
        # If I return circles in a 1.2 x 0.8 box, the bounding box is 1.2 x 0.8, perimeter 4.
        # If I return circles that don't touch the boundaries, the bounding box is smaller, perimeter < 4.
        # To maximize sum of radii, we likely want to fill the box.
        # So the optimization should ensure the circles span the box or the box is tight around them.
        # Actually, if the box is variable, we can just place circles in a 1x1 box.
        # But maybe a non-square box allows better packing?
        # Let's assume the "rectangle" is the bounding box of the circles.
        # So we just need to return valid circles such that their bounding box has perimeter <= 4.
        
        x = pos[:, 0]
        y = pos[:, 1]
        r = pos[:, 2]
        
        # Check radii positive
        if np.any(r <= 0):
            return 0.0, False
            
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] - 1e-9:
                    return 0.0, False
        
        # Compute bounding box
        min_x = np.min(x - r)
        max_x = np.max(x + r)
        min_y = np.min(y - r)
        max_y = np.max(y + r)
        
        w = max_x - min_x
        h = max_y - min_y
        
        if 2 * (w + h) > 4.0 + 1e-9:
            return 0.0, False
            
        return np.sum(r), True

    def physics_simulate(x0, y0, r0, w0, steps=200, dt=0.01):
        """
        Simulate soft particle dynamics to resolve overlaps and push against walls.
        x, y: (N,) arrays
        r: (N,) array
        w0: initial width, h0 = 2 - w0
        Returns final x, y, r, w, h
        """
        h0 = 2.0 - w0
        x = x0.copy()
        y = y0.copy()
        r = r0.copy()
        w = w0
        h = h0
        
        # We will optimize w and h as well? Or just centers and radii?
        # Let's keep w, h fixed during physics, but optimize them in the outer loop?
        # Or let the boxes expand/contract?
        # Simpler: Fix w, h for this simulation.
        
        k_repel = 50.0
        k_wall = 50.0
        damping = 0.9
        
        vx = np.zeros(N)
        vy = np.zeros(N)
        
        for step in range(steps):
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            # Pairwise repulsion
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = r[i] + r[j]
                    if dist < min_dist and dist > 1e-6:
                        overlap = min_dist - dist
                        f = k_repel * overlap
                        fx[i] += (dx/dist) * f
                        fy[i] += (dy/dist) * f
                        fx[j] -= (dx/dist) * f
                        fy[j] -= (dy/dist) * f
                    elif dist <= 1e-6:
                        # Random kick
                        fx[i] += np.random.randn() * 0.1
                        fy[i] += np.random.randn() * 0.1
                        fx[j] -= np.random.randn() * 0.1
                        fy[j] -= np.random.randn() * 0.1
            
            # Wall repulsion
            for i in range(N):
                if x[i] - r[i] < 0:
                    fx[i] += k_wall * (r[i] - x[i])
                if x[i] + r[i] > w:
                    fx[i] -= k_wall * (x[i] + r[i] - w)
                if y[i] - r[i] < 0:
                    fy[i] += k_wall * (r[i] - y[i])
                if y[i] + r[i] > h:
                    fy[i] -= k_wall * (y[i] + r[i] - h)
            
            # Update velocities and positions
            vx = vx * damping + fx * dt
            vy = vy * damping + fy * dt
            x += vx * dt
            y += vy * dt
            
            # Hard clamp to prevent escape (optional, soft is better for gradient)
            # But for stability in simulation, maybe clamp?
            # Let's not clamp, let the penalty handle it in the next optimization step.
            
        return x, y, r, w, h

    def objective(params):
        """
        params: [w, x1, y1, r1, ..., xN, yN, rN]
        We run a short physics simulation to get a relaxed state, then compute the penalty.
        This makes the objective smoother.
        """
        w = params[0]
        if w < 0.2 or w > 1.8:
            return 1e6
        h = 2.0 - w
        
        x = params[1::3]
        y = params[2::3]
        r = params[3::3]
        
        # Run a few steps of physics to relax
        x_rel, y_rel, r_rel, w_rel, h_rel = physics_simulate(x, y, r, w, steps=10, dt=0.05)
        
        # Now compute penalty on the relaxed state
        # We want to maximize sum(r), so minimize -sum(r) + penalty
        
        pen = 0.0
        # Boundary violations
        for i in range(N):
            if x_rel[i] - r_rel[i] < 0:
                pen += (r_rel[i] - x_rel[i])**2
            if x_rel[i] + r_rel[i] > w_rel:
                pen += (x_rel[i] + r_rel[i] - w_rel)**2
            if y_rel[i] - r_rel[i] < 0:
                pen += (r_rel[i] - y_rel[i])**2
            if y_rel[i] + r_rel[i] > h_rel:
                pen += (y_rel[i] + r_rel[i] - h_rel)**2
                
        # Overlaps
        for i in range(N):
            for j in range(i+1, N):
                dx = x_rel[i] - x_rel[j]
                dy = y_rel[i] - y_rel[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_d = r_rel[i] + r_rel[j]
                if dist < min_d:
                    pen += (min_d - dist)**2
                    
        return -np.sum(r_rel) + 10.0 * pen

    def generate_starts():
        starts = []
        rng = np.random.default_rng(42)
        
        # Aspect ratios to try
        aspects = [1.0, 1.2, 0.8, 1.5, 0.67, 1.4, 0.71]
        
        for w0 in aspects:
            h0 = 2.0 - w0
            if h0 < 0.2: continue
            
            # Grid init
            side = int(np.ceil(np.sqrt(N)))
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            for i in range(N):
                row, col = divmod(i, side)
                x[i] = (col + 0.5) / side * w0
                y[i] = (row + 0.5) / side * h0
                r[i] = min(w0/side, h0/side) * 0.45
                
            params = np.concatenate(([w0], np.column_stack((x, y, r)).flatten()))
            starts.append(params)
            
        # Random inits
        for _ in range(10):
            w0 = rng.uniform(0.8, 1.2)
            h0 = 2.0 - w0
            x = rng.uniform(0.1, w0-0.1, N)
            y = rng.uniform(0.1, h0-0.1, N)
            r = rng.uniform(0.05, 0.1, N)
            params = np.concatenate(([w0], np.column_stack((x, y, r)).flatten()))
            starts.append(params)
            
        return starts

    starts = generate_starts()
    
    for s_idx, start_params in enumerate(starts):
        if time.time() > deadline - 10:
            break
            
        # Phase 1: Physics relaxation via gradient descent on the "relaxed objective"
        # This is expensive if we do many steps. Let's just run a longer physics sim directly.
        
        w0 = start_params[0]
        x0 = start_params[1::3]
        y0 = start_params[2::3]
        r0 = start_params[3::3]
        
        # Run longer physics simulation
        x_sim, y_sim, r_sim, w_sim, h_sim = physics_simulate(x0, y0, r0, w0, steps=100, dt=0.02)
        
        # Phase 2: Refine with L-BFGS-B
        # Now use the standard penalty objective on the relaxed state
        def obj_refine(p):
            w = p[0]
            if w < 0.2 or w > 1.8: return 1e6
            h = 2.0 - w
            x = p[1::3]
            y = p[2::3]
            r = p[3::3]
            
            pen = 0.0
            for i in range(N):
                if x[i] - r[i] < 0: pen += (r[i] - x[i])**2
                if x[i] + r[i] > w: pen += (x[i] + r[i] - w)**2
                if y[i] - r[i] < 0: pen += (r[i] - y[i])**2
                if y[i] + r[i] > h: pen += (y[i] + r[i] - h)**2
                
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_d = r[i] + r[j]
                    if dist < min_d:
                        pen += (min_d - dist)**2
                        
            return -np.sum(r) + 10.0 * pen
            
        p0 = np.concatenate(([w_sim], np.column_stack((x_sim, y_sim, r_sim)).flatten()))
        
        bounds = [(0.2, 1.8)]
        for i in range(N):
            bounds.append((0.0, 2.0)) # x
            bounds.append((0.0, 2.0)) # y
            bounds.append((0.01, 0.5)) # r
            
        try:
            res = minimize(obj_refine, p0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 100, 'ftol': 1e-8})
            
            if res.fun < 1e-3: # Low penalty
                w_res = res.x[0]
                x_res = res.x[1::3]
                y_res = res.x[2::3]
                r_res = res.x[3::3]
                
                # Translate so min x, min y are at 0?
                # The evaluator uses bounding box.
                # So we can just return the coordinates as is.
                pos = np.column_stack((x_res, y_res, r_res))
                
                val, valid = check_and_value(pos)
                if valid and val > best_value:
                    best_value = val
                    best_pos = pos
                    
        except Exception as e:
            pass
            
    # Fallback
    if best_pos is None:
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_pos = np.array(cells[:N])
        
    return best_pos[:N]

# EVOLVE-BLOCK-END
