"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles."""
import time
import numpy as np
from scipy.optimize import minimize

N = 26
EPS = 1e-9

def construct():
    t_start = time.time()
    DEADLINE = 140.0  # Safety margin for 150s limit

    def max_radius_at(x, y, circles):
        """Calculate the maximum radius a circle at (x,y) can have given existing circles."""
        # Distance to walls
        r_max = min(x, 1.0 - x, y, 1.0 - y)
        if r_max <= 0:
            return 0.0
        
        # Distance to existing circles
        for cx, cy, cr in circles:
            dist = np.hypot(x - cx, y - cy)
            # The new circle must not overlap: dist >= r_new + r_old
            # So r_new <= dist - r_old
            r_candidate = dist - cr
            if r_candidate < r_max:
                r_max = r_candidate
        
        return max(0.0, r_max)

    def local_refine(circles, time_budget):
        """Refine positions and radii using local optimization to squeeze more space."""
        if not circles:
            return circles
            
        # Convert to numpy array for easier manipulation
        # circles is list of (x, y, r)
        # We will optimize x, y for each circle, keeping r as max possible? 
        # Actually, it's better to optimize x, y, r together but with constraints.
        # Given the complexity of constraints, let's try a simpler "push apart" or 
        # gradient ascent on sum of radii with penalty.
        
        # A more robust simple refinement: 
        # For each circle, try to move it slightly to increase its radius or allow neighbors to grow.
        # This is complex. Let's stick to a simpler strategy:
        # Re-calculate radii based on current positions (they might have shrunk if we moved them).
        # Actually, in greedy placement, radii are fixed once placed.
        # To improve, we can try to "shrink" earlier circles slightly to allow later ones to be bigger?
        # Or simply optimize the whole configuration.
        
        # Let's use a simple local search: 
        # 1. Take current positions.
        # 2. For each circle, calculate its current max radius based on others.
        # 3. If the current radius is less than max radius, increase it.
        # 4. Try random small perturbations to positions to see if it allows a larger total sum.
        
        # A better approach for refinement: 
        # Use scipy minimize on the vector of all x,y,r with penalty for overlap.
        # But we need to be careful with time.
        
        # Let's implement a simple "relaxation" step:
        # Iterate: for each circle i, fix others, maximize r_i.
        # Then for each circle i, fix r_i, try to move it to a better spot? No, that breaks the greedy nature.
        
        # Let's do a global local search on positions only, recalculating radii.
        # Objective: maximize sum of radii.
        # Constraint: no overlap, in square.
        
        # We'll define a function that takes positions (N, 2) and returns max possible radii.
        # Then we can use a simple gradient-free optimization or just random restarts on positions.
        
        # Actually, the most effective simple improvement is to allow the *radii* to be optimized 
        # given the positions, and then slightly perturb positions to find a better packing.
        
        # Let's try a specific refinement: 
        # 1. Calculate max radii for current positions.
        # 2. Define a cost function: -sum(radii) + penalty(overlaps) + penalty(walls).
        # 3. Use Nelder-Mead or Powell on the 2N position variables? 
        #    If we fix radii to their max possible for those positions, the function is non-smooth.
        #    Better: Optimize 3N variables (x, y, r) with penalty.
        
        x0 = np.array([v for c in circles for v in c]) # length 3N
        if len(x0) == 0: return circles
        
        def objective(params):
            p = params[:2*N].reshape(N, 2)
            r = params[2*N:]
            
            # Penalty for walls
            wall_pen = 0.0
            for i in range(N):
                if p[i,0] < r[i]: wall_pen += (r[i] - p[i,0])**2
                if p[i,0] > 1-r[i]: wall_pen += (p[i,0] - (1-r[i]))**2
                if p[i,1] < r[i]: wall_pen += (r[i] - p[i,1])**2
                if p[i,1] > 1-r[i]: wall_pen += (p[i,1] - (1-r[i]))**2
            
            # Penalty for overlaps
            overlap_pen = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    d = np.hypot(p[i,0]-p[j,0], p[i,1]-p[j,1])
                    if d < r[i] + r[j]:
                        overlap_pen += (r[i] + r[j] - d)**2
            
            return -np.sum(r) + 100.0 * (wall_pen + overlap_pen)

        # Use Powell method as it doesn't require gradients and is robust for small N
        # But 78 variables might be slow for Powell.
        # Let's limit the refinement time.
        
        if time.time() + 5.0 > t_start + DEADLINE:
            return circles
            
        try:
            res = minimize(objective, x0, method='Powell', 
                           options={'maxiter': 200, 'maxfev': 2000, 'disp': False})
            # Check if result is valid
            p = res.x[:2*N].reshape(N, 2)
            r = res.x[2*N:]
            
            # Validate
            valid = True
            for i in range(N):
                if r[i] < 0: valid = False
                if p[i,0] < r[i] - 1e-6 or p[i,0] > 1-r[i] + 1e-6: valid = False
                if p[i,1] < r[i] - 1e-6 or p[i,1] > 1-r[i] + 1e-6: valid = False
                for j in range(i+1, N):
                    d = np.hypot(p[i,0]-p[j,0], p[i,1]-p[j,1])
                    if d < r[i] + r[j] - 1e-6: valid = False
            
            if valid:
                new_circles = [(p[i,0], p[i,1], r[i]) for i in range(N)]
                if np.sum(r) > np.sum([c[2] for c in circles]):
                    return new_circles
        except Exception:
            pass
            
        return circles

    def greedy_place(n_circles, rng):
        """Place n_circles greedily."""
        circles = []
        for i in range(n_circles):
            if time.time() > t_start + DEADLINE:
                break
                
            best_r = 0.0
            best_pos = None
            
            # Strategy: 
            # For the first few circles, use a dense grid to find the largest possible spot.
            # For later circles, use random sampling + local refinement.
            
            if i < 4:
                # Dense grid search
                # The optimal first circle is often in the center or corner.
                # Let's sample a grid of 50x50.
                grid_size = 50
                for gx in range(grid_size):
                    for gy in range(grid_size):
                        x = (gx + 0.5) / grid_size
                        y = (gy + 0.5) / grid_size
                        r = max_radius_at(x, y, circles)
                        if r > best_r:
                            best_r = r
                            best_pos = (x, y)
            else:
                # Random sampling
                # Sample 1000 points
                samples = 1000
                # Bias towards empty space? Hard to know. Just uniform.
                for _ in range(samples):
                    x = rng.random()
                    y = rng.random()
                    r = max_radius_at(x, y, circles)
                    if r > best_r:
                        best_r = r
                        best_pos = (x, y)
                
                # If we found a good spot, try to refine it locally?
                # max_radius_at gives the exact max for that (x,y).
                # Moving (x,y) slightly might allow a larger r if we move away from constraints.
                # But greedy placement usually places at a "tight" spot.
                
                # Let's do a small local search around best_pos to see if we can get a slightly bigger radius
                # by moving the center.
                if best_pos:
                    bx, by = best_pos
                    for _ in range(20):
                        # Random jitter
                        dx = rng.normal(0, 0.01)
                        dy = rng.normal(0, 0.01)
                        nx = np.clip(bx + dx, 0, 1)
                        ny = np.clip(by + dy, 0, 1)
                        nr = max_radius_at(nx, ny, circles)
                        if nr > best_r:
                            best_r = nr
                            best_pos = (nx, ny)

            if best_pos is None or best_r < 1e-6:
                break
                
            circles.append((best_pos[0], best_pos[1], best_r))

        return circles

    best_score = -1.0
    best_circles = None
    
    # Run multiple restarts with different random seeds
    # We have ~140 seconds.
    # Greedy placement is fast. Refinement is the bottleneck.
    # Let's do 20 restarts.
    
    num_restarts = 20
    
    for restart in range(num_restarts):
        if time.time() > t_start + DEADLINE - 10.0: # Leave 10s for final refinement/return
            break
            
        rng = np.random.default_rng(42 + restart)
        
        # Vary the strategy slightly?
        # e.g., different grid density or random sample count.
        # For now, keep it consistent.
        
        circles = greedy_place(N, rng)
        
        if len(circles) < N:
            continue
            
        # Refine
        # Only refine if we have time
        if time.time() < t_start + DEADLINE - 15.0:
            circles = local_refine(circles, 10.0)
            
        if len(circles) < N:
            continue
            
        score = sum(c[2] for c in circles)
        
        if score > best_score:
            best_score = score
            best_circles = circles
            
        # Early stopping if we find a very good solution?
        # The theoretical max for 26 circles is not known exactly, but roughly:
        # Area of 26 circles = 26 * pi * r^2.
        # If they were squares of side s, 26 s^2 = 1 -> s = 1/sqrt(26) ~ 0.196.
        # Radius ~ 0.098. Sum ~ 2.5.
        # With circles, packing efficiency is higher, but boundaries hurt.
        # Let's just keep searching.

    if best_circles is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return np.array(best_circles)

if __name__ == '__main__':
    out = construct()
    print("sum of radii:", out[:, 2].sum())
    print(out)
