import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    # New Approach: Simulated Annealing with Variable Rectangle Aspect Ratio
    # State: [w, x1, y1, r1, x2, y2, r2, ...]
    # h is derived from w: h = 2 - w
    # Maximize sum(r) subject to constraints.
    
    def compute_energy(state, w_penalty=10.0, overlap_penalty=50.0):
        w = state[0]
        h = 2.0 - w
        if w <= 0 or h <= 0:
            return 1e10
        
        x = state[1::3]
        y = state[2::3]
        r = state[3::3]
        
        # Objective: Minimize -sum(r) + penalties
        obj = -np.sum(r)
        
        # Boundary Penalties
        # x must be in [r, w-r], y must be in [r, h-r]
        # If r is too large to fit, penalty is huge.
        bound_pen = 0.0
        for i in range(N):
            if r[i] > w / 2.0 or r[i] > h / 2.0:
                bound_pen += (r[i] - min(w, h)/2.0)**2 * w_penalty
            else:
                if x[i] < r[i]:
                    bound_pen += (r[i] - x[i])**2 * w_penalty
                elif x[i] > w - r[i]:
                    bound_pen += (x[i] - (w - r[i]))**2 * w_penalty
                if y[i] < r[i]:
                    bound_pen += (r[i] - y[i])**2 * w_penalty
                elif y[i] > h - r[i]:
                    bound_pen += (y[i] - (h - r[i]))**2 * w_penalty
        
        # Overlap Penalties
        overlap_pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    overlap_pen += (min_dist - dist)**2 * overlap_penalty
        
        return obj + bound_pen + overlap_pen

    def is_valid(state, tol=1e-5):
        w = state[0]
        h = 2.0 - w
        if w <= 0 or h <= 0:
            return False
        
        x = state[1::3]
        y = state[2::3]
        r = state[3::3]
        
        if np.any(r <= 0):
            return False
            
        for i in range(N):
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return False
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return False
            if r[i] > w/2 + tol or r[i] > h/2 + tol:
                return False
                
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] - tol:
                    return False
        return True

    def refine(state):
        # Use L-BFGS-B to polish the solution from SA
        # We fix w, h in the local optimization for stability, or optimize them too?
        # Let's optimize w, x, y, r together.
        
        def obj_func(vars):
            return compute_energy(vars)
            
        # Bounds
        # w in [0.1, 1.9] (since h=2-w, keep aspect ratio reasonable)
        bounds = [(0.1, 1.9)]
        for i in range(N):
            # x, y bounds depend on w, h, so we use loose bounds [0, 2]
            # The penalty function handles the strict constraints.
            # However, L-BFGS-B needs static bounds.
            # We will rely on the penalty function for strictness during refinement.
            # Initial radii are small, so x,y in [0, 2] is safe.
            bounds.append((0.0, 2.0)) # x
            bounds.append((0.0, 2.0)) # y
            bounds.append((0.0, 1.0)) # r (max radius in 1x1 is 0.5, in 0.1x1.9 is ~0.05)
            
        res = minimize(obj_func, state, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if is_valid(res.x):
            return res.x
        return state

    best_state = None
    best_score = 0.0
    
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    # Run multiple SA trials with different random seeds and initial aspect ratios
    trial = 0
    while time.time() - start_time < time_limit:
        trial += 1
        np.random.seed(42 + trial)
        
        # Initial guess
        # Try different aspect ratios
        w_init = np.random.uniform(0.5, 1.5)
        h_init = 2.0 - w_init
        
        state = np.zeros(1 + 3*N)
        state[0] = w_init
        
        # Initialize circles in a grid-like fashion within the rectangle
        side = int(np.ceil(np.sqrt(N)))
        for i in range(N):
            col = i % side
            row = i // side
            # Spread in w x h
            xi = (col + 0.5) * (w_init / side)
            yi = (row + 0.5) * (h_init / side)
            # Small initial radius
            ri = 0.05
            
            state[1 + 3*i] = xi
            state[2 + 3*i] = yi
            state[3 + 3*i] = ri
            
        current_state = state.copy()
        current_energy = compute_energy(current_state)
        
        # Simulated Annealing Parameters
        T = 1.0
        T_min = 1e-4
        alpha = 0.995 # Cooling rate
        max_iter = 5000
        
        for k in range(max_iter):
            if time.time() - start_time > time_limit:
                break
                
            # Generate neighbor
            new_state = current_state.copy()
            
            # Perturb w
            if np.random.random() < 0.1:
                new_state[0] += np.random.normal(0, 0.05)
                new_state[0] = np.clip(new_state[0], 0.1, 1.9)
            
            # Perturb positions and radii
            for i in range(N):
                # Move center
                if np.random.random() < 0.5:
                    new_state[1 + 3*i] += np.random.normal(0, 0.05 * T)
                    new_state[2 + 3*i] += np.random.normal(0, 0.05 * T)
                
                # Move radius
                if np.random.random() < 0.3:
                    new_state[3 + 3*i] += np.random.normal(0, 0.01 * T)
                    # Keep radius positive
                    if new_state[3 + 3*i] < 0.01:
                        new_state[3 + 3*i] = 0.01
                    
            new_energy = compute_energy(new_state)
            
            # Accept/Reject
            if new_energy < current_energy:
                current_state = new_state
                current_energy = new_energy
            else:
                delta_e = new_energy - current_energy
                if np.random.random() < np.exp(-delta_e / T):
                    current_state = new_state
                    current_energy = new_energy
            
            # Cool down
            T = max(T * alpha, T_min)
            
            # Periodic refinement to escape local minima or improve precision
            if k % 500 == 0 and k > 0:
                refined = refine(current_state)
                if compute_energy(refined) < current_energy:
                    current_state = refined
                    current_energy = compute_energy(refined)
        
        # Final refinement
        final_state = refine(current_state)
        
        if is_valid(final_state):
            score = np.sum(final_state[3::3])
            if score > best_score:
                best_score = score
                best_state = final_state.copy()
        
        # Check time
        if time.time() - start_time > time_limit - 10:
            break
            
    if best_state is None:
        # Fallback: Grid in 1x1
        w = 1.0
        h = 1.0
        side = int(np.ceil(np.sqrt(N)))
        result = []
        for i in range(N):
            col = i % side
            row = i // side
            xi = (col + 0.5) / side
            yi = (row + 0.5) / side
            ri = 0.5 / side - 1e-4
            result.append([xi, yi, ri])
        return np.array(result)
        
    # Extract result
    w = best_state[0]
    x = best_state[1::3]
    y = best_state[2::3]
    r = best_state[3::3]
    
    result = []
    for i in range(N):
        result.append([x[i], y[i], r[i]])
        
    return np.array(result)
