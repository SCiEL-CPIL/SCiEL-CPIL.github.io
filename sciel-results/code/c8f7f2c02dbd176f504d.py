"""Circle packing in perimeter-4 rectangle using force-directed relaxation with SLSQP refinement."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit
    
    best_value = 0.0
    best_result = None
    
    # Parameters
    penalty_weight = 500.0
    
    def unpack(z):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w
        return x, y, r, w, h
    
    def check_validity(z, tol=1e-8):
        x, y, r, w, h = unpack(z)
        
        if w <= 0.1 or h <= 0.1:
            return False, 0.0
            
        for i in range(N):
            if r[i] <= 1e-9:
                return False, 0.0
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return False, 0.0
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return False, 0.0
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d_sq = dx * dx + dy * dy
                r_sum = r[i] + r[j]
                if d_sq < r_sum * r_sum - tol:
                    return False, 0.0
        
        val = float(np.sum(r))
        return True, val
    
    def force_directed_relax(z, max_iter=500, dt=0.005):
        """Force-directed relaxation to resolve overlaps and push circles to expand."""
        z = z.copy().astype(float)
        n = N
        
        for it in range(max_iter):
            x, y, r, w, h = unpack(z)
            
            if w < 0.1 or h < 0.1:
                break
                
            fx = np.zeros(n)
            fy = np.zeros(n)
            
            # Wall repulsion - push circles inward if they cross boundary
            for i in range(n):
                if x[i] - r[i] < 0:
                    fx[i] += (x[i] - r[i]) * 50.0
                if x[i] + r[i] > w:
                    fx[i] -= (x[i] + r[i] - w) * 50.0
                if y[i] - r[i] < 0:
                    fy[i] += (y[i] - r[i]) * 50.0
                if y[i] + r[i] > h:
                    fy[i] -= (y[i] + r[i] - h) * 50.0
            
            # Pairwise repulsion
            for i in range(n):
                for j in range(i + 1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d_sq = dx * dx + dy * dy
                    min_d = r[i] + r[j]
                    if d_sq < min_d * min_d and d_sq > 1e-12:
                        d = np.sqrt(d_sq)
                        overlap = min_d - d
                        f = overlap * 20.0
                        fx[i] += (dx / d) * f
                        fy[i] += (dy / d) * f
                        fx[j] -= (dx / d) * f
                        fy[j] -= (dy / d) * f
            
            # Update positions
            z[0::3][:n] += dt * fx
            z[1::3][:n] += dt * fy
            
            # Clamp positions to stay within bounds
            for i in range(n):
                z[3 * i] = np.clip(z[3 * i], r[i], w - r[i])
                z[3 * i + 1] = np.clip(z[3 * i + 1], r[i], h - r[i])
            
            # Check if converged (no significant forces)
            if np.max(np.abs(fx)) < 1e-6 and np.max(np.abs(fy)) < 1e-6:
                break
        
        return z
    
    def expand_radii(z, max_iter=200):
        """Iteratively expand radii while maintaining validity."""
        z = z.copy().astype(float)
        n = N
        
        for _ in range(max_iter):
            x, y, r, w, h = unpack(z)
            
            # For each circle, find max possible radius
            improved = False
            for i in range(n):
                max_r = min(x[i], w - x[i], y[i], h - y[i])
                
                for j in range(n):
                    if i == j:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx * dx + dy * dy)
                    max_r = min(max_r, dist - r[j])
                
                if max_r > r[i] + 1e-10:
                    r[i] = max_r
                    improved = True
            
            if not improved:
                break
            
            z[2::3][:n] = r
        
        return z
    
    def penalty_objective(z):
        x, y, r, w, h = unpack(z)
        
        if w < 0.2 or h < 0.2 or np.any(r < 0):
            return 1e6
            
        obj = -np.sum(r)
        
        # Overlap Penalty
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d_sq = dx * dx + dy * dy
                dist_req = r[i] + r[j]
                if d_sq < dist_req * dist_req:
                    d = np.sqrt(d_sq)
                    overlap = dist_req - d
                    obj += penalty_weight * overlap * overlap
        
        # Boundary Penalty
        for i in range(N):
            if x[i] < r[i]:
                obj += penalty_weight * (r[i] - x[i]) ** 2
            elif x[i] > w - r[i]:
                obj += penalty_weight * (x[i] - (w - r[i])) ** 2
            if y[i] < r[i]:
                obj += penalty_weight * (r[i] - y[i]) ** 2
            elif y[i] > h - r[i]:
                obj += penalty_weight * (y[i] - (h - r[i])) ** 2
                
        return obj
    
    def generate_initial_guesses():
        guesses = []
        rng = np.random.default_rng(42)
        
        # Grid-based starts with varying aspect ratios
        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5, 1.2, 0.8, 0.9, 1.1]
        side = int(np.ceil(np.sqrt(N)))
        
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z = np.zeros(3 * N + 1)
            for i in range(N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w0
                cy = (row + 0.5) / side * h0
                r0 = min(w0 / side, h0 / side) * 0.45
                z[3*i] = cx
                z[3*i+1] = cy
                z[3*i+2] = r0
            z[-1] = w0
            guesses.append(z)
            
        # Random starts
        for _ in range(15):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.05, 0.12)
                cx = rng.uniform(r0, w0 - r0)
                cy = rng.uniform(r0, h0 - r0)
                z[3*i] = cx
                z[3*i+1] = cy
                z[3*i+2] = r0
            z[-1] = w0
            guesses.append(z)
            
        return guesses
    
    def slsqp_refine(z_init, time_budget):
        """Refine using SLSQP with hard constraints."""
        if time_budget < 3:
            return z_init
            
        n = N
        z = z_init.copy()
        
        # Hard constraints for SLSQP
        constraints = []
        
        # Boundary constraints
        for i in range(n):
            constraints.append({'type': 'ineq', 'fun': lambda p, idx=i: p[3*idx] - p[3*idx+2]})
            constraints.append({'type': 'ineq', 'fun': lambda p, idx=i: p[-1] - p[3*idx] - p[3*idx+2]})
            constraints.append({'type': 'ineq', 'fun': lambda p, idx=i: p[3*idx+1] - p[3*idx+2]})
            constraints.append({'type': 'ineq', 'fun': lambda p, idx=i: (2.0 - p[-1]) - p[3*idx+1] - p[3*idx+2]})
            
        # Overlap constraints
        for i in range(n):
            for j in range(i + 1, n):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda p, idx1=i, idx2=j: (
                        (p[3*idx1] - p[3*idx2])**2 + 
                        (p[3*idx1+1] - p[3*idx2+1])**2 - 
                        (p[3*idx1+2] + p[3*idx2+2])**2
                    )
                })
        
        # Bounds
        w_min, w_max = 0.3, 1.7
        bounds = []
        for i in range(n):
            bounds.append((0.01, 1.5))  # x
            bounds.append((0.01, 1.5))  # y
            bounds.append((0.01, 0.5))  # r
        bounds.append((w_min, w_max))  # w
        
        # Objective: minimize negative sum of radii
        def obj(z):
            x, y, r, w, h = unpack(z)
            return -np.sum(r)
        
        try:
            result = minimize(
                obj, z, method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-8}
            )
            
            if result.success and result.fun < 0:
                z_opt = result.x
                valid, val = check_validity(z_opt)
                if valid:
                    return z_opt
        except Exception:
            pass
        
        return z_init
    
    # Main search loop
    guesses = generate_initial_guesses()
    
    for z_init in guesses:
        if time.time() > deadline:
            break
        
        # Step 1: Force-directed relaxation
        z_relaxed = force_directed_relax(z_init, max_iter=300)
        
        # Step 2: Expand radii
        z_expanded = expand_radii(z_relaxed, max_iter=100)
        
        # Check validity
        valid, val = check_validity(z_expanded)
        if valid and val > best_value:
            best_value = val
            best_result = np.column_stack([
                z_expanded[0::3][:N],
                z_expanded[1::3][:N],
                z_expanded[2::3][:N]
            ])
        
        # Step 3: SLSQP refinement if time permits
        remaining = deadline - time.time()
        if remaining > 5:
            z_refined = slsqp_refine(z_expanded, time_budget=min(10.0, remaining))
            valid, val = check_validity(z_refined)
            if valid and val > best_value:
                best_value = val
                best_result = np.column_stack([
                    z_refined[0::3][:N],
                    z_refined[1::3][:N],
                    z_refined[2::3][:N]
                ])
    
    # Fallback
    if best_result is None:
        w, h = 1.0, 1.0
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    x = (i + 0.5) / side * w
                    y = (j + 0.5) / side * h
                    r = min(w, h) / (2 * side) * 0.9
                    cells.append([x, y, r])
        best_result = np.array(cells[:N])
    
    return best_result
