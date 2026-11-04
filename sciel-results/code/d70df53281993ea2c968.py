"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _kinetic_annealing(n, deadline, seed=0):
    """
    Kinetic annealing strategy:
    1. Initialize circles randomly.
    2. Simulate rigid bodies with repulsive forces (overlap resolution) 
       and attractive forces (towards center/walls to maximize packing).
    3. Use adaptive damping and temperature to allow exploration.
    4. Periodically try to increase radii (inflation) if no overlaps exist.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize positions and radii
    # Start with small radii to allow movement
    x = rng.uniform(0.1, 0.9, n)
    y = rng.uniform(0.1, 0.9, n)
    r = np.full(n, 0.02)
    
    # Velocities
    vx = np.zeros(n)
    vy = np.zeros(n)
    
    # Simulation parameters
    dt = 0.01
    damping = 0.95
    k_repel = 100.0  # Repulsion strength
    k_attract = 1.0   # Attraction to center strength (weak, just to keep them in)
    inflation_rate = 0.001
    temp = 1.0
    
    best_total = 0.0
    best_state = (x.copy(), y.copy(), r.copy())
    
    start_time = time.time()
    
    # Pre-allocate force arrays
    fx = np.zeros(n)
    fy = np.zeros(n)
    
    steps = 0
    max_steps = 50000
    
    while time.time() < deadline - 5.0 and steps < max_steps:
        steps += 1
        
        # Calculate forces
        fx[:] = 0
        fy[:] = 0
        
        # Repulsive forces between circles
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(dist_sq)
                min_dist = r[i] + r[j]
                
                if dist < min_dist and dist > 1e-6:
                    # Overlap! Apply strong repulsive force
                    # Force is proportional to penetration depth
                    penetration = min_dist - dist
                    force = k_repel * penetration
                    # Normalize direction
                    nx = dx / dist
                    ny = dy / dist
                    fx[i] += force * nx
                    fy[i] += force * ny
                    fx[j] -= force * nx
                    fy[j] -= force * ny
                elif dist < 1e-6:
                    # Exactly overlapping, push apart randomly
                    angle = rng.uniform(0, 2 * np.pi)
                    fx[i] += k_repel * np.cos(angle)
                    fy[i] += k_repel * np.sin(angle)
                    fx[j] -= k_repel * np.cos(angle)
                    fy[j] -= k_repel * np.sin(angle)
        
        # Attractive force towards center (to prevent drifting to corners unnecessarily, 
        # though corner packing is often optimal, this helps stability)
        # Actually, for maximizing sum of radii, we want them to spread out.
        # A better "attractive" force is to push them towards the walls if they are small?
        # No, standard MD for packing usually relies on repulsion and a container.
        # Let's use a soft potential to keep them inside [0,1]
        for i in range(n):
            # Wall constraints via soft repulsion
            # Left wall
            if x[i] < r[i]:
                fx[i] += k_repel * (r[i] - x[i])
            # Right wall
            if x[i] > 1.0 - r[i]:
                fx[i] -= k_repel * (x[i] - (1.0 - r[i]))
            # Bottom wall
            if y[i] < r[i]:
                fy[i] += k_repel * (y[i] - r[i])
            # Top wall
            if y[i] > 1.0 - r[i]:
                fy[i] -= k_repel * (y[i] - (1.0 - r[i]))
            
            # Weak centering force to prevent all circles from clumping in one corner
            # This helps distribute them initially
            fx[i] -= k_attract * (x[i] - 0.5)
            fy[i] -= k_attract * (y[i] - 0.5)
            
            # Random thermal kick
            fx[i] += rng.normal(0, temp)
            fy[i] += rng.normal(0, temp)
            
        # Update velocities and positions (Symplectic Euler)
        vx = vx * damping + fx * dt
        vy = vy * damping + fy * dt
        
        x = x + vx * dt
        y = y + vy * dt
        
        # Hard constraint clipping (safety net)
        for i in range(n):
            x[i] = np.clip(x[i], r[i], 1.0 - r[i])
            y[i] = np.clip(y[i], r[i], 1.0 - r[i])
            # If clipped, zero out velocity component
            if x[i] == r[i] or x[i] == 1.0 - r[i]:
                vx[i] = 0
            if y[i] == r[i] or y[i] == 1.0 - r[i]:
                vy[i] = 0
                
        # Check for validity and try to inflate
        valid = True
        for i in range(n):
            if x[i] < r[i] - 1e-9 or x[i] > 1.0 - r[i] + 1e-9:
                valid = False
            if y[i] < r[i] - 1e-9 or y[i] > 1.0 - r[i] + 1e-9:
                valid = False
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] - 1e-9:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Inflate radii slightly
            # Ensure inflation doesn't cause immediate overlap
            max_possible_inflation = 0.0
            for i in range(n):
                # Distance to walls
                d_wall = min(x[i], 1-x[i], y[i], 1-y[i]) - r[i]
                # Distance to others
                for j in range(i+1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    d_other = dist - r[i] - r[j]
                    if d_other < d_wall:
                        d_wall = d_other
                # Limit inflation to half the available space to avoid overlap with neighbor
                # Actually, if we inflate all by dr, overlap increases by 2*dr.
                # So max dr is min(d_available) / 2
                if d_wall < max_possible_inflation:
                    max_possible_inflation = d_wall
                    
            if max_possible_inflation > 0:
                dr = min(inflation_rate, max_possible_inflation * 0.5)
                r += dr
                
            current_total = np.sum(r)
            if current_total > best_total:
                best_total = current_total
                best_state = (x.copy(), y.copy(), r.copy())
                
        # Cool down
        temp *= 0.999
        if temp < 0.01:
            temp = 0.01
            
    return best_state[0], best_state[1], best_state[2], best_total

def _gradient_polish(x, y, r, deadline):
    """
    Refine the kinetic result using gradient-based optimization.
    Objective: Maximize sum(r)
    Constraints: 
    1. x_i >= r_i, 1-x_i >= r_i, y_i >= r_i, 1-y_i >= r_i
    2. dist_ij >= r_i + r_j
    """
    n = len(r)
    
    def flatten(xy, r):
        return np.concatenate([xy.ravel(), r])
        
    def unflatten(z):
        xy = z[:2*n].reshape(n, 2)
        r = z[2*n:]
        return xy, r
        
    def objective(z):
        xy, r = unflatten(z)
        # Return negative sum of radii to minimize
        return -np.sum(r)
        
    def constraints(z):
        xy, r = unflatten(z)
        cons = []
        # Wall constraints
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[2*i] - z[2*n+i]}) # x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: (1.0 - z[2*i]) - z[2*n+i]}) # (1-x_i) - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[2*i+1] - z[2*n+i]}) # y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: (1.0 - z[2*i+1]) - z[2*n+i]}) # (1-y_i) - r_i >= 0
        # Overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda z, i=i, j=j: np.sqrt((z[2*i]-z[2*j])**2 + (z[2*i+1]-z[2*j+1])**2) - z[2*n+i] - z[2*n+j]
                })
        return cons
        
    z0 = flatten(x, y, r)
    
    try:
        res = minimize(
            objective, 
            z0, 
            method='SLSQP', 
            constraints=constraints(z0),
            bounds=([(0, 1)] * (2*n) + [(0, 0.5)] * n),
            options={'maxiter': 500, 'ftol': 1e-9}
        )
        
        if res.success:
            xy_opt, r_opt = unflatten(res.x)
            total = np.sum(r_opt)
            
            # Final validity check
            valid = True
            for i in range(n):
                if xy_opt[i,0] < r_opt[i] - 1e-6 or xy_opt[i,0] > 1.0 - r_opt[i] + 1e-6:
                    valid = False
                if xy_opt[i,1] < r_opt[i] - 1e-6 or xy_opt[i,1] > 1.0 - r_opt[i] + 1e-6:
                    valid = False
            for i in range(n):
                for j in range(i+1, n):
                    d = np.linalg.norm(xy_opt[i] - xy_opt[j])
                    if d < r_opt[i] + r_opt[j] - 1e-6:
                        valid = False
                        
            if valid:
                return xy_opt[:,0], xy_opt[:,1], r_opt, total
            else:
                # If invalid, shrink radii slightly
                for i in range(n):
                    r_opt[i] *= 0.99
                return xy_opt[:,0], xy_opt[:,1], r_opt, np.sum(r_opt)
        else:
            return x, y, r, np.sum(r)
    except Exception:
        return x, y, r, np.sum(r)

def construct():
    start_time = time.time()
    # Evaluator kills at 150s. Leave 10s margin.
    deadline = start_time + 140.0
    
    best_total = 0.0
    best_x = None
    best_y = None
    best_r = None
    
    # Run kinetic annealing with multiple seeds
    # Each run takes some time, so we limit the number of seeds based on time
    # Estimate: one run takes ~5-10s? Let's try 5 seeds.
    
    seeds = [42, 123, 456, 789, 1011]
    
    for seed in seeds:
        if time.time() > deadline - 20.0:
            break
            
        try:
            x, y, r, total = _kinetic_annealing(N, deadline, seed=seed)
            if total > best_total:
                best_total = total
                best_x = x.copy()
                best_y = y.copy()
                best_r = r.copy()
        except Exception:
            pass
            
    # If we have time, polish the best result
    if best_x is not None and time.time() < deadline - 10.0:
        try:
            x_pol, y_pol, r_pol, total_pol = _gradient_polish(best_x, best_y, best_r, deadline)
            if total_pol > best_total:
                best_total = total_pol
                best_x = x_pol
                best_y = y_pol
                best_r = r_pol
        except Exception:
            pass
            
    # Fallback if no valid result found
    if best_x is None:
        # Simple grid fallback
        side = int(np.ceil(np.sqrt(N)))
        xs, ys = np.mgrid[0:side, 0:side]
        xs = xs.ravel()[:N]
        ys = ys.ravel()[:N]
        best_x = (xs + 0.5) / side
        best_y = (ys + 0.5) / side
        best_r = np.full(N, 0.5/side - 0.01)
        best_total = np.sum(best_r)
        
    # Final safety shrink to ensure strict validity
    for i in range(N):
        r_i = best_r[i]
        x_i = best_x[i]
        y_i = best_y[i]
        
        # Check walls
        max_r = min(x_i, 1-x_i, y_i, 1-y_i)
        if r_i > max_r:
            r_i = max_r * 0.999
            
        # Check overlaps
        for j in range(i+1, N):
            d = np.sqrt((best_x[i]-best_x[j])**2 + (best_y[i]-best_y[j])**2)
            max_r_overlap = (d - best_r[j]) * 0.999
            if r_i > max_r_overlap:
                r_i = max_r_overlap
        best_r[i] = max(0.0, r_i)
        
    # Re-validate
    valid = True
    for i in range(N):
        if best_x[i] - best_r[i] < -1e-7 or 1.0 - best_x[i] - best_r[i] < -1e-7:
            valid = False
        if best_y[i] - best_r[i] < -1e-7 or 1.0 - best_y[i] - best_r[i] < -1e-7:
            valid = False
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt((best_x[i]-best_x[j])**2 + (best_y[i]-best_y[j])**2)
            if d < best_r[i] + best_r[j] - 1e-7:
                valid = False
                
    if not valid:
        # Extreme fallback
        side = 6
        for i in range(N):
            gx = i % side
            gy = i // side
            best_x[i] = (gx + 0.5) / side
            best_y[i] = (gy + 0.5) / side
            best_r[i] = 0.5/side - 0.05
            
    result = np.zeros((N, 3))
    result[:, 0] = best_x
    result[:, 1] = best_y
    result[:, 2] = best_r
    
    return result
# EVOLVE-BLOCK-END
