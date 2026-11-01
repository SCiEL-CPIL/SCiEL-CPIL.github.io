import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import time

N = 32

def _hexagonal_init(n):
    """Generate a hexagonal grid of n points inside [0,1]^2."""
    # We need to find a good grid size
    # Approximate area per circle in hex packing is (sqrt(3)/2) * r^2
    # Total area 1, so n * (sqrt(3)/2) * r^2 <= 1 => r <= sqrt(2 / (n*sqrt(3)))
    # Let's just generate a grid and pick the best n points
    
    # Try different grid spacings
    best_points = None
    best_sum = 0
    
    # Estimate optimal spacing
    # For n points, spacing s ~ 1/sqrt(n)
    # Hex grid has rows offset by s/2, row height s*sqrt(3)/2
    
    for scale in np.linspace(0.1, 0.3, 20):
        # Generate points
        points = []
        y = scale * np.sqrt(3) / 2
        row = 0
        while y <= 1.0:
            offset = 0 if row % 2 == 0 else scale / 2
            x = offset
            while x <= 1.0:
                points.append((x, y))
                x += scale
            y += scale * np.sqrt(3) / 2
            row += 1
        
        points = np.array(points)
        if len(points) < n:
            continue
            
        # Select n points that maximize the minimum distance to boundaries and each other?
        # Or just take the first n?
        # Better: take points that are furthest from the boundary?
        # Or use a greedy selection: start with center, add point that maximizes min distance to existing points.
        
        # Greedy selection
        selected = []
        # Start with the point closest to center (0.5, 0.5)
        dists_to_center = np.sum((points - 0.5)**2, axis=1)
        idx = np.argmin(dists_to_center)
        selected.append(points[idx])
        
        for _ in range(n - 1):
            # Calculate min distance from all unselected points to all selected points
            min_dists = []
            for p in points:
                if any(np.allclose(p, s) for s in selected):
                    continue
                d = np.min([np.linalg.norm(p - s) for s in selected])
                # Also consider distance to boundary to prefer interior points
                d_boundary = np.min([p[0], 1-p[0], p[1], 1-p[1]])
                # Weighted score
                score = d + 0.1 * d_boundary 
                min_dists.append((score, p))
            
            if not min_dists:
                break
                
            min_dists.sort(key=lambda x: x[0], reverse=True)
            selected.append(min_dists[0][1])
            
        if len(selected) == n:
            selected = np.array(selected)
            # Calculate sum of potential radii (min dist to neighbor/2 and boundary)
            sum_r = 0
            for i in range(n):
                d_min = np.min([selected[i][0], 1-selected[i][0], selected[i][1], 1-selected[i][1]])
                for j in range(n):
                    if i != j:
                        d = np.linalg.norm(selected[i] - selected[j]) / 2
                        d_min = min(d_min, d)
                sum_r += d_min
            
            if sum_r > best_sum:
                best_sum = sum_r
                best_points = selected
                
    if best_points is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(n)))
        cells = [((i + 0.5) / side, (j + 0.5) / side) for i in range(side) for j in range(side)]
        best_points = np.array(cells[:n])
        
    return best_points

def construct():
    start_time = time.time()
    timeout = 100  # Keep some buffer
    
    # Initial guess
    centers = _hexagonal_init(N)
    
    # Initial radii: min distance to boundary and half distance to nearest neighbor
    radii = np.zeros(N)
    for i in range(N):
        d_bound = np.min([centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1]])
        d_neigh = np.inf
        for j in range(N):
            if i != j:
                d = np.linalg.norm(centers[i] - centers[j]) / 2
                d_neigh = min(d_neigh, d)
        radii[i] = min(d_bound, d_neigh)
        
    # Variables: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(3 * N)
    for i in range(N):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]
        
    def objective(v):
        # Maximize sum of radii -> Minimize -sum
        r = v[2::3]
        return -np.sum(r)
        
    def constraints(v):
        cons = []
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        
        # Boundary constraints: r <= x, r <= 1-x, r <= y, r <= 1-y
        # x - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda v: v[0::3] - v[2::3]})
        cons.append({'type': 'ineq', 'fun': lambda v: 1 - v[0::3] - v[2::3]})
        cons.append({'type': 'ineq', 'fun': lambda v: v[1::3] - v[2::3]})
        cons.append({'type': 'ineq', 'fun': lambda v: 1 - v[1::3] - v[2::3]})
        
        # Positivity: r >= 0
        cons.append({'type': 'ineq', 'fun': lambda v: v[2::3]})
        
        # Pairwise constraints: (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
        # (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
        for i in range(N):
            for j in range(i+1, N):
                xi, yi, ri = v[3*i], v[3*i+1], v[3*i+2]
                xj, yj, rj = v[3*j], v[3*j+1], v[3*j+2]
                # Note: SLSQP might struggle with many non-linear constraints
                # N=32 -> 32*31/2 = 496 constraints. This might be slow.
                
                # We can define a single constraint function that returns an array
                # But SLSQP expects a list of dicts or a single dict with array fun.
                # Let's use a single dict with an array for pairwise constraints to reduce overhead?
                # Actually, SLSQP handles arrays well.
                pass
        
        # Let's restructure constraints for efficiency
        # We will compute the pairwise constraints in a vectorized way if possible,
        # but the lambda inside the loop is inefficient.
        
        # Better approach: Define a single constraint function that returns an array of all constraint values.
        # SLSQP supports this if we provide a single 'ineq' constraint with an array-valued fun.
        # However, the standard interface is a list.
        # Let's stick to the list but optimize the calculation.
        
        # Actually, for N=32, 496 constraints is not huge, but the lambda overhead is.
        # Let's create the constraint functions outside the loop if possible?
        # No, they depend on v.
        
        # Let's use a single function that returns all pairwise constraints.
        # We'll modify the cons list construction.
        
        # We'll keep the boundary ones as is.
        # For pairwise, we'll add one entry that computes all.
        
        def pairwise_cons(v):
            x = v[0::3]
            y = v[1::3]
            r = v[2::3]
            cons_vals = np.zeros(N*(N-1)//2)
            k = 0
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dr = r[i] + r[j]
                    cons_vals[k] = dx*dx + dy*dy - dr*dr
                    k += 1
            return cons_vals
            
        cons.append({'type': 'ineq', 'fun': pairwise_cons})
        
        return cons
        
    # Run optimization
    # SLSQP is a good choice for this size
    # We might need to try multiple restarts
    
    best_val = objective(x0)
    best_x = x0
    
    # Try a few restarts if time permits
    restarts = 3
    for k in range(restarts):
        if time.time() - start_time > timeout:
            break
            
        # Perturb initial guess slightly
        if k > 0:
            x0_perturbed = x0.copy()
            # Randomly perturb positions and radii slightly
            noise_pos = np.random.normal(0, 0.01, size=(N, 2))
            noise_rad = np.random.normal(0, 0.005, size=N)
            
            for i in range(N):
                x0_perturbed[3*i] += noise_pos[i, 0]
                x0_perturbed[3*i+1] += noise_pos[i, 1]
                x0_perturbed[3*i+2] += noise_rad[i]
                
            # Ensure validity of perturbed start
            # Clamp positions to [0.01, 0.99] and radii to [0.001, 0.4]
            x0_perturbed[0::3] = np.clip(x0_perturbed[0::3], 0.01, 0.99)
            x0_perturbed[1::3] = np.clip(x0_perturbed[1::3], 0.01, 0.99)
            x0_perturbed[2::3] = np.clip(x0_perturbed[2::3], 0.001, 0.4)
        else:
            x0_perturbed = x0
            
        res = minimize(objective, x0_perturbed, method='SLSQP', constraints=constraints(x0_perturbed), bounds=[(0.01, 0.99)]*N + [(0.01, 0.99)]*N + [(0.001, 0.4)]*N, options={'maxiter': 1000, 'ftol': 1e-6})
        
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_x = res.x
            
    # Extract result
    x = best_x[0::3]
    y = best_x[1::3]
    r = best_x[2::3]
    
    # Ensure strict validity (due to numerical errors)
    # Check constraints again
    for i in range(N):
        r[i] = min(r[i], x[i], 1-x[i], y[i], 1-y[i])
        for j in range(i+1, N):
            d = np.linalg.norm([x[i]-x[j], y[i]-y[j]])
            if d < r[i] + r[j]:
                # Reduce radii to fix overlap
                overlap = (r[i] + r[j] - d) / 2
                r[i] -= overlap
                r[j] -= overlap
                # Ensure positive
                r[i] = max(r[i], 0.001)
                r[j] = max(r[j], 0.001)
                
    # Final check and clip
    for i in range(N):
        r[i] = min(r[i], x[i], 1-x[i], y[i], 1-y[i])
        r[i] = max(r[i], 0.0)
        
    result = np.column_stack((x, y, r))
    return result
