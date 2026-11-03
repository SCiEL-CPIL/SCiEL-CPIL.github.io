import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_r = 0.0
    best_points = None

    # Initial guesses
    guesses = []
    
    # 1. Concentric rings
    # 1, 6, 12, 18, 13
    ring_counts = [1, 6, 12, 18, 13]
    points = []
    for count in ring_counts:
        if count == 1:
            points.append([0.0, 0.0])
        else:
            angles = np.linspace(0, 2*np.pi, count, endpoint=False)
            # Rough radius estimate for ring k (1-indexed)
            # Radius ~ (k-0.5) * spacing
            # Spacing ~ 2r ~ 0.25
            # r_k ~ 0.125 * (k-0.5) ? 
            # Let's just place them at equal angular spacing and let the optimizer adjust radius.
            # Initial radius guess: 1.0 / (len(ring_counts) + 1) * (index + 1)
            pass
            
    # Better initial ring radii:
    # Total area ~ pi * r_max^2.
    # Let's just use a spiral or uniform rings.
    ring_radii = np.linspace(0.1, 0.9, len(ring_counts))
    for i, count in enumerate(ring_counts):
        r_ring = ring_radii[i]
        if count == 1:
            points.append([0.0, 0.0])
        else:
            angles = np.linspace(0, 2*np.pi, count, endpoint=False)
            for a in angles:
                points.append([r_ring * np.cos(a), r_ring * np.sin(a)])
    guesses.append(np.array(points))

    # 2. Hexagonal lattice
    spacing = 0.25
    points = []
    for i in range(-10, 11):
        for j in range(-10, 11):
            x = i * spacing + (j % 2) * spacing * 0.5
            y = j * spacing * np.sqrt(3) / 2
            if x*x + y*y < 0.9**2: # Keep well inside
                points.append([x, y])
    if len(points) > N:
        # Sort by distance to center and take closest N
        d = np.hypot(np.array(points)[:,0], np.array(points)[:,1])
        idx = np.argsort(d)[:N]
        guesses.append(np.array(points)[idx])
    else:
        guesses.append(np.array(points) + np.random.randn(len(points), 2)*0.01) # Pad if needed? No, N is fixed.
        # If we don't have enough, add random
        while len(points) < N:
            points.append([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)])
        guesses.append(np.array(points))

    # 3. Random
    guesses.append(np.random.uniform(-0.8, 0.8, (N, 2)))

    # 4. Spiral
    t = np.linspace(0, 1, N)
    r = 0.1 + 0.8 * t
    theta = 2 * np.pi * 10 * t
    points = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    guesses.append(points)

    # Optimization function
    def objective(vars):
        # vars: [x1, y1, ..., x50, y50, r]
        pts = vars[:2*N].reshape(N, 2)
        r = vars[2*N]
        return -r

    def constraints(vars):
        pts = vars[:2*N].reshape(N, 2)
        r = vars[2*N]
        cons = []
        # Pairwise distances
        for i in range(N):
            for j in range(i+1, N):
                dx = pts[i,0] - pts[j,0]
                dy = pts[i,1] - pts[j,1]
                d = np.sqrt(dx*dx + dy*dy)
                cons.append(d - 2*r)
        # Boundary
        for i in range(N):
            d = np.sqrt(pts[i,0]**2 + pts[i,1]**2)
            cons.append(1.0 - d - r)
        # r > 0
        cons.append(r - 0.01)
        
        # Convert to dict for SLSQP
        return [{'type': 'ineq', 'fun': lambda v, c=c: c(v)} for c in cons]
        # This lambda closure is slow. Let's use a vectorized approach if possible or just list of funcs.
        # SLSQP accepts a list of dicts.

    # Actually, defining 1225 constraints individually is slow for SLSQP.
    # Let's use a penalty method with L-BFGS-B or minimize_scalar?
    # Or use 'trust-constr' which handles sparse constraints better?
    # Or simply use a penalty function and minimize it.
    
    def penalty_objective(vars):
        pts = vars[:2*N].reshape(N, 2)
        r = vars[2*N]
        if r < 0: r = 0
        pen = 0.0
        # Overlaps
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((pts[i,0]-pts[j,0])**2 + (pts[i,1]-pts[j,1])**2)
                if d < 2*r:
                    pen += (2*r - d)**2
        # Boundary
        for i in range(N):
            d = np.sqrt(pts[i,0]**2 + pts[i,1]**2)
            if d > 1.0 - r:
                pen += (d - (1.0 - r))**2
        # We want to maximize r, so minimize -r + penalty
        return -r + 100.0 * pen

    # Use L-BFGS-B with penalty
    # Bounds: x,y in [-1, 1], r in [0, 0.5]
    bounds = [(-1.0, 1.0)] * (2*N) + [(0.0, 0.5)]
    
    for guess in guesses:
        if time.time() > deadline:
            break
            
        x0 = np.concatenate([guess.flatten(), [0.1]]) # Initial r guess
        
        try:
            res = minimize(penalty_objective, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.success or res.fun < 1e-5: # Check if penalty is low
                r_found = res.x[2*N]
                pts_found = res.x[:2*N].reshape(N, 2)
                
                # Validate
                valid = True
                min_dist = 1e9
                for i in range(N):
                    d_center = np.hypot(pts_found[i,0], pts_found[i,1])
                    if d_center > 1.0 - r_found + 1e-9:
                        valid = False
                        break
                    for j in range(i+1, N):
                        d = np.hypot(pts_found[i,0]-pts_found[j,0], pts_found[i,1]-pts_found[j,1])
                        if d < 2*r_found - 1e-9:
                            valid = False
                            break
                    if not valid: break
                if valid:
                    if r_found > best_r:
                        best_r = r_found
                        best_points = pts_found
        except Exception as e:
            pass

    if best_points is None:
        # Fallback to seed
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                return points[np.argsort(radius, kind="stable")[:N]]
            spacing *= 0.98
            
    return best_points
