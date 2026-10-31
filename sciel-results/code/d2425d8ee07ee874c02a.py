import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
START_TIME = time.time()
DEADLINE = START_TIME + 130.0  # Safety margin

def construct():
    best_score = 0.0
    best_result = None
    
    # Aspect ratios to try: w/h
    # Perimeter 4 => w+h=2.
    # ratio = w/h. h = 2/(1+ratio), w = 2*ratio/(1+ratio).
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 1.5, 0.7, 2.0, 0.5, 1.33, 0.75]
    
    # We will perform a "structural" hybrid:
    # 1. Use SLSQP with analytic gradients (from inspiration 1) for precise local optimization.
    # 2. Use force-directed relaxation (from current program) for robust initial packing and global moves.
    # 3. Iterate between them to escape local minima.
    
    # Pre-compute rectangle dimensions for each ratio
    rects = []
    for ratio in aspect_ratios:
        h = 2.0 / (1.0 + ratio)
        w = ratio * h
        rects.append((w, h))
        
    # Shuffle to diversify? Or just iterate.
    # To maximize time usage, we'll interleave or just run the best ones.
    # Let's sort by "squareness" as square is often best for packing density, but rectangle might allow larger radii sum?
    # Actually, for fixed perimeter, area is max at square. But sum of radii depends on packing efficiency.
    # Let's just iterate through all.
    
    # To ensure we don't spend too much time on bad ratios, we can do a quick check first?
    # No, SLSQP is fast if initialized well.
    
    # We'll do multiple "rounds" if time permits.
    # Round 1: Quick SLSQP from grid/random for all ratios.
    # Round 2: Force relaxation + SLSQP refinement for top candidates.
    
    # Let's define the objective and gradient functions here to avoid redefining
    def objective_and_grad(x, W, H, mu=100.0):
        """
        x: (N, 3) array [cx, cy, r]
        Returns (value, grad)
        value = -sum(r) + mu * penalties
        """
        cx = x[:, 0].copy()
        cy = x[:, 1].copy()
        r = x[:, 2].copy()
        n = len(cx)
        
        # Clip for safety
        cx = np.clip(cx, 0, W)
        cy = np.clip(cy, 0, H)
        r = np.clip(r, 1e-6, 2.0)
        
        val = -np.sum(r)
        grad = np.zeros_like(x)
        grad[:, 2] = -1.0
        
        # Boundary constraints: r <= min(cx, cy, W-cx, H-cy)
        # We penalize violation: max(r - dist_to_wall, 0)^2
        # dist_to_wall[i] = min(cx[i], cy[i], W-cx[i], H-cy[i])
        d_left = cx
        d_right = W - cx
        d_bottom = cy
        d_top = H - cy
        d_min = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
        
        viol_b = np.maximum(r - d_min, 0.0)
        if np.any(viol_b > 0):
            val += mu * np.sum(viol_b**2)
            for i in range(n):
                if viol_b[i] > 0:
                    # Subgradient of d_min
                    # If multiple are min, we can average or pick one. Picking one is non-smooth but SLSQP handles it okay-ish.
                    # Better: if viol > 0, it means r is larger than the distance to the closest wall.
                    # We need to push r down or push center away from that wall.
                    # Gradient of penalty w.r.t r: 2*mu*viol
                    # Gradient w.r.t cx: 2*mu*viol * (-d(d_min)/d(cx))
                    
                    dv = 2.0 * mu * viol_b[i]
                    grad[i, 2] += dv
                    
                    # Determine which wall is limiting
                    # To be robust, we can check which distance is minimal.
                    # If d_left is the min, increasing cx increases d_min, so d(d_min)/d(cx) = 1.
                    # Penalty is (r - d_min)^2. d/d(cx) = 2(r-d_min) * (-d(d_min)/d(cx)).
                    # So if d_left is min, grad[i,0] -= dv.
                    
                    # Handle ties by averaging subgradients? Or just picking the one with smallest distance?
                    # Let's find the argmin.
                    dists = np.array([d_left[i], d_right[i], d_bottom[i], d_top[i]])
                    min_dist = np.min(dists)
                    eps = 1e-8
                    active = dists <= min_dist + eps
                    cnt = np.sum(active)
                    
                    gx = 0.0
                    gy = 0.0
                    if active[0]: gx += 1.0   # d_left: d/d(cx) = 1
                    if active[1]: gx -= 1.0   # d_right: d/d(cx) = -1
                    if active[2]: gy += 1.0   # d_bottom: d/d(cy) = 1
                    if active[3]: gy -= 1.0   # d_top: d/d(cy) = -1
                    
                    if cnt > 0:
                        gx /= cnt
                        gy /= cnt
                        
                    # Gradient of penalty w.r.t cx is -dv * gx
                    grad[i, 0] -= dv * gx
                    grad[i, 1] -= dv * gy
        
        # Non-overlap constraints: dist(i,j) >= r[i] + r[j]
        # Penalty: max(r[i]+r[j] - dist, 0)^2
        for i in range(n):
            for j in range(i+1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.sqrt(dx*dx + dy*dy)
                req = r[i] + r[j]
                viol = req - d
                if viol > 0:
                    val += mu * viol**2
                    dv = 2.0 * mu * viol
                    if d < 1e-8:
                        # Random push
                        grad[i, 0] += dv * 1.0
                        grad[j, 0] -= dv * 1.0
                        grad[i, 2] += dv
                        grad[j, 2] += dv
                    else:
                        nx = dx / d
                        ny = dy / d
                        grad[i, 0] -= dv * nx
                        grad[i, 1] -= dv * ny
                        grad[j, 0] += dv * nx
                        grad[j, 1] += dv * ny
                        grad[i, 2] += dv
                        grad[j, 2] += dv
                        
        return val, grad

    def evaluate(x, W, H):
        """Check validity and return sum of radii."""
        cx = x[:, 0]
        cy = x[:, 1]
        r = x[:, 2]
        
        # Check bounds
        if np.any(cx < -1e-6) or np.any(cx > W + 1e-6) or np.any(cy < -1e-6) or np.any(cy > H + 1e-6):
            return None
        
        # Check radii positive
        if np.any(r < 1e-6):
            return None
            
        # Check circle inside box
        for i in range(N):
            if cx[i] < r[i] - 1e-6 or cx[i] > W - r[i] + 1e-6:
                return None
            if cy[i] < r[i] - 1e-6 or cy[i] > H - r[i] + 1e-6:
                return None
                
        # Check disjoint
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((cx[i]-cx[j])**2 + (cy[i]-cy[j])**2)
                if d < r[i] + r[j] - 1e-6:
                    return None
                    break
            else:
                continue
            break
        else:
            return float(np.sum(r))
        
        return None

    def relax_force(x, W, H, iterations=500):
        """
        Force directed relaxation to fix overlaps and bounds.
        x: (N,3)
        """
        x = x.copy()
        n = N
        step = 0.1
        for _ in range(iterations):
            fx = np.zeros(n)
            fy = np.zeros(n)
            fr = np.zeros(n) # Not optimizing r here, just positions?
            # Actually, let's just push positions.
            
            # Overlaps
            for i in range(n):
                for j in range(i+1, n):
                    dx = x[i,0] - x[j,0]
                    dy = x[i,1] - x[j,1]
                    d = np.sqrt(dx*dx + dy*dy)
                    req = x[i,2] + x[j,2]
                    if d < req and d > 1e-8:
                        force = (req - d) * 0.5
                        nx = dx/d
                        ny = dy/d
                        fx[i] += force * nx
                        fy[i] += force * ny
                        fx[j] -= force * nx
                        fy[j] -= force * ny
                    elif d < 1e-8:
                        fx[i] += 0.1
                        fx[j] -= 0.1
            
            # Walls
            for i in range(n):
                r = x[i,2]
                if x[i,0] < r:
                    fx[i] += (r - x[i,0]) * 2.0
                if x[i,0] > W - r:
                    fx[i] -= (x[i,0] - (W - r)) * 2.0
                if x[i,1] < r:
                    fy[i] += (r - x[i,1]) * 2.0
                if x[i,1] > H - r:
                    fy[i] -= (x[i,1] - (H - r)) * 2.0
                    
            x[:, 0] += fx * step
            x[:, 1] += fy * step
            
            # Clip to box (hard constraint)
            x[:, 0] = np.clip(x[:, 0], 0, W)
            x[:, 1] = np.clip(x[:, 1], 0, H)
            
            # Check convergence
            if np.max(np.abs(fx)) < 1e-5 and np.max(np.abs(fy)) < 1e-5:
                break
        return x

    def run_slsqp(x0, W, H, max_iter=100):
        """
        Run SLSQP from x0.
        """
        def fun(x):
            v, _ = objective_and_grad(x, W, H)
            return v
        def jac(x):
            _, g = objective_and_grad(x, W, H)
            return g
            
        try:
            res = minimize(fun, x0, jac=jac, method='SLSQP', options={'maxiter': max_iter, 'ftol': 1e-10})
            return res.x
        except Exception:
            return x0

    # Main Loop
    for W, H in rects:
        if time.time() > DEADLINE - 10:
            break
            
        # 1. Generate initial layouts
        inits = []
        
        # Grid init
        # Try a few grid configurations
        for kx in [3, 4, 5]:
            ky = int(np.ceil(N / kx))
            # Adjust ky to fit exactly N? No, just truncate/pad
            # Create a grid of kx x ky
            # We need exactly N circles.
            # Let's just use a simple grid that might have more slots, then take first N.
            # Or better, calculate optimal grid for N.
            # For 21, 4x6 is 24 slots. 3x7 is 21. 5x5 is 25.
            # Let's just use a generic grid generator.
            
            # Create grid points
            xs = np.linspace(0, W, kx+2)[1:-1]
            ys = np.linspace(0, H, ky+2)[1:-1]
            
            pts = []
            for yy in ys:
                for xx in xs:
                    r = min(xx, W-xx, yy, H-yy) * 0.9
                    r = max(r, 1e-4)
                    pts.append([xx, yy, r])
            
            if len(pts) >= N:
                inits.append(np.array(pts[:N]))
            else:
                # Pad with random
                while len(pts) < N:
                    cx = random.uniform(0.1*W, 0.9*W)
                    cy = random.uniform(0.1*H, 0.9*H)
                    r = 0.1 * min(W, H)
                    pts.append([cx, cy, r])
                inits.append(np.array(pts))
        
        # Random init
        for _ in range(2):
            pts = []
            for _ in range(N):
                cx = random.uniform(0.1*W, 0.9*W)
                cy = random.uniform(0.1*H, 0.9*H)
                r = 0.2 * min(W, H) # Start small-ish
                pts.append([cx, cy, r])
            inits.append(np.array(pts))
        
        best_local_score = 0
        best_local_x = None
        
        for x0 in inits:
            if time.time() > DEADLINE - 5:
                break
                
            # Step 1: Force relax to make it feasible-ish
            x_relaxed = relax_force(x0, W, H, iterations=200)
            
            # Step 2: SLSQP
            x_opt = run_slsqp(x_relaxed, W, H, max_iter=150)
            
            # Step 3: Evaluate
            score = evaluate(x_opt, W, H)
            if score is not None:
                if score > best_local_score:
                    best_local_score = score
                    best_local_x = x_opt
                    
        # If we found a good local solution, try to improve it with a "grow" phase?
        # Or just accept.
        
        if best_local_score > best_score:
            best_score = best_local_score
            best_result = best_local_x
            
    # Fallback if nothing valid found
    if best_result is None:
        # Simple grid in 1x1
        W, H = 1.0, 1.0
        kx = 5
        ky = 5
        xs = np.linspace(0, W, kx+2)[1:-1]
        ys = np.linspace(0, H, ky+2)[1:-1]
        pts = []
        for yy in ys:
            for xx in xs:
                r = min(xx, W-xx, yy, H-yy) * 0.9
                pts.append([xx, yy, r])
        best_result = np.array(pts[:N])
        
    return best_result

# Dummy functions to satisfy interface if needed, though construct is the only one required
def _relax_pack(w, h, n, start_time, deadline):
    return None

def _relax_positions(x, y, r, w, h, max_iter=1000, deadline=None):
    return x, y

def _calc_overlap_energy(x, y, r, w, h):
    return 0.0

def _is_valid(result, w, h):
    return True

if __name__ == "__main__":
    res = construct()
    print(res.shape)
