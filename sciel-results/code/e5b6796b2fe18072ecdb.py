import numpy as np
import time
import random

N = 21
PERIMETER = 4.0
DEADLINE_FRACTION = 0.90  # Safety margin

def construct():
    t0 = time.time()
    deadline = 120.0 * DEADLINE_FRACTION
    
    best_val = -1.0
    best_circles = None
    
    # Aspect ratios to try: W + H = 2. W in [0.5, 1.5]
    # The optimal is likely near 1.0 (square) but we will scan.
    aspect_ratios = np.linspace(0.6, 1.4, 17)
    
    # Shuffle to not always start with the same one, though order doesn't matter much if we track best
    # We will run multiple random restarts for each aspect ratio if time permits.
    
    for W in aspect_ratios:
        if time.time() - t0 > deadline:
            break
        H = 2.0 - W
        
        # Try a few random restarts for this aspect ratio
        # Budget per aspect ratio depends on remaining time
        # Let's allocate a fixed number of restarts, say 3, but check time.
        for restart in range(3):
            if time.time() - t0 > deadline:
                break
            
            circles = _grow_and_relax(W, H, t0, deadline)
            if circles is None:
                continue
            
            # circles is list of (x, y, r)
            # Scale to perimeter 4
            minx = min(c[0] - c[2] for c in circles)
            miny = min(c[1] - c[2] for c in circles)
            maxx = max(c[0] + c[2] for c in circles)
            maxy = max(c[1] + c[2] for c in circles)
            
            bw = maxx - minx
            bh = maxy - miny
            per = 2.0 * (bw + bh)
            
            if per <= 0:
                continue
                
            s = 4.0 / per
            scaled_circles = []
            for (x, y, r) in circles:
                nx = (x - minx) * s
                ny = (y - miny) * s
                nr = r * s
                scaled_circles.append((nx, ny, nr))
            
            # Validate
            if _is_valid(scaled_circles, 2.0, 2.0):
                val = sum(c[2] for c in scaled_circles)
                if val > best_val:
                    best_val = val
                    best_circles = scaled_circles
                    
    if best_circles is None:
        # Fallback: simple grid
        # 5x5 grid is 25, take 21.
        # Or 4x5 = 20, add 1? 
        # Let's just use a standard grid fallback that is definitely valid.
        side = 5
        # Create 25 points, take first 21
        pts = []
        for i in range(side):
            for j in range(side):
                if len(pts) < N:
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    r = 0.5 / side - 0.01
                    pts.append((x, y, r))
        best_circles = pts
        # Scale
        minx = min(c[0] - c[2] for c in best_circles)
        miny = min(c[1] - c[2] for c in best_circles)
        maxx = max(c[0] + c[2] for c in best_circles)
        maxy = max(c[1] + c[2] for c in best_circles)
        bw = maxx - minx
        bh = maxy - miny
        per = 2.0 * (bw + bh)
        s = 4.0 / per
        best_circles = [((x-minx)*s, (y-miny)*s, r*s) for (x,y,r) in best_circles]
        
    return np.array(best_circles)

def _is_valid(circles, W, H, tol=1e-6):
    n = len(circles)
    if n != N:
        return False
    # Check bounds
    for x, y, r in circles:
        if x - r < -tol or x + r > W + tol or y - r < -tol or y + r > H + tol:
            return False
    # Check overlaps
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            d = np.hypot(x1-x2, y1-y2)
            if d < r1 + r2 - tol:
                return False
    return True

def _grow_and_relax(W, H, t0, deadline):
    """
    Initializes small circles, grows them, and relaxes positions to avoid overlap.
    Returns list of (x, y, r) or None if failed.
    """
    # Initialize
    rng = np.random.default_rng()
    x = rng.uniform(0.1*W, 0.9*W, N)
    y = rng.uniform(0.1*H, 0.9*H, N)
    r = np.full(N, 0.01) # Start small
    
    dt = 0.05 # Learning rate for position update
    dr = 0.005 # Radius growth step
    max_iterations = 5000
    relax_steps = 20 # Number of relaxation steps per growth step
    
    for i_iter in range(max_iterations):
        if time.time() - t0 > deadline:
            return None
            
        # 1. Grow radii
        r += dr
        
        # 2. Relax positions
        # We minimize E = sum(overlap^2) + boundary_penalty
        # Using simple gradient descent on positions
        
        # Precompute pairs? No, N is small (21). O(N^2) is fine.
        
        for step in range(relax_steps):
            if time.time() - t0 > deadline:
                return None
                
            dx = np.zeros(N)
            dy = np.zeros(N)
            
            # Overlap forces
            for i in range(N):
                for j in range(i+1, N):
                    d = np.hypot(x[i]-x[j], y[i]-y[j])
                    if d < 1e-9:
                        # Random nudge
                        dx[i] += rng.uniform(-1, 1)
                        dy[i] += rng.uniform(-1, 1)
                        dx[j] -= dx[i]
                        dy[j] -= dy[i]
                        continue
                    
                    overlap = (r[i] + r[j]) - d
                    if overlap > 0:
                        # Force is proportional to overlap
                        # Direction from j to i
                        fx = (x[i] - x[j]) / d
                        fy = (y[i] - y[j]) / d
                        force = overlap * 2.0 # Gradient of 0.5 * overlap^2 is overlap
                        dx[i] += fx * force
                        dy[i] += fy * force
                        dx[j] -= fx * force
                        dy[j] -= fy * force
            
            # Boundary forces
            # Push back if x-r < 0 or x+r > W
            for i in range(N):
                if x[i] - r[i] < 0:
                    dx[i] += (x[i] - r[i]) * 5.0 # Strong push
                if x[i] + r[i] > W:
                    dx[i] -= (x[i] + r[i] - W) * 5.0
                if y[i] - r[i] < 0:
                    dy[i] += (y[i] - r[i]) * 5.0
                if y[i] + r[i] > H:
                    dy[i] -= (y[i] + r[i] - H) * 5.0
            
            # Update positions
            x += dx * dt
            y += dy * dt
            
            # Hard clamp to keep them roughly in bounds during relaxation
            # This helps stability
            x = np.clip(x, r, W - r)
            y = np.clip(y, r, H - r)
            
            # Check if converged (forces are small)
            # If max force is small, we can break early from relaxation loop
            # But we need to be careful not to break too early.
            # Let's just run fixed steps for simplicity, or check overlap sum.
            # Total overlap
            total_overlap = 0
            for i in range(N):
                for j in range(i+1, N):
                    d = np.hypot(x[i]-x[j], y[i]-y[j])
                    ov = (r[i] + r[j]) - d
                    if ov > 0:
                        total_overlap += ov * ov
            if total_overlap < 1e-6:
                break
        
        # Check if the configuration is valid enough to continue growing
        # If overlaps are too large, we might be stuck.
        # We can check the total overlap penalty.
        total_overlap = 0
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(x[i]-x[j], y[i]-y[j])
                ov = (r[i] + r[j]) - d
                if ov > 0:
                    total_overlap += ov
        
        # If total overlap is significant, we might have grown too fast or hit a local min.
        # In this simple scheme, we just continue, hoping relaxation fixes it.
        # However, if circles are forced out of bounds significantly, we might fail.
        # The hard clamp above prevents going out, but might cause high overlap.
        
        # Simple stopping criterion: if we can't reduce overlap below a threshold, stop growing?
        # For now, just run all iterations or until time limit.
        # The final result will be scaled. If it's invalid, _is_valid will reject it.
        
    # Final cleanup: ensure circles are strictly inside
    # The hard clamp did this.
    
    # Convert to list of tuples
    res = []
    for i in range(N):
        res.append((x[i], y[i], r[i]))
        
    return res

if __name__ == "__main__":
    res = construct()
    print(res.shape)
    print(res[:3])
