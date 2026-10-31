import numpy as np
import random

N = 21

def construct():
    random.seed(42)
    np.random.seed(42)
    
    def get_cost(state):
        """
        state: [w, h, x1, y1, r1, ...]
        Returns cost (lower is better).
        If valid, cost = -sum(r).
        If invalid, cost = 1e6 + penalty.
        """
        w = state[0]
        h = state[1]
        
        # Enforce perimeter
        if w + h > 2.0:
            scale = 2.0 / (w + h)
            w *= scale
            h *= scale
            
        circles = state[2:].reshape((N, 3))
        xs = circles[:, 0]
        ys = circles[:, 1]
        rs = circles[:, 2]
        
        # Boundary check
        # Vectorized
        left_viol = np.maximum(0, rs - xs)
        right_viol = np.maximum(0, xs - (w - rs))
        bottom_viol = np.maximum(0, rs - ys)
        top_viol = np.maximum(0, ys - (h - rs))
        
        boundary_penalty = np.sum(left_viol) + np.sum(right_viol) + np.sum(bottom_viol) + np.sum(top_viol)
        
        if boundary_penalty > 1e-9:
            return 1e6 + boundary_penalty
        
        # Overlap check
        # N=21, so 210 pairs. Vectorized pairwise distance is efficient.
        # Create distance matrix
        # dist[i,j] = sqrt((xi-xj)^2 + (yi-yj)^2)
        # We need dist[i,j] >= ri + rj
        
        # Using scipy.spatial.distance.cdist might be overkill or slow if imported each time? No, import once.
        # But we can just do a simple loop or numpy broadcasting.
        
        # Numpy broadcasting for 21x21 is fast.
        # dx = xs[:, None] - xs[None, :]
        # dy = ys[:, None] - ys[None, :]
        # dist = np.sqrt(dx**2 + dy**2)
        # min_dist = rs[:, None] + rs[None, :]
        # overlap = np.maximum(0, min_dist - dist)
        # overlap_penalty = np.sum(overlap) # Sum of all violations
        
        # To avoid double counting and diagonal, we can just sum the upper triangle.
        # Or just sum all, it's symmetric.
        
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        
        min_dist = rs[:, None] + rs[None, :]
        
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        
        overlap = np.where(mask, np.maximum(0, min_dist - dist), 0)
        overlap_penalty = np.sum(overlap)
        
        if overlap_penalty > 1e-9:
            return 1e6 + overlap_penalty
            
        return -np.sum(rs)

    def simulate_annealing():
        # Initialize
        # Start with a square-ish rectangle
        w = 1.0 + random.uniform(-0.2, 0.2)
        h = 2.0 - w
        
        # Initialize circles
        # Small radii, random positions
        state = [w, h]
        for _ in range(N):
            x = random.uniform(0, w)
            y = random.uniform(0, h)
            r = random.uniform(0.02, 0.05)
            state.extend([x, y, r])
            
        state = np.array(state)
        
        current_cost = get_cost(state)
        best_state = state.copy()
        best_cost = current_cost
        
        T = 1.0
        T_min = 1e-4
        cooling_rate = 0.995
        max_iters = 200000 # Adjust based on time
        
        # Estimate time per iteration to fit in 120s?
        # 200k iterations might be too many if get_cost is slow.
        # Let's do 50k iterations and check.
        
        for i in range(50000):
            # Pick random circle
            idx = random.randint(0, N-1)
            
            # Perturb x, y, r
            # Step size depends on T
            step = 0.05 * T + 0.01
            
            new_state = state.copy()
            
            # Perturb position
            new_state[2 + 3*idx] += random.gauss(0, step) # x
            new_state[2 + 3*idx + 1] += random.gauss(0, step) # y
            
            # Perturb radius
            # Bias towards growing?
            # If we just perturb, it might shrink.
            # Let's do a random walk on r.
            new_state[2 + 3*idx + 2] += random.gauss(0, step * 0.5)
            
            # Ensure r > 0
            if new_state[2 + 3*idx + 2] < 0.01:
                new_state[2 + 3*idx + 2] = 0.01
                
            # Optionally perturb w, h occasionally
            if random.random() < 0.05:
                new_state[0] += random.gauss(0, step)
                new_state[1] += random.gauss(0, step)
                
            new_cost = get_cost(new_state)
            
            # Acceptance criterion
            delta = new_cost - current_cost
            if delta < 0 or random.random() < np.exp(-delta / T):
                state = new_state
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_state = state.copy()
                    
            T *= cooling_rate
            if T < T_min:
                break
                
        return best_state, best_cost

    best_state, best_cost = simulate_annealing()
    
    # Extract result
    w = best_state[0]
    h = best_state[1]
    if w + h > 2.0:
        scale = 2.0 / (w + h)
        w *= scale
        h *= scale
        # Scale circles
        circles = best_state[2:].reshape((N, 3))
        circles[:, 0] *= scale
        circles[:, 1] *= scale
        circles[:, 2] *= scale
    else:
        circles = best_state[2:].reshape((N, 3))
        
    # Final cleanup: clamp and resolve small overlaps
    # This is crucial to ensure validity
    for _ in range(10):
        valid = True
        for i in range(N):
            cx, cy, cr = circles[i]
            if cr < 1e-6: cr = 1e-6
            if cx < cr: cx = cr
            if cx > w - cr: cx = w - cr
            if cy < cr: cy = cr
            if cy > h - cr: cy = h - cr
            circles[i] = [cx, cy, cr]
            
        for i in range(N):
            for j in range(i+1, N):
                dx = circles[i,0] - circles[j,0]
                dy = circles[i,1] - circles[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_d = circles[i,2] + circles[j,2]
                if dist < min_d:
                    # Shrink
                    excess = min_d - dist
                    shrink = 0.5 * excess + 1e-5
                    circles[i,2] -= shrink
                    circles[j,2] -= shrink
                    valid = False
        if valid:
            break
            
    return circles
