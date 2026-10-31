import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # 100s limit, leaving 20s margin
    
    best_score = 0.0
    best_solution = None
    
    # Variables: [W, x1, y1, r1, ..., x21, y21, r21]
    # H is derived as 2 - W
    # Perimeter constraint: 2(W+H) <= 4 => W+H <= 2. We set W+H=2 for max area.
    
    def unpack(v):
        W = v[0]
        H = 2.0 - W
        # Ensure W, H are positive
        if W <= 0 or H <= 0:
            return None
        xs = v[1:1+3*N:3]
        ys = v[2:1+3*N:3]
        rs = v[3:1+3*N:3]
        return W, H, xs, ys, rs

    def objective(v):
        unpacked = unpack(v)
        if unpacked is None:
            return 1e6
        W, H, xs, ys, rs = unpacked
        
        # Penalties for negative radii or too large radii
        if np.any(rs <= 0) or np.any(rs > 1):
            return 1e6
            
        # Wall penalties
        # Circle i must be inside [0, W] x [0, H]
        # dist to left wall: xs[i] - rs[i]
        # dist to right wall: W - xs[i] - rs[i]
        # dist to bottom wall: ys[i] - rs[i]
        # dist to top wall: H - ys[i] - rs[i]
        
        # If any distance < 0, penalty is square of the violation
        wall_pen = 0.0
        for i in range(N):
            d_left = xs[i] - rs[i]
            d_right = W - xs[i] - rs[i]
            d_bottom = ys[i] - rs[i]
            d_top = H - ys[i] - rs[i]
            
            if d_left < 0: wall_pen += d_left**2
            if d_right < 0: wall_pen += d_right**2
            if d_bottom < 0: wall_pen += d_bottom**2
            if d_top < 0: wall_pen += d_top**2
            
        # Overlap penalties
        overlap_pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dist_sq = (xs[i]-xs[j])**2 + (ys[i]-ys[j])**2
                dist = np.sqrt(dist_sq)
                min_dist = rs[i] + rs[j]
                if dist < min_dist:
                    overlap_pen += (min_dist - dist)**2
                    
        # Objective: Minimize -sum(r) + lambda * (wall_pen + overlap_pen)
        # We want to maximize sum(r), so minimize -sum(r).
        # Penalty weight lambda should be large enough to enforce constraints.
        # Since radii are small (sum < 2), penalties are small if close to valid.
        # Let's use a large lambda.
        lambda_w = 100.0
        score = -np.sum(rs) + lambda_w * (wall_pen + overlap_pen)
        
        return score

    def is_valid(v):
        unpacked = unpack(v)
        if unpacked is None:
            return False
        W, H, xs, ys, rs = unpacked
        if np.any(rs <= 0):
            return False
        for i in range(N):
            if xs[i] < rs[i] or xs[i] > W - rs[i]:
                return False
            if ys[i] < rs[i] or ys[i] > H - rs[i]:
                return False
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                if dist < rs[i] + rs[j]:
                    return False
        return True

    def get_score(v):
        if not is_valid(v):
            return -1e9
        unpacked = unpack(v)
        W, H, xs, ys, rs = unpacked
        return np.sum(rs)

    # Initializations
    inits = []
    
    # 1. Grid initialization in 1x1 square
    side = int(np.ceil(np.sqrt(N)))
    for i in range(side):
        for j in range(side):
            if len(inits) >= N: break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.01
            inits.append([x, y, r])
    # Pad if necessary
    while len(inits) < N:
        inits.append([0.5, 0.5, 0.01])
        
    # Convert to vector format: [W, x1, y1, r1, ...]
    # Start with W=1.0
    v0 = [1.0]
    for x, y, r in inits:
        v0.extend([x, y, r])
    
    # 2. Random initialization
    np.random.seed(42)
    v_rand = [1.0]
    for _ in range(N):
        x = np.random.uniform(0.1, 0.9)
        y = np.random.uniform(0.1, 0.9)
        r = 0.01
        v_rand.extend([x, y, r])
        
    # Add more random inits
    for seed in range(5):
        np.random.seed(seed)
        v_r = [1.0]
        for _ in range(N):
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            r = 0.01
            v_r.extend([x, y, r])
        inits_vectors = [v0, v_rand] + [v_r for seed in range(1, 6)]
        
    # Optimization loop
    for init_v in inits_vectors:
        if time.time() > deadline:
            break
            
        # Nelder-Mead is good for non-differentiable penalty functions
        res = minimize(objective, init_v, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if res.success or res.fun < 1e5:
            # Check validity
            if is_valid(res.x):
                score = get_score(res.x)
                if score > best_score:
                    best_score = score
                    best_solution = res.x.copy()
                    
    # If we found a valid solution, format it
    if best_solution is not None:
        unpacked = unpack(best_solution)
        W, H, xs, ys, rs = unpacked
        # Construct output array (n, 3)
        out = np.zeros((N, 3))
        for i in range(N):
            out[i] = [xs[i], ys[i], rs[i]]
        return out
    else:
        # Fallback to seed-like solution if optimization fails completely
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

# EVOLVE-BLOCK-END
