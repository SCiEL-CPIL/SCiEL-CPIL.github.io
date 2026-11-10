<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    time_limit = 110.0 # 10s buffer
    
    # 1. Generate initial configuration based on known optimal structure
    # Graham et al. 1998: 1 center, ring of 5, ring of 13, ring of 31.
    # We estimate radii for the rings to start near the solution.
    # Optimal r ~ 0.1258
    # Center at (0,0)
    # Ring 1: 5 points. Distance from center ~ 2*r? No, they touch each other and center?
    # In optimal packing, usually neighbors touch.
    # Let's initialize with a hexagonal grid first, then refine, OR use the ring structure.
    # Ring structure is much better for N=50.
    
    def generate_rings():
        centers = []
        # Center
        centers.append([0.0, 0.0])
        
        # Ring 1: 5 points
        # They are arranged in a pentagon.
        # Radius R1.
        # They touch the center? No, usually there's a gap or they touch.
        # Let's assume they touch each other and the center is equidistant.
        # Actually, for N=50, the structure is:
        # 1 center.
        # 5 around it.
        # 13 around that.
        # 31 around that.
        
        # Let's start with estimated radii.
        # r_opt ~ 0.126
        # Center to Ring1: ~ 2*r_opt? 
        # Distance between adjacent in Ring1: 2*r_opt.
        # Chord length 2*r_opt for 5 points on circle R1: 2*R1*sin(pi/5) = 2*r_opt => R1 = r_opt/sin(pi/5)
        
        r_est = 0.125
        R1 = r_est / np.sin(np.pi / 5)
        for i in range(5):
            angle = 2 * np.pi * i / 5 + np.pi/5 # offset
            centers.append([R1 * np.cos(angle), R1 * np.sin(angle)])
            
        # Ring 2: 13 points
        # They touch Ring 1 points and each other.
        # Approx radius R2.
        # Let's just use a hexagonal grid initial for simplicity and let SA fix it.
        # Actually, generating the specific ring structure is tricky to get the angles right without the exact solution.
        # Let's use a robust hexagonal grid initialization which is valid, then optimize.
        # But the ring structure is so much better. 
        # Let's try to build the ring structure more carefully.
        
        # Alternative: Use a dense hex grid, then prune/expand.
        # Let's stick to the hex grid init from the seed but improve the optimization.
        # The seed's hex grid is too sparse.
        
        # Better Init: 
        # Create a hex grid with spacing 2*r_est.
        # Keep points inside 1-r_est.
        # If we have > N, pick the N closest to center.
        # If we have < N, reduce spacing.
        
        return None # We will use the hex init below
    
    # Initialization: Hexagonal Grid
    # Start with a spacing that yields approximately N points.
    # Area of unit circle = pi. Area of hex cell ~ 2.598 * (s/2)^2? No.
    # Hex packing density ~ 0.9069.
    # Number of points ~ Area / Area_per_point.
    # Area_per_point ~ (2*r)^2 * sqrt(3)/2.
    # Let's just iterate spacing.
    
    def get_hex_points(spacing):
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        # Filter points that fit in circle of radius 1 - spacing/2
        # This is a rough estimate for the radius.
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            # Sort by distance to center to get the most central ones
            idx = np.argsort(radius)
            return points[idx[:N]]
        return None

    # Find a good initial spacing
    spacing = 0.25
    best_init = None
    while spacing > 0.05:
        pts = get_hex_points(spacing)
        if pts is not None:
            best_init = pts
            break
        spacing *= 0.95
        
    if best_init is None:
        # Fallback to seed behavior
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                best_init = points[np.argsort(radius, kind="stable")[:N]]
                break
            spacing *= 0.98
            if time.time() - start_time > 5.0:
                break
                
    current_centers = best_init.copy()
    
    # Calculate initial radius
    def get_min_radius(centers):
        min_r = 1.0 - np.max(np.linalg.norm(centers, axis=1))
        # Check pairwise distances
        dists = np.zeros((N, N))
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                dists[i, j] = d
                dists[j, i] = d
        # Minimum distance is 2*r
        min_d = np.min(dists[dists > 0])
        r = min_d / 2.0
        return min(min_r, r)
        
    current_r = get_min_radius(current_centers)
    
    # Simulated Annealing Optimization
    T = 0.01 # Initial temperature
    alpha = 0.9999 # Cooling factor
    # We want to run as many iterations as possible
    
    def step(centers, T):
        # Pick a random center to move
        i = np.random.randint(N)
        # Move it a small amount
        delta = np.random.normal(0, T, 2)
        new_centers = centers.copy()
        new_centers[i] += delta
        
        # Check if new centers are valid (within unit circle)
        # If not, penalty is huge
        new_r = get_min_radius(new_centers)
        old_r = get_min_radius(centers)
        
        # Actually, recalculating full O(N^2) every step is slow.
        # For N=50, 2500 ops. 10k steps = 25M ops. Python loops are slow.
        # We need a faster evaluation or vectorized evaluation.
        
        return new_centers, new_r

    # Faster Evaluation
    def calc_min_radius(centers):
        # Vectorized pairwise distances
        # (N, 1, 2) - (N, 1, 2) doesn't work directly for all pairs.
        # Use broadcasting
        # dists[i, j] = || c_i - c_j ||
        # c_i: (N, 1, 2), c_j: (1, N, 2)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Zero out diagonal
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Boundary constraint
        norms = np.linalg.norm(centers, axis=1)
        min_boundary = 1.0 - np.max(norms)
        
        return min(min_pair_dist / 2.0, min_boundary)

    current_r = calc_min_radius(current_centers)
    
    # SA Loop
    # Estimate time per iteration
    # Vectorized calc_min_radius for N=50 is very fast.
    # Let's try to run for ~100 seconds.
    
    t_start = time.time()
    max_time = time_limit - 5.0 # 5s buffer
    
    # We can also try to increase T if stuck, but simple cooling is usually fine for this size.
    # Let's use a "restart" strategy if no improvement for a long time.
    best_centers = current_centers.copy()
    best_r = current_r
    
    # Number of iterations per second?
    # Let's just run until time runs out.
    
    # To speed up, we can optimize only the "active" set, but N=50 is small.
    
    # One issue: get_min_radius is O(N^2). 
    # 100,000 iterations might be too many in Python if overhead is high.
    # Let's estimate. 10,000 iterations should be safe.
    
    # Let's try a fixed number of iterations based on time.
    # First, measure time for 100 iterations.
    t0 = time.time()
    for _ in range(100):
        i = np.random.randint(N)
        delta = np.random.normal(0, T, 2)
        new_centers = current_centers.copy()
        new_centers[i] += delta
        r = calc_min_radius(new_centers)
        _ = r
    t_elapsed = time.time() - t0
    iters_per_sec = 100.0 / t_elapsed
    total_iters = int(iters_per_sec * (max_time - (time.time() - t_start)))
    
    # Cap total iterations to prevent infinite loop if machine is very fast
    if total_iters > 1000000:
        total_iters = 1000000
        
    for it in range(total_iters):
        if time.time() - t_start > max_time:
            break
            
        i = np.random.randint(N)
        # Move size proportional to T
        # T starts at 0.01, decays.
        # Maybe use a fixed small move size and just accept/reject?
        # Better: Move size ~ T.
        delta = np.random.normal(0, T, 2)
        new_centers = current_centers.copy()
        new_centers[i] += delta
        
        new_r = calc_min_radius(new_centers)
        
        if new_r > current_r:
            current_centers = new_centers
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centers = current_centers.copy()
        else:
            # Accept if delta_r is not too bad
            delta_r = current_r - new_r
            if np.random.rand() < np.exp(-delta_r / (T * 0.1)): # Scale T for acceptance
                current_centers = new_centers
                current_r = new_r
                
        T *= alpha
        if T < 1e-5:
            T = 1e-5 # Floor

    # Final polish: Local gradient descent on the minimum radius?
    # Or just return best found.
    
    # Ensure return type is correct
    return best_centers

# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill

    # --- Initialization: ring structure close to known optimal ---
    def generate_ring_init():
        centers = np.zeros((N, 2))
        # 1 center
        centers[0] = [0.0, 0.0]
        
        r_est = 0.125
        # Ring 1: 5 points
        R1 = r_est / np.sin(np.pi / 5)
        idx = 1
        for i in range(5):
            angle = 2 * np.pi * i / 5 + np.pi / 5
            centers[idx] = [R1 * np.cos(angle), R1 * np.sin(angle)]
            idx += 1
        
        # Ring 2: 13 points
        # Estimate R2: points on ring 2 touch ring 1 points and each other
        # Approximate: R2 ~ R1 + 2*r_est + some adjustment
        R2 = R1 + 2.2 * r_est
        for i in range(13):
            angle = 2 * np.pi * i / 13 + np.pi / 13
            centers[idx] = [R2 * np.cos(angle), R2 * np.sin(angle)]
            idx += 1
        
        # Ring 3: 31 points
        R3 = R2 + 2.2 * r_est
        for i in range(31):
            angle = 2 * np.pi * i / 31 + np.pi / 31
            centers[idx] = [R3 * np.cos(angle), R3 * np.sin(angle)]
            idx += 1
        
        return centers

    current = generate_ring_init()

    # --- Soft-min objective and gradient ---
    def soft_min_value_and_grad(centers, beta):
        """
        Compute soft minimum of all constraints (pairwise distances/2 and boundary distances)
        and its gradient.
        Constraints: for i<j, d_ij/2; for i, 1 - |c_i|
        We want to maximize the minimum constraint = radius r.
        """
        n = len(centers)
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        half_dists = dists / 2.0
        
        # Boundary distances
        norms = np.linalg.norm(centers, axis=1)
        boundary = 1.0 - norms
        
        # Collect all constraint values
        # Pairwise: N*(N-1)/2 values
        iu = np.triu_indices(n, k=1)
        pair_vals = half_dists[iu]  # shape (M,)
        
        # Boundary: N values
        all_vals = np.concatenate([pair_vals, boundary])
        
        # Soft min: -log(sum(exp(-beta * vals))) / beta
        # For numerical stability, subtract max
        max_val = np.max(all_vals)
        exp_vals = np.exp(-beta * (all_vals - max_val))
        sum_exp = np.sum(exp_vals)
        soft_min = max_val - np.log(sum_exp) / beta
        
        # Gradient of soft min w.r.t. centers
        # d(soft_min)/d(vals_k) = exp(-beta*(vals_k - max_val)) / sum_exp
        weights = exp_vals / sum_exp
        
        # Gradient w.r.t. centers
        grad = np.zeros_like(centers)
        
        # Pairwise contribution: d(vals_ij)/d(c_i) = (c_i - c_j)/|c_i - c_j| * 1/2 ... 
        # Actually vals_ij = |c_i - c_j|/2, so d(vals_ij)/d(c_i) = (c_i - c_j)/|c_i - c_j|
        # Wait: d(|c_i - c_j|/2)/d(c_i) = (c_i - c_j) / (2 * |c_i - c_j|) * 2 = (c_i - c_j)/|c_i - c_j|
        # Hmm let me be careful. f = ||c_i - c_j|| / 2. df/dc_i = (c_i - c_j) / ||c_i - c_j||
        # Because d||x||/dx = x/||x||, and the 1/2 factor: d(||x||/2)/dx = (1/2)*(x/||x||) = x/(2||x||)
        # Wait no. Let me redo. f(x) = ||x||/2 where x = c_i - c_j.
        # df/dx = (1/2) * x/||x||
        # So df/dc_i = (1/2) * (c_i - c_j)/||c_i - c_j||
        # df/dc_j = -(1/2) * (c_i - c_j)/||c_i - c_j||
        
        # Boundary: d(1 - ||c_i||)/dc_i = -c_i/||c_i||
        
        # Build gradient
        for k in range(len(pair_vals)):
            i, j = iu[0][k], iu[1][k]
            w = weights[k]
            d = centers[i] - centers[j]
            dist = np.linalg.norm(d)
            if dist < 1e-12:
                continue
            grad[i] += w * d / (2.0 * dist)
            grad[j] -= w * d / (2.0 * dist)
        
        for i in range(n):
            w = weights[len(pair_vals) + i]
            norm_i = norms[i]
            if norm_i < 1e-12:
                continue
            grad[i] += w * (-centers[i] / norm_i)
        
        return soft_min, grad

    # --- Gradient-based optimization with adaptive beta annealing ---
    best_centers = current.copy()
    best_r = _true_radius(current)
    
    def _true_radius(centers):
        """Compute the true (hard) minimum radius."""
        n = len(centers)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        norms = np.linalg.norm(centers, axis=1)
        min_boundary = 1.0 - np.max(norms)
        return min(min_pair, min_boundary)

    beta = 50.0  # initial sharpness
    lr = 0.005   # learning rate
    momentum = np.zeros_like(current)
    v = 0.9      # momentum coefficient
    
    t0 = time.time()
    max_time = deadline - 5.0
    
    # Anneal beta from 50 to ~500 over the run
    n_total_steps = 20000
    step = 0
    
    for step in range(n_total_steps):
        if time.time() - t0 > max_time:
            break
        
        # Progress-based beta annealing
        progress = step / n_total_steps
        beta = 50.0 + 450.0 * progress
        
        # Compute soft min and gradient
        val, grad = soft_min_value_and_grad(current, beta)
        
        # Momentum update
        momentum = v * momentum + (1 - v) * grad
        current = current + lr * momentum
        
        # Project back: ensure no point is outside the unit circle
        norms = np.linalg.norm(current, axis=1)
        for i in range(N):
            if norms[i] > 1.0:
                current[i] *= 1.0 / norms[i]
        
        # Check true radius
        true_r = _true_radius(current)
        if true_r > best_r:
            best_r = true_r
            best_centers = current.copy()
        
        # Occasionally add small random perturbation to escape local minima
        if step % 500 == 0 and step > 0:
            current += np.random.normal(0, 0.002, (N, 2))
            norms = np.linalg.norm(current, axis=1)
            for i in range(N):
                if norms[i] > 1.0:
                    current[i] *= 1.0 / norms[i]
        
        # Decrease learning rate slightly
        lr = max(0.001, 0.005 * (1.0 - progress * 0.5))
    
    # Final: return best found
    return best_centers

# EVOLVE-BLOCK-END
>>>>>>> REPLACE
