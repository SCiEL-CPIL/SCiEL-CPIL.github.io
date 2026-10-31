import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial configuration: Hexagonal lattice
    # 1 center, 6 ring 1, 4 from ring 2
    # Scale factor: min dist must be sqrt(3)
    # In lattice units (dist 1), we need to scale by sqrt(3)
    # But we are optimizing in real coordinates.
    
    # Let's start with a configuration that satisfies constraints
    # 1 at (0,0)
    # 6 at dist sqrt(3)
    # 4 at dist sqrt(3)*sqrt(3) = 3? No.
    
    # Better: Start with a valid hexagonal cluster of 7, and push 4 out.
    # Or just start with a random spread and let the optimizer fix it.
    
    # Let's use a heuristic initialization:
    # 7 points in a hexagon of radius r1
    # 4 points in a hexagon of radius r2
    
    # Min dist d = sqrt(3)
    d = np.sqrt(3)
    
    # Init 11 points
    # 1 at 0
    # 6 at d
    # 4 at 2*d (approx)
    
    pts = np.zeros((N, 2))
    # Center
    pts[0] = [0, 0]
    
    # Ring 1
    for i in range(6):
        angle = i * np.pi / 3
        pts[i+1] = [d * np.cos(angle), d * np.sin(angle)]
        
    # Ring 2 (pick 4)
    # Place them between the ring 1 points to maximize spacing?
    # Or just at angles 0, 90, 180, 270 at distance 2*d
    angles = [0, np.pi/2, np.pi, 3*np.pi/2]
    for i in range(4):
        pts[i+7] = [2*d * np.cos(angles[i]), 2*d * np.sin(angles[i])]
        
    # Optimize
    # We want to minimize max norm, subject to min pairwise dist >= d
    
    # Simple local search:
    # Randomly perturb, accept if better
    
    best_pts = pts.copy()
    best_score = np.max(np.linalg.norm(best_pts, axis=1))
    
    # Penalty for constraint violation
    def get_cost(p):
        # Max dist
        max_d = np.max(np.linalg.norm(p, axis=1))
        # Min pairwise dist
        # Calculate all pairwise distances
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Mask diagonal
        np.fill_diagonal(dists, np.inf)
        min_d = np.min(dists)
        
        # Violation
        viol = max(0, d - min_d)
        return max_d + 100 * viol * viol, viol

    current_cost, _ = get_cost(best_pts)
    
    # Nelder-Mead or similar?
    # Let's use a simple random hill climbing with shrinking step size
    
    step_size = 0.5
    iterations = 0
    max_iterations = 50000
    
    while time.time() < deadline and iterations < max_iterations:
        iterations += 1
        # Pick a random point to move
        idx = np.random.randint(N)
        
        # Try moving it
        new_pts = best_pts.copy()
        # Random direction
        angle = np.random.uniform(0, 2*np.pi)
        # Random magnitude
        mag = np.random.uniform(0, step_size)
        
        new_pts[idx] += [mag * np.cos(angle), mag * np.sin(angle)]
        
        new_cost, new_viol = get_cost(new_pts)
        
        if new_cost < current_cost:
            best_pts = new_pts
            current_cost = new_cost
            best_score = np.max(np.linalg.norm(best_pts, axis=1))
            # Reduce step size slightly on success? Or keep it?
            # Let's keep it and reduce globally every 1000 iters
            if iterations % 1000 == 0:
                step_size *= 0.95
        else:
            # If stuck, maybe reduce step size?
            if iterations % 100 == 0:
                step_size *= 0.99

    # Final S
    S = best_score + 1.0
    
    # Format output
    inner = []
    for p in best_pts:
        inner.append([float(p[0]), float(p[1]), 0.0])
        
    return inner, (0.0, 0.0), float(S), 0.0
# EVOLVE-BLOCK-END
