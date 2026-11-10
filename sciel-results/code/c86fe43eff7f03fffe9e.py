import numpy as np
import time
from scipy.spatial.distance import cdist

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # --- 1. Initialize with a Ring-based structure ---
    # This is a much better starting point than a hex lattice or random points.
    # We distribute N points into rings.
    
    # Heuristic for number of rings and points per ring
    # Area of unit circle is pi. Area of small circle is pi*r^2.
    # Rough estimate of r ~ 0.05.
    # Let's try to fit rings.
    
    centers = np.zeros((N, 2))
    
    # Simple initialization: 
    # 1 center at origin
    # Remaining N-1 distributed in rings
    # We need to guess the ring radii. 
    # A common packing has rings with roughly 2pi * R / d points.
    
    # Let's use a deterministic initialization that is "almost" optimal for a given radius,
    # then let SA fix the radius.
    
    # Strategy: 
    # Assume a target radius r_target.
    # Fill rings until we have N points.
    
    # Let's start with a standard hexagonal packing as a fallback or better init?
    # The prompt says hex is score 1.0. 
    # Let's try to generate a ring structure directly.
    
    # Estimate r from N. r ~ 1 / sqrt(N) * constant.
    # For N=300, r is around 0.053.
    # Let's pick a few candidate r's and pick the one that fits best in a ring structure?
    # Or just start with a reasonable r and let SA adjust positions.
    
    # Let's initialize with a "spiral" or "rings" that is slightly compressed.
    # Actually, a good init is:
    # 1. Calculate a hex grid for a slightly smaller radius to ensure no overlaps.
    # 2. Assign these to the first N points.
    # 3. The rest of the points (if any) or just the first N are used.
    # Wait, the hex grid code in the prompt is a valid init.
    # Let's use the hex grid code to get a valid initial configuration.
    
    def get_hex_init():
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            mask = radius <= 1.0 - spacing / 2.0
            points = points[mask]
            if len(points) >= N:
                return points[:N].copy()
            spacing *= 0.95
    
    centers = get_hex_init()
    
    # --- 2. Simulated Annealing ---
    
    def calculate_score(pts):
        # Returns the radius r
        # r = min( min_pair_dist / 2, min(1 - |p|) )
        
        # Boundary constraint
        dists_center = np.linalg.norm(pts, axis=1)
        boundary_r = 1.0 - np.max(dists_center)
        
        # Pairwise distances
        # To be fast, we can use cdist
        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)
        
        pair_r = min_pair_dist / 2.0
        
        return min(boundary_r, pair_r)
    
    def calculate_forces(pts, r_target):
        # We want to maximize r.
        # Equivalent to minimizing -r or minimizing overlap.
        # Let's use a potential energy function.
        # E = sum( 1/d^2 for d < 2*r_target ) + penalty for boundary
        
        # Actually, simple gradient-free SA moves are easier.
        # We just need to evaluate the score.
        # But calculating full cdist every step is O(N^2) = 90k ops. 
        # 100k steps * 90k ops = 9e9 ops. Might be too slow in Python/NumPy if not optimized.
        # cdist is C-optimized. 100k calls to cdist(300,300) should be feasible in 140s?
        # Let's check. cdist(300,300) is very fast.
        
        pass # Not using forces, using score directly
    
    # SA Parameters
    T = 0.01  # Initial Temperature
    T_min = 1e-6
    cooling_rate = 0.999
    max_iter = 50000
    
    best_centers = centers.copy()
    best_score = calculate_score(centers)
    current_score = best_score
    current_centers = centers.copy()
    
    # Pre-compute some stuff? No, just loop.
    
    # To speed up, we can batch the cdist calculation? No, it's already vectorized.
    # The bottleneck is Python loop overhead.
    # Let's try to do as many steps as possible.
    
    # Optimization: Instead of full cdist, we can check only local neighbors?
    # Hard to do with arbitrary moves.
    # Let's stick to full cdist but keep the loop tight.
    
    # Better Init: 
    # The hex init is valid. 
    # We can try to "expand" the hex grid slightly? No, that causes overlaps.
    # We can try to "relax" the hex grid.
    
    # Let's run SA.
    for i in range(max_iter):
        if time.time() > deadline:
            break
        
        # Move
        # Randomly pick a few points to move? Or all?
        # Moving all points with small perturbations is often better for global structure.
        # Moving 1 point is better for local refinement.
        
        # Let's try moving a random subset of points (e.g., 10%)
        n_move = max(1, N // 10)
        idx = np.random.choice(N, n_move, replace=False)
        
        # Perturbation magnitude
        # Scale perturbation with temperature
        scale = T * 0.1
        perturbations = np.random.normal(0, scale, size=(n_move, 2))
        
        new_centers = current_centers.copy()
        new_centers[idx] += perturbations
        
        # Keep inside circle? 
        # If we push out, the penalty (boundary_r) will drop, so score drops.
        # We don't strictly need to clamp, the score function handles it.
        # But clamping might help convergence.
        # Let's clamp to unit circle to avoid huge negative scores.
        norms = np.linalg.norm(new_centers[idx], axis=1)
        mask_out = norms > 1.0
        if np.any(mask_out):
            new_centers[idx[mask_out]] /= norms[mask_out, np.newaxis]
        
        new_score = calculate_score(new_centers)
        
        # Acceptance
        delta = new_score - current_score
        if delta > 0:
            current_centers = new_centers
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_centers = current_centers.copy()
        else:
            # Accept with probability exp(delta / T)
            # delta is negative
            prob = np.exp(delta / T)
            if np.random.random() < prob:
                current_centers = new_centers
                current_score = new_score
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        # Occasionally reheat or restart?
        # If no improvement for a long time, maybe restart from best with higher T?
        # For now, simple cooling.
        
    # Final Polish:
    # Use a local optimizer on the best found?
    # We can do a few steps of "steepest ascent" on the min-distance.
    # Or just return the best SA result.
    
    return best_centers

# Test locally? No, just output.
