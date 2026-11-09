import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 200

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin for 150s limit
    
    # 1. Generate a high-quality initial configuration using a distorted hexagonal lattice
    # This avoids random initialization pitfalls and starts near the theoretical optimum.
    
    def generate_hex_packing(n):
        # Approximate grid size
        # Hex packing density is ~0.9069
        # Area per circle ~ pi * r^2. 
        # For N circles in area 1, r ~ 0.0366 (from task description)
        # dx ~ 2r ~ 0.073
        # cols ~ 1 / 0.073 ~ 13-14
        # Let's try a grid that generates slightly more than N points, 
        # then we will select the best N or optimize all if we generate exactly N.
        # Actually, for optimization, we need exactly N points.
        # We can generate a dense hex grid, and if we have more than N, 
        # we can't just drop them easily without breaking symmetry.
        # Instead, we will generate a hex grid with exactly N points by 
        # iterating rows and taking the first N points, then centering.
        
        pts = []
        # Hex spacing
        # We want to fit N points.
        # Let's estimate rows.
        # If we use a square grid of size k x k, we have k^2 points.
        # sqrt(200) ~ 14.14. So 14x14 = 196. Close.
        # Hex grid:
        # Row 0: 14 points
        # Row 1: 14 points (shifted)
        # ...
        # Let's just generate a large hex grid and take the first N.
        # To make it look like a "packing", we should probably center the cluster.
        
        # Better approach: 
        # Generate points on a hex lattice covering [-0.6, 0.6] x [-0.6, 0.6]
        # Then select the N points that are closest to the origin? 
        # No, we want them to fill the square [-0.5, 0.5].
        
        # Let's use a parameterized hex grid.
        # dx = 1.0 / k
        # dy = dx * sqrt(3) / 2
        # We want to cover the square.
        # Let's try k=14.
        # dx = 1/14 ~ 0.0714
        # dy = 0.0618
        # Height of 14 rows = 13 * dy ~ 0.8. Fits in 1.0.
        # Width of 14 cols = 13 * dx + dx/2 (shift) ~ 1.0. Fits in 1.0.
        
        k = 14
        dx = 1.0 / k
        dy = dx * np.sqrt(3) / 2.0
        
        all_pts = []
        # Generate enough points to cover the square
        # x from -0.5 - dx to 0.5 + dx
        # y from -0.5 - dy to 0.5 + dy
        
        i = 0
        y = -0.5 - dy
        while y < 0.5 + dy:
            x_start = -0.5 - dx
            if i % 2 == 0:
                x_start = -0.5 - dx
            else:
                x_start = -0.5 - dx + dx/2.0
            
            j = 0
            x = x_start
            while x < 0.5 + dx:
                # Only keep points strictly inside [-0.5, 0.5] with a small margin?
                # Or keep all and let optimization push them in?
                # Let's keep all and center the set.
                all_pts.append([x, y])
                x += dx
                j += 1
            y += dy
            i += 1
        
        all_pts = np.array(all_pts)
        
        # We have ~200-250 points.
        # We need exactly N.
        # Strategy: Select the N points that form the most "compact" cluster?
        # Or simply take the first N and center them?
        # Taking the first N creates a rectangular block.
        
        # Alternative: 
        # Use the center of the square as reference.
        # Sort points by distance to center? No, that leaves a hole in the middle.
        
        # Let's try a different construction:
        # Start with a square grid of 14x14 = 196 points.
        # Add 4 points?
        # Or just use 14x14 square grid and optimize.
        # The hex grid is denser.
        
        # Let's stick to the hex grid but select N points randomly?
        # No, deterministic is better for reproducibility and stability.
        
        # Let's generate a hex grid that is exactly N points by adjusting k.
        # For k=14, we have 14*14 = 196 points in a rectangular block if we just take k rows of k cols.
        # Let's generate k rows of k cols in a hex pattern.
        
        pts = []
        for i in range(k):
            for j in range(k):
                x = j * dx + (i % 2) * dx / 2.0
                y = i * dy
                pts.append([x, y])
        
        pts = np.array(pts)
        
        # Center the points
        center = pts.mean(axis=0)
        pts -= center
        
        # We have 196 points. We need 200.
        # We can add 4 points in the "gaps" or just expand the grid slightly?
        # If we use k=14, we have 196.
        # If we use k=15, we have 225.
        # Let's use k=15 and select 200?
        # Or use k=14 and add 4 random points near the boundary?
        
        # Let's use k=15.
        k = 15
        dx = 1.0 / k
        dy = dx * np.sqrt(3) / 2.0
        pts = []
        for i in range(k):
            for j in range(k):
                x = j * dx + (i % 2) * dx / 2.0
                y = i * dy
                pts.append([x, y])
        pts = np.array(pts)
        center = pts.mean(axis=0)
        pts -= center
        
        # We have 225 points.
        # We need to remove 25 points.
        # Removing points from the corners might be good as they are less "interior".
        # Let's sort by distance to center and remove the outermost 25?
        # That would create a circular shape, which is bad for a square.
        
        # Better: Remove points that are "least supported" by neighbors?
        # Or just remove random points?
        # Let's try to keep the structure as square as possible.
        # The hex grid is rectangular.
        # Let's just take the first 200? No, that's a rectangle.
        
        # Let's try a different initialization: 
        # Random points inside the square, but clustered?
        # No, let's use the hex grid and just optimize all 225 points, 
        # then take the 200 with the largest "local" radius?
        # That's complex.
        
        # Simplest robust approach:
        # Generate a square grid of 14x14 = 196.
        # Add 4 points at the midpoints of the sides?
        # This gives a nice symmetric start.
        
        # Let's go with 14x14 square grid + 4 extra points.
        k = 14
        dx = 1.0 / (k - 1) # Spacing so that points span [-0.5, 0.5]
        # Actually, if we want to maximize radius, we want points as spread out as possible.
        # Square grid:
        xs = np.linspace(-0.45, 0.45, k)
        ys = np.linspace(-0.45, 0.45, k)
        xx, yy = np.meshgrid(xs, ys)
        pts = np.column_stack([xx.ravel(), yy.ravel()])
        
        # We have 196 points.
        # Add 4 points. Where?
        # Maybe in the center of the 4 quadrants?
        # Or just use 15x15 = 225 and randomly drop 25?
        # Let's use 15x15 and drop 25 from the corners to maintain symmetry.
        
        k = 15
        xs = np.linspace(-0.45, 0.45, k)
        ys = np.linspace(-0.45, 0.45, k)
        xx, yy = np.meshgrid(xs, ys)
        pts = np.column_stack([xx.ravel(), yy.ravel()])
        
        # We have 225 points.
        # Sort by distance to center? No.
        # Let's just use the first 200? 
        # To make it deterministic and symmetric, let's remove points from the outer ring?
        # The outer ring has 4 * (15 - 2) = 52 points.
        # We need to remove 25.
        # Let's remove 25 points randomly from the outer ring?
        # Or just take a subset that is centered.
        
        # Actually, a 14x14 grid (196) + 4 points is cleaner.
        # Let's stick to 14x14 square grid. It's a valid initial guess.
        # The optimizer will move them.
        
        k = 14
        xs = np.linspace(-0.4, 0.4, k)
        ys = np.linspace(-0.4, 0.4, k)
        xx, yy = np.meshgrid(xs, ys)
        pts = np.column_stack([xx.ravel(), yy.ravel()])
        
        # Add 4 points to make 200.
        # Place them in the gaps of the 2x2 blocks in the center?
        # Or just add 4 random points?
        # Let's add 4 points at the midpoints of the edges of the central 2x2 block?
        # This is getting complicated.
        
        # Let's just use a random initialization with a fixed seed. 
        # It's simpler and often works well with DE.
        rng = np.random.RandomState(42)
        pts = rng.uniform(-0.4, 0.4, (N, 2))
        
        return pts

    # 2. Objective Function
    # Minimize sum of inverse powers of distances.
    # This is a smooth approximation of maximizing the minimum distance.
    
    p = 6.0 # Higher p makes it more like a min-distance problem
    eps = 1e-6
    
    def objective(x_flat):
        x = x_flat.reshape(-1, 2)
        
        # Pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        mask = ~np.eye(N, dtype=bool)
        dist_sq = dist_sq[mask]
        
        # Avoid division by zero
        dist_sq = np.maximum(dist_sq, eps**2)
        
        # Sum of inverse powers
        pair_energy = np.sum(dist_sq ** (-p/2.0))
        
        # Wall energy
        # Keep points inside [-0.5, 0.5]
        wx_neg = np.maximum(0.5 - x[:, 0], eps)
        wx_pos = np.maximum(0.5 + x[:, 0], eps)
        wy_neg = np.maximum(0.5 - x[:, 1], eps)
        wy_pos = np.maximum(0.5 + x[:, 1], eps)
        
        wall_energy = np.sum(wx_neg ** (-p) + wx_pos ** (-p) + wy_neg ** (-p) + wy_pos ** (-p))
        
        return pair_energy + wall_energy

    def gradient(x_flat):
        x = x_flat.reshape(-1, 2)
        grad = np.zeros_like(x)
        
        # Pairwise gradient
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        safe_dist_sq = np.maximum(dist_sq, eps**2)
        
        factor = safe_dist_sq ** (-p/2.0 - 1.0)
        
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        factor = np.where(mask, factor, 0.0)
        
        pair_grad = -p * np.sum(diff * factor[:, :, np.newaxis], axis=1)
        
        # Wall gradient
        wx_neg = np.maximum(0.5 - x[:, 0], eps)
        wx_pos = np.maximum(0.5 + x[:, 0], eps)
        wy_neg = np.maximum(0.5 - x[:, 1], eps)
        wy_pos = np.maximum(0.5 + x[:, 1], eps)
        
        grad[:, 0] += pair_grad[:, 0] + p * (wx_neg ** (-p - 1.0)) - p * (wx_pos ** (-p - 1.0))
        grad[:, 1] += pair_grad[:, 1] + p * (wy_neg ** (-p - 1.0)) - p * (wy_pos ** (-p - 1.0))
        
        return grad.flatten()

    # 3. Optimization Strategy
    # Start with a hex-like grid initialization.
    # We will use the 14x14 square grid + 4 points idea but implemented simply.
    
    # Let's use a hex grid initialization as planned.
    # Generate 15x15 hex grid, take first 200? No, let's just use random init for DE.
    # DE is good at global search.
    
    x0 = np.random.RandomState(42).uniform(-0.4, 0.4, (N, 2)).flatten()
    
    # Run Differential Evolution
    # DE is stochastic, so we use a fixed seed for reproducibility in this run.
    # Bounds: [-0.5, 0.5]
    bounds = [(-0.45, 0.45)] * (2 * N) # Slightly inside to allow DE to explore
    
    # Time budget check
    if time.time() - start_time > timeout:
        return x0.reshape(-1, 2)
        
    try:
        # DE options
        # maxiter: 100
        # tol: 1e-4
        # popsize: 15
        # We need to be careful with time.
        # DE on 400 variables is expensive.
        # Let's limit maxiter.
        
        res_de = differential_evolution(
            objective, 
            bounds, 
            seed=42,
            maxiter=50, 
            tol=1e-4,
            popsize=10,
            polish=False # We will polish manually
        )
        
        x_best = res_de.x
        f_best = res_de.fun
        
    except Exception as e:
        x_best = x0
        f_best = objective(x0)
    
    # Check time
    if time.time() - start_time > timeout - 20:
        # If not enough time left for polishing, return DE result
        x_final = x_best.reshape(-1, 2)
        x_final = np.clip(x_final, -0.499999, 0.499999)
        return x_final
    
    # Polish with L-BFGS-B
    # Use the result from DE as start for L-BFGS-B
    try:
        res_lbfgs = minimize(
            objective, 
            x_best, 
            jac=gradient, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if res_lbfgs.fun < f_best:
            x_best = res_lbfgs.x
            f_best = res_lbfgs.fun
            
    except Exception:
        pass
    
    # Final cleanup
    x_final = x_best.reshape(-1, 2)
    x_final = np.clip(x_final, -0.499999, 0.499999)
    
    return x_final
