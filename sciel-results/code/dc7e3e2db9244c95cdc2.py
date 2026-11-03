"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def construct():
    # Deadline guard: must finish well before 150s
    t_start = time.time()
    DEADLINE = 135.0  # seconds, safety margin
    
    # Initialize with a grid layout
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N].astype(np.float64).copy()
    
    def compute_radius(c):
        """Compute the maximum radius for given centers."""
        # Min distance between any pair / 2
        n = len(c)
        min_dist = np.inf
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(c[i] - c[j])
                if d < min_dist:
                    min_dist = d
        # Min distance to walls
        for i in range(n):
            for k in range(2):
                d_wall = 0.5 - abs(c[i][k])
                if d_wall < min_dist:
                    min_dist = d_wall
        return min_dist / 2.0
    
    best_radius = compute_radius(centers)
    
    # Simulated annealing / gradient-based optimization
    # Use a simple approach: random perturbations + local refinement
    
    np.random.seed(42)
    
    # Multi-start with different initial configurations
    starts = [centers.copy()]
    
    # Add some random starts
    for s in range(5):
        if time.time() - t_start > DEADLINE - 10:
            break
        c_rand = np.random.uniform(-0.5, 0.5, (N, 2))
        starts.append(c_rand)
    
    # Also try a hexagonal-like packing
    c_hex = np.zeros((N, 2))
    idx = 0
    spacing = 1.0 / 8
    for row in range(8):
        for col in range(8):
            if idx >= N:
                break
            x = -0.5 + (col + 0.5) * spacing
            y = -0.5 + (row + 0.5) * spacing
            if row % 2 == 1:
                x += spacing * 0.5
            # Clamp to stay in bounds
            x = np.clip(x, -0.5, 0.5)
            y = np.clip(y, -0.5, 0.5)
            c_hex[idx] = [x, y]
            idx += 1
    starts.append(c_hex)
    
    def optimize(c, max_iters=50000, lr_init=0.01, lr_final=0.0001, alpha=0.9999):
        """Local optimization using subgradient method."""
        c = c.copy()
        lr = lr_init
        best_r = compute_radius(c)
        best_c = c.copy()
        
        for it in range(max_iters):
            if time.time() - t_start > DEADLINE - 5:
                break
            
            # Compute gradient of -radius (we want to maximize radius)
            # radius = min(min_pair_dist/2, min_wall_dist)
            # Use subgradient: find which constraint is active
            
            n = len(c)
            min_val = np.inf
            active_i, active_j, is_wall = -1, -1, False
            active_wall_idx, active_wall_coord = -1, -1
            
            # Check all pairs
            for i in range(n):
                for j in range(i+1, n):
                    d = np.linalg.norm(c[i] - c[j])
                    if d < min_val:
                        min_val = d
                        active_i, active_j = i, j
                        is_wall = False
            
            # Check walls
            for i in range(n):
                for k in range(2):
                    d_wall = 0.5 - abs(c[i][k])
                    if d_wall < min_val:
                        min_val = d_wall
                        active_i = i
                        is_wall = True
                        active_wall_idx, active_wall_coord = i, k
            
            r = min_val / 2.0
            
            if r > best_r:
                best_r = r
                best_c = c.copy()
            
            # Compute subgradient of -r
            grad = np.zeros_like(c)
            
            if is_wall:
                # Wall constraint active
                i = active_wall_idx
                k = active_wall_coord
                if c[i][k] > 0:
                    grad[i][k] = -1.0  # Push away from wall
                else:
                    grad[i][k] = 1.0
            else:
                # Pair constraint active
                i, j = active_i, active_j
                diff = c[i] - c[j]
                d = np.linalg.norm(diff)
                if d > 1e-12:
                    grad[i] = diff / d
                    grad[j] = -diff / d
            
            # Update: c += lr * grad (since we're minimizing -r)
            c = c + lr * grad
            
            # Project back into the square
            c = np.clip(c, -0.5, 0.5)
            
            # Decay learning rate
            lr *= alpha
            
            # Occasionally add small random perturbation to escape local minima
            if it % 1000 == 500:
                c = c + np.random.normal(0, 0.005, c.shape)
                c = np.clip(c, -0.5, 0.5)
        
        return best_c, best_r
    
    # Optimize each starting point
    for start in starts:
        if time.time() - t_start > DEADLINE - 10:
            break
        
        # First, do a quick annealing to get into a good basin
        c_annealed = start.copy()
        for temp in [0.05, 0.03, 0.01, 0.005, 0.001]:
            for it in range(2000):
                if time.time() - t_start > DEADLINE - 5:
                    break
                i = np.random.randint(N)
                pert = np.random.normal(0, temp, 2)
                c_new = c_annealed.copy()
                c_new[i] += pert
                c_new[i] = np.clip(c_new[i], -0.5, 0.5)
                
                r_old = compute_radius(c_annealed)
                r_new = compute_radius(c_new)
                
                if r_new >= r_old or np.random.random() < np.exp((r_new - r_old) / temp):
                    c_annealed = c_new
            
            if time.time() - t_start > DEADLINE - 5:
                break
        
        # Then do local optimization
        c_opt, r_opt = optimize(c_annealed)
        
        if r_opt > best_radius:
            best_radius = r_opt
            centers = c_opt
    
    # Final polish: run more iterations on the best solution
    if time.time() - t_start < DEADLINE - 10:
        c_polish, r_polish = optimize(centers, max_iters=100000, lr_init=0.005, lr_final=0.00001, alpha=0.99999)
        if r_polish > best_radius:
            best_radius = r_polish
            centers = c_polish
    
    return centers

# EVOLVE-BLOCK-END
