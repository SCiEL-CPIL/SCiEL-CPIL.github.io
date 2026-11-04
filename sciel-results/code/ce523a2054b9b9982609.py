import numpy as np
from scipy.spatial import cKDTree

N = 100

def construct():
    # 1. Initialization: Grid
    # 100 spheres. 4x5x5 = 100.
    # Spacing needs to be such that they fit.
    # Cube side 1.
    # Grid points in [-0.4, 0.4] to allow movement.
    nx, ny, nz = 4, 5, 5
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.4, 0.4, nz)
    
    centers = np.array([(x, y, z) for x in xs for y in ys for z in zs])
    
    # 2. Force-Directed Relaxation (Repulsion)
    # We want to maximize the minimum distance.
    # We use a potential that is very steep for small distances.
    # V(d) = 1/d^4 for d < d_max, 0 otherwise.
    # This pushes close spheres apart strongly.
    
    steps = 2000
    learning_rate = 0.01
    d_max = 0.2 # Cutoff for repulsion
    
    for step in range(steps):
        # Calculate forces
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion
        # Use KDTree for efficiency? N=100, O(N^2) is 10k ops, very fast.
        for i in range(N):
            for j in range(i + 1, N):
                diff = centers[i] - centers[j]
                d = np.linalg.norm(diff)
                if d < 1e-6:
                    # Jitter if coincident
                    diff = np.random.normal(0, 0.01, 3)
                    d = np.linalg.norm(diff)
                
                if d < d_max:
                    # Force magnitude ~ 1/d^2
                    # Direction is diff/d
                    # F = diff / d^3
                    f = diff / (d**3)
                    forces[i] += f
                    forces[j] -= f
        
        # Wall repulsion
        # Push away from walls if close
        wall_dist = 0.1 # Start pushing if within 0.1 of wall
        for i in range(N):
            for k in range(3):
                # Axis k
                c_val = centers[i, k]
                # Distance to -0.5 wall
                d_neg = c_val - (-0.5)
                if d_neg < wall_dist:
                    # Push positive
                    forces[i, k] += (wall_dist - d_neg) ** 2 * 10.0
                # Distance to +0.5 wall
                d_pos = 0.5 - c_val
                if d_pos < wall_dist:
                    # Push negative
                    forces[i, k] -= (wall_dist - d_pos) ** 2 * 10.0
        
        # Update positions
        centers += learning_rate * forces
        # Clip to box
        centers = np.clip(centers, -0.5, 0.5)
        
        # Decay learning rate
        learning_rate *= 0.999

    # 3. Local Refinement (Hill Climbing)
    # The MD might leave some spheres slightly suboptimal.
    # We do a local search on the minimum radius.
    
    def calc_radius(c):
        # Vectorized min distance
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        max_abs_coord = np.max(np.abs(c), axis=1)
        min_boundary = 0.5 - np.max(max_abs_coord)
        
        return min(min_pair / 2.0, min_boundary)

    r = calc_radius(centers)
    
    # Hill Climbing with random perturbations
    hc_steps = 5000
    base_step = 0.005
    
    for _ in range(hc_steps):
        i = np.random.randint(N)
        delta = np.random.normal(0, base_step, 3)
        new_c = centers.copy()
        new_c[i] = centers[i] + delta
        new_c[i] = np.clip(new_c[i], -0.5, 0.5)
        
        new_r = calc_radius(new_c)
        
        if new_r > r:
            centers = new_c
            r = new_r
        else:
            # Try a smaller step in the same direction? Or just skip.
            pass
            
    return centers
