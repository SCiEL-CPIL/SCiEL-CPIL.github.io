import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    # Objective function and gradient
    # We maximize the minimum distance.
    # Smooth approximation: minimize - (1/beta) * log( sum( exp(-beta * d_k) ) )
    # This is equivalent to maximizing the log-sum-exp of distances, which is dominated by the min distance.
    
    beta = 50.0  # Annealing parameter for smoothness
    
    def objective_and_grad(x_flat):
        x = x_flat.reshape(N, 3)
        # Calculate pairwise distances
        # d[i, j] = ||x_i - x_j||
        # Efficient way:
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=2)
        dists = np.sqrt(dists_sq + 1e-12) # Avoid division by zero
        
        # Mask out diagonal (self-distance)
        mask = ~np.eye(N, dtype=bool)
        pair_dists = dists[mask]
        
        # Wall distances
        # x is in [-0.5, 0.5]
        # dist to wall = min(x - (-0.5), 0.5 - x) = min(x+0.5, 0.5-x)
        # Actually, distance to boundary is min(x - (-0.5), 0.5 - x) for each dim.
        # min_k (0.5 - |x_k|) is the distance to the nearest face.
        wall_dists = 0.5 - np.abs(x)
        min_wall_dists = np.min(wall_dists, axis=1)
        
        all_dists = np.concatenate([pair_dists, min_wall_dists])
        
        # Log-sum-exp trick for stability
        # S = sum( exp(-beta * d) )
        # log(S) = max_d(-beta*d) + log( sum( exp(-beta*d - max_d(-beta*d)) ) )
        vals = -beta * all_dists
        max_val = np.max(vals)
        log_sum_exp = max_val + np.log(np.sum(np.exp(vals - max_val)))
        
        obj = -log_sum_exp / beta
        
        # Gradient
        # d obj / d x = -1/beta * (1/S) * sum( d(exp(-beta d_k))/dx )
        # = -1/beta * (1/S) * sum( exp(-beta d_k) * (-beta) * d(d_k)/dx )
        # = (1/S) * sum( exp(-beta d_k) * d(d_k)/dx )
        
        exp_vals = np.exp(vals - max_val)
        S = np.sum(exp_vals)
        # We need to map exp_vals back to pairs and walls
        
        # Pair distances gradient
        # d(d_ij)/dx_i = (x_i - x_j) / d_ij
        # d(d_ij)/dx_j = (x_j - x_i) / d_ij
        
        grad = np.zeros_like(x)
        
        # Sum over pairs
        # exp_vals for pairs are the first N*(N-1) elements
        pair_exp = exp_vals[:N*(N-1)]
        # Reshape to (N, N-1)? No, mask was flattened.
        # It's easier to compute the force directly.
        # Force on i from j is proportional to exp(-beta d_ij) * (x_i - x_j)/d_ij
        
        # Let's compute the sum of weighted vectors
        # weights = exp(-beta d_ij) / S
        # This is O(N^2)
        
        # Re-calculate distances and weights for gradient
        # We already have dists and mask
        weights_pair = np.zeros((N, N))
        # Map flat pair_exp to matrix
        # The mask order is row-major.
        # Let's just recompute the weighted sum efficiently.
        
        # Weight for pair (i,j) is exp(-beta d_ij) / S
        # Note: S includes wall terms too.
        
        # Let's compute the gradient term by term.
        # Term for pair (i,j):
        # w_ij = exp(-beta d_ij) / S
        # grad_i += w_ij * (x_i - x_j) / d_ij
        # grad_j += w_ij * (x_j - x_i) / d_ij
        
        # To avoid O(N^2) python loops, we can use matrix ops.
        # diff is (N, N, 3)
        # dists is (N, N)
        # We need sum_j ( exp(-beta d_ij) / S * (x_i - x_j) / d_ij )
        
        # Create weight matrix W (N, N)
        W = np.zeros((N, N))
        # Fill W with exp(-beta * dists) / S where mask is True
        W[mask] = pair_exp / S
        
        # Sum_j W_ij * (x_i - x_j) / d_ij
        # Note: W_ij is 0 on diagonal.
        # (x_i - x_j) / d_ij is the unit vector u_ij.
        # diff / dists[:, :, np.newaxis] gives unit vectors.
        unit_vecs = diff / (dists[:, :, np.newaxis] + 1e-12)
        
        # grad_pair = sum_j W_ij * u_ij
        # This is (W @ unit_vecs)? No, W is (N,N), unit_vecs is (N,N,3).
        # grad_i = sum_j W[i,j] * unit_vecs[i,j]
        grad_pair = np.sum(W[:, :, np.newaxis] * unit_vecs, axis=1)
        
        # Wall distances gradient
        # d(d_wall_i)/dx_i = sign(x_i) * (1 if x_i > 0 else -1)?
        # d_wall_i = min_k (0.5 - |x_ik|)
        # Let m_i be the dimension of the min.
        # d(d_wall_i)/dx_ik = -sign(x_ik) if k == m_i, else 0.
        # Weight w_wall_i = exp(-beta d_wall_i) / S
        # grad_i += w_wall_i * (-sign(x_i[m_i])) e_{m_i}
        
        wall_exp = exp_vals[N*(N-1):]
        weights_wall = wall_exp / S
        
        grad_wall = np.zeros_like(x)
        # For each i, find the dimension of the min wall distance
        min_dims = np.argmin(wall_dists, axis=1)
        for i in range(N):
            k = min_dims[i]
            # Derivative of (0.5 - |x_ik|) wrt x_ik is -sign(x_ik)
            # If x_ik is 0, sign is 0, but we can use a subgradient or just 0.
            # Using sign(x_ik)
            grad_wall[i, k] = weights_wall[i] * (-np.sign(x[i, k]))
            
        grad = grad_pair + grad_wall
        
        return obj, grad

    def run_optimization(init_x):
        # Flatten
        x0 = init_x.flatten()
        
        # Bounds
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        # Use L-BFGS-B
        res = minimize(
            objective_and_grad, 
            x0, 
            method='L-BFGS-B', 
            jac=True, 
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        return res.x.reshape(N, 3)

    def calc_score(x):
        # Calculate the actual min radius
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2) + 1e-12)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        wall_dists = 0.5 - np.abs(x)
        min_wall = np.min(np.min(wall_dists, axis=1))
        
        return min(min_pair / 2.0, min_wall)

    best_x = None
    best_score = -np.inf
    
    def resolve_overlaps(x, r_target, steps=200):
        x = x.copy()
        for _ in range(steps):
            forces = np.zeros_like(x)
            # Pairwise
            for i in range(N):
                for j in range(i+1, N):
                    d = x[i] - x[j]
                    dist = np.linalg.norm(d)
                    if dist < 2 * r_target and dist > 1e-9:
                        f = (2 * r_target - dist) / dist * d
                        forces[i] += f
                        forces[j] -= f
            # Walls
            for i in range(N):
                for k in range(3):
                    # Left wall: x + 0.5
                    d_left = x[i, k] + 0.5
                    if d_left < r_target:
                        forces[i, k] += (r_target - d_left) * (-1) # Push right (positive x)
                    # Right wall: 0.5 - x
                    d_right = 0.5 - x[i, k]
                    if d_right < r_target:
                        forces[i, k] += (r_target - d_right) * (1) # Push left (negative x)
            x += forces * 0.1
            # Clamp
            x = np.clip(x, -0.5, 0.5)
        return x

    # Improved Initialization: FCC Lattice
    def generate_fcc_init(seed):
        np.random.seed(seed)
        # FCC lattice points in a unit cell
        # Points: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
        # We need a larger volume to pick 50 points.
        # Let's generate a grid of FCC points in a larger cube and pick the densest subset.
        
        # Generate FCC points in [-1, 1]
        # Step size approx 0.5
        points = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                for k in range(-2, 3):
                    # FCC basis
                    points.append([i*0.5, j*0.5, k*0.5])
                    points.append([i*0.5+0.25, j*0.5+0.25, k*0.5])
                    points.append([i*0.5+0.25, j*0.5, k*0.5+0.25])
                    points.append([i*0.5, j*0.5+0.25, k*0.5+0.25])
        
        points = np.array(points)
        
        # Filter points to be roughly inside [-0.5, 0.5]
        # Actually, we want to pack them IN the cube.
        # So we should pick points that are close to the center.
        
        # Calculate distance from center (0,0,0)
        dists = np.linalg.norm(points, axis=1)
        
        # Sort by distance
        sorted_indices = np.argsort(dists)
        
        # Pick the first N points
        x_init = points[sorted_indices[:N]].copy()
        
        # Scale to fit in cube [-0.5, 0.5]
        # The max coordinate in our generated set might be > 0.5
        # Let's normalize to fit within [-0.4, 0.4] initially to allow expansion
        max_coord = np.max(np.abs(x_init))
        if max_coord > 0.4:
            x_init *= (0.4 / max_coord)
            
        # Jitter to break symmetry
        x_init += np.random.normal(0, 0.01, x_init.shape)
        
        return x_init

    # Strategy 1: FCC Lattice + Dynamics + Optimization
    for seed in range(10):
        if time.time() > deadline:
            break
        
        x_init = generate_fcc_init(seed)
        
        # Resolve overlaps with a small radius to make it feasible
        # Start with a small radius, then increase?
        # For now, just resolve with r=0.05
        x_resolved = resolve_overlaps(x_init, 0.05, steps=50)
        
        x_opt = run_optimization(x_resolved)
        score = calc_score(x_opt)
        
        if score > best_score:
            best_score = score
            best_x = x_opt

    # Strategy 2: BCC Lattice Init
    def generate_bcc_init(seed):
        np.random.seed(seed)
        points = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                for k in range(-2, 3):
                    points.append([i*0.5, j*0.5, k*0.5])
                    points.append([i*0.5+0.25, j*0.5+0.25, k*0.5+0.25])
        
        points = np.array(points)
        dists = np.linalg.norm(points, axis=1)
        sorted_indices = np.argsort(dists)
        x_init = points[sorted_indices[:N]].copy()
        
        max_coord = np.max(np.abs(x_init))
        if max_coord > 0.4:
            x_init *= (0.4 / max_coord)
            
        x_init += np.random.normal(0, 0.01, x_init.shape)
        return x_init

    for seed in range(10, 20):
        if time.time() > deadline:
            break
        
        x_init = generate_bcc_init(seed)
        x_resolved = resolve_overlaps(x_init, 0.05, steps=50)
        x_opt = run_optimization(x_resolved)
        score = calc_score(x_opt)
        
        if score > best_score:
            best_score = score
            best_x = x_opt

    # Strategy 3: Random + Overlap Resolution (Original)
    for seed in range(20, 30):
        if time.time() > deadline:
            break
        np.random.seed(seed)
        x_init = np.random.uniform(-0.4, 0.4, (N, 3))
        x_resolved = resolve_overlaps(x_init, 0.05, steps=50)
        
        x_opt = run_optimization(x_resolved)
        score = calc_score(x_opt)
        
        if score > best_score:
            best_score = score
            best_x = x_opt

    if best_x is None:
        # Fallback to simple grid if something failed
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_x = grid[:N]
        
    return best_x
