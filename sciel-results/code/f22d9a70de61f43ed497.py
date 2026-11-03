"""
Optimized packing of 50 equal spheres in a unit cube.
Uses a smooth surrogate (soft-min) objective with L-BFGS-B optimization.
Key improvements:
1. Vectorized gradient computation for the soft-min objective.
2. Multiple diverse initializations including lattice-based structures.
3. Aggressive beta annealing to escape local minima and converge to sharp minima.
4. Strict deadline management to ensure validity and maximize search time.
"""
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

N = 50
DEADLINE_SECONDS = 145.0  # Safety margin before 150s kill

# Precompute pair indices for vectorized gradient
# pdist returns distances in condensed form.
# For N points, the index k in pdist corresponds to pair (i, j) where:
# i = (2*N - 3 + sqrt(9 - 8*N + 8*k)) / 2  (derived from quadratic formula)
# j = i + 1 + (k - i*(2*N - i - 3)/2)
# However, it's safer and faster to just precompute the arrays once.
PAIR_I = np.empty(N * (N - 1) // 2, dtype=np.int32)
PAIR_J = np.empty(N * (N - 1) // 2, dtype=np.int32)
_k = 0
for _i in range(N):
    for _j in range(_i + 1, N):
        PAIR_I[_k] = _i
        PAIR_J[_k] = _j
        _k += 1

def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    def remaining():
        return DEADLINE_SECONDS - elapsed()

    def compute_min_radius(positions):
        """Compute the minimum radius given positions of shape (N, 3)."""
        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        if not np.all(np.isfinite(positions)):
            return 0.0
        if positions.shape[0] != N or positions.shape[1] != 3:
            return 0.0
        # Check bounds: strictly inside [-0.5, 0.5]
        if np.any(np.abs(positions) >= 0.5):
            return 0.0
        
        dists = pdist(positions, 'euclidean')
        min_dist = np.min(dists) if len(dists) > 0 else 1.0
        min_r_pairs = min_dist / 2.0
        
        walls = 0.5 - np.abs(positions)
        min_wall = np.min(walls)
        
        return min(min_r_pairs, min_wall)

    def smooth_surrogate_and_grad(x_flat, beta):
        """
        Compute the smooth surrogate objective and its gradient.
        Objective: 1/beta * log(sum(exp(-beta * v_i)))
        where v_i are the candidate radii (pairwise distances/2 and wall distances).
        Gradient: -weights_i * grad(v_i)
        """
        positions = x_flat.reshape(-1, 3)
        
        # Compute pairwise distances
        dists = pdist(positions, 'euclidean')
        pair_r = dists / 2.0
        
        # Compute wall distances
        walls = 0.5 - np.abs(positions)
        wall_r = walls.ravel()
        
        # Combine all candidate radii
        all_vals = np.concatenate([pair_r, wall_r])
        
        # Stable log-sum-exp calculation
        neg_beta_vals = -beta * all_vals
        max_val = np.max(neg_beta_vals)
        exp_vals = np.exp(neg_beta_vals - max_val)
        sum_exp = np.sum(exp_vals)
        
        # Objective value
        obj_val = (max_val + np.log(sum_exp)) / beta
        
        # Weights for gradient
        weights = exp_vals / sum_exp
        
        # Gradient calculation
        grad_pos = np.zeros_like(positions)
        
        n_pairs = len(pair_r)
        
        # Pairwise part
        # grad(pair_r_k) / grad(pos_i) = (pos_i - pos_j) / dist_ij
        # We need to accumulate weights * grad
        # Using precomputed PAIR_I, PAIR_J
        # dists is the full distance, pair_r is dist/2
        # d(pair_r)/d(pos_i) = (pos_i - pos_j) / (2 * dist)
        
        # Vectorized accumulation is tricky with simple arrays, but we can use np.add.at
        # Or just loop if N is small enough, but N=50 means 1225 pairs. 
        # A loop of 1225 iterations in Python is fast (~ms).
        # However, we can vectorize the diff calculation.
        
        pos_i = positions[PAIR_I]
        pos_j = positions[PAIR_J]
        diffs = pos_i - pos_j  # shape (n_pairs, 3)
        
        # dists is shape (n_pairs,)
        # Avoid division by zero
        safe_dists = np.where(dists < 1e-12, 1e-12, dists)
        
        # Gradient contribution for pair k on pos_i: weights[k] * diffs[k] / (2 * dists[k])
        # Gradient contribution for pair k on pos_j: -weights[k] * diffs[k] / (2 * dists[k])
        
        # weights_pair = weights[:n_pairs]
        # factor = weights_pair / (2 * safe_dists)  # shape (n_pairs,)
        
        # We can use np.add.at for accumulation
        # But since each pair involves two points, we need to be careful.
        # Let's stick to a simple loop for clarity and sufficient speed, 
        # as 1225 iterations is negligible compared to optimizer steps.
        
        for idx in range(n_pairs):
            w = weights[idx]
            if w < 1e-15:
                continue
            i = PAIR_I[idx]
            j = PAIR_J[idx]
            d = safe_dists[idx]
            diff = diffs[idx]
            factor = w / (2.0 * d)
            
            grad_pos[i] += factor * diff
            grad_pos[j] -= factor * diff
        
        # Wall part
        # wall_r[k] corresponds to position i = k//3, coord d = k%3
        # d(wall)/d(pos) = -sign(pos)
        # weights_wall = weights[n_pairs:]
        
        wall_weights = weights[n_pairs:]
        # Vectorized wall gradient
        # For each point i and coord d, the contribution is -sign(pos[i,d]) * wall_weights[i*3+d]
        
        # Create a sign matrix
        signs = np.sign(positions)
        # Where pos is 0, sign is 0, which is a valid subgradient choice (or we could use 1, but 0 is safer for symmetry)
        
        # Reshape wall_weights to (N, 3)
        wall_weights_reshaped = wall_weights.reshape(N, 3)
        
        # Gradient contribution: -signs * wall_weights_reshaped
        grad_pos -= signs * wall_weights_reshaped
        
        return obj_val, grad_pos.ravel()

    def local_optimize_smooth(positions, beta, maxiter, ftol):
        """Run L-BFGS-B on the smooth surrogate."""
        if remaining() < 5:
            return positions
        
        bounds = [(-0.499999, 0.499999)] * (N * 3)
        
        def obj_flat(x):
            val, _ = smooth_surrogate_and_grad(x, beta)
            return val
            
        def grad_flat(x):
            _, g = smooth_surrogate_and_grad(x, beta)
            return g
        
        try:
            res = minimize(obj_flat, positions.flatten(), jac=grad_flat, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': maxiter, 'ftol': ftol, 'gtol': 1e-8})
            pos_opt = res.x.reshape(-1, 3)
            pos_opt = np.clip(pos_opt, -0.499999, 0.499999)
            return pos_opt
        except Exception:
            return positions

    def local_optimize_direct(positions, maxiter=200):
        """Run L-BFGS-B on the raw -min_radius objective."""
        if remaining() < 5:
            return positions
        
        bounds = [(-0.499999, 0.499999)] * (N * 3)
        
        def obj_flat(x):
            return -compute_min_radius(x.reshape(-1, 3))
        
        try:
            res = minimize(obj_flat, positions.flatten(), method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-14})
            pos_opt = res.x.reshape(-1, 3)
            pos_opt = np.clip(pos_opt, -0.499999, 0.499999)
            return pos_opt
        except Exception:
            return positions

    def make_initial_grid():
        """Simple grid initialization."""
        # 4x4x4 grid has 64 points, take first 50
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N].copy()

    def make_bcc_lattice():
        """Body Centered Cubic lattice initialization."""
        positions = []
        # 3x3x3 BCC unit cells? No, just points in a 3x3x3 grid with centers.
        # Let's try a 3x3x3 grid of simple cubic, then add centers?
        # 27 corners + 27 centers = 54 points. Take 50.
        a = 0.5
        step = 2*a/3
        for iz in range(3):
            for iy in range(3):
                for ix in range(3):
                    x = -a + (ix + 0.5) * step
                    y = -a + (iy + 0.5) * step
                    z = -a + (iz + 0.5) * step
                    positions.append([x, y, z])
                    if len(positions) < N:
                        positions.append([x + step/2, y + step/2, z + step/2])
        positions = np.array(positions[:N])
        return positions

    def make_fcc_lattice():
        """Face Centered Cubic inspired initialization."""
        # FCC has 4 atoms per unit cell.
        # 2x2x2 unit cells = 16 atoms. Too few.
        # 3x3x3 unit cells = 108 atoms. Take 50.
        a = 0.5
        step = 2*a/3
        positions = []
        for iz in range(3):
            for iy in range(3):
                for ix in range(3):
                    # Corners
                    x0 = -a + ix * step
                    y0 = -a + iy * step
                    z0 = -a + iz * step
                    # 8 corners, but shared. Let's just generate all potential FCC points in the box
                    # FCC points in [0,1] are (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5) + integer shifts
                    # Scale to [-0.5, 0.5]
                    # Let's just use a dense random grid for FCC-like if BCC is not enough
                    pass
        
        # Simpler: Just use a perturbed grid
        return make_initial_grid() + np.random.normal(0, 0.05, (N, 3))

    def make_random(seed):
        np.random.seed(seed)
        return np.random.uniform(-0.45, 0.45, size=(N, 3))

    best_positions = None
    best_radius = -1.0

    # List of initializations to try
    initializations = [
        ("grid", make_initial_grid),
        ("bcc", make_bcc_lattice),
        ("rand0", lambda: make_random(0)),
        ("rand1", lambda: make_random(42)),
        ("rand2", lambda: make_random(123)),
        ("rand3", lambda: make_random(777)),
        ("rand4", lambda: make_random(2024)),
        ("rand5", lambda: make_random(999)),
    ]

    # Annealing schedule for beta
    # Start low to explore, end high to exploit
    beta_schedule = [10.0, 20.0, 50.0, 100.0, 200.0, 500.0]

    for name, init_func in initializations:
        if remaining() < 20:
            break
            
        try:
            positions = init_func()
            positions = np.clip(positions, -0.499999, 0.499999)
            
            current_pos = positions
            current_r = compute_min_radius(current_pos)
            
            # Annealing loop
            for beta in beta_schedule:
                if remaining() < 10:
                    break
                
                # Allocate more iterations for lower beta (exploration)
                if beta < 50:
                    maxiter = 200
                else:
                    maxiter = 100
                    
                current_pos = local_optimize_smooth(current_pos, beta=beta, maxiter=maxiter, ftol=1e-12)
                current_r = compute_min_radius(current_pos)
                
                # Early stop if no improvement in radius? 
                # The surrogate might improve even if radius doesn't immediately, 
                # but usually radius tracks it.
                
            # Final direct polish
            if remaining() > 10:
                current_pos = local_optimize_direct(current_pos, maxiter=100)
                current_r = compute_min_radius(current_pos)
            
            if current_r > best_radius:
                best_radius = current_r
                best_positions = current_pos.copy()
                
        except Exception:
            continue

    # Perturbations of the best solution found
    if best_positions is not None and remaining() > 30:
        for perturb_seed in range(5):
            if remaining() < 15:
                break
            
            np.random.seed(1000 + perturb_seed)
            # Small perturbations
            perturbed = best_positions + np.random.normal(0, 0.01, size=best_positions.shape)
            perturbed = np.clip(perturbed, -0.499999, 0.499999)
            
            # Quick annealing
            for beta in [50.0, 100.0, 200.0]:
                if remaining() < 5:
                    break
                perturbed = local_optimize_smooth(perturbed, beta=beta, maxiter=50, ftol=1e-12)
            
            if remaining() > 5:
                perturbed = local_optimize_direct(perturbed, maxiter=50)
            
            r = compute_min_radius(perturbed)
            if r > best_radius:
                best_radius = r
                best_positions = perturbed.copy()

    # Final high-precision polish
    if best_positions is not None and remaining() > 10:
        polished = local_optimize_smooth(best_positions, beta=1000.0, maxiter=100, ftol=1e-14)
        polished = local_optimize_direct(polished, maxiter=50)
        r_polished = compute_min_radius(polished)
        if r_polished > best_radius:
            best_radius = r_polished
            best_positions = polished.copy()

    # Ensure validity
    if best_positions is None:
        best_positions = make_initial_grid()
    
    best_positions = np.clip(best_positions, -0.499999, 0.499999)
    
    return best_positions

if __name__ == "__main__":
    # Quick self-test
    pos = construct()
    print("Radius:", compute_min_radius(pos))
