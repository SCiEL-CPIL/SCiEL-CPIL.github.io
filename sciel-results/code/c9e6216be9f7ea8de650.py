"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the common radius r for given centers."""
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to walls
    # Cube is [-0.5, 0.5]^3
    # Distance from center c to wall is min(0.5 - c_k, c_k + 0.5) for each axis k
    # Actually, distance to the boundary of the box [-0.5, 0.5] is:
    # min(0.5 - c_x, c_x + 0.5) for x-axis, etc.
    # Let's compute min distance to any face.
    wall_dists = np.minimum(0.5 - centers, centers + 0.5)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist, min_wall_dist) / 2.0

def _surrogate_objective_and_grad(centers, beta):
    """
    Compute the negative of the LSE surrogate for the minimum radius and its gradient.
    We want to maximize r = min(d_ij/2, d_wall_i).
    Equivalent to minimizing -r.
    Let f_i = -d_i (where d_i are the distances we want to maximize).
    We want to minimize max(f_i) approximated by (1/beta) * log(sum(exp(beta * f_i))).
    
    Here, the set of distances D consists of:
    1. Pairwise distances d_ij / 2
    2. Wall distances d_wall_i / 2 (Note: radius is half the distance to wall? 
       No, radius is the distance to the wall if the sphere is tangent. 
       Wait, the problem says: "min_i min_k (1/2 - |c_ik|)" is the distance to the wall.
       And "min_{i<j} |c_i - c_j| / 2" is half the distance between centers.
       So r = min( min_pair_dist/2, min_wall_dist ).
       
       Let's re-read carefully: 
       "Score: the largest common radius r = min(min_{i<j} |c_i - c_j| / 2, min_i min_k (1/2 - |c_ik|))"
       
       So the constraints are:
       1. r <= |c_i - c_j| / 2  =>  2r <= |c_i - c_j|
       2. r <= 0.5 - |c_ik|     =>  r <= dist_to_wall
        
       So we want to maximize r such that:
       2r <= d_ij for all i<j
       r <= d_wall_i for all i,k
        
       Let's define the "slack" variables:
       s_ij = d_ij - 2r
       s_wall = d_wall - r
        
       We want to maximize r.
       
       Alternative formulation:
       Maximize r subject to:
       d_ij - 2r >= 0
       d_wall - r >= 0
       
       This is equivalent to minimizing a surrogate for -r.
       
       Let's define the "violations" if we guess a radius r0? No, that's not direct.
       
       Better: Minimize the maximum of the "tightness" of the constraints.
       Let's define the objective as the negative of the minimum of the normalized distances.
       
       Let A = { d_ij / 2 } for all i<j
       Let B = { d_wall_i } for all i
       
       We want to maximize min( min(A), min(B) ).
       
       Let's use the LSE approximation of the minimum.
       min(x) = -max(-x)
       So we want to minimize max( -d_ij/2, -d_wall_i ).
       
       Let g(x) = max( -d_ij/2, -d_wall_i ).
       We approximate g(x) with LSE:
       LSE_beta(x) = (1/beta) * log( sum( exp(beta * x_k) ) )
       
       So we minimize F(centers) = (1/beta) * log( sum_{i<j} exp(-beta * d_ij/2) + sum_i sum_k exp(-beta * d_wall_ik) )
       
       Wait, d_wall_i is the minimum distance to any wall for sphere i.
       d_wall_i = min( 0.5 - c_ix, c_ix + 0.5, 0.5 - c_iy, c_iy + 0.5, 0.5 - c_iz, c_iz + 0.5 )
       
       This is also a minimum of 6 terms.
       So the total set of terms in the max is:
       For each pair (i,j): term = -d_ij/2
       For each sphere i and each wall w: term = -d_iw
        
       Actually, d_wall_i = min_w d_iw. So -d_wall_i = max_w -d_iw.
       So max( -d_ij/2, -d_wall_i ) = max( -d_ij/2, max_w -d_iw ) = max( -d_ij/2, -d_i1, ..., -d_i6 )
       
       So we can just sum over all these terms in the LSE.
       
       Terms:
       1. For each pair i<j: t_ij = -0.5 * ||c_i - c_j||
       2. For each sphere i:
          t_ix+ = -(0.5 - c_ix) = c_ix - 0.5
          t_ix- = -(c_ix + 0.5) = -c_ix - 0.5
          t_iy+ = c_iy - 0.5
          t_iy- = -c_iy - 0.5
          t_iz+ = c_iz - 0.5
          t_iz- = -c_iz - 0.5
          
       Objective: (1/beta) * log( sum_{i<j} exp(beta * t_ij) + sum_i sum_{walls} exp(beta * t_iw) )
       
       Gradient:
       Let S = sum_k exp(beta * t_k)
       F = (1/beta) * log(S)
       dF/dc = (1/beta) * (1/S) * dS/dc
             = (1/beta) * (1/S) * sum_k exp(beta * t_k) * beta * (dt_k/dc)
             = (1/S) * sum_k exp(beta * t_k) * (dt_k/dc)
             = sum_k ( exp(beta * t_k) / S ) * (dt_k/dc)
       
       Let w_k = exp(beta * t_k) / S. These are the weights.
       
       dt_ij/dc_i = -0.5 * (c_i - c_j) / ||c_i - c_j||
       dt_ij/dc_j = -0.5 * (c_j - c_i) / ||c_i - c_j||
       
       dt_ix+/dc_ix = 1
       dt_ix-/dc_ix = -1
       etc.
    """
    n = len(centers)
    # Initialize gradient
    grad = np.zeros_like(centers)
    
    # Compute terms and weights
    # We need to be careful with numerical stability.
    # We will compute log_sum_exp in a stable way.
    
    # 1. Pairwise terms
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Only upper triangle for pairs
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu]
    pair_diffs = diff[iu] # shape (M, 3)
    
    # t_ij = -0.5 * dist
    t_pairs = -0.5 * pair_dists
    
    # 2. Wall terms
    # For each center, 6 terms
    # c = [x, y, z]
    # t_x+ = x - 0.5
    # t_x- = -x - 0.5
    # ...
    wall_terms = np.zeros((n, 6))
    wall_terms[:, 0] = centers[:, 0] - 0.5
    wall_terms[:, 1] = -centers[:, 0] - 0.5
    wall_terms[:, 2] = centers[:, 1] - 0.5
    wall_terms[:, 3] = -centers[:, 1] - 0.5
    wall_terms[:, 4] = centers[:, 2] - 0.5
    wall_terms[:, 5] = -centers[:, 2] - 0.5
    
    # Flatten wall terms
    t_walls = wall_terms.flatten()
    
    # Combine all terms
    all_terms = np.concatenate([t_pairs, t_walls])
    
    # Stable log-sum-exp
    max_term = np.max(all_terms)
    if max_term > 100: # Overflow guard, though beta is small usually
        # If terms are very positive, exp(beta*t) might overflow
        # But t are negative usually (distances are positive, so -dist is negative)
        # Wait, t = -dist. So exp(beta * -dist) is small.
        # The max term will be the one with the largest (least negative) value, i.e., smallest distance.
        # So max_term is close to 0 (if r is large) or very negative (if r is small).
        pass
        
    # Compute weights
    # w_k = exp(beta * t_k) / sum(exp(beta * t_k))
    # To avoid underflow/overflow, shift by max_term
    exp_terms = np.exp(beta * (all_terms - max_term))
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp
    
    # Objective value
    obj = (1.0 / beta) * (np.log(sum_exp) + beta * max_term)
    
    # Gradient
    # grad = sum_k w_k * dt_k/dc
    
    # Pairwise gradient contribution
    # w_pairs = weights[:len(t_pairs)]
    w_pairs = weights[:len(t_pairs)]
    
    # dt_ij/dc_i = -0.5 * (c_i - c_j) / d_ij
    # dt_ij/dc_j = -0.5 * (c_j - c_i) / d_ij
    
    # We need to accumulate these into grad.
    # pair_diffs is (M, 3). pair_dists is (M,).
    # unit_vecs = pair_diffs / pair_dists[:, np.newaxis]
    unit_vecs = pair_diffs / pair_dists[:, np.newaxis]
    
    # contrib_i = -0.5 * w_pairs * unit_vecs
    # contrib_j = -0.5 * w_pairs * (-unit_vecs) = 0.5 * w_pairs * unit_vecs
    
    # We can use np.add.at
    i_idx, j_idx = iu
    # grad[i] += -0.5 * w * unit_vec
    # grad[j] += 0.5 * w * unit_vec
    
    # Efficient way:
    # Create a matrix of contributions
    # But N is small (50), so M=1225. This is fine.
    
    # Let's just loop or use advanced indexing if possible.
    # Actually, we can use:
    # grad[i_idx] -= 0.5 * w_pairs[:, np.newaxis] * unit_vecs
    # grad[j_idx] += 0.5 * w_pairs[:, np.newaxis] * unit_vecs
    
    # But this might have race conditions if we do it in one go? No, np.add.at is safe.
    # Or we can use:
    np.add.at(grad, i_idx, -0.5 * w_pairs[:, np.newaxis] * unit_vecs)
    np.add.at(grad, j_idx, 0.5 * w_pairs[:, np.newaxis] * unit_vecs)
    
    # Wall gradient contribution
    # w_walls = weights[len(t_pairs):]
    w_walls = weights[len(t_pairs):]
    w_walls_reshaped = w_walls.reshape(n, 6)
    
    # dt/dc for walls:
    # t_x+ = x - 0.5 -> dt/dx = 1
    # t_x- = -x - 0.5 -> dt/dx = -1
    # t_y+ = y - 0.5 -> dt/dy = 1
    # t_y- = -y - 0.5 -> dt/dy = -1
    # t_z+ = z - 0.5 -> dt/dz = 1
    # t_z- = -z - 0.5 -> dt/dz = -1
    
    # grad_x += w_x+ * 1 + w_x- * (-1)
    # grad_y += w_y+ * 1 + w_y- * (-1)
    # grad_z += w_z+ * 1 + w_z- * (-1)
    
    grad[:, 0] += w_walls_reshaped[:, 0] - w_walls_reshaped[:, 1]
    grad[:, 1] += w_walls_reshaped[:, 2] - w_walls_reshaped[:, 3]
    grad[:, 2] += w_walls_reshaped[:, 4] - w_walls_reshaped[:, 5]
    
    # We want to MINIMIZE the surrogate of the MAX of negative distances.
    # So the objective returned to scipy should be the value, and grad the gradient.
    return obj, grad

def _optimize_lbgfgs(centers, max_time=30.0, beta=50.0, seed=0):
    """
    Optimize the LSE surrogate using L-BFGS-B.
    """
    start_time = time.time()
    
    # Flatten centers for L-BFGS-B
    x0 = centers.flatten()
    
    # Bounds: each coordinate in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Callback to check time and optionally adjust beta (annealing)
    # For simplicity, we will run with a fixed beta for this run, 
    # or do a few runs with increasing beta.
    
    def fun_and_grad(x):
        c = x.reshape(N, 3)
        val, grad = _surrogate_objective_and_grad(c, beta)
        return val, grad.flatten()
    
    # Use L-BFGS-B
    # maxiter is a safety limit, but we rely on time
    res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True, bounds=bounds, 
                   options={'maxiter': 1000, 'maxfun': 1000, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False})
    
    best_centers = res.x.reshape(N, 3)
    
    # Ensure validity
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers

def _generate_grid_start():
    """Generate a cubic grid start."""
    # 50 points. 4x4x4 = 64. Take first 50.
    # Or 3x4x5 = 60.
    # Let's use a 4x4x4 grid and remove corners if needed?
    # Simplest: 4x4x4 grid, take first 50.
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _generate_fcc_start():
    """Generate an FCC-like start."""
    # FCC lattice in a cube.
    # Unit cell has 4 atoms.
    # Let's just use a dense grid and jitter.
    return _generate_grid_start() + np.random.RandomState(1).randn(N, 3) * 0.01

def _generate_random_start():
    """Generate a random start."""
    return np.random.RandomState(2).uniform(-0.5, 0.5, (N, 3))

def construct():
    start_time = time.time()
    deadline = 145.0 # Safety margin
    
    # Multi-start with different initializations and beta values
    # Start with lower beta (smoother) and go to higher beta (sharper)
    
    best_centers = None
    best_radius = -np.inf
    
    initializations = [
        _generate_grid_start,
        _generate_fcc_start,
        _generate_random_start
    ]
    
    # We will try a few betas for the best initializations
    # Or just run L-BFGS-B with a moderate beta.
    
    # Strategy:
    # 1. Run with beta=20 for 3 starts.
    # 2. Take the best, refine with beta=50.
    # 3. Take the best, refine with beta=100.
    
    current_centers = None
    best_r = -np.inf
    
    # Phase 1: Coarse optimization
    for init_func in initializations:
        if time.time() > deadline - 30:
            break
        init_centers = init_func()
        # Add small jitter to break symmetry for random/grid
        init_centers = init_centers + np.random.RandomState(hash(init_func) % 100).randn(N, 3) * 0.001
        init_centers = np.clip(init_centers, -0.5, 0.5)
        
        time_left = deadline - time.time()
        if time_left < 10:
            break
        
        # Allocate time
        t_per = min(20.0, time_left / 2)
        
        # We can't easily pass a time limit to minimize, so we rely on maxiter/maxfun
        # Or we can wrap it? No, let's just set reasonable maxiter.
        # L-BFGS-B is fast.
        
        centers = _optimize_lbgfgs(init_centers, max_time=t_per, beta=20.0)
        
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
            current_centers = centers.copy()
    
    # Phase 2: Refinement with higher beta
    if current_centers is not None:
        # Try beta=50
        if time.time() < deadline - 20:
            centers = _optimize_lbgfgs(current_centers, max_time=20.0, beta=50.0)
            r = _compute_radius(centers)
            if r > best_r:
                best_r = r
                best_centers = centers.copy()
                current_centers = centers.copy()
        
        # Try beta=100
        if time.time() < deadline - 10:
            centers = _optimize_lbgfgs(current_centers, max_time=10.0, beta=100.0)
            r = _compute_radius(centers)
            if r > best_r:
                best_r = r
                best_centers = centers.copy()
        
        # Try beta=200
        if time.time() < deadline - 5:
            centers = _optimize_lbgfgs(current_centers, max_time=5.0, beta=200.0)
            r = _compute_radius(centers)
            if r > best_r:
                best_r = r
                best_centers = centers.copy()
    
    if best_centers is None:
        best_centers = _generate_grid_start()
    
    # Final safety clip
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
# EVOLVE-BLOCK-END
