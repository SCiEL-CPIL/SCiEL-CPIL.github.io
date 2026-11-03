"""Packing 50 equal spheres in a unit cube via simulated annealing + gradient-based local refinement."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

N = 50
DEADLINE = 145.0  # seconds, safety margin before 150s limit


def _score(centers):
    """Compute the maximum radius for given centers."""
    # Wall distances: min distance from each center to the boundary
    # Distance to wall is min(0.5 - x, x + 0.5) etc.
    # Equivalent to 0.5 - max(|x|, |y|, |z|) for a single point? 
    # No, min_k (0.5 - |c_ik|) is the distance to the closest face.
    # For a point c, dist_to_wall = min(0.5 - c_0, c_0 + 0.5, 0.5 - c_1, c_1 + 0.5, 0.5 - c_2, c_2 + 0.5)
    # This is equivalent to 0.5 - max(|c_0|, |c_1|, |c_2|) ONLY IF we consider the distance to the boundary of the cube [-0.5, 0.5]^3.
    # Wait. The distance from (x,y,z) to the boundary of [-0.5, 0.5]^3 is:
    # min(0.5-x, x+0.5, 0.5-y, y+0.5, 0.5-z, z+0.5)
    # This is indeed 0.5 - max(|x|, |y|, |z|).
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_pair / 2.0, min_wall)


def _init_cubic():
    """Initial cubic grid placement with slight jitter to break symmetry."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    # m=4 for N=50. 4x4x4=64 points. We take the first 50.
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    # Add small jitter to avoid perfectly symmetric local minima traps
    centers += np.random.randn(N, 3) * 0.001
    return centers


def _simulated_annealing(centers, start_time, time_budget):
    """Simulated annealing to maximize the minimum clearance."""
    best_centers = centers.copy()
    best_score = _score(centers)
    current_centers = centers.copy()
    current_score = best_score
    
    T = 0.01  # initial temperature
    cooling = 0.995
    step_size = 0.05
    
    while T > 1e-6 and (time.time() - start_time) < time_budget:
        # Perturb one or two random centers
        n_perturb = 1 if np.random.rand() < 0.7 else 2
        idx = np.random.choice(N, size=n_perturb, replace=False)
        
        new_centers = current_centers.copy()
        for i in idx:
            new_centers[i] += np.random.randn(3) * step_size
            # Clamp to cube with small margin to keep them inside
            new_centers[i] = np.clip(new_centers[i], -0.49, 0.49)
        
        new_score = _score(new_centers)
        delta = new_score - current_score
        
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            current_centers = new_centers
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_centers = current_centers.copy()
        
        T *= cooling
        step_size = max(0.005, step_size * 0.999)
        
        # Periodic check to avoid spending too much time on a single iteration if it's slow
        if time.time() - start_time > time_budget:
            break
    
    return best_centers, best_score


def _objective_and_grad(x, beta):
    """
    Objective and gradient for L-BFGS-B.
    Minimizes the negative soft-min of clearances.
    Clearances: pairwise distances / 2, and wall distances.
    """
    centers = x.reshape(N, 3)
    
    # 1. Wall distances
    # g_wall_i = 0.5 - max(|c_i|)
    # This is non-smooth at |x|=|y| etc. 
    # For gradient, we can use the subgradient or just treat the 6 wall distances separately?
    # The constraint is min(0.5-x, 0.5+x, ...). 
    # Actually, the radius is limited by min(all pair dists/2, all wall dists).
    # Let's define the pool of constraints g_k.
    # g_pair_ij = d_ij / 2
    # g_wall_i = min(0.5 - c_i0, c_i0 + 0.5, 0.5 - c_i1, c_i1 + 0.5, 0.5 - c_i2, c_i2 + 0.5)
    # To make it smooth and differentiable, it is better to include all 6 wall distances for each point in the pool?
    # If we include all 6, the min of the pool will correctly identify the closest wall.
    # Let's do that. It increases the size of g from N + N(N-1)/2 to 6N + N(N-1)/2.
    # N=50 -> 300 + 1225 = 1525 constraints. This is fine.
    
    # Wall distances for each point and each of 6 faces
    # Faces: x=0.5, x=-0.5, y=0.5, y=-0.5, z=0.5, z=-0.5
    # dists:
    # 0.5 - c[:,0]
    # c[:,0] + 0.5
    # 0.5 - c[:,1]
    # c[:,1] + 0.5
    # 0.5 - c[:,2]
    # c[:,2] + 0.5
    
    wall_dists = np.concatenate([
        0.5 - centers[:, 0],
        centers[:, 0] + 0.5,
        0.5 - centers[:, 1],
        centers[:, 1] + 0.5,
        0.5 - centers[:, 2],
        centers[:, 2] + 0.5
    ])
    
    # Pairwise distances
    # We need upper triangle to avoid double counting in the sum, but for min it doesn't matter.
    # However, for gradient accumulation, double counting would double the weight.
    # Standard practice: use unique pairs.
    iu = np.triu_indices(N, k=1)
    diff = centers[iu[0]] - centers[iu[1]]
    dists = np.linalg.norm(diff, axis=1)
    pair_dists = dists / 2.0
    
    # Combine all constraints
    g_all = np.concatenate([pair_dists, wall_dists])
    
    # Soft min: -1/beta * log(sum(exp(-beta * g)))
    # For numerical stability, use logsumexp
    # val = -1/beta * logsumexp(-beta * g_all)
    neg_beta_g = -beta * g_all
    log_sum_exp = logsumexp(neg_beta_g)
    val = -log_sum_exp / beta
    
    # Gradient
    # d(val)/dx = -1/beta * d(logsumexp)/dx
    # d(logsumexp(z))/dz = softmax(z)
    # z = -beta * g
    # dz/dx = -beta * dg/dx
    # d(val)/dx = -1/beta * sum(softmax(z)_k * dz_k/dx)
    #            = -1/beta * sum(softmax(z)_k * (-beta * dg_k/dx))
    #            = sum(softmax(z)_k * dg_k/dx)
    
    # Weights = softmax(-beta * g_all)
    weights = np.exp(neg_beta_g - log_sum_exp)
    
    grad_x = np.zeros_like(x)
    
    # Pairwise part
    # g_pair_k = 0.5 * ||c_i - c_j||
    # d(g_pair_k)/dc_i = 0.5 * (c_i - c_j) / ||c_i - c_j||
    # d(g_pair_k)/dc_j = 0.5 * (c_j - c_i) / ||c_i - c_j||
    
    # weights[0 : len(pair_dists)] correspond to pair_dists
    w_pairs = weights[:len(pair_dists)]
    
    # Vectorized gradient accumulation for pairs
    # This is tricky to vectorize fully without a loop or complex indexing.
    # N_pairs = 1225. A loop is fast enough (~1ms).
    for k in range(len(pair_dists)):
        i, j = iu[0][k], iu[1][k]
        w = w_pairs[k]
        if w < 1e-12: continue
        
        d_ij = dists[k]
        if d_ij > 1e-12:
            diff_ij = centers[i] - centers[j]
            grad_vec = 0.5 * diff_ij / d_ij
            
            # grad_x is flat (3N,)
            # c_i is at index i*3 : i*3+3
            # c_j is at index j*3 : j*3+3
            grad_x[i*3:i*3+3] += w * grad_vec
            grad_x[j*3:j*3+3] -= w * grad_vec
            
    # Wall part
    # g_wall_m corresponds to wall_dists[m]
    # wall_dists layout:
    # 0..N-1: 0.5 - c[:,0]  -> grad c[:,0] = -1
    # N..2N-1: c[:,0] + 0.5 -> grad c[:,0] = 1
    # 2N..3N-1: 0.5 - c[:,1] -> grad c[:,1] = -1
    # 3N..4N-1: c[:,1] + 0.5 -> grad c[:,1] = 1
    # 4N..5N-1: 0.5 - c[:,2] -> grad c[:,2] = -1
    # 5N..6N-1: c[:,2] + 0.5 -> grad c[:,2] = 1
    
    w_start = len(pair_dists)
    w_walls = weights[w_start:]
    
    # Accumulate gradients for walls
    # grad_x layout: [c0x, c0y, c0z, c1x, c1y, c1z, ...]
    # c_i, k is at index i*3 + k
    
    # x-coords (k=0)
    # w_walls[0:N] for 0.5-cx, grad -1
    # w_walls[N:2N] for cx+0.5, grad 1
    grad_x[0::3] -= w_walls[0:N]
    grad_x[0::3] += w_walls[N:2*N]
    
    # y-coords (k=1)
    # w_walls[2N:3N] for 0.5-cy, grad -1
    # w_walls[3N:4N] for cy+0.5, grad 1
    grad_x[1::3] -= w_walls[2*N:3*N]
    grad_x[1::3] += w_walls[3*N:4*N]
    
    # z-coords (k=2)
    # w_walls[4N:5N] for 0.5-cz, grad -1
    # w_walls[5N:6N] for cz+0.5, grad 1
    grad_x[2::3] -= w_walls[4*N:5*N]
    grad_x[2::3] += w_walls[5*N:6*N]
    
    # We want to MINIMIZE -val.
    # So objective = -val
    # Gradient = -grad_x
    return -val, -grad_x


def _local_refinement(centers, start_time, time_budget):
    """Local optimization using scipy minimize on negative soft-min score."""
    best_centers = centers.copy()
    best_score = _score(centers)
    
    # Anneal beta from low to high
    # Low beta: smooth, broad minima. High beta: sharp, precise minima.
    beta_schedule = [50.0, 100.0, 200.0, 500.0, 1000.0]
    
    current_centers = centers.copy()
    
    for beta in beta_schedule:
        if (time.time() - start_time) > time_budget:
            break
        
        x0 = current_centers.ravel()
        
        try:
            # Use L-BFGS-B
            # maxiter can be higher if we have time, but we check time inside the loop?
            # scipy minimize doesn't have a callback that easily stops based on wall clock for the *next* iteration,
            # but we can check after each run.
            # To be safe, we limit maxiter to a reasonable number that finishes quickly.
            result = minimize(_objective_and_grad, x0, method='L-BFGS-B', 
                             args=(beta,),
                             options={'maxiter': 500, 'maxls': 50, 'ftol': 1e-12, 'gtol': 1e-10})
            
            new_centers = result.x.reshape(N, 3)
            # Clip to ensure validity, though the optimizer should keep them in bounds if initialized well
            # The wall constraints push them inwards.
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            new_score = _score(new_centers)
            
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers.copy()
                current_centers = new_centers.copy()
            else:
                # If it didn't improve, keep the previous best as start for next beta
                # current_centers remains best_centers? No, if it failed to improve, 
                # the result might be worse. We should start next beta from best_centers.
                current_centers = best_centers.copy()
                
        except Exception:
            pass
            
        # Check time
        if (time.time() - start_time) > time_budget:
            break
            
    return best_centers, best_score


def construct():
    start_time = time.time()
    
    # Start with cubic grid
    centers = _init_cubic()
    best_centers = centers.copy()
    best_score = _score(centers)
    
    # Phase 1: Simulated Annealing
    # Give SA a decent chunk of time, but not too much as local refinement is more precise.
    time_budget_sa = 30.0
    centers, score = _simulated_annealing(centers, start_time, time_budget_sa)
    if score > best_score:
        best_score = score
        best_centers = centers.copy()
    
    # Phase 2: Local Refinement
    # Use remaining time minus a safety buffer
    elapsed = time.time() - start_time
    remaining_time = DEADLINE - elapsed - 5.0  # 5s buffer
    if remaining_time > 5.0:
        centers, score = _local_refinement(centers, start_time, DEADLINE - 5.0)
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
    
    # Final check
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return best_centers
        
    return best_centers
