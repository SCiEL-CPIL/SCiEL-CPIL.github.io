import numpy as np
import time
from scipy.optimize import minimize

N = 100
HALF = 0.5

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -HALF) or np.any(centers > HALF):
        return 0.0
        
    # Wall clearance
    clearance = (HALF - np.abs(centers)).min()
    if clearance <= 0:
        return 0.0
        
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Get upper triangle indices (i < j)
    iu = np.triu_indices(N, k=1)
    min_pair_dist = dists[iu].min()
    
    # Radius is limited by half the minimum pairwise distance and the wall clearance
    return min(min_pair_dist * 0.5, clearance)

def _energy_and_grad(x, target_r, beta):
    """
    Energy function for optimization using a soft-min approximation for the radius.
    Instead of minimizing violations of a fixed target_r, we maximize the soft-min radius.
    
    The soft-min radius is approximated as:
    r_soft = - (1/beta) * log( sum( exp(-beta * d_ij) for i<j) + exp(-beta * w_ik) for i,k ) )
    
    To maximize r_soft, we minimize -r_soft.
    However, the exponential sum can overflow for large beta. We use the log-sum-exp trick.
    
    Let A = [ -beta * d_ij (for all i<j), -beta * w_ik (for all i,k) ]
    r_soft = - (1/beta) * log( sum( exp(A) ) )
    
    Gradient of -r_soft:
    d(-r_soft)/dx = (1/beta) * d(log(sum(exp(A)))/dx)
                  = (1/beta) * sum( exp(A_k) / sum(exp(A)) * dA_k/dx )
                  = sum( p_k * dA_k/dx )
    where p_k = exp(A_k) / sum(exp(A)) is the softmax probability.
    
    This is equivalent to a weighted sum of gradients of the individual distances/clearances.
    The weights are exponentially concentrated on the "tightest" constraints.
    
    x: flat array of length 3N.
    target_r: not used in this formulation, kept for signature compatibility if needed, but effectively ignored.
    beta: temperature parameter.
    """
    c = x.reshape(N, 3)
    
    # --- Compute Distances and Wall Clearances ---
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Wall clearances
    wall_clearances = HALF - np.abs(c)
    
    # --- Build the Log-Sum-Exp ---
    # We want to compute LSE = log( sum( exp(-beta * d) ) )
    # To avoid overflow, we subtract the max value.
    # Let M = max( -beta * d_ij, -beta * w_ik ) = -beta * min( d_ij, w_ik )
    # Then sum( exp(A) ) = exp(M) * sum( exp(A - M) )
    # log(sum(exp(A))) = M + log(sum(exp(A - M)))
    
    # Flatten pairwise distances (upper triangle)
    iu = np.triu_indices(N, k=1)
    d_pairs = dists[iu]
    
    # Flatten wall clearances
    w_flat = wall_clearances.flatten()
    
    # Combine all values
    # We need to be careful with memory. N=100 -> 4950 pairs + 300 walls = 5250 values.
    # This is small.
    
    # Values for exponent
    # A_k = -beta * value_k
    # We want to maximize the minimum value.
    # So we look at the smallest values.
    
    # Let's define the "cost" as the negative of the distance/clearance?
    # No, standard log-sum-exp for min is:
    # min(x) ≈ -1/beta * log( sum( exp(-beta * x) ) )
    
    # So we compute S = sum( exp(-beta * x_k) )
    # where x_k are the distances and clearances.
    
    # To be numerically stable:
    # Let x_min = min(x_k)
    # S = exp(-beta * x_min) * sum( exp(-beta * (x_k - x_min)) )
    # log(S) = -beta * x_min + log( sum( exp(-beta * (x_k - x_min)) ) )
    # r_soft = -1/beta * log(S) = x_min - 1/beta * log( sum( exp(-beta * (x_k - x_min)) ) )
    
    # This is stable if beta * (x_k - x_min) is not too large positive.
    # Since x_k >= x_min, x_k - x_min >= 0.
    # If beta is large, exp(-beta * (x_k - x_min)) will be close to 0 for x_k > x_min.
    # The sum will be dominated by the terms where x_k is close to x_min.
    
    # Let's combine the arrays
    # x_all: array of all distances and wall clearances
    x_pairs = d_pairs
    x_walls = w_flat
    
    x_all = np.concatenate([x_pairs, x_walls])
    
    # Compute x_min
    x_min = np.min(x_all)
    
    # Compute shifted values
    x_shifted = x_all - x_min
    
    # Compute exponentials
    # exp(-beta * x_shifted)
    # If x_shifted is large, this underflows to 0, which is fine.
    # If x_shifted is 0, exp(0) = 1.
    
    exp_vals = np.exp(-beta * x_shifted)
    
    sum_exp = np.sum(exp_vals)
    
    if sum_exp == 0:
        # Should not happen since x_min is in the array, so at least one term is exp(0)=1
        r_soft = x_min
        # Gradient is 0? No, we need to handle this edge case carefully.
        # But practically, sum_exp >= 1.
        grad = np.zeros_like(c)
        return -r_soft, grad.flatten()
    
    # r_soft = x_min - (1/beta) * log(sum_exp)
    r_soft = x_min - (1.0 / beta) * np.log(sum_exp)
    
    # Energy is -r_soft
    energy = -r_soft
    
    # --- Gradient ---
    # d(energy)/dx = -d(r_soft)/dx
    # r_soft = x_min - (1/beta) * log( sum( exp(-beta * (x_k - x_min)) ) )
    # Note: x_min depends on x. This makes the derivative slightly tricky.
    # However, the standard soft-min approximation assumes x_min is fixed or uses the full log-sum-exp derivative.
    # Let's use the full log-sum-exp derivative formula:
    # f(x) = -1/beta * log( sum( exp(-beta * x_k) ) )
    # df/dx_i = (1/beta) * [ sum( exp(-beta * x_k) * (-beta * dx_k/dx_i) ) / sum( exp(-beta * x_k) ) ]
    # df/dx_i = - sum( p_k * dx_k/dx_i )
    # where p_k = exp(-beta * x_k) / sum( exp(-beta * x_k) )
    
    # In our shifted formulation:
    # p_k = exp(-beta * (x_k - x_min)) / sum( exp(-beta * (x_k - x_min)) ) = exp_vals / sum_exp
    
    p_vals = exp_vals / sum_exp
    
    # Now we need dx_k/dx_i for each component of x (the centers).
    # x_k corresponds to either a pair distance d_ij or a wall clearance w_ik.
    
    # Gradient contribution from pairs:
    # d_ij = ||c_i - c_j||
    # d(d_ij)/d(c_i) = (c_i - c_j) / d_ij
    # d(d_ij)/d(c_j) = (c_j - c_i) / d_ij
    
    # Gradient contribution from walls:
    # w_ik = HALF - |c_ik|
    # d(w_ik)/d(c_ik) = -sign(c_ik)
    
    # Total gradient of r_soft:
    # d(r_soft)/dx = - sum_k p_k * (dx_k/dx)
    
    # Let's compute the gradient of x_all with respect to c.
    # This is a sparse operation.
    
    grad_r_soft = np.zeros_like(c)
    
    # --- Pair Contributions ---
    # p_pairs: probabilities for the pair distances
    p_pairs = p_vals[:len(x_pairs)]
    
    # We need to accumulate gradients for each point i.
    # For each pair (i, j) with weight p_ij:
    # d(d_ij)/d(c_i) = (c_i - c_j) / d_ij
    # d(d_ij)/d(c_j) = (c_j - c_i) / d_ij
    
    # We have the flattened p_pairs corresponding to the upper triangle indices iu.
    # iu[0] are the row indices (i), iu[1] are the column indices (j).
    
    # Let's vectorize this accumulation.
    # It's hard to fully vectorize the accumulation into grad_r_soft without loops or advanced indexing.
    # Given N=100, 4950 pairs, a loop over pairs might be slow in Python.
    # Let's try to use np.add.at or similar.
    
    # diff[i, j] = c[i] - c[j]
    # unit_vecs[i, j] = diff[i, j] / d_ij
    
    # We need to add p_ij * unit_vecs[i, j] to grad_r_soft[i]
    # and -p_ij * unit_vecs[i, j] to grad_r_soft[j]
    # (Note: d(r_soft)/dx = - sum p_k dx_k/dx. So we are summing -p_k * grad_x_k)
    # Wait, let's re-verify signs.
    # f(x) = -r_soft
    # df/dx = -d(r_soft)/dx
    # d(r_soft)/dx = - sum_k p_k * (dx_k/dx)
    # So df/dx = sum_k p_k * (dx_k/dx)
    
    # So we want to accumulate:
    # grad_energy += p_ij * (d(d_ij)/d(c_i)) for point i
    # grad_energy += p_ij * (d(d_ij)/d(c_j)) for point j
    
    # d(d_ij)/d(c_i) = (c_i - c_j) / d_ij
    # d(d_ij)/d(c_j) = (c_j - c_i) / d_ij = -(c_i - c_j) / d_ij
    
    # So:
    # grad_energy[i] += p_ij * (c_i - c_j) / d_ij
    # grad_energy[j] += p_ij * (c_j - c_i) / d_ij
    
    # Let's compute the terms.
    # We have iu[0], iu[1] for the active pairs.
    # We need the corresponding unit vectors.
    
    # Get the differences for the pairs
    # diff[iu[0], iu[1]] gives c[i] - c[j] for each pair
    pair_diffs = diff[iu[0], iu[1]] # Shape (num_pairs, 3)
    
    # Get the distances
    pair_dists = dists[iu[0], iu[1]]
    
    # Safe division
    safe_pair_dists = np.where(pair_dists > 1e-12, pair_dists, 1.0)
    pair_unit_vecs = pair_diffs / safe_pair_dists[:, np.newaxis]
    
    # Weighted unit vectors
    weighted_vecs = p_pairs[:, np.newaxis] * pair_unit_vecs
    
    # Accumulate into grad_energy
    # For each pair k with indices (i, j) and vector v:
    # grad_energy[i] += v
    # grad_energy[j] -= v
    
    # Using np.add.at
    np.add.at(grad_r_soft, iu[0], weighted_vecs)
    np.add.at(grad_r_soft, iu[1], -weighted_vecs)
    
    # --- Wall Contributions ---
    p_walls = p_vals[len(x_pairs):]
    
    # Wall clearances w_ik = HALF - |c_ik|
    # d(w_ik)/d(c_ik) = -sign(c_ik)
    # So we accumulate:
    # grad_energy[i, k] += p_ik * (-sign(c_ik))
    
    # p_walls is flattened. We need to reshape it back to (N, 3).
    p_walls_reshaped = p_walls.reshape(N, 3)
    
    signs = np.sign(c)
    
    # d(w)/dc = -sign(c)
    wall_grad_terms = p_walls_reshaped * (-signs)
    
    grad_r_soft += wall_grad_terms
    
    # Final energy gradient
    # energy = -r_soft
    # grad_energy = -grad_r_soft ?
    # Let's re-check.
    # r_soft = -1/beta log(sum exp(-beta x))
    # d(r_soft)/dx = - sum p_k dx_k/dx
    # energy = -r_soft
    # d(energy)/dx = sum p_k dx_k/dx
    
    # So grad_energy = grad_r_soft (as computed above, which is sum p_k dx_k/dx)
    
    # Wait, in the pair accumulation:
    # I accumulated p_ij * (c_i - c_j)/d_ij into grad_r_soft[i].
    # This is p_ij * d(d_ij)/d(c_i).
    # So grad_r_soft is indeed sum p_k dx_k/dx.
    # Therefore, grad_energy = grad_r_soft.
    
    # Let's double check the wall part.
    # d(w_ik)/d(c_ik) = -sign(c_ik).
    # I added p_ik * (-sign(c_ik)) to grad_r_soft[i, k].
    # This is p_ik * d(w_ik)/d(c_ik).
    # Correct.
    
    total_grad = grad_r_soft.flatten()
    
    return energy, total_grad

def _optimize_soft_min(init_pts, beta_start, beta_end, max_iter, time_limit, gtol=1e-8):
    """
    Optimize the soft-min radius objective.
    """
    t_start = time.time()
    best_x = init_pts.flatten()
    best_energy = _energy_and_grad(best_x, 0.0, beta_start)[0]
    
    # We will increase beta over time to sharpen the approximation.
    # Start with low beta (smooth), end with high beta (sharp).
    
    num_steps = 10 # Number of beta annealing steps
    
    for step in range(num_steps):
        if time.time() - t_start > time_limit:
            break
            
        # Interpolate beta
        alpha = step / max(1, num_steps - 1)
        beta = beta_start * (beta_end / beta_start) ** alpha
        
        # Define objective for this beta
        def objective(x):
            if time.time() - t_start > time_limit:
                raise TimeoutError()
            return _energy_and_grad(x, 0.0, beta)
        
        try:
            result = minimize(
                objective,
                best_x,
                method='BFGS',
                jac=True,
                options={'maxiter': max_iter // num_steps, 'gtol': gtol, 'disp': False}
            )
            
            if np.all(np.isfinite(result.x)):
                current_energy = result.fun
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_x = result.x
        except TimeoutError:
            break
        except Exception:
            # If it fails, keep the previous best
            pass
    
    return best_x.reshape(N, 3)

def construct():
    start_time = time.time()
    deadline = 145.0 # Leave 5s margin for 150s limit
    
    # 1. Initialization: 5x5x4 grid
    xs = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    ys = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    zs = np.array([-0.375, -0.125, 0.125, 0.375])
    
    centers = []
    for z in zs:
        for y in ys:
            for x in xs:
                centers.append([x, y, z])
    centers = np.array(centers)
    
    if len(centers) != N:
        # Fallback
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        centers = grid[:N]

    best_pts = centers.copy()
    best_r = _compute_radius(best_pts)
    
    rng = np.random.RandomState(42)
    
    # Strategy: Soft-min annealing
    # Start with a smooth objective (low beta) and gradually increase beta.
    # This helps escape local minima and find a better global structure.
    
    # Phase 1: Annealing from low to high beta
    beta_start = 10.0
    beta_end = 500.0
    
    # We will run the annealing process a few times with different noise levels
    # to explore different basins.
    
    num_restarts = 3
    
    for restart in range(num_restarts):
        if time.time() > deadline - 20.0:
            break
        
        # Add noise to break symmetry
        # For the first restart, use the grid. For subsequent ones, use the best found so far.
        if restart == 0:
            init_pts = best_pts.copy()
        else:
            noise = rng.normal(0, 0.01, size=(N, 3))
            init_pts = np.clip(best_pts + noise, -0.49, 0.49)
        
        remaining_time = deadline - (time.time() - start_time) - 10.0
        if remaining_time < 10.0:
            break
            
        # Run soft-min optimization
        opt_pts = _optimize_soft_min(
            init_pts,
            beta_start,
            beta_end,
            max_iter=1000,
            time_limit=remaining_time,
            gtol=1e-8
        )
        
        if np.all(np.isfinite(opt_pts)):
            opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
            r_opt = _compute_radius(opt_pts)
            
            if r_opt > best_r:
                best_r = r_opt
                best_pts = opt_pts.copy()
    
    # Phase 2: Hard-constraint polishing
    # Now that we have a good configuration, we can try to push the radius slightly higher
    # using the hard-constraint violation method (from the original program) which is more precise
    # near the optimum.
    
    if time.time() < deadline - 10.0 and best_r > 0.1:
        # Try to increase the radius incrementally
        current_r = best_r
        current_pts = best_pts.copy()
        
        # Define a range of target radii to try
        # Best known is ~0.11138
        # We will try to push towards 0.112 or higher.
        
        targets = []
        t = current_r + 0.0005
        while t < current_r + 0.005:
            targets.append(t)
            t += 0.0005
        
        for target_r in targets:
            if time.time() > deadline - 5.0:
                break
            
            # Add small noise to break symmetry
            noise = rng.normal(0, 0.001, size=(N, 3))
            init_pts = np.clip(current_pts + noise, -0.49, 0.49)
            
            remaining = deadline - (time.time() - start_time) - 2.0
            if remaining < 2.0:
                break
            
            # Use BFGS with hard constraints
            def objective_hard(x):
                if time.time() > deadline - 2.0:
                    raise TimeoutError()
                return _energy_and_grad_hard(x, target_r)
            
            try:
                result = minimize(
                    objective_hard,
                    init_pts.flatten(),
                    method='BFGS',
                    jac=True,
                    options={'maxiter': 500, 'gtol': 1e-9, 'disp': False}
                )
                opt_pts = result.x.reshape(N, 3)
                
                if np.all(np.isfinite(opt_pts)):
                    opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
                    r_opt = _compute_radius(opt_pts)
                    
                    if r_opt > best_r:
                        best_r = r_opt
                        best_pts = opt_pts.copy()
                        current_pts = opt_pts.copy()
            except TimeoutError:
                break
            except Exception:
                pass

    return best_pts

def _energy_and_grad_hard(x, target_r):
    """
    Energy function for BFGS optimization with hard constraints.
    x: flat array of length 3N.
    target_r: desired radius.
    Returns: (energy, gradient).
    Energy is the sum of squared violations of the constraint that all spheres
    of radius target_r are disjoint and inside the cube.
    """
    c = x.reshape(N, 3)
    
    # --- Pairwise Part ---
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dists, 1.0)
    
    # Active pairs: those closer than 2 * target_r
    violations = 2.0 * target_r - dists
    
    # Mask out diagonal and inactive pairs
    active_mask = violations > 1e-10
    active_mask[np.diag_indices_from(active_mask)] = False
    
    # Compute gradients
    grad = np.zeros_like(c)
    
    weights = np.where(active_mask, violations, 0.0)
    
    safe_dists = np.where(dists > 1e-12, dists, 1.0)
    unit_vecs = diff / safe_dists[:, :, np.newaxis]
    
    terms = weights[:, :, np.newaxis] * unit_vecs
    
    grad_pair = np.sum(terms, axis=1)
    
    # --- Wall Part ---
    wall_violations = target_r - (0.5 - np.abs(c))
    wall_active = wall_violations > 1e-10
    
    signs = np.sign(c)
    
    grad_wall = np.where(wall_active, wall_violations * signs, 0.0)
    
    total_grad = (grad_pair + grad_wall).flatten()
    
    # Energy
    energy_pair = 0.5 * np.sum(np.where(active_mask, violations**2, 0.0))
    energy_wall = 0.5 * np.sum(np.where(wall_active, wall_violations**2, 0.0))
    total_energy = energy_pair + energy_wall
    
    return total_energy, total_grad

# EVOLVE-BLOCK-END
