"""
Strategy: Differentiable Maximin Packing via Smooth Minimum Surrogate
This approach reframes the discrete "min distance" problem as a continuous
optimization problem. We maximize a smooth approximation of the minimum
distance (using a soft-min function) using gradient ascent. This allows
efficient vectorized updates for all 100 spheres simultaneously, rather than
moving one sphere at a time.

Phases:
1. Force-Directed Relaxation: Quickly spread spheres out using repulsive
   forces to escape the initial grid's local minima and establish a rough
   packing.
2. Smooth Gradient Ascent: Maximize the soft-min radius. The gradient of
   the soft-min w.r.t. position is well-defined and can be computed in
   O(N^2) vectorized operations. We use Adam optimizer for stability.
3. Discrete Tightening: Once the smooth solution is close to the optimum,
   we identify the critical pairs (those determining the true minimum) and
   perform a final local adjustment to ensure the exact discrete minimum
   is maximized.
"""

import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # 1. Initialization: Simple grid
    # We use a slightly perturbed grid to break symmetry
    rng = np.random.default_rng(42)
    m = int(np.ceil(N ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Add small noise to break perfect symmetry
    centers += rng.normal(0, 0.01, size=centers.shape)
    # Ensure within bounds
    centers = np.clip(centers, -0.49, 0.49)

    def compute_exact_radius(c):
        """Compute the true radius for evaluation."""
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Wall clearance
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    best_centers = centers.copy()
    best_radius = compute_exact_radius(best_centers)

    # Helper for smooth min
    # soft_min(x, beta) = -1/beta * log(sum(exp(-beta * x)))
    # This is differentiable. Gradient w.r.t x_i is proportional to exp(-beta * x_i) / sum(exp(-beta * x))
    
    def compute_smooth_radius_and_grad(c, beta):
        """
        Computes a smooth approximation of the radius and its gradient.
        We treat the radius as the minimum of:
        1. Pairwise distances / 2
        2. Wall distances
        
        We combine these into a single vector of 'distances' D of length N(N-1)/2 + 3N.
        Then we compute soft_min(D, beta).
        
        Returns: smooth_radius, gradients (N, 3)
        """
        # 1. Pairwise distances
        # diff[i, j] = c[i] - c[j]
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        # We only need upper triangle for unique pairs
        i_upper, j_upper = np.triu_indices(N, k=1)
        pair_dists = dists[i_upper, j_upper]
        pair_dists_half = pair_dists / 2.0
        
        # Gradients for pairwise part
        # d/dc_i (|c_i - c_j|/2) = (c_i - c_j) / (2 * |c_i - c_j|)
        # d/dc_j (|c_i - c_j|/2) = (c_j - c_i) / (2 * |c_i - c_j|)
        # Note: diff[i,j] is c_i - c_j.
        # grad_i += diff[i,j] / dist[i,j]
        # grad_j -= diff[i,j] / dist[i,j]
        
        # To vectorize gradient accumulation efficiently:
        # We can compute the unit vectors for all pairs.
        # dists is (N, N). Set diagonal to 1 to avoid div by 0.
        dists_safe = dists.copy()
        np.fill_diagonal(dists_safe, 1.0)
        unit_vecs = diff / dists_safe[:, :, np.newaxis] # (N, N, 3)
        
        # We need to sum contributions weighted by soft-min probabilities.
        # Let's construct the full vector of distances D.
        # D has length M = N(N-1)/2 + 3N
        wall_dists = 0.5 - np.abs(c) # (N, 3)
        wall_dists_flat = wall_dists.flatten() # (3N,)
        
        # Combine distances
        D = np.concatenate([pair_dists_half, wall_dists_flat])
        
        # Compute soft min
        # To avoid overflow, subtract max
        D_max = np.max(D)
        exp_neg_beta_D = np.exp(-beta * (D - D_max))
        sum_exp = np.sum(exp_neg_beta_D)
        
        smooth_rad = D_max - (1.0 / beta) * np.log(sum_exp)
        
        # Weights for each distance term in the soft min
        # w_k = exp(-beta * D_k) / sum_exp
        weights = exp_neg_beta_D / sum_exp
        
        # Split weights
        n_pairs = len(pair_dists_half)
        w_pairs = weights[:n_pairs]
        w_walls = weights[n_pairs:]
        
        # Compute Gradients
        grads = np.zeros_like(c)
        
        # Pairwise contributions
        # We need to map w_pairs back to the (N, N) structure or accumulate.
        # w_pairs corresponds to i_upper, j_upper.
        # For each pair (i, j) with weight w:
        # grad[i] += w * (c[i] - c[j]) / (2 * dist)
        # grad[j] += w * (c[j] - c[i]) / (2 * dist)
        
        # unit_vecs[i_upper, j_upper] is (c_i - c_j) / dist
        # So grad[i] += w * 0.5 * unit_vecs[i,j]
        # grad[j] += w * -0.5 * unit_vecs[i,j]
        
        # Vectorized accumulation using np.add.at
        # We have indices i_upper, j_upper and weights w_pairs
        # term_i = w_pairs * 0.5 * unit_vecs[i_upper, j_upper]
        # term_j = -w_pairs * 0.5 * unit_vecs[i_upper, j_upper]
        
        terms = 0.5 * w_pairs[:, np.newaxis] * unit_vecs[i_upper, j_upper]
        
        # Accumulate to i
        np.add.at(grads, i_upper, terms)
        # Accumulate to j (negative)
        np.add.at(grads, j_upper, -terms)
        
        # Wall contributions
        # d/dc (0.5 - |c|) is -sign(c)
        # Weights w_walls are for flattened (N, 3) array.
        # grads += w_walls.reshape(N, 3) * (-np.sign(c))
        wall_grads = -np.sign(c) * w_walls.reshape(N, 3)
        grads += wall_grads
        
        return smooth_rad, grads

    # Phase 1: Force Directed Relaxation
    # Objective: Maximize sum of pairwise distances (or minimize energy)
    # This helps spread spheres out quickly.
    print("Phase 1: Force Directed Relaxation", flush=True)
    current_c = centers.copy()
    
    for step in range(500):
        if time.time() > deadline - 60: break # Leave time for other phases
        
        # Compute forces
        diff = current_c[:, np.newaxis, :] - current_c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, 1.0)
        
        # Repulsive force: 1/d^2
        # F = diff / dist^3 (since diff/dist is unit vector, divide by dist^2)
        forces = diff / (dists[:, :, np.newaxis] ** 3)
        
        # Sum forces on each particle
        net_forces = np.sum(forces, axis=1)
        
        # Centering force: pull towards 0 to keep in cube
        # F_center = -current_c * lambda
        centering = -current_c * 0.5
        
        # Update
        lr = 0.01
        current_c += lr * (net_forces + centering)
        
        # Clip to bounds
        current_c = np.clip(current_c, -0.49, 0.49)
        
        # Check improvement
        r = compute_exact_radius(current_c)
        if r > best_radius:
            best_radius = r
            best_centers = current_c.copy()

    # Phase 2: Smooth Gradient Ascent
    print("Phase 2: Smooth Gradient Ascent", flush=True)
    current_c = best_centers.copy()
    
    # Adam Optimizer state
    m_v = np.zeros_like(current_c)
    v_v = np.zeros_like(current_c)
    t = 0
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    # Schedule for beta (sharpness of soft min)
    # Start low (smooth), end high (sharp, approximating true min)
    # beta = 10 -> 500
    
    start_beta = 10.0
    end_beta = 500.0
    max_steps = 2000
    
    for step in range(max_steps):
        if time.time() > deadline - 20: break
        
        # Interpolate beta
        progress = step / max_steps
        # Exponential ramp
        beta = start_beta * (end_beta / start_beta) ** progress
        
        smooth_rad, grads = compute_smooth_radius_and_grad(current_c, beta)
        
        # We want to maximize smooth_rad, so gradient is +grads
        # Adam update
        t += 1
        m_v = beta1 * m_v + (1 - beta1) * grads
        v_v = beta2 * v_v + (1 - beta2) * (grads ** 2)
        
        m_hat = m_v / (1 - beta1 ** t)
        v_hat = v_v / (1 - beta2 ** t)
        
        lr = 0.01
        update = lr * m_hat / (np.sqrt(v_hat) + eps)
        
        current_c += update
        
        # Clip to bounds
        current_c = np.clip(current_c, -0.49, 0.49)
        
        # Periodic check of exact radius
        if step % 50 == 0:
            r = compute_exact_radius(current_c)
            if r > best_radius:
                best_radius = r
                best_centers = current_c.copy()

    # Phase 3: Discrete Tightening
    # The smooth min might leave spheres slightly non-optimal relative to the
    # true discrete minimum. We identify the bottleneck pairs and adjust.
    print("Phase 3: Discrete Tightening", flush=True)
    current_c = best_centers.copy()
    
    # We will perform a local search focusing on the "tight" constraints.
    # Identify all pairs with distance < 2 * best_radius * 1.01
    # And walls with distance < best_radius * 1.01
    
    # Use a small step size gradient ascent on the exact radius?
    # Exact radius is non-differentiable.
    # Instead, we do a "nudge" search.
    
    # Find the current minimum radius and the pairs/walls achieving it.
    def get_bottlenecks(c, r):
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        
        # Pairs with dist/2 <= r + epsilon
        eps = 1e-5
        tight_pairs = []
        i_upper, j_upper = np.triu_indices(N, k=1)
        pd = dists[i_upper, j_upper] / 2.0
        mask = pd <= r + eps
        tight_pairs = list(zip(i_upper[mask], j_upper[mask]))
        
        # Walls
        wall_dists = 0.5 - np.abs(c)
        tight_walls = []
        for i in range(N):
            for k in range(3):
                if wall_dists[i, k] <= r + eps:
                    tight_walls.append((i, k))
                    
        return tight_pairs, tight_walls

    # Iterative refinement
    for iter in range(50):
        if time.time() > deadline - 5: break
        
        r = compute_exact_radius(current_c)
        if r > best_radius:
            best_radius = r
            best_centers = current_c.copy()
            
        tight_pairs, tight_walls = get_bottlenecks(current_c, r)
        
        # If no tight constraints, we can expand? 
        # Actually, if we are at a local max of the smooth function, 
        # the discrete min is likely close.
        
        # Try to move spheres to alleviate the tightest constraint.
        # Simple heuristic: move each sphere in the direction that increases
        # its distance to its nearest neighbors and walls.
        
        # Calculate "pressure" on each sphere from tight constraints.
        pressure = np.zeros_like(current_c)
        
        for i, j in tight_pairs:
            diff = current_c[i] - current_c[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-9:
                # Push i away from j, j away from i
                force = diff / dist
                pressure[i] += force
                pressure[j] -= force
        
        for i, k in tight_walls:
            # Push i away from wall k
            # Wall is at 0.5 or -0.5
            # If c[i,k] > 0, wall is at 0.5, push in -k direction
            # If c[i,k] < 0, wall is at -0.5, push in +k direction
            # If c[i,k] == 0, no specific wall direction, but distance is 0.5.
            # Actually, wall distance is 0.5 - |c|.
            # Gradient of wall distance w.r.t c is -sign(c).
            # We want to MAXIMIZE wall distance, so move in direction of gradient.
            # grad = -sign(c).
            # So move c += lr * (-sign(c)) ?
            # If c > 0, sign=1, grad=-1. Move left (towards center). Correct.
            # If c < 0, sign=-1, grad=1. Move right (towards center). Correct.
            
            # But wait, if we are tight against a wall, we want to move AWAY from the wall.
            # If c is near 0.5, we want to move towards -0.5.
            # The force from the pair logic above pushes away from neighbors.
            # For walls, we should push towards the center of the cube.
            # So force is -sign(c) * e_k.
            
            e_k = np.zeros(3)
            e_k[k] = 1.0
            force = -np.sign(current_c[i, k]) * e_k
            # If c is exactly 0, sign is 0. But then wall dist is 0.5, unlikely to be tight if r < 0.5.
            if np.linalg.norm(force) == 0:
                # Random small push? Or just skip.
                pass
            else:
                pressure[i] += force
                
        # Normalize pressure to prevent large jumps
        norms = np.linalg.norm(pressure, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        pressure = pressure / norms
        
        # Apply small update
        lr = 0.001
        current_c += lr * pressure
        current_c = np.clip(current_c, -0.49, 0.49)

    # Final validation
    final_radius = compute_exact_radius(best_centers)
    # Ensure no invalid states
    if np.any(np.isnan(best_centers)) or np.any(np.isinf(best_centers)):
        return centers # Fallback to initial grid
        
    if np.any(best_centers > 0.5) or np.any(best_centers < -0.5):
        best_centers = np.clip(best_centers, -0.5, 0.5)
        # Re-check radius after clip
        final_radius = compute_exact_radius(best_centers)
        
    return best_centers

# Test block (removed for submission)
if __name__ == "__main__":
    # c = construct()
    # print("Radius:", compute_exact_radius(c))
    pass
