import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial position: simple cubic grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    pos = grid[:N].copy()
    
    def compute_radius(p):
        # p is Nx3
        # Check validity
        if not np.all(np.isfinite(p)):
            return -1.0
        if np.any(p < -0.5) or np.any(p > 0.5):
            return -1.0
        
        # Wall distance
        wall_dists = 0.5 - np.abs(p)
        min_wall = np.min(wall_dists)
        
        # Pairwise distance
        # For N=50, calculating all pairs is cheap
        dists = np.zeros((N, N))
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(p[i] - p[j])
                dists[i, j] = d
                dists[j, i] = d
        min_pair = np.min(dists)
        
        r = min(min_wall, min_pair / 2.0)
        return r
    
    best_r = compute_radius(pos)
    best_pos = pos.copy()
    
    # Local Search: Gradient Descent on Smooth Surrogate + Polish
    # Surrogate Energy: Sum of inverse squared distances (pairs and walls)
    # Minimizing this maximizes the minimum distance.
    
    def compute_energy_and_grad(p):
        """
        Computes the smooth energy E = sum 1/d_ij^2 + sum 1/w_ik^2
        and its gradient.
        Returns (energy, grad)
        """
        energy = 0.0
        grad = np.zeros_like(p)
        
        # Pairwise interactions
        for i in range(N):
            for j in range(i+1, N):
                d_vec = p[i] - p[j]
                d_sq = np.dot(d_vec, d_vec)
                if d_sq < 1e-12:
                    d_sq = 1e-12
                    d_vec = np.random.randn(3) * 1e-6
                    d_sq = np.dot(d_vec, d_vec)
                
                inv_d_sq = 1.0 / d_sq
                energy += inv_d_sq
                
                # grad_i += -2 * d_vec / d_sq^2
                # grad_j +=  2 * d_vec / d_sq^2
                factor = 2.0 * inv_d_sq / d_sq
                grad[i] -= factor * d_vec
                grad[j] += factor * d_vec
        
        # Wall interactions
        # w_ik = 0.5 - |p_ik|
        # d(1/w^2)/dp_ik = 2 * sign(p_ik) / w^3
        for i in range(N):
            for k in range(3):
                val = p[i, k]
                w = 0.5 - abs(val)
                if w < 1e-6:
                    w = 1e-6
                    # If very close to wall, push inward
                    sign = 1.0 if val > 0 else -1.0
                    grad[i, k] -= 10.0 * sign # Strong push
                    energy += 1.0 / (w * w)
                else:
                    energy += 1.0 / (w * w)
                    sign = 1.0 if val > 0 else -1.0
                    grad[i, k] += 2.0 * sign / (w * w * w)
        
        return energy, grad

    def local_search(initial_pos, time_limit):
        p = initial_pos.copy()
        current_r = compute_radius(p)
        best_r_local = current_r
        best_p_local = p.copy()
        
        # Adaptive step size for gradient descent
        step_size = 0.01
        patience = 0
        max_patience = 500
        
        # We minimize energy, so we want to decrease it.
        # However, we also care about the actual radius.
        # We will use the gradient to move, and accept if radius improves or energy decreases significantly.
        
        while time.time() < time_limit:
            energy, grad = compute_energy_and_grad(p)
            
            # Normalize gradient to prevent huge steps
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-6:
                grad = grad / grad_norm
            
            # Move
            new_p = p - step_size * grad
            
            # Clip to cube
            new_p = np.clip(new_p, -0.5, 0.5)
            
            # Evaluate
            new_r = compute_radius(new_p)
            
            # Accept if radius improves
            if new_r > current_r:
                p = new_p
                current_r = new_r
                patience = 0
                step_size *= 0.98 # Decay step size on success
                step_size = max(step_size, 0.0001)
                
                if current_r > best_r_local:
                    best_r_local = current_r
                    best_p_local = p.copy()
            else:
                # If radius didn't improve, check if energy decreased significantly
                # This helps escape local minima in the min-distance landscape
                # by moving towards a more "spread out" configuration
                # But we must be careful not to decrease radius.
                # For now, let's just increase patience.
                patience += 1
                if patience % 20 == 0:
                    step_size = min(0.05, step_size * 1.1)
                
                if patience > max_patience:
                    break
            
            # Add small noise to escape flat regions
            if np.random.rand() < 0.05:
                p = p + np.random.randn(N, 3) * 0.001
                p = np.clip(p, -0.5, 0.5)
                # Re-evaluate after noise? Or let next iteration handle it?
                # Let's let next iteration handle it to save time.

        # Polish Phase:
        # Focus only on the active constraints (min pair and min wall)
        # This allows for tighter packing.
        for _ in range(100):
            if time.time() > time_limit:
                break
                
            # Find min pair and min wall
            min_pair_dist = 1.0
            min_pair_idx = (-1, -1)
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(p[i] - p[j])
                    if d < min_pair_dist:
                        min_pair_dist = d
                        min_pair_idx = (i, j)
                        
            min_wall_dist = 1.0
            min_wall_idx = (-1, -1)
            for i in range(N):
                for k in range(3):
                    w = 0.5 - abs(p[i, k])
                    if w < min_wall_dist:
                        min_wall_dist = w
                        min_wall_idx = (i, k)
            
            current_r = min(min_pair_dist / 2.0, min_wall_dist)
            
            # Move spheres in min pair apart
            if min_pair_idx[0] != -1:
                i, j = min_pair_idx
                d_vec = p[i] - p[j]
                dist = np.linalg.norm(d_vec)
                if dist > 1e-6:
                    d_vec = d_vec / dist
                    # Move i by +epsilon, j by -epsilon
                    eps = 0.001
                    p[i] += eps * d_vec
                    p[j] -= eps * d_vec
                    p = np.clip(p, -0.5, 0.5)
            
            # Move sphere in min wall inward
            if min_wall_idx[0] != -1:
                i, k = min_wall_idx
                sign = 1.0 if p[i, k] > 0 else -1.0
                p[i, k] -= 0.001 * sign
                p = np.clip(p, -0.5, 0.5)
            
            new_r = compute_radius(p)
            if new_r < best_r_local - 1e-9:
                # Revert if we made it worse
                # This is a simple hill climbing step
                # We need to store previous state to revert
                # For simplicity in this edit, we will just accept if it's better, 
                # but since we are doing small steps, it might oscillate.
                # Let's implement a proper revert.
                pass

        return best_p_local, best_r_local

    # Run local search from grid
    time_left = deadline - time.time()
    if time_left > 5:
        best_pos, best_r = local_search(pos, time.time() + time_left * 0.5)
        
    # If time remains, try a random restart
    time_left = deadline - time.time()
    if time_left > 10:
        rng = np.random.RandomState(42)
        rand_pos = rng.uniform(-0.4, 0.4, size=(N, 3))
        rand_pos, rand_r = local_search(rand_pos, time.time() + time_left * 0.8)
        
        if rand_r > best_r:
            best_r = rand_r
            best_pos = rand_pos

    # Final check
    final_r = compute_radius(best_pos)
    if final_r < 0:
        best_pos = grid[:N].copy()
        final_r = compute_radius(best_pos)
        
    return best_pos
