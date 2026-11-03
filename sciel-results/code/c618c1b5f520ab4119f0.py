import numpy as np
import time
import math
from scipy.optimize import minimize

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def compute_radius(centers):
        """
        Computes the max radius for the given centers.
        Returns 0 if invalid.
        """
        if centers.shape != (N, 2):
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0
        # Check bounds
        if np.any(np.abs(centers) > 0.5 + 1e-9):
            return 0.0
        
        # Wall distance
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        
        # Pair distance
        # Use vectorized distance calculation
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        # Set diagonal to inf
        np.fill_diagonal(dist_sq, np.inf)
        min_dist_sq = np.min(dist_sq)
        min_pair = np.sqrt(min_dist_sq)
        
        r = min(min_wall, min_pair / 2.0)
        return max(0.0, r)

    def soft_min_neg(x, beta=100.0):
        """
        Computes -soft_min(x) = -(-1/beta * log(sum(exp(-beta * x))))
        This is a smooth approximation to -min(x).
        We want to maximize min(x), so we minimize -min(x).
        """
        # Clip x to prevent overflow in exp
        x_clipped = np.clip(x, -100, 100)
        min_val = np.min(x_clipped)
        # Shift for numerical stability
        shifted = x_clipped - min_val
        # sum(exp(-beta * shifted))
        # Since shifted >= 0, -beta * shifted <= 0, so exp is stable
        s = np.sum(np.exp(-beta * shifted))
        if s == 0:
            return min_val # Should not happen if beta > 0 and x finite
        return -(min_val - (1.0/beta) * np.log(s))

    def objective_and_grad(x, beta=100.0):
        """
        Objective to minimize: -soft_min(constraints)
        Constraints are distances to walls and between pairs.
        We want to maximize the minimum distance.
        
        x is shape (N, 2)
        """
        centers = x.reshape(N, 2)
        
        # 1. Wall distances
        # dist_i = 0.5 - |c_i|
        # We need min over i and k (x,y)
        wall_dists = 0.5 - np.abs(centers)
        # Flatten wall distances
        flat_walls = wall_dists.flatten()
        
        # 2. Pair distances
        # diff[i, j] = c_i - c_j
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(dist_sq, np.inf)
        
        # We want min dist, which is sqrt(min dist_sq).
        # Maximizing min dist is equivalent to maximizing min dist_sq.
        # Let's work with dist_sq for gradients? 
        # Actually, soft_min on distances is standard. 
        # But computing gradients for sqrt is annoying.
        # Let's use soft_min on dist_sq? No, the objective is radius.
        # Radius r = min( min_wall, min_pair/2 ).
        # Maximizing r is equivalent to maximizing min( 2*min_wall, min_pair ).
        # Let's define constraints C such that we maximize min(C).
        # C_walls = 2 * (0.5 - |c|) = 1 - 2|c|
        # C_pairs = dist (not squared)
        
        # To avoid sqrt in gradient, we can use a differentiable proxy or just compute it.
        # With N=200, N^2=40000 pairs. Sqrt is fast.
        dists = np.sqrt(dist_sq)
        flat_pairs = dists.flatten()
        
        # Combine all constraints
        # Note: We want to maximize the minimum of these values.
        # So we minimize -soft_min(values).
        
        # For efficiency, we can compute soft_min on the combined array?
        # Yes, soft_min(A union B) = soft_min(A) if min(A) << min(B) etc.
        # Just concatenate.
        all_vals = np.concatenate([flat_walls * 2.0, flat_pairs])
        
        # Calculate value
        val = soft_min_neg(all_vals, beta)
        
        # Calculate gradient
        # d/dx [-soft_min(x)]
        # Let S = sum(exp(-beta * v_i))
        # f = -(-1/beta * log S) = 1/beta * log S
        # df/dv_i = 1/beta * (1/S) * S * (-beta) * exp(-beta * v_i) * (-1) ? No.
        # f = 1/beta * log( sum(exp(-beta * v_i)) )
        # df/dv_i = 1/beta * 1/S * d/dv_i(sum(exp(-beta * v_j)))
        # d/dv_i(sum(exp(-beta * v_j))) = exp(-beta * v_i) * (-beta)
        # So df/dv_i = 1/beta * 1/S * (-beta) * exp(-beta * v_i) = -exp(-beta * v_i) / S
        # This is a probability distribution p_i = exp(-beta * v_i) / S
        # So gradient of f w.r.t v is -p.
        # We want to minimize f. So we want gradient of f.
        # grad_f = -p
        
        # Wait, let's re-verify.
        # We want to maximize min(v).
        # Proxy: maximize soft_min(v) = -1/beta * log(sum(exp(-beta * v)))
        # Let g(v) = soft_min(v). We maximize g.
        # dg/dv_i = -1/beta * 1/S * (-beta * exp(-beta * v_i)) = exp(-beta * v_i) / S = p_i
        # So gradient of soft_min is p (positive).
        # We want to maximize g, so we move in direction +grad_g.
        # Or we minimize -g.
        # Let h = -g. dh/dv_i = -p_i.
        # We minimize h. Gradient of h is -p.
        
        # Let's compute p.
        # To avoid underflow/overflow, use log-sum-exp trick.
        v_min = np.min(all_vals)
        shifted = all_vals - v_min
        exp_terms = np.exp(-beta * shifted)
        S = np.sum(exp_terms)
        p = exp_terms / S
        
        # Gradient of h (which we minimize) w.r.t all_vals is -p
        grad_all_vals = -p
        
        # Now backpropagate to centers.
        grad_centers = np.zeros_like(centers)
        
        # 1. Walls
        # v_wall_idx corresponds to flat_walls * 2
        # flat_walls[k] = 2 * (0.5 - |c_k|)
        # dv_wall / dc = -2 * sign(c)
        # grad_c += grad_v * dv/dc
        grad_walls = grad_all_vals[:2*N].reshape(N, 2)
        # d(2*(0.5 - |c|))/dc = -2 * sign(c)
        # Handle c=0 carefully, sign(0)=0 is fine for subgradient
        signs = np.sign(centers)
        grad_centers += grad_walls * (-2.0 * signs)
        
        # 2. Pairs
        # v_pair_idx corresponds to flat_pairs
        # flat_pairs is from dists matrix (N, N) flattened.
        # dists[i, j] = |c_i - c_j|
        # d dists[i, j] / dc_i = (c_i - c_j) / |c_i - c_j|
        # d dists[i, j] / dc_j = (c_j - c_i) / |c_i - c_j| = - (c_i - c_j) / |c_i - c_j|
        
        grad_pairs_flat = grad_all_vals[2*N:]
        grad_pairs = grad_pairs_flat.reshape(N, N)
        
        # Vectorized gradient calculation for pairs
        # diff[i, j] = c_i - c_j
        # dists[i, j] = norm(diff[i, j])
        # unit_diff[i, j] = diff[i, j] / dists[i, j]
        
        # Avoid division by zero (diagonal is inf, so dists is inf, unit is 0? 
        # Actually we set diagonal to inf in dist_sq, so dists is inf.
        # diff is 0. 0/inf = 0. This is safe.
        # But 0/0 is nan. Diagonal dist_sq is inf, so dists is inf.
        # So unit_diff diagonal will be 0/inf = 0.
        
        with np.errstate(divide='ignore', invalid='ignore'):
            unit_diff = diff / dists[:, :, np.newaxis]
            # Where dists was inf (diagonal), unit_diff is 0.
            # Where dists was 0 (overlapping points), unit_diff is nan.
            # We should handle overlapping points.
            # If dist == 0, gradient is undefined. We can set to 0 or random.
            # Let's set nan to 0.
            unit_diff = np.nan_to_num(unit_diff, nan=0.0, posinf=0.0, neginf=0.0)
            
        # grad_c_i = sum_j grad_pairs[i, j] * unit_diff[i, j]
        # grad_c_j = sum_i grad_pairs[i, j] * (-unit_diff[i, j])
        
        # Using matrix multiplication
        # grad_centers += grad_pairs @ unit_diff
        # Wait, grad_pairs[i, j] is the gradient w.r.t dist[i, j].
        # dL/dc_i = sum_j (dL/ddist[i,j]) * (ddist[i,j]/dc_i)
        # = sum_j grad_pairs[i, j] * unit_diff[i, j]
        
        grad_from_i = np.sum(grad_pairs * unit_diff, axis=1)
        
        # For j:
        # dL/dc_j = sum_i grad_pairs[i, j] * (-unit_diff[i, j])
        # Note unit_diff[j, i] = -unit_diff[i, j]
        # So dL/dc_j = sum_i grad_pairs[i, j] * unit_diff[j, i]
        # This is the same formula!
        # grad_from_j = np.sum(grad_pairs * unit_diff, axis=1) is NOT correct for j directly if we just swap indices?
        # Let's check.
        # grad_centers[j] += sum_i grad_pairs[i, j] * unit_diff[j, i]
        # Since unit_diff is antisymmetric and grad_pairs is symmetric (dist[i,j]=dist[j,i]),
        # grad_pairs[i, j] * unit_diff[j, i] = grad_pairs[j, i] * (-unit_diff[i, j])
        # So summing over i for j is equivalent to summing over j for i but with sign flip?
        # Actually, let's just use the property:
        # grad_centers = grad_pairs @ unit_diff?
        # (A @ B)_{ik} = sum_j A_{ij} B_{jk}
        # Here we want sum_j grad_pairs[i, j] * unit_diff[i, j]. This is element-wise product sum, not matrix mult.
        # It is sum_j (grad_pairs * unit_diff)[i, j]
        
        grad_pairs_contrib = np.sum(grad_pairs * unit_diff, axis=1)
        grad_centers += grad_pairs_contrib
        
        # Total gradient of h (to minimize)
        return val, grad_centers.flatten()

    def hexagonal_init(rows, cols, jitter=0.0, seed=0):
        rng = np.random.RandomState(seed)
        d = 1.0 / cols * 0.99
        row_spacing = d * math.sqrt(3) / 2
        total_height = (rows - 1) * row_spacing + d
        if total_height > 1.0:
            d = 1.0 / ((rows - 1) * math.sqrt(3) / 2 + 1) * 0.99
            
        pts = []
        for row in range(rows):
            y = -0.5 + d/2 + row * (d * math.sqrt(3) / 2)
            if row % 2 == 0:
                x_start = -0.5 + d/2
                for col in range(cols):
                    x = x_start + col * d
                    if jitter > 0:
                        x += rng.uniform(-jitter*d, jitter*d)
                        y += rng.uniform(-jitter*d, jitter*d)
                    pts.append([x, y])
            else:
                x_start = -0.5 + d
                for col in range(cols - 1):
                    x = x_start + col * d
                    if jitter > 0:
                        x += rng.uniform(-jitter*d, jitter*d)
                        y += rng.uniform(-jitter*d, jitter*d)
                    pts.append([x, y])
        
        pts = np.array(pts)
        if len(pts) >= N:
            return pts[:N]
        else:
            return None

    # Main Logic
    best_centers = None
    best_radius = 0.0
    
    # Generate diverse initializations
    inits = []
    
    # 1. Hexagonal grids
    for rows in [14, 15, 16]:
        cols = math.ceil(N / rows)
        for jitter in [0.0, 0.05]:
            pts = hexagonal_init(rows, cols, jitter=jitter, seed=rows*10 + int(jitter*100))
            if pts is not None:
                inits.append(pts)
                
    # 2. Square grid
    m = int(np.ceil(np.sqrt(N)))
    d = 1.0 / m
    pts = []
    for i in range(m):
        for j in range(m):
            x = -0.5 + (i + 0.5) * d
            y = -0.5 + (j + 0.5) * d
            pts.append([x, y])
    pts = np.array(pts)
    if len(pts) >= N:
        inits.append(pts[:N])
        
    # 3. Random
    for seed in range(2):
        rng = np.random.RandomState(seed)
        pts = rng.uniform(-0.49, 0.49, (N, 2))
        inits.append(pts)

    # Evaluate inits to find best starting point
    init_scores = []
    for i, init in enumerate(inits):
        r = compute_radius(init)
        init_scores.append((r, i))
        
    init_scores.sort(reverse=True)
    
    # Run L-BFGS-B on the top initializations
    # We have ~140s. 
    # L-BFGS-B is fast. Let's try optimizing the top 3-5 inits.
    # Each optimization might take 10-30s.
    
    tried_indices = set()
    
    for rank, (score, idx) in enumerate(init_scores):
        if time.time() > deadline - 20:
            break
        if idx in tried_indices:
            continue
        tried_indices.add(idx)
        
        x0 = inits[idx].flatten()
        
        # Objective function for minimize
        def obj(x, beta=100.0):
            val, _ = objective_and_grad(x, beta)
            return val
            
        def obj_grad(x, beta=100.0):
            _, grad = objective_and_grad(x, beta)
            return grad

        try:
            res = minimize(
                obj, x0, method='L-BFGS-B', 
                jac=obj_grad, 
                options={'maxiter': 200, 'gtol': 1e-8, 'disp': False}
            )
            
            if res.success or res.fun < 0: # res.fun is -soft_min, so smaller is better (more negative means larger min)
                # Actually res.fun is the value of the objective (which is -soft_min).
                # We want to minimize it.
                # Check if it improved the radius.
                centers_opt = res.x.reshape(N, 2)
                r_opt = compute_radius(centers_opt)
                
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_centers = centers_opt.copy()
                    
        except Exception as e:
            pass

    # Final Polish: If we have a good result, run a few more steps with higher beta (sharper)
    if best_centers is not None and time.time() < deadline - 5:
        # Increase beta to make soft_min closer to min
        # This requires redefining the objective with a closure or passing beta
        # For simplicity, we can just run L-BFGS-B again with a wrapper that uses higher beta
        # But objective_and_grad has beta fixed at 100.
        # Let's just run one more round with the current best, maybe with a smaller step?
        # Or just return. The gradient method is quite precise.
        
        # Let's try a local search polish if time permits
        # Simple coordinate ascent?
        # No, L-BFGS-B should have converged well.
        
        pass

    if best_centers is None:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers
