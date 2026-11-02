import numpy as np
from scipy.optimize import minimize

N = 50

def construct():
    def soft_min(x, beta):
        m = np.max(x)
        return m - (1.0 / beta) * np.log(np.sum(np.exp(beta * (x - m))))

    def clearances_and_grads(params, beta, gamma_wall=100.0):
        centers = params.reshape(N, 3)
        
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        iu = np.triu_indices(N, k=1)
        pair_dists = dists[iu]
        pair_clears = pair_dists / 2.0
        
        # Wall clearances
        abs_c = np.abs(centers)
        max_abs = np.max(abs_c, axis=1, keepdims=True)
        
        # Soft max for walls
        exp_terms_wall = np.exp(gamma_wall * (abs_c - max_abs))
        sum_exp_wall = np.sum(exp_terms_wall, axis=1, keepdims=True)
        soft_max_wall = (max_abs + (1.0 / gamma_wall) * np.log(sum_exp_wall)).flatten()
        wall_clears = 0.5 - soft_max_wall
        
        all_clears = np.concatenate([pair_clears, wall_clears])
        
        # Weights
        m_clears = np.max(all_clears)
        exp_terms = np.exp(-beta * (all_clears - m_clears))
        sum_exp = np.sum(exp_terms)
        weights = exp_terms / sum_exp
        
        soft_min_val = m_clears - (1.0 / beta) * np.log(sum_exp)
        val = -soft_min_val
        
        # Gradients
        n_pairs = len(pair_clears)
        w_pairs = weights[:n_pairs]
        w_walls = weights[n_pairs:]
        
        i_idx, j_idx = iu
        
        safe_dists = np.where(dists > 1e-8, dists, 1.0)
        unit_vecs = diff / safe_dists[:, :, np.newaxis]
        
        W_full = np.zeros((N, N))
        W_full[i_idx, j_idx] = w_pairs
        W_full[j_idx, i_idx] = w_pairs
        
        grad_pairs = 0.5 * np.einsum('ij,ijk->ik', W_full, unit_vecs)
        
        p_walls = exp_terms_wall / sum_exp_wall
        signs = np.sign(centers)
        grad_walls_vec = signs * p_walls
        
        grad_walls_term = w_walls[:, np.newaxis] * grad_walls_vec
        
        # grad f = - sum w grad C
        # grad C_pair = 0.5 * u
        # grad C_wall = - grad soft_max = - grad_walls_vec
        # sum w grad C = grad_pairs - grad_walls_term
        # grad f = -(grad_pairs - grad_walls_term) = -grad_pairs + grad_walls_term
        
        grad_total = -grad_pairs + grad_walls_term
        
        return val, grad_total.flatten()

    def optimize(centers_init, time_limit=5.0):
        import time
        start_time = time.time()
        
        # Staged beta
        betas = [50, 100, 200, 500, 1000]
        current_centers = centers_init.copy()
        
        for i, beta in enumerate(betas):
            if time.time() - start_time > time_limit - 1.0:
                break
                
            # More iterations for higher beta
            max_iter = 100 + i * 50
            
            res = minimize(
                lambda p: clearances_and_grads(p, beta)[0],
                current_centers.flatten(),
                jac=lambda p: clearances_and_grads(p, beta)[1],
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': max_iter, 'gtol': 1e-10}
            )
            current_centers = res.x.reshape(N, 3)
            
        return current_centers

    def score(centers):
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        max_c = np.max(np.abs(centers), axis=1)
        min_wall = 0.5 - np.min(max_c)
        
        return min(min_pair, min_wall)

    def greedy_init():
        centers = []
        for i in range(N):
            # Sample many candidates
            candidates = np.random.uniform(-0.45, 0.45, (1000, 3))
            best_score = -np.inf
            best_c = None
            
            for c in candidates:
                # Check walls
                wall_dist = 0.5 - np.max(np.abs(c))
                if wall_dist < 0:
                    continue
                
                # Check existing centers
                if centers:
                    dists = np.linalg.norm(np.array(centers) - c, axis=1)
                    min_dist = np.min(dists) / 2.0
                else:
                    min_dist = 1.0 # Large
                
                s = min(wall_dist, min_dist)
                if s > best_score:
                    best_score = s
                    best_c = c
            
            if best_c is not None:
                centers.append(best_c)
            
            # If we couldn't find a good one (shouldn't happen), add random
            if len(centers) <= i:
                centers.append(np.random.uniform(-0.4, 0.4, 3))
                
        return np.array(centers)

    def lattice_init():
        # FCC lattice
        # Generate points in a larger grid and clip
        a = 0.35 # Approx spacing
        pts = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                for k in range(-3, 4):
                    # FCC basis
                    x = i * a
                    y = j * a
                    z = k * a
                    pts.append([x, y, z])
                    pts.append([x + a/2, y + a/2, z])
                    pts.append([x + a/2, y, z + a/2])
                    pts.append([x, y + a/2, z + a/2])
        
        pts = np.array(pts)
        # Filter to cube
        mask = np.all(np.abs(pts) <= 0.45, axis=1)
        pts = pts[mask]
        
        # Shuffle and take N
        np.random.shuffle(pts)
        if len(pts) < N:
            # Pad with random
            extra = np.random.uniform(-0.4, 0.4, (N - len(pts), 3))
            pts = np.vstack([pts, extra])
        
        return pts[:N]

    best_centers = None
    best_score = -np.inf
    
    import time
    start_total = time.time()
    timeout = 110.0 # Leave 10s margin
    
    strategies = [
        ("random", lambda: np.random.uniform(-0.4, 0.4, (N, 3))),
        ("greedy", greedy_init),
        ("lattice", lattice_init),
    ]
    
    # Run multiple restarts
    for seed in range(50): # Up to 50 restarts
        if time.time() - start_total > timeout:
            break
            
        np.random.seed(seed)
        
        # Cycle through strategies
        strat_name, strat_func = strategies[seed % len(strategies)]
        
        try:
            c_init = strat_func()
            c_opt = optimize(c_init, time_limit=2.0)
            s = score(c_opt)
            
            if s > best_score:
                best_score = s
                best_centers = c_opt
        except Exception:
            pass

    return best_centers
