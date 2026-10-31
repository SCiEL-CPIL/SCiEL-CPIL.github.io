import numpy as np
import time

N = 16
D = 2
NUM_PAIRS = N * (N - 1) // 2

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety

    # Precompute pair indices for efficient distance computation
    iu = np.triu_indices(N, 1)
    pair_i = iu[0]
    pair_j = iu[1]

    def compute_dists_sq(pts):
        """Compute all pairwise squared distances."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        return dists_sq[iu]

    def exact_score(pts):
        d_sq = compute_dists_sq(pts)
        min_d_sq = np.min(d_sq)
        max_d_sq = np.max(d_sq)
        if min_d_sq < 1e-12:
            return 1e10
        return max_d_sq / min_d_sq

    def soft_objective(pts, beta=50.0):
        """Soft objective using log ratio for stability."""
        d_sq = np.maximum(compute_dists_sq(pts), 1e-12)
        
        max_d_sq = np.max(d_sq)
        soft_max = max_d_sq + (1.0/beta) * np.log(np.sum(np.exp(beta * (d_sq - max_d_sq))))
        
        min_d_sq = np.min(d_sq)
        soft_min = min_d_sq - (1.0/beta) * np.log(np.sum(np.exp(-beta * (d_sq - min_d_sq))))
        
        if soft_min < 1e-8 or soft_max < 1e-8:
            return 1e10
        
        return np.log(soft_max) - np.log(soft_min)

    def prepare_params(pts):
        """Normalize: fix closest pair to (0,0) and (1,0)."""
        d_sq = compute_dists_sq(pts)
        min_idx = np.argmin(d_sq)
        i0 = pair_i[min_idx]
        i1 = pair_j[min_idx]
        if i0 > i1:
            i0, i1 = i1, i0
        
        p0 = pts[i0]
        p1 = pts[i1]
        v = p1 - p0
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-8:
            return None, None
        
        cos_t = v[0] / norm_v
        sin_t = v[1] / norm_v
        R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
        
        transformed = (pts - p0) @ R_mat.T / norm_v
        
        perm = list(range(N))
        perm[0] = i0
        perm[1] = i1
        remaining = [i for i in range(N) if i != i0 and i != i1]
        perm[2:] = remaining
        
        transformed = transformed[perm]
        return transformed[2:].flatten(), perm

    def params_to_pts(params, perm):
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = params.reshape(-1, 2)
        inv_perm = np.argsort(perm)
        return pts[inv_perm]

    def finite_diff_gradient(params, perm, beta=50.0, eps=1e-5):
        """
        Compute gradient of soft_objective w.r.t. params using central differences.
        params: 28-dim vector (14 points × 2 coords)
        """
        grad = np.zeros(28)
        pts = params_to_pts(params, perm)
        base_val = soft_objective(pts, beta)
        
        for k in range(28):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[k] += eps
            params_minus[k] -= eps
            
            pts_plus = params_to_pts(params_plus, perm)
            pts_minus = params_to_pts(params_minus, perm)
            
            val_plus = soft_objective(pts_plus, beta)
            val_minus = soft_objective(pts_minus, beta)
            
            grad[k] = (val_plus - val_minus) / (2 * eps)
        
        return grad

    def gradient_flow_optimize(params, perm, max_iters=2000, time_limit=15.0):
        """
        Custom gradient flow with adaptive step size.
        Uses finite-difference gradients and adaptive learning rate.
        """
        current_params = params.copy()
        current_pts = params_to_pts(current_params, perm)
        current_score = exact_score(current_pts)
        best_params = current_params.copy()
        best_score = current_score
        
        beta = 50.0
        lr = 0.01
        lr_min = 1e-6
        lr_max = 0.1
        patience = 0
        best_iter_score = current_score
        best_iter_params = current_params.copy()
        
        t_start = time.time()
        
        for it in range(max_iters):
            # Time check
            if time.time() - t_start > time_limit:
                break
            
            # Compute gradient
            grad = finite_diff_gradient(current_params, perm, beta=beta)
            
            # Adaptive step size
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-10:
                break
            
            # Try step
            new_params = current_params - lr * grad
            new_pts = params_to_pts(new_params, perm)
            new_score = exact_score(new_pts)
            
            if new_score < current_score:
                # Accept step
                current_params = new_params
                current_score = new_score
                lr = min(lr * 1.1, lr_max)
                patience = 0
                
                if current_score < best_score:
                    best_score = current_score
                    best_params = current_params.copy()
                    best_iter_score = current_score
                    best_iter_params = current_params.copy()
            else:
                # Reject step, reduce learning rate
                lr = max(lr * 0.7, lr_min)
                patience += 1
                
                if patience > 50:
                    # Stuck, try a larger beta or break
                    if lr < lr_min * 1.5:
                        break
                    patience = 0
        
        return best_params, best_score

    def local_refine(params, perm, time_limit=8.0):
        """
        Multi-stage refinement: gradient flow with increasing beta.
        """
        current_params = params.copy()
        current_pts = params_to_pts(current_params, perm)
        current_score = exact_score(current_pts)
        
        # Stage 1: Coarse optimization with low beta
        if time.time() < deadline - time_limit:
            p1, s1 = gradient_flow_optimize(current_params, perm, max_iters=1000, 
                                            time_limit=min(5.0, deadline - time.time() - 2))
            if s1 < current_score:
                current_params = p1
                current_score = s1
        
        # Stage 2: Fine optimization with higher beta
        if time.time() < deadline - 3:
            p2, s2 = gradient_flow_optimize(current_params, perm, max_iters=1500,
                                            time_limit=min(5.0, deadline - time.time() - 1))
            if s2 < current_score:
                current_params = p2
                current_score = s2
        
        return current_params, current_score

    best_score = 1e10
    best_params = None
    best_perm = None
    
    # Generate diverse seeds
    seeds = []
    
    # 1. Random seeds
    for _ in range(3):
        pts = np.random.uniform(-1, 1, (N, 2))
        seeds.append(pts)
        
    # 2. Hexagonal lattice seed (perturbed)
    hex_pts = []
    R = 4
    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            x = q + 0.5 * r
            y = (np.sqrt(3) / 2) * r
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    hex_seed = hex_pts[:N].copy()
    hex_seed += np.random.normal(0, 0.1, hex_seed.shape)
    seeds.append(hex_seed)
    
    # 3. Concentric circles seed
    circ_pts = []
    for i in range(6):
        angle = 2 * np.pi * i / 6
        circ_pts.append([np.cos(angle), np.sin(angle)])
    for i in range(9):
        angle = 2 * np.pi * i / 9
        circ_pts.append([1.5 * np.cos(angle), 1.5 * np.sin(angle)])
    circ_pts.append([0,0])
    circ_pts = np.array(circ_pts)
    circ_seed = circ_pts.copy()
    circ_seed += np.random.normal(0, 0.1, circ_seed.shape)
    seeds.append(circ_seed)

    # 4. 3D Projection Seed
    def generate_3d_projection_seed():
        x = np.linspace(-1, 1, 4)
        y = np.linspace(-1, 1, 4)
        grid_3d = np.array([[xi, yi, 0] for xi in x for yi in y])
        grid_3d += np.random.normal(0, 0.2, grid_3d.shape)
        
        mean = np.mean(grid_3d, axis=0)
        centered = grid_3d - mean
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx]
        projection = centered @ eigvecs[:, :2]
        
        angle = np.random.uniform(0, 2*np.pi)
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        projection = projection @ rot.T
        projection = projection / np.max(np.abs(projection))
        return projection

    seeds.append(generate_3d_projection_seed())
    
    # 5. Perturbed 3D Projection Seeds
    for _ in range(2):
        seed_3d = generate_3d_projection_seed()
        seed_3d += np.random.normal(0, 0.1, seed_3d.shape)
        seeds.append(seed_3d)

    # 6. Square grid seed
    sq_pts = []
    for i in range(4):
        for j in range(4):
            sq_pts.append([i - 1.5, j - 1.5])
    sq_pts = np.array(sq_pts, dtype=float)
    sq_seed = sq_pts.copy()
    sq_seed += np.random.normal(0, 0.1, sq_seed.shape)
    seeds.append(sq_seed)

    # 7. Ring-based seed: 1 center + 5 inner + 10 outer
    ring_pts = [[0, 0]]
    for i in range(5):
        angle = 2 * np.pi * i / 5
        ring_pts.append([np.cos(angle), np.sin(angle)])
    for i in range(10):
        angle = 2 * np.pi * i / 10 + 0.3
        ring_pts.append([2.0 * np.cos(angle), 2.0 * np.sin(angle)])
    ring_pts = np.array(ring_pts, dtype=float)
    ring_seed = ring_pts.copy()
    ring_seed += np.random.normal(0, 0.05, ring_seed.shape)
    seeds.append(ring_seed)

    num_restarts = len(seeds)
    
    # Time budget: ~15s per seed for gradient flow
    for r in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        pts = seeds[r]
        
        # Ensure distinctness
        for i in range(N):
            for j in range(i+1, N):
                if np.linalg.norm(pts[i] - pts[j]) < 1e-4:
                    pts[j] += np.random.uniform(-0.1, 0.1, 2)
        
        res_prep = prepare_params(pts)
        if res_prep is None:
            continue
        init_params, perm = res_prep
        
        # Gradient flow optimization
        try:
            candidate_params, candidate_score = local_refine(init_params, perm, time_limit=12.0)
            
            if candidate_score < best_score:
                best_score = candidate_score
                best_params = candidate_params
                best_perm = perm
                
        except Exception as e:
            pass

    # Final polish on best result
    if best_params is not None and time.time() < deadline - 5:
        try:
            candidate_params, candidate_score = local_refine(best_params, best_perm, time_limit=8.0)
            if candidate_score < best_score:
                best_score = candidate_score
                best_params = candidate_params
        except Exception:
            pass

    # Construct final points
    if best_params is None:
        # Fallback: regular hexagonal packing
        pts = []
        for q in range(-2, 3):
            for r in range(-2, 3):
                x = q + 0.5 * r
                y = (np.sqrt(3) / 2) * r
                pts.append([x, y])
        pts = np.array(pts[:N])
        return pts

    final_pts = params_to_pts(best_params, best_perm)
    return final_pts
