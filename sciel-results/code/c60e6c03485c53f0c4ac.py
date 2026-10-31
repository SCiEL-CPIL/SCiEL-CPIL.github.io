"""Combined approach: gradient-flow local refinement from diverse seeds, using smooth log-sum-exp surrogates."""
import numpy as np
import time
from scipy.optimize import minimize

N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    iu = np.triu_indices(N, 1)
    n_pairs = len(iu[0])

    def get_sq_dists(pts):
        """Calculate squared pairwise distances for the upper triangle."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        return sq_dists[iu]

    def exact_score(pts):
        """Calculate the exact score: (max dist / min dist)^2"""
        sq_dists = get_sq_dists(pts)
        min_sq = np.min(sq_dists)
        max_sq = np.max(sq_dists)
        if min_sq < 1e-12:
            return 1e10
        return max_sq / min_sq

    def smooth_objective(pts, beta):
        """
        Smooth surrogate using log-sum-exp on SQUARED distances.
        Returns (soft_max_sq / soft_min_sq).
        """
        sq_dists = get_sq_dists(pts)
        sq_dists = np.maximum(sq_dists, 1e-12)
        
        # Soft Max of squared distances
        max_sq = np.max(sq_dists)
        exp_terms_max = np.exp(beta * (sq_dists - max_sq))
        sum_exp_max = np.sum(exp_terms_max)
        if sum_exp_max < 1e-300:
            soft_max_sq = max_sq
        else:
            soft_max_sq = max_sq + (1.0 / beta) * np.log(sum_exp_max)
        
        # Soft Min of squared distances
        min_sq = np.min(sq_dists)
        exp_terms_min = np.exp(-beta * (sq_dists - min_sq))
        sum_exp_min = np.sum(exp_terms_min)
        if sum_exp_min < 1e-300:
            soft_min_sq = min_sq
        else:
            soft_min_sq = min_sq - (1.0 / beta) * np.log(sum_exp_min)
            
        if soft_min_sq < 1e-12:
            return 1e10
            
        return soft_max_sq / soft_min_sq

    def normalize(pts):
        """Center at origin and scale such that RMS distance is 1."""
        pts = pts - np.mean(pts, axis=0)
        rms = np.sqrt(np.mean(np.sum(pts**2, axis=1)))
        if rms > 1e-9:
            pts = pts / rms
        return pts

    def gradient_flow(pts, beta, max_iters=5000, lr_init=0.005, lr_decay=0.999):
        """
        Gradient flow using finite-difference gradient of the smooth objective.
        Moves each point independently.
        """
        pts = pts.copy()
        current_score = smooth_objective(pts, beta)
        best_pts = pts.copy()
        best_score = current_score
        lr = lr_init
        
        rng = np.random.default_rng(12345)
        
        for i in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Compute gradient via central differences
            grad = np.zeros_like(pts)
            eps = 1e-5
            for idx in range(N):
                for d in range(D):
                    pts_plus = pts.copy()
                    pts_minus = pts.copy()
                    pts_plus[idx, d] += eps
                    pts_minus[idx, d] -= eps
                    
                    score_plus = smooth_objective(pts_plus, beta)
                    score_minus = smooth_objective(pts_minus, beta)
                    
                    grad[idx, d] = (score_plus - score_minus) / (2 * eps)
            
            # Update points
            pts = pts - lr * grad
            
            # Normalize to prevent drift
            pts = normalize(pts)
            
            new_score = smooth_objective(pts, beta)
            
            if new_score < best_score:
                best_score = new_score
                best_pts = pts.copy()
            
            # Decay learning rate
            lr *= lr_decay
            if lr < 1e-8:
                break
        
        return best_pts, best_score

    def l_bfgs_polish(pts, beta, maxiter=200):
        """
        L-BFGS-B polish on the smooth objective.
        """
        def obj_flat(x):
            p = x.reshape(N, D)
            return smooth_objective(p, beta)
        
        x0 = pts.flatten()
        bnds = [(-10, 10)] * (N * D)
        
        try:
            res = minimize(obj_flat, x0, method='L-BFGS-B', bounds=bnds,
                           options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.success or res.fun < 1e9:
                return res.x.reshape(N, D), res.fun
        except Exception:
            pass
        
        return pts, smooth_objective(pts, beta)

    def exact_polish(pts, maxiter=100):
        """
        Final polish using the exact objective with L-BFGS-B.
        """
        def obj_flat(x):
            p = x.reshape(N, D)
            return exact_score(p)
        
        x0 = pts.flatten()
        bnds = [(-10, 10)] * (N * D)
        
        try:
            res = minimize(obj_flat, x0, method='L-BFGS-B', bounds=bnds,
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.success or res.fun < 1e9:
                return res.x.reshape(N, D), res.fun
        except Exception:
            pass
        
        return pts, exact_score(pts)

    def generate_seeds():
        """Generate diverse initial configurations."""
        seeds = []
        rng = np.random.default_rng(42)
        
        # 1. Hexagonal Lattice (Top 16 closest to center)
        R = 5
        hex_pts = []
        for q in range(-R, R + 1):
            for r in range(-R, R + 1):
                x = q + 0.5 * r
                y = (np.sqrt(3) / 2) * r
                hex_pts.append([x, y])
        hex_pts = np.array(hex_pts)
        dists_origin = np.sum(hex_pts**2, axis=1)
        sorted_indices = np.argsort(dists_origin)
        init_hex = hex_pts[sorted_indices[:N]]
        seeds.append(normalize(init_hex))
        
        # 2. Regular 16-gon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        init_circle = np.column_stack([np.cos(angles), np.sin(angles)])
        seeds.append(normalize(init_circle))
        
        # 3. 4x4 Grid
        grid_x = np.linspace(-1.5, 1.5, 4)
        grid_y = np.linspace(-1.5, 1.5, 4)
        gx, gy = np.meshgrid(grid_x, grid_y)
        init_grid = np.column_stack([gx.flatten(), gy.flatten()])
        seeds.append(normalize(init_grid))
        
        # 4. Golden Angle Spiral
        init_spiral = np.zeros((N, 2))
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(N):
            r = np.sqrt(i + 0.5)
            theta = i * golden_angle
            init_spiral[i, 0] = r * np.cos(theta)
            init_spiral[i, 1] = r * np.sin(theta)
        seeds.append(normalize(init_spiral))
        
        # 5. Random
        init_random = rng.uniform(-2, 2, (N, 2))
        seeds.append(normalize(init_random))
        
        # 6. Perturbed Hex
        seeds.append(normalize(init_hex + rng.normal(0, 0.1, size=(N, 2))))
        
        # 7. Perturbed Grid
        seeds.append(normalize(init_grid + rng.normal(0, 0.1, size=(N, 2))))
        
        # 8. Two concentric rings (8 inner, 8 outer)
        angles_inner = np.linspace(0, 2*np.pi, 8, endpoint=False)
        angles_outer = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
        r_inner = 0.6
        r_outer = 1.4
        inner_pts = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
        outer_pts = np.column_stack([r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)])
        ring_pts = np.vstack([inner_pts, outer_pts])
        seeds.append(normalize(ring_pts))
        
        # 9. Three concentric rings (6+6+4)
        angles_r1 = np.linspace(0, 2*np.pi, 6, endpoint=False)
        angles_r2 = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
        angles_r3 = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4
        r1, r2, r3 = 0.4, 1.0, 1.6
        pts_r1 = np.column_stack([r1 * np.cos(angles_r1), r1 * np.sin(angles_r1)])
        pts_r2 = np.column_stack([r2 * np.cos(angles_r2), r2 * np.sin(angles_r2)])
        pts_r3 = np.column_stack([r3 * np.cos(angles_r3), r3 * np.sin(angles_r3)])
        triple_ring = np.vstack([pts_r1, pts_r2, pts_r3])
        seeds.append(normalize(triple_ring))
        
        return seeds

    # Generate seeds
    candidates = generate_seeds()
    
    # Score seeds to prioritize
    scored_seeds = [(exact_score(s), s) for s in candidates]
    scored_seeds.sort(key=lambda x: x[0])
    
    best_score = 1e10
    best_pts = None
    
    # Process top seeds
    num_to_process = min(6, len(scored_seeds))
    
    for idx in range(num_to_process):
        if time.time() > deadline - 30:
            break
            
        seed_score, init_pts = scored_seeds[idx]
        
        # Update best if seed is good
        if seed_score < best_score:
            best_score = seed_score
            best_pts = init_pts.copy()
        
        current_pts = init_pts.copy()
        
        # Phase 1: Gradient flow with low beta (broad basin)
        if time.time() < deadline - 50:
            gf_pts, gf_score = gradient_flow(current_pts, beta=10.0, max_iters=3000, lr_init=0.003)
            if gf_score < best_score:
                best_score = gf_score
                best_pts = gf_pts.copy()
            current_pts = gf_pts
        
        # Phase 2: L-BFGS polish with medium beta
        if time.time() < deadline - 30:
            lbfgs_pts, lbfgs_score = l_bfgs_polish(current_pts, beta=100.0, maxiter=150)
            if lbfgs_score < best_score:
                best_score = lbfgs_score
                best_pts = lbfgs_pts.copy()
            current_pts = lbfgs_pts
        
        # Phase 3: Gradient flow with high beta (sharper)
        if time.time() < deadline - 15:
            gf_pts2, gf_score2 = gradient_flow(current_pts, beta=1000.0, max_iters=2000, lr_init=0.001)
            if gf_score2 < best_score:
                best_score = gf_score2
                best_pts = gf_pts2.copy()
            current_pts = gf_pts2
        
        # Phase 4: Exact polish
        if time.time() < deadline - 5:
            exact_pts, exact_sc = exact_polish(current_pts, maxiter=80)
            if exact_sc < best_score:
                best_score = exact_sc
                best_pts = exact_pts.copy()

    if best_pts is None:
        # Fallback to the initial best candidate
        best_pts = scored_seeds[0][1]
        
    # Final verification
    final_score = exact_score(best_pts)
    return best_pts

# EVOLVE-BLOCK-END
