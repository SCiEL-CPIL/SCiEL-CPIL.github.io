# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from itertools import combinations

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill

    def get_dists(pts):
        pts = np.asarray(pts, dtype=np.float64)
        if pts.shape != (n, d):
            pts = pts.reshape(n, d)
        return pdist(pts)

    def exact_score(pts):
        dists = get_dists(pts)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        d_min = np.min(dists)
        return (d_min / d_max) ** 2

    def exact_score_flat(x):
        return exact_score(x.reshape(n, d))

    def exact_objective(x):
        return -exact_score_flat(x)

    best_x = None
    best_score = 0.0

    def update_best(x, score):
        nonlocal best_x, best_score
        if score > best_score:
            best_score = score
            best_x = x.copy()

    def normalize_center(pts):
        pts = pts - pts.mean(axis=0)
        dmax = np.max(pdist(pts))
        if dmax > 1e-12:
            pts = pts / dmax
        return pts

    # Phase 0: Generate diverse initial configurations
    def gen_grid():
        pts = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
        return normalize_center(pts)

    def gen_hex():
        pts = []
        for i in range(4):
            for j in range(4):
                x = j + 0.5 * (i % 2)
                y = i * np.sqrt(3) / 2
                pts.append([x, y])
        return normalize_center(np.array(pts))

    def gen_ring(k_center=1, r_inner=0.0, r_outer=1.0):
        pts = np.zeros((n, 2))
        if k_center > 0:
            pts[:k_center] = 0.0
        remaining = n - k_center
        angles = np.linspace(0, 2 * np.pi, remaining, endpoint=False)
        pts[k_center:, 0] = r_outer * np.cos(angles)
        pts[k_center:, 1] = r_outer * np.sin(angles)
        return normalize_center(pts)

    def gen_two_ring(r1=0.4, r2=1.0, offset=0.0):
        pts = np.zeros((n, 2))
        angles1 = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        angles2 = np.linspace(0, 2 * np.pi, 8, endpoint=False) + offset
        pts[:8, 0] = r1 * np.cos(angles1)
        pts[:8, 1] = r1 * np.sin(angles1)
        pts[8:, 0] = r2 * np.cos(angles2)
        pts[8:, 1] = r2 * np.sin(angles2)
        return normalize_center(pts)

    def gen_square_ring(r=1.0):
        pts = np.zeros((n, 2))
        side = r * 2
        # 4 corners
        pts[0] = [-side/2, -side/2]
        pts[1] = [side/2, -side/2]
        pts[2] = [side/2, side/2]
        pts[3] = [-side/2, side/2]
        # 3 points per edge
        for i in range(1, 4):
            t = i / 4.0
            pts[4 + (i-1)*4] = [-side/2 + t*side, -side/2]
            pts[4 + (i-1)*4 + 1] = [side/2, -side/2 + t*side]
            pts[4 + (i-1)*4 + 2] = [side/2 - t*side, side/2]
            pts[4 + (i-1)*4 + 3] = [-side/2, side/2 - t*side]
        return normalize_center(pts)

    initial_configs = [
        gen_grid(),
        gen_hex(),
        gen_ring(1, 0.0, 1.0),
        gen_ring(0, 0.0, 1.0),
        gen_two_ring(0.3, 1.0, 0.0),
        gen_two_ring(0.4, 1.0, np.pi/8),
        gen_two_ring(0.5, 1.0, np.pi/8),
        gen_square_ring(1.0),
    ]

    np.random.seed(42)
    for _ in range(8):
        initial_configs.append(np.random.uniform(-1, 1, (n, d)))

    for cfg in initial_configs:
        s = exact_score(cfg)
        update_best(cfg.ravel(), s)

    # Phase 1: Simulated Annealing - structural change from gradient-based to stochastic
    def simulated_annealing(x0, max_iter=5000, T0=0.5, T_end=0.001, cooling=0.9995):
        x = x0.copy()
        T = T0
        best_local = x.copy()
        best_local_score = exact_score_flat(x)
        
        for it in range(max_iter):
            if time.time() > deadline - 30:
                break
            
            # Perturb: random move
            delta = np.random.normal(0, 0.1, x.shape)
            x_new = x + delta
            
            score_old = exact_score_flat(x)
            score_new = exact_score_flat(x_new)
            
            if score_new >= score_old or np.random.random() < np.exp((score_new - score_old) / max(T, 1e-10)):
                x = x_new
                if score_new > best_local_score:
                    best_local_score = score_new
                    best_local = x_new.copy()
            else:
                x = x_new  # still move, but don't update best
            
            T = max(T_end, T * cooling)
        
        return best_local, best_local_score

    # Run SA from multiple starts
    sa_starts = initial_configs[:6]
    for cfg in sa_starts:
        if time.time() > deadline - 120:
            break
        try:
            sa_result, sa_score = simulated_annealing(cfg.ravel(), max_iter=3000)
            if sa_score > best_score:
                update_best(sa_result, sa_score)
        except Exception:
            pass

    # Phase 2: Pairwise-critical refinement
    # Identify min and max distance pairs, then optimize to balance them
    def critical_pair_refine(x0, max_iter=2000):
        x = x0.copy()
        for it in range(max_iter):
            if time.time() > deadline - 60:
                break
            
            pts = x.reshape(n, d)
            dists = get_dists(pts)
            d_min = np.min(dists)
            d_max = np.max(dists)
            
            if d_max < 1e-12:
                break
            
            # Find indices of min and max distance pairs
            min_idx = np.argmin(dists)
            max_idx = np.argmax(dists)
            
            # Convert pdist index to pair indices
            def pdist_to_pair(idx, n):
                # Find i,j such that pdist index = idx
                # pdist index for pair (i,j) with i<j: sum_{k=0}^{i-1} (n-1-k) + (j-i-1)
                # Use precomputed mapping
                count = 0
                for i in range(n-1):
                    for j in range(i+1, n):
                        if count == idx:
                            return i, j
                        count += 1
                return 0, 1
            
            # Precompute for speed
            if not hasattr(critical_pair_refine, 'pair_map'):
                pair_map = []
                idx = 0
                for i in range(n-1):
                    for j in range(i+1, n):
                        pair_map.append((i, j))
                        idx += 1
                critical_pair_refine.pair_map = pair_map
            
            i_min, j_min = critical_pair_refine.pair_map[min_idx]
            i_max, j_max = critical_pair_refine.pair_map[max_idx]
            
            # Move points to increase min distance and decrease max distance
            # Gradient-like step: push min pair apart, pull max pair together
            step_size = 0.01 * (d_max - d_min) / d_max + 0.001
            
            # Increase min distance
            diff_min = pts[j_min] - pts[i_min]
            norm_min = np.linalg.norm(diff_min)
            if norm_min > 1e-10:
                dir_min = diff_min / norm_min
                x[i_min*2:i_min*2+2] -= step_size * dir_min
                x[j_min*2:j_min*2+2] += step_size * dir_min
            
            # Decrease max distance
            diff_max = pts[j_max] - pts[i_max]
            norm_max = np.linalg.norm(diff_max)
            if norm_max > 1e-10:
                dir_max = diff_max / norm_max
                x[i_max*2:i_max*2+2] += step_size * 0.5 * dir_max
                x[j_max*2:j_max*2+2] -= step_size * 0.5 * dir_max
            
            # Also do a small random perturbation to escape local minima
            x += np.random.normal(0, 0.005, x.shape)
            
            # Center and normalize
            pts_new = x.reshape(n, d)
            x = normalize_center(pts_new).ravel()
        
        return x, exact_score_flat(x)

    # Run critical pair refinement from best and a few other starts
    if best_x is not None:
        if time.time() < deadline - 80:
            try:
                cp_result, cp_score = critical_pair_refine(best_x, max_iter=3000)
                if cp_score > best_score:
                    update_best(cp_result, cp_score)
            except Exception:
                pass

    # Phase 3: Nelder-Mead refinement on the exact non-smooth objective
    if best_x is not None and time.time() < deadline - 50:
        try:
            res_nm = minimize(
                exact_objective,
                best_x,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            if time.time() < deadline - 10:
                s = -res_nm.fun
                if s > best_score:
                    update_best(res_nm.x, s)
        except Exception:
            pass

    # Phase 4: More SA iterations if time permits
    if time.time() < deadline - 100:
        if best_x is not None:
            try:
                sa_result, sa_score = simulated_annealing(best_x, max_iter=4000, T0=0.3, T_end=0.0005, cooling=0.9997)
                if sa_score > best_score:
                    update_best(sa_result, sa_score)
            except Exception:
                pass

    # Phase 5: Final NM polish
    if best_x is not None and time.time() < deadline - 10:
        try:
            res_final = minimize(
                exact_objective,
                best_x,
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'xatol': 1e-9,
                    'fatol': 1e-11,
                    'adaptive': True
                }
            )
            s = -res_final.fun
            if s > best_score:
                update_best(res_final.x, s)
        except Exception:
            pass

    if best_x is None:
        best_x = gen_grid().ravel()

    return best_x.reshape(n, d)

# EVOLVE-BLOCK-END
