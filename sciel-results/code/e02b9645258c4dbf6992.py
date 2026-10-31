import numpy as np
import time
import warnings
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

warnings.filterwarnings('ignore')

N = 13

_TRIPLE_IDX = np.array([
    (i, j, k)
    for i in range(N)
    for j in range(i + 1, N)
    for k in range(j + 1, N)
], dtype=np.int32)

def _triangle_areas_vectorized(points):
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _hull_area(points):
    try:
        if len(points) < 3:
            return 0.0
        if np.allclose(points, points[0]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = np.min(areas)
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area

def _polar_to_cartesian(polar):
    angles = polar[:N]
    radii = polar[N:]
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y])

def _polar_objective(polar):
    points = _polar_to_cartesian(polar)
    points = points - np.mean(points, axis=0)
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s

def _smooth_objective(x_flat, p=50):
    points = x_flat.reshape(N, 2)
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    if np.allclose(points, 0):
        return 1e10
    
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 1e10
    
    areas = _triangle_areas_vectorized(points)
    areas = np.maximum(areas, 1e-15)
    
    neg_p_areas = -p * areas
    max_neg = np.max(neg_p_areas)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_p_areas - max_neg)))
    soft_min = -log_sum / p
    
    score = soft_min / hull_area
    if score <= 0:
        return 1e10
    return -score

def _affine_grid_objective(params):
    a, b, c, d, tx, ty = params
    
    # Base grid: 3x5 lattice (15 points) with 2 points removed to get 13
    # We will try removing different pairs later, but for the objective, 
    # we fix a specific "good" removal pattern: remove the two corners (0,0) and (2,4)
    # to maintain symmetry-ish properties or just standard corners.
    # Actually, let's use the 4x4 minus 3 points idea or 3x5 minus 2.
    # 3x5: indices 0..14.
    # Let's define the base 3x5 grid centered at origin.
    # x: -1, 0, 1; y: -2, -1, 0, 1, 2
    base = np.array([
        [-1, -2], [0, -2], [1, -2],
        [-1, -1], [0, -1], [1, -1],
        [-1,  0], [0,  0], [1,  0],
        [-1,  1], [0,  1], [1,  1],
        [-1,  2], [0,  2], [1,  2]
    ], dtype=float)
    
    # Remove indices 0 and 14 (corners (-1,-2) and (1,2))
    # This leaves a shape that is somewhat compact.
    # Let's try removing 0 and 14.
    base = np.delete(base, [0, 14], axis=0)
    
    # Apply affine transform
    x = a * base[:, 0] + b * base[:, 1] + tx
    y = c * base[:, 0] + d * base[:, 1] + ty
    points = np.column_stack([x, y])
    
    score = _score(points)
    if score <= 0:
        return 1e10
    return -score

def _objective_full(x_flat):
    points = x_flat.reshape(N, 2)
    points = points - np.mean(points, axis=0)
    score = _score(points)
    if score <= 0:
        return 1e10
    return -score

def _make_ring_config(n_outer, n_inner, r_inner, angle_offset):
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    angles_inner = np.linspace(angle_offset, 2 * np.pi + angle_offset, n_inner, endpoint=False)
    
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    inner_pts = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    
    return np.vstack([outer_pts, inner_pts])

def _coordinate_descent_polish(points, deadline, max_iter=2000):
    best_pts = points.copy()
    best_s = _score(best_pts)
    
    # Adaptive step sizes
    step = 0.05
    min_step = 1e-6
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            for coord in range(2):
                # Try decreasing step sizes
                for delta in [step, -step, step*0.5, -step*0.5, step*0.1, -step*0.1]:
                    trial = best_pts.copy()
                    trial[i, coord] += delta
                    trial_s = _score(trial)
                    if trial_s > best_s + 1e-14:
                        best_pts = trial
                        best_s = trial_s
                        improved = True
                        break
        if not improved:
            step *= 0.5
            if step < min_step:
                break
    
    return best_pts, best_s

def _local_polish(points, deadline):
    best_x = points.flatten().copy()
    best_s = _score(points)
    
    # Phase 1: Powell on smooth objective
    if time.time() < deadline - 5:
        try:
            res = minimize(
                lambda x: _smooth_objective(x, p=80),
                best_x,
                method='Powell',
                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            s = _score(res.x.reshape(N, 2))
            if s > best_s:
                best_s = s
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead on full objective
    if time.time() < deadline - 3:
        try:
            res_nm = minimize(
                _objective_full,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            s_nm = _score(res_nm.x.reshape(N, 2))
            if s_nm > best_s:
                best_s = s_nm
                best_x = res_nm.x.copy()
        except Exception:
            pass
    
    # Phase 3: Coordinate descent polish
    if time.time() < deadline - 1:
        try:
            cd_pts, cd_s = _coordinate_descent_polish(best_x.reshape(N, 2), deadline)
            if cd_s > best_s:
                best_s = cd_s
                best_x = cd_pts.flatten().copy()
        except Exception:
            pass
    
    return best_x.reshape(N, 2), best_s

def _simulated_annealing(points, deadline):
    current = points.copy()
    current_score = _score(current)
    best = current.copy()
    best_s = current_score
    
    time_budget = deadline - 3.0
    total_time = max(0.5, time_budget - time.time())
    est_iters = int(total_time * 1000)
    est_iters = max(1000, min(est_iters, 80000))
    
    T0 = 0.02
    T_end = 0.0001
    alpha = (T_end / T0) ** (1.0 / max(1, est_iters - 1))
    
    rng = np.random.default_rng(42)
    
    accepted = 0
    total = 0
    step_scale = 0.3
    
    for it in range(est_iters):
        if time.time() > time_budget:
            break
        
        T = T0 * (alpha ** it)
        noise = rng.normal(0, T * step_scale, size=(N, 2))
        candidate = current + noise
        
        cand_score = _score(candidate)
        total += 1
        
        if cand_score > current_score:
            current = candidate
            current_score = cand_score
            accepted += 1
            if current_score > best_s:
                best_s = current_score
                best = current.copy()
        else:
            delta = current_score - cand_score
            if delta < 1e-12:
                current = candidate
                current_score = cand_score
                accepted += 1
            else:
                prob = np.exp(-delta / max(T, 1e-10))
                if rng.random() < prob:
                    current = candidate
                    current_score = cand_score
                    accepted += 1
        
        if total % 500 == 0 and total > 0:
            acc_rate = accepted / total
            if acc_rate < 0.15:
                step_scale *= 0.7
            elif acc_rate > 0.4:
                step_scale *= 1.2
            step_scale = np.clip(step_scale, 0.05, 2.0)
            accepted = 0
            total = 0
    
    return best, best_s

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    best_points = None
    best_score = 0.0

    def _update_best(points):
        nonlocal best_points, best_score
        if points is None or len(points) != N:
            return
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()

    # === Phase 0: Affine Grid Optimization ===
    # This is the key structural improvement: optimize the affine transform of a fixed lattice.
    # We try several "removal" patterns from a 3x5 or 4x4 grid.
    if time.time() < deadline - 30:
        # Define base grids and removals
        configs = []
        
        # 3x5 grid
        base_35 = np.array([
            [-1, -2], [0, -2], [1, -2],
            [-1, -1], [0, -1], [1, -1],
            [-1,  0], [0,  0], [1,  0],
            [-1,  1], [0,  1], [1,  1],
            [-1,  2], [0,  2], [1,  2]
        ], dtype=float)
        
        # Removals for 3x5 (need to remove 2)
        removals_35 = [
            [0, 14], # Corners
            [0, 1],  # Top edge
            [13, 14],# Bottom edge
            [0, 13], # Left corners
            [1, 14], # Top right, bottom right
        ]
        for rem in removals_35:
            base = np.delete(base_35, rem, axis=0)
            configs.append(base)
            
        # 4x4 grid
        base_44 = np.array([
            [-1.5, -1.5], [-0.5, -1.5], [0.5, -1.5], [1.5, -1.5],
            [-1.5, -0.5], [-0.5, -0.5], [0.5, -0.5], [1.5, -0.5],
            [-1.5,  0.5], [-0.5,  0.5], [0.5,  0.5], [1.5,  0.5],
            [-1.5,  1.5], [-0.5,  1.5], [0.5,  1.5], [1.5,  1.5]
        ], dtype=float)
        
        # Removals for 4x4 (need to remove 3)
        removals_44 = [
            [0, 3, 15], # 3 corners
            [0, 15, 14], # Corner + edge
            [0, 3, 12], # Corner + edge
            [1, 2, 3], # Top row partial
            [0, 1, 2], # Top row partial
            [12, 13, 14], # Bottom row partial
            [0, 3, 12], # Diagonal-ish
            [1, 14, 13], # Center-ish
        ]
        for rem in removals_44:
            base = np.delete(base_44, rem, axis=0)
            configs.append(base)
            
        # Optimize affine transform for each config
        for base in configs:
            if time.time() > deadline - 10:
                break
            try:
                # Initial guess: Identity + small rotation
                x0 = [1.0, 0.1, -0.1, 1.0, 0.0, 0.0]
                
                res = minimize(
                    _affine_grid_objective,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
                
                # Reconstruct points from best affine params
                a, b, c, d, tx, ty = res.x
                x = a * base[:, 0] + b * base[:, 1] + tx
                y = c * base[:, 0] + d * base[:, 1] + ty
                aff_points = np.column_stack([x, y])
                _update_best(aff_points)
                
                # Also try a second NM pass with a perturbed start if time allows
                if time.time() < deadline - 15:
                    x0_2 = [1.0, -0.2, 0.2, 1.0, 0.0, 0.0]
                    res2 = minimize(
                        _affine_grid_objective,
                        x0_2,
                        method='Nelder-Mead',
                        options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                    )
                    a, b, c, d, tx, ty = res2.x
                    x = a * base[:, 0] + b * base[:, 1] + tx
                    y = c * base[:, 0] + d * base[:, 1] + ty
                    aff_points2 = np.column_stack([x, y])
                    _update_best(aff_points2)
            except Exception:
                pass

    # === Phase 1: Differential Evolution in polar space ===
    if time.time() < deadline - 40:
        bounds = [(0, 2 * np.pi)] * N + [(0.05, 3.0)] * N
        try:
            de_result = differential_evolution(
                _polar_objective,
                bounds=bounds,
                seed=42,
                maxiter=80,
                popsize=25,
                tol=1e-7,
                mutation=(0.4, 1.2),
                recombination=0.8,
                polish=False,
                updating='deferred',
                workers=1
            )
            if de_result.success or de_result.fun < 1e9:
                de_points = _polar_to_cartesian(de_result.x)
                de_points = de_points - np.mean(de_points, axis=0)
                _update_best(de_points)
        except Exception:
            pass
    
    # === Phase 2: Generate diverse initial configurations ===
    inits = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Two-ring configurations
    for n_out in [5, 6, 7, 8]:
        n_in = N - n_out
        if n_in < 3:
            continue
        for r_in in [0.3, 0.5, 0.7]:
            for offset in [0, np.pi / n_out]:
                pts = _make_ring_config(n_out, n_in, r_in, offset)
                inits.append(pts)
    
    # 3. Random starts
    rng = np.random.default_rng(12345)
    for _ in range(10):
        r = np.sqrt(rng.random(N)) * 1.5
        theta = rng.random(N) * 2 * np.pi
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        inits.append(pts)
        
    # 4. If DE or Affine found something good, add its neighborhood
    if best_points is not None:
        rng_de = np.random.default_rng(777)
        for _ in range(5):
            perturbed = best_points + rng_de.normal(0, 0.1, size=(N, 2))
            inits.append(perturbed)
            
    # Shuffle inits
    rng2 = np.random.default_rng(999)
    indices = rng2.permutation(len(inits))
    inits = [inits[i] for i in indices]
    
    # === Phase 3: Local optimization from each start ===
    for i, x0 in enumerate(inits):
        if time.time() > deadline - 15:
            break
        
        x_opt, s_opt = _local_polish(x0, deadline)
        _update_best(x_opt)
        
        if i % 5 == 0:
            elapsed = time.time() - start_time
            print(f"Init {i}/{len(inits)}: best_score={best_score:.6f}, elapsed={elapsed:.1f}s", flush=True)
    
    # === Phase 4: Simulated Annealing ===
    if best_points is not None and time.time() < deadline - 5:
        try:
            sa_best, sa_score = _simulated_annealing(best_points, deadline)
            _update_best(sa_best)
        except Exception:
            pass
    
    # === Phase 5: Final high-precision polish ===
    if best_points is not None and time.time() < deadline - 2:
        try:
            rng_final = np.random.default_rng(888)
            for attempt in range(3):
                if time.time() > deadline - 1:
                    break
                if attempt == 0:
                    start_x = best_points.flatten()
                else:
                    start_x = best_points.flatten() + rng_final.normal(0, 0.005, size=N*2)
                
                res_final = minimize(
                    _objective_full,
                    start_x,
                    method='Nelder-Mead',
                    options={'maxiter': 1500, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
                )
                _update_best(res_final.x.reshape(N, 2))
                
                if time.time() < deadline - 1:
                    cd_pts, cd_s = _coordinate_descent_polish(best_points, deadline)
                    _update_best(cd_pts)
        except Exception:
            pass
    
    # Fallback
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
