# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from itertools import combinations

def _score_fast(points):
    """Score using vectorized triangle area computation."""
    n = len(points)
    if n < 3:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    try:
        hull = ConvexHull(points)
        a_hull = hull.volume
    except Exception:
        return 0.0
    if a_hull <= 1e-12:
        return 0.0
    
    # Vectorized: compute all 286 triangle areas at once
    idx = np.triu_indices(n, k=1)
    i_idx, j_idx = idx
    k_idx = np.array([k for i in range(n) for j in range(i+1, n) for k in range(j+1, n)])
    
    x1 = points[i_idx, 0]; y1 = points[i_idx, 1]
    x2 = points[j_idx, 0]; y2 = points[j_idx, 1]
    x3 = points[k_idx, 0]; y3 = points[k_idx, 1]
    
    areas = np.abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)) * 0.5
    min_area = np.min(areas)
    
    if min_area <= 1e-12:
        return 0.0
    return min_area / a_hull

def _softmin_score(points, beta=200.0):
    """Smooth surrogate: soft-min of triangle areas / hull area.
    Higher beta -> closer to true min. Lower beta -> smoother for gradients."""
    n = len(points)
    if n < 3:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    try:
        hull = ConvexHull(points)
        a_hull = hull.volume
    except Exception:
        return 0.0
    if a_hull <= 1e-12:
        return 0.0
    
    idx = np.triu_indices(n, k=1)
    i_idx, j_idx = idx
    k_idx = np.array([k for i in range(n) for j in range(i+1, n) for k in range(j+1, n)])
    
    x1 = points[i_idx, 0]; y1 = points[i_idx, 1]
    x2 = points[j_idx, 0]; y2 = points[j_idx, 1]
    x3 = points[k_idx, 0]; y3 = points[k_idx, 1]
    
    areas = np.abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)) * 0.5
    
    # Soft-min via log-sum-exp: -log(sum(exp(-beta*a)))/beta
    # For numerical stability:
    shifted = -beta * areas
    max_shifted = np.max(shifted)
    log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    soft_min = -log_sum / beta
    
    return soft_min / a_hull

def _softmin_grad(points, beta=200.0):
    """Numerical gradient of softmin score w.r.t. all 26 coordinates."""
    eps = 1e-6
    grad = np.zeros_like(points)
    f0 = _softmin_score(points, beta)
    for d in range(points.size):
        pt = points.ravel().copy()
        pt[d] += eps
        f1 = _softmin_score(pt.reshape(-1, 2), beta)
        pt[d] -= 2*eps
        f2 = _softmin_score(pt.reshape(-1, 2), beta)
        grad.ravel()[d] = (f1 - f2) / (2*eps)
    return grad

def _score(points):
    """Compute A_min / A_hull for a set of 13 points."""
    return _score_fast(points)

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the smallest triangle area
    relative to the convex hull area.
    """
    start_time = time.time()
    deadline = start_time + 570
    
    n = 13
    rng = np.random.default_rng(seed=42)
    
    best_points = None
    best_score = 0.0
    
    def update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    # ===== Phase 1: Good initializations =====
    initializations = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    initializations.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Hexagonal lattice pattern (center + 2 rings)
    hex_pts = np.zeros((n, 2))
    hex_pts[0] = [0.0, 0.0]
    for i in range(6):
        angle = i * np.pi / 3
        hex_pts[i + 1] = [np.cos(angle), np.sin(angle)]
    for i in range(6):
        angle = i * np.pi / 3 + np.pi / 6
        hex_pts[i + 7] = [np.sqrt(3) * np.cos(angle), np.sqrt(3) * np.sin(angle)]
    initializations.append(hex_pts)
    
    # 3. Grid 3x3 + 4 center points
    grid_pts = np.array([
        [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
        [0.0, 1.0], [1.0, 1.0], [2.0, 1.0],
        [0.0, 2.0], [1.0, 2.0], [2.0, 2.0],
        [0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5],
    ], dtype=float)
    initializations.append(grid_pts)
    
    # 4. Concentric circles: 1 center + 4 inner + 8 outer
    conc_pts = np.zeros((n, 2))
    conc_pts[0] = [0.0, 0.0]
    for i in range(4):
        angle = i * np.pi / 2 + np.pi / 4
        conc_pts[i + 1] = [0.5 * np.cos(angle), 0.5 * np.sin(angle)]
    for i in range(8):
        angle = i * np.pi / 4
        conc_pts[i + 5] = [np.cos(angle), np.sin(angle)]
    initializations.append(conc_pts)
    
    # 5. Fibonacci spiral
    fib_pts = np.zeros((n, 2))
    golden = (1 + np.sqrt(5)) / 2
    for i in range(n):
        r = np.sqrt((i + 0.5) / n)
        theta = 2 * np.pi * i / golden
        fib_pts[i] = [r * np.cos(theta), r * np.sin(theta)]
    initializations.append(fib_pts)
    
    # 6. Several random points in unit square
    for _ in range(5):
        initializations.append(rng.random((n, 2)))
    
    # 7. Random points in hexagon
    for _ in range(3):
        pts = []
        for _ in range(n):
            r = rng.random() ** 0.5
            theta = rng.random() * 2 * np.pi
            pts.append([r * np.cos(theta), r * np.sin(theta)])
        initializations.append(np.array(pts))
    
    # Evaluate all initializations
    for init_pts in initializations:
        if time.time() > deadline - 90:
            break
        s = _score_fast(init_pts)
        update_best(init_pts, s)
    
    # ===== Phase 2: Multi-scale coordinate descent =====
    current_points = best_points.copy()
    current_score = best_score
    
    step_schedule = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    for base_step in step_schedule:
        if time.time() > deadline - 90:
            break
        
        for pass_num in range(3):
            if time.time() > deadline - 90:
                break
            
            improved = True
            while improved:
                improved = False
                for i in range(n):
                    if time.time() > deadline - 90:
                        break
                    
                    for _ in range(40):
                        direction = rng.normal(size=2)
                        norm = np.linalg.norm(direction)
                        if norm < 1e-10:
                            continue
                        direction /= norm
                        
                        for mult in [1.0, 0.5, 0.25, 2.0, 0.1]:
                            step = base_step * mult
                            trial = current_points.copy()
                            trial[i] = current_points[i] + step * direction
                            s = _score_fast(trial)
                            if s > current_score:
                                current_points = trial
                                current_score = s
                                improved = True
                                break
                        if improved:
                            break
                    if time.time() > deadline - 90:
                        break
            
            update_best(current_points, current_score)
    
    # ===== Phase 3: Smooth surrogate gradient ascent =====
    if time.time() < deadline - 120:
        current = best_points.copy()
        cur_score = best_score
        
        # Start with moderate beta, increase gradually
        for beta in [50.0, 100.0, 200.0, 400.0]:
            if time.time() > deadline - 90:
                break
            
            lr = 0.01
            n_steps = 200
            
            for step in range(n_steps):
                if time.time() > deadline - 90:
                    break
                
                grad = _softmin_grad(current, beta)
                
                # Normalize gradient to prevent explosion
                gnorm = np.linalg.norm(grad)
                if gnorm < 1e-12:
                    break
                grad = grad / gnorm
                
                trial = current + lr * grad
                
                # Check if valid
                s_trial = _score_fast(trial)
                if s_trial > cur_score:
                    current = trial
                    cur_score = s_trial
                    update_best(current, cur_score)
                    # Slightly increase learning rate on success
                    lr *= 1.05
                else:
                    # Decrease learning rate
                    lr *= 0.8
                    if lr < 1e-6:
                        break
            
            # After each beta stage, do a quick coordinate descent pass
            for i in range(n):
                if time.time() > deadline - 90:
                    break
                for _ in range(10):
                    direction = rng.normal(size=2)
                    norm = np.linalg.norm(direction)
                    if norm < 1e-10:
                        continue
                    direction /= norm
                    for mult in [0.01, 0.005, 0.002]:
                        trial = current.copy()
                        trial[i] = current[i] + mult * direction
                        s = _score_fast(trial)
                        if s > cur_score:
                            current = trial
                            cur_score = s
                            break
            update_best(current, cur_score)
    
    # ===== Phase 4: Simulated annealing =====
    if time.time() < deadline - 90:
        current = best_points.copy()
        cur_score = best_score
        T = 0.1
        T_min = 0.001
        cooling = 0.995
        n_sa_steps = 5000
        
        for step in range(n_sa_steps):
            if time.time() > deadline - 60:
                break
            
            i = rng.integers(n)
            perturbation = rng.normal(size=2) * T
            trial = current.copy()
            trial[i] = current[i] + perturbation
            
            s = _score_fast(trial)
            
            if s > cur_score:
                current = trial
                cur_score = s
                update_best(current, cur_score)
            else:
                if s > 0 and T > T_min:
                    ratio = (s - cur_score) / T
                    if np.log(rng.random()) < ratio:
                        current = trial
                        cur_score = s
            
            T *= cooling
            if T < T_min:
                T = T_min
    
    # ===== Phase 5: More random restarts with local optimization =====
    for restart in range(8):
        if time.time() > deadline - 60:
            break
        
        if restart % 2 == 0:
            start = best_points + rng.normal(size=(n, 2)) * 0.15
        else:
            start = rng.random((n, 2))
        
        s = _score_fast(start)
        if s <= 0:
            continue
        
        current = start.copy()
        cur_score = s
        
        for base_step in [0.1, 0.05, 0.02, 0.01, 0.005]:
            if time.time() > deadline - 60:
                break
            for i in range(n):
                if time.time() > deadline - 60:
                    break
                for _ in range(20):
                    direction = rng.normal(size=2)
                    norm = np.linalg.norm(direction)
                    if norm < 1e-10:
                        continue
                    direction /= norm
                    for mult in [1.0, 0.5, 2.0]:
                        step = base_step * mult
                        trial = current.copy()
                        trial[i] = current[i] + step * direction
                        s = _score_fast(trial)
                        if s > cur_score:
                            current = trial
                            cur_score = s
                            break
        if cur_score > best_score:
            best_points = current.copy()
            best_score = cur_score
    
    # ===== Phase 6: Final high-precision polish =====
    if time.time() < deadline - 30:
        current = best_points.copy()
        cur_score = best_score
        
        # Very fine coordinate descent
        for base_step in [0.001, 0.0005, 0.0002, 0.0001, 0.00005]:
            if time.time() > deadline - 15:
                break
            for i in range(n):
                if time.time() > deadline - 15:
                    break
                for _ in range(50):
                    direction = rng.normal(size=2)
                    norm = np.linalg.norm(direction)
                    if norm < 1e-10:
                        continue
                    direction /= norm
                    trial = current.copy()
                    trial[i] = current[i] + base_step * direction
                    s = _score_fast(trial)
                    if s > cur_score:
                        current = trial
                        cur_score = s
        if cur_score > best_score:
            best_points = current.copy()
            best_score = cur_score
        
        # Final gradient polish with high beta
        if time.time() < deadline - 15:
            for beta in [500.0, 1000.0]:
                if time.time() > deadline - 10:
                    break
                lr = 0.005
                for step in range(100):
                    if time.time() > deadline - 10:
                        break
                    grad = _softmin_grad(current, beta)
                    gnorm = np.linalg.norm(grad)
                    if gnorm < 1e-14:
                        break
                    grad = grad / gnorm
                    trial = current + lr * grad
                    s_trial = _score_fast(trial)
                    if s_trial > cur_score:
                        current = trial
                        cur_score = s_trial
                        update_best(current, cur_score)
                    else:
                        lr *= 0.7
                        if lr < 1e-8:
                            break
    
    # Final validation
    final_score = _score_fast(best_points)
    if final_score <= 0 or not np.all(np.isfinite(best_points)):
        fallback = np.array([
            [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
            [0.0, 1.0], [1.0, 1.0], [2.0, 1.0],
            [0.0, 2.0], [1.0, 2.0], [2.0, 2.0],
            [0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5],
        ], dtype=float)
        return fallback
    
    return best_points

# EVOLVE-BLOCK-END
