# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def heilbronn_convex13():
    """
    Maximize the ratio of the smallest triangle area to the convex hull area
    for 13 points using differential evolution.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill

    # Precompute triangle indices
    tri_indices = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                tri_indices.append((i, j, k))
    tri_indices = np.array(tri_indices)
    n_tri = len(tri_indices)

    def calc_areas(points):
        p1 = points[tri_indices[:, 0]]
        p2 = points[tri_indices[:, 1]]
        p3 = points[tri_indices[:, 2]]
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        return np.abs(cross) * 0.5

    def calc_hull_area(points):
        try:
            hull = ConvexHull(points)
            return hull.volume
        except Exception:
            return 0.0

    def evaluate(points):
        if not np.all(np.isfinite(points)):
            return 0.0
        areas = calc_areas(points)
        min_area = np.min(areas)
        if min_area <= 0:
            return 0.0
        hull_area = calc_hull_area(points)
        if hull_area <= 0:
            return 0.0
        return min_area / hull_area

    def smooth_min(areas, alpha=50.0):
        """Soft-min using log-sum-exp for smoothness."""
        neg_alpha_areas = -alpha * areas
        max_val = np.max(neg_alpha_areas)
        if max_val < -700:
            return np.min(areas)
        sum_exp = np.sum(np.exp(neg_alpha_areas - max_val))
        if sum_exp <= 0:
            return np.min(areas)
        return -(max_val + np.log(sum_exp)) / alpha

    def evaluate_smooth(points, alpha=50.0):
        """Smooth surrogate using soft-min."""
        if not np.all(np.isfinite(points)):
            return 0.0
        areas = calc_areas(points)
        if np.min(areas) <= 0:
            return 0.0
        smin = smooth_min(areas, alpha)
        hull_area = calc_hull_area(points)
        if hull_area <= 0:
            return 0.0
        return smin / hull_area

    def neg_score(points):
        """Objective for minimization (DE minimizes)."""
        return -evaluate(points)

    def neg_score_smooth(points, alpha=50.0):
        """Smooth objective for DE."""
        return -evaluate_smooth(points, alpha)

    def init_hex_grid():
        """Generate a hexagonal lattice initialization."""
        points = []
        s = 0.22
        for row in range(-3, 4):
            for col in range(-3, 4):
                x = 0.5 + col * s + (row % 2) * s / 2
                y = 0.5 + row * s * np.sqrt(3) / 2
                points.append([x, y])
        points = np.array(points)
        mask = (points[:, 0] - 0.5)**2 + (points[:, 1] - 0.5)**2 <= 0.48**2
        points = points[mask]
        if len(points) > n:
            # Select n most evenly spaced points
            # Use a simple greedy approach
            selected = [0]
            for _ in range(n - 1):
                best_idx = -1
                best_min_dist = -1
                for idx in range(len(points)):
                    if idx in selected:
                        continue
                    min_dist = min(np.linalg.norm(points[idx] - points[si]) for si in selected)
                    if min_dist > best_min_dist:
                        best_min_dist = min_dist
                        best_idx = idx
                selected.append(best_idx)
            points = points[selected]
        while len(points) < n:
            p = np.random.random(2) * 0.8 + 0.1
            points = np.vstack([points, p])
        return points[:n]

    def init_circle():
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x = 0.5 + 0.4 * np.cos(angles)
        y = 0.5 + 0.4 * np.sin(angles)
        return np.column_stack((x, y))

    def init_random():
        return np.random.random((n, 2)) * 0.8 + 0.1

    # --- Differential Evolution with smooth surrogate ---
    def differential_evolution(initial_points, pop_size=40, max_gen=150,
                                F=0.8, CR=0.9):
        dim = n * 2
        x0 = initial_points.flatten()
        
        pop = np.zeros((pop_size, dim))
        pop[0] = x0
        for i in range(1, pop_size):
            pop[i] = x0 + (np.random.random(dim) - 0.5) * 0.3
            np.clip(pop[i], 0, 1, out=pop[i])
        
        pop_scores = np.array([neg_score_smooth(pop[i].reshape(n, 2), alpha=50.0) for i in range(pop_size)])
        best_idx = np.argmin(pop_scores)
        best_x = pop[best_idx].copy()
        best_score = pop_scores[best_idx]
        
        for gen in range(max_gen):
            if time.time() > deadline - 15:
                break
            
            F_gen = F * (0.6 + 0.4 * np.random.random())
            CR_gen = CR * (0.6 + 0.4 * np.random.random())
            
            new_pop = pop.copy()
            new_pop_scores = pop_scores.copy()
            
            for i in range(pop_size):
                if time.time() > deadline - 10:
                    break
                
                r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                while r1 == i or r2 == i or r3 == i:
                    r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                
                v = pop[r1] + F_gen * (pop[r2] - pop[r3])
                
                trial = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.random() < CR_gen or j == j_rand:
                        trial[j] = v[j]
                
                np.clip(trial, 0, 1, out=trial)
                
                trial_score = neg_score_smooth(trial.reshape(n, 2), alpha=50.0)
                
                if trial_score <= pop_scores[i]:
                    new_pop[i] = trial
                    new_pop_scores[i] = trial_score
                    if trial_score < best_score:
                        best_score = trial_score
                        best_x = trial.copy()
            
            pop = new_pop
            pop_scores = new_pop_scores
            
            if gen % 15 == 0 and gen > 0:
                worst_idx = np.argmax(pop_scores)
                pop[worst_idx] = np.random.random(dim) * 0.8 + 0.1
                pop_scores[worst_idx] = neg_score_smooth(pop[worst_idx].reshape(n, 2), alpha=50.0)
        
        return best_x.reshape(n, 2), -best_score

    # --- Aggressive Local Search with 12 directions and adaptive steps ---
    def local_search(points, max_iters=2000):
        current_points = points.copy()
        current_score = evaluate(current_points)
        step = 0.05
        min_step = 1e-7
        stall = 0
        
        directions_2d = [
            [1, 0], [-1, 0], [0, 1], [0, -1],
            [1, 1], [-1, 1], [1, -1], [-1, -1],
            [1, 0.5], [-1, 0.5], [1, -0.5], [-1, -0.5]
        ]
        directions_2d = np.array(directions_2d)
        norms = np.linalg.norm(directions_2d, axis=1)
        directions_2d = directions_2d / norms[:, np.newaxis]
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            improved = False
            order = np.random.permutation(n)
            for i in order:
                best_dir_move = None
                best_dir_score = current_score
                for d in directions_2d:
                    new_point = current_points.copy()
                    new_point[i] += d * step
                    if np.any(new_point < -0.01) or np.any(new_point > 1.01):
                        np.clip(new_point, 0, 1, out=new_point)
                        if np.any(new_point < 0) or np.any(new_point > 1):
                            continue
                    new_score = evaluate(new_point)
                    if new_score > best_dir_score:
                        best_dir_score = new_score
                        best_dir_move = new_point.copy()
                
                if best_dir_move is not None:
                    current_points = best_dir_move
                    current_score = best_dir_score
                    improved = True
            
            if not improved:
                stall += 1
                if stall >= 3:
                    step *= 0.6
                    stall = 0
                    if step < min_step:
                        break
            else:
                stall = 0
                step *= 1.05
        
        return current_points, current_score

    # --- Main optimization loop ---
    best_points = None
    best_score = 0.0

    # Generate diverse initial candidates
    np.random.seed(42)
    candidates = [
        init_hex_grid(),
        init_circle(),
        init_random(),
        init_random(),
        init_random()
    ]

    # Evaluate initial candidates
    scored_cands = []
    for cand in candidates:
        if not np.all(np.isfinite(cand)):
            continue
        s = evaluate(cand)
        scored_cands.append((s, cand))
        if s > best_score:
            best_score = s
            best_points = cand.copy()

    # Sort by score descending
    scored_cands.sort(key=lambda x: -x[0])

    # Run DE on top candidates with smooth surrogate
    for s, cand in scored_cands[:3]:
        if time.time() > deadline - 60:
            break
        de_points, de_score = differential_evolution(cand, pop_size=35, max_gen=120)
        true_score = evaluate(de_points)
        if true_score > best_score:
            best_score = true_score
            best_points = de_points.copy()

    # Also run DE on the best initial candidate directly
    if scored_cands and time.time() < deadline - 60:
        top_cand = scored_cands[0][1]
        de_points, de_score = differential_evolution(top_cand, pop_size=30, max_gen=100)
        true_score = evaluate(de_points)
        if true_score > best_score:
            best_score = true_score
            best_points = de_points.copy()

    # Final aggressive local search polishing
    if best_points is not None and time.time() < deadline - 10:
        final_pts, final_score = local_search(best_points, max_iters=2000)
        if final_score > best_score:
            best_points = final_pts
            best_score = final_score

    # One more local search pass if time permits
    if best_points is not None and time.time() < deadline - 5:
        final_pts2, final_score2 = local_search(best_points, max_iters=500)
        if final_score2 > best_score:
            best_points = final_pts2
            best_score = final_score2

    if best_points is None:
        best_points = init_circle()

    return best_points
# EVOLVE-BLOCK-END
