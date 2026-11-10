import time
import random
import numpy as np
from collections import Counter
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    N = 30
    bounds = 30

    # Precompute all triple indices once
    triple_indices = list(combinations(range(N), 3))
    n_triples = len(triple_indices)

    def compute_all_areas(pts):
        pts_arr = np.array(pts, dtype=np.int64)
        areas = []
        for i, j, k in triple_indices:
            x1, y1 = pts_arr[i]
            x2, y2 = pts_arr[j]
            x3, y3 = pts_arr[k]
            d = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
            if d > 0:
                areas.append(d)
        if not areas:
            return Counter()
        return Counter(areas)

    def get_max_count(counts):
        if not counts:
            return 0
        return max(counts.values())

    def get_best_area(counts):
        if not counts:
            return 0
        return counts.most_common(1)[0][0]

    def compute_delta_fast(old_pt, new_pt, pts, idx):
        """Vectorized delta computation using numpy."""
        pts_arr = np.array(pts, dtype=np.int64)
        others_mask = np.ones(N, dtype=bool)
        others_mask[idx] = False
        others_idx = np.where(others_mask)[0]
        others_pts = pts_arr[others_idx]
        
        x_old, y_old = old_pt
        x_new, y_new = new_pt
        
        n_others = len(others_idx)
        removed_counts = Counter()
        added_counts = Counter()
        
        # Vectorize over pairs
        for a in range(n_others):
            x1, y1 = others_pts[a]
            for b in range(a+1, n_others):
                x2, y2 = others_pts[b]
                d_old = abs(x_old*(y1-y2) + x1*(y2-y_old) + x2*(y_old-y1))
                if d_old > 0:
                    removed_counts[d_old] += 1
                d_new = abs(x_new*(y1-y2) + x1*(y2-y_new) + x2*(y_new-y1))
                if d_new > 0:
                    added_counts[d_new] += 1
        return removed_counts, added_counts

    def compute_delta_vectorized(old_pt, new_pt, pts_arr, idx):
        """Fully vectorized delta computation."""
        x_old, y_old = old_pt
        x_new, y_new = new_pt
        
        others_mask = np.ones(N, dtype=bool)
        others_mask[idx] = False
        others_idx = np.where(others_mask)[0]
        others_pts = pts_arr[others_idx]
        
        n_others = len(others_idx)
        
        # Compute all pairs determinants with old point
        # For each pair (a,b) among others, compute det with old and new
        # Vectorize: create arrays of all pair coordinates
        pair_a = []
        pair_b = []
        for a in range(n_others):
            for b in range(a+1, n_others):
                pair_a.append(a)
                pair_b.append(b)
        
        pair_a = np.array(pair_a)
        pair_b = np.array(pair_b)
        
        # Get coordinates
        xa1 = others_pts[pair_a, 0]
        ya1 = others_pts[pair_a, 1]
        xa2 = others_pts[pair_b, 0]
        ya2 = others_pts[pair_b, 1]
        
        # Determinant with old point
        det_old = np.abs(x_old*(ya1-ya2) + xa1*(ya2-y_old) + xa2*(y_old-ya1))
        # Determinant with new point
        det_new = np.abs(x_new*(ya1-ya2) + xa1*(ya2-y_new) + xa2*(y_new-ya1))
        
        removed_counts = Counter(det_old[det_old > 0].tolist())
        added_counts = Counter(det_new[det_new > 0].tolist())
        
        return removed_counts, added_counts

    def generate_hexagonal_config():
        """Generate a hexagonal lattice configuration for better initial guess."""
        pts = set()
        for row in range(5):
            for col in range(6):
                x = col * 2
                y = row * 3
                if row % 2 == 1:
                    x += 1
                pts.add((x, y))
        if len(pts) < 30:
            for y in range(-10, 11):
                for x in range(-10, 11):
                    if len(pts) >= 30:
                        break
                    if (x, y) not in pts:
                        pts.add((x, y))
                if len(pts) >= 30:
                    break
        return list(pts)[:30]

    def generate_random_config():
        """Generate a random configuration."""
        pts = set()
        while len(pts) < N:
            x = random.randint(-bounds, bounds)
            y = random.randint(-bounds, bounds)
            pts.add((x, y))
        return list(pts)

    # Try multiple initial configurations and pick the best
    initial_configs = [
        [(i, j) for i in range(6) for j in range(5)],
        generate_hexagonal_config(),
        generate_random_config(),
        generate_random_config(),
    ]
    
    best_initial = None
    best_initial_score = -1
    for config in initial_configs:
        if len(config) != N or len(set(config)) != N:
            continue
        counts = compute_all_areas(config)
        score = get_max_count(counts)
        if score > best_initial_score:
            best_initial_score = score
            best_initial = config
    
    if best_initial is None:
        best_initial = [(i, j) for i in range(6) for j in range(5)]
        best_initial_score = get_max_count(compute_all_areas(best_initial))
    
    pts = best_initial[:]
    pts_arr = np.array(pts, dtype=np.int64)
    counts = compute_all_areas(pts)
    current_max_count = get_max_count(counts)
    best_max_count = current_max_count
    best_pts = pts[:]
    current_best_area = get_best_area(counts)
    best_area = current_best_area

    stagnation_count = 0
    max_stagnation = 3000

    while time.time() < deadline:
        if stagnation_count > max_stagnation:
            pts = best_pts[:]
            pts_arr = np.array(pts, dtype=np.int64)
            # Larger perturbation
            for _ in range(15):
                idx = random.randint(0, N-1)
                x, y = pts[idx]
                dx = random.randint(-12, 12)
                dy = random.randint(-12, 12)
                nx, ny = x + dx, y + dy
                if -bounds <= nx <= bounds and -bounds <= ny <= bounds:
                    new_pt = (nx, ny)
                    if new_pt not in pts:
                        pts[idx] = new_pt
                        pts_arr[idx] = new_pt
            counts = compute_all_areas(pts)
            current_max_count = get_max_count(counts)
            current_best_area = get_best_area(counts)
            if current_max_count > best_max_count:
                best_max_count = current_max_count
                best_pts = pts[:]
                best_area = current_best_area
            stagnation_count = 0
            continue

        idx = random.randint(0, N-1)
        x, y = pts[idx]
        
        if stagnation_count < 300:
            step_range = 2
        elif stagnation_count < 800:
            step_range = 3
        elif stagnation_count < 1500:
            step_range = 5
        else:
            step_range = 8
            
        dx = random.randint(-step_range, step_range)
        dy = random.randint(-step_range, step_range)
        if dx == 0 and dy == 0:
            continue
        nx, ny = x + dx, y + dy
        if nx < -bounds or nx > bounds or ny < -bounds or ny > bounds:
            stagnation_count += 1
            continue
        new_pt = (nx, ny)
        if new_pt in pts:
            stagnation_count += 1
            continue

        old_pt = pts[idx]
        removed, added = compute_delta_vectorized(old_pt, new_pt, pts_arr, idx)

        for area, cnt in removed.items():
            counts[area] -= cnt
            if counts[area] <= 0:
                del counts[area]
        for area, cnt in added.items():
            counts[area] += cnt

        new_max_count = get_max_count(counts)

        if new_max_count > current_max_count:
            pts[idx] = new_pt
            pts_arr[idx] = new_pt
            current_max_count = new_max_count
            current_best_area = get_best_area(counts)
            stagnation_count = 0
            if current_max_count > best_max_count:
                best_max_count = current_max_count
                best_pts = pts[:]
                best_area = current_best_area
        else:
            for area, cnt in added.items():
                counts[area] -= cnt
                if counts[area] <= 0:
                    del counts[area]
            for area, cnt in removed.items():
                counts[area] += cnt
            stagnation_count += 1

    # Final polish: focused on the best area
    if time.time() < deadline - 3:
        pts = best_pts[:]
        pts_arr = np.array(pts, dtype=np.int64)
        counts = compute_all_areas(pts)
        current_max_count = get_max_count(counts)
        current_best_area = get_best_area(counts)
        
        polish_deadline = min(deadline - 1, time.time() + 15)
        polish_stagnation = 0
        max_polish_stagnation = 800
        
        while time.time() < polish_deadline:
            if polish_stagnation > max_polish_stagnation:
                # Try a different target area
                top_areas = counts.most_common(5)
                if len(top_areas) > 1:
                    # Try second best area
                    pass
                break
            
            # Target the current best area
            target_area = get_best_area(counts)
            target_count = counts.get(target_area, 0)
            
            idx = random.randint(0, N-1)
            x, y = pts[idx]
            
            # Small steps for polish
            dx = random.randint(-2, 2)
            dy = random.randint(-2, 2)
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if nx < -bounds or nx > bounds or ny < -bounds or ny > bounds:
                polish_stagnation += 1
                continue
            new_pt = (nx, ny)
            if new_pt in pts:
                polish_stagnation += 1
                continue

            old_pt = pts[idx]
            removed, added, old_target, new_target = compute_delta_single(old_pt, new_pt, pts_arr, idx, target_area)
            
            # Quick check: does this improve target area count?
            if new_target <= old_target:
                polish_stagnation += 1
                continue
            
            # Full update
            for area, cnt in removed.items():
                counts[area] -= cnt
                if counts[area] <= 0:
                    del counts[area]
            for area, cnt in added.items():
                counts[area] += cnt

            new_max_count = get_max_count(counts)

            if new_max_count > current_max_count:
                pts[idx] = new_pt
                pts_arr[idx] = new_pt
                current_max_count = new_max_count
                current_best_area = get_best_area(counts)
                polish_stagnation = 0
                if current_max_count > best_max_count:
                    best_max_count = current_max_count
                    best_pts = pts[:]
                    best_area = current_best_area
            else:
                for area, cnt in added.items():
                    counts[area] -= cnt
                    if counts[area] <= 0:
                        del counts[area]
                for area, cnt in removed.items():
                    counts[area] += cnt
                polish_stagnation += 1

    return best_pts

# EVOLVE-BLOCK-START
# (removed - the construct function above is the only one)
# EVOLVE-BLOCK-END
