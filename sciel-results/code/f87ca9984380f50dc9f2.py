import time
import random
import numpy as np
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin for 60s limit
    
    def calc_areas(points):
        """Calculate full area frequency histogram."""
        n = len(points)
        areas = {}
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                for k in range(j + 1, n):
                    x3, y3 = points[k]
                    det = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                    if det > 0:
                        areas[det] = areas.get(det, 0) + 1
        return areas

    def get_best_score(areas):
        """Get max multiplicity and its area from histogram."""
        if not areas:
            return 0, 0
        max_mult = max(areas.values())
        # Find the area with max multiplicity
        for a, m in areas.items():
            if m == max_mult:
                return max_mult, a
        return 0, 0

    # Generate diverse initial configurations
    candidates = []
    
    # Standard grids
    for w, h in [(6, 5), (5, 6), (3, 10), (10, 3), (2, 15), (15, 2)]:
        pts = [(x, y) for x in range(w) for y in range(h)]
        if len(pts) == 30 and len(set(pts)) == 30:
            cx, cy = w // 2, h // 2
            pts = [(x - cx, y - cy) for x, y in pts]
            candidates.append(pts)
            
    # Sheared grids
    for w, h in [(6, 5), (5, 6), (3, 10)]:
        for k in [1, -1, 2, -2]:
            pts = []
            valid = True
            for x in range(w):
                for y in range(h):
                    px = x + k * y
                    py = y
                    if not (-30 <= px <= 30 and -30 <= py <= 30):
                        valid = False
                        break
                    pts.append((px, py))
                if not valid: break
            if valid and len(pts) == 30 and len(set(pts)) == 30:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                cx = (min(xs) + max(xs)) // 2
                cy = (min(ys) + max(ys)) // 2
                pts = [(x - cx, y - cy) for x, y in pts]
                candidates.append(pts)

    # More random configurations
    for _ in range(80):
        if time.time() > deadline - 20:
            break
        pts = []
        attempts = 0
        while len(pts) < 30 and attempts < 300:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            if (x, y) not in pts:
                pts.append((x, y))
            attempts += 1
        if len(pts) == 30:
            candidates.append(pts)

    # Evaluate all candidates to find best initial
    best_score = 0
    best_points = None
    best_areas = {}
    
    for pts in candidates:
        areas = calc_areas(pts)
        score, _ = get_best_score(areas)
        if score > best_score:
            best_score = score
            best_points = list(pts)
            best_areas = areas
            
    if best_points is None:
        # Fallback
        best_points = [(i, 0) for i in range(30)]
        best_areas = calc_areas(best_points)
        best_score, _ = get_best_score(best_areas)

    current_points = list(best_points)
    current_areas = dict(best_areas)
    current_score = best_score
    
    # Local search with pairwise swaps and dynamic target area
    no_improve_count = 0
    max_no_improve = 300
    steps = 0
    
    while time.time() < deadline - 3:
        steps += 1
        
        # Get current best area (dynamic target)
        _, target_area = get_best_score(current_areas)
        
        # Try random pairwise swaps
        found_improvement = False
        i, j = random.sample(range(30), 2)
        old_i, old_j = current_points[i], current_points[j]
        
        # Try swapping positions
        new_i, new_j = old_j, old_i
        
        # Check distinctness after swap (they're already distinct since i != j)
        # Calculate new areas
        new_points = current_points.copy()
        new_points[i] = new_i
        new_points[j] = new_j
        
        # Recalculate areas efficiently - only triples involving i or j change
        new_areas = dict(current_areas)
        
        # Remove contributions from triples involving i or j
        n = 30
        for a in range(n):
            if a == i or a == j:
                continue
            # Triple (i, j, a)
            x1, y1 = old_i
            x2, y2 = old_j
            x3, y3 = new_points[a]
            det_ij = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
            if det_ij > 0:
                new_areas[det_ij] = new_areas.get(det_ij, 0) - 1
                if new_areas[det_ij] == 0:
                    del new_areas[det_ij]
            
            # Triple (i, a, j) - same as (i, j, a) already counted
            
            # Triples (i, a, b) where a, b != i, j
            # and (j, a, b) where a, b != i, j
            # This is getting complex. Let's just recalculate if we have time.
        
        # For simplicity and correctness, recalculate if swap is promising
        # First, check if the swap changes the target area count
        # Quick check: compute areas of triples involving i or j in new config
        target_count_new = 0
        for a in range(n):
            if a == i or a == j:
                continue
            # (new_i, new_j, a)
            x1, y1 = new_i
            x2, y2 = new_j
            x3, y3 = new_points[a]
            det = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
            if det == target_area:
                target_count_new += 1
            
            # (new_i, a, b) for b > a, a,b != i,j
            # This is too complex for quick check. Just do full recalc if needed.
        
        # Let's use a simpler approach: just do full recalculation for swaps
        # since the point-move approach was too slow, but swaps change fewer triples
        # Actually, let's just try the swap and recalculate
        new_points = current_points.copy()
        new_points[i], new_points[j] = new_j, new_i
        
        # Quick sanity: check if this swap might improve
        # Count how many triples with target_area involve i or j in old config
        old_target_count = 0
        for a in range(n):
            if a == i or a == j:
                continue
            x1, y1 = old_i
            x2, y2 = old_j
            x3, y3 = current_points[a]
            det = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
            if det == target_area:
                old_target_count += 1
        
        # For a swap, the triples that change are those containing i or j
        # Let's just do full recalculation - it's O(n^3) = 4060 ops, very fast
        if steps % 10 == 0 or random.random() < 0.3:
            new_areas_full = calc_areas(new_points)
            new_score, _ = get_best_score(new_areas_full)
            if new_score > current_score:
                current_points = new_points
                current_areas = new_areas_full
                current_score = new_score
                found_improvement = True
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        if found_improvement:
            continue
        
        # If stuck, try some random restarts or larger moves
        if no_improve_count > max_no_improve:
            # Try a few random configurations
            for _ in range(5):
                if time.time() > deadline - 5:
                    break
                pts = []
                attempts = 0
                while len(pts) < 30 and attempts < 200:
                    x = random.randint(-30, 30)
                    y = random.randint(-30, 30)
                    if (x, y) not in pts:
                        pts.append((x, y))
                    attempts += 1
                if len(pts) == 30:
                    areas = calc_areas(pts)
                    score, _ = get_best_score(areas)
                    if score > current_score:
                        current_points = pts
                        current_areas = areas
                        current_score = score
                        no_improve_count = 0
                        break
            else:
                # Reset to best and try again
                no_improve_count = 0

    # Final evaluation
    final_areas = calc_areas(current_points)
    final_score, _ = get_best_score(final_areas)
    if final_score > best_score:
        best_score = final_score
        best_points = current_points

    return best_points

def _calculate_max_multiplicity(points):
    """Calculate the maximum multiplicity of any positive doubled area."""
    n = len(points)
    if n < 3:
        return 0, 0
        
    areas = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            for k in range(j + 1, n):
                x3, y3 = points[k]
                det = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                if det > 0:
                    areas[det] = areas.get(det, 0) + 1
                    
    if not areas:
        return 0, 0
        
    max_mult = max(areas.values())
    max_area = [a for a, m in areas.items() if m == max_mult][0]
    return max_mult, max_area
