import time
import random
import math
from collections import Counter

def construct():
    deadline = time.time() + 75.0
    random.seed(42)
    N = 30

    # Precompute all triplets
    triplets = []
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                triplets.append((i, j, k))

    # Precompute for each index which triplets it participates in
    idx_triplets = [[] for _ in range(N)]
    for t_idx, (i, j, k) in enumerate(triplets):
        idx_triplets[i].append(t_idx)
        idx_triplets[j].append(t_idx)
        idx_triplets[k].append(t_idx)

    def doubled_area(p1, p2, p3):
        return abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))

    def full_count(points):
        areas = []
        for i, j, k in triplets:
            a = doubled_area(points[i], points[j], points[k])
            if a > 0:
                areas.append(a)
        if not areas:
            return Counter(), 0, 0
        c = Counter(areas)
        max_area, max_count = c.most_common(1)[0]
        return c, max_area, max_count

    def count_target(points, target):
        count = 0
        for i, j, k in triplets:
            if doubled_area(points[i], points[j], points[k]) == target:
                count += 1
        return count

    def generate_grid_config():
        """Generate a 6x5 grid."""
        return [(x, y) for x in range(-2, 4) for y in range(-2, 3)]

    def generate_dense_config():
        """Generate a dense 5x6 grid centered at origin."""
        return [(x, y) for x in range(-2, 3) for y in range(-3, 3)]

    def generate_lattice_config(scale=1):
        """Generate a lattice-based config for better area concentration."""
        points = []
        for q in range(-4, 5):
            for r in range(-4, 5):
                x = 2*q + r
                y = r
                if -30 <= x <= 30 and -30 <= y <= 30:
                    points.append((x, y))
        random.shuffle(points)
        return points[:30]

    # Try multiple initial configurations
    initial_configs = [
        generate_grid_config(),
        generate_dense_config(),
        generate_lattice_config(1),
        generate_lattice_config(2),
        [(x, y) for x in range(-3, 3) for y in range(-3, 3) if abs(x) + abs(y) <= 4][:30],
    ]

    best_initial = None
    best_initial_count = 0

    for config in initial_configs:
        seen = set()
        filtered = []
        for p in config:
            if p not in seen and -30 <= p[0] <= 30 and -30 <= p[1] <= 30:
                seen.add(p)
                filtered.append(p)
        while len(filtered) < 30:
            p = (random.randint(-30, 30), random.randint(-30, 30))
            if p not in seen:
                seen.add(p)
                filtered.append(p)
        filtered = filtered[:30]

        counter, ba, bc = full_count(filtered)
        if bc > best_initial_count:
            best_initial_count = bc
            best_initial = filtered

    coords = list(best_initial)
    counter, target, current_count = full_count(coords)
    best_overall = current_count
    best_points = list(coords)
    best_target = target

    # CSP-inspired targeted search: try specific small target areas
    # and use backtracking-like local search to maximize triples with that area
    small_targets = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
    
    for t_area in small_targets:
        if time.time() > deadline - 15:
            break
            
        # Start from best_points and try to optimize for this target
        test_coords = list(best_points)
        test_count = count_target(test_coords, t_area)
        
        # Quick local search for this target
        T = 5.0
        cooling = 0.995
        no_imp = 0
        max_no_imp = 100
        
        for step in range(500):
            if time.time() > deadline - 15:
                break
            if no_imp > max_no_imp:
                break
                
            idx = random.randint(0, N-1)
            old_pt = test_coords[idx]
            
            # Small neighborhood
            candidates = []
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    new_pt = (old_pt[0] + dx, old_pt[1] + dy)
                    if new_pt != old_pt and new_pt not in test_coords and -30 <= new_pt[0] <= 30 and -30 <= new_pt[1] <= 30:
                        candidates.append(new_pt)
            
            if not candidates:
                no_imp += 1
                T *= cooling
                continue
            
            best_delta = -999
            best_new_pt = None
            
            old_matches = 0
            for t_idx in idx_triplets[idx]:
                i, j, k = triplets[t_idx]
                if doubled_area(test_coords[i], test_coords[j], test_coords[k]) == t_area:
                    old_matches += 1
            
            for new_pt in candidates:
                new_matches = 0
                for t_idx in idx_triplets[idx]:
                    i, j, k = triplets[t_idx]
                    if i == idx:
                        p1, p2, p3 = new_pt, test_coords[j], test_coords[k]
                    elif j == idx:
                        p1, p2, p3 = test_coords[i], new_pt, test_coords[k]
                    else:
                        p1, p2, p3 = test_coords[i], test_coords[j], new_pt
                    if doubled_area(p1, p2, p3) == t_area:
                        new_matches += 1
                
                delta = new_matches - old_matches
                if delta > best_delta:
                    best_delta = delta
                    best_new_pt = new_pt
            
            if best_new_pt is not None:
                if best_delta > 0:
                    accept = True
                else:
                    if T > 0.5:
                        accept_prob = math.exp(best_delta / T)
                        accept = random.random() < accept_prob
                    else:
                        accept = False
                
                if accept:
                    test_coords[idx] = best_new_pt
                    test_count += best_delta
                    no_imp = 0
                else:
                    no_imp += 1
            
            T *= cooling
        
        if test_count > best_overall:
            best_overall = test_count
            best_points = list(test_coords)
            best_target = t_area

    # Main simulated annealing with target-focused optimization
    coords = list(best_points)
    target = best_target
    current_count = best_overall
    
    T = 10.0
    cooling_rate = 0.99

    no_improve = 0
    max_no_improve = 150
    restarts = 0
    max_restarts = 8

    while time.time() < deadline - 5:
        if no_improve > max_no_improve:
            restarts += 1
            if restarts > max_restarts:
                break

            # Re-evaluate current state
            counter, ba, bc = full_count(coords)
            if bc > best_overall:
                best_overall = bc
                best_points = list(coords)
                best_target = ba

            # Switch to the most common area in current configuration
            target = ba
            current_count = bc
            no_improve = 0
            max_no_improve += 50
            T = 10.0

            # Random restart with perturbation
            for _ in range(3):
                idx = random.randint(0, N-1)
                new_pt = (random.randint(-30, 30), random.randint(-30, 30))
                if new_pt not in coords:
                    coords[idx] = new_pt
                    current_count = count_target(coords, target)
                    break

            if current_count < best_overall:
                coords = list(best_points)
                target = best_target
                current_count = best_overall
            continue

        # Pick a random point to move
        idx = random.randint(0, N-1)
        old_pt = coords[idx]

        # Generate candidates
        candidates = []
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                new_pt = (old_pt[0] + dx, old_pt[1] + dy)
                if new_pt != old_pt and new_pt not in coords and -30 <= new_pt[0] <= 30 and -30 <= new_pt[1] <= 30:
                    candidates.append(new_pt)
        for _ in range(5):
            new_pt = (random.randint(-30, 30), random.randint(-30, 30))
            if new_pt != old_pt and new_pt not in coords:
                candidates.append(new_pt)

        if not candidates:
            no_improve += 1
            T *= cooling_rate
            continue

        best_delta = -999
        best_new_pt = None

        # Precompute old matches for this idx
        old_matches = 0
        for t_idx in idx_triplets[idx]:
            i, j, k = triplets[t_idx]
            if doubled_area(coords[i], coords[j], coords[k]) == target:
                old_matches += 1

        for new_pt in candidates:
            new_matches = 0
            for t_idx in idx_triplets[idx]:
                i, j, k = triplets[t_idx]
                if i == idx:
                    p1, p2, p3 = new_pt, coords[j], coords[k]
                elif j == idx:
                    p1, p2, p3 = coords[i], new_pt, coords[k]
                else:
                    p1, p2, p3 = coords[i], coords[j], new_pt
                if doubled_area(p1, p2, p3) == target:
                    new_matches += 1

            delta = new_matches - old_matches
            if delta > best_delta:
                best_delta = delta
                best_new_pt = new_pt

        if best_new_pt is not None:
            if best_delta > 0:
                accept = True
            else:
                if T > 0.1:
                    accept_prob = math.exp(best_delta / T)
                    accept = random.random() < accept_prob
                else:
                    accept = False

            if accept:
                coords[idx] = best_new_pt
                current_count += best_delta
                no_improve = 0
                if current_count > best_overall:
                    best_overall = current_count
                    best_points = list(coords)
                    best_target = target
            else:
                no_improve += 1

        T *= cooling_rate

    # Final polish pass: high-precision local search
    if time.time() < deadline - 2:
        coords = list(best_points)
        counter, target, current_count = full_count(coords)
        
        # Greedy hill climbing with small steps
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for idx in range(N):
                if time.time() > deadline - 2:
                    break
                old_pt = coords[idx]
                
                # Very small neighborhood
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if dx == 0 and dy == 0:
                            continue
                        new_pt = (old_pt[0] + dx, old_pt[1] + dy)
                        if new_pt in coords or new_pt == old_pt:
                            continue
                        if not (-30 <= new_pt[0] <= 30 and -30 <= new_pt[1] <= 30):
                            continue
                        
                        # Compute delta
                        old_matches = 0
                        for t_idx in idx_triplets[idx]:
                            i, j, k = triplets[t_idx]
                            if doubled_area(coords[i], coords[j], coords[k]) == target:
                                old_matches += 1
                        
                        new_matches = 0
                        for t_idx in idx_triplets[idx]:
                            i, j, k = triplets[t_idx]
                            if i == idx:
                                p1, p2, p3 = new_pt, coords[j], coords[k]
                            elif j == idx:
                                p1, p2, p3 = coords[i], new_pt, coords[k]
                            else:
                                p1, p2, p3 = coords[i], coords[j], new_pt
                            if doubled_area(p1, p2, p3) == target:
                                new_matches += 1
                        
                        delta = new_matches - old_matches
                        if delta > 0:
                            coords[idx] = new_pt
                            current_count += delta
                            improved = True
                            if current_count > best_overall:
                                best_overall = current_count
                                best_points = list(coords)
                            break
                    if improved:
                        break

    # Final verification
    if len(best_points) != N or len(set(best_points)) != N:
        return [(x, y) for x in range(6) for y in range(5)]
    for p in best_points:
        if not (-30 <= p[0] <= 30 and -30 <= p[1] <= 30):
            return [(x, y) for x in range(6) for y in range(5)]

    return best_points
