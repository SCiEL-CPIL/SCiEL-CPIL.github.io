import time
import math
import random

def construct():
    start_time = time.time()
    deadline = 143.0  # Safety margin for 150s limit

    def elapsed():
        return time.time() - start_time

    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        g = math.gcd(A, math.gcd(B, C))
        if g == 0:
            return None
        A //= g
        B //= g
        C //= g
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
        return (A, B, C)

    def count_collinear_triples(pts):
        """Count total number of collinear triples in the point set."""
        n = len(pts)
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                for k in range(j + 1, n):
                    x3, y3 = pts[k]
                    if dx * (y3 - y1) == dy * (x3 - x1):
                        count += 1
        return count

    def has_4_collinear_fast(pts):
        """Check if any line contains 4+ points. Returns (bool, line_key if found)."""
        n = len(pts)
        seen_lines = set()
        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(pts[i], pts[j])
                if key is None or key in seen_lines:
                    continue
                seen_lines.add(key)
                A, B, C = key
                count = 0
                for k in range(n):
                    x3, y3 = pts[k]
                    if A * x3 + B * y3 + C == 0:
                        count += 1
                        if count >= 4:
                            return True, key
        return False, None

    def calculate_score(pts):
        """Exact branch and bound for largest subset with no three collinear."""
        n = len(pts)
        if n < 3:
            return n
        
        # Precompute: for each pair (i,j), bitmask of k collinear with both
        collinear_with = [[0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                mask = 0
                for k in range(n):
                    if k == i or k == j:
                        continue
                    x3, y3 = pts[k]
                    if dx * (y3 - y1) == dy * (x3 - x1):
                        mask |= (1 << k)
                collinear_with[i][j] = mask
                collinear_with[j][i] = mask

        best_score = 0
        time_limit = deadline - 2.0
        
        def bbb(step, current_mask, current_size):
            nonlocal best_score
            if elapsed() > time_limit:
                return
            if current_size > best_score:
                best_score = current_size
            if step == n:
                return
            if current_size + (n - step) <= best_score:
                return
                
            idx = step
            
            # Option 1: Exclude idx
            bbb(step + 1, current_mask, current_size)
            
            # Option 2: Include idx
            valid = True
            temp_mask = current_mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                j = lsb.bit_length() - 1
                temp_mask ^= lsb
                if collinear_with[idx][j] & current_mask:
                    valid = False
                    break
                    
            if valid:
                bbb(step + 1, current_mask | (1 << idx), current_size + 1)
                
        bbb(0, 0, 0)
        return best_score

    def make_parabola_seed():
        """Points on a perturbed parabola y = x^2 / 2."""
        pts = []
        for i in range(20):
            x = i - 10
            y = (x * x) // 2 + random.randint(-2, 2)
            if -100 <= y <= 100:
                pts.append((x, y))
            else:
                pts.append((x, 0))
        # Ensure distinct
        seen = set()
        unique = []
        for p in pts:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        while len(unique) < 20:
            p = (random.randint(-100, 100), random.randint(-100, 100))
            if p not in seen:
                seen.add(p)
                unique.append(p)
        return unique

    def make_circle_seed():
        """Points near a circle of radius 50."""
        pts = []
        for i in range(20):
            angle = 2 * math.pi * i / 20 + random.uniform(-0.1, 0.1)
            r = 50 + random.randint(-5, 5)
            x = int(round(r * math.cos(angle)))
            y = int(round(r * math.sin(angle)))
            x = max(-100, min(100, x))
            y = max(-100, min(100, y))
            pts.append((x, y))
        seen = set()
        unique = []
        for p in pts:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        while len(unique) < 20:
            p = (random.randint(-100, 100), random.randint(-100, 100))
            if p not in seen:
                seen.add(p)
                unique.append(p)
        return unique

    def make_random_seed():
        pts = []
        seen = set()
        while len(pts) < 20:
            p = (random.randint(-100, 100), random.randint(-100, 100))
            if p not in seen:
                seen.add(p)
                pts.append(p)
        return pts

    def make_grid_seed():
        """4x5 grid with small perturbations."""
        pts = []
        for y in range(4):
            for x in range(5):
                px = x * 20 - 40 + random.randint(-3, 3)
                py = y * 25 - 30 + random.randint(-3, 3)
                px = max(-100, min(100, px))
                py = max(-100, min(100, py))
                pts.append((px, py))
        seen = set()
        unique = []
        for p in pts:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        while len(unique) < 20:
            p = (random.randint(-100, 100), random.randint(-100, 100))
            if p not in seen:
                seen.add(p)
                unique.append(p)
        return unique

    def repair_4_collinear(pts):
        """Repair a point set to have no 4 collinear. Returns (pts, success)."""
        pts = list(pts)
        for _ in range(200):
            has4, _ = has_4_collinear_fast(pts)
            if not has4:
                return pts, True
            # Find a 4-collinear line and move one point
            seen_lines = set()
            moved = False
            for i in range(20):
                for j in range(i + 1, 20):
                    key = get_line_key(pts[i], pts[j])
                    if key is None or key in seen_lines:
                        continue
                    seen_lines.add(key)
                    A, B, C = key
                    line_pts = []
                    for k in range(20):
                        x3, y3 = pts[k]
                        if A * x3 + B * y3 + C == 0:
                            line_pts.append(k)
                    if len(line_pts) >= 4:
                        idx = random.choice(line_pts)
                        old_p = pts[idx]
                        for _ in range(100):
                            new_p = (old_p[0] + random.randint(-8, 8), old_p[1] + random.randint(-8, 8))
                            if new_p in pts or not (-100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100):
                                continue
                            test_pts = pts[:]
                            test_pts[idx] = new_p
                            h4, _ = has_4_collinear_fast(test_pts)
                            if not h4:
                                pts = test_pts
                                moved = True
                                break
                        if moved:
                            break
                if moved:
                    break
            if not moved:
                # Try bigger jump
                idx = random.randint(0, 19)
                for _ in range(50):
                    new_p = (random.randint(-100, 100), random.randint(-100, 100))
                    if new_p in pts:
                        continue
                    test_pts = pts[:]
                    test_pts[idx] = new_p
                    h4, _ = has_4_collinear_fast(test_pts)
                    if not h4:
                        pts = test_pts
                        moved = True
                        break
                if not moved:
                    return pts, False
        return pts, True

    def simulated_annealing(seed_pts, max_time=30.0):
        """SA that minimizes number of collinear triples while enforcing no-4-collinear."""
        current_pts, _ = repair_4_collinear(seed_pts)
        if elapsed() > deadline - 5:
            return None, float('inf')
        
        current_triples = count_collinear_triples(current_pts)
        best_pts = current_pts[:]
        best_triples = current_triples
        best_score = calculate_score(current_pts)
        
        T = 10.0
        T_min = 0.1
        cooling = 0.995
        
        while elapsed() < deadline - 5 and T > T_min:
            if elapsed() > deadline - 5:
                break
            
            idx = random.randint(0, 19)
            old_p = current_pts[idx]
            
            # Generate neighbor
            if random.random() < 0.7:
                # Small perturbation
                new_p = (old_p[0] + random.randint(-5, 5), old_p[1] + random.randint(-5, 5))
            else:
                # Larger jump
                new_p = (old_p[0] + random.randint(-15, 15), old_p[1] + random.randint(-15, 15))
            
            new_p = (max(-100, min(100, new_p[0])), max(-100, min(100, new_p[1])))
            
            if new_p in current_pts:
                continue
            
            new_pts = current_pts[:]
            new_pts[idx] = new_p
            
            # Check no-4-collinear constraint
            h4, _ = has_4_collinear_fast(new_pts)
            if h4:
                # Reject moves that violate constraint
                T *= cooling
                continue
            
            new_triples = count_collinear_triples(new_pts)
            delta = new_triples - current_triples
            
            # Accept if better or with probability exp(-delta/T)
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_pts = new_pts
                current_triples = new_triples
                
                if current_triples < best_triples:
                    best_triples = current_triples
                    best_pts = current_pts[:]
                    # Only recompute exact score periodically to save time
                    if elapsed() < deadline - 10:
                        s = calculate_score(current_pts)
                        if s < best_score:
                            best_score = s
                elif current_triples == best_triples and random.random() < 0.01:
                    if elapsed() < deadline - 10:
                        s = calculate_score(current_pts)
                        if s < best_score:
                            best_score = s
                            best_pts = current_pts[:]
            
            T *= cooling
            
            # Occasional restart if stuck
            if random.random() < 0.005:
                # Perturb a random point significantly
                idx2 = random.randint(0, 19)
                new_p2 = (random.randint(-100, 100), random.randint(-100, 100))
                if new_p2 not in current_pts:
                    test = current_pts[:]
                    test[idx2] = new_p2
                    h4, _ = has_4_collinear_fast(test)
                    if not h4:
                        current_pts = test
                        current_triples = count_collinear_triples(current_pts)
        
        # Final exact score
        if elapsed() < deadline - 3:
            final_score = calculate_score(best_pts)
            if final_score < best_score:
                best_score = final_score
        
        return best_pts, best_score

    best_points = None
    best_val = float('inf')
    
    # Generate diverse seeds
    seeds = [
        make_parabola_seed(),
        make_circle_seed(),
        make_random_seed(),
        make_grid_seed(),
        make_random_seed(),
        make_parabola_seed(),
        make_circle_seed(),
        make_grid_seed(),
    ]
    random.shuffle(seeds)
    
    # Allocate time: ~30s per SA run, up to 4 runs
    sa_time_per_run = 30.0
    
    for seed in seeds:
        if elapsed() > deadline - 15:
            break
        
        pts, score = simulated_annealing(seed, max_time=sa_time_per_run)
        
        if pts is not None and score < best_val:
            best_val = score
            best_points = pts
            # If we found a good score, we can try more runs
            if best_val <= 8:
                sa_time_per_run = 20.0
    
    # If we have time, do one more focused SA from the best point
    if best_points is not None and elapsed() < deadline - 20:
        # Perturb the best point slightly and run SA again
        perturbed = best_points[:]
        for _ in range(3):
            idx = random.randint(0, 19)
            old = perturbed[idx]
            new = (old[0] + random.randint(-10, 10), old[1] + random.randint(-10, 10))
            new = (max(-100, min(100, new[0])), max(-100, min(100, new[1])))
            if new not in perturbed:
                perturbed[idx] = new
        
        pts, score = simulated_annealing(perturbed, max_time=25.0)
        if pts is not None and score < best_val:
            best_val = score
            best_points = pts
    
    if best_points is None:
        # Fallback
        best_points, _ = repair_4_collinear(make_random_seed())
        if has_4_collinear_fast(best_points)[0]:
            # Last resort: fully random
            best_points = make_random_seed()
            # Try to repair
            best_points, _ = repair_4_collinear(best_points)

    # Final validation
    final_pts = list(best_points)
    h4, _ = has_4_collinear_fast(final_pts)
    if h4:
        final_pts, _ = repair_4_collinear(final_pts)
    
    return [[x, y] for x, y in final_pts]

if __name__ == "__main__":
    pts = construct()
    print(pts)
