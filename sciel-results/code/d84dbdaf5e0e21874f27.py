import random
import math
import itertools

def construct():
    def gcd(a, b):
        a, b = abs(a), abs(b)
        while b:
            a, b = b, a % b
        return a

    def score(pts):
        """Score: count lines with exactly 4 points, penalize 5+ collinear."""
        n = len(pts)
        if n != 16:
            return -1000000
        
        # Check distinct
        if len(set(map(tuple, pts))) != 16:
            return -1000000
        
        # Check bounds
        for p in pts:
            if abs(p[0]) > 1000 or abs(p[1]) > 1000:
                return -1000000
        
        # Count points on each line
        line_points = {}
        
        for i in range(16):
            for j in range(i + 1, 16):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                
                A = y2 - y1
                B = x1 - x2
                C = A * x1 + B * y1
                
                g = gcd(gcd(A, B), C)
                if g == 0:
                    continue
                A_n = A // g
                B_n = B // g
                C_n = C // g
                
                if A_n < 0 or (A_n == 0 and B_n < 0) or (A_n == 0 and B_n == 0 and C_n < 0):
                    A_n = -A_n
                    B_n = -B_n
                    C_n = -C_n
                    
                key = (A_n, B_n, C_n)
                
                if key not in line_points:
                    line_points[key] = set()
                line_points[key].add(i)
                line_points[key].add(j)
        
        count_4 = 0
        for key, point_set in line_points.items():
            k = len(point_set)
            if k > 4:
                return -1000000
            elif k == 4:
                count_4 += 1
        
        return count_4

    # Start from a known good configuration: 4x4 grid
    base_points = [[x, y] for x in range(4) for y in range(4)]
    
    def perturb(points, strength):
        """Create a perturbed version of points."""
        new_pts = [p[:] for p in points]
        for idx in range(16):
            if random.random() < strength:
                dx = random.randint(-3, 3)
                dy = random.randint(-3, 3)
                new_pts[idx][0] += dx
                new_pts[idx][1] += dy
                if abs(new_pts[idx][0]) > 1000 or abs(new_pts[idx][1]) > 1000:
                    new_pts[idx][0] = random.randint(-1000, 1000)
                    new_pts[idx][1] = random.randint(-1000, 1000)
        return new_pts

    def simulate_annealing(initial_pts, max_iters=20000, t_start=100, t_end=1):
        """Simulated annealing to find better configurations."""
        current_pts = [p[:] for p in initial_pts]
        current_score = score(current_pts)
        best_pts = [p[:] for p in current_pts]
        best_score = current_score
        
        for it in range(max_iters):
            # Annealing schedule
            t = t_start * (t_end / t_start) ** (it / max_iters)
            
            # Perturb
            new_pts = perturb(current_pts, 0.3)
            new_score = score(new_pts)
            
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
                if new_score > best_score:
                    best_pts = [p[:] for p in new_pts]
                    best_score = new_score
            else:
                # Accept worse with probability exp((new - current)/t)
                if random.random() < math.exp((new_score - current_score) / max(t, 0.01)):
                    current_pts = new_pts
                    current_score = new_score
        
        return best_pts, best_score

    # Try multiple starting configurations
    candidates = [
        base_points,
        # Slightly modified grids
        [[x, y] for x in range(4) for y in range(4)] if False else None,
    ]
    
    # Generate some alternative initial configurations
    alt_configs = []
    
    # Configuration 1: Standard 4x4 grid
    alt_configs.append(base_points)
    
    # Configuration 2: Shifted grid
    alt_configs.append([[x + 1, y + 1] for x in range(4) for y in range(4)])
    
    # Configuration 3: Different spacing
    alt_configs.append([[x * 2, y * 2] for x in range(4) for y in range(4)])
    
    # Configuration 4: Parallelogram-like
    alt_configs.append([[x + y // 2, y] for x in range(4) for y in range(4)])
    
    best_overall = base_points
    best_overall_score = -1
    
    random.seed(42)
    
    for config in alt_configs:
        # Run simulated annealing from this config
        result_pts, result_score = simulate_annealing(config, max_iters=15000)
        
        if result_score > best_overall_score:
            best_overall = result_pts
            best_overall_score = result_score
    
    # Final polish: local search on the best configuration
    current_pts = [p[:] for p in best_overall]
    current_score = score(current_pts)
    
    for _ in range(10000):
        idx = random.randint(0, 15)
        old_point = current_pts[idx][:]
        
        # Try several moves
        best_local_move = None
        best_local_score = current_score
        
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                if dx == 0 and dy == 0:
                    continue
                new_point = [old_point[0] + dx, old_point[1] + dy]
                if abs(new_point[0]) > 1000 or abs(new_point[1]) > 1000:
                    continue
                
                # Check distinctness
                is_distinct = True
                for j, p in enumerate(current_pts):
                    if j != idx and p == new_point:
                        is_distinct = False
                        break
                if not is_distinct:
                    continue
                
                current_pts[idx] = new_point
                s = score(current_pts)
                current_pts[idx] = old_point
                
                if s > best_local_score:
                    best_local_score = s
                    best_local_move = new_point
        
        if best_local_move:
            current_pts[idx] = best_local_move
            current_score = best_local_score
            if current_score > best_overall_score:
                best_overall = [p[:] for p in current_pts]
                best_overall_score = current_score
        else:
            # No improvement, try a random jump
            if random.random() < 0.1:
                idx2 = random.randint(0, 15)
                current_pts[idx2][0] = random.randint(-1000, 1000)
                current_pts[idx2][1] = random.randint(-1000, 1000)
                s = score(current_pts)
                if s > best_overall_score:
                    best_overall = [p[:] for p in current_pts]
                    best_overall_score = s
    
    return best_overall
