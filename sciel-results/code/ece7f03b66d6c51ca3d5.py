import time
import random
import math

DEADLINE = 115.0

def construct():
    start_time = time.time()
    random.seed(42)
    
    def are_collinear(p1, p2, p3):
        return (p2[0]-p1[0])*(p3[1]-p1[1]) == (p3[0]-p1[0])*(p2[1]-p1[1])
    
    def has_four_collinear(points):
        n = len(points)
        for i in range(n):
            slopes = {}
            xi, yi = points[i]
            for j in range(n):
                if i == j:
                    continue
                xj, yj = points[j]
                dx = xj - xi
                dy = yj - yi
                if dx == 0 and dy == 0:
                    continue
                if dx == 0:
                    slope = (0, 1)
                elif dy == 0:
                    slope = (1, 0)
                else:
                    g = math.gcd(abs(dx), abs(dy))
                    dx, dy = dx // g, dy // g
                    if dx < 0:
                        dx, dy = -dx, -dy
                    slope = (dx, dy)
                if slope in slopes:
                    slopes[slope] += 1
                    if slopes[slope] >= 3:
                        return True
                else:
                    slopes[slope] = 1
        return False
    
    def count_triples(points):
        """Count collinear triples - fast surrogate."""
        count = 0
        for i in range(20):
            slopes = {}
            xi, yi = points[i]
            for j in range(20):
                if i == j:
                    continue
                xj, yj = points[j]
                dx = xj - xi
                dy = yj - yi
                if dx == 0 and dy == 0:
                    continue
                if dx == 0:
                    slope = (0, 1)
                elif dy == 0:
                    slope = (1, 0)
                else:
                    g = math.gcd(abs(dx), abs(dy))
                    dx, dy = dx // g, dy // g
                    if dx < 0:
                        dx, dy = -dx, -dy
                    slope = (dx, dy)
                if slope in slopes:
                    slopes[slope] += 1
                else:
                    slopes[slope] = 1
            for k in slopes.values():
                if k >= 2:
                    count += k * (k - 1) // 2
        return count
    
    def calc_score(points):
        """Exact B&B for largest subset with no 3 collinear, using bitmask."""
        n = len(points)
        
        # Precompute collinear triples
        triple_count = [0] * n
        triples = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        triple_count[i] += 1
                        triple_count[j] += 1
                        triple_count[k] += 1
                        triples.append((i, j, k))
        
        # Sort by triple count descending (hard vertices first)
        order = sorted(range(n), key=lambda v: -triple_count[v])
        pos = {v: i for i, v in enumerate(order)}
        
        # For each position, store forbidden pairs
        forbidden = [[] for _ in range(n)]
        for i, j, k in triples:
            pi, pj, pk = pos[i], pos[j], pos[k]
            forbidden[pi].append((pj, pk))
            forbidden[pj].append((pi, pk))
            forbidden[pk].append((pi, pj))
        
        best = [0]
        chosen = [False] * n
        
        def backtrack(pos_idx, current_size):
            if pos_idx == n:
                if current_size > best[0]:
                    best[0] = current_size
                return
            if current_size + (n - pos_idx) <= best[0]:
                return
            
            v = pos_idx
            valid = True
            for u, w in forbidden[v]:
                if chosen[u] and chosen[w]:
                    valid = False
                    break
            
            if valid:
                chosen[v] = True
                backtrack(pos_idx + 1, current_size + 1)
                chosen[v] = False
            
            backtrack(pos_idx + 1, current_size)
        
        backtrack(0, 0)
        return best[0]
    
    def generate_structured(seed_val):
        """Generate structured point configurations with many collinear triples."""
        random.seed(seed_val)
        patterns = []
        
        # Pattern 1: Grid-like with perturbations
        for offset in range(3):
            pts = []
            for i in range(4):
                for j in range(5):
                    x = -40 + i * 20 + random.randint(-3, 3)
                    y = -40 + j * 20 + random.randint(-3, 3)
                    x = max(-100, min(100, x))
                    y = max(-100, min(100, y))
                    pts.append((x, y))
            unique = list(dict.fromkeys(pts))
            while len(unique) < 20:
                unique.append((random.randint(-100, 100), random.randint(-100, 100)))
            patterns.append(unique[:20])
        
        # Pattern 2: Circle with perturbations
        for angle_off in [0, 0.1, 0.2]:
            pts = []
            for i in range(20):
                angle = 2 * math.pi * i / 20 + angle_off
                r = 50 + 8 * math.sin(i * 1.5)
                x = int(round(r * math.cos(angle)))
                y = int(round(r * math.sin(angle)))
                x = max(-100, min(100, x))
                y = max(-100, min(100, y))
                pts.append((x, y))
            unique = list(dict.fromkeys(pts))
            while len(unique) < 20:
                unique.append((random.randint(-100, 100), random.randint(-100, 100)))
            patterns.append(unique[:20])
        
        # Pattern 3: Clustered around a few centers
        for cx, cy in [(0,0), (20,20), (-20,-20)]:
            pts = []
            for i in range(20):
                x = cx + random.randint(-30, 30)
                y = cy + random.randint(-30, 30)
                x = max(-100, min(100, x))
                y = max(-100, min(100, y))
                pts.append((x, y))
            unique = list(dict.fromkeys(pts))
            while len(unique) < 20:
                unique.append((random.randint(-100, 100), random.randint(-100, 100)))
            patterns.append(unique[:20])
        
        # Pattern 4: Lines with points on them
        for trial in range(3):
            pts = []
            # Create 5 lines, 4 points each
            for line_idx in range(5):
                # Random line
                x1 = random.randint(-80, 80)
                y1 = random.randint(-80, 80)
                angle = random.uniform(0, math.pi)
                dx = int(round(40 * math.cos(angle)))
                dy = int(round(40 * math.sin(angle)))
                if dx == 0 and dy == 0:
                    dx, dy = 10, 0
                for t in range(4):
                    x = x1 + t * dx
                    y = y1 + t * dy
                    x = max(-100, min(100, x))
                    y = max(-100, min(100, y))
                    pts.append((x, y))
            unique = list(dict.fromkeys(pts))
            while len(unique) < 20:
                unique.append((random.randint(-100, 100), random.randint(-100, 100)))
            patterns.append(unique[:20])
        
        return patterns
    
    def random_points():
        pts = []
        used = set()
        while len(pts) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in used:
                used.add((x, y))
                pts.append((x, y))
        return pts
    
    def repair_four_collinear(points):
        """Try to fix 4-collinear by moving one point."""
        for _ in range(100):
            if not has_four_collinear(points):
                return True
            idx = random.randint(0, 19)
            old_p = points[idx]
            for _ in range(20):
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                new_p = (x, y)
                if new_p == old_p or new_p in points:
                    continue
                points[idx] = new_p
                if not has_four_collinear(points):
                    return True
                points[idx] = old_p
            points[idx] = old_p
        return not has_four_collinear(points)
    
    best_points = None
    best_score = 21
    best_triples = 0
    
    # Try structured initializations
    structured = generate_structured(42)
    for pts in structured:
        if time.time() - start_time > DEADLINE:
            break
        if not has_four_collinear(pts):
            score = calc_score(pts)
            triples = count_triples(pts)
            if score < best_score or (score == best_score and triples > best_triples):
                best_score = score
                best_triples = triples
                best_points = pts[:]
    
    # Also try random initializations
    for trial in range(10):
        if time.time() - start_time > DEADLINE:
            break
        pts = random_points()
        if not has_four_collinear(pts):
            score = calc_score(pts)
            triples = count_triples(pts)
            if score < best_score or (score == best_score and triples > best_triples):
                best_score = score
                best_triples = triples
                best_points = pts[:]
    
    # Simulated annealing refinement
    if best_points is not None:
        points = best_points[:]
        current_triples = best_triples
        T = 5.0
        cooling = 0.995
        step_count = 0
        
        while time.time() - start_time < DEADLINE:
            step_count += 1
            idx = random.randint(0, 19)
            old_p = points[idx]
            
            # Choose move type
            move_type = random.random()
            if move_type < 0.5:
                # Random move
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                new_p = (x, y)
            elif move_type < 0.8:
                # Small perturbation
                dx = random.randint(-5, 5)
                dy = random.randint(-5, 5)
                x = max(-100, min(100, old_p[0] + dx))
                y = max(-100, min(100, old_p[1] + dy))
                new_p = (x, y)
            else:
                # Try to align with existing points
                # Pick another point and try to be collinear with it and a third
                j = random.randint(0, 19)
                if j == idx:
                    j = (j + 1) % 20
                k = random.randint(0, 19)
                if k == idx or k == j:
                    k = (k + 1) % 20
                p1 = points[j]
                p2 = points[k]
                # Generate point on line through p1, p2
                t = random.randint(-2, 2)
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                x = max(-100, min(100, x))
                y = max(-100, min(100, y))
                new_p = (x, y)
            
            if new_p == old_p:
                continue
            if new_p in points:
                continue
            
            points[idx] = new_p
            
            if has_four_collinear(points):
                points[idx] = old_p
                continue
            
            new_triples = count_triples(points)
            
            # Acceptance criterion
            delta = new_triples - current_triples
            if delta > 0 or random.random() < math.exp(delta / T):
                current_triples = new_triples
                
                # Check exact score periodically or if triples improved
                if step_count % 50 == 0 or new_triples > best_triples:
                    exact_score = calc_score(points)
                    if exact_score < best_score:
                        best_score = exact_score
                        best_points = points[:]
                        best_triples = new_triples
                    elif exact_score == best_score and new_triples > best_triples:
                        best_triples = new_triples
                        best_points = points[:]
            else:
                points[idx] = old_p
            
            T *= cooling
            if T < 0.01:
                T = 0.01
    
    if best_points is None:
        # Fallback: generate random valid points
        points = random_points()
        repair_four_collinear(points)
        best_points = points
    
    return [[p[0], p[1]] for p in best_points]
