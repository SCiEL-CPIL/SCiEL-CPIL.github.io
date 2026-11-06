"""Rewritten program with faster scoring and stronger search."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

def _count_lines_exactly_4(points):
    """Count lines containing exactly 4 of the given points using vectorized ops."""
    n = len(points)
    pts = np.array(points, dtype=np.int64)
    count = 0
    lines_seen = set()
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            nx = dx // g
            ny = dy // g
            
            if nx < 0 or (nx == 0 and ny < 0):
                nx = -nx
                ny = -ny
            
            ref_x, ref_y = x1, y1
            if (x2, y2) < (x1, y1):
                ref_x, ref_y = x2, y2
            
            line_key = (nx, ny, ref_x, ref_y)
            if line_key in lines_seen:
                continue
            lines_seen.add(line_key)
            
            # Vectorized cross product check
            px = pts[:, 0] - ref_x
            py = pts[:, 1] - ref_y
            cross = px * ny - py * nx
            cnt = np.count_nonzero(cross == 0)
            
            if cnt == 4:
                count += 1
    
    return count

def _max_collinear(points):
    """Return the maximum number of collinear points."""
    n = len(points)
    pts = np.array(points, dtype=np.int64)
    max_cnt = 0
    lines_seen = set()
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            nx = dx // g
            ny = dy // g
            
            if nx < 0 or (nx == 0 and ny < 0):
                nx = -nx
                ny = -ny
            
            ref_x, ref_y = x1, y1
            if (x2, y2) < (x1, y1):
                ref_x, ref_y = x2, y2
            
            line_key = (nx, ny, ref_x, ref_y)
            if line_key in lines_seen:
                continue
            lines_seen.add(line_key)
            
            px = pts[:, 0] - ref_x
            py = pts[:, 1] - ref_y
            cross = px * ny - py * nx
            cnt = np.count_nonzero(cross == 0)
            if cnt > max_cnt:
                max_cnt = cnt
                if max_cnt >= 5:
                    return max_cnt
    
    return max_cnt

def _valid(points):
    """Check validity: 16 distinct integer points, |x|,|y|<=1000, no 5 collinear."""
    if len(points) != 16:
        return False
    seen = set()
    for p in points:
        x, y = p
        if abs(x) > 1000 or abs(y) > 1000:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    if _max_collinear(points) >= 5:
        return False
    return True

def _score(points):
    """Score: number of lines with exactly 4 points. Return -1 if invalid."""
    if not _valid(points):
        return -1
    return _count_lines_exactly_4(points)

def _hill_climb(points, deadline, rng):
    """Hill-climb: try single-point moves and 2-point swaps."""
    best = [p[:] for p in points]
    best_score = _score(best)
    if best_score < 0:
        return best, best_score
    
    no_improve = 0
    while time.time() < deadline:
        improved = False
        # Try single-point moves
        for _ in range(200):
            idx = rng.randint(0, 15)
            old_x, old_y = best[idx]
            
            if rng.random() < 0.4:
                # Local move
                new_x = old_x + rng.randint(-5, 5)
                new_y = old_y + rng.randint(-5, 5)
            elif rng.random() < 0.7:
                # Medium move
                new_x = rng.randint(-200, 200)
                new_y = rng.randint(-200, 200)
            else:
                # Global move
                new_x = rng.randint(-1000, 1000)
                new_y = rng.randint(-1000, 1000)
            
            if abs(new_x) > 1000 or abs(new_y) > 1000:
                continue
            if any(best[k][0] == new_x and best[k][1] == new_y for k in range(16) if k != idx):
                continue
            
            candidate = [p[:] for p in best]
            candidate[idx] = [new_x, new_y]
            s = _score(candidate)
            if s > best_score:
                best_score = s
                best = candidate
                improved = True
                no_improve = 0
        
        # Try 2-point swaps
        for _ in range(100):
            i, j = rng.sample(range(16), 2)
            candidate = [p[:] for p in best]
            candidate[i], candidate[j] = candidate[j][:], candidate[i][:]
            s = _score(candidate)
            if s > best_score:
                best_score = s
                best = candidate
                improved = True
                no_improve = 0
        
        if not improved:
            no_improve += 1
            if no_improve > 3:
                break
    
    return best, best_score

def _generate_initial_configs():
    """Generate diverse initial configurations."""
    configs = []
    
    # 4x4 grid
    configs.append([[x, y] for x in range(4) for y in range(4)])
    
    # 4x4 grid shifted/scaled variants
    for sx in range(-2, 3):
        for sy in range(-2, 3):
            configs.append([[x + sx, y + sy] for x in range(4) for y in range(4)])
    
    # Affine transforms of 4x4 grid
    # y = x + c (diagonal grid)
    for c in range(-3, 4):
        configs.append([[x, x + c] for x in range(4)] + [[x, x + c + 4] for x in range(4)])
    
    # Parabolic: (x, x^2) for x in some set
    for offset in range(-2, 3):
        pts = [[x, x * x + offset] for x in range(-3, 5)]
        if len(pts) == 16 and all(abs(p[0]) <= 1000 and abs(p[1]) <= 1000 for p in pts):
            configs.append(pts)
    
    # 5x5 grid minus one point (pick various points to remove)
    grid5 = [[x, y] for x in range(5) for y in range(5)]
    for remove_idx in [0, 4, 10, 15, 20, 24, 7, 12, 17, 22]:
        pts = [p for i, p in enumerate(grid5) if i != remove_idx]
        if len(pts) == 16:
            configs.append(pts)
    
    # Random configurations
    rng = random.Random(12345)
    for _ in range(20):
        pts = []
        attempts = 0
        while len(pts) < 16 and attempts < 1000:
            x = rng.randint(-50, 50)
            y = rng.randint(-50, 50)
            if (x, y) not in pts:
                pts.append([x, y])
            attempts += 1
        if len(pts) == 16:
            configs.append(pts)
    
    # Structured: two interleaved grids
    configs.append([[x, y] for x in range(0, 8, 2) for y in range(0, 8, 2)] +
                   [[x + 1, y + 1] for x in range(0, 8, 2) for y in range(0, 8, 2)])
    
    # 4x4 with one point perturbed
    base = [[x, y] for x in range(4) for y in range(4)]
    for idx in range(16):
        for dx in [-2, -1, 1, 2]:
            for dy in [-2, -1, 1, 2]:
                pts = [p[:] for p in base]
                pts[idx] = [pts[idx][0] + dx, pts[idx][1] + dy]
                if all(abs(p[0]) <= 1000 and abs(p[1]) <= 1000 for p in pts):
                    if len(set(tuple(p) for p in pts)) == 16:
                        configs.append(pts)
    
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    rng = random.Random(42)
    
    best_points = None
    best_score = -1
    
    # Generate and evaluate initial configurations
    configs = _generate_initial_configs()
    
    # Evaluate all configs
    for cfg in configs:
        if time.time() > deadline:
            break
        s = _score(cfg)
        if s > best_score:
            best_score = s
            best_points = [p[:] for p in cfg]
    
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(4)]
        best_score = _score(best_points)
    
    # Hill-climb from best initial config
    best_points, best_score = _hill_climb(best_points, deadline, rng)
    
    # Try hill-climbing from other good initial configs
    for cfg in configs[:30]:
        if time.time() > deadline - 20:
            break
        s = _score(cfg)
        if s >= best_score - 1:  # Close to best
            candidate, cand_score = _hill_climb(cfg, deadline, rng)
            if cand_score > best_score:
                best_score = cand_score
                best_points = candidate
    
    # Additional random restarts if time permits
    while time.time() < deadline - 10:
        pts = []
        attempts = 0
        while len(pts) < 16 and attempts < 500:
            x = rng.randint(-100, 100)
            y = rng.randint(-100, 100)
            if (x, y) not in pts:
                pts.append([x, y])
            attempts += 1
        if len(pts) == 16:
            candidate, cand_score = _hill_climb(pts, deadline, rng)
            if cand_score > best_score:
                best_score = cand_score
                best_points = candidate
    
    return [[p[0], p[1]] for p in best_points]
# EVOLVE-BLOCK-END
