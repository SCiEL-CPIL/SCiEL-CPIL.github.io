import time
import random
from math import gcd

def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 3:
        return 0
    line_counts = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i+1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx_n = dx // g
            dy_n = dy // g
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
            if dx_n == 0:
                key = (0, 0, x1)
            else:
                c = dy_n * x1 - dx_n * y1
                key = (dx_n, dy_n, c)
            line_counts[key] = line_counts.get(key, 0) + 1
    ordinary = 0
    for key, cnt in line_counts.items():
        if cnt == 1:
            ordinary += 1
    return ordinary

def _not_all_collinear(points):
    if len(points) < 3:
        return True
    x1, y1 = points[0]
    x2, y2 = points[1]
    dx = x2 - x1
    dy = y2 - y1
    for i in range(2, len(points)):
        x, y = points[i]
        if dx * (y - y1) != dy * (x - x1):
            return True
    return False

def _generate_structured_candidates():
    """Generate diverse structured configurations designed to minimize ordinary lines."""
    candidates = []
    
    # 3x4 grid
    candidates.append([[x, y] for x in range(4) for y in range(3)])
    
    # 4x3 grid
    candidates.append([[x, y] for x in range(3) for y in range(4)])
    
    # Two rows of 6
    candidates.append([[x, 0] for x in range(6)] + [[x, 1] for x in range(6)])
    
    # Three rows of 4
    candidates.append([[x, y] for x in range(4) for y in range(3)])
    
    # Hexagonal arrangement
    hex_pts = [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1],
               [1, -1], [-1, 1], [2, 0], [-2, 0], [0, 2]]
    candidates.append(hex_pts)
    
    # 5x5 grid subset - keep points on diagonals and center lines
    base_5x5 = [[x, y] for x in range(5) for y in range(5)]
    special = []
    for x in range(5):
        for y in range(5):
            if x == y or x + y == 4 or x == 2 or y == 2:
                special.append([x, y])
    if len(special) >= 12:
        candidates.append(special[:12])
    elif len(special) > 0:
        remaining = [p for p in base_5x5 if p not in special]
        while len(special) < 12 and remaining:
            special.append(remaining.pop(0))
        if len(special) == 12:
            candidates.append(special)
    
    # Parabolic arrangement
    parab = [[x, x*x] for x in range(-2, 4)]
    extra = [[-2, -1], [-1, -1], [0, -1]]
    candidates.append(parab + extra)
    
    # 2x6 grid
    candidates.append([[x, y] for x in range(6) for y in range(2)])
    
    # Diagonal-shifted grid
    diag_pts = []
    for row in range(3):
        offset = row * 2
        for col in range(4):
            diag_pts.append([col + offset, row])
    candidates.append(diag_pts)
    
    # 6x2 with different offset
    candidates.append([[x, y] for x in range(2) for y in range(6)])
    
    # Square ring: 3x3 outer minus corners, plus center
    ring = [[x, y] for x in range(-1, 2) for y in range(-1, 2) if not (x == 0 and y == 0)]
    ring.append([0, 0])
    # Add 3 more to make 12
    ring += [[2, 0], [-2, 0], [0, 2]]
    candidates.append(ring)
    
    # Triangular arrangement
    tri = [[0, 0], [1, 0], [2, 0], [3, 0],
           [0, 1], [1, 1], [2, 1],
           [0, 2], [1, 2],
           [0, 3],
           [1, 3], [2, 3]]
    candidates.append(tri)
    
    # 4x3 with perturbation
    base = [[x, y] for x in range(4) for y in range(3)]
    perturbed = [p[:] for p in base]
    perturbed[0] = [0, 0]
    perturbed[1] = [1, 0]
    perturbed[2] = [2, 0]
    perturbed[3] = [3, 0]
    perturbed[4] = [0, 1]
    perturbed[5] = [1, 1]
    perturbed[6] = [2, 1]
    perturbed[7] = [3, 1]
    perturbed[8] = [0, 2]
    perturbed[9] = [1, 2]
    perturbed[10] = [2, 2]
    perturbed[11] = [3, 2]
    candidates.append(perturbed)
    
    # Concentric structure
    conc = [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1],
            [2, 0], [-2, 0], [0, 2], [0, -2],
            [1, 1], [-1, -1], [1, -1]]
    candidates.append(conc)
    
    return candidates

def _local_search(points, deadline):
    """Local search with point moves and pair swaps."""
    current = [p[:] for p in points]
    current_score = _count_ordinary(current)
    best = [p[:] for p in current]
    best_score = current_score
    
    while time.time() < deadline:
        improved = False
        
        # Try moving each point to nearby positions
        for idx in range(12):
            if time.time() > deadline:
                break
            old_x, old_y = current[idx]
            for dx in range(-4, 5):
                if time.time() > deadline:
                    break
                for dy in range(-4, 5):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = old_x + dx
                    new_y = old_y + dy
                    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                        continue
                    is_dup = False
                    for j in range(12):
                        if j != idx and current[j][0] == new_x and current[j][1] == new_y:
                            is_dup = True
                            break
                    if is_dup:
                        continue
                    current[idx] = [new_x, new_y]
                    if not _not_all_collinear(current):
                        current[idx] = [old_x, old_y]
                        continue
                    new_score = _count_ordinary(current)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        if new_score < best_score:
                            best_score = new_score
                            best = [p[:] for p in current]
                        break
                    else:
                        current[idx] = [old_x, old_y]
                else:
                    continue
                break
            if improved:
                break
        
        # Try pair moves if no improvement
        if not improved and time.time() < deadline:
            for i in range(12):
                if time.time() > deadline:
                    break
                for j in range(i+1, 12):
                    if time.time() > deadline:
                        break
                    x1, y1 = current[i]
                    x2, y2 = current[j]
                    for dx1 in range(-3, 4):
                        for dy1 in range(-3, 4):
                            if dx1 == 0 and dy1 == 0:
                                continue
                            nx1, ny1 = x1 + dx1, y1 + dy1
                            if abs(nx1) > 10**9 or abs(ny1) > 10**9:
                                continue
                            for dx2 in range(-3, 4):
                                for dy2 in range(-3, 4):
                                    if dx2 == 0 and dy2 == 0:
                                        continue
                                    nx2, ny2 = x2 + dx2, y2 + dy2
                                    if abs(nx2) > 10**9 or abs(ny2) > 10**9:
                                        continue
                                    # Check distinctness
                                    is_dup = False
                                    for k in range(12):
                                        if k != i and k != j:
                                            if (current[k][0] == nx1 and current[k][1] == ny1) or \
                                               (current[k][0] == nx2 and current[k][1] == ny2):
                                                is_dup = True
                                                break
                                        if k == i and (nx2 == current[i][0] and ny2 == current[i][1]):
                                            is_dup = True
                                            break
                                        if k == j and (nx1 == current[j][0] and ny1 == current[j][1]):
                                            is_dup = True
                                            break
                                    if is_dup:
                                        continue
                                    current[i] = [nx1, ny1]
                                    current[j] = [nx2, ny2]
                                    if not _not_all_collinear(current):
                                        current[i] = [x1, y1]
                                        current[j] = [x2, y2]
                                        continue
                                    new_score = _count_ordinary(current)
                                    if new_score < current_score:
                                        current_score = new_score
                                        improved = True
                                        if new_score < best_score:
                                            best_score = new_score
                                            best = [p[:] for p in current]
                                        break
                                    else:
                                        current[i] = [x1, y1]
                                        current[j] = [x2, y2]
                                else:
                                    continue
                                break
                            else:
                                continue
                            break
                        else:
                            continue
                        break
                    else:
                        continue
                    break
                else:
                    continue
                break
            if improved:
                continue
        
        # Random perturbation if stuck
        if not improved:
            idx = random.randint(0, 11)
            old_x, old_y = current[idx]
            found = False
            for _ in range(100):
                new_x = old_x + random.randint(-8, 8)
                new_y = old_y + random.randint(-8, 8)
                if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                    continue
                is_dup = False
                for j in range(12):
                    if j != idx and current[j][0] == new_x and current[j][1] == new_y:
                        is_dup = True
                        break
                if is_dup:
                    continue
                current[idx] = [new_x, new_y]
                if not _not_all_collinear(current):
                    current[idx] = [old_x, old_y]
                    continue
                new_score = _count_ordinary(current)
                if new_score <= current_score:
                    current_score = new_score
                    improved = True
                    found = True
                    if new_score < best_score:
                        best_score = new_score
                        best = [p[:] for p in current]
                    break
                else:
                    current[idx] = [old_x, old_y]
            if not found:
                break
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    random.seed(42)
    
    # Generate diverse structured candidates
    candidates = _generate_structured_candidates()
    
    # Add random configurations
    for _ in range(30):
        pts = []
        while len(pts) < 12:
            x = random.randint(-10, 10)
            y = random.randint(-10, 10)
            if [x, y] not in pts:
                pts.append([x, y])
        if _not_all_collinear(pts):
            candidates.append(pts)
    
    # Evaluate all candidates
    best_config = None
    best_score = float('inf')
    
    for cfg in candidates:
        if time.time() > deadline:
            break
        score = _count_ordinary(cfg)
        if score < best_score:
            best_score = score
            best_config = [p[:] for p in cfg]
    
    # Local search from best initial
    if best_config and time.time() < deadline:
        best_config, best_score = _local_search(best_config, deadline)
    
    # Try local search from other promising configs if time permits
    for cfg in candidates:
        if time.time() > deadline:
            break
        score = _count_ordinary(cfg)
        if score <= best_score + 3:
            cfg_best, cfg_score = _local_search(cfg, deadline)
            if cfg_score < best_score:
                best_score = cfg_score
                best_config = cfg_best
    
    return best_config
