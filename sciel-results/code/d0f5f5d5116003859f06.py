"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from collections import defaultdict

def normalize_line(p1, p2):
    """Return canonical line key from two points."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    if dx == 0:
        return (0, 1, x1)
    if dy == 0:
        return (1, 0, y1)
    g = gcd(abs(dx), abs(dy))
    dx_n = dx // g
    dy_n = dy // g
    if dx_n < 0:
        dx_n = -dx_n
        dy_n = -dy_n
    c = dy_n * x1 - dx_n * y1
    return (dx_n, dy_n, c)

def count_4_lines_fast(points):
    """Fast count of lines with exactly 4 points. Returns -1 if any 5 collinear."""
    n = len(points)
    if n != 16:
        return 0
    
    line_mask = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = normalize_line(points[i], points[j])
            if key is None:
                continue
            mask = (1 << i) | (1 << j)
            if key in line_mask:
                line_mask[key] |= mask
            else:
                line_mask[key] = mask
    
    count4 = 0
    for key, mask in line_mask.items():
        popcount = bin(mask).count('1')
        if popcount >= 5:
            return -1
        if popcount == 4:
            count4 += 1
    return count4

def compute_line_masks(points):
    """Compute line masks and return (line_mask_dict, point_lines dict)."""
    n = len(points)
    line_mask = {}
    point_lines = [[] for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            key = normalize_line(points[i], points[j])
            if key is None:
                continue
            mask = (1 << i) | (1 << j)
            if key in line_mask:
                line_mask[key] |= mask
            else:
                line_mask[key] = mask
            point_lines[i].append(key)
            point_lines[j].append(key)
    
    for i in range(n):
        point_lines[i] = list(set(point_lines[i]))
    
    return line_mask, point_lines

def get_score_from_masks(line_mask):
    """Get score from line masks."""
    count4 = 0
    for key, mask in line_mask.items():
        popcount = bin(mask).count('1')
        if popcount >= 5:
            return -1
        if popcount == 4:
            count4 += 1
    return count4

def local_search_incremental(current, line_mask, point_lines, current_score, best_score, best_points, deadline):
    """Perform local search with incremental updates. Returns (current, line_mask, point_lines, current_score, best_score, best_points)."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(16):
            if time.time() > deadline:
                return current, line_mask, point_lines, current_score, best_score, best_points
            old = current[idx][:]
            old_point_lines = list(point_lines[idx])
            
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = old[0] + dx, old[1] + dy
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        continue
                    current[idx] = [nx, ny]
                    
                    pts_set = set(tuple(p) for p in current)
                    if len(pts_set) < 16:
                        current[idx] = old
                        continue
                    
                    # Remove old lines involving idx
                    for key in old_point_lines:
                        if key in line_mask:
                            line_mask[key] &= ~(1 << idx)
                            if line_mask[key] == 0:
                                del line_mask[key]
                    
                    # Add new lines involving idx
                    new_point_lines = []
                    for j in range(16):
                        if j == idx:
                            continue
                        key = normalize_line(current[idx], current[j])
                        if key is None:
                            continue
                        new_point_lines.append(key)
                        if key in line_mask:
                            line_mask[key] |= (1 << idx)
                        else:
                            line_mask[key] = (1 << idx)
                    
                    point_lines[idx] = list(set(new_point_lines))
                    
                    s = get_score_from_masks(line_mask)
                    if s > current_score:
                        current_score = s
                        improved = True
                        if s > best_score:
                            best_score = s
                            best_points = [p[:] for p in current]
                    else:
                        # Rollback
                        current[idx] = old
                        for key in point_lines[idx]:
                            if key in line_mask:
                                line_mask[key] &= ~(1 << idx)
                                if line_mask[key] == 0:
                                    del line_mask[key]
                        for key in old_point_lines:
                            if key in line_mask:
                                line_mask[key] |= (1 << idx)
                            else:
                                line_mask[key] = (1 << idx)
                        point_lines[idx] = old_point_lines
    
    return current, line_mask, point_lines, current_score, best_score, best_points

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with the 4x4 grid
    grid = [[x, y] for x in range(4) for y in range(4)]
    best_points = [p[:] for p in grid]
    best_score = count_4_lines_fast(best_points)
    
    random.seed(12345)
    
    # Phase 1: Local search from grid with incremental updates
    current = [p[:] for p in grid]
    line_mask, point_lines = compute_line_masks(current)
    current_score = get_score_from_masks(line_mask)
    
    current, line_mask, point_lines, current_score, best_score, best_points = local_search_incremental(
        current, line_mask, point_lines, current_score, best_score, best_points, deadline
    )
    
    # Phase 2: Iterated local search with random restarts
    for restart in range(200):
        if time.time() > deadline:
            return best_points
        
        if restart % 3 == 0:
            current = [p[:] for p in best_points]
            for _ in range(random.randint(1, 3)):
                idx = random.randint(0, 15)
                current[idx] = [current[idx][0] + random.randint(-4, 4),
                               current[idx][1] + random.randint(-4, 4)]
        elif restart % 3 == 1:
            current = []
            for x in range(4):
                for y in range(4):
                    current.append([x + random.randint(-3, 3), y + random.randint(-3, 3)])
        else:
            current = [[random.randint(-15, 15), random.randint(-15, 15)] for _ in range(16)]
        
        for _ in range(200):
            if len(set(tuple(p) for p in current)) == 16:
                break
            idx = random.randint(0, 15)
            current[idx] = [random.randint(-15, 15), random.randint(-15, 15)]
        
        if len(set(tuple(p) for p in current)) != 16:
            continue
        
        line_mask, point_lines = compute_line_masks(current)
        current_score = get_score_from_masks(line_mask)
        if current_score < 0:
            continue
        
        if current_score > best_score:
            best_score = current_score
            best_points = [p[:] for p in current]
        
        current, line_mask, point_lines, current_score, best_score, best_points = local_search_incremental(
            current, line_mask, point_lines, current_score, best_score, best_points, deadline
        )
    
    return best_points
# EVOLVE-BLOCK-END
