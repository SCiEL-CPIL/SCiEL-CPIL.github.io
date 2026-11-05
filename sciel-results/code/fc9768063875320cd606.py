"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations
import numpy as np

def _gcd2(x, y):
    x, y = abs(x), abs(y)
    while y:
        x, y = y, x % y
    return x

def _gcd3(a, b, c):
    return _gcd2(_gcd2(a, b), c)

def _line_key(x1, y1, x2, y2):
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    g = _gcd3(a, b, c)
    if g != 0:
        a //= g
        b //= g
        c //= g
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a = -a
        b = -b
        c = -c
    return (a, b, c)

def _count_exact_k(line_counts, k):
    """Count lines with exactly k points from pair counts."""
    count = 0
    for pair_count in line_counts.values():
        val = 1 + 8 * pair_count
        root = math.isqrt(val)
        if root * root != val:
            continue
        k_val = (1 + root) // 2
        if k_val == k:
            count += 1
    return count

def _has_five_or_more(line_counts):
    for pair_count in line_counts.values():
        val = 1 + 8 * pair_count
        root = math.isqrt(val)
        if root * root != val:
            continue
        k_val = (1 + root) // 2
        if k_val >= 5:
            return True
    return False

def _full_score(points):
    """Full O(n^2) score computation. Returns (score, is_valid, line_counts)."""
    n = len(points)
    if n < 2:
        return 0, True, {}
    
    line_counts = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            key = _line_key(x1, y1, x2, y2)
            line_counts[key] = line_counts.get(key, 0) + 1
    
    if _has_five_or_more(line_counts):
        return 0, False, line_counts
    
    four_lines = _count_exact_k(line_counts, 4)
    return four_lines, True, line_counts

def _incremental_score(points, line_counts, moved_idx, old_pos, new_pos):
    """
    Update line_counts after moving point moved_idx from old_pos to new_pos.
    Returns (new_score, is_valid, updated_line_counts).
    """
    n = len(points)
    old_x, old_y = old_pos
    new_x, new_y = new_pos
    
    # Remove old lines through moved_idx
    for j in range(n):
        if j == moved_idx:
            continue
        x2, y2 = points[j]
        key = _line_key(old_x, old_y, x2, y2)
        line_counts[key] = line_counts.get(key, 0) - 1
        if line_counts[key] <= 0:
            del line_counts[key]
    
    # Add new lines through moved_idx
    for j in range(n):
        if j == moved_idx:
            continue
        x2, y2 = points[j]
        key = _line_key(new_x, new_y, x2, y2)
        line_counts[key] = line_counts.get(key, 0) + 1
    
    # Check for 5+ collinear
    if _has_five_or_more(line_counts):
        return 0, False, line_counts
    
    four_lines = _count_exact_k(line_counts, 4)
    return four_lines, True, line_counts

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Start with 4x4 grid
    base_points = [[i, j] for i in range(4) for j in range(4)]
    
    best_points = [p[:] for p in base_points]
    best_score, _, best_line_counts = _full_score(best_points)
    
    current_points = [p[:] for p in best_points]
    current_score = best_score
    current_line_counts = dict(best_line_counts)
    
    n = 16
    temperature = 10.0
    cooling = 0.9995
    plateau_prob = 0.15
    
    while time.time() < deadline:
        # Pick a random point to move
        i = random.randint(0, n - 1)
        old_x, old_y = current_points[i]
        
        # Generate new position
        r = random.random()
        if r < 0.5:
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
        elif r < 0.8:
            dx = random.randint(-15, 15)
            dy = random.randint(-15, 15)
        else:
            dx = random.randint(-50, 50)
            dy = random.randint(-50, 50)
            
        new_x = old_x + dx
        new_y = old_y + dy
        
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
        
        # Check distinctness
        dup = False
        for j in range(n):
            if i != j and current_points[j][0] == new_x and current_points[j][1] == new_y:
                dup = True
                break
        if dup:
            continue
        
        # Use incremental scoring
        old_pos = [old_x, old_y]
        new_pos = [new_x, new_y]
        current_points[i] = new_pos
        new_score, valid, current_line_counts = _incremental_score(
            current_points, current_line_counts, i, old_pos, new_pos
        )
        
        if not valid:
            current_points[i] = old_pos
            # Rebuild line_counts from scratch if we got into invalid state
            current_score, _, current_line_counts = _full_score(current_points)
            continue
        
        # Simulated annealing acceptance with plateau moves
        delta = new_score - current_score
        accept = False
        if delta > 0:
            accept = True
        elif delta == 0 and random.random() < plateau_prob:
            accept = True
        elif temperature > 0.1:
            try:
                accept = random.random() < math.exp(delta / temperature)
            except OverflowError:
                accept = False
        
        if accept:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
                best_line_counts = dict(current_line_counts)
        else:
            current_points[i] = old_pos
            # Revert line_counts
            current_score, _, current_line_counts = _full_score(current_points)
        
        temperature *= cooling
        
        # Periodic restart if stuck
        if temperature < 0.1:
            temperature = 5.0
            for _ in range(15):
                current_points = [p[:] for p in best_points]
                current_score = best_score
                current_line_counts = dict(best_line_counts)
                for _ in range(8):
                    idx = random.randint(0, n - 1)
                    ox, oy = current_points[idx]
                    nx = ox + random.randint(-6, 6)
                    ny = oy + random.randint(-6, 6)
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        continue
                    dup = False
                    for j in range(n):
                        if idx != j and current_points[j][0] == nx and current_points[j][1] == ny:
                            dup = True
                            break
                    if dup:
                        continue
                    old_pos = [ox, oy]
                    new_pos = [nx, ny]
                    current_points[idx] = new_pos
                    s, v, current_line_counts = _incremental_score(
                        current_points, current_line_counts, idx, old_pos, new_pos
                    )
                    if v and s >= current_score:
                        current_score = s
                        if s > best_score:
                            best_score = s
                            best_points = [p[:] for p in current_points]
                            best_line_counts = dict(current_line_counts)
                    else:
                        current_points[idx] = old_pos
                        current_score, _, current_line_counts = _full_score(current_points)
        
        # If we hit the theoretical upper bound, stop
        if best_score >= 15:
            break
    
    # Final validation
    final_score, valid, _ = _full_score(best_points)
    if not valid or final_score == 0:
        best_points = [[i, j] for i in range(4) for j in range(4)]
    
    return best_points
# EVOLVE-BLOCK-END
