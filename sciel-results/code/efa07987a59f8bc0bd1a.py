"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np
from itertools import combinations

_start_time = time.time()
_DEADLINE = 110  # seconds, safety margin for 120s kill

def _count_four_lines_fast(points):
    """Count lines containing exactly 4 of the given points using vectorized ops."""
    n = len(points)
    pts = np.array(points, dtype=np.int64)
    count = 0
    seen = set()
    for i in range(n):
        for j in range(i+1, n):
            p1 = pts[i]
            p2 = pts[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            # Canonical direction
            if dx < 0 or (dx == 0 and dy < 0):
                dx, dy = -dx, -dy
            # Reduce by gcd
            a, b = abs(dx), abs(dy)
            while b:
                a, b = b, a % b
            g = a if a else 1
            if g > 1:
                dx //= g
                dy //= g
            # Count points on this line: (x - p1[0])*dy == (y - p1[1])*dx
            # Vectorized check
            diffs_x = pts[:, 0] - p1[0]
            diffs_y = pts[:, 1] - p1[1]
            on_line = (diffs_x * dy == diffs_y * dx)
            cnt = int(np.sum(on_line))
            if cnt == 4:
                # Get the indices for canonical key
                idxs = np.where(on_line)[0]
                key = tuple(sorted(idxs.tolist()))
                if key not in seen:
                    seen.add(key)
                    count += 1
    return count

def _is_valid_fast(points):
    """Check validity: distinct, bounds, no five collinear."""
    if len(points) != 16:
        return False
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return False
    if len(set(tuple(p) for p in points)) != 16:
        return False
    pts = np.array(points, dtype=np.int64)
    n = 16
    for i in range(n):
        for j in range(i+1, n):
            p1 = pts[i]
            p2 = pts[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            if dx < 0 or (dx == 0 and dy < 0):
                dx, dy = -dx, -dy
            a, b = abs(dx), abs(dy)
            while b:
                a, b = b, a % b
            g = a if a else 1
            if g > 1:
                dx //= g
                dy //= g
            diffs_x = pts[:, 0] - p1[0]
            diffs_y = pts[:, 1] - p1[1]
            on_line = (diffs_x * dy == diffs_y * dx)
            cnt = int(np.sum(on_line))
            if cnt >= 5:
                return False
    return True

def _try_move_fast(points, idx, dx, dy):
    """Try moving point idx by (dx, dy). Return new points or None if invalid."""
    new_pts = [list(p) for p in points]
    new_pts[idx] = [new_pts[idx][0] + dx, new_pts[idx][1] + dy]
    x, y = new_pts[idx]
    if abs(x) > 1000 or abs(y) > 1000:
        return None
    # Check distinct
    for j in range(16):
        if j != idx and new_pts[j] == new_pts[idx]:
            return None
    # Check no five collinear (fast vectorized)
    pts = np.array(new_pts, dtype=np.int64)
    n = 16
    for i in range(n):
        for j in range(i+1, n):
            p1 = pts[i]
            p2 = pts[j]
            ddx = p2[0] - p1[0]
            ddy = p2[1] - p1[1]
            if ddx < 0 or (ddx == 0 and ddy < 0):
                ddx, ddy = -ddx, -ddy
            a, b = abs(ddx), abs(ddy)
            while b:
                a, b = b, a % b
            g = a if a else 1
            if g > 1:
                ddx //= g
                ddy //= g
            diffs_x = pts[:, 0] - p1[0]
            diffs_y = pts[:, 1] - p1[1]
            on_line = (diffs_x * ddy == diffs_y * ddx)
            cnt = int(np.sum(on_line))
            if cnt >= 5:
                return None
    return new_pts

def _initial_grid():
    return [[x, y] for x in range(4) for y in range(4)]

def _local_search(points, score, max_iter=5000, deadline=None):
    """Local search: try small moves and accept improvements."""
    current = [list(p) for p in points]
    current_score = score
    best = [list(p) for p in current]
    best_score = current_score
    no_improve = 0
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        idx = random.randint(0, 15)
        # Try a few random small moves
        improved = False
        for _ in range(10):
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
            if dx == 0 and dy == 0:
                continue
            new_pts = _try_move_fast(current, idx, dx, dy)
            if new_pts is None:
                continue
            new_score = _count_four_lines_fast(new_pts)
            if new_score > current_score:
                current = new_pts
                current_score = new_score
                if current_score > best_score:
                    best = [list(p) for p in current]
                    best_score = current_score
                no_improve = 0
                improved = True
                break
        if not improved:
            no_improve += 1
            if no_improve > 200:
                break
    return best, best_score

def construct():
    global _start_time
    random.seed(42)
    
    # Start with 4x4 grid
    best = _initial_grid()
    best_score = _count_four_lines_fast(best)
    
    # Multiple restarts with different initial perturbations
    deadline = _start_time + _DEADLINE
    
    # Phase 1: Local search from grid
    best, best_score = _local_search(best, best_score, max_iter=10000, deadline=deadline)
    
    # Phase 2: Try different initial configurations
    # Try 4x4 grid with slight variations
    for trial in range(20):
        if time.time() > deadline:
            break
        # Create a perturbed grid
        pts = [[x, y] for x in range(4) for y in range(4)]
        # Randomly perturb a few points
        for _ in range(random.randint(1, 4)):
            idx = random.randint(0, 15)
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
            pts[idx] = [pts[idx][0] + dx, pts[idx][1] + dy]
        # Check validity
        if not _is_valid_fast(pts):
            continue
        score = _count_four_lines_fast(pts)
        if score > best_score:
            best, best_score = _local_search(pts, score, max_iter=5000, deadline=deadline)
    
    # Phase 3: Final polish - try all single-point moves with larger steps
    if time.time() < deadline - 10:
        current = [list(p) for p in best]
        current_score = best_score
        for _ in range(2000):
            if time.time() > deadline:
                break
            idx = random.randint(0, 15)
            for dx in range(-15, 16):
                for dy in range(-15, 16):
                    if dx == 0 and dy == 0:
                        continue
                    new_pts = _try_move_fast(current, idx, dx, dy)
                    if new_pts is None:
                        continue
                    new_score = _count_four_lines_fast(new_pts)
                    if new_score > current_score:
                        current = new_pts
                        current_score = new_score
                        if current_score > best_score:
                            best = [list(p) for p in current]
                            best_score = current_score
                        break
                else:
                    continue
                break
    
    return [[int(x), int(y)] for x, y in best]

# EVOLVE-BLOCK-END
