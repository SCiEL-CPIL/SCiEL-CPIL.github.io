import time
import random
import math
import numpy as np
from collections import Counter, defaultdict

def _line_key(dx, dy, x0, y0):
    """Compute normalized line key (direction, offset)."""
    if dx == 0 and dy == 0:
        return (0, 0, 0)
    ax, ay = abs(dx), abs(dy)
    g = math.gcd(ax, ay)
    if g == 0:
        g = 1
    dxn = dx // g
    dyn = dy // g
    if dxn < 0 or (dxn == 0 and dyn < 0):
        dxn = -dxn
        dyn = -dyn
    c = dyn * x0 - dxn * y0
    return (dxn, dyn, c)

def _build_line_index(points):
    """Build line index for 16 points. Returns (line_to_points, score, max_count, valid)."""
    n = len(points)
    if n != 16:
        return None, 0, 0, False
    xs = np.array([p[0] for p in points], dtype=np.int64)
    ys = np.array([p[1] for p in points], dtype=np.int64)
    if np.any(np.abs(xs) > 1000) or np.any(np.abs(ys) > 1000):
        return None, 0, 0, False
    pts = np.column_stack((xs, ys))
    if len(set(map(tuple, pts.tolist()))) != 16:
        return None, 0, 0, False
    
    line_to_points = defaultdict(set)
    for i in range(16):
        for j in range(i+1, 16):
            dx = xs[j] - xs[i]
            dy = ys[j] - ys[i]
            key = _line_key(int(dx), int(dy), int(xs[i]), int(ys[i]))
            line_to_points[key].add(i)
            line_to_points[key].add(j)
    
    score = 0
    max_count = 0
    for key, pt_set in line_to_points.items():
        cnt = len(pt_set)
        if cnt > max_count:
            max_count = cnt
        if cnt == 4:
            score += 1
    
    if max_count >= 5:
        return None, 0, 0, False
    
    return line_to_points, score, max_count, True

def _evaluate_from_index(line_to_points):
    """Compute score from existing line index."""
    score = 0
    max_count = 0
    for key, pt_set in line_to_points.items():
        cnt = len(pt_set)
        if cnt > max_count:
            max_count = cnt
        if cnt == 4:
            score += 1
    if max_count >= 5:
        return 0, False
    return score, True

def _incremental_update(points, old_idx, old_pt, new_pt):
    """Incrementally update line index when moving point old_idx from old_pt to new_pt.
    Returns (new_line_to_points, new_score, new_max_count, valid)."""
    xs = np.array([p[0] for p in points], dtype=np.int64)
    ys = np.array([p[1] for p in points], dtype=np.int64)
    
    # Check bounds and uniqueness
    if abs(new_pt[0]) > 1000 or abs(new_pt[1]) > 1000:
        return None, 0, 0, False
    for i in range(16):
        if i != old_idx and points[i] == new_pt:
            return None, 0, 0, False
    
    # Build line index from scratch (simpler and still fast enough)
    return _build_line_index(points)

def _evaluate_fast(points):
    """Evaluate 16 [x,y] points. Returns (is_valid, score)."""
    line_to_points, score, max_count, valid = _build_line_index(points)
    if not valid:
        return False, 0
    return True, score

def _affine_seed():
    """Generate an affinely transformed 4x4 grid."""
    base = np.array([[x, y] for y in range(4) for x in range(4)], dtype=np.int64)
    for _ in range(300):
        A = np.array([[random.randint(-4, 4), random.randint(-4, 4)],
                       [random.randint(-4, 4), random.randint(-4, 4)]], dtype=np.int64)
        det = int(A[0,0]*A[1,1] - A[0,1]*A[1,0])
        if det == 0:
            continue
        t = A @ base.T
        if np.any(np.abs(t) > 900):
            continue
        pts = t.T.tolist()
        is_valid, score = _evaluate_fast(pts)
        if is_valid:
            return pts, score
    return None, 0

def _plateau_move(points, idx):
    """Make a 'plateau' move: move point to a position that preserves most line structure.
    This helps escape local optima by exploring the plateau of equivalent configurations."""
    old = points[idx]
    # Try small perturbations that might preserve line structure
    for _ in range(5):
        dx = random.randint(-2, 2)
        dy = random.randint(-2, 2)
        if dx == 0 and dy == 0:
            continue
        new_pt = [old[0] + dx, old[1] + dy]
        if abs(new_pt[0]) > 1000 or abs(new_pt[1]) > 1000:
            continue
        # Check uniqueness
        valid = True
        for i in range(16):
            if i != idx and points[i] == new_pt:
                valid = False
                break
        if valid:
            return new_pt
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    best_points = None
    best_score = -1

    # Try diverse affine seeds
    for _ in range(50):
        if time.time() > deadline:
            break
        pts, score = _affine_seed()
        if pts is not None and score > best_score:
            best_score = score
            best_points = [p[:] for p in pts]
        if best_score >= 14:
            break

    # Fallback
    if best_points is None:
        best_points = [[x, y] for y in range(4) for x in range(4)]
        is_valid, best_score = _evaluate_fast(best_points)

    current_points = [p[:] for p in best_points]
    current_score = best_score

    # Simulated annealing local search with incremental evaluation
    T = 3.0
    T_min = 0.1
    cool_rate = 0.997
    no_improve_count = 0
    max_no_improve = 2000

    while time.time() < deadline:
        idx = random.randint(0, 15)
        old = current_points[idx]

        r = random.random()
        if r < 0.15:
            # Plateau move: try to move within the plateau of equivalent configurations
            new_pt = _plateau_move(current_points, idx)
            if new_pt is None:
                nx = old[0] + random.randint(-2, 2)
                ny = old[1] + random.randint(-2, 2)
            else:
                nx, ny = new_pt
        elif r < 0.40:
            nx = old[0] + random.randint(-3, 3)
            ny = old[1] + random.randint(-3, 3)
        elif r < 0.65:
            nx = old[0] + random.randint(-10, 10)
            ny = old[1] + random.randint(-10, 10)
        elif r < 0.85:
            nx = old[0] + random.randint(-30, 30)
            ny = old[1] + random.randint(-30, 30)
        else:
            nx = random.randint(-1000, 1000)
            ny = random.randint(-1000, 1000)

        if abs(nx) > 1000 or abs(ny) > 1000:
            continue

        current_points[idx] = [nx, ny]
        is_valid, new_score = _evaluate_fast(current_points)

        if not is_valid:
            current_points[idx] = old
            continue

        delta = new_score - current_score
        if delta >= 0:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Accept with probability exp(delta/T)
            if random.random() < np.exp(delta / T):
                current_score = new_score
                no_improve_count += 1
            else:
                current_points[idx] = old

        T = max(T_min, T * cool_rate)

        # Restart from best with perturbation if stuck
        if (no_improve_count > max_no_improve or random.random() < 0.003) and time.time() < deadline - 10:
            current_points = [p[:] for p in best_points]
            current_score = best_score
            no_improve_count = 0
            # Perturb a few points more aggressively
            for _ in range(random.randint(2, 5)):
                pi = random.randint(0, 15)
                current_points[pi] = [
                    best_points[pi][0] + random.randint(-30, 30),
                    best_points[pi][1] + random.randint(-30, 30)
                ]
                if abs(current_points[pi][0]) > 1000 or abs(current_points[pi][1]) > 1000:
                    current_points[pi] = [random.randint(-1000, 1000), random.randint(-1000, 1000)]
            is_valid, current_score = _evaluate_fast(current_points)
            if not is_valid:
                current_points = [p[:] for p in best_points]
                current_score = best_score

    return best_points
