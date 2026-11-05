import time
import random
import numpy as np
from collections import Counter

# Precompute pair indices (120 pairs for 16 points)
_iu, _ju = np.triu_indices(16, k=1)

def _evaluate_fast(points):
    """Fast evaluator using vectorized numpy operations.
    Returns (is_valid, score) where score = number of lines with exactly 4 points.
    """
    xs = np.array([p[0] for p in points], dtype=np.int64)
    ys = np.array([p[1] for p in points], dtype=np.int64)
    if np.any(np.abs(xs) > 1000) or np.any(np.abs(ys) > 1000):
        return False, 0
    # distinctness
    pts = np.column_stack((xs, ys))
    if len(set(map(tuple, pts.tolist()))) != 16:
        return False, 0
    # Compute line keys for all 120 pairs
    dx = xs[_ju] - xs[_iu]
    dy = ys[_ju] - ys[_iu]
    g = np.gcd(np.abs(dx), np.abs(dy))
    g_safe = np.where(g == 0, 1, g)
    dxn = dx // g_safe
    dyn = dy // g_safe
    # Canonicalize sign
    neg = (dxn < 0) | ((dxn == 0) & (dyn < 0))
    dxn = np.where(neg, -dxn, dxn)
    dyn = np.where(neg, -dyn, dyn)
    # c = dyn*x1 - dxn*y1
    c = dyn * xs[_iu] - dxn * ys[_iu]
    # Pack into a single integer key for fast hashing
    # dxn, dyn are small (bounded by ~2000), c can be larger
    # Use tuple hashing via numpy structured array or just Counter on tuples
    keys = np.column_stack((dxn, dyn, c))
    keys_t = [tuple(k) for k in keys.tolist()]
    cnt = Counter(keys_t)
    score = 0
    max_pairs = 0
    for v in cnt.values():
        if v == 6:
            score += 1
        if v > max_pairs:
            max_pairs = v
    if max_pairs >= 10:
        return False, 0
    return True, score

def _affine_seed():
    """Generate a 4x4 grid affinely transformed, try many random transforms."""
    base = np.array([[x, y] for y in range(4) for x in range(4)], dtype=np.int64)
    for _ in range(300):
        A = np.array([[random.randint(-4, 4), random.randint(-4, 4)],
                       [random.randint(-4, 4), random.randint(-4, 4)]], dtype=np.int64)
        det = int(A[0,0]*A[1,1] - A[0,1]*A[1,0])
        if det == 0:
            continue
        t = A @ base.T
        if np.any(np.abs(t) > 950):
            continue
        pts = t.T.tolist()
        is_valid, score = _evaluate_fast(pts)
        if is_valid and score >= 4:
            return pts, score
    return None, 0

def _count_lines_exactly4(points):
    """Return number of lines with exactly 4 points (assumes validity already checked)."""
    xs = np.array([p[0] for p in points], dtype=np.int64)
    ys = np.array([p[1] for p in points], dtype=np.int64)
    dx = xs[_ju] - xs[_iu]
    dy = ys[_ju] - ys[_iu]
    g = np.gcd(np.abs(dx), np.abs(dy))
    g_safe = np.where(g == 0, 1, g)
    dxn = dx // g_safe
    dyn = dy // g_safe
    neg = (dxn < 0) | ((dxn == 0) & (dyn < 0))
    dxn = np.where(neg, -dxn, dxn)
    dyn = np.where(neg, -dyn, dyn)
    c = dyn * xs[_iu] - dxn * ys[_iu]
    keys = np.column_stack((dxn, dyn, c))
    keys_t = [tuple(k) for k in keys.tolist()]
    cnt = Counter(keys_t)
    score = 0
    max_pairs = 0
    for v in cnt.values():
        if v == 6:
            score += 1
        if v > max_pairs:
            max_pairs = v
    return score, max_pairs

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    best_points = None
    best_score = -1

    # Phase 1: Find good starting configurations via affine seeds
    for _ in range(50):
        if time.time() > deadline - 30:
            break
        pts, score = _affine_seed()
        if pts is not None and score > best_score:
            best_score = score
            best_points = pts
        if best_score >= 12:
            break

    # Fallback
    if best_points is None:
        best_points = [[x, y] for y in range(4) for x in range(4)]
        is_valid, best_score = _evaluate_fast(best_points)

    current_points = [p[:] for p in best_points]
    current_score = best_score

    # Phase 2: Local search with multiple strategies
    # Strategy: pick a point, try moving it. Use adaptive step sizes.
    # Also try "swap" moves: pick two points and swap their positions.
    # And "line-preserving" moves: move a point along a line that already has 3 of its points.

    while time.time() < deadline:
        # Check time more frequently
        if time.time() > deadline:
            break

        r = random.random()

        if r < 0.40:
            # Single point move
            idx = random.randint(0, 15)
            old = current_points[idx][:]
            # Adaptive step: mix of small, medium, large
            rr = random.random()
            if rr < 0.4:
                step = random.randint(-3, 3)
                nx = old[0] + step
                ny = old[1] + random.randint(-3, 3)
            elif rr < 0.7:
                step = random.randint(-10, 10)
                nx = old[0] + step
                ny = old[1] + random.randint(-10, 10)
            else:
                nx = random.randint(-1000, 1000)
                ny = random.randint(-1000, 1000)
            if abs(nx) > 1000 or abs(ny) > 1000:
                continue
            current_points[idx] = [nx, ny]
            is_valid, new_score = _evaluate_fast(current_points)
            if is_valid:
                if new_score > best_score:
                    best_score = new_score
                    best_points = [p[:] for p in current_points]
                # Accept with simulated annealing-like probability
                if new_score >= current_score:
                    current_score = new_score
                else:
                    # Accept worse moves occasionally to escape local optima
                    if random.random() < 0.15:
                        current_score = new_score
                    else:
                        current_points[idx] = old
            else:
                current_points[idx] = old

        elif r < 0.55:
            # Swap two points
            i, j = random.sample(range(16), 2)
            old_i = current_points[i][:]
            old_j = current_points[j][:]
            current_points[i] = old_j
            current_points[j] = old_i
            is_valid, new_score = _evaluate_fast(current_points)
            if is_valid:
                if new_score > best_score:
                    best_score = new_score
                    best_points = [p[:] for p in current_points]
                if new_score >= current_score or random.random() < 0.1:
                    current_score = new_score
                else:
                    current_points[i] = old_i
                    current_points[j] = old_j
            else:
                current_points[i] = old_i
                current_points[j] = old_j

        elif r < 0.70:
            # Targeted: move a point along a direction that might create/destroy 4-lines
            idx = random.randint(0, 15)
            old = current_points[idx][:]
            # Pick a random other point and move toward/away from it
            other = random.randint(0, 15)
            if other == idx:
                other = (idx + 1) % 16
            ox, oy = current_points[other]
            dx = ox - old[0]
            dy = oy - old[1]
            # Move by a fraction of the way
            frac = random.choice([1, -1, 2, -2, 3, -3])
            nx = old[0] + dx * frac // max(abs(dx), 1) if dx != 0 else old[0] + random.randint(-5, 5)
            ny = old[1] + dy * frac // max(abs(dy), 1) if dy != 0 else old[1] + random.randint(-5, 5)
            # Simpler: just add a small perturbation biased toward the other point
            nx = old[0] + random.randint(-5, 5) + (dx // 4 if dx != 0 else 0)
            ny = old[1] + random.randint(-5, 5) + (dy // 4 if dy != 0 else 0)
            if abs(nx) > 1000 or abs(ny) > 1000:
                continue
            current_points[idx] = [nx, ny]
            is_valid, new_score = _evaluate_fast(current_points)
            if is_valid:
                if new_score > best_score:
                    best_score = new_score
                    best_points = [p[:] for p in current_points]
                if new_score >= current_score or random.random() < 0.12:
                    current_score = new_score
                else:
                    current_points[idx] = old
            else:
                current_points[idx] = old

        else:
            # Restart from best with perturbation
            if best_points is not None:
                current_points = [p[:] for p in best_points]
                current_score = best_score
                # Perturb 2-3 random points
                for _ in range(random.randint(2, 3)):
                    idx = random.randint(0, 15)
                    current_points[idx] = [
                        max(-1000, min(1000, current_points[idx][0] + random.randint(-8, 8))),
                        max(-1000, min(1000, current_points[idx][1] + random.randint(-8, 8)))
                    ]
                is_valid, new_score = _evaluate_fast(current_points)
                if not is_valid:
                    current_points = [p[:] for p in best_points]
                    current_score = best_score

    return best_points
