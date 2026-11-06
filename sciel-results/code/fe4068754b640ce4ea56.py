"""Structural rewrite: line-based constructive search + fast pair-line scorer."""
# EVOLVE-BLOCK-START
import time
import math
import random
from collections import defaultdict

def _line_key(p1, p2):
    """Return canonical (a,b,c) for line through p1,p2 where a*x+b*y=c, gcd-normalized."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    g = math.gcd(abs(dx), abs(dy))
    if g == 0:
        return None
    dx_n, dy_n = dx // g, dy // g
    if dx_n < 0 or (dx_n == 0 and dy_n < 0):
        dx_n, dy_n = -dx_n, -dy_n
    a, b = dy_n, -dx_n
    c = a * x1 + b * y1
    g2 = math.gcd(abs(a), abs(b))
    if g2 > 1:
        a, b, c = a // g2, b // g2, c // g2
    if a < 0 or (a == 0 and b < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

def _score_fast(points):
    """Count exact-4 lines and detect 5+ collinear. Returns (exact4, has5, line_counts)."""
    n = len(points)
    line_pts = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            line_pts[key].add(i)
            line_pts[key].add(j)
    exact4 = 0
    has5 = False
    for key, s in line_pts.items():
        a, b, c = key
        cnt = 0
        for i in range(n):
            px, py = points[i]
            if a * px + b * py == c:
                cnt += 1
        if cnt == 4:
            exact4 += 1
        elif cnt >= 5:
            has5 = True
    return exact4, has5

def _try_place_line(points, idx, rng, max_coord=1000):
    """Try to move point idx so it completes a 4-collinear line with 3 existing points.
    Returns (new_point, new_line_key) or (None, None)."""
    n = len(points)
    # Find all triples of other points that are collinear
    others = [i for i in range(n) if i != idx]
    rng.shuffle(others)
    for a in range(len(others)):
        for b in range(a + 1, len(others)):
            for c in range(b + 1, len(others)):
                i, j, k = others[a], others[b], others[c]
                key = _line_key(points[i], points[j])
                if key is None:
                    continue
                # Check if k is on this line
                a_, b_, c_ = key
                if a_ * points[k][0] + b_ * points[k][1] != c_:
                    continue
                # This line has at least 3 of our points. Check if it already has 4+
                # (we want to add a 4th, not create a 5th)
                cnt = 0
                for m in range(n):
                    if m == idx:
                        continue
                    px, py = points[m]
                    if a_ * px + b_ * py == c_:
                        cnt += 1
                if cnt >= 4:
                    continue  # would make 5
                # Find a 4th point on this line within bounds, distinct from existing
                # Line: a_*x + b_*y = c_
                # We need integer solutions. Parametrize.
                # If a_ != 0: x = (c_ - b_*y) / a_
                # If b_ != 0: y = (c_ - a_*x) / b_
                # Try a range of values
                candidates = []
                if a_ != 0:
                    # x = (c_ - b_*y) / a_
                    # y must be such that (c_ - b_*y) % a_ == 0
                    rem = c_ % a_ if a_ > 0 else (-c_) % (-a_)
                    # We need b_*y ≡ c_ (mod a_)
                    # Try y in a range
                    for y_try in range(-1000, 1001):
                        if (c_ - b_ * y_try) % a_ != 0:
                            continue
                        x_try = (c_ - b_ * y_try) // a_
                        if abs(x_try) > max_coord or abs(y_try) > max_coord:
                            continue
                        pt = (x_try, y_try)
                        # Check distinctness
                        if any(pt == points[m] for m in range(n) if m != idx):
                            continue
                        candidates.append(pt)
                elif b_ != 0:
                    for x_try in range(-1000, 1001):
                        if (c_ - a_ * x_try) % b_ != 0:
                            continue
                        y_try = (c_ - a_ * x_try) // b_
                        if abs(x_try) > max_coord or abs(y_try) > max_coord:
                            continue
                        pt = (x_try, y_try)
                        if any(pt == points[m] for m in range(n) if m != idx):
                            continue
                        candidates.append(pt)
                else:
                    continue
                if candidates:
                    # Pick a random candidate
                    pt = rng.choice(candidates)
                    return pt, key
    return None, None

def construct():
    deadline = time.time() + 140
    rng = random.Random(42)
    
    # Start with a 4x4 grid (known to have many 4-collinear lines)
    base = [[x, y] for x in range(4) for y in range(4)]
    best_score, _ = _score_fast(base)
    best_points = [p[:] for p in base]
    
    # Phase 1: Line-completion constructive search
    for restart in range(60):
        if time.time() > deadline - 20:
            break
        
        if restart == 0:
            current = [p[:] for p in base]
        elif rng.random() < 0.5:
            current = [p[:] for p in best_points]
        else:
            # Perturbed grid
            current = [[x, y] for x in range(4) for y in range(4)]
            for _ in range(rng.randint(1, 6)):
                idx = rng.randint(0, 15)
                current[idx] = [current[idx][0] + rng.randint(-3, 3),
                                 current[idx][1] + rng.randint(-3, 3)]
                # Ensure distinctness
                for j in range(16):
                    if j != idx and current[j] == current[idx]:
                        current[idx] = [current[idx][0] + 1, current[idx][1]]
        
        s, h5 = _score_fast(current)
        if h5:
            continue
        current_score = s
        
        # Line-completion moves (60% of the time)
        steps = 0
        no_improve = 0
        max_no_improve = 300
        T = 0.8
        
        while time.time() < deadline - 15 and no_improve < max_no_improve:
            steps += 1
            idx = rng.randint(0, 15)
            old = current[idx][:]
            
            if rng.random() < 0.6:
                # Try to complete a line
                new_pt, _ = _try_place_line(current, idx, rng)
                if new_pt is None:
                    # Fallback: small random move
                    new_pt = [old[0] + rng.randint(-4, 4), old[1] + rng.randint(-4, 4)]
            else:
                # Random small move
                new_pt = [old[0] + rng.randint(-5, 5), old[1] + rng.randint(-5, 5)]
            
            if abs(new_pt[0]) > 1000 or abs(new_pt[1]) > 1000:
                no_improve += 1
                continue
            if any(new_pt == p for p in current if p is not current[idx]):
                no_improve += 1
                continue
            
            current[idx] = new_pt
            s_new, h5_new = _score_fast(current)
            
            if h5_new:
                current[idx] = old
                no_improve += 1
                continue
            
            delta = s_new - current_score
            if delta > 0 or (T > 0.1 and rng.random() < math.exp(delta / T)):
                if delta > 0:
                    no_improve = 0
                    if s_new > best_score:
                        best_score = s_new
                        best_points = [p[:] for p in current]
                else:
                    no_improve += 1
                current_score = s_new
            else:
                current[idx] = old
                no_improve += 1
            
            T = max(0.1, T * 0.998)
    
    return best_points
# EVOLVE-BLOCK-END
