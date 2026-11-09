"""Combined: target-area construction + multi-scale local search with incremental scoring."""
import time
import random
from collections import Counter

T_START = time.time()

def _elapsed():
    return time.time() - T_START

def _deadline_ok(margin=8.0):
    return _elapsed() < 90.0 - margin

def _doubled_areas(points):
    n = len(points)
    areas = []
    for i in range(n):
        ax, ay = points[i]
        for j in range(i+1, n):
            bx, by = points[j]
            dx1, dy1 = bx-ax, by-ay
            for k in range(j+1, n):
                cx, cy = points[k]
                dx2, dy2 = cx-ax, cy-ay
                det = abs(dx1*dy2 - dy1*dx2)
                if det > 0:
                    areas.append(det)
    return areas

def _valid(pts):
    if len(pts) != 30:
        return False
    if len(set(pts)) != 30:
        return False
    for x, y in pts:
        if not (isinstance(x, int) and isinstance(y, int)):
            return False
        if x < -30 or x > 30 or y < -30 or y > 30:
            return False
    return True

def _score(pts):
    if not _valid(pts):
        return 0
    return max(Counter(_doubled_areas(pts)).values(), default=0)

def _full_counter(pts):
    return Counter(_doubled_areas(pts))

def _fast_move(points, idx, old_pt, new_pt, counter):
    """Update counter after moving point idx from old_pt to new_pt."""
    n = len(points)
    c = Counter(counter)
    ox, oy = old_pt
    nx, ny = new_pt
    others = [i for i in range(n) if i != idx]
    # Remove triples containing idx with old_pt
    for ii in range(len(others)):
        i = others[ii]
        for jj in range(ii+1, len(others)):
            j = others[jj]
            ax, ay = points[i]
            bx, by = points[j]
            det = abs((ax-ox)*(by-oy) - (ay-oy)*(bx-ox))
            if det > 0:
                c[det] -= 1
                if c[det] <= 0:
                    del c[det]
    # Add triples containing idx with new_pt
    for ii in range(len(others)):
        i = others[ii]
        for jj in range(ii+1, len(others)):
            j = others[jj]
            ax, ay = points[i]
            bx, by = points[j]
            det = abs((ax-nx)*(by-ny) - (ay-ny)*(bx-nx))
            if det > 0:
                c[det] += 1
    mx = max(c.values(), default=0)
    return c, mx

def _construct_for_A(A, rng):
    """Construct 30 points with many triples of doubled area A using pivot + direction method."""
    directions = [
        (1, 0), (0, 1), (1, 1), (1, -1), (2, 1), (1, 2), (2, -1), (3, 1),
        (1, 3), (3, 2), (2, 3), (-1, 1), (4, 1), (1, 4), (4, 3), (3, 4),
        (5, 2), (2, 5), (5, 1), (1, 5), (5, 3), (3, 5), (5, 4), (4, 5)
    ]
    rng.shuffle(directions)
    
    best_pts = None
    best_score = 0
    
    pivot_cands = []
    for x in range(-15, 16, 3):
        for y in range(-15, 16, 3):
            pivot_cands.append((x, y))
    rng.shuffle(pivot_cands)
    
    for pivot in pivot_cands[:15]:
        px, py = pivot
        candidates = []
        for vx, vy in directions:
            if vx == 0:
                if vy != 0 and A % vy == 0:
                    y = py - A // vy
                    if -30 <= y <= 30:
                        for x in range(-30, 31):
                            if (x, y) != (px, py):
                                candidates.append((x, y))
            elif vy == 0:
                if A % vx == 0:
                    x = px + A // vx
                    if -30 <= x <= 30:
                        for y in range(-30, 31):
                            if (x, y) != (px, py):
                                candidates.append((x, y))
            else:
                for y in range(-30, 31):
                    num = A + (y - py) * vx
                    if num % vy == 0:
                        x = px + num // vy
                        if -30 <= x <= 30 and (x, y) != (px, py):
                            candidates.append((x, y))
        
        if len(candidates) < 29:
            continue
        
        # Sort by distance to pivot for compactness
        seen = set()
        unique_cands = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique_cands.append(c)
        
        unique_cands.sort(key=lambda c: c[0]**2 + c[1]**2)
        
        chosen = [(px, py)]
        used = set(chosen)
        for c in unique_cands:
            if c not in used:
                chosen.append(c)
                used.add(c)
            if len(chosen) == 30:
                break
        
        if len(chosen) < 30:
            # Fill with random points
            while len(chosen) < 30:
                rx = rng.randint(-30, 30)
                ry = rng.randint(-30, 30)
                if (rx, ry) not in used:
                    chosen.append((rx, ry))
                    used.add((rx, ry))
        
        s = _score(chosen)
        if s > best_score:
            best_score = s
            best_pts = list(chosen)
    
    return best_pts, best_score

def _local_search(pts, counter, max_iter=300, margin=10.0):
    """Multi-scale hill-climb with incremental scoring."""
    n = len(pts)
    best_pts = list(pts)
    best_counter = Counter(counter)
    best_score = max(best_counter.values(), default=0)
    occupied = set(best_pts)
    
    # Multi-scale neighborhoods
    nh_small = [(dx, dy) for dx in range(-3, 4) for dy in range(-3, 4) if (dx, dy) != (0, 0)]
    nh_med = [(dx, dy) for dx in range(-6, 7) for dy in range(-6, 7) if (dx, dy) != (0, 0)]
    nh_large = [(dx, dy) for dx in range(-12, 13) for dy in range(-12, 13) if (dx, dy) != (0, 0)]
    
    rng = random.Random(hash(tuple(pts)) % (2**32))
    
    for it in range(max_iter):
        if not _deadline_ok(margin):
            break
        
        # Choose neighborhood scale based on iteration
        if it < max_iter // 3:
            nh = nh_small
        elif it < 2 * max_iter // 3:
            nh = nh_med
        else:
            nh = nh_large
        
        improved = False
        order = list(range(n))
        rng.shuffle(order)
        
        for idx in order:
            if not _deadline_ok(margin):
                break
            ox, oy = best_pts[idx]
            # Shuffle neighbors for diversity
            nh_copy = list(nh)
            rng.shuffle(nh_copy)
            
            for dx, dy in nh_copy:
                nx, ny = ox+dx, oy+dy
                if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                    continue
                np = (nx, ny)
                if np in occupied:
                    continue
                new_c, new_s = _fast_move(best_pts, idx, (ox, oy), np, best_counter)
                if new_s > best_score:
                    best_pts[idx] = np
                    best_counter = new_c
                    best_score = new_s
                    occupied.discard((ox, oy))
                    occupied.add(np)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            # Try a random jump to escape local optimum
            if rng.random() < 0.3:
                idx = rng.randint(0, n-1)
                ox, oy = best_pts[idx]
                for _ in range(20):
                    nx = rng.randint(-30, 30)
                    ny = rng.randint(-30, 30)
                    np = (nx, ny)
                    if np not in occupied:
                        new_c, new_s = _fast_move(best_pts, idx, (ox, oy), np, best_counter)
                        if new_s >= best_score:
                            best_pts[idx] = np
                            best_counter = new_c
                            best_score = new_s
                            occupied.discard((ox, oy))
                            occupied.add(np)
                            improved = True
                        break
            if not improved:
                break
    
    return best_pts, best_score, best_counter

def _generate_seeds():
    """Diverse structured seeds."""
    seeds = []
    # 6x5 grid
    seeds.append([(x, y) for x in range(-3, 4) for y in range(-2, 3)])
    # 5x6 grid
    seeds.append([(x, y) for x in range(-2, 3) for y in range(-3, 4)])
    # 10x3
    seeds.append([(x, y) for x in range(-5, 5) for y in range(-1, 2)])
    # 3x10
    seeds.append([(x, y) for x in range(-1, 2) for y in range(-5, 5)])
    # 15x2
    seeds.append([(x, y) for x in range(-7, 8) for y in range(-1, 1)])
    # 2x15
    seeds.append([(x, y) for x in range(-1, 1) for y in range(-7, 8)])
    # 7x4 grid
    seeds.append([(x, y) for x in range(-3, 4) for y in range(-1, 3)])
    # 4x7 grid
    seeds.append([(x, y) for x in range(-1, 3) for y in range(-3, 4)])
    # 8x4 grid
    seeds.append([(x, y) for x in range(-4, 4) for y in range(-1, 3)])
    # 9x3 grid
    seeds.append([(x, y) for x in range(-4, 5) for y in range(-1, 2)])
    # Diamond
    pts = []
    for r in range(-7, 8):
        for c in range(-7, 8):
            if abs(r) + abs(c) <= 7 and len(pts) < 30:
                pts.append((r, c))
    if len(pts) == 30:
        seeds.append(pts)
    # Staggered grid
    pts = []
    for row in range(5):
        for col in range(6):
            x = col - 2 + (row % 2)
            y = row - 2
            pts.append((x, y))
    if len(pts) == 30 and _valid(pts):
        seeds.append(pts)
    return seeds

def construct():
    if not _deadline_ok(15):
        return [(x, y) for x in range(6) for y in range(5)]
    
    best_pts = None
    best_score = 0
    
    # Phase 1: score seeds, polish each
    seeds = _generate_seeds()
    for seed in seeds:
        if not _deadline_ok(15):
            break
        if not _valid(seed):
            continue
        s = _score(seed)
        if s >= best_score:
            counter = _full_counter(seed)
            p, _, c = _local_search(seed, counter, max_iter=150, margin=15)
            ps = max(c.values(), default=0)
            if ps > best_score:
                best_score = ps
                best_pts = list(p)
    
    # Phase 2: model-and-optimize for various target areas A
    rng = random.Random(42)
    target_areas = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 30, 32, 36, 40, 48, 56, 60, 72, 80, 90, 100]
    rng.shuffle(target_areas)
    
    for A in target_areas:
        if not _deadline_ok(12):
            break
        p, s = _construct_for_A(A, rng)
        if p is None:
            continue
        if s >= best_score:
            counter = _full_counter(p)
            pp, _, c = _local_search(p, counter, max_iter=150, margin=12)
            ps = max(c.values(), default=0)
            if ps > best_score:
                best_score = ps
                best_pts = list(pp)
    
    # Phase 3: random restarts with local search
    for _ in range(50):
        if not _deadline_ok(10):
            break
        pts = []
        used = set()
        for _ in range(30):
            x = rng.randint(-25, 25)
            y = rng.randint(-25, 25)
            if (x, y) not in used:
                pts.append((x, y))
                used.add((x, y))
        s = _score(pts)
        if s >= best_score - 5:
            counter = _full_counter(pts)
            p, _, c = _local_search(pts, counter, max_iter=100, margin=10)
            ps = max(c.values(), default=0)
            if ps > best_score:
                best_score = ps
                best_pts = list(p)
    
    # Final polish with more iterations
    if best_pts is not None and _deadline_ok(8):
        counter = _full_counter(best_pts)
        p, _, c = _local_search(best_pts, counter, max_iter=300, margin=8)
        ps = max(c.values(), default=0)
        if ps >= best_score:
            best_pts = list(p)
            best_score = ps
    
    if best_pts is None:
        best_pts = [(x, y) for x in range(6) for y in range(5)]
    
    return best_pts
