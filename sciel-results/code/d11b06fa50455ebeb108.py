"""Optimized: delta-based local search with multi-target tracking and
systematic lattice construction for maximizing area multiplicity."""
import time
import random
from collections import Counter
import numpy as np

T_START = time.time()

def _elapsed():
    return time.time() - T_START

def _deadline_ok(margin=8.0):
    return _elapsed() < 88.0 - margin

def _doubled_areas(points):
    """Compute all positive doubled areas for triples."""
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
    """Compute full area counter using vectorized approach."""
    n = len(pts)
    areas = []
    for i in range(n):
        ax, ay = pts[i]
        for j in range(i+1, n):
            bx, by = pts[j]
            dx1, dy1 = bx-ax, by-ay
            for k in range(j+1, n):
                cx, cy = pts[k]
                dx2, dy2 = cx-ax, cy-ay
                det = abs(dx1*dy2 - dy1*dx2)
                if det > 0:
                    areas.append(det)
    return Counter(areas)

def _compute_target_counter(pts, targets):
    """Compute counts for specific target areas only."""
    n = len(pts)
    counts = {t: 0 for t in targets}
    for i in range(n):
        ax, ay = pts[i]
        for j in range(i+1, n):
            bx, by = pts[j]
            dx1, dy1 = bx-ax, by-ay
            for k in range(j+1, n):
                cx, cy = pts[k]
                dx2, dy2 = cx-ax, cy-ay
                det = abs(dx1*dy2 - dy1*dx2)
                if det > 0 and det in counts:
                    counts[det] += 1
    return counts

def _fast_move_delta(pts, idx, old_pt, new_pt, target_counts, targets):
    """Compute delta in target counts when moving point idx.
    Returns dict of target -> count change."""
    n = len(pts)
    ox, oy = old_pt
    nx, ny = new_pt
    delta = {t: 0 for t in targets}
    others = [i for i in range(n) if i != idx]
    
    # Remove contribution of old_pt
    for ii in range(len(others)):
        i = others[ii]
        for jj in range(ii+1, len(others)):
            j = others[jj]
            ax, ay = pts[i]
            bx, by = pts[j]
            det = abs((ax-ox)*(by-oy) - (ay-oy)*(bx-ox))
            if det > 0 and det in delta:
                delta[det] -= 1
    
    # Add contribution of new_pt
    for ii in range(len(others)):
        i = others[ii]
        for jj in range(ii+1, len(others)):
            j = others[jj]
            ax, ay = pts[i]
            bx, by = pts[j]
            det = abs((ax-nx)*(by-ny) - (ay-ny)*(bx-nx))
            if det > 0 and det in delta:
                delta[det] += 1
    
    return delta

def _fast_swap_delta(pts, i, j, target_counts, targets):
    """Compute delta in target counts when swapping points i and j."""
    n = len(pts)
    pi = pts[i]
    pj = pts[j]
    delta = {t: 0 for t in targets}
    others = [k for k in range(n) if k != i and k != j]
    
    # Remove old contributions
    for ii in range(len(others)):
        a = others[ii]
        for jj in range(ii+1, len(others)):
            b = others[jj]
            ax, ay = pts[a]
            bx, by = pts[b]
            det_i = abs((ax-pi[0])*(by-pi[1]) - (ay-pi[1])*(bx-pi[0]))
            if det_i > 0 and det_i in delta:
                delta[det_i] -= 1
            det_j = abs((ax-pj[0])*(by-pj[1]) - (ay-pj[1])*(bx-pj[0]))
            if det_j > 0 and det_j in delta:
                delta[det_j] -= 1
    
    # Add new contributions (i now has pj, j now has pi)
    for ii in range(len(others)):
        a = others[ii]
        for jj in range(ii+1, len(others)):
            b = others[jj]
            ax, ay = pts[a]
            bx, by = pts[b]
            det_i_new = abs((ax-pj[0])*(by-pj[1]) - (ay-pj[1])*(bx-pj[0]))
            if det_i_new > 0 and det_i_new in delta:
                delta[det_i_new] += 1
            det_j_new = abs((ax-pi[0])*(by-pi[1]) - (ay-pi[1])*(bx-pi[0]))
            if det_j_new > 0 and det_j_new in delta:
                delta[det_j_new] += 1
    
    return delta

def _local_search_delta(pts, counter, max_iter=500, margin=10.0):
    """Fast local search using delta-based evaluation with multi-target tracking."""
    n = len(pts)
    best_pts = list(pts)
    best_counter = Counter(counter)
    best_score = max(best_counter.values(), default=0)
    occupied = set(best_pts)
    
    # Track top 3 areas as targets
    top_areas = [a for a, c in counter.most_common(3)]
    if len(top_areas) < 3:
        for a in range(1, 100):
            if a not in top_areas:
                top_areas.append(a)
            if len(top_areas) == 3:
                break
    
    targets = set(top_areas)
    target_counts = _compute_target_counter(best_pts, targets)
    
    # Adaptive neighborhood
    base_nbs = [(dx, dy) for dx in range(-6, 7) for dy in range(-6, 7) if (dx, dy) != (0, 0)]
    rng = random.Random(hash(tuple(pts)) % (2**32))
    
    no_improve = 0
    for it in range(max_iter):
        if not _deadline_ok(margin):
            break
        
        # Update targets periodically
        if it % 50 == 0:
            top_areas = [a for a, c in best_counter.most_common(3)]
            if len(top_areas) < 3:
                for a in range(1, 100):
                    if a not in top_areas:
                        top_areas.append(a)
                    if len(top_areas) == 3:
                        break
            targets = set(top_areas)
            target_counts = _compute_target_counter(best_pts, targets)
        
        improved = False
        order = list(range(n))
        rng.shuffle(order)
        
        for idx in order:
            if not _deadline_ok(margin):
                break
            ox, oy = best_pts[idx]
            # Try moves in random order
            moves = base_nbs[:]
            rng.shuffle(moves)
            for dx, dy in moves:
                nx, ny = ox+dx, oy+dy
                if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                    continue
                np = (nx, ny)
                if np in occupied:
                    continue
                
                delta = _fast_move_delta(best_pts, idx, (ox, oy), np, target_counts, targets)
                # Check if any target improves
                new_max = 0
                for t in targets:
                    new_val = target_counts[t] + delta[t]
                    if new_val > new_max:
                        new_max = new_val
                
                if new_max > best_score:
                    # Apply move
                    best_pts[idx] = np
                    occupied.discard((ox, oy))
                    occupied.add(np)
                    # Update target counts
                    for t in targets:
                        target_counts[t] += delta[t]
                    best_score = new_max
                    # Update full counter
                    best_counter = _full_counter(best_pts)
                    improved = True
                    no_improve = 0
                    break
            if improved:
                break
        
        if not improved:
            # Try swaps
            improved = False
            for _ in range(30):
                if not _deadline_ok(margin):
                    break
                i, j = rng.sample(range(n), 2)
                delta = _fast_swap_delta(best_pts, i, j, target_counts, targets)
                new_max = 0
                for t in targets:
                    new_val = target_counts[t] + delta[t]
                    if new_val > new_max:
                        new_max = new_val
                if new_max > best_score:
                    best_pts[i], best_pts[j] = best_pts[j], best_pts[i]
                    for t in targets:
                        target_counts[t] += delta[t]
                    best_score = new_max
                    best_counter = _full_counter(best_pts)
                    improved = True
                    no_improve = 0
                    break
        
        if not improved:
            no_improve += 1
            if no_improve > 30:
                break
    return best_pts, best_score, best_counter

def _generate_lattice_seeds():
    """Generate diverse lattice-based seeds."""
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
    
    # Diamond
    pts = []
    for r in range(-7, 8):
        for c in range(-7, 8):
            if abs(r) + abs(c) <= 7 and len(pts) < 30:
                pts.append((r, c))
    if len(pts) == 30:
        seeds.append(pts)
    
    # Hexagonal-ish
    pts = []
    for r in range(-4, 5):
        for c in range(-4, 5):
            if abs(r) <= 4 and abs(c) <= 4 and len(pts) < 30:
                pts.append((r*2, c*2 + (r % 2)))
    if len(pts) == 30:
        seeds.append(pts)
    
    # Triangular lattice
    pts = []
    for row in range(-3, 4):
        for col in range(-3, 4):
            x = col + (row % 2)
            y = row
            if -30 <= x <= 30 and -30 <= y <= 30 and len(pts) < 30:
                pts.append((x, y))
    if len(pts) == 30:
        seeds.append(pts)
    
    # Square lattice with offset
    pts = []
    for row in range(-3, 4):
        for col in range(-3, 4):
            x = col * 2
            y = row * 2
            if -30 <= x <= 30 and -30 <= y <= 30 and len(pts) < 30:
                pts.append((x, y))
    if len(pts) == 30:
        seeds.append(pts)
    
    # Concentric squares
    pts = []
    for r in range(0, 8):
        if r == 0:
            pts.append((0, 0))
        else:
            for i in range(-r, r+1):
                for j in range(-r, r+1):
                    if abs(i) == r or abs(j) == r:
                        if -30 <= i <= 30 and -30 <= j <= 30 and len(pts) < 30:
                            pts.append((i, j))
    if len(pts) == 30:
        seeds.append(pts)
    
    # Circle points
    pts = []
    for r in range(1, 8):
        for angle in range(0, 360, 45):
            import math
            x = round(r * math.cos(math.radians(angle)))
            y = round(r * math.sin(math.radians(angle)))
            if -30 <= x <= 30 and -30 <= y <= 30 and (x, y) not in pts and len(pts) < 30:
                pts.append((x, y))
    if len(pts) == 30:
        seeds.append(pts)
    
    return seeds

def _construct_for_A(A, rng, max_pivots=12):
    """Construct 30 points with many triples of doubled area A using lattice lines."""
    # For area A, we want points on lines such that cross products give A
    # Use multiple parallel lines with specific spacing
    
    best_pts = None
    best_score = 0
    
    # Try different pivot points
    pivot_cands = []
    for x in range(-10, 11, 3):
        for y in range(-10, 11, 3):
            pivot_cands.append((x, y))
    rng.shuffle(pivot_cands)
    
    for pivot in pivot_cands[:max_pivots]:
        px, py = pivot
        candidates = []
        
        # For each direction, find points on line at distance giving area A
        # Cross product of (p - pivot) and direction = A
        # So we want |cross(p - pivot, d)| = A for various directions d
        
        directions = [(1,0), (0,1), (1,1), (1,-1), (2,1), (1,2), (2,-1), (3,1), 
                      (1,3), (3,2), (2,3), (-1,1), (4,1), (1,4), (3,-1), (-1,3),
                      (5,1), (1,5), (4,3), (3,4), (5,2), (2,5), (5,3), (3,5)]
        rng.shuffle(directions)
        
        for vx, vy in directions:
            # Points (x,y) such that (x-px)*vy - (y-py)*vx = ±A
            # Parametrize: x = px + t, y = py + (t*vy ± A)/vx if vx != 0
            # Or: y = py + t, x = px + (t*vx ± A)/vy if vy != 0
            
            for sign in [1, -1]:
                target = sign * A
                if vx == 0:
                    # -vy*(y-py) = target => y = py - target/vy
                    if target % vy == 0:
                        y = py - target // vy
                        if -30 <= y <= 30:
                            for x in range(-30, 31):
                                if (x, y) != (px, py):
                                    candidates.append((x, y))
                elif vy == 0:
                    # vx*(x-px) = target => x = px + target/vx
                    if target % vx == 0:
                        x = px + target // vx
                        if -30 <= x <= 30:
                            for y in range(-30, 31):
                                if (x, y) != (px, py):
                                    candidates.append((x, y))
                else:
                    # (x-px)*vy - (y-py)*vx = target
                    # x = px + (target + (y-py)*vx) / vy
                    for y in range(-30, 31):
                        num = target + (y - py) * vx
                        if num % vy == 0:
                            x = px + num // vy
                            if -30 <= x <= 30 and (x, y) != (px, py):
                                candidates.append((x, y))
        
        if len(candidates) < 29:
            continue
        
        # Remove duplicates
        unique_cands = list(set(candidates))
        if len(unique_cands) < 29:
            continue
        
        # Greedy selection: prefer points that create more A-area triples
        rng.shuffle(unique_cands)
        chosen = [(px, py)]
        used = set(chosen)
        
        # Score each candidate by how many A-area triples it creates with existing points
        def candidate_score(pt):
            count = 0
            for i in range(len(chosen)):
                for j in range(i+1, len(chosen)):
                    ax, ay = chosen[i]
                    bx, by = chosen[j]
                    det = abs((pt[0]-ax)*(by-ay) - (pt[1]-ay)*(bx-ax))
                    if det == A:
                        count += 1
            return count
        
        # Iteratively add best candidate
        remaining = [c for c in unique_cands if c not in used]
        while len(chosen) < 30 and remaining:
            best_c = max(remaining, key=candidate_score)
            chosen.append(best_c)
            used.add(best_c)
            remaining.remove(best_c)
        
        if len(chosen) < 30:
            continue
        
        s = _score(chosen)
        if s > best_score:
            best_score = s
            best_pts = list(chosen)
    
    return best_pts, best_score

def _simulated_annealing(pts, counter, max_iter=300, margin=8.0):
    """Simulated annealing to escape local optima."""
    n = len(pts)
    best_pts = list(pts)
    best_counter = Counter(counter)
    best_score = max(best_counter.values(), default=0)
    occupied = set(best_pts)
    
    current_pts = list(pts)
    current_score = best_score
    occupied = set(current_pts)
    
    T = 10.0
    T_min = 0.1
    cooling = 0.995
    
    nbs = [(dx, dy) for dx in range(-8, 9) for dy in range(-8, 9) if (dx, dy) != (0, 0)]
    rng = random.Random(hash(tuple(pts)) % (2**32))
    
    for it in range(max_iter):
        if not _deadline_ok(margin):
            break
        
        T = max(T * cooling, T_min)
        
        # Random move
        idx = rng.randint(0, n-1)
        ox, oy = current_pts[idx]
        dx, dy = rng.choice(nbs)
        nx, ny = ox+dx, oy+dy
        if nx < -30 or nx > 30 or ny < -30 or ny > 30:
            continue
        np = (nx, ny)
        if np in occupied:
            continue
        
        # Evaluate
        old_counter = Counter(_doubled_areas(current_pts))
        current_pts[idx] = np
        new_counter = Counter(_doubled_areas(current_pts))
        new_score = max(new_counter.values(), default=0)
        
        delta = new_score - current_score
        if delta >= 0 or rng.random() < np.exp(delta / T):
            occupied.discard((ox, oy))
            occupied.add(np)
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_pts = list(current_pts)
                best_counter = new_counter
        else:
            current_pts[idx] = (ox, oy)
    
    return best_pts, best_score, best_counter

def construct():
    if not _deadline_ok(15):
        return [(x, y) for x in range(6) for y in range(5)]
    
    best_pts = None
    best_score = 0
    
    # Phase 1: Score seeds, polish each with fast local search
    seeds = _generate_lattice_seeds()
    for seed in seeds:
        if not _deadline_ok(12):
            break
        if not _valid(seed):
            continue
        s = _score(seed)
        if s >= best_score:
            counter = _full_counter(seed)
            p, ps, c = _local_search_delta(seed, counter, max_iter=300, margin=12)
            if ps > best_score:
                best_score = ps
                best_pts = list(p)
    
    # Phase 2: Model-and-optimize for various target areas
    rng = random.Random(42)
    target_areas = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 30, 32, 36, 40, 48, 56, 60, 64, 72, 80]
    rng.shuffle(target_areas)
    
    for A in target_areas:
        if not _deadline_ok(8):
            break
        p, s = _construct_for_A(A, rng, max_pivots=8)
        if p is None:
            continue
        if s >= best_score - 5:
            counter = _full_counter(p)
            pp, ps, c = _local_search_delta(p, counter, max_iter=200, margin=8)
            if ps > best_score:
                best_score = ps
                best_pts = list(pp)
    
    # Phase 3: Simulated annealing on best result
    if best_pts is not None and _deadline_ok(10):
        counter = _full_counter(best_pts)
        p, ps, c = _simulated_annealing(best_pts, counter, max_iter=200, margin=10)
        if ps > best_score:
            best_score = ps
            best_pts = list(p)
    
    # Phase 4: Final polish
    if best_pts is not None and _deadline_ok(6):
        counter = _full_counter(best_pts)
        p, ps, c = _local_search_delta(best_pts, counter, max_iter=400, margin=6)
        if ps >= best_score:
            best_pts = list(p)
            best_score = ps
    
    if best_pts is None:
        best_pts = [(x, y) for x in range(6) for y in range(5)]
    
    return best_pts
