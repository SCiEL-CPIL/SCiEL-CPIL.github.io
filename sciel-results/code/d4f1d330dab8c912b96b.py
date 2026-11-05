import random
import time
from math import gcd

def _get_line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    A = y1 - y2
    B = x2 - x1
    C = -(A * x1 + B * y1)
    
    g = gcd(abs(A), abs(B))
    if g == 0:
        g = abs(C)
    if g != 0:
        A //= g
        B //= g
        C //= g
        
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
        
    return (A, B, C)

def _build_line_index(points):
    n = len(points)
    line_points = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _get_line_key(points[i], points[j])
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    return line_points

def _evaluate_from_index(points, line_points):
    valid = True
    score = 0
    for key, idx_set in line_points.items():
        cnt = len(idx_set)
        if cnt >= 5:
            valid = False
            break
        elif cnt == 4:
            score += 1
    return valid, score

def _incremental_update(points, line_points, old_idx, old_pt, new_pt):
    n = len(points)
    lines_to_remove = []
    for key, idx_set in line_points.items():
        if old_idx in idx_set:
            lines_to_remove.append(key)
    
    for key in lines_to_remove:
        idx_set = line_points[key]
        idx_set.discard(old_idx)
        if len(idx_set) <= 1:
            del line_points[key]
    
    for j in range(n):
        if j == old_idx:
            continue
        key = _get_line_key(new_pt, points[j])
        if key not in line_points:
            line_points[key] = set()
        line_points[key].add(old_idx)
        line_points[key].add(j)
    
    return line_points

def _generate_informed_moves(points, line_points, idx, rng):
    """Generate candidate moves that are geometrically informed."""
    old_x, old_y = points[idx]
    moves = []
    
    # 1) Small perturbations around current position
    for _ in range(3):
        dx = rng.randint(-3, 3)
        dy = rng.randint(-3, 3)
        moves.append([old_x + dx, old_y + dy])
    
    # 2) Try to align with existing 3-point lines to create 4-point lines
    # Find lines with exactly 3 points that don't include idx
    for key, idx_set in list(line_points.items()):
        if len(idx_set) == 3 and idx not in idx_set:
            # This line has 3 points; if we move idx onto it, we get 4
            # Pick a point on the line and generate a candidate near the line
            pts_on_line = [points[i] for i in idx_set]
            # Use the line equation to find a good integer point
            A, B, C = key
            # Try several integer points on this line
            for t in range(-5, 6):
                # Parametrize: if B != 0, x = t, y = (-C - A*t) / B
                if B != 0:
                    num = -C - A * t
                    if num % B == 0:
                        y_val = num // B
                        if abs(t) <= 1000 and abs(y_val) <= 1000:
                            cand = [t, y_val]
                            # Check distinctness
                            is_distinct = True
                            for i, p in enumerate(points):
                                if i != idx and p[0] == t and p[1] == y_val:
                                    is_distinct = False
                                    break
                            if is_distinct:
                                moves.append(cand)
                elif A != 0:
                    num = -C - B * t
                    if num % A == 0:
                        x_val = num // A
                        if abs(x_val) <= 1000 and abs(t) <= 1000:
                            cand = [x_val, t]
                            is_distinct = True
                            for i, p in enumerate(points):
                                if i != idx and p[0] == x_val and p[1] == t:
                                    is_distinct = False
                                    break
                            if is_distinct:
                                moves.append(cand)
                if len(moves) > 15:
                    break
        if len(moves) > 15:
            break
    
    # 3) Medium random jumps
    for _ in range(3):
        moves.append([rng.randint(-10, 10), rng.randint(-10, 10)])
    
    # 4) Larger exploration jumps
    moves.append([rng.randint(-50, 50), rng.randint(-50, 50)])
    
    return moves

def _anneal_from_start(points, line_points, start_score, deadline, rng, initial_temp=1.0):
    """Run simulated annealing from a starting configuration."""
    current_score = start_score
    best_score = start_score
    best_points = [p[:] for p in points]
    
    temp = initial_temp
    cool_rate = 0.995
    
    while time.time() < deadline and temp > 0.01:
        idx = rng.randint(0, 15)
        old_pt = [points[idx][0], points[idx][1]]
        
        # Generate informed moves
        candidates = _generate_informed_moves(points, line_points, idx, rng)
        
        if not candidates:
            temp *= cool_rate
            continue
        
        # Pick a random candidate
        new_pt = candidates[rng.randint(0, len(candidates) - 1)]
        
        # Check bounds and distinctness
        if abs(new_pt[0]) > 1000 or abs(new_pt[1]) > 1000:
            temp *= cool_rate
            continue
        
        is_distinct = True
        for i, p in enumerate(points):
            if i != idx and p[0] == new_pt[0] and p[1] == new_pt[1]:
                is_distinct = False
                break
        if not is_distinct:
            temp *= cool_rate
            continue
        
        # Apply move
        points[idx] = new_pt
        new_line_points = _incremental_update(points, line_points, idx, old_pt, new_pt)
        is_valid, new_score = _evaluate_from_index(points, new_line_points)
        
        # Annealing acceptance
        if is_valid and new_score >= current_score:
            # Always accept valid moves that don't decrease score
            accept = True
        elif is_valid and new_score < current_score:
            # Accept worse valid moves with probability exp(-delta/temp)
            delta = current_score - new_score
            if delta == 0:
                accept = True
            else:
                prob = 2.718281828 ** (-delta / max(temp, 0.001))
                accept = rng.random() < prob
        else:
            # Invalid: rarely accept if very hot
            accept = temp > 5.0 and rng.random() < 0.01
        
        if accept:
            current_score = new_score
            line_points = new_line_points
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in points]
        else:
            # Revert
            points[idx] = old_pt
            line_points = _build_line_index(points)
            is_valid, current_score = _evaluate_from_index(points, line_points)
        
        temp *= cool_rate
    
    return best_points, best_score, points, line_points, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s margin before 120s kill
    
    rng = random.Random(42)
    
    # Initial: 4x4 grid
    points = [[x, y] for x in range(4) for y in range(4)]
    line_points = _build_line_index(points)
    is_valid, current_score = _evaluate_from_index(points, line_points)
    
    best_points = [p[:] for p in points]
    best_score = current_score
    
    if not is_valid:
        return best_points
    
    # Phase 1: Annealing from 4x4 grid
    phase1_deadline = start_time + 60
    best_points, best_score, points, line_points, current_score = _anneal_from_start(
        points, line_points, current_score, phase1_deadline, rng, initial_temp=2.0
    )
    
    # Phase 2: Multi-restart from perturbed best
    remaining_deadline = deadline
    
    for restart in range(10):
        if time.time() >= remaining_deadline:
            break
        
        # Perturb best solution to create diversity
        perturbed = [p[:] for p in best_points]
        for _ in range(3):
            idx = rng.randint(0, 15)
            perturbed[idx] = [
                perturbed[idx][0] + rng.randint(-5, 5),
                perturbed[idx][1] + rng.randint(-5, 5)
            ]
            # Clamp to bounds
            perturbed[idx][0] = max(-1000, min(1000, perturbed[idx][0]))
            perturbed[idx][1] = max(-1000, min(1000, perturbed[idx][1]))
        
        # Check distinctness
        if len(set(tuple(p) for p in perturbed)) != 16:
            continue
        
        pline_points = _build_line_index(perturbed)
        is_valid, p_score = _evaluate_from_index(perturbed, pline_points)
        
        if not is_valid:
            continue
        
        # Anneal from perturbed start
        phase_deadline = min(remaining_deadline, time.time() + 8)
        new_best_points, new_best_score, _, _, _ = _anneal_from_start(
            perturbed, pline_points, p_score, phase_deadline, rng, initial_temp=1.0
        )
        
        if new_best_score > best_score:
            best_score = new_best_score
            best_points = [p[:] for p in new_best_points]
    
    return best_points
