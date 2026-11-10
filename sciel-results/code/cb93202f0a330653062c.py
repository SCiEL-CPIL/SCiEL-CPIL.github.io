# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random
from itertools import combinations

def _normalize_line(A, B, C):
    A, B, C = int(A), int(B), int(C)
    g = math.gcd(abs(A), abs(B))
    g = math.gcd(g, abs(C))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidence(points, lines):
    """Count incidences between points (N,2) and lines (M,3)."""
    if len(points) == 0 or len(lines) == 0:
        return 0
    px = points[:, 0]
    py = points[:, 1]
    A = lines[:, 0]
    B = lines[:, 1]
    C = lines[:, 2]
    # (M, N) matrix
    vals = A[:, None] * px[None, :] + B[:, None] * py[None, :] + C[:, None]
    return int(np.sum(vals == 0))

def _find_best_lines_for_points(points, k=27, deadline=None):
    """Find the k lines with maximum total incidence for the given points."""
    n = len(points)
    points_int = points.astype(np.int64)
    
    # Collect all distinct lines from pairs of points
    seen_lines = set()
    candidate_lines = []
    
    for i in range(n):
        x1, y1 = int(points_int[i, 0]), int(points_int[i, 1])
        for j in range(i + 1, n):
            x2, y2 = int(points_int[j, 0]), int(points_int[j, 1])
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue
            key = _normalize_line(A, B, C)
            if key not in seen_lines:
                seen_lines.add(key)
                candidate_lines.append(key)
    
    # Also add lines with small integer coefficients not captured by point pairs
    # These can sometimes give better incidences (e.g., diagonal lines through grid)
    for A in range(-4, 5):
        for B in range(-4, 5):
            if A == 0 and B == 0:
                continue
            # Try several C values
            for C in range(-20, 21):
                key = _normalize_line(A, B, C)
                if key not in seen_lines:
                    seen_lines.add(key)
                    candidate_lines.append(key)
    
    if len(candidate_lines) == 0:
        return np.array([[_normalize_line(0, 1, -c)] for c in range(-27, 0)], dtype=np.int64)
    
    arr = np.array(candidate_lines, dtype=np.int64)
    
    px = points_int[:, 0:1].T  # (1, N)
    py = points_int[:, 1:2].T  # (1, N)
    A = arr[:, 0:1]
    B = arr[:, 1:2]
    C = arr[:, 2:3]
    
    # Compute incidence counts for all candidates
    vals = A * px + B * py + C
    counts = np.sum(vals == 0, axis=1)
    
    # Sort by count descending
    if len(counts) > k:
        idx = np.argsort(-counts)
        top_idx = idx[:k]
    else:
        top_idx = np.arange(len(counts))
    
    return arr[top_idx].astype(np.int64)

def _generate_point_sets(deadline):
    """Generate a variety of point configurations."""
    sets = []
    
    # 1. Rectangular Grid 6x9 and 9x6
    for w, h in [(6, 9), (9, 6), (7, 8), (8, 7)]:
        if w * h == 54:
            pts = np.array([[i, j] for i in range(w) for j in range(h)], dtype=np.int64)
            sets.append(pts)
    
    # 1b. Scaled grids (spacing > 1) to allow more line diversity
    for scale in [2, 3]:
        for w, h in [(6, 9), (9, 6)]:
            if w * h == 54:
                pts = np.array([[i * scale, j * scale] for i in range(w) for j in range(h)], dtype=np.int64)
                sets.append(pts)
    
    # 1c. Skewed grids (sheared)
    for shear in [1, 2, -1]:
        for w, h in [(6, 9)]:
            pts = np.array([[i + shear * j, j] for i in range(w) for j in range(h)], dtype=np.int64)
            if len(set(map(tuple, pts.tolist()))) == 54:
                sets.append(pts)
    
    # 2. Triangular/hexagonal lattice patterns
    # Pattern A: staggered rows
    for stagger in [0, 1]:
        pts = []
        for j in range(9):
            offset = (j % 2) * stagger
            for i in range(6):
                pts.append([i + offset, j])
        pts = np.array(pts, dtype=np.int64)
        if len(set(map(tuple, pts))) == 54:
            sets.append(pts)
    
    # Pattern B: diagonal lattice
    pts = []
    for j in range(9):
        for i in range(6):
            pts.append([i + j // 3, j])
    pts = np.array(pts, dtype=np.int64)
    if len(set(map(tuple, pts))) == 54:
        sets.append(pts)
    
    # Pattern C: compact diamond shape
    cx, cy = 27, 27
    pts = []
    for dy in range(-5, 6):
        for dx in range(-5, 6):
            if abs(dx) + abs(dy) <= 5 and len(pts) < 54:
                pts.append([cx + dx, cy + dy])
    if len(pts) == 54:
        pts = np.array(pts, dtype=np.int64)
        sets.append(pts)
    
    # Pattern D: two interleaved grids
    pts1 = np.array([[i, j] for i in range(6) for j in range(9) if (i + j) % 2 == 0], dtype=np.int64)
    pts2 = np.array([[i + 0.5, j + 0.5] for i in range(6) for j in range(9) if (i + j) % 2 == 1], dtype=np.int64)
    # Scale to integers: multiply by 2
    pts1 = pts1 * 2
    pts2 = pts2 * 2
    pts = np.vstack([pts1, pts2])
    if len(pts) == 54 and len(set(map(tuple, pts))) == 54:
        sets.append(pts.astype(np.int64))
    
    # Pattern E: spiral-like
    pts = []
    x, y = 0, 0
    dx, dy = 1, 0
    for step in range(54):
        pts.append([x, y])
        for _ in range(step % 2 + 1):
            x += dx
            y += dy
        dx, dy = -dy, dx
    pts = np.array(pts, dtype=np.int64)
    if len(set(map(tuple, pts))) == 54:
        sets.append(pts)
    
    # Pattern F: 3x18 grid (very elongated)
    pts = np.array([[i, j] for i in range(3) for j in range(18)], dtype=np.int64)
    sets.append(pts)
    
    # Pattern G: 18x3 grid
    pts = np.array([[i, j] for i in range(18) for j in range(3)], dtype=np.int64)
    sets.append(pts)
    
    # Pattern H: L-shape
    pts = []
    # Horizontal bar: 18 points
    for i in range(18):
        pts.append([i, 0])
    # Vertical bar: 36 points
    for j in range(1, 37):
        pts.append([0, j])
    pts = np.array(pts, dtype=np.int64)
    if len(pts) == 54 and len(set(map(tuple, pts))) == 54:
        sets.append(pts)
    
    # Pattern I: T-shape
    pts = []
    # Top bar: 18 points
    for i in range(18):
        pts.append([i, 0])
    # Stem: 36 points
    for j in range(1, 37):
        pts.append([8, j])
    pts = np.array(pts, dtype=np.int64)
    if len(pts) == 54 and len(set(map(tuple, pts))) == 54:
        sets.append(pts)
    
    return sets

def _evaluate_set(points, deadline):
    """Evaluate a specific point set by finding best lines."""
    lines = _find_best_lines_for_points(points, 27, deadline)
    score = _count_incidence(points, lines)
    return points, lines, score

def _refine_set(points, lines, deadline, max_steps=5000):
    """
    Refine the configuration by moving points.
    Uses incremental incidence tracking for speed.
    """
    current_score = _count_incidence(points, lines)
    best_points = points.copy()
    best_lines = lines.copy()
    best_score = current_score
    
    no_improve_count = 0
    point_set = set(map(tuple, best_points.tolist()))
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        # Pick a random point to move
        i = random.randint(0, 53)
        ox, oy = int(best_points[i, 0]), int(best_points[i, 1])
        
        # Try moving to a nearby location
        moved = False
        # Try multiple offsets in random order
        offsets = [(dx, dy) for dx in range(-3, 4) for dy in range(-3, 4) if (dx, dy) != (0, 0)]
        random.shuffle(offsets)
        
        for dx, dy in offsets:
            nx, ny = ox + dx, oy + dy
            
            # Check if this position is already taken
            if (nx, ny) in point_set:
                continue
            
            # Create new points
            new_points = best_points.copy()
            new_points[i, 0] = nx
            new_points[i, 1] = ny
            
            # Find best lines for new points
            new_lines = _find_best_lines_for_points(new_points, 27)
            new_score = _count_incidence(new_points, new_lines)
            
            if new_score > current_score:
                best_points = new_points
                best_lines = new_lines
                current_score = new_score
                best_score = current_score
                no_improve_count = 0
                point_set = set(map(tuple, best_points.tolist()))
                moved = True
                break
            elif new_score == current_score and random.random() < 0.05:
                # Accept equal score moves occasionally to escape local optima
                best_points = new_points
                best_lines = new_lines
                point_set = set(map(tuple, best_points.tolist()))
                no_improve_count = 0
                moved = True
                break
        
        if not moved:
            no_improve_count += 1
            if no_improve_count > 100:
                # Try a bigger jump
                i = random.randint(0, 53)
                ox, oy = int(best_points[i, 0]), int(best_points[i, 1])
                nx = ox + random.randint(-8, 8)
                ny = oy + random.randint(-8, 8)
                
                if (nx, ny) not in point_set:
                    new_points = best_points.copy()
                    new_points[i, 0] = nx
                    new_points[i, 1] = ny
                    new_lines = _find_best_lines_for_points(new_points, 27)
                    new_score = _count_incidence(new_points, new_lines)
                    
                    if new_score >= current_score:
                        best_points = new_points
                        best_lines = new_lines
                        current_score = new_score
                        best_score = max(best_score, current_score)
                        point_set = set(map(tuple, best_points.tolist()))
                        no_improve_count = 0
    
    return best_points, best_lines, best_score

def _polish_set(points, lines, deadline):
    """
    Final polish: try all single-point moves with larger radius, accept only improvements.
    """
    current_score = _count_incidence(points, lines)
    best_points = points.copy()
    best_lines = lines.copy()
    best_score = current_score
    
    point_set = set(map(tuple, best_points.tolist()))
    
    # Try multiple passes
    for pass_num in range(3):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(54):
            if time.time() > deadline:
                break
            
            ox, oy = int(best_points[i, 0]), int(best_points[i, 1])
            
            # Try all moves in a larger radius
            best_move = None
            best_move_score = current_score
            
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = ox + dx, oy + dy
                    
                    if (nx, ny) in point_set:
                        continue
                    
                    new_points = best_points.copy()
                    new_points[i, 0] = nx
                    new_points[i, 1] = ny
                    
                    new_lines = _find_best_lines_for_points(new_points, 27)
                    new_score = _count_incidence(new_points, new_lines)
                    
                    if new_score > best_move_score:
                        best_move_score = new_score
                        best_move = (nx, ny)
            
            if best_move is not None:
                nx, ny = best_move
                best_points[i, 0] = nx
                best_points[i, 1] = ny
                point_set.remove((ox, oy))
                point_set.add((nx, ny))
                best_lines = _find_best_lines_for_points(best_points, 27)
                current_score = best_move_score
                best_score = max(best_score, current_score)
                improved = True
        
        if not improved:
            break
    
    return best_points, best_lines, best_score

def szemeredi_trotter_incidences():
    start_time = time.time()
    # Safety margin: return before 620s
    deadline = start_time + 590
    
    best_points = None
    best_lines = None
    best_score = -1
    
    # Phase 1: Evaluate initial candidate sets
    candidate_sets = _generate_point_sets(deadline)
    
    for pts in candidate_sets:
        if time.time() > deadline:
            break
        pts, lines, score = _evaluate_set(pts, deadline)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
            best_lines = lines.copy()
    
    # Phase 1b: Try random perturbations of the best grid found
    if best_points is not None:
        for trial in range(5):
            if time.time() > deadline:
                break
            perturbed = best_points.copy()
            for _ in range(10):
                idx = random.randint(0, 53)
                perturbed[idx, 0] += random.randint(-2, 2)
                perturbed[idx, 1] += random.randint(-2, 2)
            # Ensure no duplicates
            seen_pts = set()
            valid = []
            for p in perturbed:
                key = (int(p[0]), int(p[1]))
                if key not in seen_pts:
                    seen_pts.add(key)
                    valid.append(key)
            if len(valid) == 54:
                perturbed = np.array(valid, dtype=np.int64)
                pts, lines, score = _evaluate_set(perturbed, deadline)
                if score > best_score:
                    best_score = score
                    best_points = pts.copy()
                    best_lines = lines.copy()
    
    # Phase 2: Refinement
    if best_points is not None:
        elapsed = time.time() - start_time
        remaining = deadline - start_time - elapsed
        
        # Allocate time: 55% for refinement, 20% for polish, 25% for second refinement
        refine_budget = int(remaining * 0.55)
        polish_budget = int(remaining * 0.20)
        refine2_budget = int(remaining * 0.25)
        
        refine_deadline = start_time + elapsed + refine_budget
        
        max_steps = min(10000, int(refine_budget * 50))
        if max_steps < 500:
            max_steps = 500
        
        best_points, best_lines, best_score = _refine_set(
            best_points, best_lines, refine_deadline, max_steps
        )
        
        # Phase 3: Polish
        polish_deadline = start_time + elapsed + refine_budget + polish_budget
        best_points, best_lines, best_score = _polish_set(
            best_points, best_lines, polish_deadline
        )
        
        # Phase 4: Second refinement pass with fresh random seed
        if time.time() < deadline - 5:
            random.seed(int(time.time() * 1000) % 100000)
            refine2_deadline = deadline - 5
            max_steps2 = min(5000, int(refine2_budget * 50))
            if max_steps2 < 200:
                max_steps2 = 200
            best_points, best_lines, best_score = _refine_set(
                best_points, best_lines, refine2_deadline, max_steps2
            )
    
    # Final Validation and Cleanup
    if best_points is None or len(best_points) != 54:
        best_points = np.array([[i, j] for i in range(6) for j in range(9)], dtype=np.int64)
        best_lines = _find_best_lines_for_points(best_points, 27)
    
    points = np.array(best_points, dtype=np.int64)
    lines = np.array(best_lines, dtype=np.int64)
    
    if len(points) > 54:
        points = points[:54]
    elif len(points) < 54:
        while len(points) < 54:
            points = np.vstack([points, [len(points)*100, len(points)*200]])
    
    valid_lines = []
    seen = set()
    for l in lines:
        key = _normalize_line(int(l[0]), int(l[1]), int(l[2]))
        if key not in seen and (key[0] != 0 or key[1] != 0):
            seen.add(key)
            valid_lines.append(key)
    
    lines = np.array(valid_lines, dtype=np.int64)
    
    if len(lines) > 27:
        lines = lines[:27]
    elif len(lines) < 27:
        c = 1000
        while len(lines) < 27:
            key = _normalize_line(0, 1, -c)
            if key not in seen:
                seen.add(key)
                lines = np.vstack([lines, np.array(key, dtype=np.int64)])
            c += 1
    
    return points, lines

# EVOLVE-BLOCK-END
