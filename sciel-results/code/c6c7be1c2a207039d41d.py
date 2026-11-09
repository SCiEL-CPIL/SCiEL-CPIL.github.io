# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from math import gcd
from itertools import product

def _count_incidences(points, lines):
    """Count point-line incidences exactly using int64 arithmetic."""
    n = len(points)
    m = len(lines)
    count = 0
    for i in range(n):
        x, y = points[i]
        for j in range(m):
            A, B, C = lines[j]
            if A * x + B * y + C == 0:
                count += 1
    return count

def _normalize_line(A, B, C):
    """Normalize line (A, B, C) so gcd(|A|,|B|,|C|) = 1 and first nonzero is positive."""
    # Handle zeros in gcd
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g > 0:
        A //= g
        B //= g
        C //= g
    # Make first nonzero entry positive
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def _is_valid(points, lines):
    """Check if the configuration is valid."""
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    
    # Check points are integers
    if not np.all(np.isclose(points, np.round(points), atol=1e-6)):
        return False
    points_int = np.round(points).astype(np.int64)
    
    # Check no repeated points
    if len(set(map(tuple, points_int.tolist()))) != 54:
        return False
    
    # Check lines
    line_set = set()
    for i in range(27):
        A, B, C = lines[i]
        if not all(np.isclose(v, np.round(v), atol=1e-6) for v in [A, B, C]):
            return False
        A, B, C = int(np.round(A)), int(np.round(B)), int(np.round(C))
        if A == 0 and B == 0:
            return False
        norm = _normalize_line(A, B, C)
        if norm in line_set:
            return False
        line_set.add(norm)
    
    return True

def _make_grid(cols, rows):
    """Create a grid of points and lines."""
    xs = np.arange(1, cols + 1)
    ys = np.arange(1, rows + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    lines = []
    # Vertical lines
    for x in range(1, cols + 1):
        lines.append((1, 0, -x))
    # Horizontal lines
    for y in range(1, rows + 1):
        lines.append((0, 1, -y))
    # Diagonals (slope 1)
    # y = x + d => x - y + d = 0
    # d ranges from -(cols-1) to (rows-1)
    for d in range(-(cols - 1), rows):
        # Check if any point lies on this line
        # y = x + d
        # 1 <= y <= rows, 1 <= x <= cols
        # 1 <= x + d <= rows => 1-d <= x <= rows-d
        # x must be in [1, cols]
        x_min = max(1, 1 - d)
        x_max = min(cols, rows - d)
        if x_min <= x_max:
            lines.append((1, -1, d))
            
    # We need exactly 27 lines. 
    # Grid cols*rows = 54.
    # If cols=6, rows=9: 6 vert + 9 horiz = 15. Need 12 diagonals.
    # If cols=9, rows=6: 9 vert + 6 horiz = 15. Need 12 diagonals.
    # If cols=3, rows=18: 3 vert + 18 horiz = 21. Need 6 diagonals.
    # If cols=18, rows=3: 18 vert + 3 horiz = 21. Need 6 diagonals.
    # If cols=2, rows=27: 2 vert + 27 horiz = 29 > 27. Trim?
    # If cols=27, rows=2: 27 vert + 2 horiz = 29 > 27. Trim?
    # If cols=1, rows=54: 1 vert + 54 horiz = 55 > 27.
    
    # Prioritize lines with more incidences?
    # For a grid, vertical/horizontal have cols/rows incidences.
    # Diagonals have varying incidences.
    # Let's just take the first 27.
    if len(lines) > 27:
        lines = lines[:27]
    elif len(lines) < 27:
        # Add arbitrary lines? Or just return invalid if we can't fill?
        # Actually, we can add lines passing through pairs of points if needed, 
        # but for a grid, we usually have plenty of candidate lines.
        # If we have too few, we might need to add "noise" lines or accept fewer incidences.
        # But the contract requires 27 lines.
        # Let's add lines passing through random pairs of points if we are short.
        while len(lines) < 27:
            p1 = random.choice(points)
            p2 = random.choice(points)
            if p1[0] == p2[0] and p1[1] == p2[1]:
                continue
            A = p2[1] - p1[1]
            B = p1[0] - p2[0]
            C = -(A * p1[0] + B * p1[1])
            norm = _normalize_line(A, B, C)
            if norm not in [tuple(l) for l in lines]:
                lines.append(norm)
            
    lines = np.array(lines, dtype=np.int64)
    return points, lines

def _local_search_fast(points, lines, deadline):
    """
    Fast local search using incidence matrix updates.
    """
    points = points.copy()
    lines = lines.copy()
    n_pts = len(points)
    n_lines = len(lines)
    
    # Initial incidence matrix
    # M[i, j] = 1 if points[i] is on lines[j]
    M = np.zeros((n_pts, n_lines), dtype=np.int8)
    for i in range(n_pts):
        x, y = points[i]
        for j in range(n_lines):
            A, B, C = lines[j]
            if A * x + B * y + C == 0:
                M[i, j] = 1
                
    current_score = np.sum(M)
    best_score = current_score
    best_points = points.copy()
    best_lines = lines.copy()
    
    # Precompute line coefficients for speed
    A = lines[:, 0]
    B = lines[:, 1]
    C = lines[:, 2]
    
    # Helper to check if line is valid (not duplicate, A!=0 or B!=0)
    def is_line_valid_and_unique(line_tuple, current_lines):
        Al, Bl, Cl = line_tuple
        if Al == 0 and Bl == 0:
            return False
        norm = _normalize_line(Al, Bl, Cl)
        for l in current_lines:
            if _normalize_line(l[0], l[1], l[2]) == norm:
                return False
        return True

    while time.time() < deadline:
        improved = False
        
        # 1. Try moving points
        # For each point, try moving to neighbors
        # To speed up, we can sample a subset of points if too slow, but 54 is small.
        for i in range(n_pts):
            if time.time() >= deadline:
                break
            
            orig_x, orig_y = points[i]
            
            # Try a set of neighbor offsets
            # We want to find the best move for this point.
            best_delta = 0
            best_new_pos = None
            
            # Search range
            # Points are usually close in grid. Range 5 is enough.
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = orig_x + dx, orig_y + dy
                    
                    # Check duplicate point
                    # Simple check: is (nx, ny) in points (excluding i)?
                    # Since N is small, linear scan is okay, or use a set.
                    # Let's use a set of points for O(1) lookup.
                    # But updating the set every time is overhead.
                    # Given 54 points, linear scan in Python is ~54 ops.
                    # 54 points * 121 neighbors * 54 checks = ~350k ops per point.
                    # Total ~19M ops. Might be slow in Python.
                    # Let's restrict neighbors to a smaller set or use numpy.
                    
                    # Actually, let's just evaluate the delta for incidence.
                    # Delta = sum_j (M_new[i,j] - M_old[i,j])
                    # M_old[i,j] is known.
                    # M_new[i,j] = 1 if A[j]*nx + B[j]*ny + C[j] == 0
                    
                    # We can vectorize this for a fixed point i.
                    pass # We will do this differently below for speed.
            
            # Vectorized evaluation for point i
            # Current incidences for point i: M[i, :]
            # Try all candidate positions in a box around orig.
            # Candidates: (orig_x + dx, orig_y + dy) for dx, dy in [-5, 5]
            # We can generate a list of candidates, filter duplicates, and compute incidences.
            
            candidates = []
            seen = set()
            # Add original position to seen to skip it
            seen.add((orig_x, orig_y))
            
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = orig_x + dx, orig_y + dy
                    if (nx, ny) not in seen:
                        # Check if this position is occupied by another point
                        # This check is expensive if done naively.
                        # But we can defer it? No, invalid state.
                        # Let's check against all other points.
                        # points[0:i] + points[i+1:]
                        # If (nx, ny) is in that set, skip.
                        # Using numpy:
                        # other_pts = np.vstack([points[:i], points[i+1:]])
                        # if np.any((other_pts[:,0] == nx) & (other_pts[:,1] == ny)): skip
                        # This is O(N) per candidate.
                        
                        # Optimization: Precompute a set of all points?
                        # But points change.
                        # Let's just do the numpy check, it's fast enough for 54 points.
                        other_pts = np.vstack([points[:i], points[i+1:]]) if i > 0 and i < n_pts - 1 else (points[:i] if i < n_pts - 1 else points[i+1:])
                        if len(other_pts) > 0:
                            if np.any((other_pts[:,0] == nx) & (other_pts[:,1] == ny)):
                                continue
                        
                        candidates.append((nx, ny))
                        seen.add((nx, ny))
            
            if not candidates:
                continue
            
            # Compute incidences for all candidates at once
            # candidates is a list of (x, y).
            # cand_x = np.array([c[0] for c in candidates])
            # cand_y = np.array([c[1] for c in candidates])
            # Incidences: (cand_x[:, None] * A[None, :] + cand_y[:, None] * B[None, :] + C[None, :]) == 0
            # Shape: (num_candidates, n_lines)
            
            cand_x = np.array([c[0] for c in candidates])
            cand_y = np.array([c[1] for c in candidates])
            
            # Compute new M row for each candidate
            # val = cand_x[:, None] * A[None, :] + cand_y[:, None] * B[None, :] + C[None, :]
            # new_M = (val == 0).astype(np.int8)
            # scores = np.sum(new_M, axis=1)
            # old_score = np.sum(M[i, :])
            # deltas = scores - old_score
            
            # To save memory/time, we can just compute the sum directly.
            # But we need to find the max delta.
            
            # Vectorized calculation
            vals = cand_x[:, None] * A[None, :] + cand_y[:, None] * B[None, :] + C[None, :]
            new_M = (vals == 0).astype(np.int8)
            scores = np.sum(new_M, axis=1)
            old_score = np.sum(M[i, :])
            
            max_score = np.max(scores)
            if max_score > old_score:
                idx = np.argmax(scores)
                nx, ny = candidates[idx]
                
                # Apply move
                points[i] = [nx, ny]
                M[i, :] = new_M[idx, :]
                current_score += (max_score - old_score)
                improved = True
                
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
                    best_lines = lines.copy()
                    
            # If no improvement for this point, we leave it as is.
            # We don't "try" a bad move and revert. We just don't move.
            # This is "Best Improvement" per point.
            
        # 2. Try changing lines
        # For each line, try replacing it with a line passing through 2 random points?
        # Or try shifting the line?
        # Changing a line is more complex to vectorize against all points.
        # Let's try: For line j, calculate its current score (sum of M[:, j]).
        # Generate candidate lines:
        # a) Lines passing through pairs of points (sample a few).
        # b) Lines parallel to existing lines, shifted.
        
        for j in range(n_lines):
            if time.time() >= deadline:
                break
            
            old_score_line = np.sum(M[:, j])
            
            # Candidate 1: Line through two random points
            # Sample 10 pairs
            best_new_score = -1
            best_new_line = None
            best_new_M_col = None
            
            for _ in range(20): # 20 random pairs
                p1_idx = random.randint(0, n_pts - 1)
                p2_idx = random.randint(0, n_pts - 1)
                if p1_idx == p2_idx:
                    continue
                x1, y1 = points[p1_idx]
                x2, y2 = points[p2_idx]
                
                Al = y2 - y1
                Bl = x1 - x2
                Cl = -(Al * x1 + Bl * y1)
                
                if Al == 0 and Bl == 0:
                    continue
                
                # Check if this line is the same as any existing line (except j)
                # And check if it's valid
                norm = _normalize_line(Al, Bl, Cl)
                
                # Check duplicate with other lines
                dup = False
                for k in range(n_lines):
                    if k == j:
                        continue
                    if _normalize_line(lines[k,0], lines[k,1], lines[k,2]) == norm:
                        dup = True
                        break
                if dup:
                    continue
                
                # Calculate incidences for this new line
                # val = points[:, 0] * Al + points[:, 1] * Bl + Cl
                # new_M_col = (val == 0).astype(np.int8)
                # score = np.sum(new_M_col)
                
                val = points[:, 0] * Al + points[:, 1] * Bl + Cl
                new_M_col = (val == 0).astype(np.int8)
                score = np.sum(new_M_col)
                
                if score > best_new_score:
                    best_new_score = score
                    best_new_line = norm
                    best_new_M_col = new_M_col
            
            if best_new_line is not None and best_new_score > old_score_line:
                # Apply change
                lines[j] = best_new_line
                M[:, j] = best_new_M_col
                current_score += (best_new_score - old_score_line)
                improved = True
                
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
                    best_lines = lines.copy()
                    
        # If no improvement in this entire pass, we are stuck.
        # We can try a random perturbation to escape local minimum.
        if not improved:
            # Perturb a random point significantly
            i = random.randint(0, n_pts - 1)
            orig_x, orig_y = points[i]
            dx = random.randint(-10, 10)
            dy = random.randint(-10, 10)
            nx, ny = orig_x + dx, orig_y + dy
            
            # Check duplicate
            other_pts = np.vstack([points[:i], points[i+1:]]) if i > 0 and i < n_pts - 1 else (points[:i] if i < n_pts - 1 else points[i+1:])
            if len(other_pts) > 0:
                if np.any((other_pts[:,0] == nx) & (other_pts[:,1] == ny)):
                    continue # Skip perturbation if invalid
            
            # Calculate new M row
            val = nx * A + ny * B + C
            new_M_row = (val == 0).astype(np.int8)
            
            # Accept if score doesn't drop too much (Simulated Annealing step)
            # Or just accept if it's better. Since we are stuck, maybe accept equal?
            # Let's just accept if it's better or equal to encourage exploration.
            new_score = np.sum(new_M_row)
            old_score = np.sum(M[i, :])
            
            if new_score >= old_score:
                points[i] = [nx, ny]
                M[i, :] = new_M_row
                current_score += (new_score - old_score)
                
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
                    best_lines = lines.copy()
            else:
                # Accept with probability exp(-delta/T) ? 
                # For simplicity, let's just reject if worse, but we are stuck.
                # Maybe we should restart from best_points?
                # Let's break and restart if stuck for a while?
                # For now, just break the loop and restart from best.
                pass
                
            # If we just broke without improving, we might be in a local min.
            # Let's restart from the best configuration found so far.
            # This is a simple restart strategy.
            points = best_points.copy()
            lines = best_lines.copy()
            # Recalculate M
            M = np.zeros((n_pts, n_lines), dtype=np.int8)
            for i in range(n_pts):
                x, y = points[i]
                for j in range(n_lines):
                    A, B, C = lines[j]
                    if A * x + B * y + C == 0:
                        M[i, j] = 1
            current_score = np.sum(M)
            
            # Update A, B, C arrays
            A = lines[:, 0]
            B = lines[:, 1]
            C = lines[:, 2]
            
            # Restart the while loop? 
            # To avoid infinite loops of restarting, we can add a counter.
            # But for now, let's just continue the outer loop which will check time.
            # Actually, if we restart, we should break out of the "pass" logic and start a new pass.
            # The while loop will continue.
            continue # Next iteration of while loop (which is a new pass)

    return best_points, best_lines, best_score

def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization.
    """
    deadline = time.time() + 580  # Leave 20s safety margin
    
    # Try different grid constructions
    best_points = None
    best_lines = None
    best_count = 0
    
    # Aspect ratios for 54 points
    # 54 = 1*54, 2*27, 3*18, 6*9, 9*6, 18*3, 27*2, 54*1
    # 6x9 and 9x6 are most "square"
    # 3x18 and 18x3 are rectangular
    
    configs = [
        (6, 9),
        (9, 6),
        (3, 18),
        (18, 3),
        (2, 27), # Might have too many lines? 2 vert + 27 horiz = 29. We trim to 27.
        (27, 2),
        (1, 54), # 1 vert + 54 horiz = 55. Trim to 27.
        (54, 1),
    ]
    
    for cols, rows in configs:
        if time.time() > deadline:
            break
        
        pts, lns = _make_grid(cols, rows)
        if _is_valid(pts, lns):
            # Run local search
            deadline_ls = deadline - 5 # Leave 5s for final checks
            if time.time() < deadline_ls:
                pts, lns, cnt = _local_search_fast(pts, lns, deadline_ls)
                if cnt > best_count:
                    best_count = cnt
                    best_points = pts
                    best_lines = lns
            else:
                cnt = _count_incidences(pts, lns)
                if cnt > best_count:
                    best_count = cnt
                    best_points = pts
                    best_lines = lns

    # If we didn't find anything (shouldn't happen), use baseline
    if best_points is None:
        best_points, best_lines = _make_grid(6, 9)
        
    # Ensure valid output
    best_points = np.round(best_points).astype(np.int64)
    best_lines = np.round(best_lines).astype(np.int64)
    
    return best_points, best_lines

# EVOLVE-BLOCK-END
