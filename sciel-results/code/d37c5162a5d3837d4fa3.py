# EVOLVE-BLOCK-START
import numpy as np
import time
from math import gcd
from itertools import combinations

def _normalize_line(A, B, C):
    """Normalize a line equation Ax + By + C = 0 to canonical form."""
    if A == 0 and B == 0:
        return None
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return None
    A //= g
    B //= g
    C //= g
    # Make first nonzero entry positive
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def _count_incidence_for_line(A, B, C, points):
    """Count how many points lie on the line Ax + By + C = 0."""
    vals = A * points[:, 0] + B * points[:, 1] + C
    return int(np.sum(vals == 0))

def _get_all_lines_and_counts(points):
    """
    Generate all lines through pairs of points and count incidences.
    Returns:
        lines_dict: { (A,B,C): count }
        point_lines: list of sets of line keys for each point
    """
    n = points.shape[0]
    lines_dict = {}
    point_lines = [set() for _ in range(n)]
    
    # Generate lines from all pairs
    for i in range(n):
        p1 = points[i]
        for j in range(i + 1, n):
            p2 = points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            A = dy
            B = -dx
            C = dx * p1[1] - dy * p1[0]
            key = _normalize_line(A, B, C)
            if key is not None:
                if key not in lines_dict:
                    lines_dict[key] = 0
                point_lines[i].add(key)
                point_lines[j].add(key)
                
    # Count incidences accurately for each line
    for key in lines_dict:
        A, B, C = key
        lines_dict[key] = _count_incidence_for_line(A, B, C, points)
        
    return lines_dict, point_lines

def _top_k_sum(counts, k=27):
    """Return sum of top k values in a dict."""
    if len(counts) < k:
        return sum(counts.values())
    sorted_vals = sorted(counts.values(), reverse=True)
    return sum(sorted_vals[:k])

def _select_top_lines(lines_dict, k=27):
    """Select top k lines by incidence count."""
    sorted_lines = sorted(lines_dict.items(), key=lambda x: (-x[1], x[0]))
    selected = [l for l, c in sorted_lines[:k]]
    if len(selected) < k:
        # Pad with dummy lines that have 0 incidence
        sel_keys = set(selected)
        for key in lines_dict:
            if len(selected) >= k:
                break
            if key not in sel_keys:
                selected.append(key)
        while len(selected) < k:
            selected.append((1, 0, 1000000 + len(selected)))
    return np.array(selected, dtype=np.int64)

def _evaluate_removal(base_points, num_remove, deadline):
    """
    Evaluate removing num_remove points from base_points to maximize top-27 line incidences.
    Uses incremental updates for efficiency.
    """
    n = base_points.shape[0]
    if n < num_remove:
        return -1, None
    
    lines_dict, point_lines = _get_all_lines_and_counts(base_points)
    
    best_score = -1
    best_mask = None
    
    if num_remove == 0:
        score = _top_k_sum(lines_dict)
        if score > best_score:
            best_score = score
            best_mask = np.ones(n, dtype=bool)
        return best_score, base_points[best_mask] if best_mask is not None else None
    
    if num_remove == 1:
        for i in range(n):
            if time.time() > deadline:
                break
            # Create adjusted counts
            temp_counts = dict(lines_dict)
            for key in point_lines[i]:
                temp_counts[key] -= 1
            score = _top_k_sum(temp_counts)
            if score > best_score:
                best_score = score
                best_mask = np.ones(n, dtype=bool)
                best_mask[i] = False
                
    elif num_remove == 2:
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                temp_counts = dict(lines_dict)
                for key in point_lines[i]:
                    temp_counts[key] -= 1
                for key in point_lines[j]:
                    temp_counts[key] -= 1
                score = _top_k_sum(temp_counts)
                if score > best_score:
                    best_score = score
                    best_mask = np.ones(n, dtype=bool)
                    best_mask[i] = False
                    best_mask[j] = False
    
    elif num_remove == 3:
        # Try removing triples - only feasible for small n
        if n <= 60:
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        break
                    for k in range(j + 1, n):
                        if time.time() > deadline:
                            break
                        temp_counts = dict(lines_dict)
                        for key in point_lines[i]:
                            temp_counts[key] -= 1
                        for key in point_lines[j]:
                            temp_counts[key] -= 1
                        for key in point_lines[k]:
                            temp_counts[key] -= 1
                        score = _top_k_sum(temp_counts)
                        if score > best_score:
                            best_score = score
                            best_mask = np.ones(n, dtype=bool)
                            best_mask[i] = False
                            best_mask[j] = False
                            best_mask[k] = False
    
    if best_mask is None:
        return -1, None
    return best_score, base_points[best_mask]

def _generate_grid(a, b):
    """Generate an a x b grid of points."""
    xs = np.arange(1, a + 1)
    ys = np.arange(1, b + 1)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    return np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

def _try_grid_with_removal(a, b, num_remove, deadline):
    """Try a grid of size a*b, removing num_remove points to get 54."""
    if a * b != 54 + num_remove:
        return None, None
    pts_base = _generate_grid(a, b)
    score, pts_best = _evaluate_removal(pts_base, num_remove, deadline)
    if pts_best is not None and score > 0:
        lines_dict, _ = _get_all_lines_and_counts(pts_best)
        lines_arr = _select_top_lines(lines_dict)
        return pts_best, lines_arr
    return None, None

def _try_exact_grid(a, b, deadline):
    """Try an exact a*b = 54 grid."""
    if a * b != 54:
        return None, None
    pts = _generate_grid(a, b)
    lines_dict, _ = _get_all_lines_and_counts(pts)
    lines_arr = _select_top_lines(lines_dict)
    return pts, lines_arr

def _smooth_incidence_score(points, lines, sigma=0.5):
    """
    Compute a smooth approximation of the incidence count using Gaussian kernels.
    For each point-line pair, compute exp(-(Ax+By+C)^2 / (2*sigma^2)).
    """
    n_pts = points.shape[0]
    n_lines = lines.shape[0]
    score = 0.0
    for i in range(n_pts):
        x, y = points[i]
        for j in range(n_lines):
            A, B, C = lines[j]
            val = A * x + B * y + C
            score += np.exp(-(val ** 2) / (2 * sigma ** 2))
    return score

def _continuous_optimize(initial_points, initial_lines, deadline, max_iter=500, lr=0.1):
    """
    Use gradient ascent on a smoothed incidence objective to find better configurations.
    Points and line coefficients are treated as real-valued variables.
    """
    n_pts = 54
    n_lines = 27
    sigma = 0.5
    
    # Start from integer configurations
    pts = initial_points.astype(np.float64).copy()
    lines = initial_lines.astype(np.float64).copy()
    
    best_score = -np.inf
    best_pts = pts.copy()
    best_lines = lines.copy()
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        # Compute smooth score
        current_score = _smooth_incidence_score(pts, lines, sigma)
        
        if current_score > best_score:
            best_score = current_score
            best_pts = pts.copy()
            best_lines = lines.copy()
        
        # Compute gradients with respect to points
        grad_pts = np.zeros_like(pts)
        for i in range(n_pts):
            for j in range(n_lines):
                A, B, C = lines[j]
                val = A * pts[i, 0] + B * pts[i, 1] + C
                kernel = np.exp(-(val ** 2) / (2 * sigma ** 2))
                # d/dx of exp(-(Ax+By+C)^2/(2*sigma^2)) = -A*val/sigma^2 * kernel
                grad_pts[i, 0] += -A * val / (sigma ** 2) * kernel
                grad_pts[i, 1] += -B * val / (sigma ** 2) * kernel
        
        # Compute gradients with respect to line coefficients
        grad_lines = np.zeros_like(lines)
        for i in range(n_pts):
            x, y = pts[i]
            for j in range(n_lines):
                A, B, C = lines[j]
                val = A * x + B * y + C
                kernel = np.exp(-(val ** 2) / (2 * sigma ** 2))
                grad_lines[j, 0] += -x * val / (sigma ** 2) * kernel
                grad_lines[j, 1] += -y * val / (sigma ** 2) * kernel
                grad_lines[j, 2] += -val / (sigma ** 2) * kernel
        
        # Gradient ascent step
        pts += lr * grad_pts
        lines += lr * grad_lines
        
        # Keep points in reasonable range
        pts = np.clip(pts, 0.5, 100.5)
        
        # Normalize lines to prevent drift
        for j in range(n_lines):
            norm = np.sqrt(lines[j, 0]**2 + lines[j, 1]**2 + lines[j, 2]**2)
            if norm > 1e-10:
                lines[j] /= norm
                # Keep A positive if possible
                if lines[j, 0] < 0:
                    lines[j] *= -1
                elif lines[j, 0] == 0 and lines[j, 1] < 0:
                    lines[j] *= -1
    
    return best_pts, best_lines, best_score

def _round_and_validate(pts_float, lines_float):
    """
    Round continuous solutions to integers and validate.
    Returns valid (points, lines) or None if invalid.
    """
    # Round points
    pts_int = np.round(pts_float).astype(np.int64)
    
    # Check for duplicate points
    for i in range(54):
        for j in range(i + 1, 54):
            if pts_int[i, 0] == pts_int[j, 0] and pts_int[i, 1] == pts_int[j, 1]:
                return None
    
    # Round and normalize lines
    lines_int = []
    seen_lines = set()
    for j in range(27):
        A = int(round(lines_float[j, 0]))
        B = int(round(lines_float[j, 1]))
        C = int(round(lines_float[j, 2]))
        
        if A == 0 and B == 0:
            # Try to find a better line
            A, B, C = 1, 0, 1000000 + j
        else:
            key = _normalize_line(A, B, C)
            if key is None:
                A, B, C = 1, 0, 1000000 + j
            else:
                A, B, C = key
        
        if (A, B, C) not in seen_lines:
            lines_int.append((A, B, C))
            seen_lines.add((A, B, C))
        else:
            # Find a different line
            for c in range(-100, 101):
                new_key = _normalize_line(1, 0, c)
                if new_key and new_key not in seen_lines:
                    lines_int.append(new_key)
                    seen_lines.add(new_key)
                    break
            else:
                for c in range(-100, 101):
                    new_key = _normalize_line(0, 1, c)
                    if new_key and new_key not in seen_lines:
                        lines_int.append(new_key)
                        seen_lines.add(new_key)
                        break
                else:
                    lines_int.append((1, 0, 1000000 + len(lines_int)))
                    seen_lines.add((1, 0, 1000000 + len(lines_int)))
    
    if len(lines_int) < 27:
        return None
    
    lines_arr = np.array(lines_int[:27], dtype=np.int64)
    
    # Verify exact incidence
    I = 0
    for i in range(54):
        for j in range(27):
            if lines_arr[j, 0] * pts_int[i, 0] + lines_arr[j, 1] * pts_int[i, 1] + lines_arr[j, 2] == 0:
                I += 1
    
    return pts_int, lines_arr, I

def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences for 54 points and 27 lines.
    Uses systematic search over grid configurations with point removals,
    plus continuous geometric optimization with Gaussian kernel smoothing.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    best_score = -1
    best_points = None
    best_lines = None
    
    def update_best(pts, lines, score):
        nonlocal best_score, best_points, best_lines
        if score > best_score:
            best_score = score
            best_points = pts
            best_lines = lines
    
    # Phase 1: Try exact grids (a*b = 54)
    exact_grids = [
        (6, 9), (9, 6), (3, 18), (18, 3), (2, 27), (27, 2), (1, 54), (54, 1)
    ]
    for a, b in exact_grids:
        if time.time() > deadline - 60:
            break
        pts, lines = _try_exact_grid(a, b, deadline)
        if pts is not None:
            # Compute actual score
            I = 0
            for i in range(54):
                for j in range(27):
                    if lines[j, 0] * pts[i, 0] + lines[j, 1] * pts[i, 1] + lines[j, 2] == 0:
                        I += 1
            update_best(pts, lines, I)
    
    # Phase 2: Try near-rectangular grids with removals
    near_grids = [
        (7, 8, 2), (8, 7, 2), (5, 11, 1), (11, 5, 1),
        (4, 14, 2), (14, 4, 2), (3, 19, 3), (19, 3, 3),
        (2, 28, 2), (28, 2, 2)
    ]
    for a, b, rem in near_grids:
        if time.time() > deadline - 60:
            break
        pts, lines = _try_grid_with_removal(a, b, rem, deadline)
        if pts is not None:
            I = 0
            for i in range(54):
                for j in range(27):
                    if lines[j, 0] * pts[i, 0] + lines[j, 1] * pts[i, 1] + lines[j, 2] == 0:
                        I += 1
            update_best(pts, lines, I)
    
    # Phase 3: Continuous geometric optimization
    if best_points is not None and time.time() < deadline - 120:
        # Start from the best grid configuration
        init_pts = best_points.astype(np.float64)
        init_lines = best_lines.astype(np.float64)
        
        # Try multiple initializations with different sigmas
        for sigma in [0.3, 0.5, 0.8, 1.0]:
            if time.time() > deadline - 30:
                break
            
            # Run continuous optimization
            opt_pts, opt_lines, smooth_score = _continuous_optimize(
                init_pts, init_lines, deadline - 30, max_iter=300, lr=0.05
            )
            
            # Round and validate
            result = _round_and_validate(opt_pts, opt_lines)
            if result is not None:
                pts_int, lines_int, I = result
                update_best(pts_int, lines_int, I)
                
                # Try to improve further from this rounded solution
                if time.time() < deadline - 60:
                    init_pts2 = pts_int.astype(np.float64)
                    init_lines2 = lines_int.astype(np.float64)
                    opt_pts2, opt_lines2, _ = _continuous_optimize(
                        init_pts2, init_lines2, deadline - 60, max_iter=200, lr=0.03
                    )
                    result2 = _round_and_validate(opt_pts2, opt_lines2)
                    if result2 is not None:
                        pts_int2, lines_int2, I2 = result2
                        update_best(pts_int2, lines_int2, I2)
    
    # Phase 4: Local refinement - try moving individual points
    if best_points is not None and time.time() < deadline - 60:
        pts = best_points.copy()
        current_score = best_score
        improved = True
        while improved and time.time() < deadline - 30:
            improved = False
            lines_dict, _ = _get_all_lines_and_counts(pts)
            
            for idx in range(54):
                if time.time() > deadline - 30:
                    break
                x0, y0 = pts[idx]
                # Try nearby positions
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if dx == 0 and dy == 0:
                            continue
                        new_x, new_y = x0 + dx, y0 + dy
                        if new_x < 1 or new_y < 1:
                            continue
                        if np.any((pts[:, 0] == new_x) & (pts[:, 1] == new_y)):
                            continue
                        
                        # Try this move
                        old_pt = pts[idx].copy()
                        pts[idx] = [new_x, new_y]
                        lines_dict_new, _ = _get_all_lines_and_counts(pts)
                        score_new = _top_k_sum(lines_dict_new)
                        
                        if score_new > current_score:
                            current_score = score_new
                            improved = True
                            update_best(pts, _select_top_lines(lines_dict_new), score_new)
                        else:
                            pts[idx] = old_pt
                        
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
    
    # Final validation and return
    if best_points is None:
        # Fallback to 6x9 grid
        xs = np.arange(1, 7)
        ys = np.arange(1, 10)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        best_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines_list = []
        for c in range(1, 7):
            lines_list.append((1, 0, -c))
        for c in range(1, 10):
            lines_list.append((0, 1, -c))
        for c in [0, 1, 2, 3]:
            lines_list.append((1, -1, c))
        for c in [7, 8, 9, 10]:
            lines_list.append((1, 1, -c))
        for c in [4, -1]:
            lines_list.append((1, -1, c))
        for c in [6, 11]:
            lines_list.append((1, 1, -c))
        best_lines = np.array(lines_list, dtype=np.int64)
    else:
        # Re-select top lines for the final point set
        lines_dict, _ = _get_all_lines_and_counts(best_points)
        best_lines = _select_top_lines(lines_dict)
    
    return best_points, best_lines

# EVOLVE-BLOCK-END
