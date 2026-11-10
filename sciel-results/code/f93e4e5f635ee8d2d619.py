import numpy as np
import time
import math
from itertools import combinations
from math import gcd

def _normalize_line(A, B, C):
    """Normalize a line (A, B, C) by dividing by gcd and making first nonzero positive."""
    A, B, C = int(A), int(B), int(C)
    if A == 0 and B == 0:
        return None
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g > 1:
        A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidences(points, lines):
    """Count exact integer incidences between points and lines."""
    pts = np.asarray(points, dtype=np.int64)
    lns = np.asarray(lines, dtype=np.int64)
    total = 0
    for i in range(len(lns)):
        A, B, C = lns[i]
        val = A * pts[:, 0] + B * pts[:, 1] + C
        total += int(np.sum(val == 0))
    return total

def _is_valid(points, lines):
    """Validate points and lines per the task contract."""
    pts = np.asarray(points, dtype=np.int64)
    lns = np.asarray(lines, dtype=np.int64)
    if pts.shape != (54, 2) or lns.shape != (27, 3):
        return False
    if not np.all(np.isfinite(pts.astype(np.float64))):
        return False
    if not np.all(np.isfinite(lns.astype(np.float64))):
        return False
    seen_pts = set()
    for p in pts:
        t = (int(p[0]), int(p[1]))
        if t in seen_pts:
            return False
        seen_pts.add(t)
    seen_lines = set()
    for l in lns:
        A, B, C = int(l[0]), int(l[1]), int(l[2])
        if A == 0 and B == 0:
            return False
        norm = _normalize_line(A, B, C)
        if norm is None:
            return False
        if norm in seen_lines:
            return False
        seen_lines.add(norm)
    return True

def _build_optimal_grid_6x9():
    """
    Builds a 6x9 grid and selects the 27 lines with the highest number of incidences.
    """
    xs = np.arange(6)
    ys = np.arange(9)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # Find all unique lines defined by pairs of points
    unique_lines = set()
    n_pts = len(points)
    
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            
            norm = _normalize_line(A, B, C)
            if norm is not None:
                unique_lines.add(norm)
    
    # Count incidences for each unique line
    line_incidences = []
    for A, B, C in unique_lines:
        val = A * points[:, 0] + B * points[:, 1] + C
        cnt = int(np.sum(val == 0))
        line_incidences.append((cnt, (A, B, C)))
        
    # Sort by incidence count descending
    line_incidences.sort(key=lambda x: x[0], reverse=True)
    
    # Pick top 27
    top_lines = [ln for _, ln in line_incidences[:27]]
    lines_arr = np.array(top_lines, dtype=np.int64)
    
    return points, lines_arr

def _local_search(points, lines, deadline):
    """
    Perform local search to improve the (points, lines) configuration.
    Try swapping individual points with nearby candidates, and replacing lines with better ones.
    """
    pts = np.array(points, dtype=np.int64, copy=True)
    lns = np.array(lines, dtype=np.int64, copy=True)
    current_score = _count_incidences(pts, lns)
    
    # Generate candidate points in a larger grid
    min_x, max_x = min(pts[:, 0]), max(pts[:, 0])
    min_y, max_y = min(pts[:, 1]), max(pts[:, 1])
    
    # Expand search region
    expand = 5
    all_candidates = set()
    for x in range(min_x - expand, max_x + expand + 1):
        for y in range(min_y - expand, max_y + expand + 1):
            all_candidates.add((x, y))
    
    # Remove current points from candidates
    for p in pts:
        all_candidates.discard((int(p[0]), int(p[1])))
    
    candidate_list = list(all_candidates)
    
    improved = True
    while improved and time.time() < deadline - 30:
        improved = False
        
        # Try swapping each point with each candidate
        for i in range(len(pts)):
            if time.time() >= deadline - 30:
                break
            orig_p = pts[i].copy()
            
            # Sort candidates by distance to original point for efficiency
            dists = [(np.sum((np.array(c) - orig_p) ** 2), c) for c in candidate_list]
            dists.sort()
            
            # Try closest candidates first
            for dist, cand in dists[:20]:  # Try top 20 closest
                if time.time() >= deadline - 30:
                    break
                new_pts = pts.copy()
                new_pts[i] = cand
                new_score = _count_incidences(new_pts, lns)
                
                if new_score > current_score:
                    pts = new_pts
                    candidate_list.remove(cand)
                    candidate_list.append(tuple(orig_p))
                    current_score = new_score
                    improved = True
                    break
        
        # Try replacing each line with a better one
        for i in range(len(lns)):
            if time.time() >= deadline - 30:
                break
            
            # Find all lines defined by pairs of current points
            unique_lines = set()
            for a in range(len(pts)):
                for b in range(a + 1, len(pts)):
                    x1, y1 = pts[a]
                    x2, y2 = pts[b]
                    A = y1 - y2
                    B = x2 - x1
                    C = x1 * y2 - x2 * y1
                    norm = _normalize_line(A, B, C)
                    if norm is not None:
                        unique_lines.add(norm)
            
            # Remove current lines from candidates
            for l in lns:
                norm = _normalize_line(int(l[0]), int(l[1]), int(l[2]))
                if norm is not None:
                    unique_lines.discard(norm)
            
            # Score remaining lines
            line_scores = []
            for A, B, C in unique_lines:
                val = A * pts[:, 0] + B * pts[:, 1] + C
                cnt = int(np.sum(val == 0))
                line_scores.append((cnt, (A, B, C)))
            
            line_scores.sort(key=lambda x: x[0], reverse=True)
            
            # Try replacing with best line
            if line_scores:
                best_cnt, best_line = line_scores[0]
                new_ln = lns.copy()
                new_ln[i] = best_line
                new_score = _count_incidences(pts, new_ln)
                
                if new_score > current_score:
                    lns = new_ln
                    current_score = new_score
                    improved = True
    
    return pts, lns, current_score

def szemeredi_trotter_incidences():
    deadline = time.time() + 580.0
    
    # The 6x9 grid with optimal line selection is the strongest candidate.
    try:
        pts, lns = _build_optimal_grid_6x9()
        if _is_valid(pts, lns):
            # Apply local search to improve
            if time.time() < deadline - 60:
                pts_opt, lns_opt, score = _local_search(pts, lns, deadline)
                if _is_valid(pts_opt, lns_opt):
                    return pts_opt, lns_opt
            return pts, lns
    except Exception:
        pass
        
    # Fallback to a simple construction if optimal fails
    xs = np.arange(1, 4)
    ys = np.arange(1, 19)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    lines = []
    for a in range(1, 4):
        for b in range(1, 10):
            lines.append((a, -1, b))
    lines_arr = np.array(lines, dtype=np.int64)
    
    return points, lines_arr
