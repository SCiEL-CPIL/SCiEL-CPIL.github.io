# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
from math import gcd
import time


def _normalize_line(A, B, C):
    """Normalize line (A, B, C) so gcd(|A|,|B|,|C|)=1 and first nonzero is positive."""
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        return (0, 0, 0)
    A //= g
    B //= g
    C //= g
    if A < 0 or (A == 0 and (B < 0 or (B == 0 and C < 0))):
        A, B, C = -A, -B, -C
    return (A, B, C)


def _line_from_two_points(p1, p2):
    """Get normalized line through two points."""
    x1, y1 = p1
    x2, y2 = p2
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    return _normalize_line(A, B, C)


def _count_incidence(points, lines):
    """Count total point-line incidences."""
    count = 0
    for p in points:
        for l in lines:
            if l[0] * p[0] + l[1] * p[1] + l[2] == 0:
                count += 1
    return count


def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 600.0  # 600s limit, use 590s to be safe
    
    # Start with the grid construction as baseline
    k = 3
    xs = np.arange(1, k + 1)
    ys = np.arange(1, 2 * k * k + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    base_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    rows = []
    for a in range(1, k + 1):
        for b in range(1, k * k + 1):
            rows.append((a, -1, b))
    base_lines = np.array(rows, dtype=np.int64)
    
    best_points = base_points.copy()
    best_lines = base_lines.copy()
    best_score = _count_incidence(best_points.tolist(), best_lines.tolist())
    
    # Try 6x9 grid
    if time.time() < deadline - 30:
        xs69 = np.arange(1, 7)
        ys69 = np.arange(1, 10)
        px69, py69 = np.meshgrid(xs69, ys69, indexing="ij")
        grid_points = np.stack([px69.ravel(), py69.ravel()], axis=1).astype(np.int64)
        
        # Find all lines through pairs of points in the grid
        point_list = grid_points.tolist()
        line_set = set()
        for i in range(len(point_list)):
            for j in range(i + 1, len(point_list)):
                ln = _line_from_two_points(point_list[i], point_list[j])
                line_set.add(ln)
        
        # Count incidences for each line
        line_incidents = {}
        for ln in line_set:
            A, B, C = ln
            cnt = 0
            for p in point_list:
                if A * p[0] + B * p[1] + C == 0:
                    cnt += 1
            line_incidents[ln] = cnt
        
        # Sort lines by number of incidences (descending)
        sorted_lines = sorted(line_incidents.keys(), key=lambda x: -line_incidents[x])
        
        # Take top 27 lines
        top_27 = sorted_lines[:27]
        grid_lines = np.array(top_27, dtype=np.int64)
        
        score69 = _count_incidence(point_list, grid_lines.tolist())
        if score69 > best_score:
            best_points = grid_points.copy()
            best_lines = grid_lines.copy()
            best_score = score69
    
    # Try 9x6 grid (transpose)
    if time.time() < deadline - 30:
        xs96 = np.arange(1, 10)
        ys96 = np.arange(1, 7)
        px96, py96 = np.meshgrid(xs96, ys96, indexing="ij")
        grid_points96 = np.stack([px96.ravel(), py96.ravel()], axis=1).astype(np.int64)
        
        point_list96 = grid_points96.tolist()
        line_set96 = set()
        for i in range(len(point_list96)):
            for j in range(i + 1, len(point_list96)):
                ln = _line_from_two_points(point_list96[i], point_list96[j])
                line_set96.add(ln)
        
        line_incidents96 = {}
        for ln in line_set96:
            A, B, C = ln
            cnt = 0
            for p in point_list96:
                if A * p[0] + B * p[1] + C == 0:
                    cnt += 1
            line_incidents96[ln] = cnt
        
        sorted_lines96 = sorted(line_incidents96.keys(), key=lambda x: -line_incidents96[x])
        top_27_96 = sorted_lines96[:27]
        grid_lines96 = np.array(top_27_96, dtype=np.int64)
        
        score96 = _count_incidence(point_list96, grid_lines96.tolist())
        if score96 > best_score:
            best_points = grid_points96.copy()
            best_lines = grid_lines96.copy()
            best_score = score96
    
    # Try a 5x11 grid plus 4 extra points
    if time.time() < deadline - 60:
        xs511 = np.arange(1, 6)
        ys511 = np.arange(1, 12)
        px511, py511 = np.meshgrid(xs511, ys511, indexing="ij")
        grid_points511 = np.stack([px511.ravel(), py511.ravel()], axis=1).astype(np.int64)
        
        # Add 4 extra points to make 54
        extra_pts = [(6, 1), (6, 2), (6, 3), (6, 4)]
        all_pts511 = np.vstack([grid_points511, np.array(extra_pts, dtype=np.int64)])
        
        point_list511 = all_pts511.tolist()
        line_set511 = set()
        for i in range(len(point_list511)):
            for j in range(i + 1, len(point_list511)):
                ln = _line_from_two_points(point_list511[i], point_list511[j])
                line_set511.add(ln)
        
        line_incidents511 = {}
        for ln in line_set511:
            A, B, C = ln
            cnt = 0
            for p in point_list511:
                if A * p[0] + B * p[1] + C == 0:
                    cnt += 1
            line_incidents511[ln] = cnt
        
        sorted_lines511 = sorted(line_incidents511.keys(), key=lambda x: -line_incidents511[x])
        top_27_511 = sorted_lines511[:27]
        grid_lines511 = np.array(top_27_511, dtype=np.int64)
        
        score511 = _count_incidence(point_list511, grid_lines511.tolist())
        if score511 > best_score:
            best_points = all_pts511.copy()
            best_lines = grid_lines511.copy()
            best_score = score511
    
    # Try a 4x14 grid minus 2 points plus 2 extra to make 54
    if time.time() < deadline - 60:
        xs414 = np.arange(1, 5)
        ys414 = np.arange(1, 15)
        px414, py414 = np.meshgrid(xs414, ys414, indexing="ij")
        grid_points414 = np.stack([px414.ravel(), py414.ravel()], axis=1).astype(np.int64)
        
        # Remove 2 points and add 2 extras to keep 54
        # Remove (1,1) and (4,14), add (5,1) and (5,2)
        mask = ~((grid_points414[:, 0] == 1) & (grid_points414[:, 1] == 1)) & \
               ~((grid_points414[:, 0] == 4) & (grid_points414[:, 1] == 14))
        grid_points414 = grid_points414[mask]
        extra_pts414 = np.array([(5, 1), (5, 2)], dtype=np.int64)
        all_pts414 = np.vstack([grid_points414, extra_pts414])
        
        point_list414 = all_pts414.tolist()
        line_set414 = set()
        for i in range(len(point_list414)):
            for j in range(i + 1, len(point_list414)):
                ln = _line_from_two_points(point_list414[i], point_list414[j])
                line_set414.add(ln)
        
        line_incidents414 = {}
        for ln in line_set414:
            A, B, C = ln
            cnt = 0
            for p in point_list414:
                if A * p[0] + B * p[1] + C == 0:
                    cnt += 1
            line_incidents414[ln] = cnt
        
        sorted_lines414 = sorted(line_incidents414.keys(), key=lambda x: -line_incidents414[x])
        top_27_414 = sorted_lines414[:27]
        grid_lines414 = np.array(top_27_414, dtype=np.int64)
        
        score414 = _count_incidence(point_list414, grid_lines414.tolist())
        if score414 > best_score:
            best_points = all_pts414.copy()
            best_lines = grid_lines414.copy()
            best_score = score414
    
    return best_points, best_lines


# EVOLVE-BLOCK-END
