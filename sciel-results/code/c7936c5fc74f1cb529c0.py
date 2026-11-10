# EVOLVE-BLOCK-START
import numpy as np
from math import gcd
import time
import itertools

DEADLINE = 620  # seconds; leave 10s margin before 630s kill

def _normalize_line(A, B, C):
    """Normalize (A,B,C) so gcd=1 and first nonzero is positive."""
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return None
    A //= g
    B //= g
    C //= g
    if A < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B < 0:
        B, C = -B, -C
    return (A, B, C)

def _compute_incidence(points, lines):
    """Compute total incidences for given points and lines."""
    total = 0
    for A, B, C in lines:
        total += np.sum(A * points[:, 0] + B * points[:, 1] + C == 0)
    return total

def _find_best_lines(points, max_lines=27):
    """Find the best max_lines lines for given points."""
    n = len(points)
    line_counts = {}
    
    # Generate all lines from point pairs
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            key = _normalize_line(A, B, C)
            if key is not None and key not in line_counts:
                line_counts[key] = 0
    
    # Count incidences
    for key in line_counts:
        A, B, C = key
        line_counts[key] = np.sum(A * points[:, 0] + B * points[:, 1] + C == 0)
    
    # Sort and select top lines
    sorted_lines = sorted(line_counts.items(), key=lambda x: x[1], reverse=True)
    selected = [k for k, v in sorted_lines[:max_lines]]
    
    while len(selected) < max_lines:
        selected.append((1, 0, -1000))
    
    return np.array(selected, dtype=np.int64)

def _local_search(points, start_time, deadline):
    """Local search: try moving each point to improve total incidence."""
    n = len(points)
    best_score = -1
    best_points = points.copy()
    
    # Initial evaluation
    lines = _find_best_lines(best_points)
    best_score = _compute_incidence(best_points, lines)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() >= deadline:
                break
            orig = best_points[i].copy()
            # Try small perturbations
            for dx in [-2, -1, 0, 1, 2]:
                for dy in [-2, -1, 0, 1, 2]:
                    if dx == 0 and dy == 0:
                        continue
                    trial = best_points.copy()
                    trial[i, 0] = orig[0] + dx
                    trial[i, 1] = orig[1] + dy
                    
                    # Check for duplicates
                    if np.any(np.all(trial == trial[i], axis=1) and np.arange(n) != i):
                        continue
                    
                    trial_lines = _find_best_lines(trial)
                    score = _compute_incidence(trial, trial_lines)
                    
                    if score > best_score:
                        best_score = score
                        best_points = trial
                        improved = True
                        break
                if improved:
                    break
    
    return best_points, best_score

def szemeredi_trotter_incidences():
    """
    Maximizes point-line incidences for 54 points and 27 lines on the integer lattice.
    
    Strategy:
    1. Start with a 6x9 grid of points (54 points total).
    2. Use local search to improve the configuration by moving individual points.
    3. Select the 27 lines with the highest incidence counts.
    """
    start_time = time.time()
    
    # 1. Define the 54 points on a 6x9 grid
    xs = np.arange(6)
    ys = np.arange(9)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # 2. Local search to improve
    points, score = _local_search(points, start_time, DEADLINE)
    
    # 3. Find best lines for final configuration
    lines = _find_best_lines(points)
    
    return points, lines

# EVOLVE-BLOCK-END
