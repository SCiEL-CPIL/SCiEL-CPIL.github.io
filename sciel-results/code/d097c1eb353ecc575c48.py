# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from itertools import product

def _incidence_count(points, lines):
    """Compute number of (point, line) incidences. points: (n,2), lines: (m,3)."""
    px = points[:, 0:1]
    py = points[:, 1:2]
    A = lines[0:1, :]
    B = lines[1:2, :]
    C = lines[2:3, :]
    total = px * A + py * B + C
    return int(np.sum(total == 0))


def _gcd3(a, b, c):
    """Compute gcd of three integers."""
    g = math.gcd(abs(int(a)), abs(int(b)))
    g = math.gcd(g, abs(int(c)))
    return g if g != 0 else 1


def _normalize_line(A, B, C):
    """Normalize a line so it's unique."""
    A, B, C = int(A), int(B), int(C)
    g = _gcd3(A, B, C)
    A, B, C = A // g, B // g, C // g
    # Make first nonzero entry positive
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)


def _validate(points, lines):
    """Validate the points and lines arrays. Returns True if valid."""
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    
    # Check for NaN or inf
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(lines)):
        return False
    
    # Check entries are close to integers
    if np.any(np.abs(points - np.round(points)) > 1e-6):
        return False
    if np.any(np.abs(lines - np.round(lines)) > 1e-6):
        return False
    
    # Round to int64
    points_int = np.round(points).astype(np.int64)
    lines_int = np.round(lines).astype(np.int64)
    
    # Check for repeated points
    if len(set(map(tuple, points_int))) != 54:
        return False
    
    # Check for invalid lines (A = B = 0)
    for i in range(27):
        A, B, C = lines_int[i]
        if A == 0 and B == 0:
            return False
    
    # Check for duplicate lines
    normalized = set()
    for i in range(27):
        A, B, C = lines_int[i]
        norm = _normalize_line(A, B, C)
        if norm in normalized:
            return False
        normalized.add(norm)
    
    return True


def _grid_construction(k=3):
    """Classical Szemeredi-Trotter grid construction."""
    xs = np.arange(1, k + 1)
    ys = np.arange(1, 2 * k * k + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    rows = []
    for a in range(1, k + 1):
        for b in range(1, k * k + 1):
            rows.append((a, -1, b))
    lines = np.array(rows, dtype=np.int64)
    
    return points, lines


def _better_construction():
    """Try to find a better construction using a different approach."""
    # Use a denser grid with more structure
    # Try: points on a k x k grid, lines with various slopes
    
    # Alternative: use points from {1,...,6} x {1,...,9} = 54 points
    # Lines: try to maximize incidences
    
    # For now, let's try a different parameterization
    # Points: {1,...,6} x {1,...,9}
    xs = np.arange(1, 7)
    ys = np.arange(1, 10)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # Lines: y = a*x + b with various a, b
    # To get 27 lines, try a in {0, 1, 2}, b in {0, 1, ..., 8}
    rows = []
    for a in range(0, 3):
        for b in range(0, 9):
            rows.append((a, -1, b))
    lines = np.array(rows, dtype=np.int64)
    
    return points, lines


def _search_better(deadline):
    """Search for a better construction within the time budget."""
    best_points, best_lines = _grid_construction(3)
    best_score = _incidence_count(best_points, best_lines)
    
    start_time = time.time()
    
    # Try different grid configurations
    for k in [2, 3, 4, 5, 6]:
        if time.time() > deadline:
            break
        
        # Try k x (54//k) grids
        if 54 % k != 0:
            continue
        
        m = 54 // k
        xs = np.arange(1, k + 1)
        ys = np.arange(1, m + 1)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        
        # Try different line configurations
        # Lines: y = a*x + b
        # We need 27 lines
        # Try a in {0, 1, ..., A-1}, b in {0, 1, ..., B-1} where A*B = 27
        for A in [1, 3, 9, 27]:
            if 27 % A != 0:
                continue
            B = 27 // A
            
            rows = []
            for a in range(0, A):
                for b in range(0, B):
                    rows.append((a, -1, b))
            lines = np.array(rows, dtype=np.int64)
            
            if _validate(points, lines):
                score = _incidence_count(points, lines)
                if score > best_score:
                    best_score = score
                    best_points = points.copy()
                    best_lines = lines.copy()
        
        # Also try lines with different B values
        # Lines: A*x + B*y + C = 0
        # Try B = -1, -2, -3, etc.
        for B_val in [-1, -2, -3]:
            rows = []
            count = 0
            for a in range(0, 10):
                for b in range(0, 10):
                    if count >= 27:
                        break
                    rows.append((a, B_val, b))
                    count += 1
                if count >= 27:
                    break
            
            if len(rows) == 27:
                lines = np.array(rows, dtype=np.int64)
                if _validate(points, lines):
                    score = _incidence_count(points, lines)
                    if score > best_score:
                        best_score = score
                        best_points = points.copy()
                        best_lines = lines.copy()
        
        if time.time() > deadline:
            break
    
    return best_points, best_lines, best_score


def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization. Construct a set of n points and a set of
    m lines in the plane (on the integer lattice) so as to maximize the number of
    point-line incidences (pairs (p, l) with p lying on l). Here the instance is fixed
    at n = 54 points and m = 27 lines.

    Returns:
        points: np.ndarray of shape (54, 2), integer (x, y) lattice coordinates.
        lines : np.ndarray of shape (27, 3), integer (A, B, C) with A*x + B*y + C = 0.
    """
    start_time = time.time()
    deadline = start_time + 580  # Leave 20s safety margin before 600s limit
    
    # Start with the baseline construction
    points, lines = _grid_construction(3)
    
    # Try to find a better construction
    if time.time() < deadline:
        better_points, better_lines, better_score = _search_better(deadline)
        baseline_score = _incidence_count(points, lines)
        
        if better_score > baseline_score:
            points, lines = better_points, better_lines
    
    # Final validation and rounding
    points = np.round(points).astype(np.int64)
    lines = np.round(lines).astype(np.int64)
    
    return points, lines


# EVOLVE-BLOCK-END
