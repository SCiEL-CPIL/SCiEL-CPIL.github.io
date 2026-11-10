# EVOLVE-BLOCK-START
import numpy as np
from math import gcd


def _normalize_line(A, B, C):
    """Normalize a line equation so that two equivalent lines map to the same key."""
    if A == 0 and B == 0 and C == 0:
        return (0, 0, 0)
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g > 0:
        A //= g
        B //= g
        C //= g
    # Make first nonzero entry positive
    if A < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B < 0:
        A, B, C = -A, -B, -C
    return (A, B, C)


def szemeredi_trotter_incidences():
    """
    Constructs a set of 54 points and 27 lines in the integer plane to maximize
    point-line incidences.
    
    Uses a 6x9 grid of points: {(x, y) | 0 <= x <= 5, 0 <= y <= 8}
    
    Strategy:
    1. Generate all unique lines through pairs of grid points.
    2. Count exact incidences for each line.
    3. Greedily select the 27 lines with the highest incidence counts.
    
    This systematically finds the optimal set of lines, including verticals,
    horizontals, diagonals, and any other high-incidence lines.
    """
    # Points: 6x9 grid (0-indexed for simplicity)
    k = 6  # columns
    m = 9  # rows
    xs = np.arange(k)
    ys = np.arange(m)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    n_points = len(points)
    n_lines_wanted = 27
    
    # Generate all unique lines through pairs of grid points
    all_lines = {}  # normalized line -> (A, B, C)
    
    for i in range(n_points):
        for j in range(i + 1, n_points):
            x1, y1 = int(points[i, 0]), int(points[i, 1])
            x2, y2 = int(points[j, 0]), int(points[j, 1])
            A = y2 - y1
            B = x1 - x2
            C = -(A * x1 + B * y1)
            if A == 0 and B == 0:
                continue
            norm = _normalize_line(A, B, C)
            if norm not in all_lines:
                all_lines[norm] = (A, B, C)
    
    # Count incidences for each line
    line_incidents = {}
    for norm, (A, B, C) in all_lines.items():
        count = 0
        for idx in range(n_points):
            x = int(points[idx, 0])
            y = int(points[idx, 1])
            if A * x + B * y + C == 0:
                count += 1
        line_incidents[norm] = count
    
    # Sort lines by incidence count (descending)
    sorted_lines = sorted(all_lines.items(), key=lambda x: -line_incidents[x[0]])
    
    # Greedily select the top 27 lines
    selected = []
    selected_norms = set()
    for norm, (A, B, C) in sorted_lines:
        if norm in selected_norms:
            continue
        selected.append((A, B, C))
        selected_norms.add(norm)
        if len(selected) >= n_lines_wanted:
            break
    
    # If we somehow don't have enough (shouldn't happen with 54 points), pad
    if len(selected) < n_lines_wanted:
        for norm, (A, B, C) in sorted_lines:
            if norm not in selected_norms:
                selected.append((A, B, C))
                selected_norms.add(norm)
                if len(selected) >= n_lines_wanted:
                    break
    
    lines_array = np.array(selected[:n_lines_wanted], dtype=np.int64)
    
    # Verify shapes
    assert points.shape == (54, 2), f"Points shape: {points.shape}"
    assert lines_array.shape == (27, 3), f"Lines shape: {lines_array.shape}"
    
    return points, lines_array


# EVOLVE-BLOCK-END
