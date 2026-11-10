import numpy as np
import time
from math import gcd
from itertools import combinations

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = 620.0  # safety margin before 630s kill
    
    def normalize_line(A, B, C):
        if A == 0 and B == 0:
            return None
        g = gcd(gcd(abs(A), abs(B)), abs(C))
        if g > 0:
            A, B, C = A // g, B // g, C // g
        if A < 0 or (A == 0 and B < 0):
            A, B, C = -A, -B, -C
        return (A, B, C)
    
    def count_incidence(points, lines):
        """Count incidences between points (N,2) and lines (M,3)."""
        if len(points) == 0 or len(lines) == 0:
            return 0
        A = lines[:, 0:1]
        B = lines[:, 1:2]
        C = lines[:, 2:3]
        x = points[:, 0:1].T
        y = points[:, 1:1].T
        res = A * x + B * y + C
        return int(np.sum(res == 0))
    
    def generate_candidates(nx, ny, base_points):
        """Generate a rich set of candidate lines for an nx x ny grid."""
        candidates = set()
        
        # Vertical lines x = c
        for c in range(nx):
            candidates.add((1, 0, -c))
            
        # Horizontal lines y = c
        for c in range(ny):
            candidates.add((0, 1, -c))
        
        # Diagonals and other rational slopes
        # We want lines A*x + B*y + C = 0 where gcd(A,B)=1
        # For a line to pass through at least 2 grid points, the slope must be rational
        # and the intercept must be such that it hits integer coordinates.
        
        # Iterate over primitive (A, B) pairs
        # A and B can be negative. We normalize later, but for generation we can be systematic.
        # Let's iterate over dx, dy where gcd(dx, dy)=1.
        # Line: dy * x - dx * y + C = 0  => y = (dy/dx) x + C/dx
        # Actually, let's use the form: A = dy, B = -dx.
        # Then A*x + B*y + C = 0 => dy*x - dx*y + C = 0.
        # C = dx*y - dy*x.
        # For points (x, y) in the grid, C takes specific values.
        
        max_slope = max(nx, ny) + 1
        for dx in range(1, max_slope + 1):
            for dy in range(-max_slope, max_slope + 1):
                if dy == 0:
                    continue
                if gcd(dx, abs(dy)) != 1:
                    continue
                
                A = dy
                B = -dx
                # C = dx*y - dy*x
                C_vals = dx * base_points[:, 1] - dy * base_points[:, 0]
                vals = np.unique(C_vals)
                for C in vals:
                    candidates.add((A, B, C))
                    
        # Normalize and deduplicate
        unique_lines = []
        seen = set()
        for line in candidates:
            norm = normalize_line(*line)
            if norm and norm not in seen:
                seen.add(norm)
                unique_lines.append(norm)
                
        return np.array(unique_lines, dtype=np.int64)
    
    best_overall_score = -1
    best_overall_points = None
    best_overall_lines = None
    
    # Try different grid dimensions
    # The current best uses 7x8 (56 pts, remove 2).
    # Inspiration 1 tries 8x7 and 7x8.
    # Let's also try 6x9 (54 pts, no removal) and maybe 9x6.
    # And maybe 5x11 (55 pts, remove 1)?
    
    grid_configs = [
        (7, 8),  # 56 points
        (8, 7),  # 56 points
        (6, 9),  # 54 points
        (9, 6),  # 54 points
        (5, 11), # 55 points
        (11, 5), # 55 points
    ]
    
    for nx, ny in grid_configs:
        if time.time() > deadline - 10:
            break
            
        n_grid = nx * ny
        if n_grid < 54:
            continue
            
        xs = np.arange(nx, dtype=np.int64)
        ys = np.arange(ny, dtype=np.int64)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        base_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        
        # Generate candidates
        candidates = generate_candidates(nx, ny, base_points)
        
        if len(candidates) == 0:
            continue
        
        # Precompute incidence matrix for full grid
        # point_line_incidence[k, m] = 1 if point k lies on line m
        # This is (n_grid, n_lines)
        A = candidates[:, 0:1]
        B = candidates[:, 1:2]
        C = candidates[:, 2:3]
        x = base_points[:, 0:1].T
        y = base_points[:, 1:1].T
        res = A * x + B * y + C
        point_line_incidence = (res == 0).astype(np.int8)
        
        total_line_cov = point_line_incidence.sum(axis=0)
        
        # Strategy depends on n_grid
        if n_grid == 54:
            # No points to remove. Just pick top 27 lines.
            top_indices = np.argsort(-total_line_cov)[:27]
            score = int(total_line_cov[top_indices].sum())
            
            if score > best_overall_score:
                best_overall_score = score
                best_overall_points = base_points.copy()
                best_overall_lines = candidates[top_indices].copy()
                
        elif n_grid == 55:
            # Remove 1 point
            # line_incidents = total_line_cov - point_line_incidence[i]
            # We want to maximize sum of top 27 (total_line_cov - point_line_incidence[i])
            # This is equivalent to minimizing sum of top 27 point_line_incidence[i] weighted by whether they are in top 27 lines?
            # Actually, we just iterate i.
            for i in range(n_grid):
                if time.time() > deadline - 10:
                    break
                line_incidents = total_line_cov - point_line_incidence[i]
                top_indices = np.argsort(-line_incidents)[:27]
                score = int(line_incidents[top_indices].sum())
                
                if score > best_overall_score:
                    best_overall_score = score
                    mask = np.ones(n_grid, dtype=bool)
                    mask[i] = False
                    best_overall_points = base_points[mask].copy()
                    best_overall_lines = candidates[top_indices].copy()
                    
        elif n_grid == 56:
            # Remove 2 points
            # 56 choose 2 = 1540 pairs. Fast enough.
            for i, j in combinations(range(n_grid), 2):
                if time.time() > deadline - 10:
                    break
                line_incidents = total_line_cov - point_line_incidence[i] - point_line_incidence[j]
                top_indices = np.argsort(-line_incidents)[:27]
                score = int(line_incidents[top_indices].sum())
                
                if score > best_overall_score:
                    best_overall_score = score
                    mask = np.ones(n_grid, dtype=bool)
                    mask[i] = False
                    mask[j] = False
                    best_overall_points = base_points[mask].copy()
                    best_overall_lines = candidates[top_indices].copy()
        
        # Optional: Local refinement
        # If we have time, try to swap points or lines to improve.
        # But given the small search space for removals, the global optimum for the grid is likely found.
        # The main gain comes from trying multiple grids.
        
    # Fallback
    if best_overall_points is None:
        # Should not happen, but just in case
        best_overall_points = np.zeros((54, 2), dtype=np.int64)
        best_overall_lines = np.zeros((27, 3), dtype=np.int64)
        
    return best_overall_points, best_overall_lines
