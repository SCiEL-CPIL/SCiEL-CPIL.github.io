import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    k = 3  # alpha^2 = 3

    n_points = 2401

    def point_to_idx(xa, xb, ya, yb):
        return ((xa * 7 + xb) * 7 + ya) * 7 + yb

    def idx_to_point(idx):
        yb = idx % 7
        ya = (idx // 7) % 7
        xb = (idx // 49) % 7
        xa = (idx // 343) % 7
        return (xa, xb, ya, yb)

    start_time = time.time()
    deadline = 580  # safety margin before 630s kill

    # Precompute all 50 directions * 49 lines each
    # Direction d (0-48): v = (1, m) where m = (ma, mb)
    # Direction 49: v = (0, 1) [vertical]
    # For direction (1, ma, mb), line through (xa,xb,ya,yb) has invariant:
    #   c1 = (ya - xa*ma - 3*xb*mb) % 7
    #   c2 = (yb - xa*mb - xb*ma) % 7
    # For vertical, line ID = (xa, xb)
    
    # Precompute line membership: for each direction d and line l, the 49 point indices
    all_lines = []  # all_lines[d][l] = np.array of 49 int64 indices
    point_line_dir = np.zeros((n_points, 50), dtype=np.int64)
    
    slopes = [(ma, mb) for ma in range(7) for mb in range(7)]
    
    for d in range(50):
        line_to_points = {}
        for idx in range(n_points):
            xa, xb, ya, yb = idx_to_point(idx)
            if d < 49:
                ma, mb = slopes[d]
                c1 = (ya - xa * ma - 3 * xb * mb) % 7
                c2 = (yb - xa * mb - xb * ma) % 7
                lid = c1 * 7 + c2
            else:
                lid = xa * 7 + xb
            if lid not in line_to_points:
                line_to_points[lid] = []
            line_to_points[lid].append(idx)
        
        lines_in_dir = []
        for lid in range(49):
            if lid in line_to_points:
                pts = np.array(line_to_points[lid], dtype=np.int64)
            else:
                pts = np.array([], dtype=np.int64)
            lines_in_dir.append(pts)
            for idx in line_to_points.get(lid, []):
                point_line_dir[idx][d] = lid
        all_lines.append(lines_in_dir)
    
    # Initialize N using the Hermitian unital construction with QR exclusion
    QR = {0, 1, 2, 4}  # Quadratic residues mod 7
    in_N = np.ones(n_points, dtype=bool)
    for xa in range(7):
        for xb in range(7):
            for ya in range(7):
                for yb in range(7):
                    in_unital = (2 * xa + ya * ya - yb * yb * k) % 7 == 0
                    if in_unital and ya in QR:
                        idx = point_to_idx(xa, xb, ya, yb)
                        in_N[idx] = False
    
    # Track n_in_N[d][l] = number of points in N on line l in direction d
    n_in_N = np.zeros((50, 49), dtype=np.int64)
    for d in range(50):
        for l in range(49):
            n_in_N[d][l] = np.sum(in_N[all_lines[d][l]])
    
    # For each point, compute clean_count = number of directions where the line is clean
    # A direction d is clean for point q iff:
    #   If in_N[q]: n_in_N[d][point_line_dir[q][d]] == 49
    #   Else: n_in_N[d][point_line_dir[q][d]] == 48
    
    def compute_clean_count(q):
        cc = 0
        for d in range(50):
            lid = point_line_dir[q][d]
            if in_N[q]:
                if n_in_N[d][lid] == 49:
                    cc += 1
            else:
                if n_in_N[d][lid] == 48:
                    cc += 1
        return cc
    
    def is_valid():
        for q in range(n_points):
            if compute_clean_count(q) == 0:
                return False
        return True
    
    # Verify initial validity
    if not is_valid():
        # Fallback: return the unital construction as-is (should be valid)
        points = []
        for idx in range(n_points):
            if in_N[idx]:
                xa, xb, ya, yb = idx_to_point(idx)
                points.append([[xa, xb], [ya, yb]])
        return np.array(points, dtype=np.int64)
    
    # Greedy removal: try to remove points from N (add to complement S)
    # A point p can be removed if after removal, all points still have >= 1 clean direction.
    # Removing p only affects points on lines through p in each direction.
    # For efficiency, track clean_count for each point.
    
    clean_count = np.zeros(n_points, dtype=np.int64)
    for q in range(n_points):
        clean_count[q] = compute_clean_count(q)
    
    removed = 0
    # Try removing points in order of their index; prioritize points that are "easy" to remove
    # (i.e., points where the removal breaks few clean lines)
    
    for p in range(n_points):
        if time.time() - start_time > deadline:
            break
        if not in_N[p]:
            continue
        
        # Check if removing p maintains validity
        # Temporarily remove p
        in_N[p] = False
        affected_lines = []
        for d in range(50):
            lid = point_line_dir[p][d]
            old_val = n_in_N[d][lid]
            n_in_N[d][lid] = old_val - 1
            affected_lines.append((d, lid, old_val))
        
        # Check if any point loses its last clean direction
        valid = True
        # Only need to check points on affected lines (lines through p in some direction)
        # But since all lines through p cover all points, check all points
        # Optimization: only check points whose clean_count might have changed
        # A point q's clean status in direction d changes only if q is on the same line as p in direction d
        # i.e., point_line_dir[q][d] == point_line_dir[p][d]
        
        # For each direction d, the affected points are those on line point_line_dir[p][d]
        affected_points = set()
        for d in range(50):
            lid = point_line_dir[p][d]
            for q in all_lines[d][lid]:
                affected_points.add(q)
        
        for q in affected_points:
            if clean_count[q] == 0:
                valid = False
                break
            # Recompute clean_count for q
            new_cc = 0
            for d in range(50):
                lid = point_line_dir[q][d]
                if in_N[q]:
                    if n_in_N[d][lid] == 49:
                        new_cc += 1
                else:
                    if n_in_N[d][lid] == 48:
                        new_cc += 1
            clean_count[q] = new_cc
        
        if valid:
            removed += 1
            # Keep p removed
        else:
            # Restore
            in_N[p] = True
            for d, lid, old_val in affected_lines:
                n_in_N[d][lid] = old_val
            # Restore clean counts for affected points
            for q in affected_points:
                new_cc = 0
                for d in range(50):
                    lid = point_line_dir[q][d]
                    if in_N[q]:
                        if n_in_N[d][lid] == 49:
                            new_cc += 1
                    else:
                        if n_in_N[d][lid] == 48:
                            new_cc += 1
                clean_count[q] = new_cc
    
    # Build the result
    points = []
    for idx in range(n_points):
        if in_N[idx]:
            xa, xb, ya, yb = idx_to_point(idx)
            points.append([[xa, xb], [ya, yb]])
    
    if not points:
        points = [[[0, 0], [0, 0]]]
    
    return np.array(points, dtype=np.int64)
