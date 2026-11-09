import time
import itertools
import numpy as np
from collections import Counter
import random

# Precompute indices for all triples (i, j, k) with i < j < k for n=30
_N = 30
_triple_indices = None

def _get_triple_indices():
    global _triple_indices
    if _triple_indices is None:
        indices = []
        for i in range(_N):
            for j in range(i + 1, _N):
                for k in range(j + 1, _N):
                    indices.append((i, j, k))
        _triple_indices = np.array(indices, dtype=np.int32)
    return _triple_indices

def _valid(points):
    """Check 30 distinct integer points in [-30,30]^2."""
    if len(points) != 30:
        return False
    if len(set(points)) != 30:
        return False
    for x, y in points:
        if not (isinstance(x, int) and isinstance(y, int)):
            return False
        if x < -30 or x > 30 or y < -30 or y > 30:
            return False
    return True

def _compute_all_areas(points):
    """
    Compute all doubled areas for the given points.
    Returns a numpy array of positive areas.
    """
    pts = np.array(points, dtype=np.int64)
    idx = _get_triple_indices()
    
    i_idx = idx[:, 0]
    j_idx = idx[:, 1]
    k_idx = idx[:, 2]
    
    xi = pts[i_idx, 0]
    yi = pts[i_idx, 1]
    xj = pts[j_idx, 0]
    yj = pts[j_idx, 1]
    xk = pts[k_idx, 0]
    yk = pts[k_idx, 1]
    
    dx1 = xj - xi
    dy1 = yj - yi
    dx2 = xk - xi
    dy2 = yk - yi
    
    areas = np.abs(dx1 * dy2 - dy1 * dx2)
    
    # Filter out zero areas
    positive_areas = areas[areas > 0]
    
    return positive_areas

def _score(points):
    """
    Compute the maximum multiplicity of a single positive doubled area.
    """
    if not _valid(points):
        return 0
    
    a_pos = _compute_all_areas(points)
    
    if len(a_pos) == 0:
        return 0
        
    # Count frequencies
    unique_areas, counts = np.unique(a_pos, return_counts=True)
    max_mult = np.max(counts)
    
    return int(max_mult)

def _generate_square_grid(rows, cols, spacing=1, center=(0,0)):
    """Generate a grid of points."""
    points = []
    xs = [center[0] + (c - (cols - 1) / 2.0) * spacing for c in range(cols)]
    ys = [center[1] + (r - (rows - 1) / 2.0) * spacing for r in range(rows)]
    
    for x in xs:
        for y in ys:
            ix, iy = int(round(x)), int(round(y))
            if -30 <= ix <= 30 and -30 <= iy <= 30:
                points.append((ix, iy))
                
    return points

def _generate_candidates():
    """
    Generate a diverse set of high-symmetry initial candidates.
    """
    candidates = []
    
    # 1. Square Grids (various aspect ratios and spacings)
    grid_configs = [
        (5, 6, 4),
        (6, 5, 4),
        (10, 3, 6),
        (3, 10, 6),
        (15, 2, 8),
        (2, 15, 8),
        (5, 6, 3),
        (6, 5, 3),
        (10, 3, 5),
        (3, 10, 5),
        (15, 2, 6),
        (2, 15, 6),
        (5, 6, 2),
        (6, 5, 2),
        (10, 3, 4),
        (3, 10, 4),
    ]
    
    for rows, cols, spacing in grid_configs:
        pts = _generate_square_grid(rows, cols, spacing)
        if len(pts) == 30 and _valid(pts):
            candidates.append(pts)
            
    # 2. Hexagonal-like shifted grids
    hex_configs = [
        (5, 6, 4, 3),
        (6, 5, 4, 3),
    ]
    
    for rows, cols, xs, ys in hex_configs:
        pts = []
        for r in range(rows):
            y = (r - (rows-1)/2.0) * ys
            offset = (r % 2) * (xs / 2.0)
            for c in range(cols):
                x = (c - (cols-1)/2.0) * xs + offset
                ix, iy = int(round(x)), int(round(y))
                if -30 <= ix <= 30 and -30 <= iy <= 30:
                    pts.append((ix, iy))
        if len(pts) == 30 and _valid(pts):
            candidates.append(pts)

    # 3. Diagonal Lines
    line_pts = []
    for i in range(-10, 11, 2):
        line_pts.append((i, i))
    for i in range(-10, 11, 2):
        p = (i, -i)
        if p not in line_pts:
            line_pts.append(p)
    for i in range(-10, 11, 2):
        p = (0, i)
        if p not in line_pts:
            line_pts.append(p)
            
    if len(line_pts) >= 30:
        candidates.append(line_pts[:30])
        
    # 4. Random High-Symmetry
    for _ in range(5):
        pts = []
        while len(pts) < 30:
            x = random.randint(-15, 15) * 2
            y = random.randint(-15, 15) * 2
            if (x, y) not in pts:
                pts.append((x, y))
        if _valid(pts):
            candidates.append(pts)

    return candidates

class IncrementalAreaCounter:
    def __init__(self, points):
        self.points = [list(p) for p in points]
        self.n = len(self.points)
        self.area_counts = Counter()
        self.current_max_mult = 0
        self._update_all()
        
    def _calc_area(self, i, j, k):
        x1, y1 = self.points[i]
        x2, y2 = self.points[j]
        x3, y3 = self.points[k]
        return abs((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))
        
    def _update_all(self):
        self.area_counts = Counter()
        for i in range(self.n):
            for j in range(i+1, self.n):
                for k in range(j+1, self.n):
                    a = self._calc_area(i, j, k)
                    if a > 0:
                        self.area_counts[a] += 1
        if self.area_counts:
            self.current_max_mult = max(self.area_counts.values())
        else:
            self.current_max_mult = 0
            
    def try_move(self, idx, new_pos):
        """
        Try moving point idx to new_pos.
        Returns True if the move improved the score, False otherwise.
        If True, the move is kept.
        """
        old_pos = self.points[idx][:]
        
        # Check bounds and collision
        if new_pos[0] < -30 or new_pos[0] > 30 or new_pos[1] < -30 or new_pos[1] > 30:
            return False
        if new_pos in [tuple(p) for p in self.points if p != self.points[idx]]:
            return False
            
        # Calculate delta counts
        old_areas = []
        new_areas = []
        
        for j in range(self.n):
            if j == idx: continue
            for k in range(j+1, self.n):
                if k == idx: continue
                
                a_old = self._calc_area(idx, j, k)
                
                x1, y1 = new_pos
                x2, y2 = self.points[j]
                x3, y3 = self.points[k]
                a_new = abs((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))
                
                if a_old > 0:
                    old_areas.append(a_old)
                if a_new > 0:
                    new_areas.append(a_new)
                    
        # Apply updates
        for a in old_areas:
            self.area_counts[a] -= 1
            if self.area_counts[a] == 0:
                del self.area_counts[a]
                
        for a in new_areas:
            self.area_counts[a] += 1
            
        # Update max
        if self.area_counts:
            new_max = max(self.area_counts.values())
        else:
            new_max = 0
            
        if new_max > self.current_max_mult:
            self.points[idx] = list(new_pos)
            self.current_max_mult = new_max
            return True
        else:
            # Revert
            for a in old_areas:
                if a in self.area_counts:
                    self.area_counts[a] += 1
                else:
                    self.area_counts[a] = 1
            for a in new_areas:
                self.area_counts[a] -= 1
                if self.area_counts[a] == 0:
                    del self.area_counts[a]
            return False

    def try_swap(self, idx1, idx2, new_pos1, new_pos2):
        """
        Try swapping: move idx1 to new_pos1 and idx2 to new_pos2.
        Returns True if the swap improved the score, False otherwise.
        """
        old_pos1 = self.points[idx1][:]
        old_pos2 = self.points[idx2][:]
        
        # Check bounds
        for pos in [new_pos1, new_pos2]:
            if pos[0] < -30 or pos[0] > 30 or pos[1] < -30 or pos[1] > 30:
                return False
        
        # Check collision with other points
        other_points = [tuple(p) for i, p in enumerate(self.points) if i not in [idx1, idx2]]
        if new_pos1 in other_points or new_pos2 in other_points:
            return False
        if new_pos1 == new_pos2:
            return False
            
        # Calculate delta counts for all triples involving idx1 or idx2
        old_areas = []
        new_areas = []
        
        for i in range(self.n):
            for j in range(i+1, self.n):
                for k in range(j+1, self.n):
                    # Check if this triple involves idx1 or idx2
                    if idx1 in (i, j, k) or idx2 in (i, j, k):
                        a_old = self._calc_area(i, j, k)
                        
                        # Calculate new area with swapped positions
                        pts_new = self.points[:]
                        pts_new[idx1] = list(new_pos1)
                        pts_new[idx2] = list(new_pos2)
                        
                        x1, y1 = pts_new[i]
                        x2, y2 = pts_new[j]
                        x3, y3 = pts_new[k]
                        a_new = abs((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))
                        
                        if a_old > 0:
                            old_areas.append(a_old)
                        if a_new > 0:
                            new_areas.append(a_new)
        
        # Apply updates
        for a in old_areas:
            self.area_counts[a] -= 1
            if self.area_counts[a] == 0:
                del self.area_counts[a]
                
        for a in new_areas:
            self.area_counts[a] += 1
            
        # Update max
        if self.area_counts:
            new_max = max(self.area_counts.values())
        else:
            new_max = 0
            
        if new_max > self.current_max_mult:
            self.points[idx1] = list(new_pos1)
            self.points[idx2] = list(new_pos2)
            self.current_max_mult = new_max
            return True
        else:
            # Revert
            for a in old_areas:
                if a in self.area_counts:
                    self.area_counts[a] += 1
                else:
                    self.area_counts[a] = 1
            for a in new_areas:
                self.area_counts[a] -= 1
                if self.area_counts[a] == 0:
                    del self.area_counts[a]
            return False

def _local_search(points, time_limit=10.0):
    """
    Hill Climbing with Incremental Scoring and swap operations.
    """
    start_time = time.time()
    counter = IncrementalAreaCounter(points)
    best_points = [tuple(p) for p in counter.points]
    best_score = counter.current_max_mult
    
    n = len(points)
    no_improve_count = 0
    max_no_improve = 30  # Increased from 20
    
    while time.time() - start_time < time_limit:
        if no_improve_count > max_no_improve:
            # Random restart perturbation: move 3-5 points randomly
            num_moves = random.randint(3, 5)
            for _ in range(num_moves):
                idx = random.randint(0, n-1)
                px, py = counter.points[idx]
                step = random.randint(5, 15)
                nx = px + random.choice([-step, step])
                ny = py + random.choice([-step, step])
                if -30 <= nx <= 30 and -30 <= ny <= 30:
                    counter.try_move(idx, (nx, ny))
            no_improve_count = 0
            continue
            
        idx = random.randint(0, n-1)
        px, py = counter.points[idx]
        
        # Try larger neighborhood (±5 instead of ±3)
        improved = False
        for _ in range(15):  # Increased from 10
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
            nx, ny = px + dx, py + dy
            
            if counter.try_move(idx, (nx, ny)):
                improved = True
                best_points = [tuple(p) for p in counter.points]
                best_score = counter.current_max_mult
                break
                
        if not improved:
            # Try swap operation
            idx2 = random.randint(0, n-1)
            if idx2 != idx:
                px2, py2 = counter.points[idx2]
                for _ in range(5):  # Try a few swap attempts
                    dx1 = random.randint(-5, 5)
                    dy1 = random.randint(-5, 5)
                    dx2 = random.randint(-5, 5)
                    dy2 = random.randint(-5, 5)
                    nx1, ny1 = px + dx1, py + dy1
                    nx2, ny2 = px2 + dx2, py2 + dy2
                    
                    if counter.try_swap(idx, idx2, (nx1, ny1), (nx2, ny2)):
                        improved = True
                        best_points = [tuple(p) for p in counter.points]
                        best_score = counter.current_max_mult
                        break
        
        if not improved:
            no_improve_count += 1
            
    return best_points, best_score

def construct():
    """
    Construct 30 points maximizing the multiplicity of a single doubled area.
    """
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    candidates = _generate_candidates()
    
    if not candidates:
        # Fallback
        return [(x, y) for x in range(6) for y in range(5)]
        
    best_points = None
    best_score = 0
    
    # Score all candidates
    for cand in candidates:
        if not _valid(cand):
            continue
        s = _score(cand)
        if s > best_score:
            best_score = s
            best_points = list(cand)
            
    if best_points is None:
        return [(x, y) for x in range(6) for y in range(5)]
        
    # Run local search on the best candidate
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        search_time = min(remaining_time - 2, 40)
        searched_pts, s = _local_search(best_points, time_limit=search_time)
        if s > best_score:
            best_score = s
            best_points = searched_pts
            
    # If we have significant time left, try local search on other top candidates
    if deadline - time.time() > 10:
        # Sort candidates by score
        scored_cands = []
        for cand in candidates:
            if not _valid(cand):
                continue
            s = _score(cand)
            scored_cands.append((s, cand))
        scored_cands.sort(key=lambda x: x[0], reverse=True)
        
        # Try the next best few
        for s, cand in scored_cands[1:5]:
            if time.time() > deadline - 5:
                break
            if cand == best_points:
                continue
            # Only search if it's close to the best
            if s >= best_score - 5:
                remaining = deadline - time.time() - 3
                if remaining > 5:
                    cand_search, s2 = _local_search(cand, time_limit=min(remaining, 10))
                    if s2 > best_score:
                        best_score = s2
                        best_points = cand_search
                        
    # Final check
    if not _valid(best_points):
        return [(x, y) for x in range(6) for y in range(5)]
        
    return best_points

# EVOLVE-BLOCK-END
