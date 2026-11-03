import numpy as np
import time
from scipy.spatial.distance import cdist

N = 200
HALF = 0.5
_TIME_LIMIT = 140  # Leave 10s margin for 150s kill


def _score(c):
    """Return the common radius for a configuration (higher is better)."""
    c = np.asarray(c, dtype=np.float64)
    if c.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # inside square
    if np.any(np.abs(c) > HALF + 1e-12):
        return 0.0
    # pairwise min distance
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    # wall distance
    wall = np.min([
        np.min(c[:, 0]) + HALF,
        HALF - np.max(c[:, 0]),
        np.min(c[:, 1]) + HALF,
        HALF - np.max(c[:, 1]),
    ])
    r = min(wall, min_pair / 2.0)
    return r if r > 0 else 0.0


def _hex_grid_points(offset_x=0.0, offset_y=0.0, scale=1.0):
    """
    Generate a hexagonal lattice of points covering the unit square plus margin.
    Returns array of shape (M, 2) for M points.
    """
    # Hex lattice spacing: for packing circles of radius r, the hex spacing is 2r.
    # We generate a dense hex grid and let greedy select the best N.
    # Base spacing ~ 1/15 ≈ 0.0667, so r ~ 0.0333. We want slightly denser.
    spacing = 1.0 / 16.0 * scale
    dx = spacing
    dy = spacing * np.sqrt(3) / 2.0
    
    # Generate grid over a range that covers [-0.5, 0.5] plus margin
    margin = 0.1
    x_min, x_max = -HALF - margin, HALF + margin
    y_min, y_max = -HALF - margin, HALF + margin
    
    nx = int(np.ceil((x_max - x_min) / dx)) + 1
    ny = int(np.ceil((y_max - y_min) / dy)) + 1
    
    points = []
    for j in range(ny):
        y = y_min + j * dy + offset_y
        x_offset = 0.0 if j % 2 == 0 else dx / 2.0
        for i in range(nx):
            x = x_min + i * dx + x_offset + offset_x
            points.append([x, y])
    
    return np.array(points)


def _greedy_select(candidate_points, n_select=N):
    """
    Greedily select n_select points from candidate_points to maximize the
    minimum distance (both pairwise and to walls).
    """
    # Start with all candidates that are inside the square
    mask = np.all(np.abs(candidate_points) <= HALF, axis=1)
    candidates = candidate_points[mask]
    
    if len(candidates) < n_select:
        # Not enough candidates; pad with random points inside square
        extra = np.random.uniform(-HALF, HALF, (n_select - len(candidates), 2))
        candidates = np.vstack([candidates, extra])
    
    selected = []
    remaining = list(range(len(candidates)))
    
    # Pre-compute distances to walls for each candidate
    wall_dists = np.min([
        candidates[:, 0] + HALF,
        HALF - candidates[:, 0],
        candidates[:, 1] + HALF,
        HALF - candidates[:, 1],
    ], axis=0)
    
    # We'll build up the selected set one point at a time.
    # At each step, pick the candidate that maximizes the minimum of:
    #   - its distance to all already-selected points (divided by 2 for radius)
    #   - its wall distance
    # But we want to maximize the overall minimum radius, so we use a
    # "bottleneck" greedy: at each step, pick the point that maximizes the
    # minimum distance to any already-selected point AND to walls.
    
    # For efficiency, maintain an array of current min-distances to selected set
    # for each remaining candidate.
    min_pair_dist = np.full(len(candidates), np.inf)
    
    for step in range(n_select):
        if not remaining:
            break
        
        # For each remaining candidate, compute the min distance to selected set
        # and take min with wall distance
        best_idx = -1
        best_val = -1.0
        
        for idx in remaining:
            if selected:
                d_to_sel = np.min(np.linalg.norm(candidates[idx] - candidates[selected], axis=1))
            else:
                d_to_sel = np.inf
            val = min(d_to_sel / 2.0, wall_dists[idx])
            if val > best_val:
                best_val = val
                best_idx = idx
        
        if best_idx == -1:
            break
        
        selected.append(best_idx)
        remaining.remove(best_idx)
        
        # Update min_pair_dist for remaining candidates
        if remaining:
            d_new = np.linalg.norm(candidates[remaining] - candidates[best_idx], axis=1)
            min_pair_dist[remaining] = np.minimum(min_pair_dist[remaining], d_new)
    
    if len(selected) < n_select:
        # Should not happen if we have enough candidates
        while len(selected) < n_select:
            selected.append(np.random.randint(len(candidates)))
    
    return candidates[selected]


def _greedy_select_fast(candidate_points, n_select=N):
    """
    Faster greedy: use a priority-queue-like approach with vectorized updates.
    At each step, pick the candidate with the largest min(dist_to_selected/2, wall_dist).
    """
    mask = np.all(np.abs(candidate_points) <= HALF, axis=1)
    candidates = candidate_points[mask]
    
    if len(candidates) < n_select:
        extra = np.random.uniform(-HALF, HALF, (n_select - len(candidates), 2))
        candidates = np.vstack([candidates, extra])
    
    n_cand = len(candidates)
    
    wall_dists = np.min([
        candidates[:, 0] + HALF,
        HALF - candidates[:, 0],
        candidates[:, 1] + HALF,
        HALF - candidates[:, 1],
    ], axis=0)
    
    # current_min_dist[i] = min distance from candidate i to any selected point
    current_min_dist = np.full(n_cand, np.inf)
    selected = np.zeros(n_select, dtype=int)
    used = np.zeros(n_cand, dtype=bool)
    
    for step in range(n_select):
        # Score for each unused candidate: min(current_min_dist[i]/2, wall_dists[i])
        score = np.minimum(current_min_dist / 2.0, wall_dists)
        score[used] = -1.0
        
        best_idx = np.argmax(score)
        selected[step] = best_idx
        used[best_idx] = True
        
        # Update distances for all unused candidates
        if step < n_select - 1:
            d_new = np.linalg.norm(candidates - candidates[best_idx], axis=1)
            current_min_dist = np.minimum(current_min_dist, d_new)
            current_min_dist[used] = np.inf
    
    return candidates[selected]


def _local_refine(pts, max_iters=200, lr=0.01):
    """
    Simple gradient-free local refinement: iteratively move each point in the
    direction that increases the minimum distance to its nearest neighbors and walls.
    Uses a simple hill-climbing approach with small steps.
    """
    pts = pts.copy()
    n = len(pts)
    
    for it in range(max_iters):
        improved = False
        
        for i in range(n):
            # Compute distances to all other points
            d = np.linalg.norm(pts - pts[i], axis=1)
            d[i] = np.inf
            nearest_j = np.argmin(d)
            min_d = d[nearest_j]
            
            # Wall distances
            w_left = pts[i, 0] + HALF
            w_right = HALF - pts[i, 0]
            w_bottom = pts[i, 1] + HALF
            w_top = HALF - pts[i, 1]
            min_wall = min(w_left, w_right, w_bottom, w_top)
            
            # Determine the constraining factor
            r_current = min(min_d / 2.0, min_wall)
            
            # Move in direction away from nearest neighbor and toward center of walls
            # Direction away from nearest neighbor
            if min_d < 2.0 * min_wall:
                # Constrained by pair distance
                direction = pts[i] - pts[nearest_j]
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    direction /= norm
                    # Move a small step in that direction
                    new_pt = pts[i] + lr * direction
                    # Check if it stays in bounds
                    if np.all(np.abs(new_pt) <= HALF):
                        # Check if it improves the min distance
                        new_d = np.linalg.norm(new_pt - pts[nearest_j])
                        if new_d > min_d * (1 + 1e-6):
                            pts[i] = new_pt
                            improved = True
            else:
                # Constrained by wall
                # Move toward the center
                direction = -pts[i]
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    direction /= norm
                    new_pt = pts[i] + lr * direction
                    if np.all(np.abs(new_pt) <= HALF):
                        # Check wall distance
                        new_w = min(new_pt[0] + HALF, HALF - new_pt[0],
                                    new_pt[1] + HALF, HALF - new_pt[1])
                        if new_w > min_wall * (1 + 1e-6):
                            pts[i] = new_pt
                            improved = True
        
        if not improved:
            break
    
    return pts


def _hill_climb(pts, max_rounds=50, step_size=0.005):
    """
    Coordinate-wise hill climbing: for each point, try small perturbations
    in each coordinate direction and accept if the score improves.
    """
    pts = pts.copy()
    current_score = _score(pts)
    
    for round_ in range(max_rounds):
        improved = False
        for i in range(N):
            for coord in range(2):
                for direction in [-1, 1]:
                    new_pts = pts.copy()
                    new_pts[i, coord] += direction * step_size
                    if np.all(np.abs(new_pts) <= HALF):
                        new_score = _score(new_pts)
                        if new_score > current_score + 1e-12:
                            pts = new_pts
                            current_score = new_score
                            improved = True
                            break
                else:
                    continue
                break
        if not improved:
            break
    
    return pts


def construct():
    t0 = time.time()
    best_r = 0.0
    best_pts = None
    
    # Strategy: hex-grid greedy with multiple offsets, then local refinement.
    # This is a structural change from the previous direct coordinate optimization.
    
    np.random.seed(42)
    
    # Generate hex grids with different offsets and scales
    offsets_x = [0.0, 0.02, 0.04, 0.01, 0.03]
    offsets_y = [0.0, 0.02, 0.04, 0.01, 0.03]
    scales = [1.0, 0.95, 1.05]
    
    attempts = 0
    max_attempts = 30
    
    for scale in scales:
        for ox in offsets_x:
            for oy in offsets_y:
                if time.time() - t0 > _TIME_LIMIT - 20:
                    break
                if attempts >= max_attempts:
                    break
                
                # Generate hex grid
                grid = _hex_grid_points(offset_x=ox, offset_y=oy, scale=scale)
                
                # Greedy select
                pts = _greedy_select_fast(grid)
                attempts += 1
                
                # Local refinement
                if time.time() - t0 < _TIME_LIMIT - 15:
                    pts = _local_refine(pts, max_iters=100, lr=0.005)
                
                r = _score(pts)
                if r > best_r:
                    best_r = r
                    best_pts = pts.copy()
                
                # If we have time, do a hill-climb on the best
                if time.time() - t0 < _TIME_LIMIT - 10 and best_pts is not None:
                    refined = _hill_climb(best_pts, max_rounds=20, step_size=0.003)
                    r_refined = _score(refined)
                    if r_refined > best_r:
                        best_r = r_refined
                        best_pts = refined.copy()
        
        if time.time() - t0 > _TIME_LIMIT - 20:
            break
    
    # Final hill-climb on the best result if time permits
    if best_pts is not None and time.time() - t0 < _TIME_LIMIT - 5:
        final = _hill_climb(best_pts, max_rounds=30, step_size=0.002)
        r_final = _score(final)
        if r_final > best_r:
            best_r = r_final
            best_pts = final.copy()
    
    # Fallback
    if best_pts is None:
        best_pts = np.random.uniform(-HALF, HALF, (N, 2))
    
    return best_pts
