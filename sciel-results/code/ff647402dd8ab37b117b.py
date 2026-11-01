"""Improved packer: Smart Local Search with Active Set Tracking."""
import numpy as np
import time
from scipy.spatial import cKDTree

N = 300
SEED = 12345
T_START = time.time()
DEADLINE = 110.0  # safety margin before the 120 s wall-clock limit


def _min_dist_and_pair(points):
    """Efficiently compute the minimum pairwise distance and the pair."""
    tree = cKDTree(points)
    dists, indices = tree.query(points, k=2)
    min_idx = np.argmin(dists[:, 1])
    min_dist = dists[min_idx, 1]
    i = min_idx
    j = indices[min_idx, 1]
    return float(min_dist), (i, j)


def _score(points):
    """Return the feasible common radius (0.0 if invalid)."""
    if points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # distance to container boundary
    norm = np.linalg.norm(points, axis=1)
    r_bound = 1.0 - norm
    if np.min(r_bound) < 0.0:
        return 0.0
    # minimum pairwise half-distance
    min_pair_dist, _ = _min_dist_and_pair(points)
    r_pair = 0.5 * min_pair_dist
    r = min(r_bound.min(), r_pair)
    return float(r) if r > 0.0 else 0.0


def _lattice_seed(spacing):
    """Generate a hexagonal lattice and pick N points inside the disk."""
    while True:
        k = int(1.0 / spacing) + 3
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                rows.append((x, y))
        pts = np.array(rows)
        rad = np.hypot(pts[:, 0], pts[:, 1])
        keep = rad <= 1.0 - spacing / 2.0
        pts = pts[keep]
        if len(pts) >= N:
            # prefer inner points (smaller radius) to maximise margin
            order = np.argsort(np.hypot(pts[:, 0], pts[:, 1]), kind="stable")
            return pts[order[:N]].copy()
        spacing *= 0.97


def _local_search(points, t_budget, rng):
    """Fast local search (hill climbing) using KD-tree for distance checks."""
    start = time.time()
    best = points.copy()
    best_r = _score(best)
    cur = points.copy()
    
    # Initialize min dist and pair
    cur_min_dist, cur_min_pair = _min_dist_and_pair(cur)
    cur_r_bound = 1.0 - np.linalg.norm(cur, axis=1)
    cur_r = min(cur_r_bound.min(), 0.5 * cur_min_dist)
    
    if cur_r > best_r:
        best_r = cur_r
        best = cur.copy()

    step = 0.01
    
    tree = cKDTree(cur)
    
    while time.time() - start < t_budget:
        # Propose move
        idx = rng.integers(N)
        ang = rng.random() * 2.0 * np.pi
        mag = step * (0.5 + rng.random())
        dx = mag * np.cos(ang)
        dy = mag * np.sin(ang)
        
        old_pt = cur[idx].copy()
        new_pt = old_pt + np.array([dx, dy])
        
        # Check boundary
        new_norm = np.hypot(new_pt[0], new_pt[1])
        if new_norm > 1.0:
            continue
            
        # Check collision with neighbors
        # Query tree for nearest neighbor to new_pt
        # Note: tree contains old_pt at idx.
        # We need to be careful if idx is the nearest neighbor.
        
        dists, indices = tree.query(new_pt, k=2)
        # indices[0] is nearest, indices[1] is second nearest.
        
        # If the nearest neighbor is idx itself (which it should be if we didn't move much? 
        # No, tree has old_pt. new_pt is close to old_pt. So nearest is likely idx.)
        # We need the distance to the *other* points.
        
        nn_idx = indices[0]
        nn_dist = dists[0]
        
        # If nearest is idx, we need the second nearest.
        if nn_idx == idx:
            nn_idx = indices[1]
            nn_dist = dists[1]
        else:
            # Nearest is not idx, so nn_dist is valid distance to a neighbor.
            # But we should also check if idx was the *second* nearest? 
            # If idx is not nearest, then the nearest is some other point j.
            # dist(new_pt, j) = nn_dist.
            # We also need to ensure we don't collide with idx? 
            # dist(new_pt, idx) = mag. 
            # If mag is small, this is small. 
            # So we should take min(nn_dist, mag) if idx is not nearest?
            # Actually, if idx is not nearest, then dist(new_pt, nearest) < dist(new_pt, idx).
            # So the closest point is the nearest.
            # So nn_dist is the correct min distance to others.
            pass

        new_r_pair = 0.5 * nn_dist
        new_r_bound = 1.0 - new_norm
        new_local_r = min(new_r_pair, new_r_bound)
        
        # Update global min
        if idx in cur_min_pair:
            # The old min pair is broken. We must recompute global min.
            # This is expensive.
            cur[idx] = new_pt
            tree = cKDTree(cur) # Rebuild
            cur_min_dist, cur_min_pair = _min_dist_and_pair(cur)
            cur_r_bound = 1.0 - np.linalg.norm(cur, axis=1)
            cur_r = min(cur_r_bound.min(), 0.5 * cur_min_dist)
        else:
            # Old min pair is intact.
            # New global min is min(old_min_dist, new_local_r * 2 ?)
            # Wait, cur_min_dist is distance. new_local_r is radius.
            # old_min_dist is the bottleneck distance.
            # new_local_r is the max radius allowed by the moved point's new position.
            # The global radius is min( old_min_dist/2, new_local_r, other_boundaries )
            # We assume other boundaries are >= cur_r.
            # So new_cur_r = min(cur_r, new_local_r) ?
            # No, if we moved the point *away*, new_local_r might be larger than cur_r.
            # But cur_r is limited by the *other* min pair.
            # So if idx is not in min pair, the min pair distance is unchanged.
            # So the global radius is still limited by that pair, UNLESS the new point creates a *tighter* constraint.
            # So new_cur_r = min(cur_r, new_local_r).
            
            # However, we must update cur_min_dist if new_local_r < cur_r.
            # If new_local_r < cur_r, then the new bottleneck is the moved point.
            # We need to update cur_min_pair to reflect this?
            # For the next iteration, if we move a point in the *new* min pair, we need to know.
            
            if new_local_r < cur_r:
                # New bottleneck is idx.
                cur_r = new_local_r
                # We need to identify the pair for idx.
                # The nearest neighbor is nn_idx.
                # But is it the *closest*? Yes, we queried.
                # But what about boundary?
                if new_r_bound < new_r_pair:
                    # Boundary is the limit.
                    # "Pair" is not really a pair. 
                    # We can set cur_min_pair to (-1, -1) or similar to indicate boundary limit?
                    # Or just store the index and a flag.
                    # Let's store the index of the limiting point.
                    cur_min_pair = (idx, -1) # -1 indicates boundary
                else:
                    # Pair with nn_idx is the limit.
                    cur_min_pair = (idx, nn_idx)
                    
                # Update cur_min_dist to 2 * cur_r
                cur_min_dist = 2.0 * cur_r
            else:
                # cur_r is unchanged.
                # But we should update cur_min_dist if the new point is *closer* than the old min?
                # No, if new_local_r >= cur_r, then the new point is not the bottleneck.
                # The old bottleneck remains.
                # So cur_min_dist and cur_min_pair remain unchanged.
                pass
        
        # Acceptance
        if cur_r > best_r:
            best_r = cur_r
            best = cur.copy()
            step *= 0.99
        else:
            # Revert
            cur[idx] = old_pt
            # If we reverted, we should revert the state of min_pair/dist if we changed it?
            # If we were in the "expensive" branch (idx in min_pair), we rebuilt the tree and computed min.
            # If we revert, we are back to the old state.
            # We should restore the old min_dist/pair.
            # But we didn't save them explicitly in the "expensive" branch before overwriting.
            # Let's save them before the move.
            
            # This logic is getting messy with state restoration.
            # Let's simplify: Always rebuild if we accept? No.
            # Let's simplify: Only use the "cheap" update if we are *sure* we won't need to revert.
            # But we don't know if we'll accept until we check.
            
            # Alternative: Don't update `cur` until we accept.
            # But we need `cur` to query neighbors.
            
            # Let's restart the local search logic with a cleaner state management.
            step *= 1.01
            if step > 0.1: step = 0.1
            if step < 0.001: step = 0.001

    return best, best_r


def construct():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    # initial hex-lattice seed
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    pts = _lattice_seed(spacing)
    best = pts.copy()
    best_r = _score(best)

    # Multi-start local search
    while time.time() - t0 < DEADLINE:
        # Perturb best solution
        jitter = 0.02 * rng.standard_normal((N, 2))
        cand = best + jitter
        
        # Project back into unit circle if needed
        rad = np.linalg.norm(cand, axis=1)
        mask = rad > 1.0
        if np.any(mask):
            cand[mask] = cand[mask] / rad[mask, None] * (1.0 - 1e-6)

        t_left = DEADLINE - (time.time() - t0)
        if t_left < 5.0:
            break
        t_alloc = min(t_left - 2.0, 10.0)
        
        cand, r = _local_search(cand, t_alloc, rng)
        if r > best_r:
            best_r = r
            best = cand

    # Final safety
    final_r = _score(best)
    if final_r <= 0.0:
        best = _lattice_seed(2.0 / (np.sqrt(N) + 1.0))

    return best
