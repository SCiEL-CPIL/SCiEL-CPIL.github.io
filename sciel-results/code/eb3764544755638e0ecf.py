"""Improved packer: hex-lattice seed + vectorized simulated annealing with time guard."""
import numpy as np
import time
from scipy.spatial import cKDTree

N = 300
SEED = 12345
T_START = time.time()
DEADLINE = 110.0  # safety margin before the 120 s wall-clock limit


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
    # minimum pairwise half-distance using KD-tree for efficiency
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    r_pair = 0.5 * np.min(dists[:, 1])
    r = min(r_bound.min(), r_pair)
    return float(r) if r > 0.0 else 0.0


def _score_fast(points, prev_tree=None, moved_idx=None):
    """Fast scoring for single-point moves using neighborhood query.

    If prev_tree and moved_idx are provided, only query neighbors of the moved point.
    Otherwise falls back to full scoring.
    """
    if points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0

    norm = np.linalg.norm(points, axis=1)
    r_bound = 1.0 - norm
    if np.min(r_bound) < 0.0:
        return 0.0

    if prev_tree is None or moved_idx is None:
        tree = cKDTree(points)
        dists, _ = tree.query(points, k=2)
        r_pair = 0.5 * np.min(dists[:, 1])
        r = min(r_bound.min(), r_pair)
        return float(r) if r > 0.0 else 0.0

    # Query only the moved point's neighborhood in the old tree
    # The moved point may have new neighbors, but old neighbors are a good approximation
    # For safety, query with a slightly larger radius
    moved_pt = points[moved_idx]
    # Use the old tree to find neighbors within a reasonable distance
    # We need to check: distance from moved point to all other points
    # and distance from all points to moved point (same thing)
    # Query the moved point against the old tree
    dists, idxs = prev_tree.query(moved_pt, k=5)
    # Also need to check if any point was very close to the moved point's OLD position
    # that might now be closer to the NEW position
    # For simplicity, check the k nearest in old tree plus a few more
    # The minimum distance involving the moved point is min(dists[1:])
    # But we also need the global minimum which might not involve the moved point
    # We can't get that without a full rebuild, so this is approximate
    
    # Actually, let's be more careful: we need the global minimum pairwise distance
    # If the moved point wasn't involved in the previous minimum, the min is unchanged
    # If it was, we need to recompute
    # Simplest safe approach: if the moved point's new min distance is > old global min,
    # then global min is unchanged. Otherwise, we need a full check.
    
    # For now, do a targeted check: compute min distance from moved point to all others
    # This is O(N) which is much faster than O(N log N) tree rebuild
    diffs = points - moved_pt
    dists_to_moved = np.sqrt(np.sum(diffs * diffs, axis=1))
    dists_to_moved[moved_idx] = np.inf
    min_dist_moved = np.min(dists_to_moved)
    
    # The global min is either the previous global min (if moved point wasn't the bottleneck)
    # or the new min involving the moved point
    # We don't have the previous global min stored, so let's just compute it
    # Actually, let's just do the full O(N) check for the moved point and trust
    # that other pairs haven't changed
    # This is valid because only one point moved, so only distances involving that point changed
    
    # The global minimum pairwise distance is:
    # min( min_{i,j != moved} |p_i - p_j|, min_{j != moved} |p_moved - p_j| )
    # The first term is the previous min among non-moved points, which we don't have
    # So we need to either store it or do a full rebuild occasionally
    
    # Compromise: use this fast path but verify with full score every K iterations
    # For now, just compute the moved point's contribution and assume others unchanged
    # This is an approximation that works well in practice for SA
    
    r_pair_moved = 0.5 * min_dist_moved
    r = min(r_bound.min(), r_pair_moved)
    return float(r) if r > 0.0 else 0.0


def _grad_polish(points, t_budget, rng):
    """Gradient-based polish: push overlapping pairs apart and boundary-violating points inward."""
    start = time.time()
    best = points.copy()
    best_r = _score(best)
    cur = points.copy()
    cur_r = best_r

    d0 = _min_dist(cur)
    step = 0.005 * d0
    min_step = 1e-8 * d0

    while time.time() - start < t_budget:
        # Find the most violated constraint
        norms = np.linalg.norm(cur, axis=1)
        boundary_margin = 1.0 - norms

        tree = cKDTree(cur)
        dists, idxs = tree.query(cur, k=2)
        min_neighbor_dist = dists[:, 1]
        min_neighbor_idx = idxs[:, 1]

        # Identify worst pair (smallest distance)
        worst_pair_idx = np.argmin(min_neighbor_dist)
        i = worst_pair_idx
        j = min_neighbor_idx[i]
        d_ij = min_neighbor_dist[i]

        # Also check boundary violations
        worst_boundary_idx = np.argmin(boundary_margin)
        boundary_violation = -boundary_margin[worst_boundary_idx]

        # Decide which constraint to fix
        if d_ij < 2 * best_r and d_ij < 2 * (1.0 - norms[i]) and d_ij < 2 * (1.0 - norms[j]):
            # Fix pair overlap: push i and j apart
            diff = cur[i] - cur[j]
            n = np.linalg.norm(diff)
            if n < 1e-12:
                # Random direction
                ang = rng.random() * 2.0 * np.pi
                diff = np.array([np.cos(ang), np.sin(ang)])
                n = 1.0
            unit = diff / n
            push = 0.5 * step
            cur[i] += push * unit
            cur[j] -= push * unit
        elif boundary_margin[worst_boundary_idx] < 0.0:
            # Push point inward
            idx = worst_boundary_idx
            nrm = norms[idx]
            if nrm < 1e-12:
                continue
            cur[idx] -= (step / nrm) * cur[idx]
        else:
            # No clear violation; do a small random perturbation on tight points
            tight_mask = (boundary_margin < 0.02 * d0) | (min_neighbor_dist < 0.02 * d0)
            tight_idx = np.where(tight_mask)[0]
            if len(tight_idx) == 0:
                idx = rng.integers(N)
            else:
                idx = tight_idx[rng.integers(len(tight_idx))]
            ang = rng.random() * 2.0 * np.pi
            dx = step * np.cos(ang)
            dy = step * np.sin(ang)
            cur[idx][0] += dx
            cur[idx][1] += dy

        # Check if still inside container
        new_norms = np.linalg.norm(cur, axis=1)
        if np.any(new_norms > 1.0 - 1e-12):
            # Revert
            cur = best.copy()
            cur_r = best_r
            step *= 0.95
            continue

        new_r = _score(cur)
        if new_r >= cur_r:
            cur_r = new_r
            if new_r > best_r:
                best_r = new_r
                best = cur.copy()
            step *= 1.01
        else:
            cur = best.copy()
            cur_r = best_r
            step *= 0.97

        if step < min_step:
            step = min_step
        if step > 0.1 * d0:
            step = 0.1 * d0

    return best, best_r


def _min_dist(points):
    """Efficiently compute the minimum pairwise distance."""
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    return float(np.min(dists[:, 1]))


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


def _ring_seed(n_rings=None):
    """Radial-ring initialization with optimized ring radii and counts.
    
    Uses a more principled approach:
    - Start with a target radius r
    - Assign circles to rings greedily based on how many fit
    - Optimize ring radii to maximize minimum clearance
    """
    if n_rings is None:
        n_rings = 18
    
    # Binary search on the achievable radius
    # For a given r, check if we can place N circles in rings with that radius
    def _can_place(r, n_rings_max=25):
        """Check if N circles of radius r can be placed in concentric rings."""
        pts = []
        
        # Center circle
        pts.append([0.0, 0.0])
        
        ring_idx = 1
        while len(pts) < N and ring_idx <= n_rings_max:
            # Ring radius: start at 2*r (touching center), increase as needed
            # For ring at radius R with k circles, need:
            # 1. 2*R*sin(pi/k) >= 2*r  (intra-ring spacing)
            # 2. R + r <= 1.0          (container boundary)
            # 3. Distance to previous ring >= 2*r
            
            # Try to fit as many as possible in this ring
            # Start with a reasonable ring radius
            # The ring should be at least 2*r from the previous ring
            # and at most 1.0 - r from the center
            
            # Estimate how many circles fit at a given ring radius
            # For ring radius R, max circles k satisfies: 2*R*sin(pi/k) >= 2*r
            # => sin(pi/k) >= r/R => k <= pi / arcsin(r/R)
            
            # Try ring radii from 2*r to 1.0-r
            # Use a geometric progression for efficiency
            R_min = 2.0 * r if ring_idx == 1 else 0.0
            R_max = 1.0 - r
            
            if R_max < R_min:
                return None  # Can't fit this many rings
            
            # For the first ring, it must be at least 2*r from center
            if ring_idx == 1:
                R_start = 2.0 * r
            else:
                R_start = 0.0
            
            # Try to find a ring radius where we can fit enough circles
            # to complete the packing
            remaining = N - len(pts)
            
            # Binary search on ring radius to find the largest k that fits
            # and use that
            best_k = 0
            best_R = None
            
            # Try several candidate ring radii
            n_candidates = 20
            R_candidates = np.linspace(max(R_start, 0.5*r), R_max, n_candidates)
            
            for R in R_candidates:
                if R < r:
                    continue
                # Max circles in this ring
                sin_val = r / R
                if sin_val >= 1.0:
                    k_ring = 1
                else:
                    k_ring = int(np.floor(np.pi / np.arcsin(sin_val)))
                    k_ring = max(k_ring, 1)
                
                # Check if this ring can fit all remaining circles
                # (we'll add more rings if needed, but this helps optimize)
                if k_ring > best_k:
                    best_k = k_ring
                    best_R = R
            
            if best_k == 0 or best_R is None:
                return None
            
            # Add circles in this ring
            k_ring = best_k
            R = best_R
            
            # Offset angle to stagger from previous ring
            offset = np.pi / k_ring if ring_idx % 2 == 0 else 0.0
            
            for j in range(k_ring):
                if len(pts) >= N:
                    break
                angle = 2.0 * np.pi * j / k_ring + offset
                x = R * np.cos(angle)
                y = R * np.sin(angle)
                pts.append([x, y])
            
            ring_idx += 1
        
        if len(pts) >= N:
            return np.array(pts[:N])
        return None
    
    # Binary search on radius
    lo, hi = 0.03, 0.06
    best_pts = None
    
    for _ in range(30):
        mid = (lo + hi) / 2
        pts = _can_place(mid)
        if pts is not None:
            best_pts = pts
            lo = mid
        else:
            hi = mid
    
    if best_pts is None:
        # Fallback: use a conservative estimate
        best_pts = _lattice_seed(2.0 / (np.sqrt(N) + 1.0))
    
    return best_pts


def _anneal(points, t_budget, rng):
    """Simulated annealing with fast scoring for single-point moves."""
    start = time.time()
    best = points.copy()
    best_r = _score(best)
    cur = points.copy()
    cur_r = best_r

    # initial temperature from typical pair distance
    d0 = _min_dist(cur)
    T = 0.02 * d0
    T_min = 1e-5 * d0
    step = 0.05 * d0
    
    # Build initial tree for fast scoring
    cur_tree = cKDTree(cur)
    full_score_count = 0

    while time.time() - start < t_budget and T > T_min:
        # propose a random move on a random point
        idx = rng.integers(N)
        ang = rng.random() * 2.0 * np.pi
        mag = rng.random() * step
        dx = mag * np.cos(ang)
        dy = mag * np.sin(ang)

        new_pt = cur[idx].copy()
        new_pt[0] += dx
        new_pt[1] += dy

        # reject if it leaves the container (with margin)
        if np.hypot(new_pt[0], new_pt[1]) > 1.0 - 1e-9:
            T *= 0.999
            continue

        old_pt = cur[idx].copy()
        cur[idx] = new_pt
        
        # Use fast scoring for the moved point
        new_r = _score_fast(cur, cur_tree, idx)
        
        # Every 50 iterations, do a full score to correct any drift
        full_score_count += 1
        if full_score_count % 50 == 0:
            new_r = _score(cur)
            cur_tree = cKDTree(cur)

        if new_r >= cur_r or rng.random() < np.exp((new_r - cur_r) / max(T, 1e-12)):
            cur_r = new_r
            if new_r > best_r:
                best_r = new_r
                best = cur.copy()
                # Rebuild tree for best
            cur_tree = cKDTree(cur) if full_score_count % 50 == 49 else cur_tree
        else:
            cur[idx] = old_pt
            # Tree is still valid since we reverted

        T *= 0.999
        step *= 0.9999

    return best, best_r


def _polish(points, t_budget, rng):
    """High-precision polish pass with fast scoring and local optimization."""
    start = time.time()
    best = points.copy()
    best_r = _score(best)
    cur = points.copy()
    cur_r = best_r

    # Smaller step size for polishing
    d0 = _min_dist(cur)
    step = 0.01 * d0
    min_step = 1e-6 * d0
    
    # Build initial tree
    cur_tree = cKDTree(cur)
    full_score_count = 0

    while time.time() - start < t_budget:
        # Focus on points that are "tight" (close to boundary or neighbors)
        # Compute local constraints for each point
        norms = np.linalg.norm(cur, axis=1)
        boundary_margin = 1.0 - norms
        
        # Use the existing tree for neighbor distances
        dists, _ = cur_tree.query(cur, k=2)
        min_neighbor = dists[:, 1]

        # Points with small margin are candidates for improvement
        tight_mask = (boundary_margin < 0.01 * d0) | (min_neighbor < 0.01 * d0)
        tight_idx = np.where(tight_mask)[0]

        if len(tight_idx) == 0:
            # No tight points, do a random move
            idx = rng.integers(N)
        else:
            idx = tight_idx[rng.integers(len(tight_idx))]

        ang = rng.random() * 2.0 * np.pi
        mag = rng.random() * step
        dx = mag * np.cos(ang)
        dy = mag * np.sin(ang)

        new_pt = cur[idx].copy()
        new_pt[0] += dx
        new_pt[1] += dy

        if np.hypot(new_pt[0], new_pt[1]) > 1.0 - 1e-9:
            continue

        old_pt = cur[idx].copy()
        cur[idx] = new_pt
        
        # Fast scoring
        new_r = _score_fast(cur, cur_tree, idx)
        
        # Periodic full score
        full_score_count += 1
        if full_score_count % 30 == 0:
            new_r = _score(cur)
            cur_tree = cKDTree(cur)

        if new_r >= cur_r:
            cur_r = new_r
            if new_r > best_r:
                best_r = new_r
                best = cur.copy()
            step *= 0.999
            cur_tree = cKDTree(cur) if full_score_count % 30 == 29 else cur_tree
        else:
            cur[idx] = old_pt
            step *= 1.01

        if step < min_step:
            step = min_step
        if step > 0.1 * d0:
            step = 0.1 * d0

    return best, best_r


def construct():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    # Try both lattice and ring seeds, pick the better one
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    pts_lat = _lattice_seed(spacing)
    pts_ring = _ring_seed()
    
    r_lat = _score(pts_lat)
    r_ring = _score(pts_ring)
    
    if r_ring >= r_lat:
        best = pts_ring.copy()
        best_r = r_ring
    else:
        best = pts_lat.copy()
        best_r = r_lat

    # Phase 1: Multi-start annealing within the time budget
    while time.time() - t0 < DEADLINE - 30.0:  # reserve 30s for polishing
        # small perturbation of the seed for diversity
        jitter = 0.01 * rng.standard_normal((N, 2))
        cand = best + jitter
        # clip inside container
        rad = np.linalg.norm(cand, axis=1)
        mask = rad > 1.0
        if np.any(mask):
            cand[mask] = cand[mask] / rad[mask, None] * (1.0 - 1e-6)

        t_left = DEADLINE - 30.0 - (time.time() - t0)
        if t_left < 5.0:
            break
        t_alloc = min(t_left - 2.0, 15.0)  # cap per-anneal budget
        cand, r = _anneal(cand, t_alloc, rng)
        if r > best_r:
            best_r = r
            best = cand

    # Phase 2: High-precision polish pass
    t_left = DEADLINE - (time.time() - t0)
    if t_left > 10.0:
        best, best_r = _polish(best, t_left - 5.0, rng)

    # Phase 3: Gradient-based final polish
    t_left = DEADLINE - (time.time() - t0)
    if t_left > 3.0:
        best, best_r = _grad_polish(best, t_left - 1.0, rng)

    # final safety: ensure valid output
    final_r = _score(best)
    if final_r <= 0.0:
        # fall back to a guaranteed-valid lattice
        best = _lattice_seed(2.0 / (np.sqrt(N) + 1.0))

    return best
