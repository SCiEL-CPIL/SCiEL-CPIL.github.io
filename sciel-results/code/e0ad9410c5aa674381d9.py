<<<<<<< SEARCH
def _score_local(points, idx, old_pos, new_pos, tree, cached_dists):
    """
    Fast score after moving point idx from old_pos to new_pos.
    Uses cached KDTree distances for all other pairs; only recomputes
    distances involving idx. Returns the new feasible radius.
    """
    # New boundary for moved point
    new_norm = np.linalg.norm(new_pos)
    # The minimum boundary is min over all points of (1 - |c_i|)
    # For un-moved points, their (1-|c_i|) is unchanged; for moved point it's (1-new_norm)
    # We need the global min boundary. We can track: old_min_boundary was cached.
    # But simpler: compute new boundary for moved point, and the rest is unchanged.
    # Actually we need min over ALL i of (1 - |c_i|).
    # Let's just compute it properly but cheaply:
    # norms for all other points are unchanged; only idx changes.
    # We'll pass in the old min_boundary and old norm of idx.
    # For simplicity in this version, recompute norms (cheap for N=300).
    norms = np.linalg.norm(points, axis=1)
    min_boundary = float(1.0 - np.max(norms))
    if min_boundary <= 0:
        return 0.0

    # For pair distances: all pairs not involving idx are unchanged.
    # Pairs involving idx: distance from new_pos to points[j] for all j != idx.
    # We need min of: (cached min pair dist among non-idx pairs), and (min_j!=idx |new_pos - points[j]|)
    # cached_dists[:, 1] gives nearest neighbor distance for each point.
    # The global min pair distance is min(cached_dists[:, 1]).
    # But if the moved point was involved in the minimum pair, we need to update.
    # Simpler approach: compute all distances from new_pos to other points.
    diff = points - new_pos
    diff[idx, :] = 0.0  # exclude self
    dists_to_others = np.linalg.norm(diff, axis=1)
    dists_to_others[idx] = np.inf  # exclude self
    min_new_pair = float(np.min(dists_to_others))

    # For pairs not involving idx, the min is the minimum of cached_dists[:,1]
    # but we must exclude pairs where idx was the nearest neighbor of some point,
    # or where idx was the second nearest. This is complex.
    # For N=300, just compute the full min pair distance cheaply:
    # min_pair = min( min over j!=idx of |new_pos - points[j]|, min over pairs not involving idx )
    # The latter: from cached_dists, the min among all points' nearest-neighbor dist,
    # but we need to be careful if idx was the nearest neighbor of some point.
    # Given N=300, let's just do a vectorized min over all pairs not involving idx.
    # Actually, for N=300, a full pairwise is 300*299/2 = 44850 ops, still fast.
    # But let's try the cached approach: the min pair dist not involving idx is
    # the min of cached_dists[:,1] for points whose nearest neighbor is not idx,
    # plus for points whose nearest neighbor IS idx, their second-nearest distance.
    # This is getting complex. For N=300, let's just compute it directly but vectorized.

    # Full pairwise min excluding self, but only need min:
    # min over all i<j of |points[i]-points[j]|
    # Vectorized: compute all pairwise distances, take min.
    # For N=300 this is fine.
    # Actually, let's use the fact that only distances involving idx changed.
    # min_pair = min( min_j!=idx |new_pos - points[j]|, min_pair_not_involving_idx )
    # min_pair_not_involving_idx: from cached_dists, for each point j, if its nearest
    # neighbor is not idx, then cached_dists[j,1] is valid. If its nearest neighbor IS idx,
    # then we need its second nearest (which we don't have cached).
    # Simplest correct approach for N=300: just compute the full min.
    # It's 44850 distance computations, ~0.1ms.

    # Even simpler: since we're doing this in a tight loop, let's precompute
    # the min pair distance not involving idx once per outer iteration.
    # But for now, let's just do the full computation - it's still fast.

    # Compute min pair distance properly:
    # We have dists_to_others (dist from new_pos to all others).
    # For pairs not involving idx: use cached info.
    # Let's just compute min of:
    #   a) min_j!=idx |new_pos - points[j]|  = min_new_pair
    #   b) min over i<j, i,j != idx of |points[i]-points[j]|
    # For (b), we can use: min of cached_dists[:,1] where the nearest neighbor is not idx,
    # and for points whose nearest neighbor is idx, we need their second nearest.
    # To keep it simple and correct, let's just do the full pairwise min for (b).
    # With N=300, this is acceptable.

    # Actually, let me reconsider. The full _score with KDTree takes maybe 0.5ms.
    # The local score with full pairwise min takes similar time.
    # The real speedup comes from NOT rebuilding the KDTree.
    # Let me just compute the min pair distance via broadcasting for N=300.

    # Min pair distance not involving idx:
    mask = np.ones(N, dtype=bool)
    mask[idx] = False
    sub_pts = points[mask]
    n_sub = len(sub_pts)
    if n_sub >= 2:
        # Compute min pairwise distance in sub_pts
        # Using KDTree on sub_pts would be fast
        sub_tree = cKDTree(sub_pts)
        sub_dists, _ = sub_tree.query(sub_pts, k=2)
        min_sub_pair = float(np.min(sub_dists[:, 1]))
    else:
        min_sub_pair = np.inf

    min_pair = min(min_new_pair, min_sub_pair)
    return min(min_boundary, min_pair / 2.0)
=======
def _score_local(points, idx, old_pos, new_pos, tree, cached_dists):
    """
    Fast score after moving point idx from old_pos to new_pos.
    Uses cached KDTree distances for all other pairs; only recomputes
    distances involving idx. Returns the new feasible radius.
    """
    norms = np.linalg.norm(points, axis=1)
    min_boundary = float(1.0 - np.max(norms))
    if min_boundary <= 0:
        return 0.0

    diff = points - new_pos
    diff[idx, :] = 0.0
    dists_to_others = np.linalg.norm(diff, axis=1)
    dists_to_others[idx] = np.inf
    min_new_pair = float(np.min(dists_to_others))

    mask = np.ones(N, dtype=bool)
    mask[idx] = False
    sub_pts = points[mask]
    n_sub = len(sub_pts)
    if n_sub >= 2:
        sub_tree = cKDTree(sub_pts)
        sub_dists, _ = sub_tree.query(sub_pts, k=2)
        min_sub_pair = float(np.min(sub_dists[:, 1]))
    else:
        min_sub_pair = np.inf

    min_pair = min(min_new_pair, min_sub_pair)
    return min(min_boundary, min_pair / 2.0)


def _soft_min_objective(x, beta=50.0):
    """
    Smooth surrogate for the packing objective.
    Maximizes min(min_pair_dist/2, min_boundary) using soft-min.
    Returns negative of soft-min (to minimize).
    """
    pts = x.reshape(N, 2)
    norms = np.linalg.norm(pts, axis=1)
    boundary_terms = 1.0 - norms  # shape (N,)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dists = np.min(dists, axis=1)  # shape (N,), min distance for each point
    pair_terms = min_dists / 2.0  # shape (N,)
    
    # Combine both sets of terms
    all_terms = np.concatenate([boundary_terms, pair_terms])
    
    # Soft min via log-sum-exp
    m = np.min(all_terms)
    if m < -50:
        return -m  # very infeasible
    log_sum = np.log(np.sum(np.exp(-beta * (all_terms - m)))) / beta
    soft_min_val = m - log_sum
    return -soft_min_val


def _soft_min_gradient(x, beta=50.0):
    """Gradient of _soft_min_objective."""
    pts = x.reshape(N, 2)
    norms = np.linalg.norm(pts, axis=1)
    boundary_terms = 1.0 - norms
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dists = np.min(dists, axis=1)
    pair_terms = min_dists / 2.0
    
    all_terms = np.concatenate([boundary_terms, pair_terms])
    m = np.min(all_terms)
    if m < -50:
        # Return gradient pointing toward feasible region
        grad = np.zeros_like(x)
        for i in range(N):
            if norms[i] > 0.5:
                grad[2*i] = -pts[2*i] / (2.0 * norms[i])
                grad[2*i+1] = -pts[2*i+1] / (2.0 * norms[i])
        return grad
    
    exps = np.exp(-beta * (all_terms - m))
    weights = exps / np.sum(exps)
    
    grad = np.zeros_like(x)
    
    # Gradient of boundary terms: d/dx_i (1 - |x_i|) = -x_i / |x_i|
    for i in range(N):
        w = weights[i]
        if norms[i] > 1e-10:
            grad[2*i] = -w * pts[2*i] / norms[i]
            grad[2*i+1] = -w * pts[2*i+1] / norms[i]
    
    # Gradient of pair terms: more complex, approximate with nearest neighbor direction
    for i in range(N):
        w = weights[N + i]
        if w < 1e-10:
            continue
        # Find nearest neighbor
        j = np.argmin(dists[i])
        if j == i:
            continue
        d = dists[i, j]
        if d < 1e-10:
            continue
        # d/dx_i (|x_i - x_j|/2) = (x_i - x_j) / (2*d)
        grad[2*i] += -w * (pts[i, 0] - pts[j, 0]) / (2.0 * d)
        grad[2*i+1] += -w * (pts[i, 1] - pts[j, 1]) / (2.0 * d)
        # d/dx_j (|x_i - x_j|/2) = (x_j - x_i) / (2*d)
        grad[2*j] += -w * (pts[j, 0] - pts[i, 0]) / (2.0 * d)
        grad[2*j+1] += -w * (pts[j, 1] - pts[i, 1]) / (2.0 * d)
    
    return grad


def _refine_lbfgs(points, deadline, max_iter=200, beta_start=30.0, beta_end=100.0):
    """
    Smooth L-BFGS-B refinement using soft-min surrogate.
    """
    from scipy.optimize import minimize
    
    x0 = points.flatten().copy()
    best_x = x0.copy()
    best_obj = _soft_min_objective(x0, beta_start)
    
    # Progressive beta schedule
    n_phases = 3
    for phase in range(n_phases):
        if time.time() > deadline:
            break
        beta = beta_start + (beta_end - beta_start) * phase / max(1, n_phases - 1)
        phase_deadline = deadline if phase == n_phases - 1 else min(deadline, time.time() + 30.0)
        
        res = minimize(
            _soft_min_objective,
            x0,
            jac=_soft_min_gradient,
            method='L-BFGS-B',
            args=(beta,),
            options={'maxiter': max_iter // n_phases, 'ftol': 1e-12, 'gtol': 1e-10},
        )
        
        if res.fun < best_obj:
            best_obj = res.fun
            best_x = res.x.copy()
        
        x0 = res.x.copy()
        if time.time() > phase_deadline - 5:
            break
    
    pts = best_x.reshape(N, 2)
    # Check feasibility
    r = _score(pts)
    if r > 0:
        return pts, r
    return None, 0.0
>>>>>>> REPLACE

