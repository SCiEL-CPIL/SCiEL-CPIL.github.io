import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from scipy.sparse.csgraph import laplacian as sparse_laplacian
from scipy.sparse import coo_matrix
from scipy.linalg import eigh

N = 300

def _hex_init(n_points, seed=0):
    """Generate a hexagonal lattice and select n_points best points."""
    rng = np.random.default_rng(seed)
    best_pts = None
    best_r = 0.0

    for attempt in range(20):
        base_spacing = 2.0 / (np.sqrt(n_points) + 1.0) * 0.92
        spacing = base_spacing * (0.93 + 0.015 * attempt)

        k = int(1.0 / spacing) + 6
        rows = []

        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0 - spacing / 2.0 - 0.01:
                    rows.append((x, y, r))

        if len(rows) < n_points:
            continue

        pts = np.array([(r[0], r[1]) for r in rows])
        radii = np.array([r[2] for r in rows])
        order = np.argsort(radii, kind="stable")

        if attempt % 3 == 0:
            selected = pts[order[:n_points]]
        elif attempt % 3 == 1:
            top_k = min(len(pts), int(n_points * 1.5))
            top_indices = order[:top_k]
            selected_indices = rng.choice(top_indices, size=n_points, replace=False)
            selected = pts[selected_indices]
        else:
            top_k = min(len(pts), int(n_points * 0.8))
            top_indices = order[:top_k]
            rest_indices = order[top_k:]
            if len(rest_indices) > 0:
                rest_selected = rng.choice(rest_indices, size=n_points - top_k, replace=False)
                selected = np.vstack([pts[top_indices], pts[rest_selected]])
            else:
                selected = pts[order[:n_points]]

        if attempt > 2:
            noise_scale = spacing * 0.03
            selected = selected + rng.normal(0, noise_scale, selected.shape)
            norms = np.linalg.norm(selected, axis=1)
            mask = norms < 1.0 - 0.01
            selected = selected[mask]
            if len(selected) < n_points:
                continue
            selected = selected[:n_points]

        d = cdist(selected, selected)
        np.fill_diagonal(d, np.inf)
        min_pair = d.min() / 2.0
        min_bound = 1.0 - np.linalg.norm(selected, axis=1).max()
        r = min(min_pair, min_bound)

        if r > best_r:
            best_r = r
            best_pts = selected.copy()

    if best_pts is None:
        spacing = 2.0 / (np.sqrt(n_points) + 1.0) * 0.92
        k = int(1.0 / spacing) + 6
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                    rows.append((x, y))
        pts = np.array(rows)
        order = np.argsort(np.linalg.norm(pts, axis=1), kind="stable")
        best_pts = pts[order[:n_points]]

    return best_pts

def _ring_init(n_points, seed=0):
    """Initialize with concentric rings."""
    rng = np.random.default_rng(seed)
    # Estimate number of rings
    # Area ~ pi, each ring has ~ 2*pi*r*dr points
    # Let's try different numbers of rings
    best_pts = None
    best_r = 0.0

    for n_rings in range(8, 20):
        radii = np.linspace(0, 0.95, n_rings)
        if n_rings == 1:
            pts = np.zeros((n_points, 2))
        else:
            # Distribute points among rings proportional to circumference
            circs = 2 * np.pi * radii
            circs[0] = 1e-6  # avoid division by zero
            frac = circs / circs.sum()
            counts = np.round(frac * n_points).astype(int)
            # Adjust to sum to n_points
            diff = n_points - counts.sum()
            idx = np.argsort(circs)
            for i in range(abs(diff)):
                if diff > 0:
                    counts[idx[i]] += 1
                else:
                    if counts[idx[-i-1]] > 0:
                        counts[idx[-i-1]] -= 1

            pts_list = []
            for i, (r, c) in enumerate(zip(radii, counts)):
                if c == 0:
                    continue
                angles = np.linspace(0, 2*np.pi, c, endpoint=False) + rng.uniform(0, 2*np.pi/c)
                xs = r * np.cos(angles)
                ys = r * np.sin(angles)
                pts_list.append(np.column_stack([xs, ys]))
            pts = np.vstack(pts_list) if pts_list else np.zeros((n_points, 2))
            pts = pts[:n_points]

        if len(pts) < n_points:
            continue

        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        min_pair = d.min() / 2.0
        min_bound = 1.0 - np.linalg.norm(pts, axis=1).max()
        r = min(min_pair, min_bound)

        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    if best_pts is None:
        best_pts = _hex_init(n_points, seed=seed)

    return best_pts

def _compute_radius(pts):
    """Compute the packing radius for given points."""
    norms = np.linalg.norm(pts, axis=1)
    min_bound = 1.0 - norms.max()
    if min_bound <= 0:
        return 0.0
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    min_pair = d.min() / 2.0
    return min(min_pair, min_bound)

def _soft_min_logsumexp(values, T):
    """Stable soft minimum: -T * logsumexp(-values/T)"""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    neg_vals = -values / T
    max_neg = np.max(neg_vals)
    if max_neg > 700:
        return values.min()
    exp_shifted = np.exp(neg_vals - max_neg)
    log_sum = np.log(np.sum(exp_shifted)) + max_neg
    return -T * log_sum

def _get_active_edges(pts, k=5):
    """Get k-nearest neighbor edges."""
    n = len(pts)
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    edges = set()
    for i in range(n):
        nn = np.argsort(d[i])[:k]
        for j in nn:
            if i < j:
                edges.add((i, j))
            else:
                edges.add((j, i))
    return list(edges)

def _spectral_laplacian(pts):
    """Compute the normalized graph Laplacian eigenvalues for Delaunay triangulation."""
    n = len(pts)
    try:
        tri = Delaunay(pts)
    except Exception:
        return None, None

    # Build edge list from Delaunay
    edge_set = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i+1, 3):
                a, b = simplex[i], simplex[j]
                if a > b:
                    a, b = b, a
                edge_set.add((a, b))

    if len(edge_set) == 0:
        return None, None

    edges = list(edge_set)
    n_edges = len(edges)

    # Build adjacency matrix
    row = np.zeros(n_edges)
    col = np.zeros(n_edges)
    data = np.ones(n_edges)

    for idx, (i, j) in enumerate(edges):
        row[idx] = i
        col[idx] = j
        data[idx] = 1.0

    # Symmetric adjacency
    A = coo_matrix((data, (row, col)), shape=(n, n))
    A = (A + A.T).tocsr()

    # Degree
    deg = np.array(A.sum(axis=1)).flatten()

    # Laplacian L = D - A
    from scipy.sparse import diags
    D = diags(deg)
    L = (D - A).tocsr()

    # Normalized Laplacian L_sym = I - D^{-1/2} L D^{-1/2}
    # But for spectral connectivity, we use the smallest non-zero eigenvalue of L
    # (algebraic connectivity / Fiedler value)

    # Compute eigenvalues of L (symmetric)
    # For efficiency, only compute a few smallest eigenvalues
    try:
        from scipy.sparse.linalg import eigsh
        # We want the smallest non-zero eigenvalue
        # Use shift-invert to get smallest eigenvalues
        # L is positive semidefinite, smallest eigenvalue is 0 (for connected graph)
        # Next smallest is algebraic connectivity

        # eigsh with smallest algebraic
        # sigma=0 for smallest eigenvalue, but 0 is always there
        # Use sigma=-1 to get eigenvalues near 0? No, that's for complex.
        # Let's just compute the first few smallest eigenvalues
        # L is PSD, so eigenvalues are >= 0
        try:
            evals, evecs = eigsh(L, k=2, which='SM')  # 2 smallest
            # evals[0] should be ~0, evals[1] is algebraic connectivity
            lambda_2 = evals[1]
        except Exception:
            # Fallback: dense
            L_dense = L.toarray()
            evals = np.linalg.eigvalsh(L_dense)
            lambda_2 = evals[1]

        return lambda_2, edges
    except Exception:
        return None, None

def _spectral_objective_and_grad(x, T, edges, lambda_2_target):
    """
    Objective: combine soft-min of constraints with spectral regularizer.
    The spectral term encourages uniform spacing by maximizing algebraic connectivity.
    """
    pts = x.reshape(-1, 2)
    n = len(pts)

    norms = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - norms

    # Pairwise distances for active edges
    pair_half = np.array([np.linalg.norm(pts[i] - pts[j]) / 2.0 for i, j in edges])

    all_vals = np.concatenate([pair_half, boundary])
    obj_soft = -_soft_min_logsumexp(all_vals, T)

    # Spectral regularizer: we want to maximize lambda_2 (algebraic connectivity)
    # This encourages a more uniform, rigid structure
    # Use a smooth approximation: penalize small lambda_2
    # For now, just add a small constant penalty for non-finite
    spectral_term = 0.0
    # Computing lambda_2 in the objective is expensive, so we only use it
    # in a separate optimization phase or as a post-processing step

    return obj_soft, None

def _spectral_refine(pts, n_steps=500, deadline=None):
    """
    Refine the packing using spectral information.
    Use the Delaunay triangulation to identify poorly connected regions
    and apply targeted forces to improve connectivity and uniformity.
    """
    pos = pts.copy().astype(np.float64)
    best_pos = pos.copy()
    best_r = _compute_radius(best_pos)
    n = len(pos)

    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break

        # Recompute Delaunay and spectral info periodically
        if step % 50 == 0:
            lambda_2, edges = _spectral_laplacian(pos)
            if edges is None:
                edges = _get_active_edges(pos, k=5)

        # Force-directed relaxation with spectral guidance
        forces = np.zeros_like(pos)

        # Pairwise repulsion
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)

        # Use active edges for pairwise forces
        for i, j in edges:
            d = dists[i, j]
            if d < 1e-10:
                continue
            target = 2.0 * _compute_radius(pos) * 1.05
            if d < target:
                excess = target - d
                unit_vec = (pos[i] - pos[j]) / d
                forces[i] += 2.0 * excess * unit_vec
                forces[j] -= 2.0 * excess * unit_vec

        # Boundary force
        norms = np.linalg.norm(pos, axis=1)
        for i in range(n):
            if norms[i] > 0.95:
                excess_b = norms[i] - 0.95
                if excess_b > 0:
                    f_mag = 10.0 * excess_b
                    if norms[i] > 1e-10:
                        direction = -pos[i] / norms[i]
                    else:
                        direction = np.array([1.0, 0.0])
                    forces[i] += f_mag * direction

        # Spectral correction: push points toward more uniform distribution
        # Use the eigenvector corresponding to lambda_2 to identify
        # regions that need adjustment
        if step % 50 == 0 and edges is not None:
            try:
                # Recompute eigenvector
                from scipy.sparse.linalg import eigsh
                from scipy.sparse import coo_matrix, diags
                edge_set = set(edges)
                n_edges = len(edge_set)
                if n_edges > 0:
                    row = np.zeros(n_edges)
                    col = np.zeros(n_edges)
                    data = np.ones(n_edges)
                    edges_list = list(edge_set)
                    for idx, (i, j) in enumerate(edges_list):
                        row[idx] = i
                        col[idx] = j
                        data[idx] = 1.0
                    A = coo_matrix((data, (row, col)), shape=(n, n))
                    A = (A + A.T).tocsr()
                    deg = np.array(A.sum(axis=1)).flatten()
                    D = diags(deg)
                    L = (D - A).tocsr()
                    evals, evecs = eigsh(L, k=2, which='SM')
                    # evecs[:, 1] is the Fiedler vector
                    fiedler = evecs[:, 1]
                    # Points with large |fiedler| are at the "ends" of the
                    # low-connectivity mode; push them inward slightly
                    # Points with small |fiedler| are at the "center"
                    # This is a heuristic to encourage more uniform spacing
                    for i in range(n):
                        if abs(fiedler[i]) > 0.5:
                            # Push slightly toward center
                            if norms[i] > 1e-10:
                                forces[i] += 0.01 * (-pos[i] / norms[i])
            except Exception:
                pass

        # Update positions
        lr = 0.005 * (0.999 ** step)
        pos = pos + lr * forces

        # Project back into unit disk
        norms = np.linalg.norm(pos, axis=1)
        mask_out = norms > 0.995
        if np.any(mask_out):
            pos[mask_out] = pos[mask_out] / norms[mask_out, np.newaxis] * 0.994

        # Update best
        if step % 20 == 0:
            r = _compute_radius(pos)
            if r > best_r:
                best_r = r
                best_pos = pos.copy()

    return best_pos, best_r

def _force_relax(pts, n_steps=2000, deadline=None):
    """Fast force-directed relaxation."""
    pos = pts.copy().astype(np.float64)
    best_pos = pos.copy()
    best_r = _compute_radius(best_pos)
    n = len(pos)

    r_est = best_r
    lr = 0.008
    cooldown = 0.998

    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break

        forces = np.zeros_like(pos)

        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)

        target = 2.0 * r_est * 1.02
        mask = dists < target

        safe_dists = np.where(mask, dists, 1.0)
        unit_vecs = diff / safe_dists[:, :, np.newaxis]

        excess = np.where(mask, target - dists, 0.0)
        force_mag = 3.0 * excess / safe_dists

        forces += np.sum(force_mag[:, :, np.newaxis] * unit_vecs, axis=1)

        norms = np.linalg.norm(pos, axis=1)
        wall_margin = 0.08
        for i in range(n):
            if norms[i] > 1.0 - wall_margin:
                excess_b = norms[i] - (1.0 - wall_margin)
                if excess_b > 0:
                    f_mag = 15.0 * excess_b
                    if norms[i] > 1e-10:
                        direction = -pos[i] / norms[i]
                    else:
                        direction = np.array([1.0, 0.0])
                    forces[i] += f_mag * direction

        pos = pos + lr * forces

        norms = np.linalg.norm(pos, axis=1)
        mask_out = norms > 0.995
        if np.any(mask_out):
            pos[mask_out] = pos[mask_out] / norms[mask_out, np.newaxis] * 0.994

        if step % 30 == 0:
            r = _compute_radius(pos)
            if r > best_r:
                best_r = r
                best_pos = pos.copy()
                r_est = r

        lr *= cooldown

    return best_pos, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 148.0

    best_pts = None
    best_r = 0.0

    # Phase 1: Multiple initializations
    inits = [
        _hex_init(N, seed=42),
        _hex_init(N, seed=123),
        _hex_init(N, seed=777),
        _ring_init(N, seed=7),
        _ring_init(N, seed=99),
        _hex_init(N, seed=314),
        _ring_init(N, seed=555),
    ]

    for pts in inits:
        if time.time() > deadline - 40.0:
            break

        r = _compute_radius(pts)

        # Force relaxation
        relaxed_pts, relaxed_r = _force_relax(pts, n_steps=1500, deadline=deadline - 30.0)

        if relaxed_r > r:
            pts = relaxed_pts
            r = relaxed_r

        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    if best_pts is None:
        best_pts = _hex_init(N, seed=0)
        best_r = _compute_radius(best_pts)

    # Phase 2: Spectral refinement
    if time.time() < deadline - 15.0:
        spectral_pts, spectral_r = _spectral_refine(best_pts, n_steps=800, deadline=deadline - 10.0)
        if spectral_r > best_r:
            best_r = spectral_r
            best_pts = spectral_pts.copy()

    # Phase 3: Global optimization with Soft-Min
    if time.time() < deadline - 12.0:
        temperatures = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
        x0 = best_pts.ravel()

        for T in temperatures:
            if time.time() > deadline - 8.0:
                break
            try:
                # Use active edges for sparse optimization
                edges = _get_active_edges(best_pts, k=5)
                result = minimize(
                    lambda x: _objective_and_grad(x, T, edges)[0],
                    x0,
                    jac=lambda x: _objective_and_grad(x, T, edges)[1],
                    method='L-BFGS-B',
                    options={
                        'maxiter': 150,
                        'ftol': 1e-13,
                        'gtol': 1e-11,
                        'maxls': 60,
                        'maxfun': 400,
                    }
                )
                new_pts = result.x.reshape(-1, 2)
                norms = np.linalg.norm(new_pts, axis=1)
                if np.any(norms >= 1.0) or not np.all(np.isfinite(new_pts)):
                    continue
                new_r = _compute_radius(new_pts)
                if new_r > best_r:
                    best_r = new_r
                    best_pts = new_pts.copy()
                    x0 = new_pts.ravel()
            except Exception:
                continue

    # Phase 4: Local Polish with Sparse Gradient
    if time.time() < deadline - 4.0 and best_pts is not None:
        try:
            tri = Delaunay(best_pts)
            edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i+1, 3):
                        a, b = simplex[i], simplex[j]
                        if a > b:
                            a, b = b, a
                        edges.add((a, b))
            edge_list = list(edges)
        except Exception:
            edge_list = _get_active_edges(best_pts, k=5)

        if edge_list:
            edge_dists = [(np.linalg.norm(best_pts[a] - best_pts[b]), (a, b)) for a, b in edge_list]
            edge_dists.sort(key=lambda x: x[0])
            edge_list = [e for _, e in edge_dists[:200]]

            temperatures = [0.0001, 0.00005, 0.00002, 0.00001, 0.000005]
            x0 = best_pts.ravel()

            for T in temperatures:
                if time.time() > deadline - 2.0:
                    break
                try:
                    result = minimize(
                        lambda x: _objective_and_grad(x, T, edge_list)[0],
                        x0,
                        jac=lambda x: _objective_and_grad(x, T, edge_list)[1],
                        method='L-BFGS-B',
                        options={
                            'maxiter': 250,
                            'ftol': 1e-15,
                            'gtol': 1e-13,
                            'maxls': 100,
                            'maxfun': 600,
                        }
                    )
                    new_pts = result.x.reshape(-1, 2)
                    norms = np.linalg.norm(new_pts, axis=1)
                    if np.all(norms < 1.0) and np.all(np.isfinite(new_pts)):
                        new_r = _compute_radius(new_pts)
                        if new_r > best_r:
                            best_r = new_r
                            best_pts = new_pts.copy()
                            x0 = new_pts.ravel()
                except Exception:
                    pass

    # Final validity check
    if best_pts is None:
        best_pts = _hex_init(N, seed=0)
    norms = np.linalg.norm(best_pts, axis=1)
    mask = norms < 1.0
    if not np.all(mask) or not np.all(np.isfinite(best_pts)):
        best_pts = _hex_init(N, seed=0)

    return best_pts

def _objective_and_grad(x, T, edge_indices=None):
    """
    Objective: minimize -softmin(constraints)
    If edge_indices is provided, only consider those pairs for the pairwise term.
    """
    pts = x.reshape(-1, 2)
    n = len(pts)

    norms = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - norms

    if edge_indices is None:
        d = cdist(pts, pts)
        iu = np.triu_indices(n, k=1)
        pair_half = d[iu] / 2.0
        n_pairs = len(pair_half)
    else:
        pair_half = np.array([np.linalg.norm(pts[i] - pts[j]) / 2.0 for i, j in edge_indices])
        n_pairs = len(pair_half)

    all_vals = np.concatenate([pair_half, boundary])

    obj = -_soft_min_logsumexp(all_vals, T)

    neg_vals_T = -all_vals / T
    max_nv = np.max(neg_vals_T)
    if max_nv > 700:
        min_idx = np.argmin(all_vals)
        grad_all = np.zeros_like(all_vals)
        grad_all[min_idx] = -1.0
    else:
        exp_shifted = np.exp(neg_vals_T - max_nv)
        weights = exp_shifted / np.sum(exp_shifted)
        grad_all = -weights

    grad_pts = np.zeros_like(pts)

    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    grad_boundary = -pts / safe_norms[:, np.newaxis] * grad_all[len(pair_half):][:, np.newaxis]
    grad_pts += grad_boundary

    if edge_indices is None:
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        safe_dists = np.where(dists > 1e-10, dists, 1.0)
        unit_vecs = diff / safe_dists[:, :, np.newaxis]

        W = np.zeros((n, n))
        iu = np.triu_indices(n, k=1)
        W[iu] = grad_all[:n_pairs]
        W[iu[1], iu[0]] = grad_all[:n_pairs]

        weighted_unit = W[:, :, np.newaxis] * unit_vecs
        grad_pairs = 0.5 * np.sum(weighted_unit, axis=1)
        grad_pts += grad_pairs
    else:
        for k, (i, j) in enumerate(edge_indices):
            w = grad_all[k]
            if w == 0:
                continue
            diff = pts[i] - pts[j]
            d = np.linalg.norm(diff)
            if d < 1e-10:
                continue
            unit_vec = diff / d
            grad_pts[i] += 0.5 * w * unit_vec
            grad_pts[j] -= 0.5 * w * unit_vec

    return obj, grad_pts.ravel()

from scipy.spatial.distance import cdist
