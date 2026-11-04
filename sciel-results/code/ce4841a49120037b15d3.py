import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 300

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill

    def compute_score(pts):
        """Compute the packing radius score."""
        tree = cKDTree(pts)
        k = min(20, N - 1)
        dists_k, idx_k = tree.query(pts, k=k + 1)
        rad = np.linalg.norm(pts, axis=1)
        bound_dists = 1.0 - rad
        
        min_pair_dist = np.inf
        for i in range(N):
            for j in range(1, k + 1):
                d = dists_k[i, j]
                if d > 1e-10:
                    if d < min_pair_dist:
                        min_pair_dist = d
        
        min_bound = np.min(bound_dists)
        return min(min_pair_dist / 2.0, min_bound)

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

    def _objective_and_grad(x, T, edge_indices=None):
        """
        Objective: minimize -softmin(constraints)
        """
        pts = x.reshape(-1, 2)
        n = len(pts)
        
        norms = np.linalg.norm(pts, axis=1)
        boundary = 1.0 - norms
        
        if edge_indices is None:
            d = cdist(pts, pts)
            iu = np.triu_indices(n, k=1)
            pair_half = d[iu] / 2.0
        else:
            pair_half = np.array([np.linalg.norm(pts[i] - pts[j]) / 2.0 for i, j in edge_indices])
            
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
            n_pairs = len(pair_half)
            iu = np.triu_indices(n, k=1)
            W[iu] = grad_all[:n_pairs]
            W[iu[1], iu[0]] = grad_all[:n_pairs]
            
            weighted_unit = W[:, :, np.newaxis] * unit_vecs
            grad_pairs = 0.5 * np.sum(weighted_unit, axis=1)
            grad_pts += grad_pairs
        else:
            for k, (i, j) in enumerate(edge_indices):
                w = grad_all[k]
                if w == 0: continue
                diff = pts[i] - pts[j]
                d = np.linalg.norm(diff)
                if d < 1e-10: continue
                unit_vec = diff / d
                grad_pts[i] += 0.5 * w * unit_vec
                grad_pts[j] -= 0.5 * w * unit_vec
                
        return obj, grad_pts.ravel()

    def _force_relax(pts, n_steps=2000, deadline=None):
        """Fast force-directed relaxation."""
        pos = pts.copy().astype(np.float64)
        best_pos = pos.copy()
        best_r = compute_radius(pos)
        n = len(pos)
        
        r_est = best_r
        lr = 0.008
        cooldown = 0.9995
        
        for step in range(n_steps):
            if deadline is not None and time.time() > deadline:
                break
                
            forces = np.zeros_like(pos)
            
            diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            
            target = 2.0 * r_est
            too_close = dists < target * 1.5
            safe_dists = np.where(too_close, dists, 1.0)
            
            excess = np.where(too_close, target - dists, 0.0)
            force_mag = np.where(too_close, excess / safe_dists, 0.0)
            
            unit_vecs = diff / safe_dists[:, :, np.newaxis]
            
            forces += np.sum(force_mag[:, :, np.newaxis] * unit_vecs, axis=1)
            
            norms = np.linalg.norm(pos, axis=1)
            wall_margin = 0.08
            in_wall = norms > (1.0 - wall_margin)
            if np.any(in_wall):
                excess_b = np.where(in_wall, norms - (1.0 - wall_margin), 0.0)
                f_mag = 15.0 * excess_b
                safe_norms = np.where(norms > 1e-10, norms, 1.0)
                boundary_force = -pos / safe_norms[:, np.newaxis] * f_mag[:, np.newaxis]
                forces += boundary_force
            
            pos = pos + lr * forces
            
            norms = np.linalg.norm(pos, axis=1)
            mask_out = norms > 1.0
            if np.any(mask_out):
                pos[mask_out] = pos[mask_out] / norms[mask_out, np.newaxis] * 0.998
                
            if step % 50 == 0:
                r = compute_radius(pos)
                if r > best_r:
                    best_r = r
                    best_pos = pos.copy()
                    r_est = min(r, 0.06)
                    
            lr *= cooldown
            
        return best_pos

    def compute_radius(pts):
        norms = np.linalg.norm(pts, axis=1)
        min_bound = 1.0 - norms.max()
        if min_bound <= 0:
            return 0.0
        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        min_pair = d.min() / 2.0
        return min(min_pair, min_bound)

    def _hex_init(n_points, seed=0):
        """Generate a hexagonal lattice and select n_points best points."""
        rng = np.random.default_rng(seed)
        best_pts = None
        best_r = 0.0
        
        for attempt in range(15):
            base_spacing = 2.0 / (np.sqrt(n_points) + 1.0) * 0.95
            spacing = base_spacing * (0.92 + 0.02 * attempt)
            
            k = int(1.0 / spacing) + 6
            lattice_pts = []
            for j in range(-k, k + 1):
                for i in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * np.sqrt(3.0) / 2.0
                    lattice_pts.append((x, y))
            lattice_pts = np.array(lattice_pts)
            
            angle = rng.uniform(0, 2 * np.pi)
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            rotated = np.column_stack([
                lattice_pts[:, 0] * cos_a - lattice_pts[:, 1] * sin_a,
                lattice_pts[:, 0] * sin_a + lattice_pts[:, 1] * cos_a
            ])
            
            norms = np.linalg.norm(rotated, axis=1)
            margin = spacing * 0.5
            mask = norms <= 1.0 - margin
            if mask.sum() < n_points:
                continue
            
            candidates = rotated[mask]
            cand_norms = norms[mask]
            
            if attempt > 0:
                candidates = candidates + rng.normal(0, spacing * 0.03, candidates.shape)
                norms2 = np.linalg.norm(candidates, axis=1)
                valid = norms2 < 1.0
                if valid.sum() < n_points:
                    continue
                candidates = candidates[valid]
                cand_norms = norms2[valid]
            
            if len(candidates) > n_points:
                order = np.argsort(cand_norms, kind="stable")
                selected = candidates[order[:n_points]]
            else:
                selected = candidates
                
            d = cdist(selected, selected)
            np.fill_diagonal(d, np.inf)
            min_pair = d.min() / 2.0
            min_bound = 1.0 - np.linalg.norm(selected, axis=1).max()
            r = min(min_pair, min_bound)
            
            if r > best_r:
                best_r = r
                best_pts = selected.copy()
                
        if best_pts is None:
            spacing = 2.0 / (np.sqrt(n_points) + 1.0) * 0.95
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

    def _get_delaunay_edges(pts):
        """Get Delaunay triangulation edges as list of (i, j) pairs."""
        try:
            tri = Delaunay(pts)
            edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i+1, 3):
                        a, b = simplex[i], simplex[j]
                        if a > b:
                            a, b = b, a
                        edges.add((a, b))
            return list(edges)
        except Exception:
            # Fallback: use k-nearest neighbors
            tree = cKDTree(pts)
            k = min(8, N-1)
            _, idx = tree.query(pts, k=k+1)
            edges = set()
            for i in range(N):
                for j_idx in idx[i, 1:]:
                    a, b = i, j_idx
                    if a > b:
                        a, b = b, a
                    edges.add((a, b))
            return list(edges)

    def _topological_swap(pts, edges, rng):
        """
        Perform a topological swap: pick two non-adjacent points and swap them.
        This can escape local minima by changing the global geometry while
        preserving local neighborhood structure.
        """
        n = len(pts)
        # Build adjacency set for quick lookup
        adj = [set() for _ in range(n)]
        for i, j in edges:
            adj[i].add(j)
            adj[j].add(i)
        
        # Pick two random points that are not adjacent
        for _ in range(100):
            i = rng.integers(0, n)
            j = rng.integers(0, n)
            if i == j or j in adj[i]:
                continue
            # Swap positions
            new_pts = pts.copy()
            new_pts[i], new_pts[j] = new_pts[j], new_pts[i]
            return new_pts
        return pts

    # Main optimization loop
    best_pts = None
    best_score = 0.0
    
    # Phase 1: Generate multiple hexagonal lattice initializations
    base_spacing = np.sqrt(2 * np.pi / (np.sqrt(3) * N))
    
    # Try several different initializations
    init_candidates = []
    for seed in range(10):
        if time.time() - start_time > time_limit * 0.3:
            break
        pts = _hex_init(N, seed=seed)
        init_candidates.append(pts)
    
    # Also add a random initialization
    if time.time() - start_time < time_limit * 0.3:
        rng = np.random.default_rng(42)
        pts_random = rng.uniform(-0.95, 0.95, (N, 2))
        norms = np.linalg.norm(pts_random, axis=1)
        mask = norms > 0.95
        pts_random[mask] *= 0.95 / norms[mask, np.newaxis]
        init_candidates.append(pts_random)
    
    # Phase 2: Optimize each initialization
    for init_idx, pts_init in enumerate(init_candidates):
        if time.time() - start_time > time_limit * 0.85:
            break
        
        # First, do a quick force relaxation to get a better starting point
        deadline = start_time + time_limit * 0.1
        pts_relaxed = _force_relax(pts_init, n_steps=1000, deadline=deadline)
        
        # Compute initial score
        score_init = compute_score(pts_relaxed)
        if score_init > best_score:
            best_score = score_init
            best_pts = pts_relaxed.copy()
        
        # Get Delaunay edges for the relaxed configuration
        edges = _get_delaunay_edges(pts_relaxed)
        
        # Run L-BFGS-B optimization with soft-min objective
        T_start = 0.005
        T_end = 0.0005
        state = {'T': T_start, 'iter': 0}
        
        time_budget = min(8.0, (time_limit * 0.85 - (time.time() - start_time)) * 0.8)
        end_time = start_time + time_budget
        
        def time_callback(xk):
            state['iter'] += 1
            if time.time() > end_time:
                return True
            if state['iter'] % 30 == 0:
                state['T'] = max(state['T'] * 0.85, T_end)
            return False
        
        x0 = pts_relaxed.flatten()
        
        try:
            def obj_wrapper(x):
                return _objective_and_grad(x, state['T'])[0]
            
            def grad_wrapper(x):
                return _objective_and_grad(x, state['T'])[1]
            
            result = minimize(
                obj_wrapper,
                x0,
                jac=grad_wrapper,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000},
                callback=time_callback
            )
        except Exception:
            result = None
        
        if result is not None:
            pts_opt = result.x.reshape(N, 2)
            rad = np.linalg.norm(pts_opt, axis=1)
            mask_out = rad > 1.0
            if np.any(mask_out):
                pts_opt[mask_out] /= rad[mask_out, np.newaxis]
            
            score = compute_score(pts_opt)
            if score > best_score:
                best_score = score
                best_pts = pts_opt.copy()
        
        # Phase 3: Topological swaps to escape local minima
        if best_pts is not None and time.time() - start_time < time_limit * 0.9:
            rng = np.random.default_rng(123 + init_idx)
            current_pts = best_pts.copy()
            current_score = best_score
            
            for swap_iter in range(5):
                if time.time() - start_time > time_limit * 0.9:
                    break
                
                # Get new Delaunay edges
                edges = _get_delaunay_edges(current_pts)
                
                # Perform a topological swap
                swapped_pts = _topological_swap(current_pts, edges, rng)
                
                # Optimize the swapped configuration
                T_start = 0.003
                T_end = 0.0003
                state = {'T': T_start, 'iter': 0}
                
                time_budget = min(4.0, (time_limit * 0.9 - (time.time() - start_time)) * 0.5)
                end_time = start_time + time_budget
                
                def time_callback_swap(xk):
                    state['iter'] += 1
                    if time.time() > end_time:
                        return True
                    if state['iter'] % 20 == 0:
                        state['T'] = max(state['T'] * 0.85, T_end)
                    return False
                
                x0 = swapped_pts.flatten()
                
                try:
                    def obj_wrapper_swap(x):
                        return _objective_and_grad(x, state['T'])[0]
                    
                    def grad_wrapper_swap(x):
                        return _objective_and_grad(x, state['T'])[1]
                    
                    result_swap = minimize(
                        obj_wrapper_swap,
                        x0,
                        jac=grad_wrapper_swap,
                        method='L-BFGS-B',
                        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 3000},
                        callback=time_callback_swap
                    )
                except Exception:
                    result_swap = None
                
                if result_swap is not None:
                    pts_opt = result_swap.x.reshape(N, 2)
                    rad = np.linalg.norm(pts_opt, axis=1)
                    mask_out = rad > 1.0
                    if np.any(mask_out):
                        pts_opt[mask_out] /= rad[mask_out, np.newaxis]
                    
                    score = compute_score(pts_opt)
                    if score > current_score:
                        current_score = score
                        current_pts = pts_opt.copy()
                        if score > best_score:
                            best_score = score
                            best_pts = pts_opt.copy()
    
    # Final polish with high T if time remains
    if best_pts is not None and time.time() - start_time < time_limit * 0.95:
        remaining = time_limit * 0.98 - time.time()
        if remaining > 2.0:
            T_start = 0.0005
            T_end = 0.0001
            state = {'T': T_start, 'iter': 0}
            
            end_time = start_time + min(remaining, 10.0)
            
            def time_callback_final(xk):
                state['iter'] += 1
                if time.time() > end_time:
                    return True
                if state['iter'] % 10 == 0:
                    state['T'] = max(state['T'] * 0.9, T_end)
                return False
            
            x0 = best_pts.flatten()
            
            try:
                def obj_wrapper_final(x):
                    return _objective_and_grad(x, state['T'])[0]
                
                def grad_wrapper_final(x):
                    return _objective_and_grad(x, state['T'])[1]
                
                result_final = minimize(
                    obj_wrapper_final,
                    x0,
                    jac=grad_wrapper_final,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 3000},
                    callback=time_callback_final
                )
            except Exception:
                result_final = None
            
            if result_final is not None:
                pts_opt = result_final.x.reshape(N, 2)
                rad = np.linalg.norm(pts_opt, axis=1)
                mask_out = rad > 1.0
                if np.any(mask_out):
                    pts_opt[mask_out] /= rad[mask_out, np.newaxis]
                
                score = compute_score(pts_opt)
                if score > best_score:
                    best_score = score
                    best_pts = pts_opt.copy()
    
    # Ensure all points are inside the circle
    if best_pts is None:
        best_pts = _hex_init(N, seed=0)
    
    rad = np.linalg.norm(best_pts, axis=1)
    mask_out = rad > 1.0
    if np.any(mask_out):
        best_pts[mask_out] /= rad[mask_out, np.newaxis]
    
    return best_pts
