import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 16
D = 2

def _pairwise_distances(pts):
    """Compute all pairwise Euclidean distances."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))

def _score(pts):
    """Compute (max/min)^2 for pairwise distances."""
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    dmin = d.min()
    dmax = d.max()
    if dmin <= 1e-12:
        return np.inf
    return (dmax / dmin) ** 2

def _smooth_objective(pts_flat, tau):
    """
    Smooth approximation of the log-ratio of max to min distance.
    Minimizing this is equivalent to minimizing the ratio.
    We use log-sum-exp for max and min of log-distances.
    """
    pts = pts_flat.reshape(N, 2)
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, 1.0) # Avoid log(0)
    
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    d_valid = d[mask]
    
    if np.any(d_valid < 1e-12):
        return 1e10
        
    log_d = np.log(d_valid)
    
    # Numerical stability: subtract max/min
    max_log_d = np.max(log_d)
    min_log_d = np.min(log_d)
    
    # Softmax of log_d
    exp_pos = np.exp(tau * (log_d - max_log_d))
    soft_max = max_log_d + np.log(np.sum(exp_pos)) / tau
    
    # Softmin of log_d
    exp_neg = np.exp(-tau * (log_d - min_log_d))
    soft_min = min_log_d - np.log(np.sum(exp_neg)) / tau
    
    # Objective: soft_max - soft_min
    return soft_max - soft_min

def _make_hexagonal_cluster(n, spread=6):
    """Generate a hexagonal lattice cluster of n points centered at origin."""
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3) / 2])

    pts = []
    for i in range(-spread, spread + 1):
        for j in range(-spread, spread + 1):
            p = i * v1 + j * v2
            pts.append(p)
    pts = np.array(pts)

    # Center the cluster
    pts -= pts.mean(axis=0)

    # Sort by distance to center and pick the closest n
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    return pts[order[:n]]

def _make_circular_config(n, radius=1.0):
    """Points equally spaced on a circle, plus center if odd, or two concentric rings."""
    if n == 16:
        # 1 on center + 7 on inner ring + 8 on outer ring
        pts = [np.array([0.0, 0.0])]
        r1 = 1.0
        r2 = 2.0
        for k in range(7):
            theta = 2 * np.pi * k / 7
            pts.append(r1 * np.array([np.cos(theta), np.sin(theta)]))
        for k in range(8):
            theta = 2 * np.pi * k / 8 + np.pi / 8
            pts.append(r2 * np.array([np.cos(theta), np.sin(theta)]))
        return np.array(pts)
    # Generic fallback
    angles = 2 * np.pi * np.arange(n) / n
    return radius * np.column_stack([np.cos(angles), np.sin(angles)])

def _make_grid_config(n):
    """4x4 grid, centered."""
    x, y = np.meshgrid(np.arange(4), np.arange(4))
    pts = np.column_stack([x.ravel(), y.ravel()]).astype(float)
    pts -= pts.mean(axis=0)
    return pts

def _make_random_config(n, seed=0):
    """Random points in unit square, centered."""
    rng = np.random.default_rng(seed)
    pts = rng.random((n, 2))
    pts -= pts.mean(axis=0)
    return pts

def _make_concentric_rings(n):
    """Create points on concentric rings with optimized radii and angles."""
    if n == 16:
        # Try: 1 center, 5 on r=1, 5 on r=1.5, 5 on r=2.0
        pts = [np.array([0.0, 0.0])]
        rings = [(5, 1.0), (5, 1.5), (5, 2.0)]
        for count, r in rings:
            for k in range(count):
                theta = 2 * np.pi * k / count + np.pi / count
                pts.append(r * np.array([np.cos(theta), np.sin(theta)]))
        return np.array(pts)
    return _make_circular_config(n)

def _make_optimized_hexagonal(n):
    """Optimized hexagonal arrangement for 16 points."""
    if n == 16:
        # Hexagonal packing: 1 center, 6 in first ring, 9 in second ring (partial)
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.5, np.sqrt(3) / 2])
        
        pts = [np.array([0.0, 0.0])]
        
        # First ring: 6 points
        for k in range(6):
            theta = 2 * np.pi * k / 6
            pts.append(np.array([np.cos(theta), np.sin(theta)]))
        
        # Second ring: 9 points (hexagonal lattice positions)
        second_ring = [
            (2, 0), (1, 1), (0, 2), (-1, 1), (-2, 0), (-1, -1),
            (0, -2), (1, -1), (2, 0)  # Some duplicates, will dedupe
        ]
        seen = set()
        for i, j in second_ring:
            p = i * v1 + j * v2
            key = (round(p[0], 6), round(p[1], 6))
            if key not in seen:
                seen.add(key)
                pts.append(p)
        
        # Take first 16
        return np.array(pts[:16])
    return _make_hexagonal_cluster(n)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    best_pts = None
    best_score = np.inf

    def update_best(pts, score):
        nonlocal best_pts, best_score
        if score < best_score:
            best_score = score
            best_pts = pts.copy()

    # 1. Generate diverse initial candidates
    candidates = [
        _make_hexagonal_cluster(N),
        _make_circular_config(N),
        _make_grid_config(N),
        _make_concentric_rings(N),
        _make_optimized_hexagonal(N),
        _make_random_config(N, seed=1),
        _make_random_config(N, seed=2),
        _make_random_config(N, seed=3),
        _make_random_config(N, seed=4),
        _make_random_config(N, seed=5),
    ]

    # Evaluate all candidates
    for c in candidates:
        s = _score(c)
        update_best(c, s)

    # 2. Lattice-based Swap Search (Inspiration 1)
    # This explores the discrete structure of the hexagonal lattice which is often optimal for packing.
    if time.time() < deadline - 30:
        # Generate a large hexagonal lattice
        R = 10
        lattice_pts = []
        for q in range(-R, R + 1):
            for r in range(-R, R + 1):
                x = q + 0.5 * r
                y = (np.sqrt(3) / 2) * r
                lattice_pts.append((x, y))
        lattice_pts = np.array(lattice_pts)
        
        # Sort by distance to origin
        dists_from_origin = np.sum(lattice_pts**2, axis=1)
        sorted_indices = np.argsort(dists_from_origin)
        
        # Pool of candidate indices
        pool_size = 200
        candidate_indices = sorted_indices[:pool_size]
        
        # Initial subset: top 16
        current_indices = sorted_indices[:N].copy()
        
        def score_subset(indices):
            sub_pts = lattice_pts[indices]
            # Centering helps with symmetry but score is translation invariant.
            # However, keeping them centered might help if we were optimizing, but here we just pick.
            diff = sub_pts[:, np.newaxis, :] - sub_pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            iu = np.triu_indices(N, 1)
            d = dists[iu]
            min_d = np.min(d)
            max_d = np.max(d)
            if min_d < 1e-8:
                return 1e10
            return (max_d / min_d)**2

        current_score = score_subset(current_indices)
        in_subset = set(current_indices)
        outside_candidates = [i for i in candidate_indices if i not in in_subset]
        
        improved = True
        while improved and time.time() < deadline - 30:
            improved = False
            # Try swapping each point in subset with each outside candidate
            # We iterate through current indices and try swapping with outside
            # To speed up, we can check if the swap improves things.
            # Simple hill climbing: accept any improvement.
            
            # To avoid O(N * Pool) full score recalculations being too slow, 
            # we rely on N=16 being small.
            # Score calc is ~120 pairwise distances. 
            # 16 * 184 swaps * 120 dists ~ 350k ops per pass. 
            # Python loops are slow. Let's limit passes or use numpy vectorization for swaps?
            # Vectorizing swaps is tricky. Let's just do it and see time.
            
            for i_idx in range(N):
                if time.time() > deadline - 30:
                    break
                i = current_indices[i_idx]
                for j in outside_candidates:
                    if time.time() > deadline - 30:
                        break
                    # Swap
                    new_indices = current_indices.copy()
                    new_indices[i_idx] = j
                    new_score = score_subset(new_indices)
                    if new_score < current_score:
                        current_score = new_score
                        current_indices = new_indices
                        in_subset = set(current_indices)
                        outside_candidates = [k for k in candidate_indices if k not in in_subset]
                        improved = True
                        break # Restart outer loop logic effectively by breaking inner and letting while loop restart
            if improved:
                continue
            else:
                break
        
        if current_score < best_score:
            best_pts = lattice_pts[current_indices].copy()
            best_score = current_score
            update_best(best_pts, best_score)

    # 3. Simulated Annealing for global exploration
    def simulated_annealing(pts_init, time_limit, seed=42):
        """Run simulated annealing to explore the search space."""
        rng = np.random.default_rng(seed)
        current_pts = pts_init.copy()
        current_score = _score(current_pts)
        best_sa_pts = current_pts.copy()
        best_sa_score = current_score
        
        T = 0.1  # Initial temperature
        T_min = 0.0001
        cooling = 0.995
        max_steps = 50000
        
        for step in range(max_steps):
            if time.time() > time_limit:
                break
            if T < T_min:
                break
                
            # Generate perturbation
            noise_scale = T * 0.5
            new_pts = current_pts + rng.normal(0, noise_scale, size=current_pts.shape)
            
            new_score = _score(new_pts)
            
            # Accept if better or with probability exp(-(new-old)/T)
            if new_score < current_score:
                current_pts = new_pts
                current_score = new_score
                if new_score < best_sa_score:
                    best_sa_score = new_score
                    best_sa_pts = new_pts.copy()
            else:
                delta = new_score - current_score
                if delta < 0:
                    current_pts = new_pts
                    current_score = new_score
                else:
                    prob = np.exp(-delta / T)
                    if rng.random() < prob:
                        current_pts = new_pts
                        current_score = new_score
            
            T *= cooling
        
        return best_sa_pts, best_sa_score

    # Run simulated annealing from top candidates
    if time.time() < deadline - 30:
        # Sort candidates by score
        scored_candidates = [(c, _score(c)) for c in candidates]
        scored_candidates.sort(key=lambda x: x[1])
        
        # Run SA from top 3 candidates
        for c, s in scored_candidates[:3]:
            if time.time() > deadline - 25:
                break
            try:
                sa_pts, sa_score = simulated_annealing(c, deadline - 25, seed=42)
                update_best(sa_pts, sa_score)
            except Exception:
                pass

    # 4. Global Search using Differential Evolution
    def de_objective(pts_flat):
        # Use a moderate tau for DE to keep it smooth but representative
        return _smooth_objective(pts_flat, tau=10.0)

    if time.time() < deadline - 20:
        try:
            bounds = [(-5, 5)] * (N * 2)
            result = differential_evolution(
                de_objective,
                bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-5,
                polish=False,
                init='sobol',
            )
            if result.fun < np.inf:
                new_pts = result.x.reshape(N, 2)
                new_score = _score(new_pts)
                update_best(new_pts, new_score)
        except Exception:
            pass

    # 5. Local Refinement using L-BFGS-B on the smooth objective
    def local_refine(pts_init, time_limit):
        """Refine a configuration using L-BFGS-B with annealing tau."""
        current_pts = pts_init.copy()
        current_score = _score(current_pts)
        
        # Annealing schedule: start with low tau (smooth), increase to high tau (sharp)
        taus = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
        
        for tau in taus:
            if time.time() > time_limit:
                break
                
            # Define the objective for this specific tau
            def obj_flat(x):
                return _smooth_objective(x, tau)
                
            try:
                res = minimize(
                    obj_flat,
                    current_pts.ravel(),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14}
                )
                new_pts = res.x.reshape(N, 2)
                new_score = _score(new_pts)
                
                if new_score < current_score:
                    current_pts = new_pts
                    current_score = new_score
            except Exception:
                pass
        
        return current_pts, current_score

    # Refine top candidates
    all_candidates = candidates.copy()
    if best_pts is not None:
        all_candidates.append(best_pts)
    
    # Sort by raw score to prioritize best ones
    all_candidates.sort(key=lambda x: _score(x))
    
    # Refine the top 4 distinct candidates
    refined_count = 0
    for c in all_candidates:
        if time.time() > deadline - 10:
            break
        if refined_count >= 4:
            break
            
        # Avoid refining exact duplicates
        is_dup = False
        for r in all_candidates[:refined_count]:
            if np.allclose(c, r, atol=1e-5):
                is_dup = True
                break
        if is_dup:
            continue
            
        try:
            refined_pts, refined_score = local_refine(c, deadline - 10)
            update_best(refined_pts, refined_score)
            refined_count += 1
        except Exception:
            pass

    # 6. Final Polish with Nelder-Mead on the raw objective
    if best_pts is not None and time.time() < deadline - 3:
        try:
            def raw_obj(x):
                pts = x.reshape(N, 2)
                return _score(pts)
            
            res = minimize(
                raw_obj,
                best_pts.ravel(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'maxfev': 20000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            if res.fun < best_score:
                new_pts = res.x.reshape(N, 2)
                new_score = _score(new_pts)
                update_best(new_pts, new_score)
        except Exception:
            pass

    # 7. Additional local refinement if time permits
    if best_pts is not None and time.time() < deadline - 2:
        try:
            # Run another round of L-BFGS-B refinement with higher tau
            def high_tau_obj(x):
                return _smooth_objective(x, tau=1000.0)
            
            res = minimize(
                high_tau_obj,
                best_pts.ravel(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            new_pts = res.x.reshape(N, 2)
            new_score = _score(new_pts)
            if new_score < best_score:
                update_best(new_pts, new_score)
        except Exception:
            pass

    if best_pts is None:
        return _make_hexagonal_cluster(N)

    return best_pts
