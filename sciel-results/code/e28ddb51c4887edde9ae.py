# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time
from scipy.optimize import minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    start_time = time.time()
    deadline = start_time + 620.0  # Safety margin of 10 seconds

    def compute_score(points):
        if points.shape != (16, 2):
            return -np.inf
        d = pdist(points)
        if d.size == 0:
            return 0.0
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio

    def compute_score_smooth(points, beta=50.0):
        """
        Smooth surrogate for the min/max ratio using log-sum-exp.
        We maximize (d_min/d_max)^2.
        Using soft-min and soft-max approximations.
        """
        if points.shape != (16, 2):
            return -np.inf
        d = pdist(points)
        if d.size == 0:
            return 0.0
        
        # Soft minimum: approximates min(d)
        # min(d) = -max(-d)
        neg_d = -d
        max_neg_d = np.max(neg_d)
        soft_min = -1.0 / beta * (max_neg_d + np.log(np.sum(np.exp(neg_d - max_neg_d))))
        
        # Soft maximum: approximates max(d)
        max_d = np.max(d)
        soft_max = 1.0 / beta * (max_d + np.log(np.sum(np.exp(d - max_d))))
        
        if soft_max <= 0:
            return 0.0
        ratio = soft_min / soft_max
        return ratio * ratio

    # ------------------------------------------------------------------
    # 1. Generate diverse initial configurations
    # ------------------------------------------------------------------
    configs = []

    # 1.1 4x4 Grid (Standard)
    grid = np.array([[i, j] for i in range(4) for j in range(4)], dtype=float)
    grid = (grid - 1.5) / 1.5
    configs.append(grid)

    # 1.2 Regular 16-gon
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))

    # 1.3 Two concentric squares (4 inner, 12 outer - actually 4+12=16)
    # Inner: 4 points
    inner_r = 0.5
    outer_r = 1.0
    inner = np.array([[inner_r, inner_r], [-inner_r, inner_r],
                      [-inner_r, -inner_r], [inner_r, -inner_r]])
    outer = []
    # 4 corners + 8 midpoints? No, let's use a 12-gon for outer to be safe, or just optimize 12 points on a circle.
    # Let's stick to the previous working config: 4 inner, 12 outer (12-gon)
    outer_a = np.linspace(0, 2*np.pi, 12, endpoint=False)
    outer = np.column_stack([outer_r*np.cos(outer_a), outer_r*np.sin(outer_a)])
    configs.append(np.vstack([inner, outer]))

    # 1.4 Two concentric octagons (8+8)
    inner_a = np.linspace(0, 2*np.pi, 8, endpoint=False)
    outer_a = inner_a + np.pi/8
    inner_r4 = 0.55
    outer_r4 = 1.0
    inner4 = np.column_stack([inner_r4*np.cos(inner_a), inner_r4*np.sin(inner_a)])
    outer4 = np.column_stack([outer_r4*np.cos(outer_a), outer_r4*np.sin(outer_a)])
    configs.append(np.vstack([inner4, outer4]))

    # 1.5 Center + 5-ring + 10-ring
    c = np.array([[0.0, 0.0]])
    r5 = 0.5
    a5 = np.linspace(0, 2*np.pi, 5, endpoint=False)
    ring5 = np.column_stack([r5*np.cos(a5), r5*np.sin(a5)])
    r10 = 1.0
    a10 = np.linspace(0, 2*np.pi, 10, endpoint=False)
    ring10 = np.column_stack([r10*np.cos(a10), r10*np.sin(a10)])
    configs.append(np.vstack([c, ring5, ring10]))

    # 1.6 Hexagonal lattice selection
    s = 1.0
    h = s * np.sqrt(3) / 2
    lattice = []
    for row in range(-3, 4):
        for col in range(-3, 4):
            x = col * s + (row % 2) * s / 2
            y = row * h
            lattice.append([x, y])
    lattice = np.array(lattice)
    centroid = np.mean(lattice, axis=0)
    dists = np.sum((lattice - centroid) ** 2, axis=1)
    idx = np.argsort(dists)[:16]
    cfg6 = lattice[idx]
    cfg6 = (cfg6 - np.mean(cfg6, axis=0))
    norm = np.max(np.max(np.abs(cfg6), axis=0))
    if norm > 0:
        cfg6 = cfg6 / norm
    configs.append(cfg6)

    # 1.7 Two concentric hexagons + extras
    hex_a = np.linspace(0, 2*np.pi, 6, endpoint=False)
    inner_hex_r = 0.5
    outer_hex_r = 1.0
    inner_hex = np.column_stack([inner_hex_r*np.cos(hex_a), inner_hex_r*np.sin(hex_a)])
    outer_hex = np.column_stack([outer_hex_r*np.cos(hex_a + np.pi/6), outer_hex_r*np.sin(hex_a + np.pi/6)])
    extra = np.array([[0, 1.0], [1.0, 0], [0, -1.0], [-1.0, 0.0]])
    configs.append(np.vstack([inner_hex, outer_hex, extra]))

    # 1.8 Random perturbations of grid
    for seed in range(3):
        np.random.seed(seed * 7 + 1)
        pert = grid + 0.1 * np.random.randn(16, 2)
        pert = (pert - np.mean(pert, axis=0))
        std = np.std(pert)
        if std > 0:
            pert = pert / std
        configs.append(pert)

    # 1.9 Fibonacci spiral
    n = 16
    golden = (1 + np.sqrt(5)) / 2
    indices = np.arange(n)
    r = np.sqrt(indices / n)
    theta = 2 * np.pi * indices / golden
    fib_pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    configs.append(fib_pts)

    # 1.10 Ring + inner ring
    outer_ring_a = np.linspace(0, 2*np.pi, 12, endpoint=False)
    inner_ring_a = outer_ring_a + np.pi/12
    outer_ring_r = 1.0
    inner_ring_r = 0.45
    outer_ring = np.column_stack([outer_ring_r*np.cos(outer_ring_a), outer_ring_r*np.sin(outer_ring_a)])
    inner_ring = np.column_stack([inner_ring_r*np.cos(inner_ring_a), inner_ring_r*np.sin(inner_ring_a)])
    configs.append(np.vstack([inner_ring, outer_ring]))

    # ------------------------------------------------------------------
    # 2. NEW STRUCTURAL STRATEGY: Symmetry-Constrained Projective/Radial Warps
    # ------------------------------------------------------------------
    # The idea: Instead of treating all 32 coordinates as free variables,
    # we exploit the D4 symmetry of the 4x4 grid.
    # We define a radial warp function f(r) that maps the radius of the grid points.
    # Grid points have radii:
    # (0,0) -> 0 (center)
    # (0,1), (1,0), etc -> 1 (8 points)
    # (1,1), etc -> sqrt(2) (4 points)
    # (0,2), etc -> 2 (4 points)
    # (1,2), etc -> sqrt(5) (8 points)
    # (2,2), etc -> sqrt(8) (4 points)
    # Wait, the grid is 4x4, indices 0..3. Center is 1.5.
    # Relative coords: -1.5, -0.5, 0.5, 1.5
    # Radii squared:
    # (0.5, 0.5) -> 0.5 (4 points)
    # (0.5, 1.5) -> 2.5 (8 points)
    # (1.5, 1.5) -> 4.5 (4 points)
    
    # Let's parameterize the warp by scaling factors for each "ring" of points.
    # Ring 1: 4 points at (±0.5, ±0.5)
    # Ring 2: 8 points at (±0.5, ±1.5) and (±1.5, ±0.5)
    # Ring 3: 4 points at (±1.5, ±1.5)
    
    # We can optimize 3 parameters: s1, s2, s3 for the scaling of each ring.
    # Additionally, we can add a global rotation (though grid is symmetric, rotation breaks the axis alignment which might help if we allow non-orthogonal distortion, but let's stick to radial scaling first).
    # Actually, a better projective-like warp: x' = x * (1 + k * r^2).
    # Let's try optimizing 3 scaling factors for the 3 concentric squares of the 4x4 grid.
    
    def generate_warp_config(s1, s2, s3):
        """
        Generate a configuration by scaling the 3 rings of the 4x4 grid.
        s1: scale for inner 4 points (±0.5, ±0.5)
        s2: scale for middle 8 points (±0.5, ±1.5), (±1.5, ±0.5)
        s3: scale for outer 4 points (±1.5, ±1.5)
        """
        pts = []
        # Inner 4
        for x in [0.5, -0.5]:
            for y in [0.5, -0.5]:
                pts.append([x * s1, y * s1])
        # Middle 8
        for x in [0.5, -0.5]:
            for y in [1.5, -1.5]:
                pts.append([x * s2, y * s2])
        for x in [1.5, -1.5]:
            for y in [0.5, -0.5]:
                pts.append([x * s2, y * s2])
        # Outer 4
        for x in [1.5, -1.5]:
            for y in [1.5, -1.5]:
                pts.append([x * s3, y * s3])
        
        return np.array(pts, dtype=float)

    # Add a few initial guesses for the warp
    configs.append(generate_warp_config(1.0, 1.0, 1.0)) # Standard grid
    configs.append(generate_warp_config(0.8, 1.0, 1.2)) # Slightly expanded outer
    configs.append(generate_warp_config(1.2, 1.0, 0.8)) # Slightly contracted outer
    configs.append(generate_warp_config(0.5, 1.0, 1.5)) # Strong expansion

    # ------------------------------------------------------------------
    # 3. Evaluate initial configs
    # ------------------------------------------------------------------
    best_score = -np.inf
    best_points = None
    for cfg in configs:
        score = compute_score(cfg)
        if score > best_score:
            best_score = score
            best_points = cfg.copy()

    # ------------------------------------------------------------------
    # 4. Optimization Phases
    # ------------------------------------------------------------------

    # Phase 1: Smooth Nelder-Mead from best config
    if best_points is not None and time.time() < deadline - 150:
        def objective_smooth(pts_flat, beta=50.0):
            pts = pts_flat.reshape(16, 2)
            s = compute_score_smooth(pts, beta)
            if not np.isfinite(s):
                return 1e10
            return -s

        try:
            result = minimize(objective_smooth, best_points.flatten(), method='Nelder-Mead',
                             options={'maxiter': 5000, 'fatol': 1e-14, 'xatol': 1e-14})
            opt_pts = result.x.reshape(16, 2)
            opt_score = compute_score(opt_pts)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts.copy()
        except Exception:
            pass

    # Phase 2: Coarse-to-fine beta annealing
    if best_points is not None and time.time() < deadline - 120:
        betas = [20.0, 40.0, 80.0, 150.0, 300.0]
        for beta in betas:
            if time.time() > deadline - 30:
                break
            def objective_beta(pts_flat):
                pts = pts_flat.reshape(16, 2)
                s = compute_score_smooth(pts, beta)
                if not np.isfinite(s):
                    return 1e10
                return -s

            try:
                result = minimize(objective_beta, best_points.flatten(), method='Nelder-Mead',
                                 options={'maxiter': 3000, 'fatol': 1e-14, 'xatol': 1e-14})
                opt_pts = result.x.reshape(16, 2)
                opt_score = compute_score(opt_pts)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_pts.copy()
            except Exception:
                pass

    # Phase 3: Optimizing the Symmetry-Constrained Warp (New Strategy)
    # We specifically optimize the 3 parameters of the radial warp on the 4x4 grid structure.
    # This is a lower-dimensional optimization problem (3D) which is easier to solve globally.
    if time.time() < deadline - 90:
        def warp_objective(params):
            s1, s2, s3 = params
            # Ensure positive scales
            s1 = max(0.1, s1)
            s2 = max(0.1, s2)
            s3 = max(0.1, s3)
            pts = generate_warp_config(s1, s2, s3)
            s = compute_score(pts)
            if not np.isfinite(s):
                return 1e10
            return -s

        # Try multiple starting points for the warp optimization
        warp_starts = [
            [1.0, 1.0, 1.0],
            [0.8, 1.0, 1.2],
            [1.2, 1.0, 0.8],
            [0.5, 1.5, 1.0],
            [1.5, 0.5, 1.0],
            [1.0, 0.8, 1.2],
            [1.0, 1.2, 0.8]
        ]
        
        for start in warp_starts:
            if time.time() > deadline - 45:
                break
            try:
                result = minimize(warp_objective, start, method='Nelder-Mead',
                                 options={'maxiter': 2000, 'fatol': 1e-12, 'xatol': 1e-12})
                opt_params = result.x
                opt_pts = generate_warp_config(opt_params[0], opt_params[1], opt_params[2])
                opt_score = compute_score(opt_pts)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_pts.copy()
            except Exception:
                pass

    # Phase 4: Random restarts with smooth objective
    if time.time() < deadline - 60:
        for seed in range(15):
            if time.time() > deadline - 30:
                break
            np.random.seed(seed * 31 + 17)
            rand_pts = np.random.randn(16, 2)
            rand_pts = rand_pts - np.mean(rand_pts, axis=0)
            std = np.std(rand_pts)
            if std > 0:
                rand_pts = rand_pts / std

            def obj_restart(pts_flat):
                pts = pts_flat.reshape(16, 2)
                s = compute_score_smooth(pts, 80.0)
                if not np.isfinite(s):
                    return 1e10
                return -s

            try:
                result = minimize(obj_restart, rand_pts.flatten(), method='Nelder-Mead',
                                 options={'maxiter': 2000, 'fatol': 1e-13, 'xatol': 1e-13})
                opt_pts = result.x.reshape(16, 2)
                opt_score = compute_score(opt_pts)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_pts.copy()
            except Exception:
                pass

    # Phase 5: Gradient-based refinement using L-BFGS-B with smooth objective
    if best_points is not None and time.time() < deadline - 30:
        def objective_lbfgs(pts_flat, beta=200.0):
            pts = pts_flat.reshape(16, 2)
            s = compute_score_smooth(pts, beta)
            if not np.isfinite(s):
                return 1e10
            return -s

        try:
            result = minimize(objective_lbfgs, best_points.flatten(), method='L-BFGS-B',
                             options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-15})
            opt_pts = result.x.reshape(16, 2)
            opt_score = compute_score(opt_pts)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts.copy()
        except Exception:
            pass

    # Phase 6: High-precision polish with raw objective
    if best_points is not None and time.time() < deadline - 10:
        def objective_polish(pts_flat):
            pts = pts_flat.reshape(16, 2)
            s = compute_score(pts)
            if not np.isfinite(s):
                return 1e10
            return -s

        try:
            result = minimize(objective_polish, best_points.flatten(), method='Nelder-Mead',
                             options={'maxiter': 10000, 'fatol': 1e-18, 'xatol': 1e-18})
            opt_pts = result.x.reshape(16, 2)
            opt_score = compute_score(opt_pts)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts.copy()
        except Exception:
            pass

    # Phase 7: Final micro-polish
    if best_points is not None and time.time() < deadline - 2:
        def objective_micro(pts_flat):
            pts = pts_flat.reshape(16, 2)
            s = compute_score(pts)
            if not np.isfinite(s):
                return 1e10
            return -s

        try:
            result = minimize(objective_micro, best_points.flatten(), method='Nelder-Mead',
                             options={'maxiter': 20000, 'fatol': 1e-20, 'xatol': 1e-20})
            opt_pts = result.x.reshape(16, 2)
            opt_score = compute_score(opt_pts)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts.copy()
        except Exception:
            pass

    if best_points is None:
        best_points = configs[0]

    return best_points

# EVOLVE-BLOCK-END
