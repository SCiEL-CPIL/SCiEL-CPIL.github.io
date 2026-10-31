import numpy as np
import time
from scipy.optimize import minimize

N = 16
D = 2

def _particle_equilibrium_init(repulsion_power=4, attraction_power=2, iterations=500, dt=0.01):
    """
    Generate initial configuration via particle equilibrium:
    - Short-range repulsion (strong at small distances)
    - Long-range attraction (keeps points bounded)
    This balances local equidistance with global scale.
    """
    np.random.seed(42)
    pts = np.random.randn(N, D) * 0.5
    
    for it in range(iterations):
        # Compute pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        
        # Force matrix
        forces = np.zeros_like(pts)
        
        for i in range(N):
            for j in range(i+1, N):
                d = dist[i, j]
                if d < 1e-10:
                    # Random small displacement for coincident points
                    disp = np.random.randn(D) * 0.01
                    forces[i] += disp
                    forces[j] -= disp
                    continue
                
                # Repulsive force: ~1/d^(repulsion_power-1) in magnitude, direction away
                rep_mag = 1.0 / (d ** (repulsion_power - 1))
                # Attractive force: ~d^(attraction_power-1) in magnitude, direction toward
                attr_mag = d ** (attraction_power - 1)
                
                # Net force on i from j (repulsion pushes i away from j)
                direction = (pts[i] - pts[j]) / d
                net_force = (rep_mag - attr_mag) * direction
                
                forces[i] += net_force
                forces[j] -= net_force
        
        # Update positions with velocity damping (simple Euler with damping)
        pts += forces * dt
        
        # Center the configuration
        pts -= pts.mean(axis=0)
    
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    # Precompute upper triangle indices
    iu = np.triu_indices(N, 1)
    n_pairs = len(iu[0])
    
    def objective(pts_flat):
        """Exact objective: (max dist / min dist)^2"""
        pts = pts_flat.reshape(N, D)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        d_sq = dists_sq[iu]
        d_min_sq = np.min(d_sq)
        d_max_sq = np.max(d_sq)
        
        if d_min_sq < 1e-12:
            return 1e10
        return d_max_sq / d_min_sq
    
    def log_soft_objective(pts_flat, beta):
        """
        Smooth surrogate: log(soft_max) - log(soft_min)
        This is equivalent to log(max/min) which is smoother than max/min.
        """
        pts = pts_flat.reshape(N, D)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        d_sq = dists_sq[iu]
        
        max_d = np.max(d_sq)
        min_d = np.min(d_sq)
        
        if min_d < 1e-12:
            return 1e10
        
        # log(soft_max) = log((1/beta) * (beta*max_d + log(sum(exp(beta*(d_sq - max_d))))))
        #               = -log(beta) + log(beta*max_d + log(sum(exp(beta*(d_sq - max_d)))))
        exp_max_terms = np.exp(beta * (d_sq - max_d))
        log_sum_max = np.log(np.sum(exp_max_terms))
        log_soft_max = -np.log(beta) + np.log(beta * max_d + log_sum_max)
        
        # log(soft_min) = log(-(1/beta) * (-beta*min_d + log(sum(exp(-beta*(d_sq - min_d))))))
        #               = -log(beta) + log(beta*min_d - log(sum(exp(-beta*(d_sq - min_d)))))
        exp_min_terms = np.exp(-beta * (d_sq - min_d))
        log_sum_min = np.log(np.sum(exp_min_terms))
        # soft_min = (1/beta) * (beta*min_d - log_sum_min)
        soft_min_val = (1.0/beta) * (beta * min_d - log_sum_min)
        if soft_min_val < 1e-12:
            return 1e10
        log_soft_min = np.log(soft_min_val)
        
        return log_soft_max - log_soft_min

    def hexagonal_init():
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.5, np.sqrt(3)/2])
        pts = []
        # 4x4 block of hex lattice
        # Centers:
        # (-1, 0), (0, 0), (1, 0), (2, 0) ?
        # Let's just take a compact cluster from the infinite lattice
        for i in range(-2, 3):
            for j in range(-2, 3):
                p = i * v1 + j * v2
                pts.append(p)
        pts = np.array(pts)
        # Sort by distance from origin to get the most central 16
        dists = np.sum(pts**2, axis=1)
        sorted_idx = np.argsort(dists)
        return pts[sorted_idx[:16]]

    def square_init():
        side = 4
        x = np.arange(side, dtype=float)
        y = np.arange(side, dtype=float)
        grid = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
        grid -= grid.mean(axis=0)
        return grid

    def circle_init():
        # Points on a circle, slightly perturbed
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        r = 1.0
        pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        return pts

    def random_init():
        return np.random.rand(N, 2) - 0.5

    def torus_init():
        """
        Initialize points on a discrete torus (4x4 grid with periodic boundaries).
        The minimum image convention gives uniform spacing.
        We optimize aspect ratio and scaling to find best geometry.
        """
        # 4x4 grid on a torus
        n_per_side = 4
        # Try several aspect ratios
        best_score = np.inf
        best_pts = None
        
        for aspect in [0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 2.0]:
            # Grid spacing: width = 1, height = aspect
            # Points at (i/n, j/(n*aspect)) for i,j in range(n)
            xs = np.linspace(0, 1, n_per_side, endpoint=False)
            ys = np.linspace(0, 1, n_per_side, endpoint=False)
            grid = np.array(np.meshgrid(xs, ys)).T.reshape(-1, 2)
            grid[:, 1] *= aspect  # Scale y by aspect ratio
            
            # Apply minimum image convention to compute distances
            # For a torus of size (1, aspect), the distance between points
            # is computed with wrapping
            pts = grid.copy()
            # Center the grid
            pts -= pts.mean(axis=0)
            
            # Compute pairwise distances with minimum image
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            # Wrap: subtract 1 from x if |dx| > 0.5, subtract aspect from y if |dy| > aspect/2
            diff[:, :, 0] = diff[:, :, 0] - np.round(diff[:, :, 0]) * 1.0
            diff[:, :, 1] = diff[:, :, 1] - np.round(diff[:, :, 1] / aspect) * aspect
            d_sq = np.sum(diff**2, axis=2)
            d_sq = d_sq[iu]
            d_min_sq = np.min(d_sq)
            d_max_sq = np.max(d_sq)
            
            if d_min_sq < 1e-12:
                continue
            
            score = d_max_sq / d_min_sq
            if score < best_score:
                best_score = score
                best_pts = pts.copy()
        
        if best_pts is None:
            return random_init()
        
        return best_pts

    def run_optimization(init_pts, beta_schedule):
        """
        Run optimization with annealed beta (softness).
        """
        current = init_pts.flatten().copy()
        best_score_local = objective(current)
        best_pts_local = current.copy()
        
        for beta in beta_schedule:
            if time.time() > deadline - 5:
                break
            
            # Optimize the log-soft objective
            try:
                res = minimize(
                    lambda x: log_soft_objective(x, beta),
                    current,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
                )
                if res.fun < 1e9: # Sanity check
                    current = res.x
                    # Evaluate on exact objective
                    score = objective(current)
                    if score < best_score_local:
                        best_score_local = score
                        best_pts_local = current.copy()
            except Exception:
                pass
                
        return best_score_local, best_pts_local

    best_score_global = np.inf
    best_pts_global = None

    # Define beta schedules: Start soft, get stiff (more gradual)
    # Use more aggressive schedule to converge faster, leaving time for polish
    beta_schedule = [10, 30, 100, 300, 1000, 3000]
    
    initial_guesses = [
        hexagonal_init(),
        square_init(),
        circle_init(),
        torus_init(),
        random_init(),
        random_init(),
        _particle_equilibrium_init(repulsion_power=4, attraction_power=2, iterations=500, dt=0.01),
        _particle_equilibrium_init(repulsion_power=6, attraction_power=2, iterations=500, dt=0.01),
        _particle_equilibrium_init(repulsion_power=4, attraction_power=3, iterations=500, dt=0.01),
    ]

    for init_pts in initial_guesses:
        if time.time() > deadline - 15:
            break
            
        score, pts = run_optimization(init_pts, beta_schedule)
        
        if score < best_score_global:
            best_score_global = score
            best_pts_global = pts

        # Refine with exact objective using multiple methods
        if time.time() > deadline - 5:
            continue
            
        # Powell refinement
        try:
            res = minimize(
                objective,
                best_pts_global,
                method='Powell',
                options={'maxiter': 800, 'disp': False}
            )
            if res.fun < best_score_global:
                best_score_global = res.fun
                best_pts_global = res.x.copy()
        except Exception:
            pass
        
        # Nelder-Mead refinement
        if time.time() > deadline - 5:
            continue
        try:
            res = minimize(
                objective,
                best_pts_global,
                method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-10, 'fatol': 1e-10, 'disp': False}
            )
            if res.fun < best_score_global:
                best_score_global = res.fun
                best_pts_global = res.x.copy()
        except Exception:
            pass
        
        # Final high-precision polish with tighter tolerance
        if time.time() > deadline - 3:
            continue
        try:
            res = minimize(
                objective,
                best_pts_global,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-13, 'fatol': 1e-13, 'disp': False}
            )
            if res.fun < best_score_global:
                best_score_global = res.fun
                best_pts_global = res.x.copy()
        except Exception:
            pass
        
        # Additional L-BFGS polish with very high beta
        if time.time() > deadline - 2:
            continue
        try:
            res = minimize(
                lambda x: log_soft_objective(x, 10000),
                best_pts_global,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'disp': False}
            )
            if res.fun < 1e9:
                score = objective(res.x)
                if score < best_score_global:
                    best_score_global = score
                    best_pts_global = res.x.copy()
        except Exception:
            pass
        
        # SLSQP final polish with tighter tolerances
        if time.time() > deadline - 1:
            continue
        try:
            res = minimize(
                objective,
                best_pts_global,
                method='SLSQP',
                options={'maxiter': 200, 'ftol': 1e-14, 'disp': False}
            )
            if res.fun < best_score_global:
                best_score_global = res.fun
                best_pts_global = res.x.copy()
        except Exception:
            pass

    if best_pts_global is None:
        # Fallback
        best_pts_global = hexagonal_init().flatten()

    return best_pts_global.reshape(N, D)
