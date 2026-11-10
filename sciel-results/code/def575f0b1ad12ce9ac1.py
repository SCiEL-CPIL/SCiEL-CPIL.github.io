# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution
import time


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 630s kill

    def score(points):
        """Calculate R = (d_min/d_max)^2"""
        dists = pdist(points, 'euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-10:
            return 0.0
        ratio = d_min / d_max
        return ratio ** 2

    def make_hex_init():
        """Hexagonal lattice arrangement"""
        points = []
        # Center
        points.append([0.0, 0.0])
        # First ring (6 points)
        for i in range(6):
            angle = i * np.pi / 3
            points.append([1.0 * np.cos(angle), 1.0 * np.sin(angle)])
        # Second ring (9 points) - hexagonal pattern
        for i in range(9):
            if i < 6:
                angle = i * np.pi / 3 + np.pi / 6
                r = 2.0
            else:
                angle = (i - 6) * 2 * np.pi / 3
                r = 2.0
            points.append([r * np.cos(angle), r * np.sin(angle)])
        return np.array(points[:n])

    def make_interlocking_init():
        """Two interlocking square patterns"""
        points = []
        # Square 1: corners at (±1, ±1)
        for sx in [-1, 1]:
            for sy in [-1, 1]:
                points.append([sx, sy])
        # Square 2: rotated 45°, scaled
        s = 1.4
        for sx in [-1, 1]:
            for sy in [-1, 1]:
                x = s * (sx + sy) / np.sqrt(2)
                y = s * (sx - sy) / np.sqrt(2)
                points.append([x, y])
        # Fill remaining with inner points
        for i in range(n - 8):
            angle = i * 2 * np.pi / (n - 8)
            points.append([0.5 * np.cos(angle), 0.5 * np.sin(angle)])
        return np.array(points[:n])

    def make_circular_init():
        """Two concentric circles with offset"""
        points = []
        # Outer circle: 8 points
        for i in range(8):
            angle = i * 2 * np.pi / 8
            points.append([1.0 * np.cos(angle), 1.0 * np.sin(angle)])
        # Inner circle: 8 points, offset by half-step
        for i in range(8):
            angle = i * 2 * np.pi / 8 + np.pi / 8
            points.append([0.5 * np.cos(angle), 0.5 * np.sin(angle)])
        return np.array(points[:n])

    def make_random_init(scale=1.0):
        return np.random.randn(n, 2) * scale

    np.random.seed(42)
    inits = [
        make_hex_init(),
        make_interlocking_init(),
        make_circular_init(),
        make_random_init(1.0),
        make_random_init(0.7),
        make_random_init(1.3),
    ]

    best_points = inits[0].copy()
    best_score = score(inits[0])

    # Phase 1: Differential evolution for global exploration
    if time.time() < deadline - 200:
        def de_objective(params):
            points = params.reshape((n, 2))
            return -score(points)
        
        bounds = [(-2, 2)] * (n * 2)
        try:
            de_result = differential_evolution(
                de_objective, bounds, 
                maxiter=100, popsize=15, 
                tol=1e-8, seed=42,
                polish=False,
                init='sobol'
            )
            candidate = de_result.x.reshape((n, 2))
            trial_score = score(candidate)
            if trial_score > best_score:
                best_score = trial_score
                best_points = candidate.copy()
        except Exception:
            pass

    # Phase 2: Simulated annealing with adaptive step size
    if time.time() < deadline - 120:
        def sa_optimize(init_points, n_steps=5000):
            current = init_points.copy()
            current_score = score(current)
            best_sa = current.copy()
            best_sa_score = current_score
            
            # Adaptive step size
            base_step = 0.1
            step_scale = 1.0
            T = 0.1
            cooling = 0.9997
            
            for step in range(n_steps):
                if time.time() > deadline - 60:
                    break
                    
                # Perturb
                delta = np.random.randn(n, 2) * base_step * step_scale
                proposed = current + delta
                
                proposed_score = score(proposed)
                
                if proposed_score > current_score:
                    current = proposed
                    current_score = proposed_score
                    if current_score > best_sa_score:
                        best_sa = current.copy()
                        best_sa_score = current_score
                    # Reduce step on improvement
                    step_scale *= 0.95
                else:
                    # Accept with probability exp((new-old)/T)
                    if T > 1e-10:
                        prob = np.exp((proposed_score - current_score) / T)
                        if np.random.random() < prob:
                            current = proposed
                            current_score = proposed_score
                            step_scale *= 1.05
                    # Increase step on rejection
                    step_scale = min(step_scale * 1.02, 2.0)
                
                T *= cooling
            
            return best_sa, best_sa_score
        
        # Run SA from multiple starting points
        sa_starts = [best_points]
        for init in inits:
            if time.time() > deadline - 80:
                break
            sa_starts.append(init.copy())
        
        for sa_init in sa_starts:
            if time.time() > deadline - 60:
                break
            sa_result, sa_score = sa_optimize(sa_init, n_steps=3000)
            if sa_score > best_score:
                best_score = sa_score
                best_points = sa_result.copy()

    # Phase 3: Nelder-Mead local refinement
    if time.time() < deadline - 40:
        def nm_objective(params):
            points = params.reshape((n, 2))
            return -score(points)
        
        # Multiple Nelder-Mead runs from perturbed best points
        nm_runs = 5
        for r in range(nm_runs):
            if time.time() > deadline - 20:
                break
            
            if r == 0:
                init = best_points.copy()
            else:
                init = best_points + np.random.randn(n, 2) * 0.03
            
            try:
                nm_result = minimize(
                    nm_objective, init.flatten(), 
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 
                             'adaptive': True}
                )
                candidate = nm_result.x.reshape((n, 2))
                trial_score = score(candidate)
                if trial_score > best_score:
                    best_score = trial_score
                    best_points = candidate.copy()
            except Exception:
                pass

    # Phase 4: Final tight refinement with L-BFGS-B
    if time.time() < deadline - 10:
        def lbfgs_objective(params):
            points = params.reshape((n, 2))
            return -score(points)
        
        try:
            lbfgs_result = minimize(
                lbfgs_objective, best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            candidate = lbfgs_result.x.reshape((n, 2))
            trial_score = score(candidate)
            if trial_score > best_score:
                best_points = candidate.copy()
        except Exception:
            pass

    return best_points

# EVOLVE-BLOCK-END
