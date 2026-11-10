# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
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

    def soft_score(params, beta=50.0):
        """
        Smooth surrogate using soft-min and soft-max.
        Returns -soft_R where soft_R = (soft_min/soft_max)^2
        """
        points = params.reshape((n, 2))
        dists = pdist(points, 'euclidean')
        # Soft min: -logsumexp(-dists/beta) * beta
        # Soft max: logsumexp(dists/beta) * beta
        scaled = dists / beta
        # Numerically stable logsumexp
        max_s = np.max(scaled)
        logsumexp_max = max_s + np.log(np.sum(np.exp(scaled - max_s)))
        soft_max = logsumexp_max * beta
        
        min_s = -scaled
        max_min_s = np.max(min_s)
        logsumexp_min = max_min_s + np.log(np.sum(np.exp(min_s - max_min_s)))
        soft_min = -logsumexp_min * beta
        
        if soft_max < 1e-10:
            return 0.0
        ratio = soft_min / soft_max
        return -(ratio ** 2)

    def objective(params):
        """Minimize -R using exact score"""
        points = params.reshape((n, 2))
        return -score(points)

    def soft_objective(params):
        """Minimize -soft_R using smooth surrogate"""
        return soft_score(params, beta=50.0)

    def soft_objective_tight(params):
        """Minimize -soft_R using tighter smooth surrogate"""
        return soft_score(params, beta=200.0)

    # Generate multiple diverse initial configurations
    def make_grid_init():
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        X, Y = np.meshgrid(x, y)
        return np.column_stack([X.flatten(), Y.flatten()])

    def make_circular_init():
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        radii = np.ones(n)
        # Two concentric rings
        radii[:8] = 1.0
        radii[8:] = 0.5
        return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    def make_random_init(scale=1.0):
        return np.random.randn(n, 2) * scale

    # Initial configurations to try
    np.random.seed(42)
    inits = [
        make_grid_init(),
        make_circular_init(),
        make_random_init(1.0),
        make_random_init(0.5),
        make_random_init(2.0),
    ]

    best_points = inits[0].copy()
    best_score = score(inits[0])

    # Phase 1: Use soft surrogate for broader exploration
    for trial, init in enumerate(inits):
        if time.time() > deadline - 120:
            break

        # First optimize with soft objective (broader landscape)
        result = minimize(soft_objective, init.flatten(), method='L-BFGS-B',
                         options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8})
        if result.success:
            candidate = result.x.reshape((n, 2))
            # Then refine with tighter soft objective
            result2 = minimize(soft_objective_tight, candidate.flatten(), method='L-BFGS-B',
                             options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8})
            if result2.success:
                candidate = result2.x.reshape((n, 2))
            
            trial_score = score(candidate)
            if trial_score > best_score:
                best_score = trial_score
                best_points = candidate.copy()

    # Phase 2: Local refinement with exact objective from best point
    if time.time() < deadline - 90:
        # Multiple local refinements from perturbed best points
        n_refine = 10
        for r in range(n_refine):
            if time.time() > deadline - 60:
                break
            
            if r == 0:
                init = best_points.copy()
            else:
                init = best_points + np.random.randn(n, 2) * 0.05

            result = minimize(objective, init.flatten(), method='L-BFGS-B',
                             options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-10})
            if result.success:
                candidate = result.x.reshape((n, 2))
                trial_score = score(candidate)
                if trial_score > best_score:
                    best_score = trial_score
                    best_points = candidate.copy()

    # Phase 3: Final tight refinement
    if time.time() < deadline - 30:
        result = minimize(objective, best_points.flatten(), method='L-BFGS-B',
                         options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12})
        if result.success:
            candidate = result.x.reshape((n, 2))
            trial_score = score(candidate)
            if trial_score > best_score:
                best_points = candidate.copy()

    return best_points

# EVOLVE-BLOCK-END
