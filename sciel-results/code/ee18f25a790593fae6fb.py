# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    deadline = time.time() + 570  # safety margin before 630s kill

    def score(points):
        """Compute R = (d_min / d_max)^2"""
        if points.shape != (n, d):
            return 0.0
        d = pdist(points)
        if len(d) == 0:
            return 0.0
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def objective(x):
        """Negative of the smooth surrogate for optimization.
        We maximize min(d) / max(d), which is equivalent to maximizing log(min(d)) - log(max(d)).
        We use soft-min and soft-max via log-sum-exp for smoothness.
        """
        points = x.reshape(n, d)
        # Fix scale: set first point to origin, and constrain second point to have distance 1 from first
        # Actually, let's just optimize the ratio directly with a smooth surrogate
        
        # Compute all pairwise squared distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        sq_dists = np.sum(diff ** 2, axis=2)
        # Set diagonal to inf to ignore self-distances
        np.fill_diagonal(sq_dists, np.inf)
        # Take upper triangle
        iu = np.triu_indices(n, k=1)
        sq_d = sq_dists[iu]
        
        # Soft-min of sq_dists (we want to maximize the minimum distance)
        # soft_min = -logsumexp(-log(sq_d)) but sq_d can be 0, so use a large constant
        # Actually, let's work with distances not squared distances for the ratio
        dists = np.sqrt(np.maximum(sq_d, 1e-12))
        
        # Soft-min: -log(sum(exp(-dists/tau))) * tau
        # Soft-max: log(sum(exp(dists/tau))) * tau
        tau = 0.01  # temperature for smoothness
        
        # Soft min of distances
        neg_dists = -dists / tau
        neg_dists -= np.max(neg_dists)  # for numerical stability
        soft_min = -tau * np.log(np.sum(np.exp(neg_dists)))
        
        # Soft max of distances
        pos_dists = dists / tau
        pos_dists -= np.max(pos_dists)  # for numerical stability
        soft_max = tau * np.log(np.sum(np.exp(pos_dists)))
        
        if soft_max < 1e-12:
            return 0.0
        
        # We want to maximize soft_min / soft_max, so minimize -log(soft_min/soft_max) = -log(soft_min) + log(soft_max)
        return -np.log(soft_min) + np.log(soft_max)

    def objective_grad(x):
        """Gradient of the objective function."""
        points = x.reshape(n, d)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        sq_dists = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        iu = np.triu_indices(n, k=1)
        sq_d = sq_dists[iu]
        dists = np.sqrt(np.maximum(sq_d, 1e-12))
        
        tau = 0.01
        
        # Soft min gradient
        neg_dists = -dists / tau
        neg_dists -= np.max(neg_dists)
        exp_neg = np.exp(neg_dists)
        sum_exp_neg = np.sum(exp_neg)
        # d(soft_min)/d(dists) = exp_neg / sum_exp_neg (positive, since soft_min = -tau*log(sum_exp_neg))
        # Actually: soft_min = -tau * log(sum(exp(-dists/tau)))
        # d(soft_min)/d(dists_i) = -tau * (1/sum_exp_neg) * (-exp_neg_i / tau) = exp_neg_i / sum_exp_neg
        
        # Soft max gradient
        pos_dists = dists / tau
        pos_dists -= np.max(pos_dists)
        exp_pos = np.exp(pos_dists)
        sum_exp_pos = np.sum(exp_pos)
        # d(soft_max)/d(dists_i) = tau * (1/sum_exp_pos) * (exp_pos_i / tau) = exp_pos_i / sum_exp_pos
        
        # Objective = -log(soft_min) + log(soft_max)
        # d(obj)/d(dists_i) = -1/soft_min * d(soft_min)/d(dists_i) + 1/soft_max * d(soft_max)/d(dists_i)
        
        d_soft_min_dd = exp_neg / sum_exp_neg
        d_soft_max_dd = exp_pos / sum_exp_pos
        
        d_obj_dd = -d_soft_min_dd / soft_min + d_soft_max_dd / soft_max
        
        # Now backpropagate to points
        # dists_i = sqrt(sum(diff_ij^2)) where diff_ij = points_i - points_j
        # d(dists_i)/d(points_k) involves the diff terms
        
        grad_points = np.zeros((n, d))
        for idx in range(len(iu)):
            i, j = iu[0][idx], iu[1][idx]
            diff_ij = points[i] - points[j]
            # d(dists_i)/d(points_i) = diff_ij / dists_i
            # d(dists_i)/d(points_j) = -diff_ij / dists_i
            coeff = d_obj_dd[idx] / max(dists[idx], 1e-12)
            grad_points[i] += coeff * diff_ij
            grad_points[j] -= coeff * diff_ij
        
        return grad_points.flatten()

    # Initial configuration: points on a circle
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()

    best_x = x0.copy()
    best_score = score(x0.reshape(n, d))

    # Try multiple starting points and optimize
    rng = np.random.RandomState(42)
    n_starts = 20
    for start_i in range(n_starts):
        if time.time() > deadline:
            break
        
        if start_i == 0:
            x = x0.copy()
        else:
            # Perturbed circle
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.randn(n) * 0.1
            radii = 1.0 + rng.randn(n) * 0.1
            x = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]).flatten()
        
        # Optimize
        try:
            result = minimize(objective, x, method='L-BFGS-B', jac=objective_grad,
                              options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            if result.success or result.fun < objective(x):
                x_opt = result.x
                s = score(x_opt.reshape(n, d))
                if s > best_score:
                    best_score = s
                    best_x = x_opt.copy()
        except Exception:
            pass

    # Polish with tighter parameters
    if time.time() < deadline - 60:
        try:
            # Use a smaller tau for more precise optimization
            def objective_tight(x):
                points = x.reshape(n, d)
                diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
                sq_dists = np.sum(diff ** 2, axis=2)
                np.fill_diagonal(sq_dists, np.inf)
                iu = np.triu_indices(n, k=1)
                sq_d = sq_dists[iu]
                dists = np.sqrt(np.maximum(sq_d, 1e-12))
                
                tau = 0.001
                neg_dists = -dists / tau
                neg_dists -= np.max(neg_dists)
                exp_neg = np.exp(neg_dists)
                sum_exp_neg = np.sum(exp_neg)
                soft_min = -tau * np.log(sum_exp_neg)
                
                pos_dists = dists / tau
                pos_dists -= np.max(pos_dists)
                exp_pos = np.exp(pos_dists)
                sum_exp_pos = np.sum(exp_pos)
                soft_max = tau * np.log(sum_exp_pos)
                
                if soft_max < 1e-12 or soft_min < 1e-12:
                    return 0.0
                return -np.log(soft_min) + np.log(soft_max)

            def objective_grad_tight(x):
                points = x.reshape(n, d)
                diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
                sq_dists = np.sum(diff ** 2, axis=2)
                np.fill_diagonal(sq_dists, np.inf)
                iu = np.triu_indices(n, k=1)
                sq_d = sq_dists[iu]
                dists = np.sqrt(np.maximum(sq_d, 1e-12))
                
                tau = 0.001
                neg_dists = -dists / tau
                neg_dists -= np.max(neg_dists)
                exp_neg = np.exp(neg_dists)
                sum_exp_neg = np.sum(exp_neg)
                soft_min = -tau * np.log(sum_exp_neg)
                
                pos_dists = dists / tau
                pos_dists -= np.max(pos_dists)
                exp_pos = np.exp(pos_dists)
                sum_exp_pos = np.sum(exp_pos)
                soft_max = tau * np.log(sum_exp_pos)
                
                d_soft_min_dd = exp_neg / sum_exp_neg
                d_soft_max_dd = exp_pos / sum_exp_pos
                
                d_obj_dd = -d_soft_min_dd / soft_min + d_soft_max_dd / soft_max
                
                grad_points = np.zeros((n, d))
                for idx in range(len(iu)):
                    i, j = iu[0][idx], iu[1][idx]
                    diff_ij = points[i] - points[j]
                    coeff = d_obj_dd[idx] / max(dists[idx], 1e-12)
                    grad_points[i] += coeff * diff_ij
                    grad_points[j] -= coeff * diff_ij
                return grad_points.flatten()

            result = minimize(objective_tight, best_x, method='L-BFGS-B', jac=objective_grad_tight,
                              options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
            if result.success or result.fun < objective_tight(best_x):
                x_opt = result.x
                s = score(x_opt.reshape(n, d))
                if s > best_score:
                    best_score = s
                    best_x = x_opt.copy()
        except Exception:
            pass

    return best_x.reshape(n, d)


# EVOLVE-BLOCK-END
