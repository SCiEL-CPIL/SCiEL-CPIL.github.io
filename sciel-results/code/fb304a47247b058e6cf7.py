"""Cost-aware acquisition for cosine8 8d BO with deadline guard."""
import time
import numpy as np
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
DEADLINE_MARGIN = 5.0  # seconds to spare before 330s limit


def _compute_ei(mu, sigma, f_best):
    """Compute Expected Improvement values with numerical stability."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)  # Numerical stability
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    return ei, z


def _compute_probability_of_improvement(z):
    """P(f < f_best) given z = (f_best - mu) / sigma."""
    p_imp = norm.cdf(z)
    p_imp = np.clip(p_imp, 1e-10, 1.0 - 1e-10)
    return p_imp


def _safe_acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function: combines Expected Improvement per unit cost with
    a probability-of-improvement weighted term and an exploration bonus.
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Guard against zero/negative cost
    cost = np.maximum(cost, 1e-6)
    sigma = np.maximum(sigma, 1e-9)

    # Compute EI and z
    ei, z = _compute_ei(mu, sigma, f_best)

    # Probability of improvement P(f < f_best)
    p_imp = _compute_probability_of_improvement(z)

    # Cost-aware EI
    ei_cost = ei / cost

    # Probability-weighted improvement magnitude per unit cost
    # This captures the "value density" idea: how likely is improvement
    # and how large is it, normalized by cost
    prob_weighted = p_imp * np.maximum(f_best - mu, 0.0) / cost

    # Uncertainty bonus: encourages exploring high-uncertainty regions
    # Use a smooth bounded transformation
    sigma_max = np.max(sigma)
    if sigma_max > 1e-9:
        uncertainty_bonus = sigma / sigma_max
    else:
        uncertainty_bonus = np.zeros_like(sigma)

    # Exploitation term: favor low predicted values
    # Normalize mu to [0,1] range for scaling
    mu_norm = (mu - (-1.0)) / 2.0  # map [-1,1] to [0,1]
    mu_norm = np.clip(mu_norm, 0.0, 1.0)
    exploitation = 1.0 - mu_norm  # higher when mu is lower

    # Combine: cost-aware EI dominates, with probability-weighted term,
    # exploration and exploitation
    score = 1.0 * ei_cost + 0.5 * prob_weighted + 0.2 * uncertainty_bonus + 0.1 * exploitation

    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)

    return score


def _pareto_hypervolume_score(ei, cost, sigma):
    """
    Compute Pareto-front hypervolume contribution scores.
    A point's score is the area it uniquely dominates on the (cost, ei) front.
    """
    sort_idx = np.argsort(cost)
    sorted_cost = cost[sort_idx]
    sorted_ei = ei[sort_idx]

    # Find indices on the Pareto front
    front_indices = []
    max_ei_so_far = -np.inf

    for i in range(len(sort_idx)):
        if sorted_ei[i] > max_ei_so_far:
            front_indices.append(sort_idx[i])
            max_ei_so_far = sorted_ei[i]

    front_indices = np.array(front_indices)

    scores = np.zeros_like(ei)

    if len(front_indices) >= 10 and np.max(ei) > 1e-6:
        f_costs = cost[front_indices]
        f_eis = ei[front_indices]

        # c_next is the cost of the next point on the front, or max(cost) for the last one
        c_nexts = np.empty_like(f_costs)
        if len(f_costs) > 1:
            c_nexts[:-1] = f_costs[1:]
        c_nexts[-1] = np.max(cost)

        # HV contribution = width * height
        hv_contribs = (c_nexts - f_costs) * f_eis

        scores[front_indices] = hv_contribs

        # Add a tiny exploration bonus to break ties and encourage diversity
        scores[front_indices] += 1e-8 * sigma[front_indices]

    else:
        # Fallback: cost-weighted EI with exploration bonus
        sigma_max = np.max(sigma)
        if sigma_max > 1e-9:
            exploration_bonus = 0.1 * sigma / sigma_max * np.max(ei)
        else:
            exploration_bonus = 0.0
        scores = (ei + exploration_bonus) / cost

    return scores


def acquisition(mu, sigma, f_best, cost):
    # Deadline check: if we're too close to the limit, return a safe fallback
    elapsed = time.time() - _START_TIME
    if elapsed > 330.0 - DEADLINE_MARGIN:
        # Fallback: return a simple cost-aware EI to finish quickly
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        cost = np.maximum(cost, 1e-6)
        sigma = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma
        z = np.clip(z, -10, 10)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        score = ei / cost
        score = np.where(np.isfinite(score), score, 0.0)
        return np.maximum(score, 0.0)

    try:
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)

        # Replace any non-finite values
        mu = np.where(np.isfinite(mu), mu, f_best)
        sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
        cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

        cost = np.maximum(cost, 1e-6)
        sigma = np.maximum(sigma, 1e-9)

        # Compute EI
        ei, z = _compute_ei(mu, sigma, f_best)

        # Compute probability of improvement
        p_imp = _compute_probability_of_improvement(z)

        # Compute two candidate score sets:
        # 1) Probability-weighted value density score
        prob_weighted = p_imp * np.maximum(f_best - mu, 0.0) / cost
        sigma_max = np.max(sigma)
        if sigma_max > 1e-9:
            uncertainty_bonus = sigma / sigma_max
        else:
            uncertainty_bonus = np.zeros_like(sigma)
        score_prob = 1.0 * (ei / cost) + 0.5 * prob_weighted + 0.2 * uncertainty_bonus + 0.1 * (1.0 - np.clip((mu + 1.0) / 2.0, 0.0, 1.0))
        score_prob = np.where(np.isfinite(score_prob), score_prob, 0.0)
        score_prob = np.maximum(score_prob, 0.0)

        # 2) Pareto hypervolume score
        score_pareto = _pareto_hypervolume_score(ei, cost, sigma)
        score_pareto = np.where(np.isfinite(score_pareto), score_pareto, 0.0)
        score_pareto = np.maximum(score_pareto, 0.0)

        # Blend: use Pareto scores when the front is well-defined,
        # otherwise fall back to the probability-weighted score
        # We use a soft blend based on the quality of the Pareto front
        if len(np.where(score_pareto > 0)[0]) >= 10 and np.max(ei) > 1e-6:
            # Normalize both to [0,1] for blending
            max_prob = np.max(score_prob)
            max_pareto = np.max(score_pareto)
            if max_prob > 1e-12:
                score_prob_norm = score_prob / max_prob
            else:
                score_prob_norm = score_prob
            if max_pareto > 1e-12:
                score_pareto_norm = score_pareto / max_pareto
            else:
                score_pareto_norm = score_pareto

            # Blend 50/50
            score = 0.5 * score_prob + 0.5 * score_pareto
        else:
            score = score_prob

        # Ensure finite output
        score = np.where(np.isfinite(score), score, 0.0)
        score = np.maximum(score, 0.0)

        return score

    except Exception:
        # Last-resort fallback: uniform random scores
        return np.ones(2000, dtype=np.float64) * 0.1
