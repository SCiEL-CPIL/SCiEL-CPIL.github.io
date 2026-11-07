"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard state
_START_TIME = None

def acquisition(mu, sigma, f_best, cost):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Check deadline: 330s limit, leave 10s margin
    if time.time() - _START_TIME > 320:
        # Return a neutral score if we are running out of time
        return np.zeros(len(mu))
    
    # Ensure inputs are valid and 1D
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    # Clip cost to avoid log(0) or division by zero
    # Cost is in [1, 2] typically, but ensure positive
    cost = np.maximum(cost, 1e-8)
    
    # Strategy: Cost-Effective Expected Hypervolume Improvement (CE-EHI)
    # 
    # The standard EI/UCB approaches treat the objective as a single dimension.
    # Here we treat the problem as a bi-objective optimization: minimize objective, minimize cost.
    # However, since we must return a scalar score to pick ONE point, we approximate the 
    # expected improvement in the hypervolume of the (cost, objective) space.
    #
    # The hypervolume improvement (HI) of a new point (c, y) over a reference point (c_ref, y_ref)
    # is the area of the rectangle defined by (c, y) and (c_ref, y_ref) if the point dominates the reference.
    # Expected HI (EHI) integrates this over the GP posterior.
    #
    # For a single point selection, a common approximation for EHI in 2D (cost, obj) is:
    # EHI(x) ≈ E[ (max(0, y_ref - Y(x)) * max(0, c_ref - c(x)) ) ]
    # where Y(x) ~ N(mu(x), sigma(x)^2) and c(x) is deterministic.
    #
    # Since cost is deterministic, c_ref - c(x) is a constant factor for each candidate.
    # We need E[ max(0, y_ref - Y) ].
    # Let Z = (y_ref - mu) / sigma.
    # E[ max(0, y_ref - Y) ] = (y_ref - mu) * Phi(Z) + sigma * phi(Z)
    # This is exactly the Expected Improvement (EI) formula with y_ref as the threshold.
    #
    # So, EHI(x) = EI_obj(x) * (c_ref - c(x)) if c(x) < c_ref, else 0.
    #
    # To make it "Cost-Effective", we divide by the cost of the evaluation or use a normalized form.
    # The directive suggests: "dividing the expected hypervolume gain by the evaluation cost".
    #
    # Reference Point Selection:
    # We don't have the full Pareto front, but we have f_best (best objective seen).
    # We can set the reference objective y_ref = f_best.
    # For the cost reference c_ref, we can use the maximum possible cost (2.0) or the mean cost of the current population.
    # Using max cost (2.0) ensures that any point with cost < 2.0 has a positive cost term.
    # However, to be more adaptive, we can use a reference cost that represents "average" or "current best" cost.
    # Let's use c_ref = 2.0 (max possible) to be conservative about the hypervolume box size, 
    # or c_ref = np.mean(cost) to be adaptive.
    # Actually, the hypervolume is relative to a reference point that is worse than all points.
    # Let's use y_ref = f_best and c_ref = 2.0 (since cost is in [1,2]).
    # Wait, if y_ref = f_best, then EI is the expected improvement over f_best.
    # This is standard EI.
    # The hypervolume interpretation adds the cost dimension.
    # HI = (y_ref - y) * (c_ref - c).
    # EHI = E[(y_ref - Y) * (c_ref - c)]_+ = E[(y_ref - Y)]_+ * (c_ref - c)
    #
    # So the score is:
    # score = EI(f_best, mu, sigma) * (c_ref - cost) / cost
    #
    # Why divide by cost? The directive says "dividing ... by the evaluation cost".
    # This normalizes the gain by the resource spent.
    
    # Calculate Z-score for EI
    z = (f_best - mu) / sigma
    
    # Calculate EI (Expected Improvement)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Define reference cost. 
    # Using a fixed max cost (2.0) makes the (c_ref - cost) term larger for cheaper points.
    # Using mean(cost) adapts to the current candidate set.
    # Let's use a fixed reference cost of 2.0 (max possible) to maximize the relative weight of cheap points.
    c_ref = 2.0
    
    # Hypervolume term: (c_ref - cost)
    # If cost > c_ref, this is negative. We should clip it to 0?
    # In hypervolume, if the point is worse than the reference in one dimension, it doesn't contribute.
    # So we clip (c_ref - cost) to be non-negative.
    hv_cost_term = np.maximum(c_ref - cost, 0.0)
    
    # Expected Hypervolume Improvement
    ehi = ei * hv_cost_term
    
    # Cost-Effective Score: Divide by cost
    # This penalizes expensive points more heavily in the numerator AND denominator?
    # No, the directive says "dividing the expected hypervolume gain by the evaluation cost".
    # So: score = ehi / cost
    
    # To avoid division by zero, cost is already clipped.
    score = ehi / cost
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
