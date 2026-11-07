"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Strategy: Dynamic acquisition using a hybrid of Expected Improvement (EI)
    # and Lower Confidence Bound (LCB), adapting the balance over time.
    #
    # Rationale:
    # 1. Early on (few observations), the GP is very uncertain. We should explore
    #    to gather information. LCB with a high beta is good for this.
    # 2. Later on (many observations), we should exploit the best found region
    #    to refine the minimum. EI is good for this as it focuses on points
    #    likely to improve upon f_best.
    #
    # We infer the "step" by tracking the number of calls to acquisition.
    # The loop structure is:
    #   - 5 initial points (acquisition not called)
    #   - Then, at each step: draw 2000 candidates, compute posterior, call
    #     acquisition, evaluate 1 point.
    #   - Total evaluations = 30.
    #   - So acquisition is called 25 times (for evaluations 6 through 30).
    #
    # We use a global counter to track the step.
    # Step 0 corresponds to the 6th evaluation (5 points in GP).
    # Step 24 corresponds to the 30th evaluation (29 points in GP).
    
    if not hasattr(acquisition, '_call_count'):
        acquisition._call_count = 0
    
    step = acquisition._call_count
    acquisition._call_count += 1
    
    # Total steps = 25
    total_steps = 25
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # where Phi is the CDF and phi is the PDF of the standard normal.
    # We want to MAXIMIZE EI, so higher EI is better.
    # The evaluator picks the candidate with the HIGHEST score.
    
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues in norm.cdf/pdf
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0)  # EI is non-negative
    
    # Calculate LCB score for minimization
    # LCB = mu - beta * sigma
    # We want to MINIMIZE LCB, so we return -LCB = -mu + beta * sigma
    # Higher -LCB is better (means lower LCB, i.e., more optimistic lower bound)
    
    # Dynamic beta: start high (exploration), decrease to low (exploitation)
    # Linear schedule from 4.0 to 1.0
    beta = 4.0 - 3.0 * (step / total_steps)
    
    lcb_score = -mu + beta * sigma
    
    # Blend EI and LCB
    # Early: weight towards LCB (exploration)
    # Late: weight towards EI (exploitation)
    # Use a sigmoid or linear blend
    # Let's use a linear blend:
    # weight_ei goes from 0.2 to 0.8
    # weight_lcb goes from 0.8 to 0.2
    
    # Actually, let's think about this differently.
    # EI is already a good acquisition function that balances exploration and exploitation.
    # LCB is also good.
    # Maybe we should just use EI with a dynamic "temperature" or just EI?
    # Or perhaps use EI but add a small exploration term?
    
    # Let's try a simpler approach: use EI directly, but maybe adjust it.
    # Or use a weighted combination.
    
    # Let's normalize both to have zero mean and unit variance for blending
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + 1e-9
    ei_norm = (ei - ei_mean) / ei_std
    
    lcb_mean = np.mean(lcb_score)
    lcb_std = np.std(lcb_score) + 1e-9
    lcb_norm = (lcb_score - lcb_mean) / lcb_std
    
    # Blend weight: start with more LCB, end with more EI
    # At step 0: weight_ei = 0.3, weight_lcb = 0.7
    # At step 24: weight_ei = 0.7, weight_lcb = 0.3
    weight_ei = 0.3 + 0.4 * (step / total_steps)
    weight_lcb = 1.0 - weight_ei
    
    scores = weight_ei * ei_norm + weight_lcb * lcb_norm
    
    # Ensure finiteness
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
