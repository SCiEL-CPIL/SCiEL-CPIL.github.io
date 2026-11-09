# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global state to persist across calls within the same seed run
_acq_state = {
    'call_count': 0,
    'last_f_best': None,
    'f_best_history': []
}

def acquisition(mu, sigma, f_best):
    t_start = time.time()
    # The evaluator kills after 330s. The function call itself has a 300s limit.
    # We use a tight deadline to ensure we return well before any timeout.
    deadline = t_start + 290.0
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    n = len(mu)
    
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Ensure sigma is safe
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Update state
    _acq_state['call_count'] += 1
    calls = _acq_state['call_count']
    
    # Knowledge Gradient (KG) Acquisition Function
    # KG_i = E[ min_j (f_j | x_i evaluated) ] - min_j (f_j)
    # We want to maximize this reduction in expected minimum.
    # Since we are minimizing, a larger positive KG means a better candidate.
    # Note: The "min_j (f_j)" term is the same for all i, so we can just compute
    # the first term and it will preserve the ranking.
    
    # Monte Carlo approximation:
    # 1. Draw a sample from the joint posterior of all candidates.
    #    Since we don't have the covariance matrix, we approximate the joint distribution
    #    by assuming conditional independence given the current information? 
    #    No, that's just sampling each independently. 
    #    However, without coordinates, we cannot compute the true GP covariance 
    #    between the 2000 candidates. 
    #    
    #    BUT, we can use a property of the GP: 
    #    If we assume the candidates are effectively "independent" in the posterior 
    #    (which is a strong approximation, often used when the candidate set is large 
    #    and spread out, or as a heuristic "Independent Knowledge Gradient"), 
    #    we can estimate the expected minimum.
    #    
    #    A better heuristic for "Independent KG":
    #    For each candidate i, we sample its value y_i ~ N(mu_i, sigma_i^2).
    #    The expected minimum over the set S, given that we evaluate i, is:
    #    E[ min( y_i, min_{j!=i} y_j ) ]
    #    
    #    If we assume independence of y_i and y_j for j!=i:
    #    Let M_{-i} = min_{j!=i} y_j.
    #    Then E[ min(y_i, M_{-i}) ] = E[ y_i * I(y_i < M_{-i}) ] + E[ M_{-i} * I(M_{-i} < y_i) ]
    #    
    #    This is complex to compute exactly without the joint distribution.
    #    
    #    Alternative Structural Approach: **Probability of Improvement (PI) vs Expected Improvement (EI)**
    #    Current program is TS. EI is deterministic and often outperforms TS in low dimensions.
    #    However, the directive asks for a *different* strategy than the current one (TS).
    #    EI is a different family (Deterministic vs Stochastic).
    #    But let's try to stick to the KG idea if possible, or a hybrid.
    #    
    #    Actually, a very effective and structurally different approach from TS is 
    #    **Expected Improvement (EI)**. TS samples the function; EI calculates the 
    #    expected gain in the *minimum* value. 
    #    
    #    Let's implement a robust **Expected Improvement (EI)** with a small 
    #    exploration bonus, as it is the gold standard for BO and structurally 
    #    distinct from the stochastic TS in the parent.
    #    
    #    EI_i = E[ max(0, f_best - f(x_i)) ]
    #    = (f_best - mu_i) * Phi(z) + sigma_i * phi(z)
    #    where z = (f_best - mu_i) / sigma_i
    #    
    #    To avoid premature convergence (a known weakness of pure EI), we can 
    #    use a "lower confidence bound" (LCB) term or a "probability of improvement" 
    #    mix. But pure EI is a clean, distinct structural change from TS.
    
    # Calculate z-scores
    z = (f_best - mu) / sigma_safe
    
    # Standard Normal CDF and PDF
    # Using scipy.stats.norm for accuracy
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # EI Score
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Clip small negative values due to numerical precision
    ei = np.maximum(ei, 0.0)
    
    # To make it more robust against local optima (which TS also suffers from, 
    # but in a different way), we can add a small amount of exploration.
    # However, the prompt asks for a structural change. EI is the standard 
    # deterministic counterpart.
    
    # Let's enhance EI with a "U-Upper Confidence Bound" (UCB) style exploration 
    # term to ensure we don't get stuck. 
    # UCB = mu + kappa * sigma
    # For minimization, we want to minimize UCB, so we maximize -UCB? 
    # No, UCB for minimization is LCB = mu - kappa * sigma.
    # We want to pick the point with the lowest LCB.
    # So score = -LCB = -mu + kappa * sigma.
    
    # A common hybrid is EI + Exploration Bonus.
    # Or simply: Score = EI + lambda * sigma
    # This ensures points with high uncertainty get a boost.
    
    # Let's try a mix: 0.8 * EI + 0.2 * (sigma_safe / max(sigma_safe))
    # This normalizes the exploration term.
    
    max_sigma = np.max(sigma_safe)
    if max_sigma > 0:
        exploration_term = sigma_safe / max_sigma
    else:
        exploration_term = np.zeros_like(sigma_safe)
    
    # Weights
    w_ei = 0.8
    w_exp = 0.2
    
    scores = w_ei * ei + w_exp * exploration_term
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    
    # If any scores are -inf, they won't be picked.
    # If any are inf, they will be picked.
    
    # Time check
    if time.time() > deadline:
        pass # Return what we have
        
    return scores

# EVOLVE-BLOCK-END
