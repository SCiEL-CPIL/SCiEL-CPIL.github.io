import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Model-and-Optimize Framework:
    # 1. Pre-compute Expected Improvement (EI) for all candidates
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ei = (f_best - mu) * cdf + sigma * pdf
    
    cost_safe = np.maximum(cost, 1e-10)
    
    # 2. Optimize exploration parameter beta
    # We test a set of beta values for UCB and select the one that 
    # maximizes the EI of the best candidate selected by that UCB.
    beta_candidates = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
    
    best_beta = 2.0
    best_improvement = -np.inf
    
    for beta in beta_candidates:
        # Compute UCB scores (higher is better for selection)
        # Minimization: Score ~ beta*sigma - mu
        ucb_scores = (beta * sigma - mu) / cost_safe
        
        # Find the index of the best candidate for this beta
        best_idx = np.argmax(ucb_scores)
        
        # Evaluate the EI of this specific candidate
        current_improvement = ei[best_idx]
        
        # Update best beta if this selection yields higher predicted improvement
        if current_improvement > best_improvement + 1e-6:
            best_improvement = current_improvement
            best_beta = beta
            
    # 3. Return scores using the optimized beta
    final_scores = (best_beta * sigma - mu) / cost_safe
    
    # Ensure finite scores
    final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return final_scores
