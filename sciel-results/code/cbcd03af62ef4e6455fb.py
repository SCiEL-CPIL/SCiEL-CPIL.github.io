"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    deadline = start_time + 280.0
    
    # Ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # 1. Compute base Expected Improvement (EI) scores
    # This serves as a strong exploitation baseline.
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    # Handle zero variance safely
    sigma_safe = np.where(sigma > 1e-8, sigma, 1e-8)
    z = (f_best - mu) / sigma_safe
    
    # Use scipy.stats.norm for cdf and pdf
    from scipy.stats import norm
    ei_scores = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure finite scores for EI
    ei_scores = np.where(np.isfinite(ei_scores), ei_scores, 0.0)
    
    # 2. Compute the new Quantile-based Exploration Score
    # Strategy: 
    # - Identify candidates where mu < f_best (potential to improve)
    # - Among these, rank by sigma (uncertainty)
    # - Select the top 5% of these candidates by uncertainty
    # - Assign them a high score, potentially boosted by their EI or just a fixed high value?
    # The prompt suggests: "acquisition score is the negative predicted mean plus a penalty derived from the empirical rank of the uncertainty among candidates"
    # Wait, the prompt says: "effectively selecting points that are in the top 5% of uncertainty but only if their mean is below the current best."
    # And: "score is the negative predicted mean plus a penalty derived from the empirical rank of the uncertainty"
    
    # Let's interpret "penalty derived from empirical rank". 
    # Usually, we want high scores for good candidates.
    # If we want to select top 5% uncertainty, we can create a mask.
    # But the score must be a continuous value for the 2000 candidates.
    
    # Let's construct the score as:
    # score = -mu + lambda * rank_score
    # Where rank_score is high for high uncertainty.
    # But we need to enforce the condition "mu < f_best".
    # If mu >= f_best, the score should be low (or 0, or -inf).
    
    # Let's compute the rank of sigma among all candidates.
    # Rank 0 is lowest sigma, Rank N-1 is highest sigma.
    # We want high rank to have high score contribution.
    
    # Step 2a: Compute ranks of sigma
    sigma_ranks = np.argsort(np.argsort(sigma))
    # Normalize ranks to [0, 1]
    n_candidates = len(sigma)
    sigma_rank_norm = sigma_ranks / (n_candidates - 1) if n_candidates > 1 else np.zeros_like(sigma)
    
    # Step 2b: Create the exploration score component
    # We want to favor high sigma_rank_norm.
    # Let's use a threshold: only consider candidates in the top 5% of sigma_rank_norm.
    # Top 5% means sigma_rank_norm >= 0.95.
    
    # However, the prompt says "penalty derived from the empirical rank".
    # Let's try a score that is:
    # base_score = -mu
    # exploration_bonus = 0 if mu >= f_best else (sigma_rank_norm ** 2) * max_sigma_scale
    
    # Actually, let's stick closer to the "quantile-based" description.
    # "selecting points that are in the top 5% of uncertainty but only if their mean is below the current best"
    
    # Let's create a score that is:
    # 1. If mu >= f_best: score = -mu (pure exploitation of low mean, even if not improving? No, if mu >= f_best, it's unlikely to improve. EI would be low. Let's set score to 0 or very low.)
    # 2. If mu < f_best: score = -mu + bonus * indicator(top 5% sigma)
    
    # Let's refine:
    # The goal is to return 2000 scores. Higher is better.
    # We can combine EI and this new quantile strategy.
    # But the directive says "Make a STRUCTURAL change... different design family".
    # So I should replace EI with this new logic, or heavily weight it.
    
    # Let's define the new score:
    # Condition: mu < f_best
    # Among those, rank sigma.
    # Top 5% get a high bonus.
    
    # Calculate the 95th percentile of sigma
    sigma_95 = np.percentile(sigma, 95)
    
    # Mask for candidates with mu < f_best
    mask_improve = mu < f_best
    
    # Mask for candidates in top 5% of sigma
    mask_top_unc = sigma >= sigma_95
    
    # Combined mask: mu < f_best AND sigma >= sigma_95
    mask_selected = mask_improve & mask_top_unc
    
    # Initialize scores
    # Base score: -mu (lower mu is better)
    scores = -mu
    
    # For selected candidates, add a large bonus to ensure they are picked.
    # What is a good bonus?
    # The scale of -mu is roughly the function values (e.g., -3 to 0).
    # Let's add a bonus that dominates the -mu term for selected candidates.
    # Bonus = 10 * max(-mu) ? Or just a fixed large number?
    # If we add a fixed large number, all selected candidates have the same score.
    # The evaluator picks the one with the highest score. If tied, it might pick arbitrarily or the first one.
    # To break ties among selected candidates, we should differentiate them.
    # We can use their EI score or their specific sigma rank as a tie-breaker.
    
    # Let's use EI as the tie-breaker for selected candidates.
    # For non-selected candidates, we use EI as well? Or just -mu?
    # If we use EI for everyone, it's not a structural change.
    # Let's try:
    # If selected: Score = 100 + EI (ensures selection)
    # If not selected: Score = EI (standard behavior)
    
    # This is a hybrid. Is it structural?
    # It changes the selection logic from "max EI" to "max EI among top 5% uncertainty (if improving), else max EI".
    # This forces exploration of high uncertainty regions that have a chance of improvement.
    
    # Let's implement this.
    
    final_scores = ei_scores.copy()
    
    if np.any(mask_selected):
        # For selected candidates, boost the score significantly
        # We want them to be higher than any non-selected candidate.
        # Max EI is usually small. Let's add a large constant.
        # But we must ensure finite scores.
        boost = 1000.0
        final_scores[mask_selected] += boost
        
        # To differentiate among selected candidates, we already have EI added.
        # EI varies, so they won't all be exactly 1000.
        # This ensures the top EI among the top 5% uncertainty (with mu < f_best) is picked.
        # If no candidates satisfy the condition, we fall back to standard EI.
        
    # Check deadline
    if time.time() > deadline:
         # Return best effort
         pass
         
    # Ensure finite
    final_scores = np.where(np.isfinite(final_scores), final_scores, -np.inf)
    
    return final_scores
# EVOLVE-BLOCK-END
