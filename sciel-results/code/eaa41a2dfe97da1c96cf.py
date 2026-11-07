"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Batch-aware Look-Ahead acquisition function.
    
    Evaluates the top 20 candidates by simulating the next 5 evaluations
    starting from each candidate, using proper posterior variance updates.
    The score of a candidate is the sum of the EIs of the 5 picks in that simulation.
    Other candidates retain their standard EI score.
    """
    n = len(mu)
    
    # Compute base EI scores for all points
    z = (f_best - mu) / (sigma + 1e-9)
    ei_scores = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Identify top 20 candidates for detailed look-ahead
    top_k = 20
    top_indices = np.argsort(ei_scores)[-top_k:][::-1]
    
    # Final scores array, initialized with base EI
    final_scores = ei_scores.copy()
    
    # Helper to simulate a batch starting from a given index
    def simulate_batch(start_idx, mu_init, sigma_init):
        mu_work = mu_init.copy()
        sigma_work = sigma_init.copy()
        used = np.zeros(n, dtype=bool)
        used[start_idx] = True
        
        # Initial score is the EI of the start point
        z_start = (f_best - mu_work[start_idx]) / (sigma_work[start_idx] + 1e-9)
        batch_sum = (f_best - mu_work[start_idx]) * norm.cdf(z_start) + sigma_work[start_idx] * norm.pdf(z_start)
        
        current_scores = ei_scores.copy()
        current_scores[start_idx] = -np.inf
        
        for step in range(1, 5):
            # Pick next best
            best_idx = np.argmax(current_scores)
            if best_idx == -1 or not np.isfinite(best_idx):
                break
            used[best_idx] = True
            
            # Update variance for nearby points
            window = 600
            i_start = max(0, best_idx - window)
            i_end = min(n, best_idx + window + 1)
            
            indices = np.arange(i_start, i_end)
            dists = np.abs(indices - best_idx)
            r = dists / 400.0
            sqrt3 = np.sqrt(3.0)
            corr = (1.0 + sqrt3 * r + r**2) * np.exp(-sqrt3 * r)
            
            # Correct variance update
            mask_unused = ~used[i_start:i_end]
            if np.any(mask_unused):
                idx_to_update = indices[mask_unused]
                sigma_work[idx_to_update] *= np.sqrt(1.0 - corr[mask_unused]**2 + 1e-12)
            
            # Recompute scores for affected region
            z_local = (f_best - mu_work[i_start:i_end]) / (sigma_work[i_start:i_end] + 1e-9)
            new_ei = (f_best - mu_work[i_start:i_end]) * norm.cdf(z_local) + sigma_work[i_start:i_end] * norm.pdf(z_local)
            
            # Update current_scores
            current_scores[i_start:i_end] = new_ei
            current_scores[i_start:i_end][used[i_start:i_end]] = -np.inf
            
            # Add this step's EI to the batch sum
            batch_sum += current_scores[best_idx]
            
        return batch_sum

    # For each top candidate, simulate and update its score
    for idx in top_indices:
        # We use the original mu and sigma for each simulation
        # This approximates the "expected" batch value if we picked this point first
        batch_val = simulate_batch(idx, mu, sigma)
        
        # The score for this candidate is the batch value
        # We might want to blend it with the original EI to avoid instability
        # But the directive says "scoring candidates based on their expected contribution"
        # So we replace the score with the batch value
        final_scores[idx] = batch_val
        
    # Ensure finite values
    final_scores = np.where(np.isfinite(final_scores), final_scores, 0.0)
    
    return final_scores
# EVOLVE-BLOCK-END
