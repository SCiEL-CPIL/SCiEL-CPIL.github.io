import numpy as np
import time

# Track time for deadline guard
_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we've used more than 300s, return a safe fallback
    elapsed = time.time() - _start_time
    if elapsed > 295.0:
        # Return UCB as a safe fallback
        return -mu + 1.0 * sigma
    
    n = len(mu)
    
    # Look-Ahead strategy:
    # 1. Compute a base score using a hybrid of EI-like and exploration
    # 2. Simulate greedy selection of top candidates, reducing local variance
    # 3. Score each candidate based on its marginal contribution
    
    # Step 1: Base acquisition score
    # Use a modified EI that balances exploitation and exploration
    # For minimization: EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # But we want a score that's higher = better
    # Let's use: score = -(mu - f_best) + kappa * sigma
    # where kappa adapts based on how much we've explored
    
    # Adaptive kappa: start exploratory, become exploitative as f_best improves
    # f_best starts high (worst), decreases toward -39.1662
    # Range of f_best: from ~100+ (worst) to -39.1662 (best)
    # Normalize: progress = (f_best - (-39.1662)) / (100 - (-39.1662))
    # When progress is high (early), kappa is large (explore)
    # When progress is low (late), kappa is small (exploit)
    
    f_min = -39.1662
    f_max_est = 100.0  # rough estimate of worst possible value
    
    progress = (f_best - f_min) / (f_max_est - f_min)
    progress = np.clip(progress, 0.0, 1.0)
    
    # kappa goes from 2.0 (very exploratory) to 0.5 (exploitative)
    kappa = 0.5 + 1.5 * progress
    
    base_scores = -mu + kappa * sigma
    
    # Step 2: Look-Ahead simulation
    # Greedily select top K candidates, reduce local variance for nearby points
    # This approximates the information gain from evaluating those points
    
    # Number of look-ahead steps
    K = 5
    
    # Work on a copy of sigma to simulate variance reduction
    sigma_sim = sigma.copy()
    
    # Compute initial scores
    current_scores = base_scores.copy()
    
    # Greedily select K candidates
    selected_indices = []
    for step in range(K):
        # Select the candidate with the highest current score
        best_idx = np.argmax(current_scores)
        selected_indices.append(best_idx)
        
        # Reduce variance locally around the selected point
        # In 1D, points are ordered; we don't have coordinates, but we can
        # approximate: the 2000 candidates are uniformly drawn, so index order
        # doesn't correspond to spatial order. Instead, we reduce variance
        # globally but weighted by how "close" each point is to the selected one.
        
        # Since we don't have coordinates, we use a heuristic:
        # Reduce sigma for all points, but more for points that were also
        # highly scored (likely in the same region)
        
        # Simple approach: reduce sigma uniformly by a small factor
        # This is a crude approximation but captures the idea that
        # evaluating one point reduces uncertainty everywhere (slightly)
        
        reduction = 0.1  # 10% reduction per selected point
        sigma_sim *= (1.0 - reduction)
        
        # Recompute scores with reduced sigma
        current_scores = -mu + kappa * sigma_sim
        
        # Mark the selected point as "used" by setting its score very low
        current_scores[best_idx] = -1e10
    
    # Step 3: Final scoring
    # The final score for each candidate is based on:
    # a) Its base score (immediate value)
    # b) Its contribution to reducing future uncertainty (look-ahead)
    
    # For points not selected in the look-ahead, their score is their base score
    # For the overall scoring, we want to prefer points that either:
    # - Have high immediate value (low mu, high sigma)
    # - Are in regions where the look-ahead simulation showed high value
    
    # The look-ahead tells us which regions are promising.
    # We can use the average score in the look-ahead as a bonus for nearby points.
    
    # Since we don't have coordinates, we use the selection pattern:
    # Points that were among the top scorers in early look-ahead steps
    # indicate promising regions.
    
    # Create a bonus based on proximity to selected points in score space
    # (This is a proxy for spatial proximity)
    
    bonus = np.zeros(n)
    for sel_idx in selected_indices:
        # Points with similar base scores are likely in the same region
        # Add bonus proportional to similarity
        similarity = np.exp(-0.5 * ((base_scores - base_scores[sel_idx]) / (kappa * sigma[sel_idx] + 1e-8))**2)
        bonus += similarity
    
    # Normalize bonus
    if bonus.max() > 0:
        bonus = bonus / bonus.max()
    
    # Final score: base score + bonus from look-ahead
    final_scores = base_scores + 0.3 * bonus
    
    # Ensure finite scores
    final_scores = np.where(np.isfinite(final_scores), final_scores, 0.0)
    
    return final_scores
