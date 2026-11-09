# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state for the acquisition function
# This persists across calls within the same seed/evaluation run.
_acq_state = {
    'history': [],       # List of (scores_vector, f_best_before, f_best_after)
    'alphas': np.array([1.0, 1.0, 1.0, 1.0, 1.0]), # Dirichlet priors for 5 candidates
    'candidates': None,  # List of lambda functions (mu, sigma, f_best) -> scores
    'initialized': False
}

def _init_candidates():
    """Initialize the set of acquisition candidate functions."""
    # Candidate 1: Expected Improvement (EI)
    def ei(mu, sigma, f_best):
        z = (f_best - mu) / sigma
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Candidate 2: LCB with beta=1.0 (kappa=1.0)
    def lcb_1(mu, sigma, f_best):
        return 1.0 * sigma - mu
        
    # Candidate 3: LCB with beta=2.0 (kappa=2.0)
    def lcb_2(mu, sigma, f_best):
        return 2.0 * sigma - mu
        
    # Candidate 4: LCB with beta=4.0 (kappa=4.0)
    def lcb_4(mu, sigma, f_best):
        return 4.0 * sigma - mu
        
    # Candidate 5: PI (Probability of Improvement)
    def pi(mu, sigma, f_best):
        z = (f_best - mu) / sigma
        return norm.cdf(z)
        
    _acq_state['candidates'] = [ei, lcb_1, lcb_2, lcb_4, pi]

def acquisition(mu, sigma, f_best):
    t_start = time.time()
    deadline = t_start + 290.0 # Safety margin

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    n = len(mu)
    
    if n == 0:
        return np.array([], dtype=np.float64)

    if not _acq_state['initialized']:
        _init_candidates()
        _acq_state['initialized'] = True

    sigma_safe = np.where(sigma > 1e-10, sigma, 1e-10)
    
    # 1. Update weights based on history
    # We have a history of past steps. For each step, we know which candidate
    # was selected (implicitly the one that produced the max score in the ensemble, 
    # but in Thompson Sampling we usually sample one candidate. 
    # However, the interface requires returning a score vector for the *next* point.
    # The loop picks argmax(score).
    # 
    # Wait, the standard Thompson Sampling approach in this context (where we must return a score vector 
    # that dictates a single point) is slightly different.
    # 
    # Let's refine the strategy:
    # Instead of returning a blended score, we can use the "Optimistic" or "Bayesian" selection.
    # But the most effective use of the "score vector" interface with adaptive weights is:
    # 
    # Actually, a simpler and very robust "Adaptive" strategy that fits the "Score Vector" interface:
    # We maintain a set of candidate acquisition functions.
    # At each step, we compute the scores for all candidates.
    # We then select a candidate index k with probability proportional to its "success weight".
    # But we must return a single score vector.
    # 
    # Alternative: **Weighted Sum**.
    # We maintain weights $w_k$ for each candidate $k$.
    # We return $S(x) = \sum_k w_k S_k(x)$.
    # The weights are updated based on whether the previous selection (argmax of the weighted sum) 
    # led to an improvement.
    # 
    # Let's implement a Weighted Sum with adaptive weights.
    # Weights start at 1.0.
    # If the previous point improved f_best, we increase the weight of the candidate that 
    # contributed most to the selection? Or simply increase all? 
    # Better: Attribute the success to the candidate that had the highest *relative* score 
    # at the selected point.
    
    history = _acq_state['history']
    weights = _acq_state['alphas'] # Reusing alphas as weights for simplicity, or use a separate list
    
    # Compute scores for all candidates
    scores_list = []
    for func in _acq_state['candidates']:
        s = func(mu, sigma_safe, f_best)
        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        scores_list.append(s)
    
    scores_array = np.vstack(scores_list) # Shape: (K, N)
    
    # Normalize each candidate's scores to [0, 1] or similar to make them comparable?
    # Or just use raw scores if they are on similar scales.
    # EI and LCB have different scales. Normalizing to mean 0, std 1 is safer.
    for k in range(len(_acq_state['candidates'])):
        s_k = scores_array[k]
        std_k = np.std(s_k)
        if std_k > 1e-9:
            scores_array[k] = (s_k - np.mean(s_k)) / std_k
        else:
            scores_array[k] = 0.0
            
    # Current weighted sum
    # Weights are in _acq_state['alphas']
    # Ensure weights are positive
    weights = np.maximum(_acq_state['alphas'], 1e-6)
    weights = weights / np.sum(weights)
    
    final_scores = np.dot(weights, scores_array)
    
    # Update logic:
    # We need to know if the *previous* step resulted in an improvement.
    # The state `history` stores tuples: (f_best_before, f_best_after, selected_idx_hint)
    # But we don't know which candidate "won" in the previous step if we used a weighted sum.
    # 
    # Let's switch to a simpler adaptive heuristic that doesn't require complex attribution:
    # **UCB for Acquisition Functions**.
    # We track the number of times each candidate has been "dominant" (i.e., its normalized score was highest at the selected point) 
    # and the average improvement when it was dominant.
    # 
    # However, implementing full attribution in this single-call interface is tricky because 
    # we don't get explicit feedback on *which* part of the weighted sum caused the selection.
    # 
    # Let's try a different, cleaner strategy: **Pure Thompson Sampling on the Candidate Index**.
    # 1. Compute normalized scores for all K candidates.
    # 2. Sample a candidate index $k^*$ from a Categorical distribution defined by the Dirichlet posterior.
    # 3. Return the scores from candidate $k^*$ as the final scores.
    # 4. After the evaluator picks the point and calls us again, we can't easily update the posterior 
    #    unless we store the *selected point's features* or just rely on the fact that 
    #    if f_best improved, the sampled candidate was "good".
    # 
    # Problem: We don't know if the *sampled* candidate was the one that effectively dictated the move 
    # if we just return its scores. Yes we do! If we return $S_{k^*}(x)$, the evaluator picks $x^* = \arg\max S_{k^*}(x)$.
    # So the move is entirely attributable to candidate $k^*$.
    # 
    # So the algorithm is:
    # 1. Compute normalized scores for all K candidates.
    # 2. Sample $k^*$ from Dirichlet-multinomial or just Categorical using current alphas.
    # 3. Return scores from candidate $k^*$.
    # 4. Store (k^*, f_best_before) in a pending buffer.
    # 5. On the NEXT call, compare new f_best with stored f_best_before.
    #    If improved, increment alpha[k^*].
    #    If not, decrement alpha[k^*] (or leave it, but usually we want to penalize failure).
    #    Actually, standard TS updates the reward. Reward = 1 if improved, 0 otherwise.
    #    Beta updates: alpha[k^*] += reward, beta[k^*] += (1-reward).
    #    We only track alpha (successes) and assume total trials? 
    #    Better to track alpha and beta.
    
    if 'betas' not in _acq_state:
        _acq_state['betas'] = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        _acq_state['pending'] = None # (k_selected, f_best_before)

    # Get current alphas and betas
    alphas = _acq_state['alphas']
    betas = _acq_state['betas']
    
    # Update pending from previous step
    if _acq_state['pending'] is not None:
        k_prev, f_best_prev = _acq_state['pending']
        if f_best < f_best_prev:
            # Improvement
            alphas[k_prev] += 1.0
        else:
            # No improvement
            betas[k_prev] += 1.0
        _acq_state['pending'] = None
        _acq_state['alphas'] = alphas
        _acq_state['betas'] = betas

    # Sample a candidate index k
    # Use Categorical distribution with probabilities proportional to alphas?
    # Or sample from Dirichlet to get weights, then pick argmax?
    # Standard TS for bandits: Sample $\theta_k \sim \text{Beta}(\alpha_k, \beta_k)$ and pick $\arg\max \theta_k$.
    # But here we want to pick a candidate to *use* for the score vector.
    # Let's sample the "quality" of each candidate and pick the best one.
    
    theta = np.random.beta(alphas, betas)
    k_selected = np.argmax(theta)
    
    # Store this selection for future update
    _acq_state['pending'] = (k_selected, f_best)
    
    # Return the scores from the selected candidate
    final_scores = scores_array[k_selected]
    
    # Ensure finite
    final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    if time.time() > deadline:
        pass
        
    return final_scores

# EVOLVE-BLOCK-END
