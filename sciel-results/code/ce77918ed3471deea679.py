"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Thompson Sampling acquisition via a finite mixture approximation.
    # We discretize the uncertainty field into components and draw a
    # sample from each, then score candidates by the mixture-weighted
    # posterior sample.
    n = len(mu)
    
    # Guard: ensure sigma is positive
    sigma = np.maximum(sigma, 1e-12)
    
    # Build a finite mixture by binning the sigma values into quantile bins.
    # This creates a GMM where each component represents a range of uncertainty.
    n_components = 10
    sigma_min = np.min(sigma)
    sigma_max = np.max(sigma)
    if sigma_max - sigma_min < 1e-12:
        # All similar uncertainty; just use single sample
        sample = mu + sigma * np.random.randn(n)
        return -sample  # Higher score means lower predicted value
    
    # Create bin edges at quantiles of sigma
    edges = np.linspace(sigma_min, sigma_max, n_components + 1)
    
    # Assign each candidate to a component based on its sigma value
    bin_indices = np.digitize(sigma, edges[1:-1])  # 0 to n_components-1
    
    # For each component, compute the mean and variance of the sampled values
    # Component weights are based on the number of candidates in each bin
    component_weights = np.zeros(n_components)
    component_means = np.zeros(n_components)
    component_vars = np.zeros(n_components)
    
    for c in range(n_components):
        mask = bin_indices == c
        count = np.sum(mask)
        if count == 0:
            component_weights[c] = 0.0
            component_means[c] = 0.0
            component_vars[c] = 0.0
        else:
            component_weights[c] = count / n
            # Use the midpoint of the bin for the component's representative sigma
            comp_sigma = 0.5 * (edges[c] + edges[c + 1])
            component_means[c] = np.mean(mu[mask])
            component_vars[c] = comp_sigma ** 2
    
    # Normalize weights
    total_weight = np.sum(component_weights)
    if total_weight < 1e-12:
        component_weights = np.ones(n_components) / n_components
    else:
        component_weights = component_weights / total_weight
    
    # Draw a sample from the mixture
    # Instead of drawing one sample from the full mixture, we compute the
    # expected value under the mixture for each candidate.
    
    # For each candidate, its score is the weighted sum over components of
    # the probability that the candidate belongs to that component times
    # the sampled value from that component.
    
    # More efficiently: for each candidate, we compute a weighted combination
    # of samples from each component, weighted by how well the candidate fits
    # that component's uncertainty level.
    
    # Simplified approach: draw one sample from each component and compute
    # a weighted average for each candidate.
    
    samples = np.zeros(n_components)
    for c in range(n_components):
        if component_weights[c] > 0:
            samples[c] = component_means[c] + np.sqrt(component_vars[c]) * np.random.randn()
        else:
            samples[c] = 0.0
    
    # For each candidate, compute the mixture-weighted sample
    # The weight for candidate i in component c is proportional to how close
    # sigma[i] is to the component's representative sigma
    scores = np.zeros(n)
    for c in range(n_components):
        if component_weights[c] <= 0:
            continue
        # Weight based on proximity to component center
        comp_center = 0.5 * (edges[c] + edges[c + 1])
        comp_bw = max(edges[1] - edges[0], 1e-12)
        # Gaussian weight
        w = np.exp(-0.5 * ((sigma - comp_center) / comp_bw) ** 2)
        w *= component_weights[c]
        scores += w * samples[c]
    
    # Normalize scores
    norm_w = np.sum(component_weights)
    if norm_w > 1e-12:
        scores /= norm_w
    
    # Higher score means lower predicted value, so negate
    # Also add a small exploration bonus for high uncertainty
    scores = -scores + 0.05 * sigma
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
