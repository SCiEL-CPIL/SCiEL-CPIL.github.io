"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guarding
_START_TIME = time.time()
# Safety margin in seconds (300s limit, leave 10s buffer)
_DEADLINE_MARGIN = 290.0


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    Combines Expected Improvement (EI) with a Thompson Sampling-like stochastic exploration term.
    """
    # Deadline guard: if we are too close to the time limit, return a safe, fast result
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE_MARGIN:
        # Fallback to a simple exploitation score to ensure we return quickly
        return -mu
    
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Compute Expected Improvement (EI)
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Compute a deterministic exploration bonus based on uncertainty
    # This mimics the diversity of Thompson Sampling but in a deterministic way
    # We scale the bonus by the sigma to encourage exploration in uncertain regions
    # but weight it so that EI remains the dominant factor for exploitation.
    # A common heuristic is to add a term proportional to sigma, but we want to
    # combine it with EI.
    # Let's try a blend: score = EI + beta * sigma
    # However, the inspiration suggests TS (sampling) was better.
    # A deterministic approximation of "potential improvement" considering uncertainty
    # is often just EI. But TS draws a sample.
    # Let's try a "Stochastic EI" approximation or a weighted sum.
    # The prompt says "Combine the strategy of the current program (EI) with the strongest ideas of the inspirations (TS)".
    # The strongest idea of TS is that it explores more.
    # We can approximate the expected value of the max of a sample and the current best?
    # Or simply use a UCB-like term added to EI?
    # Let's try: Score = EI + kappa * sigma
    # This is similar to UCB but added to EI.
    # Alternatively, we can use a "Probability of Improvement" (PI) term?
    # PI = Phi(z).
    # Let's stick to a robust blend.
    # The TS program used a fixed seed and returned -samples.
    # Since we cannot use randomness easily in a deterministic acquisition (unless we use a fixed seed per call, which is deterministic per state but random across calls? No, the function is called each step. If we use a fixed seed, it's deterministic. If we use a random seed, it's non-deterministic. The prompt implies determinism is preferred or at least stable. The TS program used a fixed seed 42. This means the "random" draw is actually a deterministic perturbation based on the input size and seed.
    # Let's replicate the deterministic Thompson Sampling logic but blend it with EI.
    
    # Deterministic TS sample
    rng = np.random.default_rng(42)
    # Note: rng.standard_normal(len(mu)) will always produce the same sequence of numbers for the same length.
    # This means the "noise" is fixed for the 2000 points.
    ts_samples = mu + sigma_safe * rng.standard_normal(len(mu))
    
    # We want to select the point with the lowest function value.
    # So for TS, the score should be -ts_samples.
    # For EI, the score is ei.
    
    # How to combine?
    # Maybe we can take the maximum of the two? No, they have different scales.
    # Maybe a weighted sum?
    # Let's look at the scores.
    # EI is non-negative.
    # -ts_samples can be anything.
    # If we add them, the scales might not match.
    # Let's try to normalize or just use a simple heuristic.
    # Actually, the TS program scored 3.48, EI 1.73.
    # This suggests that for this problem, the exploration provided by TS (even deterministic) was much better.
    # Why? Maybe because EI gets stuck in local minima or doesn't explore enough with 30 evals.
    # So we should lean heavily towards the TS-like behavior.
    # Let's try returning a score that is dominated by the TS sample but slightly guided by EI.
    # Or simply return -ts_samples? That is exactly the TS program.
    # The directive says "Combine... do not just concatenate".
    # Maybe we can use EI to guide the *variance* of the TS sample?
    # Or use a "Softmax" selection? No, we must return scores.
    
    # Let's try a different combination:
    # Score = -ts_samples + alpha * EI
    # If alpha is small, it's mostly TS.
    # If alpha is large, it's mostly EI.
    # Given TS >> EI, let's start with a small alpha.
    # But wait, EI is in "improvement" units, while -ts_samples is in "function value" units.
    # f_best is around 0-1. mu is around 0-1.
    # EI is small (improvement).
    # So adding EI to -ts_samples might be negligible.
    
    # Alternative: Use the TS sample but perturb it with EI?
    # Or, use a "Bayesian" combination?
    
    # Let's look at the "virtual population" strategy in the directive:
    # "assign each candidate a score based on its marginal contribution (Shapley-like value) to this global improvement"
    # This is complex.
    
    # Let's stick to the simpler TS + EI blend.
    # To make the scales comparable, we can scale EI by f_best or something?
    # No, let's just try a weighted sum with a small weight for EI.
    # Or, perhaps, use EI to *modulate* the TS exploration?
    # e.g., if EI is high, we are exploiting. If EI is low, we explore more?
    
    # Actually, the simplest "structural change" that combines them is to use the TS sample as the primary signal, but add a small EI component to break ties or guide slightly.
    # However, since the TS program was deterministic with a fixed seed, the "randomness" is actually a fixed offset.
    # Let's try:
    # score = -ts_samples
    # This is the TS program.
    # To combine, let's add a small fraction of EI.
    # score = -ts_samples + 0.1 * EI
    
    # But wait, if the TS program already scored 3.48, and we just add a small EI term, it might not change much.
    # The directive says "Make a STRUCTURAL change".
    # The TS program is already a structural change from EI.
    # The current program is EI.
    # So combining them means moving from pure EI to a hybrid.
    
    # Let's try a different approach:
    # Use EI to determine the "confidence" and TS to determine the "exploration".
    # Score = EI * (1 + beta * (sigma / sigma_max)) ?
    
    # Let's go with the weighted sum, but normalize EI to have similar magnitude to the function values?
    # No, that's risky.
    
    # Let's try:
    # score = -ts_samples
    # This is the best performing inspiration.
    # The current program is EI.
    # The combination should be a blend.
    # Let's try: score = -ts_samples + 0.5 * EI
    # If this doesn't work, we can adjust.
    
    # One issue: ts_samples are in function units. EI is in improvement units.
    # If f_best is 0.1, and mu is 0.2, EI is small.
    # If we add 0.5 * EI to -ts_samples (which is around -0.2), it's a small perturbation.
    
    # Let's try this.
    
    scores = -ts_samples + 0.5 * ei
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores

# EVOLVE-BLOCK-END
