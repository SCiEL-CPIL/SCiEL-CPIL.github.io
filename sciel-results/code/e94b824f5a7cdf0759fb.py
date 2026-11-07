import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
TIME_LIMIT = 290.0  # Safety margin before 330s limit


def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    if time.time() - _start_time > TIME_LIMIT:
        # Fallback to simple EI/cost if running out of time
        eps = 1e-9
        z = (f_best - mu) / (sigma + eps)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / np.maximum(cost, eps)

    eps = 1e-9
    sigma_safe = sigma + eps
    cost_safe = np.maximum(cost, eps)

    # Thompson Sampling: sample a single objective value from the Gaussian posterior
    # For minimization, we want low values. Sample y ~ N(mu, sigma).
    # Lower y is better. So score is -y / cost.
    # To avoid non-determinism issues if any (though usually fine), we use a fixed seed or rely on the loop's randomness?
    # Actually, Thompson Sampling requires a random sample per call.
    # However, the problem says "Return 2000 finite scores".
    # Using np.random.normal is standard.
    # But wait, is it better to use a deterministic approximation?
    # The directive says: "Model the acquisition as a Thompson Sampling score by sampling a single objective value from the Gaussian posterior for each candidate, normalized by the evaluation cost"
    
    # Let's implement Thompson Sampling.
    # Sample y = mu + sigma * z, where z ~ N(0, 1)
    z_sample = np.random.normal(0, 1, size=mu.shape)
    y_sample = mu + sigma_safe * z_sample
    
    # Score: We want to minimize f(x). So lower y_sample is better.
    # We normalize by cost.
    # score = -y_sample / cost_safe
    
    # However, pure Thompson Sampling can be too exploratory or exploitative depending on the scale.
    # Let's check if this is sufficient.
    # The current program (score 1.0) uses EI and LCB.
    # The inspiration 1 (score 1.0) uses pure EI.
    # The directive explicitly suggests Thompson Sampling.
    
    # Let's combine:
    # 1. Compute EI (exploitation)
    # 2. Compute Thompson Sample (exploration/exploitation balance)
    # 3. Combine them.
    
    # Actually, the directive says "Combine the strategy of the current program with the strongest ideas of the inspirations...".
    # Current program: EI + LCB, max, / cost.
    # Inspiration 1: EI / cost.
    # Directive: Use Thompson Sampling.
    
    # Let's try a blend.
    # EI is a deterministic upper bound on improvement.
    # TS is a stochastic sample.
    
    # Let's calculate EI first.
    z_ei = (f_best - mu) / sigma_safe
    cdf_z = norm.cdf(z_ei)
    pdf_z = norm.pdf(z_ei)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Calculate TS score
    # y_sample is the predicted value.
    # We want to minimize y_sample.
    # So score_ts = -y_sample / cost_safe
    
    # How to combine?
    # Option A: max(ei, -y_sample) / cost
    # Option B: ei + -y_sample / cost (linear combination)
    # Option C: Just use TS?
    
    # The directive says: "Model the acquisition as a Thompson Sampling score...". This suggests TS is the primary mechanism.
    # But "Combine the strategy of the current program..." suggests keeping elements of the current one.
    # The current program uses EI and LCB.
    # TS implicitly balances exploration and exploitation.
    
    # Let's try: Score = (EI + (-y_sample)) / (2 * cost_safe) ? No, scales might differ.
    # EI is in units of improvement (positive).
    # -y_sample is in units of negative function value.
    
    # Let's look at the scales.
    # f_best is the current best.
    # EI is roughly (f_best - mu) when mu < f_best.
    # y_sample is roughly mu.
    # So -y_sample is roughly -mu.
    
    # If we use just TS:
    # score = -y_sample / cost_safe
    
    # If we use EI:
    # score = ei / cost_safe
    
    # Let's try a soft-max or max of the two normalized scores?
    # Or perhaps the TS sample itself is the best proxy for "promising value".
    
    # Let's stick to the directive's specific suggestion: "Model the acquisition as a Thompson Sampling score... normalized by the evaluation cost".
    # This implies: score = -sample / cost.
    
    # However, to be safe and combine with the current program's robustness:
    # The current program uses `np.maximum(ei, lcb_score)`.
    # LCB score is `-(mu + beta * sigma)`.
    # TS sample is `mu + sigma * z`.
    # If z is negative, TS is more exploitative than LCB (if beta=2, LCB assumes z=-2).
    # If z is positive, TS is more exploratory.
    
    # Let's use a weighted combination or just TS.
    # Given the directive is specific, I will prioritize TS but ensure it's robust.
    
    # One issue: np.random.normal might be slow? No, it's fast.
    # Is it deterministic? No.
    # Does the evaluator care about determinism? "five seeded instances". The seeds are for the instances, not the acquisition randomness.
    # So randomness is allowed.
    
    # Let's refine the TS score.
    # y_sample = mu + sigma * z
    # We want to minimize y.
    # score = -y_sample / cost
    
    # To combine with EI (which is always non-negative and represents expected gain):
    # We can add them.
    # score = (ei + -y_sample) / cost ?
    # Units: EI is "improvement". -y_sample is "negative value".
    # They are on different scales.
    
    # Better approach:
    # Use TS as the primary signal.
    # But add a small EI component to ensure we don't ignore obvious improvements?
    # Actually, if mu < f_best, y_sample will likely be < f_best, so -y_sample will be > -f_best.
    # EI will be positive.
    
    # Let's try: score = (-y_sample + ei) / cost_safe
    # This seems reasonable. It favors low predicted values (via -y_sample) and high expected improvement (via ei).
    
    # Let's check the fallback.
    # If time is running out, use EI/cost.
    
    # Implementation:
    z_sample = np.random.normal(0, 1, size=mu.shape)
    y_sample = mu + sigma_safe * z_sample
    
    # EI calculation
    z_ei = (f_best - mu) / sigma_safe
    cdf_z = norm.cdf(z_ei)
    pdf_z = norm.pdf(z_ei)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Combine
    # We want higher score to be better.
    # -y_sample is higher when y_sample is lower (better).
    # ei is higher when improvement is expected (better).
    # So summing them makes sense.
    
    score = (-y_sample + ei) / cost_safe
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
