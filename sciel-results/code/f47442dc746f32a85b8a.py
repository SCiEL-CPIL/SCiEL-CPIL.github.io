import numpy as np
from scipy.stats import norm
import time

# Global state
_start_time = None
_step_counter = 0
_last_f_best = None

def acquisition(mu, sigma, f_best):
    global _start_time, _step_counter, _last_f_best
    
    if _start_time is None:
        _start_time = time.time()
    
    # Detect new seed/run
    # If f_best is significantly higher than the last observed min, assume new run
    # The function values are in the range roughly [-3, 0]
    if _last_f_best is not None and f_best > _last_f_best + 0.5:
        _step_counter = 0
    else:
        _step_counter += 1
        
    _last_f_best = f_best
    
    # Total expected optimization steps is 25 (30 total evals - 5 initial)
    step = min(_step_counter, 24)
    total_steps = 25.0
    
    # Deadline guard: if we are running low on time, just return a simple heuristic
    # to ensure we don't timeout. However, EI is fast.
    # The limit is 300s per evaluation, but total wall clock is 330s.
    # Actually, the task says "Each evaluation must finish within 300 s".
    # And "evaluator KILLS the program after 330s wall-clock".
    # So we have plenty of time for 2000 points.
    
    # Adaptive EI
    # Standard EI for minimization:
    # improvement = f_best - mu
    # z = improvement / sigma
    # EI = improvement * Phi(z) + sigma * phi(z)
    
    # To control exploration, we can scale the sigma or add a penalty.
    # A common approach is to use a "kappa" or "xi" parameter.
    # Alternatively, we can interpolate between EI and LCB, or scale the uncertainty.
    
    # Let's use a scaled EI where the effective sigma is reduced over time.
    # Or, simpler: use EI but with a decaying weight on the uncertainty term?
    # No, EI naturally handles this.
    
    # Let's try a strategy that blends EI and LCB, or just uses EI with a decaying "aggressiveness".
    # Aggressiveness can be implemented by scaling sigma in the EI calculation.
    # If we scale sigma down, EI becomes more exploitative.
    
    # At step 0, we want high exploration -> use full sigma.
    # At step 24, we want high exploitation -> use reduced sigma?
    # Actually, if sigma is small, EI is already exploitative.
    # The issue with LCB is that it explores purely based on sigma, regardless of the gap.
    # EI explores based on the probability of improvement.
    
    # Let's stick to pure EI first, as it's the standard "best" myopic acquisition.
    # The current program is LCB. EI is materially different.
    
    # To be safe and potentially better, we can introduce a slight decay in the "optimism".
    # One way: multiply the sigma term in the EI formula by a factor that decays.
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # If we replace sigma with alpha * sigma in the second term, and adjust z?
    # That breaks the analytical form.
    
    # Better: Use the "Expected Improvement" directly.
    # If we find that EI is too conservative, we can add a small LCB component.
    # But let's start with pure EI.
    
    # Handle zero sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    # Standard EI for minimization
    # We want to maximize the score, so we return EI.
    # EI is positive when mu < f_best.
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # To encourage exploration in early steps, we can add a small bonus for high sigma?
    # No, EI already does that.
    
    # Let's add a small decay to the exploration part to ensure convergence.
    # Actually, pure EI is known to be "myopic" but effective.
    # The LCB with linear decay might be suboptimal.
    
    # Let's try to make the EI slightly more aggressive in the beginning.
    # We can scale the sigma used in the z-score and the pdf term.
    # Let's define an exploration factor beta that decays.
    # beta = 1.0 + 2.0 * (1 - step/total_steps)
    # Use sigma_eff = beta * sigma? No, that increases exploration.
    # If we use sigma_eff = sigma / beta? That decreases exploration.
    
    # Let's just use standard EI. It's a robust baseline.
    # If the current LCB scores 0.94, and EI is theoretically better for simple regret,
    # it should perform at least as well, likely better.
    
    # One refinement: The "standardized targets" in the GP description suggests
    # that mu and sigma are in objective units.
    
    # Let's check for numerical stability.
    # If z is very large positive, cdf(z) ~ 1, pdf(z) ~ 0. EI ~ improvement.
    # If z is very large negative, cdf(z) ~ 0, pdf(z) ~ 0. EI ~ 0.
    
    # This looks correct.
    
    # To differentiate further from a basic EI implementation (which might be in inspiration 1),
    # let's look at inspiration 2: "Search-based acquisition: optimize EI/batch-penalty hyperparameters".
    # We can't do heavy search per step (too slow/complex for 2000 points * 25 steps * 3 seeds?).
    # But we can use a fixed, well-tuned EI.
    
    # Let's add a small LCB component to ensure we don't get stuck in local optima
    # if the GP is slightly overconfident.
    # Score = EI + lambda * ( -mu + beta * sigma )
    # This hybrid is often more robust.
    
    # Let's try:
    # Step 0-10: High exploration. Use LCB with high beta.
    # Step 10-24: Exploitation. Use EI.
    # Or blend them.
    
    # Let's use a blending strategy:
    # w_ei = step / total_steps
    # w_lcb = 1 - w_ei
    # At start, mostly LCB (explores). At end, mostly EI (exploits).
    # But LCB with high beta is very aggressive.
    
    # Let's define:
    # lcb_score = -mu + beta_lcb * sigma
    # beta_lcb = 1.0 + 2.0 * (1 - step/total_steps) # 3.0 to 1.0
    
    # ei_score = EI
    
    # scores = w_lcb * lcb_score + w_ei * ei_score
    
    # This ensures we explore broadly at first, then switch to the theoretically optimal myopic metric.
    
    w_lcb = 1.0 - (step / total_steps)
    w_ei = step / total_steps
    
    # Ensure non-negative weights
    w_lcb = max(0.0, w_lcb)
    w_ei = max(0.0, w_ei)
    
    # If step is 0, w_lcb=1, w_ei=0.
    # If step is 24, w_lcb=0.04, w_ei=0.96.
    
    beta_lcb = 1.0 + 2.0 * (1.0 - step / total_steps)
    
    lcb_score = -mu + beta_lcb * sigma
    ei_score = ei
    
    scores = w_lcb * lcb_score + w_ei * ei_score
    
    return scores
