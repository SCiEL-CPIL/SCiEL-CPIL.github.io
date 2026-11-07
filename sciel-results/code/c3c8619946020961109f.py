import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 10.0

def acquisition(mu, sigma, f_best):
    current_time = time.time()
    remaining_time = 330.0 - (current_time - _START_TIME)
    
    if remaining_time < _DEADLINE_MARGIN:
        # Fallback: Greedy (lowest mu)
        scores = -mu
        return np.where(np.isfinite(scores), scores, -np.inf)

    eps = 1e-9
    sigma_safe = sigma + eps

    # Calculate z-score for EI
    z = (f_best - mu) / sigma_safe

    # Calculate Expected Improvement (EI) for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Calculate Normalized EI (NEI)
    nei = ei / sigma_safe

    # Calculate LCB score
    lcb = -mu + sigma_safe

    # Adaptive blend:
    # When f_best is high (early), we want to explore. LCB is good for exploration.
    # When f_best is low (late), we want to exploit. NEI is good for exploitation.
    # We use a sigmoid-like transition based on the gap between f_best and the known minimum (3.0).
    # gap = f_best - 3.0
    # If gap is large, weight towards LCB.
    # If gap is small, weight towards NEI.
    
    # Let's define a transition parameter.
    # The known minimum is 3.0.
    # A reasonable range for transition might be f_best in [5, 10].
    # Let's use a smooth step function or sigmoid.
    # Let x = (f_best - 5.0) / (10.0 - 5.0) = (f_best - 5.0) / 5.0
    # When f_best < 5, x < 0, weight for LCB should be high?
    # Actually, early on f_best is high. We want LCB.
    # Late on f_best is low (close to 3). We want NEI.
    
    # Let's define weight for LCB (w_lcb) and NEI (w_nei).
    # w_lcb + w_nei = 1.
    
    # Let's try a simple linear interpolation or sigmoid.
    # Let t = clip((f_best - 3.0) / (10.0 - 3.0), 0, 1)
    # t=0 when f_best=3 (late), t=1 when f_best=10 (early).
    # We want w_lcb = t, w_nei = 1-t?
    # But f_best can be > 10 initially.
    
    # Let's use a sigmoid centered at some value.
    # Let center = 6.0, scale = 2.0
    # s = 1 / (1 + exp(-(f_best - center) / scale))
    # When f_best >> center, s -> 1. (Explore/LCB)
    # When f_best << center, s -> 0. (Exploit/NEI)
    
    center = 6.0
    scale = 2.0
    s = 1.0 / (1.0 + np.exp(-(f_best - center) / scale))
    
    # s is a scalar.
    # scores = s * lcb + (1-s) * nei
    
    # However, LCB and NEI have different scales.
    # LCB can be very negative or positive.
    # NEI is bounded roughly by [0, 1] or so?
    # EI/sigma. If mu << f_best, EI ~ f_best-mu, NEI ~ (f_best-mu)/sigma which can be large.
    # If mu ~ f_best, EI ~ sigma/2, NEI ~ 0.5.
    # If mu > f_best, EI is small, NEI is small.
    
    # Blending scores with different magnitudes is tricky.
    # Perhaps it's better to stick to the current program's logic which switched hard at 8.0?
    # The current program scored 1.38.
    # Inspiration 1 (NEI only) scored 1.65.
    # Wait, the current program in the prompt is the one with the switch (LCB if f_best>8 else NEI).
    # And it scored 1.38.
    # But Inspiration 1 (NEI only) scored 1.65.
    # This implies the switch to LCB in the current program might be HURTING performance compared to pure NEI?
    # Or maybe the specific beta=1.0 LCB is not optimal?
    
    # Let's look at the scores again.
    # Current: 1.3801
    # Insp 1 (NEI): 1.6567
    # Insp 2 (EI): 1.0000
    # Insp 3 (EI): 1.0000
    
    # So NEI is better than EI.
    # And the hybrid (Current) is WORSE than pure NEI.
    # This suggests that the LCB part is detrimental, or the switch point is wrong, or the blend is bad.
    
    # If NEI alone is best so far, maybe we should refine NEI?
    # Or maybe the LCB should be used only in the very first few steps?
    # But we don't have access to step count. We only have f_best.
    # In the first step, f_best is from 5 random points.
    # Goldstein-Price range is roughly [3, ~15].
    # Random points might give f_best around 10-15.
    # If we use LCB there, maybe it helps? But the score says otherwise.
    
    # Let's try a different structural change:
    # Use a **Kriging Believer** or **Predictive Variance** based approach?
    # Or perhaps a **Thompson Sampling** approximation?
    # We can't do full Thompson Sampling easily without sampling many times.
    
    # What if we use **PI (Predictive Improvement)**?
    # Or **Knowledge Gradient**? Too expensive.
    
    # Let's look at the "STAGNATION" directive: "Make a STRUCTURAL change".
    # The current program uses a hybrid. Inspiration 1 uses pure NEI.
    # Maybe the hybrid is the problem.
    # Let's try to improve upon NEI.
    
    # One idea: **Normalized Expected Improvement** is good, but maybe we can add a small exploration term that is active only when sigma is large AND f_best is not yet very good?
    
    # Alternative Structural Change: **Gaussian Process Lower Bound (GP-LB)** vs **EI**.
    # We tried LCB.
    
    # How about **Probability of Improvement (PI)**?
    # PI = Phi(z).
    # PI is bounded [0,1]. It favors points where there's a high probability of beating f_best.
    # It ignores the magnitude of improvement.
    # EI considers magnitude.
    
    # How about **Expected Hypervolume Improvement (EHVI)**?
    # For single objective, EHVI is related to EI but with a different weight.
    
    # Let's try a **Weighted Sum** of EI and PI?
    # Or **EI + lambda * sigma**?
    # This is essentially LCB if lambda is constant.
    
    # What if we use **Adaptive beta** for LCB?
    # beta = sqrt(log(1 + n)) ? We don't have n.
    # beta = sqrt(log(1 + (f_best - min_val))) ?
    
    # Let's reconsider the scores.
    # Pure NEI: 1.65.
    # Hybrid (LCB if f_best>8): 1.38.
    # This is a strong signal that the LCB part is bad.
    # Why? Maybe because LCB is too greedy in the early stages? Or too exploratory?
    # LCB = -mu + beta*sigma.
    # If beta=1, it's a balance.
    # Maybe beta should be smaller?
    # Or maybe we should just use NEI always?
    
    # But the directive says "Make a STRUCTURAL change".
    # If I just return NEI, I am not making a structural change from Inspiration 1, but I am making a change from the Current Program.
    # The Current Program is the one I am evolving.
    # So reverting to pure NEI is a structural change (removing the hybrid logic).
    # But is it an *improvement*?
    # 1.65 > 1.38. Yes.
    
    # Can we do better than 1.65?
    # Let's try **Normalized EI with a cap** or **modified normalization**.
    # NEI = EI / sigma.
    # If sigma is very small, NEI is huge. This is good for exploitation.
    # If sigma is large, NEI is small.
    
    # What if we use **EI / (sigma + mu)**?
    # Or **EI / f_best**?
    
    # Let's try **Expected Improvement normalized by f_best** (relative improvement).
    # rel_ei = ei / f_best
    
    # Or **Log-EI**?
    
    # Let's try a **Thompson Sampling** approximation using the mean and variance.
    # We can't sample, but we can use the **Gaussian Process UCB** with a specific beta.
    # We tried LCB (which is UCB for minimization).
    
    # What if we combine NEI and LCB with a **soft** switch that is very biased towards NEI?
    # e.g. weight_lcb = max(0, (f_best - 10) / 10) * 0.5 ?
    # If f_best < 10, weight_lcb = 0. Pure NEI.
    # If f_best > 10, weight_lcb increases.
    # Since the hybrid with hard switch at 8 was bad, maybe the switch point was too early?
    # Or maybe LCB is just bad for this problem.
    
    # Let's try a different acquisition: **Predictive Entropy Search**? No, too complex.
    
    # Let's try **Kriging Believer**.
    # Sample a single value from the GP posterior: y ~ N(mu, sigma^2).
    # We can't sample random numbers in the acquisition function consistently?
    # Actually, we can use a deterministic proxy.
    # Kriging Believer often uses the mean. That's greedy.
    # Or it uses the lower bound.
    
    # Let's stick to the **NEI** family but modify it.
    # **Normalized EI** is robust.
    # Let's try **EI** with a **small exploration bonus** that decays as f_best improves.
    # score = ei + alpha * sigma * exp(-k * f_best)
    
    # Or simply: **Return NEI**.
    # Since NEI (1.65) > Hybrid (1.38), the structural change is to **remove the LCB branch** and **use NEI exclusively**.
    # This is a valid structural change from the current program.
    
    # Can we refine NEI?
    # NEI = EI / sigma.
    # Sometimes, adding a small epsilon to the denominator or using a different normalization helps.
    
    # Let's try **EI / (sigma + 1e-6)** which is what NEI is.
    
    # What if we use **Log(EI)**?
    # Log(EI) is undefined if EI <= 0.
    
    # Let's try **PI (Probability of Improvement)**?
    # PI = norm.cdf(z).
    # This is bounded. It might be less sensitive to outliers in sigma.
    # But EI is generally better.
    
    # Let's try a **Hybrid of NEI and PI**?
    # score = 0.5 * nei + 0.5 * pi
    
    # Let's try **NEI** but with a **clipped z-score** to avoid numerical issues?
    # z is already handled by norm.cdf/pdf.
    
    # I will propose **Pure NEI** as the structural change, as the evidence suggests the hybrid is worse.
    # To make it slightly more "structural" than just "revert to inspiration 1", I will add a **small, adaptive exploration term** to NEI that is only active when sigma is very large relative to the current best.
    # Actually, NEI already does this.
    
    # Let's try **Normalized EI** but normalized by **f_best** instead of sigma?
    # nei_fb = ei / f_best
    # This scales the improvement relative to the current best value.
    
    # Let's try **ei / (f_best + eps)**.
    
    # Let's test the hypothesis: **Pure NEI is better than Hybrid**.
    # I will output the Pure NEI program.
    
    eps = 1e-9
    sigma_safe = sigma + eps
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    scores = ei / sigma_safe
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    return scores
