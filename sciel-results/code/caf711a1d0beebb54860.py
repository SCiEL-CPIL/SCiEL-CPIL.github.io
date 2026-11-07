"""
Acquisition function using Monte Carlo Knowledge Gradient (KG) approximation.
This strategy estimates the expected improvement of the global best by simulating
the posterior distribution of the objective function at the candidate points.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global state for deadline guard
_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: 330s total. Return safe EI if time is running out.
    if time.time() - _start_time > 325.0:
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, -1e9)

    n = len(mu)
    
    # KG Estimation via Monte Carlo
    # We approximate E[ max(0, f_best - min(f(x))) ]
    # By sampling f(x) ~ N(mu(x), sigma(x)^2) for each x.
    # To be efficient, we sample a fixed number of trajectories (S) 
    # and for each trajectory, compute the min over all x.
    # However, doing S=1000 samples * 2000 points is 2M ops, which is fine.
    # But we need to be careful about the variance of the estimate.
    
    # A more stable approach for KG in BO often uses a lower number of samples 
    # if computational budget is tight, but 330s is generous for 2000 points.
    # Let's use S=200 samples to keep it fast and robust.
    S = 200
    
    # Precompute standard normal random numbers for all samples and points
    # Shape: (S, n)
    z_samples = np.random.standard_normal((S, n))
    
    # Simulate function values: f_sim = mu + sigma * z
    # mu and sigma are (n,), z_samples is (S, n)
    # Broadcasting: (1, n) * (S, n) -> (S, n)
    f_sim = mu[None, :] + sigma[None, :] * z_samples
    
    # For each sample trajectory s, the new best value is min(f_sim[s])
    # The improvement is f_best - min(f_sim[s]) if positive, else 0.
    # We want the expected value of this improvement for each x?
    # Wait, standard KG evaluates the expected improvement of the *best* after evaluating x*.
    # The evaluator picks the x with the highest KG score.
    # The KG score for x* is E[ max(0, f_best - min(f(x*))) ]? No.
    
    # Correct KG definition:
    # KG(x) = E[ max(0, f_best - min_{x' in X} f(x')) ] where X includes x.
    # Actually, the standard definition is:
    # KG(x) = E[ max(0, f_best - min(f(x), min_{x' in X_obs} f(x'))) ]
    # Since X_obs is fixed (the 5 initial points + previous evals), min_{x' in X_obs} is f_best.
    # So KG(x) = E[ max(0, f_best - min(f(x), f_best)) ]
    # This simplifies to E[ max(0, f_best - f(x)) ] which is just Expected Improvement (EI)!
    
    # Wait, is that right?
    # If we evaluate x, the new best is min(f_best, f(x)).
    # The improvement is f_best - min(f_best, f(x)) = max(0, f_best - f(x)).
    # The expectation of this is exactly EI.
    
    # So why is KG considered different/better?
    # Ah, the "Knowledge Gradient" in the context of *global* search often refers to
    # the expected improvement of the *global minimum* of the function, not just the
    # improvement of the current best.
    # However, with only point estimates and a GP, the "global minimum" is unknown.
    
    # Let's re-read the standard literature.
    # KG(x) = E[ max(0, f_best - min(f(x))) ] is indeed EI.
    # BUT, there is a variant called "Expected Improvement of the Global Minimum" or
    # "Global EI".
    
    # Another interpretation:
    # The "Search" directive might imply we should use a method that considers the
    # *uncertainty* of the minimum more broadly.
    
    # Let's look at the "Inspiration 2" which used KG approximation and scored 1.0.
    # It says "Knowledge Gradient (KG) approximation using a large number of Monte Carlo samples".
    # If my derivation is correct that pointwise KG == EI, then Inspiration 2 might have
    # implemented something slightly different, or my derivation is missing a nuance.
    
    # Nuance: In some BO frameworks, the "current best" f_best is not just the min of observed values,
    # but the min of the *posterior mean* or something else? No, the task says f_best is lowest observed.
    
    # Let's consider **Expected Improvement of the Global Minimum (EIGM)** or **Probability of Improvement (PI)** variants.
    
    # Actually, a common "better than EI" strategy is **UCB** or **Thompson Sampling**.
    # Thompson Sampling: Sample a function from the GP posterior, pick the argmin of that sample.
    # This is equivalent to maximizing the probability of being the global minimizer (under some assumptions).
    
    # Let's implement **Thompson Sampling** via Monte Carlo.
    # 1. Sample S functions from the GP posterior: f_s(x) ~ N(mu(x), sigma(x)^2)
    # 2. For each sample s, find x_s = argmin f_s(x)
    # 3. The score for each x is the fraction of samples where x was the minimizer (or similar).
    #    Actually, to return a score for ALL 2000 points, we can't just return a single index.
    #    We can return the *expected value* of the function under the sampled distributions?
    #    Or we can return the **Posterior Mean** of the minimum?
    
    # Let's try **Expected Minimum (EM)** or **Softmin** of the posterior.
    # Score(x) = -E[min(f)]? No, that's a constant.
    
    # Let's go back to **UCB**.
    # The current program uses a portfolio of EI/UCB.
    # What if we use **Thompson Sampling** logic to generate scores?
    # Score(x) = -E[f(x) | f(x) is the minimizer of a sample]?
    
    # A simpler, robust alternative to EI that is distinct:
    # **Probability of Improvement (PI)** with a small xi?
    # PI(x) = P(f(x) < f_best - xi) = Phi((f_best - xi - mu) / sigma)
    # This is purely probabilistic.
    
    # Let's try **Thompson Sampling** based scores:
    # For each point x, we want a score that reflects how likely it is to be the new best.
    # A common proxy is the **Expected Improvement** but we already have that.
    
    # Let's look at the "Search" directive again.
    # "Generate many varied candidate answers and keep the best one found within the time budget."
    # This implies we can run *multiple* acquisition evaluations and pick one?
    # No, we return ONE vector.
    
    # Let's try **Ensemble of Acquisitions** but with a different selection criterion.
    # Or, let's try **Expected Hypervolume Improvement (EHI)**? Too complex.
    
    # Let's try **Knowledge Gradient** properly.
    # If I define the "value" of evaluating x as the expected reduction in the *regret* of the best estimate,
    # and if we assume the "best estimate" is the posterior mean of the global min...
    
    # Actually, the most distinct and powerful method here is **Thompson Sampling** implemented as:
    # Score(x) = - (1/S) * sum_{s=1}^S [ f_s(x) - min_{x'} f_s(x') ]
    # This rewards points that are close to the minimum in the sampled functions.
    # This is equivalent to the **Expected Improvement of the Global Minimum** (approx).
    
    # Let's implement this.
    # 1. Generate S samples of the function: F (S, n)
    # 2. For each sample s, compute min_val_s = min(F[s, :])
    # 3. Compute improvement_s = min_val_s - F[s, :]  (Note: F is the function value, lower is better.
    #    So if F[s, x] is close to min_val_s, improvement is small?
    #    We want high score for good points.
    #    Score should be high if F[s, x] is LOW.
    #    So Score_s(x) = min_val_s - F[s, x] is negative or zero.
    #    We want to MAXIMIZE the score.
    #    So we want F[s, x] to be as small as possible.
    #    Let's define Score(x) = - (1/S) * sum_{s=1}^S F[s, x] ? That's just -mu.
    
    # Let's use the **Probability of being the global minimizer** in the samples.
    # For each sample s, let x_s = argmin F[s, :].
    # Count how many times each x was the argmin.
    # Score(x) = count(x) / S.
    # This is a discrete score. Many points will have 0.
    # To make it continuous and smooth for the evaluator (which picks max), we can use a **Softmin** kernel.
    # Score(x) = - log( sum_{x'} exp(-beta * F_s(x')) )? No.
    
    # Let's use the **Expected Value of the Sampled Minimum** centered at x?
    # No, let's stick to a known strong performer: **UCB with adaptive beta** is often better than EI for stiff functions.
    # But the current program already tries UCB.
    
    # What is DIFFERENT?
    # **Thompson Sampling** is fundamentally different from EI/UCB.
    # Let's implement Thompson Sampling scores as:
    # Score(x) = E[ -f(x) | x is sampled ]?
    
    # A practical implementation of TS scores for a vector:
    # 1. Sample S functions F (S, n).
    # 2. For each sample s, find the index of the minimum: idx_s = argmin F[s, :]
    # 3. We can't just return a count because it's sparse.
    # 4. Instead, we can return the **Expected Improvement** of the sample minimum?
    
    # Let's try a different angle: **Entropy-based acquisition**?
    # No.
    
    # Let's try **Expected Improvement of the Global Minimum (EIGM)** approximated by:
    # Score(x) = E[ max(0, f_best - f(x)) ] - lambda * E[ max(0, f_best - min(f)) ]?
    
    # Actually, the most distinct and computationally feasible strategy that often outperforms EI
    # in 1D stiff functions is **UCB with a beta that decays with the number of evaluations**.
    # But we don't have the step count.
    
    # Let's use **f_best** as a proxy for progress.
    # If f_best is close to -39.1662, beta should be small.
    # If f_best is far, beta should be large.
    
    # The current program uses a portfolio.
    # Let's implement a **Single, Adaptive UCB** with a specific tuning.
    # Or, let's implement **Thompson Sampling** by returning the **Negative Expected Value of the Function**
    # weighted by the probability of it being the minimum?
    
    # Let's go with **Thompson Sampling** via **Soft-Argmin**.
    # For each sample s, compute the probability that x is the minimum using a softmax over the negative values.
    # P(x is min | sample s) = exp(-F[s, x] / T) / sum_{x'} exp(-F[s, x'] / T)
    # Score(x) = (1/S) sum_s P(x is min | sample s)
    # This gives a smooth, dense score vector.
    # Higher score = more likely to be the global minimizer in the posterior samples.
    
    # This is a very different design family from EI/UCB (which are myopic/improvement-based).
    # This is a **Global Optimization** based score.
    
    # Parameters:
    # S = 50 samples (fast)
    # T (temperature) = 1.0 (tunable, maybe scale with sigma?)
    
    # Let's implement this.

    S = 100  # Number of Thompson samples
    # Generate samples
    z_samples = np.random.standard_normal((S, n))
    F = mu[None, :] + sigma[None, :] * z_samples
    
    # To compute softmax probabilities efficiently:
    # Subtract max for numerical stability
    # We want P(x is min) ~ exp(-F/T).
    # Let G = -F / T. Then we want softmax(G).
    # G has shape (S, n).
    
    # Temperature T:
    # If T is small, it's like hard argmin (sparse).
    # If T is large, it's uniform.
    # We want T to be related to the scale of F.
    # Let's use T = 1.0 * (max(F) - min(F)) / n? Or just a fixed small value?
    # Since F is in objective units (around -39 to 10), T=1.0 might be too small (too sparse).
    # Let's use T = 0.5 * np.std(F, axis=1, keepdims=True) ?
    # Or simpler: T = 1.0.
    
    # Actually, a better temperature is the standard deviation of the function values.
    # But let's start with a fixed T relative to the range.
    # Range of F is roughly 50. T=1.0 makes exp(-50) ~ 0. So it will be very sparse.
    # We want a smooth distribution.
    # Let's use T = 5.0.
    
    T = 5.0
    G = -F / T
    
    # Softmax over axis 1 (for each sample s)
    # max_G = np.max(G, axis=1, keepdims=True)
    # exp_G = np.exp(G - max_G)
    # sum_exp = np.sum(exp_G, axis=1, keepdims=True)
    # probs = exp_G / sum_exp  # Shape (S, n)
    
    # Score(x) = mean of probs over samples
    # scores = np.mean(probs, axis=0)
    
    # This score is between 0 and 1. Higher is better.
    # Is this a good acquisition?
    # It favors points that are often the minimum in the posterior samples.
    # This encourages exploration of regions that *could* contain the global min.
    
    # Let's add a safety check: if scores are all zero or non-finite, fallback to EI.
    
    # Compute scores
    max_G = np.max(G, axis=1, keepdims=True)
    exp_G = np.exp(G - max_G)
    sum_exp = np.sum(exp_G, axis=1, keepdims=True)
    
    # Avoid division by zero
    sum_exp = np.where(sum_exp == 0, 1e-9, sum_exp)
    
    probs = exp_G / sum_exp
    scores = np.mean(probs, axis=0)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # If the scores are all identical or very low, it might be a degenerate case.
    # But usually, this should work.
    
    # One issue: This score doesn't directly account for f_best.
    # If f_best is very good, we should exploit more.
    # The TS score naturally handles this: if the posterior is tight around a low value,
    # the samples will cluster there, and the score will be high there.
    
    # Let's return these scores.
    
    return scores

# EVOLVE-BLOCK-END
