import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Strategy:
    1. Sample a value from the posterior predictive distribution N(mu, sigma^2).
       This is Thompson Sampling: we act as if the sampled value is the true value.
    2. The "score" for a candidate is the negative of the sampled value divided by 
       the evaluation cost. 
       - Minimizing the objective corresponds to maximizing the negative value.
       - Dividing by cost prioritizes cheaper evaluations, effectively treating 
         cost as a resource constraint.
    3. We use a deterministic seed based on the current time or a fixed seed to ensure
       reproducibility within a single run, but the sampling itself provides the 
       stochastic exploration inherent to TS.
       Note: The task says "Each function runs on five seeded instances...". 
       The acquisition function is called repeatedly in the loop. 
       Standard TS requires a new sample each time. 
       Since we don't have access to the global RNG state easily in a stateless function 
       without passing it, and the prompt implies we can use standard libs, 
       we can use np.random.default_rng with a seed derived from the input data 
       (e.g., hash of mu/sigma) or just use a fixed seed for the RNG instance 
       created per call? 
       
       Actually, if we use a fixed seed per call, we get the same sample every time 
       for the same mu/sigma, which effectively makes it deterministic but with 
       a different "noise" structure than EI. 
       
       However, true Thompson Sampling benefits from different samples over time. 
       But the function signature is stateless. 
       
       Let's look at the inspiration: "Adaptive beta...". 
       
       A common trick in stateless TS implementations for BO loops is to use the 
       current iteration count or a hash of the inputs to seed the RNG. 
       But we don't have the iteration count. 
       
       Alternative: Use the "probability of improvement" (PoI) which is the 
       probability that f(x) < f_best. 
       PoI = Phi((f_best - mu) / sigma). 
       This is a deterministic function of mu and sigma. 
       It is structurally different from EI (which is an expectation). 
       PoI is more "exploratory" in the sense that it gives high scores to regions 
       with high uncertainty (high sigma) even if the mean is not great, 
       whereas EI requires the mean to be reasonably close to f_best. 
       
       Let's try a hybrid: 
       Score = PoI * (1 / cost) 
       Or better, let's stick to the Thompson Sampling idea but implement it 
       in a way that is "stochastic" per candidate in a deterministic way? 
       
       Actually, the prompt says: "Replace the deterministic Expected Improvement 
       with a stochastic Thompson Sampling approach that samples from the posterior 
       predictive distribution... and selects the candidate with the lowest sampled 
       objective value divided by the evaluation cost."
       
       If I sample from N(mu, sigma), I get a vector of samples. 
       If I do this with a fixed seed (e.g., 42), the "randomness" is fixed relative 
       to the index. This creates a deterministic landscape that is a noisy version 
       of the mean. 
       
       Let's implement exactly what the directive suggests: 
       1. Generate samples from N(mu, sigma).
       2. Score = -sample / cost.
       
       To make it robust and avoid the "same noise every time" issue if the GP 
       doesn't change much, we can vary the seed based on the sum of mu or something 
       unique to the call context? No, that's fragile. 
       
       Let's look at the "Inspiration 1" which scored 1.41. It used "Adaptive beta 
       with logarithmic schedule". That suggests a dynamic balance parameter. 
       
       The current program uses EI/cost. 
       
       Let's try **Probability of Improvement (PoI)** adjusted for cost. 
       PoI is simpler and often more robust than EI in low-dimensional problems 
       because it doesn't suffer from the "greedy" trap of EI as much when 
       uncertainty is high. 
       
       Formula: 
       z = (f_best - mu) / sigma
       PoI = Phi(z)
       Score = PoI / cost
       
       This is a different mathematical family (CDF vs Expectation). 
       
       However, the directive specifically asks for **Thompson Sampling**. 
       
       Let's implement Thompson Sampling with a "pseudo-random" seed that changes 
       slightly with the state of the problem to simulate fresh samples? 
       Or just use a fixed seed? 
       
       If I use `np.random.default_rng(0)`, the samples are fixed for a given index. 
       This effectively adds a fixed noise pattern to the objective. 
       
       Let's try this:
       1. Create a generator.
       2. Sample `samples = mu + sigma * rng.standard_normal(2000)`.
       3. `scores = -samples / cost`.
       
       This is a valid acquisition function. It will pick the point that *looks* 
       best under a specific realization of the uncertainty. 
       
       To improve on the "fixed seed" issue, we can incorporate `f_best` into the 
       seed or use a hash of the input arrays to create a unique seed for each 
       call, ensuring that the "noise" pattern changes as the GP updates (since 
       mu/sigma change). 
       
       Let's use a seed derived from the data: 
       seed = int(np.sum(mu) * 1e6) % (2**32)
       
       This ensures that as the GP updates (mu changes), the "random" samples 
       change, effectively providing fresh Thompson samples without global state.
       
       Let's refine the cost adjustment. 
       Dividing by cost is linear. 
       Maybe `1 / (1 + cost)`? The cost is 1 + mean(x). So cost is in [1, 2]. 
       Linear division is fine.
       
       Code structure:
       1. Sanitize inputs.
       2. Calculate seed from mu/sigma.
       3. Generate samples.
       4. Calculate scores.
       5. Return scores.
       
       Deadline guard:
       The operation is vectorized and fast (2000 samples). 
       Time should be negligible (< 1ms). 
       No complex loops needed.
    """
    
    # Sanitize inputs
    mu_safe = np.nan_to_num(mu, nan=f_best, posinf=f_best, neginf=-np.inf)
    sigma_safe = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    sigma_safe = np.maximum(sigma_safe, 1e-6)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Generate a seed based on the current state of the GP to simulate 
    # fresh Thompson samples for each call without global state.
    # Using a simple hash of the mean values.
    # We ensure the seed is an integer.
    try:
        # Use the first few values to create a seed
        seed_val = int(np.sum(mu_safe[:10]) * 10000) % (2**32)
        if seed_val < 0:
            seed_val = seed_val + 2**32
    except:
        seed_val = 42
        
    # Create RNG
    rng = np.random.default_rng(seed_val)
    
    # Thompson Sampling: Draw one sample per candidate
    # samples[i] ~ N(mu[i], sigma[i]^2)
    samples = mu_safe + sigma_safe * rng.standard_normal(len(mu_safe))
    
    # For minimization, we want low values. 
    # The score should be high for "good" candidates.
    # Good candidate: Low sampled value, Low cost.
    # Score = -sampled_value / cost
    # Note: If sampled value is very negative (good), -sampled is positive (high score).
    # If sampled value is high (bad), -sampled is negative (low score).
    
    scores = -samples / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Optional: If all scores are -inf or very bad, fallback to something?
    # The task says "Return 2000 finite scores". 
    # If scores are -inf, that's not finite. 
    # Let's replace -inf with a very small number or 0?
    # "anything else invalidates the whole result". 
    # So we must ensure finiteness.
    
    scores = np.nan_to_num(scores, nan=-1e10, posinf=1e10, neginf=-1e10)
    
    return scores
