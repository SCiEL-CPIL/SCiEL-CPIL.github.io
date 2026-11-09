import numpy as np
from scipy.stats import norm
import time

# Global state for tracking selected points and time
_selected_points = []
_start_time = None
_last_f_best = None
_step_counter = 0

def acquisition(mu, sigma, f_best):
    global _selected_points, _start_time, _last_f_best, _step_counter
    
    if _start_time is None:
        _start_time = time.time()
        
    # Detect new seed/run
    # If f_best is significantly higher than the last observed min, assume new run
    # Function values are roughly in [-3, 0]
    if _last_f_best is not None and f_best > _last_f_best + 0.5:
        _step_counter = 0
        _selected_points = []
    else:
        _step_counter += 1
        
    _last_f_best = f_best
    
    # Total expected optimization steps is 25 (30 total evals - 5 initial)
    # We need to handle the first 5 initial points which are uniform random.
    # The acquisition function is called after the GP is updated.
    # Step 0 in our counter corresponds to the 6th evaluation (1st acquisition call).
    step = min(_step_counter, 24)
    total_steps = 25.0
    
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 320:
        # Return a simple heuristic to avoid timeout
        return -mu + sigma

    # Calculate Expected Improvement (EI)
    sigma_safe = np.maximum(sigma, 1e-9)
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    # Standard EI for minimization
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Diversity Penalty Strategy
    # We want to encourage exploring new areas.
    # Since we don't have the coordinates of the 2000 candidates passed in, 
    # we cannot directly compute distances to _selected_points in coordinate space.
    # HOWEVER, the task says "No coordinates are passed". 
    # Wait, if coordinates are not passed, how can we enforce spatial coverage?
    
    # Re-reading the contract: "mu and sigma have shape (2000,). Return 2000 finite scores... No coordinates are passed."
    # This is a crucial constraint. We CANNOT compute distances between candidates and previous points 
    # because we don't know where the candidates are in [0,1]^2.
    
    # Therefore, a "diversity-aware set of candidates" based on coordinates is IMPOSSIBLE 
    # unless we infer coordinates from the GP state, which is not provided.
    
    # Alternative Interpretation of "Diversity":
    # If we can't use coordinates, we must use the GP posterior itself to define "diversity".
    # High sigma regions are "diverse" from the perspective of information gain.
    # The current program already uses EI which balances this.
    
    # Is there a different structural change?
    # The prompt suggests: "greedy batch selection strategy that constructs a diversity-aware set... using a greedy maximization of a hypervolume or spacing-based objective over the top-k scored points".
    # This usually implies we have access to the candidate coordinates.
    # If the contract explicitly says "No coordinates are passed", then this specific strategy (spatial coverage via coordinates) is invalid for this interface.
    
    # Let's look at the inspirations again.
    # Inspiration 3: "deterministic, adaptive Expected Improvement... step-aware, non-monotonic exploration parameter".
    # Inspiration 2: "Search-based acquisition".
    
    # If I cannot use coordinates, I must change the *acquisition function form* itself.
    # The current program blends LCB and EI.
    # A structurally different approach: **U-PI (Upper Probability of Improvement)** or **Expected Simple Regret** with a specific tuning, or **Knowledge Gradient**.
    
    # Let's try **Knowledge Gradient (KG)** approximation or **Probability of Improvement (PI)** with aggressive tuning.
    # Actually, a very different structural change is to use **Thompson Sampling** (deterministic version? No, TS is stochastic).
    # But the prompt asks for a "greedy batch selection... spacing-based".
    
    # If I *must* follow the directive's suggested strategy despite the "No coordinates" constraint, 
    # perhaps I can *estimate* coordinates? No, that's unreliable.
    
    # Let's reconsider the "No coordinates" constraint. 
    # In many BO tasks, the acquisition function receives X (coordinates). 
    # Here it says "No coordinates are passed". 
    # This implies the evaluator generates candidates, computes mu/sigma, and passes ONLY mu/sigma. 
    # The evaluator then picks the index with the max score and evaluates THAT candidate.
    
    # If I cannot see the coordinates, I cannot enforce spatial diversity directly.
    # However, I can enforce diversity in the *posterior uncertainty space*.
    # But that's what EI/LCB do.
    
    # Let's look at the observed changes.
    # The current program (p12) scores 2.25.
    # A child (p28) that removed comments scored 0.02 (bad).
    # A child (p24) that changed module/acquisition scored 0.04 (bad).
    
    # Maybe the "No coordinates" constraint is the key to a *different* strategy:
    # **Bayesian Optimization with a Fixed Grid?** No, candidates are drawn fresh.
    
    # Let's try **Gaussian Process Upper Confidence Bound (GP-UCB)** with a specific beta schedule that is structurally different from the linear blend.
    # Or **Expected Regret** minimization.
    
    # Actually, there is a strategy that doesn't need coordinates: **Deterministic Annealing**.
    # But that's still tuning.
    
    # Let's look at the "Spacing-based" hint again. 
    # If I can't use coordinates, maybe I can use the *indices*? No, indices are random.
    
    # Is it possible the contract description "No coordinates are passed" is a simplification and I *should* assume I can't do spatial diversity?
    # If so, what is a *structurally different* design family?
    # 1. Myopic EI (Current)
    # 2. Non-myopic: Knowledge Gradient (KG). KG is the gold standard for non-myopic.
    #    Approximate KG: 
    #    KG(x) = E[ max(0, f_best - f(x)) ] - E[ max(0, f_best - min(f(x), f_best)) ] ... complicated.
    #    A simpler non-myopic proxy is **Expected Improvement with a look-ahead term**.
    
    # Let's try **Probability of Improvement (PI)** with a very high exploration weight?
    # PI = Phi(z). It is more aggressive than EI in exploration (EI is 0 when z is very negative, PI is small but non-zero? No, PI is 0.5 when z=0).
    # Actually, PI is often less efficient than EI.
    
    # Let's try **Negative Mean plus Beta * Sigma (LCB)** but with a **Non-linear Beta Schedule**.
    # The current program uses a linear blend of LCB and EI.
    # A structurally different approach: **Pure LCB with a specific Beta Schedule** (e.g., logarithmic or exponential decay) instead of a blend.
    # Or **Pure EI** (which the current program is blending away from at step 0).
    
    # Let's try **Pure EI** but with a **Step-Dependent Xi (Improvement Threshold)**.
    # EI_xi = (f_best - xi - mu) * Phi(z_xi) + sigma * phi(z_xi)
    # Where z_xi = (f_best - xi - mu) / sigma.
    # By increasing xi over time, we force the algorithm to only pursue points that are likely to improve by a significant margin, which can prevent "wandering" in flat regions.
    # Or decreasing xi to encourage exploration?
    # Standard EI uses xi=0.
    # If we use a *negative* xi (i.e., f_best + |xi|), we are asking for points that are better than f_best by |xi|. This is very exploitative.
    # If we use a *positive* xi (i.e., f_best - xi), we are asking for points that are better than f_best - xi. This allows for smaller improvements, effectively increasing exploration.
    
    # Let's try **EI with a decaying positive xi** (increasing exploration tolerance early on).
    # xi_t = xi_max * (1 - t/T)
    # Early: xi is large -> we accept points that are just slightly worse than f_best? No, f_best - xi is lower. 
    # Wait, for minimization:
    # Improvement = f_best - f(x).
    # We want to maximize E[max(0, f_best - f(x) - xi)].
    # If xi > 0, we require an improvement of at least xi. This is MORE exploitative.
    # If xi < 0, we accept improvements of at least xi (which is negative, so even if f(x) > f_best, as long as it's not too much worse, we get a score? No, max(0, ...) cuts it off).
    # Actually, standard EI is xi=0.
    
    # Let's try **U-PI (Upper Probability of Improvement)**:
    # U-PI(x) = Phi( (f_best - mu) / sigma ) * (1 + alpha * sigma)
    # This is not standard.
    
    # Let's go back to the directive: "greedy batch selection... spacing-based".
    # Since I cannot do this without coordinates, I will interpret "diversity" as **diversity in the posterior mean and variance**.
    # However, a truly different structural change is to use **Deterministic Annealing** on the temperature of the posterior.
    
    # Let's try a **Hybrid of EI and Inverse Variance**.
    # Score = EI - lambda * (1/sigma)
    # This penalizes high confidence (low sigma) points, forcing exploration of low confidence (high sigma) points even if their mean is not great.
    # This is structurally different from LCB (which adds sigma) or EI (which multiplies sigma).
    # It explicitly favors "unknown" regions.
    
    # Let's implement **EI + Exploration Bonus based on 1/sigma**.
    # Wait, 1/sigma blows up when sigma is small. We want to reward HIGH sigma.
    # So: Score = EI + lambda * sigma^2 ?
    # That's just LCB.
    
    # How about **Score = EI * (1 + lambda * sigma)**?
    # This amplifies EI in high uncertainty regions.
    
    # Let's try **Score = EI + lambda * sigma * (1 - sigma)**? No.
    
    # Let's try **Knowledge Gradient Approximation**:
    # KG is hard to compute fast for 2000 points.
    
    # Let's try **Expected Simple Regret (ESR)**?
    # ESR is similar to EI.
    
    # Let's look at the "Spacing" hint again. 
    # Is it possible to use the **Index** of the candidate as a proxy for diversity? No.
    
    # What if I use **Random Forest** or another surrogate? No, GP is fixed.
    
    # Let's try **Pure LCB with a Hyperbolic Beta Schedule**.
    # Beta(t) = beta_0 / (1 + t * decay)
    # This is a different decay profile than linear.
    
    # Let's try **Pure EI** (no blend). The current program blends LCB and EI.
    # At step 0, it's 100% LCB. LCB with beta=3.0 is VERY aggressive.
    # Maybe pure EI is better?
    
    # Let's try **EI with a Constant Xi = 0.1**.
    # This makes it slightly more exploitative than standard EI.
    
    # Let's try **Thompson Sampling** (Stochastic).
    # The prompt says "deterministic" in inspiration 3, but TS is a valid strategy.
    # However, TS requires sampling from the GP posterior.
    # We have mu and sigma. We can sample y ~ N(mu, sigma).
    # But we need to return 2000 scores.
    # We can sample a single y for each candidate?
    # If we sample y_i ~ N(mu_i, sigma_i), and return y_i, the evaluator picks the min y_i.
    # This is valid.
    # TS is known to be more efficient than EI in some high-dimensional or complex landscapes.
    # It naturally balances exploration and exploitation via the sampling.
    
    # Let's implement **Thompson Sampling**.
    # We need to be careful with the seed. The evaluator draws 2000 fresh candidates.
    # We should use a fixed seed for the TS sampling to be deterministic?
    # Or use a random seed?
    # If we use a random seed, the result is stochastic. The evaluator runs 3 seeds.
    # Stochasticity might average out.
    # However, the task says "fixed seeded random RBF mixture". The GP is deterministic given the data.
    # If we add stochasticity in the acquisition, it might help escape local optima.
    
    # Let's try **Deterministic Thompson Sampling** using a fixed pseudo-random offset per step?
    # Or just **Pure EI** first, as it's a robust baseline that is different from the LCB-blend.
    
    # Actually, the current program is a BLEND.
    # A structurally different approach is **Pure EI**.
    # Another is **Pure LCB**.
    # Another is **PI**.
    # Another is **KG**.
    
    # Let's try **Pure EI** with a **Step-Dependent Sigma Scaling**.
    # sigma_eff = sigma * (1 + decay * (1 - step/total))
    # This makes early steps more exploratory by artificially inflating sigma in the EI calculation.
    # This is structurally different from blending LCB.
    
    # Let's implement **Scaled EI**.
    # alpha = 2.0 * (1 - step/total_steps) + 0.5
    # sigma_scaled = sigma * alpha
    # z = (f_best - mu) / sigma_scaled
    # ei_scaled = (f_best - mu) * Phi(z) + sigma_scaled * phi(z)
    
    # This effectively changes the "optimism" of the EI.
    
    # Let's also add a **Deadline Guard**.
    
    # One more idea: **Maximize the Probability of being the best** (PI) but with a twist.
    
    # I will go with **Scaled EI** (also known as EI with temperature).
    # It is a single, coherent acquisition function, not a blend of two different families.
    
    # Implementation:
    # 1. Calculate step.
    # 2. Calculate alpha (exploration factor).
    # 3. Calculate scaled sigma.
    # 4. Calculate EI with scaled sigma.
    # 5. Return scores.
    
    # This is materially different from the "Linear Blend of LCB and EI" because it modifies the EI kernel itself rather than adding a separate LCB term.

    # Exploration factor: starts high, decays to 1.0 (standard EI)
    # alpha = 1.0 + 2.0 * (1.0 - step / total_steps)
    
    alpha = 1.0 + 2.0 * (1.0 - step / total_steps)
    
    sigma_scaled = sigma_safe * alpha
    
    # Recalculate z with scaled sigma
    z_scaled = improvement / sigma_scaled
    
    # Calculate EI with scaled sigma
    # Note: The improvement term (f_best - mu) is unchanged.
    # The PDF term uses sigma_scaled.
    # The CDF term uses z_scaled.
    
    ei_scaled = improvement * norm.cdf(z_scaled) + sigma_scaled * norm.pdf(z_scaled)
    
    return ei_scaled
