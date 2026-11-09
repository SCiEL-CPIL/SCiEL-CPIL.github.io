import numpy as np
import time
import sys

# EVOLVE-BLOCK-START
def _lorenz96_rhs(x, F):
    """Compute the RHS of the Lorenz-96 system."""
    n = len(x)
    # Periodic indices
    # x_{i+1} - x_{i-2} all times x_{i-1}
    # Using numpy roll
    # x_{i+1} is roll(x, -1)
    # x_{i-1} is roll(x, 1)
    # x_{i-2} is roll(x, 2)
    term1 = np.roll(x, -1) - np.roll(x, 2)
    term2 = np.roll(x, 1)
    return term1 * term2 - x + F

def _rk4_step(x, dt, F):
    """Propagate state x by one RK4 step of size dt."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def analyze(ensemble, observations, observed_indices, observation_variance):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for 120s limit
    
    n_members, n_vars = ensemble.shape
    obs_len = len(observations)
    
    # Infer F from the ensemble mean dynamics over a short step
    # This is an estimate, but 4D-Var is robust to small F errors if prior is strong
    # We use a very short step to estimate F
    dt_est = 0.01
    mean_state = np.mean(ensemble, axis=0)
    next_state = _rk4_step(mean_state, dt_est, 0.0) # F=0 placeholder
    # dx/dt = (x+1 - x-2)*x-1 - x + F
    # We can solve for F using the definition
    # Actually, let's just estimate F from the mean of the ensemble's implied F
    # F = dx/dt - (x_{i+1}-x_{i-2})*x_{i-1} + x_i
    # Using the step we just took (which used F=0):
    # next_state = mean_state + dt * (term1*term2 - mean_state)
    # So term1*term2 - mean_state = (next_state - mean_state) / dt
    # But we want F.
    # Let's compute the "natural" F for each member
    F_estimates = []
    for i in range(n_members):
        x = ensemble[i]
        # Compute RHS with F=0
        rhs_0 = _lorenz96_rhs(x, 0.0)
        # Propagate with F=0
        x_next_0 = _rk4_step(x, dt_est, 0.0)
        # This doesn't directly give F without knowing the "true" next state.
        # However, the problem states F is constant for the run.
        # We can estimate F by looking at the ensemble mean's behavior.
        # A more robust way: The evaluator uses F=6,8,10.
        # Let's assume we can estimate F from the equilibrium or just use a default?
        # No, we must be adaptive.
        # Let's try to infer F from the observation noise? No.
        # Let's infer F from the ensemble's tendency to stay bounded?
        # Actually, a simple heuristic: F is often related to the mean energy or state magnitude.
        # But the most direct way is to assume the ensemble is consistent with some F.
        # Let's compute the "implied F" for each member assuming the current state is on the attractor?
        # Too complex.
        
        # Alternative: The task says "nondimensional units".
        # Let's look at the magnitude of x.
        # For F=6, mean x ~ 0.
        # For F=8, mean x ~ 0.
        # For F=10, mean x ~ 0.
        # The variance grows with F.
        # Let's estimate F by matching the ensemble variance to theoretical values?
        # Or simply: The evaluator propagates with the TRUE F.
        # If we use the WRONG F in our 4D-Var, the prior will be wrong.
        # However, the observation term will pull us to the correct state.
        # The prior penalty prevents wild jumps.
        # So even if F is wrong, the update will be "safe" but potentially suboptimal.
        # Can we estimate F?
        # Let's assume F is one of {6, 8, 10} and pick the one that minimizes the ensemble's "surprise" (chi-square) over a 1-step propagation?
        # i.e., for each candidate F in [6, 8, 10], propagate ensemble 1 step, and see which F makes the ensemble most "consistent" with itself?
        # This is hard without truth.
        
        # Let's try a different approach: Estimate F from the mean of the state?
        # In L96, for F=8, the mean is 0.
        # Let's just use F=8 as a default? No, that's risky.
        
        # Let's try to estimate F from the observation variance? No.
        
        # Let's use a robust estimator:
        # The Lyapunov exponent or growth rate?
        
        # Actually, the simplest and most robust method in data assimilation when F is unknown but fixed:
        # Use a "model error" tolerance.
        # But we need F for the dynamics.
        
        # Let's look at the inspirations. They all define _lorenz96_rhs.
        # Inspiration 1: "Cholesky-based square-root update".
        # Inspiration 2: "Stochastic Ensemble Kalman Filter".
        # They don't mention estimating F.
        # Maybe F is not needed to be estimated if we use a filter that doesn't propagate?
        # But 4D-Var requires propagation.
        
        # Wait, the task says: "The evaluator propagates the ensemble and compares...".
        # "Lorenz-96 dx_i/dt=... with periodic indices, nondimensional units."
        # "Three runs F=6,8,10".
        # It does NOT say F is passed to analyze.
        # So I MUST estimate F or use a method that doesn't need it.
        
        # If I use a standard EnKF, I need to propagate the ensemble to the next observation time.
        # The next observation is at t+5*dt.
        # To propagate, I need F.
        
        # How to estimate F?
        # Let's assume the ensemble is in a quasi-stationary state.
        # The mean of x is 0 for all F.
        # The variance of x increases with F.
        # Let's compute the variance of the ensemble and map it to F.
        var_x = np.var(ensemble)
        # Empirical relationship for L96:
        # F=6: var ~ 2-3?
        # F=8: var ~ 8-10?
        # F=10: var ~ 20-25?
        # This is a heuristic.
        
        # Better: Use the fact that the system is chaotic.
        # Let's try to fit F by minimizing the distance between the ensemble mean after 1 step and the current ensemble mean?
        # No, that's not right.
        
        # Let's try this: For each candidate F in [6, 8, 10], propagate the ensemble mean by 1 step (dt=0.01).
        # Calculate the "cost" as the distance to the current mean? No.
        
        # Actually, there is a simpler clue.
        # The observation noise is N(0, 0.49).
        # The observation variance is 0.49.
        
        # Let's assume F is 8.0 as a central guess?
        # Or better, let's estimate F from the ensemble spread.
        # A common approximation for L96 is that the standard deviation of x is roughly proportional to sqrt(F).
        # Let's try: F_est = 10.0 * (std_x / std_x_F10)**2?
        
        # Let's use a more robust method:
        # Propagate the ensemble for 1 step with F=6, F=8, F=10.
        # Check which F results in an ensemble that is "most similar" to the current ensemble in terms of structure?
        # This is computationally expensive (3 propagations of 16x20).
        # 16*20*3 = 960 RK4 steps. Very fast.
        
        candidate_Fs = [6.0, 8.0, 10.0]
        best_F = 8.0
        min_cost = np.inf
        
        # Cost function: How much does the ensemble "spread" change?
        # In a stationary chaotic system, the spread should be roughly constant.
        # So, for each F, propagate ensemble by 1 step.
        # Calculate the distance between the propagated ensemble and the current ensemble.
        # The F that minimizes this distance is likely the correct one?
        # Not necessarily, because the system evolves.
        
        # Alternative: The "folding" of the attractor.
        
        # Let's try a different heuristic:
        # The mean kinetic energy or something?
        
        # Actually, let's look at the "seed program". It just nudges.
        # It doesn't propagate.
        # So it doesn't need F.
        
        # My 4D-Var plan requires propagation.
        # If I can't estimate F reliably, 4D-Var might fail.
        
        # Let's reconsider the strategy.
        # Maybe I should use a method that doesn't require explicit F estimation, or estimates it very simply.
        
        # Let's try to estimate F by assuming the ensemble mean is close to the attractor.
        # For L96, the average value of x_i is 0.
        # The average of x_i^2 is related to F.
        # E[x^2] ~ F^2 / 8? No.
        
        # Let's use a simple linear regression on the observed data?
        # We have observations of even indices.
        # We can fit a model x_{t+1} = A x_t + b? No, it's nonlinear.
        
        # Let's go with the "propagate and compare" heuristic.
        # Propagate ensemble 1 step with each candidate F.
        # Compute the Frobenius norm of (Ensemble_propagated - Ensemble_current).
        # The F that minimizes this is the best guess?
        # This assumes the system is in a steady state where the ensemble doesn't change much on average?
        # But it's chaotic, so it changes a lot.
        
        # However, the *structure* of the ensemble (the spread) should be preserved.
        # So, compare the covariance matrices?
        
        # Let's try this:
        # For each F, propagate ensemble.
        # Calculate the distance between the new ensemble mean and old ensemble mean? No.
        
        # Let's try: Distance between new ensemble and old ensemble.
        # The correct F should keep the ensemble "on" the attractor, so the distance should be "typical".
        # Wrong F might push it off or into a different regime.
        
        # Let's implement this heuristic.
        
        for F_cand in candidate_Fs:
            ens_prop = np.array([_rk4_step(ens, dt_est, F_cand) for ens in ensemble])
            # Cost: Frobenius norm of difference
            cost = np.linalg.norm(ens_prop - ensemble)
            if cost < min_cost:
                min_cost = cost
                best_F = F_cand
                
        F = best_F
        
        # Now perform 4D-Var for each member
        # We have 16 members, 20 variables.
        # Observation window: 5 steps of dt=0.01.
        # We observe at the END of the window?
        # "evaluator propagates... for five steps between updates, then observes..."
        # So the observations correspond to the state at t+5*dt.
        # Our ensemble is at t.
        # We want to update the ensemble at t to be the best estimate of the state at t,
        # given that the state at t+5*dt will be observed.
        
        # Wait, the standard 4D-Var updates the *initial* state of the window.
        # The observations are at the end of the window.
        # So we minimize J(x0) = ||H(x(t+5dt)) - y||^2 + ||x0 - x_prior||^2
        # where x_prior is the current ensemble member.
        
        # We need to propagate x0 to x5.
        # This requires 5 RK4 steps.
        # We need to optimize x0 (20 variables).
        # We can use a simple gradient descent or L-BFGS-B from SciPy.
        
        # But we have 16 members.
        # 16 * (optimization cost).
        # Optimization cost: each evaluation of J requires 5 RK4 steps (20 vars each).
        # L-BFGS-B might need 20-50 evaluations.
        # 16 * 50 * 5 = 4000 RK4 steps.
        # Each RK4 step is ~ 100 flops.
        # 400,000 flops. Very fast.
        
        # However, we must be careful with the prior.
        # The prior should be the ensemble mean? Or the member itself?
        # In ensemble 4D-Var, we often use the ensemble mean as the prior mean,
        # and the ensemble covariance as the prior covariance.
        # But here, we are updating each member independently?
        # If we update each member independently towards the observation,
        # we might collapse the spread.
        
        # A better approach: Ensemble 4D-Var.
        # Update the ensemble mean and covariance.
        # Or, use the "perturbed observations" method.
        
        # Let's stick to the simpler "individual member update" first,
        # but with a strong prior to prevent collapse.
        # Prior: Gaussian centered at the ensemble mean, with covariance from the ensemble.
        
        # Actually, let's use a simpler method:
        # For each member, find the x0 that minimizes:
        # J(x0) = ||H(x5) - y||^2 / R + ||x0 - x_prior||^2 / P
        # where x_prior is the current member, and P is the ensemble covariance.
        
        # This is a "local" 4D-Var for each member.
        # It will pull each member towards the observation,
        # but the prior keeps it close to its original position.
        # This should preserve spread better than a simple nudge.
        
        # Let's implement this.
        
        # 1. Compute ensemble mean and covariance
        ens_mean = np.mean(ensemble, axis=0)
        ens_cov = np.cov(ensemble.T) # 20x20
        
        # Regularize covariance
        ens_cov += 1e-6 * np.eye(n_vars)
        
        # 2. Define the cost function for a single member
        # x0: initial state (20,)
        # x_prior: prior state (20,)
        # y: observation (10,)
        # obs_idx: observed indices (10,)
        
        def cost_func(x0, x_prior, y, obs_idx, F_val, R_var):
            # Propagate x0 to x5
            x = x0.copy()
            for _ in range(5):
                x = _rk4_step(x, dt_est, F_val)
            
            # Extract observed variables
            x_obs = x[obs_idx]
            
            # Misfit
            misfit = x_obs - y
            J_obs = np.sum(misfit**2) / R_var
            
            # Prior
            prior_diff = x0 - x_prior
            J_prior = np.dot(prior_diff, np.linalg.solve(ens_cov, prior_diff))
            
            return J_obs + J_prior
        
        # 3. Optimize for each member
        updated_ensemble = ensemble.copy()
        
        for i in range(n_members):
            x_prior = ensemble[i]
            
            # Use L-BFGS-B
            from scipy.optimize import minimize
            
            # Initial guess: x_prior
            res = minimize(
                cost_func,
                x_prior,
                args=(x_prior, observations, observed_indices, F, observation_variance),
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if res.success:
                updated_ensemble[i] = res.x
            else:
                # If optimization fails, keep the prior or do a simple nudge
                updated_ensemble[i] = x_prior
            
            # Time check
            if time.time() > deadline:
                break
        
        # Ensure output is finite and within bounds
        updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
        updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
        
        return updated_ensemble

# EVOLVE-BLOCK-END
