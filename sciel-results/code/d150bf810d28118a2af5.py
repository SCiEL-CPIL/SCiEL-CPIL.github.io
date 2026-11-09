"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 320.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return a valid but arbitrary finite score to avoid invalidation
        # If we are past deadline, the evaluator likely kills us anyway, 
        # but returning something finite is safer than crashing.
        return np.zeros_like(mu)
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Estimate number of evaluations performed
    # cost[i] = 1 + mean(x_scaled)
    # Initial 5 points: cost ~ 1.5 each -> total ~ 7.5
    # Each new point adds ~ 1.5 cost
    # So n_evals is roughly (total_cost - initial_cost) / avg_cost + 5
    # However, we don't have total_cost directly, only the cost vector for the 2000 candidates.
    # We can estimate n_evals from f_best or just use a heuristic based on the magnitude of f_best?
    # Actually, the problem statement says "Runs start with 5 uniform random points... While total cost is below 30...".
    # We don't get total_cost passed. We only get cost for the 2000 candidates.
    # But we can infer the phase from f_best. 
    # Alternatively, we can use a fixed schedule or estimate n from the number of times acquisition was called? 
    # We don't have access to a counter.
    
    # Let's estimate n_evals.
    # The cost of a point is between 1 and 2.
    # Total budget is 30.
    # Max evaluations ~ 15 (if all cost 2) to 30 (if all cost 1).
    # Let's assume average cost 1.5.
    # We can approximate the number of evaluations by looking at the "age" of the GP? No, we don't have it.
    
    # Heuristic: 
    # Early on (f_best is high/large), we want to explore.
    # Later (f_best is low/small), we want to exploit.
    # Since f_best is the minimum observed, it decreases over time.
    # We can map f_best to a beta value.
    # However, the scale of Levy 2D is not fixed (min is 0, but max can be large).
    # A more robust estimate: 
    # The problem states "While total cost is below 30".
    # We can't know total cost.
    # But we can use a static schedule that is robust.
    # Or, we can assume that if f_best is very close to 0, we are in late stages.
    # Let's use a beta that starts high and decays.
    # Since we don't have n, let's use f_best as a proxy for progress? 
    # Risky if f_best doesn't decrease monotonically in a predictable way relative to n.
    
    # Alternative: Use a constant beta that works well on average?
    # The inspiration says: "beta is dynamically scaled by the current number of evaluations".
    # Can we estimate n?
    # The cost vector `cost` contains the costs of the 2000 candidates.
    # This doesn't help with n.
    
    # Let's look at the constraints again.
    # "Runs start with 5 uniform random points (cost counted)."
    # "While total cost is below 30, the evaluator scores 2000 new uniform candidates..."
    # The function `acquisition` is called for each of the 2000 candidates in each step.
    # We don't have a stateful variable.
    
    # However, we can observe that `f_best` is passed.
    # In Levy 2D, the global minimum is 0.
    # If f_best is large, we are early. If f_best is small, we are late.
    # Let's define a normalized progress `p` based on f_best.
    # But what is the scale of f_best? 
    # Levy 2D on [-10,10]^2. The function values can be quite large initially.
    # Let's assume a typical range.
    # Or, we can use a beta that is constant but tuned for cost-awareness.
    # The key difference from EI is the linear penalty on sigma/cost.
    
    # Let's try a constant beta first, or a very simple heuristic.
    # If we assume max ~ 30 evaluations, and we want to explore in the first 10.
    # Let's use a beta that decays with f_best?
    # Actually, without n, a constant beta is the safest "deterministic LCB".
    # The inspiration mentions dynamic scaling, but without n, we have to be clever.
    # Maybe we can estimate n from the fact that we are in a loop?
    # No, stateless.
    
    # Let's assume a beta that balances exploration and exploitation for the given cost range.
    # Cost is in [1, 2].
    # sigma is in objective units.
    # mu is in objective units.
    # f_best is in objective units.
    
    # Let's try beta = 1.0.
    # Score = mu - beta * sigma / cost
    # We want to MINIMIZE this score? 
    # Wait, the task says: "Return 2000 finite scores (higher = evaluate next)".
    # So we need to MAXIMIZE the acquisition value.
    # LCB typically minimizes mu - beta*sigma.
    # So Acquisition = -(mu - beta * sigma / cost) = -mu + beta * sigma / cost.
    
    # Let's refine beta.
    # If we are early, we want larger beta. If late, smaller beta.
    # Since we can't know n, let's use f_best.
    # Normalize f_best? 
    # If f_best is > 10, assume early. If f_best < 1, assume late.
    # This is fragile.
    
    # Let's stick to a robust constant or a very simple dynamic based on f_best magnitude.
    # Given the "Levy 2D" context, values can vary.
    # Let's try a beta that is inversely proportional to f_best? No.
    
    # Let's use a constant beta = 2.0. This is a common choice for LCB in BO.
    # The cost adjustment (dividing sigma by cost) is the key innovation here.
    # Low cost -> higher penalty for uncertainty -> more exploration in cheap regions.
    # High cost -> lower penalty for uncertainty -> less exploration in expensive regions.
    
    # Ensure no division by zero or negative costs (cost is always >= 1).
    # Ensure sigma is non-negative.
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Dynamic beta estimation:
    # Let's assume that if f_best is significantly larger than the expected minimum (0), we are early.
    # But we don't know the scale.
    # Let's use a fixed beta for stability, as dynamic scaling without n is risky.
    # However, the directive says "dynamically scaled".
    # Can we use the median of mu? No.
    
    # Let's try to estimate n from f_best.
    # In many BO tasks, f_best decreases roughly exponentially or linearly.
    # Let's assume a maximum f_best of ~100 (safe upper bound for Levy on this domain).
    # progress = 1 - f_best / 100 ?
    # If f_best > 100, progress = 0.
    # If f_best < 0, progress = 1.
    
    # Let's use a soft cap.
    # beta = 2.0 * (1.0 - np.clip(f_best / 50.0, 0.0, 1.0)) + 0.1
    # This starts at beta=2.0 when f_best=0? No, that's backwards.
    # If f_best is high (early), we want high beta.
    # If f_best is low (late), we want low beta.
    
    # Let's assume f_best starts around 20-50 and goes to 0.
    # beta = 0.5 + 1.5 * np.clip(f_best / 20.0, 0.0, 1.0)
    # If f_best = 20, beta = 2.0.
    # If f_best = 0, beta = 0.5.
    
    # This is a guess. Let's make it more robust.
    # Use a constant beta = 1.5.
    
    beta = 1.5
    
    # Acquisition = -mu + beta * sigma / cost
    # Higher is better.
    acq = -mu + beta * sigma_safe / cost_safe
    
    # Ensure finite
    acq = np.where(np.isfinite(acq), acq, 0.0)
    
    return acq
# EVOLVE-BLOCK-END
