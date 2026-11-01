import time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 14
D = 3
DEADLINE = 145.0
_t0 = None

def _true_score(pts):
    """
    Calculate the squared ratio of max to min pairwise distance.
    """
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    mn = d.min()
    mx = d.max()
    if mn < 1e-9:
        return np.inf
    return (mx / mn) ** 2

def _soft_ratio_objective(pts_flat, beta):
    """
    Smooth surrogate for the max/min ratio squared.
    Uses Log-Sum-Exp to approximate max and min.
    R = max(d) / min(d)
    R^2 = max(d)^2 / min(d)^2
    
    To make it smooth and differentiable:
    log(max(d)) ≈ (1/beta) * log(sum(exp(beta * d)))
    log(min(d)) ≈ -(1/beta) * log(sum(exp(-beta * d)))
    
    We want to minimize R^2.
    log(R^2) = 2 * (log(max(d)) - log(min(d)))
    
    Let L_max = (1/beta) * log(sum(exp(beta * d)))
    Let L_min = -(1/beta) * log(sum(exp(-beta * d)))
    
    Objective = 2 * (L_max - L_min)
    
    Note: d contains inf on diagonal. We must mask it.
    """
    pts = pts_flat.reshape(-1, 3)
    d = cdist(pts, pts)
    
    # Mask diagonal
    mask = ~np.eye(len(pts), dtype=bool)
    d_masked = d[mask]
    
    if len(d_masked) == 0:
        return 1e10
        
    # Numerical stability for LogSumExp
    # L_max = (1/beta) * (max_d + log(sum(exp(beta * (d - max_d)))))
    max_d = np.max(d_masked)
    min_d = np.min(d_masked)
    
    # If min_d is very small, the ratio is huge.
    # Also, if points coincide, min_d ~ 0.
    # We add a small epsilon to min_d to avoid log(0) or division by zero in the ratio if we were doing direct division.
    # But in the log-sum-exp form:
    
    # Term for max:
    # sum(exp(beta * d)) can overflow.
    # exp(beta * d) = exp(beta * max_d) * exp(beta * (d - max_d))
    # log(sum(exp(beta * d))) = beta * max_d + log(sum(exp(beta * (d - max_d))))
    
    # Term for min:
    # sum(exp(-beta * d)) = exp(-beta * min_d) * sum(exp(-beta * (d - min_d)))
    # log(sum(exp(-beta * d))) = -beta * min_d + log(sum(exp(-beta * (d - min_d))))
    
    # L_max = max_d + (1/beta) * log(sum(exp(beta * (d - max_d))))
    # L_min = min_d - (1/beta) * log(sum(exp(-beta * (d - min_d))))
    
    # Objective = 2 * (L_max - L_min)
    
    # For large beta, this approaches 2 * (max_d - min_d) ? No.
    # If beta -> inf, L_max -> max_d, L_min -> min_d.
    # So Objective -> 2 * (max_d - min_d).
    # But we want to minimize (max/min)^2.
    # Minimizing (max-min) is NOT the same as minimizing (max/min)^2.
    # Example: max=10, min=1 -> diff=9, ratio^2=100
    #          max=2, min=1 -> diff=1, ratio^2=4
    #          max=1.1, min=1 -> diff=0.1, ratio^2=1.21
    # Minimizing diff tends to shrink the whole configuration if not constrained, or just make them close.
    # However, we are in 3D space, unconstrained scale.
    # Wait, the score is scale invariant! (max/min)^2 is scale invariant.
    # So we can fix the scale. E.g., fix sum of squared distances or center of mass.
    # If we don't fix scale, the optimizer might shrink everything to 0 to minimize max-min?
    # No, max-min scales linearly with size. Ratio is invariant.
    # If we minimize 2*(max-min), and scale is free, we can make max-min arbitrarily small by scaling down?
    # No, max-min scales with size. If we scale down by k, max-min becomes k*(max-min).
    # So minimizing max-min will drive the scale to 0.
    # We MUST constrain the scale.
    
    # Better surrogate:
    # Minimize log(R^2) = 2 * (log(max) - log(min))
    # This is scale invariant.
    
    # Let's use the LogSumExp approximation for log(max) and log(min).
    # log(max(d)) ≈ (1/beta) * log(sum(exp(beta * d)))
    # log(min(d)) ≈ -(1/beta) * log(sum(exp(-beta * d)))
    
    # obj = 2 * [ (1/beta)*log(sum(exp(beta*d))) - (-(1/beta)*log(sum(exp(-beta*d)))) ]
    # obj = (2/beta) * [ log(sum(exp(beta*d))) + log(sum(exp(-beta*d))) ]
    
    # Let's compute this carefully.
    # sum_exp_pos = sum(exp(beta * d))
    # sum_exp_neg = sum(exp(-beta * d))
    
    # To avoid overflow:
    # log_sum_exp_pos = max_d + (1/beta) * log(sum(exp(beta * (d - max_d))))
    # log_sum_exp_neg = -min_d + (1/beta) * log(sum(exp(-beta * (d - min_d))))
    
    # obj = (2/beta) * (log_sum_exp_pos + log_sum_exp_neg)
    # obj = (2/beta) * (max_d - min_d + (1/beta)*(log(sum(exp(beta*(d-max_d)))) + log(sum(exp(-beta*(d-min_d))))))
    
    # As beta -> inf, the log terms go to log(1) = 0 (since only the max term in pos sum and min term in neg sum survive).
    # So obj -> (2/beta) * (max_d - min_d).
    # This still has the scale issue if we just minimize this.
    # Wait, log(max) is not max.
    # The approximation log(max) ≈ (1/beta) log(sum(exp(beta d))) is correct for large beta.
    # So obj ≈ 2 * (log(max) - log(min)).
    # This is scale invariant.
    
    # Let's implement:
    # L_pos = log(sum(exp(beta * d)))
    # L_neg = log(sum(exp(-beta * d)))
    
    # Stable calculation:
    # L_pos = max_d + log(sum(exp(beta * (d - max_d))))
    # L_neg = -min_d + log(sum(exp(-beta * (d - min_d))))
    
    # Note: sum(exp(beta*(d-max_d))) is dominated by d=max_d. exp(0)=1. Others < 1.
    # sum(exp(-beta*(d-min_d))) is dominated by d=min_d. exp(0)=1. Others < 1.
    
    # So L_pos ≈ max_d + log(1 + small) ≈ max_d
    # L_neg ≈ -min_d + log(1 + small) ≈ -min_d
    
    # obj = (2/beta) * (L_pos + L_neg)
    # obj ≈ (2/beta) * (max_d - min_d)
    
    # This is NOT log(max/min). It is (max-min)/beta.
    # I made a mistake in the derivation.
    # log(max) ≈ (1/beta) L_pos.
    # log(min) ≈ (1/beta) L_neg ? No.
    # min(d) = exp(log(min(d))).
    # log(min(d)) = log(1 / max(1/d)) = -log(max(1/d)).
    # Let e_i = 1/d_i. max(e) = 1/min(d).
    # log(max(e)) ≈ (1/beta) log(sum(exp(beta * e))) = (1/beta) log(sum(exp(-beta * d))).
    # So log(min(d)) = - (1/beta) log(sum(exp(-beta * d))).
    
    # So:
    # log(max(d)) ≈ (1/beta) L_pos
    # log(min(d)) ≈ - (1/beta) L_neg
    
    # Target: minimize 2 * (log(max) - log(min))
    # ≈ (2/beta) * (L_pos + L_neg)
    
    # This matches my formula.
    # But does it converge to 2*(log(max)-log(min))?
    # L_pos ≈ beta * max_d. So (1/beta)L_pos ≈ max_d.
    # L_neg ≈ -beta * min_d. So (1/beta)L_neg ≈ -min_d.
    # So (2/beta)(L_pos + L_neg) ≈ 2(max_d - min_d).
    
    # The approximation log(max) ≈ max is ONLY true if we are in log-space? No.
    # The identity is: lim_{beta->inf} (1/beta) log(sum(exp(beta x_i))) = max(x_i).
    # So (1/beta) L_pos -> max(d).
    # So my objective converges to 2 * (max(d) - min(d)).
    # This is NOT scale invariant.
    # If I scale points by k, max->k*max, min->k*min.
    # Obj -> 2*k*(max-min).
    # Minimizer will drive k -> 0.
    
    # I need to minimize log(max/min).
    # log(max/min) = log(max) - log(min).
    # log(max) ≈ (1/beta) L_pos.
    # log(min) ≈ - (1/beta) L_neg.
    # So log(max/min) ≈ (1/beta) L_pos + (1/beta) L_neg = (1/beta) (L_pos + L_neg).
    # Wait.
    # If (1/beta) L_pos -> max(d), then log(max) is approximated by max(d)?
    # No! The limit is max(d). So the approximation for log(max(d)) is max(d)?
    # That implies log(max(d)) ≈ max(d), which is false.
    
    # Let's re-evaluate.
    # We want to approximate f(x) = log(max(x)).
    # Softmax: max(x) ≈ (1/beta) log(sum(exp(beta x))).
    # So log(max(x)) ≈ log( (1/beta) log(sum(exp(beta x))) )? No.
    # The Softmax trick approximates MAX, not LOG-MAX.
    # max(x) ≈ (1/beta) log(sum(exp(beta x))).
    # So log(max(x)) ≈ log( (1/beta) log(sum(exp(beta x))) ).
    
    # This is getting complicated and potentially unstable.
    
    # Alternative:
    # Minimize R^2 directly using a smooth max/min.
    # R = max(d) / min(d).
    # max(d) ≈ (1/beta) log(sum(exp(beta d))).
    # min(d) ≈ 1 / ( (1/beta) log(sum(exp(-beta d))) )?
    # min(d) = 1 / max(1/d).
    # max(1/d) ≈ (1/beta) log(sum(exp(beta/d))).
    # So min(d) ≈ beta / log(sum(exp(beta/d))).
    
    # R ≈ [ (1/beta) log(sum(exp(beta d))) ] / [ beta / log(sum(exp(beta/d))) ]
    # R ≈ (1/beta^2) * log(sum(exp(beta d))) * log(sum(exp(beta/d)))
    
    # This is a valid smooth approximation for R.
    # We want to minimize R^2.
    
    # Let's implement this.
    # d_masked: (M,)
    # term1 = log(sum(exp(beta * d)))
    # term2 = log(sum(exp(beta / d)))  # Note: d must be > 0.
    
    # Stability:
    # term1 = max_d + log(sum(exp(beta * (d - max_d))))
    # term2 = max_inv + log(sum(exp(beta * (1/d - max_inv))))
    # where max_inv = max(1/d) = 1/min_d.
    
    # R_approx = (1/beta^2) * term1 * term2
    # Obj = R_approx^2
    
    # As beta -> inf:
    # term1 -> beta * max_d
    # term2 -> beta * max_inv = beta / min_d
    # R_approx -> (1/beta^2) * (beta * max_d) * (beta / min_d) = max_d / min_d.
    # Correct!
    
    # So the objective is:
    # R_approx = (term1 * term2) / (beta * beta)
    # return R_approx * R_approx
    
    if np.any(d_masked <= 0):
        return 1e10
        
    max_d = np.max(d_masked)
    min_d = np.min(d_masked)
    max_inv = 1.0 / min_d
    
    # term1
    exp_args_pos = beta * (d_masked - max_d)
    # exp_args_pos <= 0
    sum_exp_pos = np.sum(np.exp(exp_args_pos))
    term1 = max_d + (1.0 / beta) * np.log(sum_exp_pos)
    
    # term2
    inv_d = 1.0 / d_masked
    exp_args_inv = beta * (inv_d - max_inv)
    # exp_args_inv <= 0
    sum_exp_inv = np.sum(np.exp(exp_args_inv))
    term2 = max_inv + (1.0 / beta) * np.log(sum_exp_inv)
    
    R_approx = (term1 * term2) / (beta * beta)
    
    # Add a small penalty for min_d being too small to avoid numerical issues in 1/d
    # But R_approx handles it.
    
    return R_approx * R_approx

def _fib_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n)
    phi_angle = np.arccos(1 - 2 * (i + 0.5) / n)
    theta_angle = np.pi * (1 + 5 ** 0.5) * i
    x = np.sin(phi_angle) * np.cos(theta_angle)
    y = np.sin(phi_angle) * np.sin(theta_angle)
    z = np.cos(phi_angle)
    return np.column_stack([x, y, z])

def construct():
    global _t0
    if _t0 is None:
        _t0 = time.time()
    
    # Initialize with Fibonacci sphere
    pts = _fib_sphere(N)
    best_pts = pts.copy()
    best_score = _true_score(pts)
    
    # Simulated Annealing Parameters
    T_start = 1.0
    T_end = 1e-6
    alpha = 0.99
    beta_start = 10.0
    beta_end = 1000.0
    
    # Number of steps
    # We have ~145s.
    # Each step: cdist (14x14) + exps. Very fast.
    # Let's run as many steps as possible.
    
    rng = np.random.default_rng(42)
    
    T = T_start
    beta = beta_start
    
    # Pre-allocate
    n_points = N
    
    while time.time() - _t0 < DEADLINE - 5:
        # Update beta
        # Linearly increase beta to sharpen the max/min approximation
        # Or keep it constant? Increasing beta makes the landscape more rugged (closer to true max/min).
        # Start with small beta (smooth), end with large beta (sharp).
        
        # Generate random perturbation
        # Gaussian noise
        step_size = T
        noise = rng.normal(0, step_size, (n_points, 3))
        new_pts = pts + noise
        
        # Evaluate
        obj_new = _soft_ratio_objective(new_pts.flatten(), beta)
        obj_cur = _soft_ratio_objective(pts.flatten(), beta)
        
        # Acceptance probability
        if obj_new < obj_cur:
            accept = True
        else:
            delta_obj = obj_new - obj_cur
            # P(accept) = exp(-delta_obj / T)
            # If T is small, only accept if delta is negative (which we already checked) or very small positive.
            if delta_obj / T < -700: # underflow protection
                accept = False
            else:
                accept = rng.random() < np.exp(-delta_obj / T)
                
        if accept:
            pts = new_pts
            # Check true score occasionally? 
            # The soft objective is a proxy. 
            # We should track the true score to ensure we are improving the actual metric.
            # But SA minimizes the proxy. 
            # If the proxy is a good approximation, it should correlate.
            # To be safe, we can check true score every 100 steps.
            
        # Cooldown
        T *= alpha
        if T < T_end:
            T = T_end
            
        # Increase beta slowly
        beta *= 1.001
        if beta > beta_end:
            beta = beta_end

        # Periodic check of true score
        # This is expensive but ensures we don't diverge
        # Let's do it every 50 steps
        # Actually, let's just check at the end or if we have time.
        # The soft objective with high beta is very close to true score.
        
        # If we get stuck, we might need to restart or increase T.
        # For now, simple SA.
        
    # Final polish with L-BFGS-B on the soft objective with high beta
    if time.time() - _t0 < DEADLINE - 2:
        final_beta = 1000.0
        try:
            res = minimize(
                lambda x: _soft_ratio_objective(x, final_beta),
                pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-5}
            )
            if res.success:
                new_pts = res.x.reshape(-1, 3)
                new_score = _true_score(new_pts)
                if new_score < best_score:
                    best_pts = new_pts
                    best_score = new_score
        except Exception:
            pass
    
    # One last check on the current pts
    final_score = _true_score(pts)
    if final_score < best_score:
        best_pts = pts
        best_score = final_score
        
    return best_pts
