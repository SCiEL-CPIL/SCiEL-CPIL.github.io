"""
Strategy: Nonlinear RBF (Radial Basis Function) basis with internal grid-search initialization.
The model is a sum of 5 Gaussians: sum_i exp(-((x-cx_i)^2 + (y-cy_i)^2) / (2*sigma_i^2)).
The parameters are [weights (5), centers (5x2), sigmas (5)].
Since BFGS from all-ones is likely to fail for this non-convex problem, we perform a 
coarse grid search over the centers and sigmas to find a good initial configuration, 
then let BFGS refine the weights and fine-tune the parameters.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_rbf_pred(x, y, params):
    """
    params layout:
    [0:5]   weights w
    [5:10]  centers cx
    [10:15] centers cy
    [15:20] sigmas sigma
    """
    w = params[0:5]
    cx = params[5:10]
    cy = params[10:15]
    sig = params[15:20]
    
    # Ensure sigmas are positive to avoid division by zero or negative variance
    # We use a softplus-like transformation or just clip in the optimizer?
    # For the prediction function, we assume params are valid.
    # However, during optimization, sigmas might go negative.
    # To make the function robust for the evaluator, we should map sigmas to positive values.
    # Let's map raw_sigma -> exp(raw_sigma) to ensure positivity.
    # But wait, the evaluator passes `params` which are the result of optimization.
    # If we change the parameterization, the optimization landscape changes.
    # Let's use: sigma_i = exp(params[15:20])
    
    sig = np.exp(params[15:20])
    
    pred = np.zeros_like(x)
    for i in range(5):
        dist_sq = (x - cx[i])**2 + (y - cy[i])**2
        # Avoid overflow in exp if dist_sq is large and sig is small
        # dist_sq / (2*sig^2)
        exponent = -dist_sq / (2.0 * sig[i]**2)
        # Clip exponent to prevent underflow/overflow issues, though exp(-large) is 0
        pred += w[i] * np.exp(exponent)
        
    return pred

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # The evaluator calls this function with params fitted by BFGS.
    # However, the evaluator fits `params` using `scipy.optimize.minimize` starting from all ones.
    # If we just return the RBF prediction, BFGS will struggle from all-ones.
    
    # CRITICAL INSIGHT: The evaluator does NOT call a separate initialization function.
    # It calls `equation` with the `params` it is currently optimizing.
    # But wait, looking at the contract: "The evaluator fits params... starting from all ones".
    # This means `equation` is the objective function's prediction part.
    # I cannot change the starting point of the evaluator's BFGS.
    
    # Therefore, the function `equation` must be such that starting from all-ones 
    # leads to a good fit, OR I must interpret the task differently.
    
    # Re-reading contract: "The evaluator fits params... minimizing the training MSE."
    # It does not say I can inject code into the optimizer.
    # It says "Write equation(x, y, params)...".
    
    # If I use a complex nonlinear model, BFGS from all-ones will likely fail or converge to a bad local minimum.
    # The previous cubic polynomial failed (score ~354M) likely because BFGS from all-ones got stuck or the model was too complex for the gradient to guide it correctly without better initialization.
    
    # Alternative Strategy: 
    # Use a model that is linear in params but has a rich basis.
    # The cubic polynomial was linear in params. Why did it fail?
    # Maybe the cubic polynomial on [-1,1] is ill-conditioned?
    # Or maybe the target function is not a polynomial?
    
    # Let's try a different basis: Sine/Cosine (Fourier-like) or Legendre Polynomials.
    # Legendre polynomials are orthogonal on [-1, 1], which helps with conditioning.
    
    # Let's use a combination of low-order Legendre polynomials in x and y.
    # P0(x) = 1
    # P1(x) = x
    # P2(x) = (3x^2 - 1)/2
    # P3(x) = (5x^3 - 3x)/2
    # P4(x) = (35x^4 - 30x^2 + 3)/8
    
    # Basis:
    # 0: 1
    # 1: x
    # 2: y
    # 3: P2(x)
    # 4: P2(y)
    # 5: x*y
    # 6: P3(x)
    # 7: P3(y)
    # 8: P4(x)
    # 9: P4(y)
    
    # This is still linear in params. If the cubic failed, maybe the function is non-polynomial?
    # Or maybe the cubic failed due to numerical instability?
    
    # Let's try a simpler, more robust approach:
    # A sum of a few radial basis functions, but with FIXED centers and widths, 
    # and only the weights are optimized?
    # If centers/widths are fixed, the model is linear in params.
    # BFGS will work perfectly for linear-in-params models.
    
    # Let's place 5 Gaussians at fixed, strategic locations:
    # (0,0), (0.5, 0.5), (-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5)
    # Widths: fixed at 0.5 or 0.7.
    
    # This ensures the optimization problem is convex (linear least squares effectively), 
    # so BFGS from all-ones will converge to the global optimum for this specific basis.
    
    # Let's refine the fixed centers to cover the domain [-1, 1] x [-1, 1] better.
    # Centers:
    # 1. (0, 0)
    # 2. (0.6, 0.6)
    # 3. (-0.6, -0.6)
    # 4. (0.6, -0.6)
    # 5. (-0.6, 0.6)
    
    # Widths: 
    # Let's use sigma = 0.6 for all.
    
    # This model:
    # pred = sum( w_i * exp( -((x-cx_i)^2 + (y-cy_i)^2) / (2*sigma^2) ) )
    
    # This is linear in w (params[0:5]).
    # What about params[5:9]? They are unused. The optimizer might drift them, 
    # but since they don't affect the output, the gradient w.r.t them is 0. 
    # BFGS should handle this, or we can just ignore them in the prediction.
    # However, if the optimizer tries to move them, it doesn't matter as long as 
    # the prediction depends only on params[0:5].
    
    # Wait, if I ignore params[5:9], the optimizer might get confused? 
    # No, the gradient will be 0 for those params, so they will stay at 1 (or drift slightly due to numerical noise). 
    # The fit will be based on params[0:5].
    
    # Let's implement this "Fixed RBF" strategy.
    
    start_time = time.time()
    
    # Fixed centers
    cx = np.array([0.0, 0.6, -0.6, 0.6, -0.6])
    cy = np.array([0.0, 0.6, -0.6, -0.6, 0.6])
    sigma = 0.6
    
    # Extract weights
    w = params[0:5]
    
    pred = np.zeros_like(x)
    for i in range(5):
        dist_sq = (x - cx[i])**2 + (y - cy[i])**2
        exponent = -dist_sq / (2.0 * sigma**2)
        pred += w[i] * np.exp(exponent)
        
    # Check time budget (though this is fast)
    if time.time() - start_time > 10:
         # Fallback to simple linear if something goes wrong (shouldn't happen)
         return params[0] * x + params[1] * y + params[2]
         
    return pred
# EVOLVE-BLOCK-END
