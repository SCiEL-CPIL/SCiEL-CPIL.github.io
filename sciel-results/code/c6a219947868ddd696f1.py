"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Construct a signed step function to minimize the C3 upper bound.
    Metric: max_k |h*h_k| / (sum_i |h_i|)^2
    Strategy:
    1. Relax h_i to be in (-1, 1) using h_i = tanh(x_i).
    2. Minimize a smooth approximation of the peak auto-correlation.
    3. Since the metric is scale-invariant, we can assume sum |h_i| is fixed or just minimize the peak.
       However, if we let h vary continuously, the optimizer might shrink h to 0 to minimize the numerator,
       but the denominator also shrinks. The ratio is scale invariant.
       To avoid the trivial solution h=0 (where 0/0 is undefined or 0), we should constrain the norm.
       Easiest constraint: force h_i to be close to +/- 1.
       We can do this by optimizing x, where h = tanh(x), and adding a penalty if |h| is not close to 1.
       Or simply, since the problem asks for a "signed step function", discrete +/-1 is the natural interpretation.
       We will optimize x to minimize the peak auto-correlation of tanh(x), then round to +/-1.
       
    The objective function to minimize is the log-sum-exp of the squared auto-correlations,
    which approximates the max squared auto-correlation.
    """
    
    def objective(x):
        h = np.tanh(x)
        conv = np.convolve(h, h, mode='full')
        # We want to minimize max(conv^2).
        # Use log-sum-exp with temperature T.
        # As T decreases, it approaches max.
        # Start with a moderate T and maybe anneal? 
        # For simplicity, use a fixed T that is small enough to be selective but large enough for gradients.
        # conv values for N=400, h in [-1,1] are in [-400, 400]. conv^2 in [0, 160000].
        # T=100 is reasonable.
        T = 100.0
        conv_sq = conv**2
        # Subtract max for numerical stability
        max_val = np.max(conv_sq)
        log_sum_exp = T * (np.log(np.sum(np.exp((conv_sq - max_val) / T))) + max_val / T)
        
        # Add a penalty to encourage |h_i| to be close to 1 (binary-like)
        # This helps the optimizer find a discrete solution.
        # Penalty: sum(1 - h_i^2). If h_i is +/-1, penalty is 0.
        # If h_i is 0, penalty is N.
        penalty = 0.1 * np.sum(1.0 - h**2)
        
        return log_sum_exp + penalty

    # Initialize with random signs to break symmetry
    # x0 such that tanh(x0) is roughly +/- 1
    # random.uniform(-5, 5) gives tanh values in approx (-0.9999, 0.9999)
    np.random.seed(42) # For reproducibility during search, though evaluator might not care
    x0 = np.random.uniform(-5.0, 5.0, N)
    
    # Optimize
    # L-BFGS-B is good for box-constrained problems, but x is unbounded here.
    # We don't need bounds on x if tanh handles the saturation.
    res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
    
    h_opt = np.tanh(res.x)
    
    # Round to nearest sign +/- 1
    # The problem asks for "signed step heights". 
    # If we return floats, it's valid. If we return +/-1, it's also valid and likely intended for "signed step".
    # Let's return the rounded values to ensure we are on the discrete manifold if possible.
    # Actually, returning the continuous h_opt might be slightly better if the optimizer found a non-binary minimum?
    # No, for binary sequences, the peak is integer. For continuous, it can be lower?
    # Let's check: If we allow h_i to be non-binary, can we get lower peak?
    # If we scale h, ratio is invariant.
    # If we change shapes, e.g. some h_i = 0.5, some 1.
    # The problem is "signed step function". This usually implies piecewise constant.
    # The seed used np.ones.
    # Let's stick to the continuous result h_opt, as it might have slightly lower peak than the rounded version.
    # However, the "signed step" might imply discrete levels.
    # Let's return h_opt. If the evaluator expects integers, it might fail? 
    # "sequence of signed step heights" - heights can be real.
    
    return h_opt
# EVOLVE-BLOCK-END
