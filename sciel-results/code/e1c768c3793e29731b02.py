# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution
from scipy.stats import beta as beta_dist

def _compute_P_fast(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] exactly for discrete measures.
    Optimized for speed using sorting and searchsorted.
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    w = weights / np.sum(weights)
    
    # Pairwise sums: s_i + s_j for all (i,j)
    # We need sum_{i,j} w_i w_j * 1[s_i + s_j < 2*s_l - s_k]
    
    # Sort support to potentially help with structure, though not strictly necessary for searchsorted
    # Actually, let's just build the full n x n matrix of pair sums and pair weights
    pair_sums = support[:, None] + support[None, :]
    pair_weights = w[:, None] * w[None, :]
    
    # Flatten
    flat_sums = pair_sums.ravel()
    flat_w = pair_weights.ravel()
    
    # Sort by sum value
    order = np.argsort(flat_sums, kind='mergesort')
    sorted_sums = flat_sums[order]
    sorted_w = flat_w[order]
    
    # Cumulative sum of weights
    # cumsum_w[i] is sum of sorted_w[0..i-1]
    cumsum_w = np.cumsum(sorted_w)
    
    # We need for each (k, l) pair:
    # threshold = 2*s_l - s_k
    # Sum of weights where pair_sum < threshold
    # np.searchsorted(sorted_sums, threshold, side='left') gives index of first element >= threshold
    # So elements at indices [0, idx) are < threshold.
    # Sum is cumsum_w[idx-1] if idx > 0 else 0.
    # Or simply: prepend 0 to cumsum_w and index with idx.
    
    padded_cumsum = np.concatenate([[0.0], cumsum_w])
    
    # Thresholds matrix: shape (n, n) where [k, l] corresponds to s_k, s_l
    # P = sum_k sum_l w_k w_l * (sum_{i,j: s_i+s_j < 2*s_l - s_k} w_i w_j)
    
    # thresholds[k, l] = 2 * s_l - s_k
    # Note: support is 1D array of length n.
    # s_l is support[l], s_k is support[k].
    # We want a matrix T where T[k, l] = 2 * support[l] - support[k]
    
    # support[:, None] is (n, 1) -> s_k
    # support[None, :] is (1, n) -> s_l
    # T = 2 * support[None, :] - support[:, None]
    
    thresholds = 2.0 * support[None, :] - support[:, None]
    
    # Get indices
    # searchsorted returns (n, n) array of indices
    indices = np.searchsorted(sorted_sums, thresholds, side='left')
    
    # Get cumulative sums at these indices
    # inner[k, l] = padded_cumsum[indices[k, l]]
    inner = padded_cumsum[indices]
    
    # Final sum
    # P = sum_k sum_l w_k w_l inner[k, l]
    P = np.sum(w[:, None] * w[None, :] * inner)
    
    return P

def _continuous_P(params, grid_size=200):
    """
    Evaluate P for a continuous Beta distribution parameterized by alpha, beta, shift.
    We discretize the distribution on [shift, shift+1] using grid_size points.
    Returns -P to be minimized.
    """
    alpha, beta_val, shift = params
    
    # Ensure parameters are valid for Beta distribution
    if alpha < 0.1 or beta_val < 0.1 or shift < -1e-9:
        return 1.0 # High penalty (low score)
    
    # Create support points
    # Uniform grid on [shift, shift + 1]
    support = shift + np.linspace(0, 1, grid_size)
    
    # Calculate PDF values
    # Beta distribution is defined on [0,1]. We are scaling it to [shift, shift+1].
    # If X ~ Beta(alpha, beta), then Y = shift + X ~ Shifted Beta.
    # PDF of Y at y is PDF_X(y - shift).
    x_scaled = support - shift
    # Clip to [0, 1] to avoid numerical issues at boundaries
    x_scaled = np.clip(x_scaled, 0.0, 1.0)
    
    pdf_vals = beta_dist.pdf(x_scaled, alpha, beta_val)
    
    # Handle cases where pdf might be 0 or inf
    pdf_vals = np.where(np.isfinite(pdf_vals), pdf_vals, 0.0)
    
    # Normalize weights to sum to 1
    w_sum = np.sum(pdf_vals)
    if w_sum < 1e-9:
        return 1.0
        
    weights = pdf_vals / w_sum
    
    # Compute P
    P = _compute_P_fast(support, weights)
    
    return -P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game.
    Strategy: Optimize a continuous Shifted Beta distribution, then discretize.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600s wall clock limit
    
    best_P = -np.inf
    best_support = None
    best_weights = None
    
    # Define bounds for differential evolution
    # alpha, beta in [0.1, 10.0]
    # shift in [0.0, 1.0] (Since support must be >= 0, and we map [0,1] to [shift, shift+1])
    # Actually, the support can be anywhere >= 0.
    # Let's allow shift in [0, 5] to cover a reasonable range.
    bounds = [(0.1, 10.0), (0.1, 10.0), (0.0, 5.0)]
    
    # Use differential evolution for global optimization
    # maxiter and tol can be adjusted based on time budget
    # DE is stochastic, so we set a seed for reproducibility if needed, but here we want exploration.
    
    try:
        # Run DE with a moderate population to fit in time
        # maxiter=50, popsize=20 -> 1000 evaluations
        # Each evaluation takes ~0.1-0.5s depending on grid size.
        # Let's use grid_size=100 for speed during optimization, then refine with grid_size=200.
        
        # Phase 1: Coarse search
        def objective_coarse(params):
            return _continuous_P(params, grid_size=50)
            
        res_coarse = differential_evolution(objective_coarse, bounds, 
                                            maxiter=30, 
                                            popsize=15,
                                            tol=1e-4,
                                            seed=42,
                                            polish=True)
        
        if time.time() > deadline - 60:
            # If we are running out of time, just use the best found so far
            pass
        else:
            # Phase 2: Fine search around the best point
            x_best_coarse = res_coarse.x
            
            # Define a local neighborhood
            local_bounds = [
                (max(0.1, x_best_coarse[0] - 1.0), min(10.0, x_best_coarse[0] + 1.0)),
                (max(0.1, x_best_coarse[1] - 1.0), min(10.0, x_best_coarse[1] + 1.0)),
                (max(0.0, x_best_coarse[2] - 0.5), min(5.0, x_best_coarse[2] + 0.5))
            ]
            
            def objective_fine(params):
                return _continuous_P(params, grid_size=100)
                
        try:
            res_fine = differential_evolution(objective_fine, local_bounds,
                                              maxiter=50,
                                              popsize=10,
                                              tol=1e-5,
                                              seed=43,
                                              polish=True)
            
            x_opt = res_fine.x
        except Exception as e:
            print(f"DE failed: {e}")
            x_opt = np.array([1.0, 1.0, 0.0]) # Fallback to uniform
            
        # Final high-resolution discretization
        alpha, beta_val, shift = x_opt
        
        # Use a larger grid for the final answer to get a better approximation of the continuous integral
        # The problem asks for a discrete measure. A finer grid approximates the continuous one better.
        # However, the evaluator computes P exactly on the discrete support.
        # So we want the discrete measure that maximizes P.
        # The continuous optimum is a good guide.
        
        grid_sizes = [100, 200, 300, 400]
        best_grid_P = -np.inf
        best_grid_support = None
        best_grid_weights = None
        
        for gs in grid_sizes:
            if time.time() > deadline - 30:
                break
                
            support = shift + np.linspace(0, 1, gs)
            x_scaled = np.clip(support - shift, 0.0, 1.0)
            pdf_vals = beta_dist.pdf(x_scaled, alpha, beta_val)
            pdf_vals = np.where(np.isfinite(pdf_vals), pdf_vals, 0.0)
            
            w_sum = np.sum(pdf_vals)
            if w_sum < 1e-9:
                continue
                
            weights = pdf_vals / w_sum
            
            P = _compute_P_fast(support, weights)
            
            if P > best_grid_P:
                best_grid_P = P
                best_grid_support = support.copy()
                best_grid_weights = weights.copy()
                
        # If DE didn't find anything good, fallback to a simple baseline
        if best_grid_support is None:
            n = 100
            best_grid_support = np.linspace(0, 1, n)
            best_grid_weights = np.ones(n) / n
            
        # Return the best discrete measure found
        return best_grid_support, best_grid_weights

    except Exception as e:
        print(f"Critical error: {e}")
        # Absolute fallback
        n = 10
        support = np.arange(n, dtype=np.float64)
        weights = np.ones(n, dtype=np.float64)
        return support, weights

# EVOLVE-BLOCK-END
