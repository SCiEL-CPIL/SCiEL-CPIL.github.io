"""
Strategy: Adaptive RBF Boosting (Greedy Residual Minimization).
Instead of fitting a global basis (Fourier/Chebyshev) or inverting a full kernel matrix,
this approach builds the model incrementally. It starts with a constant offset and then
iteratively selects the training point that best reduces the residual sum of squares
when a Gaussian RBF is added at that location. The weight of each RBF is determined
analytically (closed-form least squares for a single new feature).
This is materially different from the current ensemble because:
1. It is adaptive: the basis functions are placed where the model is currently failing.
2. It avoids solving large linear systems: each step is O(N^2) for residual update, 
   but we can optimize this. Actually, with N=200, O(N^2) per step is fine for ~100 steps.
3. It naturally handles localized features and sharp transitions better than global polynomials.
4. No cross-validation grid search is needed for hyperparameters like 'degree' or 'gamma' 
   in the same way; the 'gamma' (width) can be fixed or lightly tuned, and the number of 
   terms is determined by a stopping criterion (e.g., residual reduction threshold).
"""
import numpy as np
import time
import warnings

warnings.filterwarnings("ignore")

_START_TIME = None
TIME_LIMIT = 330.0
SAFETY_MARGIN = 30.0

def _time_left():
    global _START_TIME
    if _START_TIME is None:
        return TIME_LIMIT - SAFETY_MARGIN
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - _START_TIME)

def fit(X_train, y_train):
    global _START_TIME
    _START_TIME = time.time()
    
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    
    # Ensure finite
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    
    if len(X) < 10:
        y_mean = y.mean() if len(y) > 0 else 0.0
        return lambda X_in: np.full(len(X_in), y_mean)
        
    x = X[:, 0]
    
    # Normalize x to [0, 1] for stable RBF widths? 
    # Or keep in original scale but pick gamma based on data spread.
    x_min, x_max = x.min(), x.max()
    x_range = x_max - x_min
    if x_range < 1e-6:
        x_range = 1.0
    
    # Heuristic for gamma: 
    # If we use K = exp(-gamma * (x - c)^2), 
    # a typical width is ~ x_range / 10. 
    # gamma = 1 / width^2 = 100 / x_range^2.
    # Let's try a few gammas or just one good one. 
    # Given the "boosting" nature, a slightly narrower RBF allows more precise localization.
    # Let's use gamma = 50 / x_range^2 as a starting point.
    gamma = 50.0 / (x_range ** 2)
    
    # Initialize model
    # Model: y_pred = c0 + sum(w_i * exp(-gamma * (x - c_i)^2))
    c0 = y.mean()
    centers = []
    weights = []
    
    residual = y - c0
    current_ssr = np.sum(residual ** 2)
    
    # Pre-compute distances between all training points? 
    # dist[i, j] = (x[i] - x[j])^2
    # K[i, j] = exp(-gamma * dist[i, j])
    # This is 200x200 = 40,000 elements. Very fast.
    diff = x[:, None] - x[None, :]
    dist_sq = diff ** 2
    K = np.exp(-gamma * dist_sq)
    
    # Greedy boosting loop
    # We want to find c_k, w_k that minimize ||residual - w_k * K[:, k]||^2
    # Optimal w_k = <residual, K[:, k]> / ||K[:, k]||^2
    # Reduction in SSR = w_k^2 * ||K[:, k]||^2 = <residual, K[:, k]>^2 / ||K[:, k]||^2
    
    max_iter = 100  # Cap to ensure speed
    min_ssr_reduction = 1e-6 * current_ssr  # Stop if improvement is tiny
    
    for step in range(max_iter):
        if _time_left() < 5:
            break
            
        # Compute <residual, K[:, j]> for all j
        # This is residual @ K
        dot_res_K = residual @ K
        
        # Compute ||K[:, j]||^2 for all j
        # This is np.sum(K**2, axis=0)
        # K is symmetric, so K**2 is also symmetric.
        # ||K[:, j]||^2 = sum_i K[i,j]^2
        norm_sq_K = np.sum(K ** 2, axis=0)
        
        # Avoid division by zero (shouldn't happen as K[i,i]=1)
        norm_sq_K = np.maximum(norm_sq_K, 1e-12)
        
        # SSR reduction for each candidate center j
        ssr_reduction = (dot_res_K ** 2) / norm_sq_K
        
        # Find best center
        best_idx = np.argmax(ssr_reduction)
        best_reduction = ssr_reduction[best_idx]
        
        if best_reduction < min_ssr_reduction:
            break
            
        # Update model
        c_best = x[best_idx]
        w_best = dot_res_K[best_idx] / norm_sq_K[best_idx]
        
        centers.append(c_best)
        weights.append(w_best)
        
        # Update residual
        # residual = residual - w_best * K[:, best_idx]
        residual = residual - w_best * K[:, best_idx]
        
        # Update SSR (optional, for logging/debugging, not strictly needed for next step 
        # as we recompute dot products from residual)
        current_ssr = np.sum(residual ** 2)
        
    # If no RBFs were added, just return the constant
    if not centers:
        return lambda X_in: np.full(len(X_in), c0)
    
    centers = np.array(centers)
    weights = np.array(weights)
    
    def predict(X_in):
        xi = X_in[:, 0]
        # Compute distances to centers
        # xi shape (N_test, 1), centers shape (1, N_centers)
        dists = xi[:, None] - centers[None, :]
        K_test = np.exp(-gamma * dists ** 2)
        return c0 + K_test @ weights
        
    return predict
