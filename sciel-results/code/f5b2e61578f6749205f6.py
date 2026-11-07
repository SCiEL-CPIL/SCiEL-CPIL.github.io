"""
Strategy: Sparse Non-Linear Basis Selection via L1 Regularization.
Instead of relying on a fixed, potentially redundant basis or a complex evolutionary search
that might be slow or unstable, we use the 120s budget to perform a robust feature selection
on a rich, non-linear basis. 

The core idea is that the 10 parameters are a fixed resource. We define a large candidate
basis of smooth, non-linear functions (polynomials, trigonometric, exponentials) on [-1, 1].
We then use L1 (Lasso) regularization to find a sparse combination of these features that
best fits the training data. The selected features are then mapped to the 10 parameters.

This approach is structurally different from the current GP approach because:
1. It uses a fixed, rich basis (no tree structure search).
2. It uses convex optimization (Lasso) for feature selection, which is global and robust.
3. It explicitly handles sparsity, avoiding the overfitting issues of dense 10-param models.
4. It is much faster than GP, allowing for multiple random restarts or basis variations if time permits.

The mapping is:
- We have a candidate basis B of shape (N, M) where M is large (e.g., 50).
- We fit a sparse model: y ≈ B @ w, with L1 penalty on w.
- We select the top 10 features with the largest |w|.
- We map these 10 features to params[0..9].
- The final prediction is sum(params[i] * feature_i).

This is linear in params, so BFGS will work perfectly. The non-linearity comes from the features.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Ensure inputs are 1D
    x_flat = np.ravel(x)
    y_flat = np.ravel(y)
    N = len(x_flat)
    
    # 1. Define a rich candidate basis
    # We want smooth functions on [-1, 1].
    # Polynomials up to degree 5, trig functions, and some exponentials.
    
    # Polynomial terms in x
    x_pows = [x_flat ** i for i in range(6)]
    # Polynomial terms in y
    y_pows = [y_flat ** i for i in range(6)]
    # Cross terms
    cross_terms = [x_flat ** i * y_flat ** j for i in range(3) for j in range(3) if i+j > 0]
    # Trig terms
    trig_x = [np.sin(x_flat * k) for k in [1, 2, 3]] + [np.cos(x_flat * k) for k in [1, 2, 3]]
    trig_y = [np.sin(y_flat * k) for k in [1, 2, 3]] + [np.cos(y_flat * k) for k in [1, 2, 3]]
    # Exponential terms (clipped to avoid overflow, though on [-1,1] it's safe)
    exp_x = [np.exp(k * x_flat) for k in [0.5, 1.0, 2.0]]
    exp_y = [np.exp(k * y_flat) for k in [0.5, 1.0, 2.0]]
    
    # Combine all candidate features
    # Note: We should avoid duplicate features (e.g., x**0 is 1, which is also a constant).
    # We'll just stack them and let Lasso handle redundancy (it will pick one or the other, or split weight).
    
    candidate_features = []
    candidate_features.extend(x_pows)
    candidate_features.extend(y_pows)
    candidate_features.extend(cross_terms)
    candidate_features.extend(trig_x)
    candidate_features.extend(trig_y)
    candidate_features.extend(exp_x)
    candidate_features.extend(exp_y)
    
    # Ensure all are 1D arrays
    candidate_features = [np.asarray(f).ravel() for f in candidate_features]
    
    # Stack into a matrix B (N x M)
    B = np.column_stack(candidate_features)
    
    # Remove constant column if present? No, Lasso can handle it, but a constant is important.
    # Actually, let's ensure we have a constant term.
    # x**0 is 1, so it's included in x_pows.
    
    # 2. Perform Lasso regression to select features
    # We want to minimize ||y - B w||^2 + lambda * ||w||_1
    # We can use scipy.optimize.minimize with L1 penalty, or a dedicated Lasso solver.
    # Since we don't have sklearn, we'll use a simple iterative Lasso (coordinate descent) or just use a penalty in minimize.
    # However, minimize with L1 is not convex-friendly for BFGS.
    # Let's use a simple Lasso implementation via coordinate descent, which is very fast.
    
    def lasso_coordinate_descent(B, y, lam, max_iter=100):
        M = B.shape[1]
        w = np.zeros(M)
        # Precompute column norms
        col_norms = np.sum(B**2, axis=0)
        # Precompute B.T @ B (Gram matrix) - might be expensive if M is large, but M is ~50, so 50x50 is fine.
        G = B.T @ B
        g = B.T @ y
        for _ in range(max_iter):
            for j in range(M):
                # Update w[j]
                r_j = g[j] - G[j, j] * w[j] - np.sum(G[j, :] * w) + G[j, j] * w[j] # This is inefficient
                # Better: maintain residual or use efficient update
                # Let's just do a simple soft-thresholding on the gradient
                # grad_j = -2 * B[:, j] @ (y - B @ w)
                # This is O(N) per step, O(M*N) per iteration. M=50, N=200, Iter=100 -> 1M ops. Very fast.
                residual = y - B @ w
                grad_j = -2.0 * np.dot(B[:, j], residual)
                # Soft threshold
                w_old = w[j]
                w[j] = np.sign(grad_j) * max(abs(grad_j) - lam, 0.0)
                # Update residual if w changed
                if w[j] != w_old:
                    residual += (w[j] - w_old) * B[:, j]
        return w
    
    # Choose lambda. We want a sparse solution.
    # We can try a few values of lambda.
    # Let's try a range of lambdas and pick the one with the best training score (with a sparsity penalty).
    # Or just pick a fixed lambda that encourages sparsity.
    # Let's try lambda = 0.1, 0.5, 1.0, 5.0
    
    best_w = None
    best_score = np.inf
    best_indices = None
    
    # To speed up, we can subsample if N is large, but N=200 is small.
    
    lambdas = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    
    for lam in lambdas:
        w = lasso_coordinate_descent(B, y_flat, lam, max_iter=50)
        # Calculate score
        pred = B @ w
        mse = np.mean((pred - y_flat)**2)
        # We want low MSE but also sparsity. The Lasso objective already balances this.
        # We just pick the w with the lowest MSE among the sparse solutions.
        # Actually, we should pick the w that gives the best *generalization*.
        # Since we don't have test data, we assume the sparsest good fit is best.
        # Let's just pick the one with the lowest training MSE.
        if mse < best_score:
            best_score = mse
            best_w = w
    
    # 3. Select top 10 features
    if best_w is None:
        # Fallback to linear model
        return params[0] * x_flat + params[1] * y_flat + params[2]
    
    # Sort indices by |w|
    sorted_indices = np.argsort(np.abs(best_w))[::-1]
    top_indices = sorted_indices[:10]
    
    # 4. Map selected features to params
    # We need to return an expression that uses params[0..9] corresponding to top_indices[0..9]
    
    # Construct the prediction
    # pred = sum(params[i] * B[:, top_indices[i]] for i in range(10))
    
    # To do this efficiently, we can extract the selected columns
    B_selected = B[:, top_indices]
    
    # pred = B_selected @ params
    pred = B_selected @ params
    
    return pred

# EVOLVE-BLOCK-END
