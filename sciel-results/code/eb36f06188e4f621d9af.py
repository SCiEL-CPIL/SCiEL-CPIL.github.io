"""Symbolic regression via Sparse Elastic Net on a rich symbolic basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr

def fit(X_train, y_train):
    start_time = time.time()
    # 300s limit. Reserve 10s for safety/overhead, leaving 290s for computation.
    deadline = start_time + 290.0
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64).flatten()
    n_samples, n_features = X.shape
    
    # Normalize inputs to [0, 1] for numerical stability in basis construction
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0
    X_norm = (X - X_min) / X_range
    
    # Standardize y for numerical stability
    y_mean = y.mean()
    y_std = y.std()
    if y_std == 0:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std

    # Build a rich basis of candidate symbolic terms
    # This approach aims to find a sparse linear combination of these terms that fits y.
    # Terms include:
    # 1. Constant
    # 2. Linear terms (x_i)
    # 3. Polynomial terms (x_i^k, x_i*x_j, etc.) up to degree 3
    # 4. Trigonometric terms (sin(pi*k*x_i), cos(pi*k*x_i)) for small k
    # 5. Rational terms (1/x_i) if safe
    
    def build_basis(X):
        n, d = X.shape
        terms = []
        term_names = []
        
        # 1. Constant
        terms.append(np.ones((n, 1)))
        term_names.append("const")
        
        # 2. Linear terms
        for i in range(d):
            terms.append(X[:, i:i+1])
            term_names.append(f"x{i}")
            
        # 3. Polynomial terms (degree 2 and 3)
        # To avoid explosion, we limit the number of terms.
        # For d=1,2,3, we can include all monomials up to degree 3 or 4.
        max_degree = 3 if d <= 2 else 2
        
        def get_monomials(d, max_deg):
            from itertools import product
            # Generate all exponent tuples (e1, ..., ed) such that sum(ei) <= max_deg
            # and sum(ei) >= 1 (linear is already handled, but we can include them again for simplicity 
            # or just start from degree 2 to avoid redundancy if we want strict sparsity selection 
            # to prefer simpler terms. However, Elastic Net will handle collinearity.
            # Let's include degree 2 and 3 explicitly to avoid redundancy with linear terms already added.
            monomials = []
            # Degree 2
            for e1 in range(0, max_deg + 1):
                for e2 in range(0, max_deg + 1 - e1):
                    # ... this is getting complex for variable d. 
                    # Simpler: generate all tuples of length d with sum between 2 and max_deg
                    pass
            
            # Efficient way for small d:
            from itertools import product
            exponents = []
            # Degree 2
            for combo in product(range(3), repeat=d):
                if sum(combo) == 2:
                    exponents.append(combo)
            # Degree 3
            for combo in product(range(4), repeat=d):
                if sum(combo) == 3:
                    exponents.append(combo)
            return exponents

        exponents = get_monomials(d, max_degree)
        for exp_tuple in exponents:
            term = np.ones((n, 1))
            name_parts = []
            for i, e in enumerate(exp_tuple):
                if e > 0:
                    term = term * (X[:, i] ** e)
                    if e == 1:
                        name_parts.append(f"x{i}")
                    else:
                        name_parts.append(f"x{i}^{e}")
            if name_parts:
                terms.append(term)
                term_names.append("*".join(name_parts))
        
        # 4. Trigonometric terms
        # Only for univariate or low-dimensional to keep basis size manageable
        if d <= 3:
            for i in range(d):
                for k in [1, 2, 3]:
                    terms.append(np.sin(np.pi * k * X[:, i]).reshape(-1, 1))
                    term_names.append(f"sin(pi*{k}*x{i})")
                    terms.append(np.cos(np.pi * k * X[:, i]).reshape(-1, 1))
                    term_names.append(f"cos(pi*{k}*x{i})")
        
        # 5. Rational terms (1/x)
        # Only if min value is sufficiently away from 0
        for i in range(d):
            if X[:, i].min() > 0.5: # Safe to compute 1/x
                terms.append((1.0 / X[:, i]).reshape(-1, 1))
                term_names.append(f"1/x{i}")
                
        # Combine into matrix
        # We use sparse matrix if large, but for n=200 and ~50-100 terms, dense is fine and faster for lsqr
        B = np.hstack(terms)
        return B, term_names

    B, term_names = build_basis(X_norm)
    n_terms = B.shape[1]
    
    # We want to solve min ||B*w - y||^2 + lambda * ||w||_1
    # This is LASSO. We can use scipy's lsqr with regularization, but it's L2.
    # For L1, we need a solver. scipy.optimize.minimize with L1 penalty is slow.
    # Alternative: Use Coordinate Descent (sklearn-style) or a simple iterative solver.
    # Since we don't have sklearn, let's implement a simple Soft-Thresholding Coordinate Descent for LASSO.
    
    # Precompute B^T B and B^T y for speed in coordinate descent
    # B is (200, ~100). B^T B is (100, 100). Very fast.
    BtB = B.T @ B
    Bty = B.T @ y_norm
    
    # Initialize weights to 0
    w = np.zeros(n_terms)
    
    # Coordinate Descent for LASSO
    # Objective: 0.5 * ||Bw - y||^2 + alpha * ||w||_1
    # Gradient of smooth part: B^T (Bw - y)
    # Update for w_j: w_j = S(Bty_j - sum_{k!=j} BtB[j,k] w_k, alpha) / BtB[j,j]
    # where S is soft-thresholding.
    
    def soft_threshold(z, lam):
        return np.sign(z) * np.maximum(np.abs(z) - lam, 0)
    
    # We will try a few alpha values to balance sparsity and fit.
    # Since data is noiseless, we want very small alpha, but not zero to allow sparsity.
    # However, if the true function is a single term, alpha should be large enough to zero out others.
    # Let's try a grid of alphas.
    
    best_w = None
    best_train_mse = np.inf
    best_alpha_used = None
    
    # Heuristic for alpha range
    # Max gradient magnitude
    max_grad = np.max(np.abs(Bty))
    alphas = np.linspace(0.001 * max_grad, 0.1 * max_grad, 5)
    
    # Add a very small alpha for dense fit if sparse fails
    alphas = np.append(alphas, 1e-6 * max_grad)
    
    for alpha in alphas:
        if time.time() > deadline - 5:
            break
            
        w = np.zeros(n_terms)
        
        # Coordinate Descent
        # Convergence: max change < tol
        tol = 1e-5
        max_iter = 500
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            max_change = 0
            for j in range(n_terms):
                if BtB[j, j] == 0:
                    continue
                # Calculate current residual contribution for feature j
                # Bty_j - sum_{k!=j} BtB[j,k] w_k
                # We can maintain a residual vector r = y - Bw
                # But updating r at each step is O(N).
                # With N=200, D=100, it's 20k ops per step, 500 steps = 10M ops. Fast.
                
                # Let's use the efficient update with residual
                # Actually, let's just compute the term directly for simplicity first, 
                # but maintaining residual is better.
                
                # For now, direct computation is fine for this size.
                # term = Bty[j]
                # for k in range(n_terms):
                #    if k != j:
                #       term -= BtB[j, k] * w[k]
                
                # Faster: 
                # current_val = Bty[j] - (BtB[j, :] @ w) + BtB[j, j] * w[j]
                # This is O(D) per j, so O(D^2) per iter. 100^2 = 10k. 500 iters = 5M. OK.
                
                current_val = Bty[j] - (BtB[j, :] @ w) + BtB[j, j] * w[j]
                w_new = soft_threshold(current_val, alpha) / BtB[j, j]
                change = np.abs(w_new - w[j])
                if change > max_change:
                    max_change = change
                w[j] = w_new
            
            if max_change < tol:
                break
        
        # Calculate training MSE
        pred = B @ w
        mse = np.mean((pred - y_norm)**2)
        
        if mse < best_train_mse:
            best_train_mse = mse
            best_w = w.copy()
            best_alpha_used = alpha

    if best_w is None:
        # Fallback: Linear Regression (Ridge)
        # Use np.linalg.lstsq
        try:
            best_w, residuals, rank, s = np.linalg.lstsq(B, y_norm, rcond=None)
        except:
            best_w = np.zeros(n_terms)
            best_w[0] = 1.0 # Constant term

    # Final prediction function
    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        X_test_norm = (X_test - X_min) / X_range
        B_test, _ = build_basis(X_test_norm)
        pred_norm = B_test @ best_w
        pred = pred_norm * y_std + y_mean
        return pred

    return predict
# EVOLVE-BLOCK-END
