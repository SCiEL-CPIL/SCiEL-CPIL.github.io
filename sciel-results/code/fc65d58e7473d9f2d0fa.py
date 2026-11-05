import numpy as np
import time
from math import comb

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    DEADLINE = 115.0

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 6:
        return np.zeros(6)

    # Map x to [-1, 1]
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max - x_min < 1e-12:
        # Constant x, return linear fit if possible or zero
        # If x is constant, only c0 matters? No, if x is constant, x^i is constant.
        # But usually x varies. If not, we can't fit higher orders meaningfully.
        # Return mean y for c0, 0 for rest?
        # Actually, if x is constant, the polynomial is just a constant function.
        return np.array([np.mean(y), 0, 0, 0, 0, 0])

    alpha = 2.0 / (x_max - x_min)
    beta = -1.0 - 2.0 * x_min / (x_max - x_min) # t = alpha * x + beta
    # Check: if x=x_min, t = alpha*x_min + beta = 2*x_min/(range) - 1 - 2*x_min/range = -1. Correct.
    # if x=x_max, t = 2*x_max/range - 1 - 2*x_min/range = 2*(x_max-x_min)/range - 1 = 2 - 1 = 1. Correct.

    t = alpha * x + beta

    # Build Chebyshev design matrix
    # T_k(t)
    # Use np.polynomial.chebyshev.chebvander
    C = np.polynomial.chebyshev.chebvander(t, 6) # Shape (n, 6)
    # Columns are T_0, T_1, ..., T_5

    # Weight matrix for regularization
    # Penalize higher degrees more
    weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    W = np.diag(weights ** 2) # Quadratic penalty on degree

    # K-Fold Cross Validation for Lambda
    # Use 5 folds if n is large enough
    k_folds = 5
    if n < k_folds * 10:
        k_folds = max(2, n // 10)
    
    lambdas = np.logspace(-8, 2, 10)
    
    best_lambda = 1.0
    min_val_mse = np.inf
    
    if n > 50 and (time.time() - start_time) < (DEADLINE - 10.0):
        indices = np.arange(n)
        np.random.shuffle(indices)
        fold_sizes = np.full(k_folds, n // k_folds)
        fold_sizes[:n % k_folds] += 1
        folds = np.split(indices, np.cumsum(fold_sizes)[:-1])
        
        for lam in lambdas:
            if time.time() > DEADLINE - 5.0:
                break
            val_mses = []
            for i in range(k_folds):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != i])
                
                C_train = C[train_idx]
                y_train = y[train_idx]
                C_val = C[val_idx]
                y_val = y[val_idx]
                
                # Solve (C^T C + lambda W) a = C^T y
                A = C_train.T @ C_train + lam * W
                b = C_train.T @ y_train
                
                try:
                    a = np.linalg.solve(A, b)
                    pred_val = C_val @ a
                    mse = np.mean((pred_val - y_val) ** 2)
                    val_mses.append(mse)
                except np.linalg.LinAlgError:
                    val_mses.append(np.inf)
            
            mean_val_mse = np.mean(val_mses)
            if mean_val_mse < min_val_mse:
                min_val_mse = mean_val_mse
                best_lambda = lam

    # Final fit with best_lambda
    A = C.T @ C + best_lambda * W
    b = C.T @ y
    try:
        a_cheb = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Fallback to unregularized or smaller lambda
        a_cheb = np.linalg.lstsq(C, y, rcond=None)[0]

    # Convert Chebyshev coefficients a_cheb to Monomial coefficients in t
    # P(t) = sum a_k T_k(t)
    # cheb2poly converts Chebyshev coeffs to Power coeffs in t
    b_t = np.polynomial.chebyshev.cheb2poly(a_cheb)
    # b_t[j] is coeff for t^j

    # Convert from t-basis to x-basis
    # t = alpha * x + beta
    # P(x) = sum_{j=0}^5 b_t[j] * (alpha * x + beta)^j
    # (alpha * x + beta)^j = sum_{m=0}^j binom(j, m) * alpha^m * beta^(j-m) * x^m
    c_x = np.zeros(6)
    for j in range(6):
        bj = b_t[j]
        if bj == 0:
            continue
        for m in range(j + 1):
            c_x[m] += bj * comb(j, m) * (alpha ** m) * (beta ** (j - m))

    return c_x
