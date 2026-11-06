import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = 115.0 # Leave 5s margin
    
    x = np.asarray(training_x, dtype=float)
    y = np.asarray(training_subgrid_tendency, dtype=float)
    
    if len(x) == 0:
        return np.zeros(6)
        
    # Create Vandermonde matrix for x (degrees 0 to 5)
    V = np.vander(x, N=6, increasing=True)
    
    # Cross-validation parameters
    lambdas = np.logspace(-8, 0, 15) # Wider range
    k = 5
    n = len(x)
    
    # Shuffle indices
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_size = n // k
    if fold_size == 0:
        # If data is too small for CV, just pick a moderate lambda
        best_lambda = 1e-4
    else:
        folds = [indices[i*fold_size:(i+1)*fold_size] for i in range(k)]
        # Handle remainder
        if n % k != 0:
            folds[-1] = np.concatenate([folds[-1], indices[k*fold_size:]])
        
        best_lambda = lambdas[0]
        best_cv_score = np.inf
        
        for lam in lambdas:
            if time.time() > deadline - 10.0:
                break # Stop searching if running out of time
            cv_errors = []
            for i in range(k):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
                
                V_train = V[train_idx]
                y_train = y[train_idx]
                V_val = V[val_idx]
                y_val = y[val_idx]
                
                # Ridge regression
                try:
                    A = V_train.T @ V_train + lam * np.eye(6)
                    b = V_train.T @ y_train
                    c = np.linalg.solve(A, b)
                    
                    pred_val = V_val @ c
                    mse = np.mean((pred_val - y_val)**2)
                    cv_errors.append(mse)
                except np.linalg.LinAlgError:
                    cv_errors.append(np.inf)
            
            mean_cv_error = np.mean(cv_errors)
            if mean_cv_error < best_cv_score:
                best_cv_score = mean_cv_error
                best_lambda = lam
                
    # Final fit
    A = V.T @ V + best_lambda * np.eye(6)
    b = V.T @ y
    try:
        c = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        c = np.linalg.lstsq(A, b, rcond=None)[0]
        
    # Ensure finite
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    
    return c
