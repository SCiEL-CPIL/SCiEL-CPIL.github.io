import numpy as np
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280.0

    x_train = X_train.flatten()
    y_train = y_train.flatten()
    n = len(x_train)

    # Sort data
    sort_idx = np.argsort(x_train)
    x_train = x_train[sort_idx]
    y_train = y_train[sort_idx]

    # Pre-compute squared distances for training set
    # D is n x n
    diff = x_train[:, None] - x_train[None, :]
    D_train = diff ** 2

    # Hyperparameter grid
    l_grid = np.logspace(np.log10(0.05), np.log10(20.0), 20)
    alpha_grid = np.logspace(-6, 2, 20)

    # Deterministic CV folds
    k_folds = 5
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, k_folds)

    def get_kernel(D, l):
        return np.exp(-D / (2 * l**2))

    best_mse = float('inf')
    best_l = l_grid[0]
    best_alpha = alpha_grid[0]

    # Pre-compute fold masks
    fold_masks = []
    for i in range(k_folds):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != i])
        fold_masks.append((train_idx, test_idx))

    for l in l_grid:
        for alpha in alpha_grid:
            if time.time() > deadline - 10.0:
                break

            mse_sum = 0.0
            for train_idx, test_idx in fold_masks:
                x_tr = x_train[train_idx]
                y_tr = y_train[train_idx]
                x_te = x_train[test_idx]
                y_te = y_train[test_idx]

                # Extract sub-matrix for training
                # D_train[train_idx][:, train_idx]
                K_train = get_kernel(D_train[np.ix_(train_idx, train_idx)], l)
                
                # Compute test-train kernel
                # We need distances between test points and training points
                # D_test_train = (x_te[:, None] - x_tr[None, :]) ** 2
                diff_te_tr = x_te[:, None] - x_tr[None, :]
                D_test_train = diff_te_tr ** 2
                K_test_train = get_kernel(D_test_train, l)

                K_reg = K_train + alpha * np.eye(len(x_tr))
                
                try:
                    w = np.linalg.solve(K_reg, y_tr)
                    y_pred = K_test_train @ w
                    mse = np.mean((y_pred - y_te) ** 2)
                    mse_sum += mse
                except np.linalg.LinAlgError:
                    mse_sum = float('inf')
                    break
                    
            if mse_sum < best_mse:
                best_mse = mse_sum / k_folds
                best_l = l
                best_alpha = alpha

        if time.time() > deadline - 10.0:
            break

    # Final fit
    K_train = get_kernel(D_train, best_l)
    K_reg = K_train + best_alpha * np.eye(n)
    
    # Add small jitter for numerical stability
    K_reg += 1e-10 * np.eye(n)
    
    try:
        w = np.linalg.solve(K_reg, y_train)
    except np.linalg.LinAlgError:
        # Fallback: Use polynomial fit if solve fails
        # This is unlikely but safe
        w = np.linalg.lstsq(np.vander(x_train, 10, increasing=True), y_train, rcond=None)[0]
        # If we fall back to polynomial, the predict function needs to know
        # To keep it simple, assume solve works. If not, return a constant.
        mean_y = np.mean(y_train)
        def predict_const(X):
            return np.full(X.shape[0], mean_y)
        return predict_const

    def predict(X):
        x_test = X.flatten()
        # Compute distances between test and train
        diff = x_test[:, None] - x_train[None, :]
        D_test_train = diff ** 2
        K_test_train = get_kernel(D_test_train, best_l)
        return K_test_train @ w

    return predict
