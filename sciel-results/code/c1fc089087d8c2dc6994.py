"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit(X_train, y_train):
    t0 = time.time()
    DEADLINE = 330.0 - 12.0  # safety margin before the 330s kill

    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float).ravel()
    n, d = X.shape

    # --- Standardize inputs to unit scale for stable kernel math ---
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd

    def predict_raw(Xq, alpha, l):
        Xqs = (np.asarray(Xq, dtype=float) - mu) / sd
        sq = np.sum((Xqs[:, None, :] - Xs[None, :, :]) ** 2, axis=2)
        K = np.exp(-0.5 * (l ** 2) * sq)
        return K @ alpha

    # --- Precompute the base squared-distance matrix (independent of l) ---
    diff = Xs[:, None, :] - Xs[None, :, :]
    sq = np.sum(diff ** 2, axis=2)
    del diff

    # --- Grid search over length-scale l, minimizing train MSE ---
    best_alpha = None
    best_l = None
    best_mse = np.inf
    lam = 1e-6

    for l in (0.15, 0.3, 0.5, 0.8, 1.2, 1.8, 2.8, 4.5, 7.5):
        if time.time() > DEADLINE:
            break
        K = np.exp(-0.5 * (l ** 2) * sq)
        A = K + lam * np.eye(n)
        try:
            alpha = np.linalg.solve(A, y)
        except np.linalg.LinAlgError:
            continue
        pred = K @ alpha
        mse = float(np.mean((pred - y) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha
            best_l = l

    # Fallback: if nothing was solved, use a simple mean predictor
    if best_alpha is None:
        mean_y = float(np.mean(y))
        return lambda X: np.full(len(np.asarray(X)), mean_y, dtype=float)

    l = best_l
    alpha = best_alpha

    def predict(Xq):
        return predict_raw(Xq, alpha, l)

    return predict
# EVOLVE-BLOCK-END
