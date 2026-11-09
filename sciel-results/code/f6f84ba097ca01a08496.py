"""Symbolic regression: Sum-of-Sines (SOS) basis function expansion with
Ridge Regression and Cross-Validation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit(X_train, y_train):
    t0 = time.time()
    DEADLINE = 315.0  # safety margin before 330s kill

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape

    # Scale inputs to [0, 1] for consistent frequency interpretation
    xmin = X.min(axis=0)
    xmax = X.max(axis=0)
    xrange = xmax - xmin
    xrange[xrange < 1e-12] = 1.0

    def scale_inputs(Xraw):
        Xr = np.asarray(Xraw, dtype=np.float64)
        return (Xr - xmin) / xrange

    Xs = scale_inputs(X)
    y_mean = y.mean()
    yc = y - y_mean
    yvar = max(np.var(y), 1e-12)

    def build_sos_features(X_scaled, max_freq):
        """
        Build Sum-of-Sines features.
        For each dimension i, include sin(k * 2*pi * x_i) and cos(k * 2*pi * x_i)
        for k = 1 to max_freq.
        Also include a bias term.
        """
        n_pts = X_scaled.shape[0]
        feats = [np.ones((n_pts, 1))]
        
        for i in range(d):
            xi = X_scaled[:, i]
            for k in range(1, max_freq + 1):
                angle = 2 * np.pi * k * xi
                feats.append(np.sin(angle).reshape(-1, 1))
                feats.append(np.cos(angle).reshape(-1, 1))
                
        # If dimension is small, add pairwise interactions of the base SOS features?
        # To keep it fast and robust, we stick to univariate SOS for now.
        # If d <= 3, we might add simple interactions like sin(x1)*sin(x2)
        if d <= 3 and max_freq >= 2:
            # Add interactions for low frequencies only to control dimensionality
            for i in range(d):
                for j in range(i + 1, d):
                    xi = X_scaled[:, i]
                    xj = X_scaled[:, j]
                    # Just the first harmonic interaction
                    feats.append((np.sin(2 * np.pi * xi) * np.sin(2 * np.pi * xj)).reshape(-1, 1))
                    feats.append((np.sin(2 * np.pi * xi) * np.cos(2 * np.pi * xj)).reshape(-1, 1))
                    feats.append((np.cos(2 * np.pi * xi) * np.sin(2 * np.pi * xj)).reshape(-1, 1))
                    feats.append((np.cos(2 * np.pi * xi) * np.cos(2 * np.pi * xj)).reshape(-1, 1))

        return np.hstack(feats)

    # Hyperparameters to search
    cand_max_freqs = [1, 2, 3, 4, 5, 6, 8]
    cand_alphas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    # Cross-validation setup
    rng_cv = np.random.RandomState(0)
    idx = np.arange(n)
    perm = rng_cv.permutation(n)
    n_folds = 5
    folds = np.array_split(perm, n_folds)

    def evaluate_hyp(max_freq, alpha):
        """Return mean CV normalized MSE."""
        total = 0.0
        cnt = 0
        for fo in folds:
            tr = np.setdiff1d(idx, fo, assume_unique=False)
            va = fo
            if len(tr) < 5:
                continue
            
            Xtr_s = Xs[tr]
            ytr_c = yc[tr]
            Xva_s = Xs[va]
            yva_c = yc[va]
            
            Ftr = build_sos_features(Xtr_s, max_freq)
            Fva = build_sos_features(Xva_s, max_freq)
            
            n_feat = Ftr.shape[1]
            reg_matrix = Ftr.T @ Ftr + alpha * np.eye(n_feat)
            try:
                w = np.linalg.solve(reg_matrix, Ftr.T @ ytr_c)
            except np.linalg.LinAlgError:
                try:
                    w = np.linalg.lstsq(reg_matrix, Ftr.T @ ytr_c, rcond=None)[0]
                except Exception:
                    return 1e12
            
            pred = Fva @ w
            mse = np.mean((pred - yva_c) ** 2)
            total += mse / yvar
            cnt += 1
        return total / max(cnt, 1)

    best_cv = np.inf
    best_max_freq = 2
    best_alpha = 1e-4

    # Grid search
    for mf in cand_max_freqs:
        if time.time() - t0 > DEADLINE - 30:
            break
        for alpha in cand_alphas:
            if time.time() - t0 > DEADLINE - 10:
                break
            cv = evaluate_hyp(mf, alpha)
            if cv < best_cv:
                best_cv = cv
                best_max_freq = mf
                best_alpha = alpha

    # Final fit
    Ftr = build_sos_features(Xs, best_max_freq)
    n_feat = Ftr.shape[1]
    reg_matrix = Ftr.T @ Ftr + best_alpha * np.eye(n_feat)
    try:
        w = np.linalg.solve(reg_matrix, Ftr.T @ yc)
    except np.linalg.LinAlgError:
        try:
            w = np.linalg.lstsq(reg_matrix, Ftr.T @ yc, rcond=None)[0]
        except Exception:
            # Fallback to mean
            return lambda Xraw: np.full(len(np.asarray(Xraw, dtype=np.float64)), y_mean)

    def predict(Xraw):
        Xr = scale_inputs(Xraw)
        Fva = build_sos_features(Xr, best_max_freq)
        return Fva @ w + y_mean

    return predict
# EVOLVE-BLOCK-END
