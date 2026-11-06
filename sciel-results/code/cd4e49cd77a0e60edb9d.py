"""Climate dynamics: unresolved-tendency closure via degree-five polynomial fit."""
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved fast-scale forcing.
    
    Uses the physical antisymmetry: the fast-slow coupling in Lorenz-96 is
    antisymmetric in x (the slow state), so the subgrid tendency should be
    an odd function of x. We restrict to odd powers (x^1, x^3, x^5) which
    reduces parameters from 6 to 3, improving generalization to unseen F=10.
    
    We also try a full degree-5 fit and pick whichever gives better
    cross-validated performance on the training data.
    """
    start_time = time.time()
    DEADLINE = 120.0  # seconds
    
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Remove any NaN/inf if present
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) == 0:
        return np.zeros(6)
    
    def _elapsed_ok():
        return (time.time() - start_time) < (DEADLINE - 2.0)
    
    # Build Vandermonde for odd powers: x^1, x^3, x^5
    V_odd = np.column_stack([x, x**3, x**5])
    
    # Build Vandermonde for full degree 5: x^0, x^1, x^2, x^3, x^4, x^5
    V_full = np.column_stack([np.ones_like(x), x, x**2, x**3, x**4, x**5])
    
    # Simple k-fold cross-validation to choose model
    def _cv_score(V, k=5, seed=42):
        n = len(x)
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n)
        folds = np.array_split(idx, k)
        scores = []
        for i in range(k):
            val_idx = folds[i]
            trn_idx = np.concatenate([folds[j] for j in range(k) if j != i])
            try:
                Vtr = V[trn_idx]
                ytr = y[trn_idx]
                Vva = V[val_idx]
                yva = y[val_idx]
                # Solve least squares
                coeffs, _, _, _ = np.linalg.lstsq(Vtr, ytr, rcond=None)
                pred = Vva @ coeffs
                mse = np.mean((pred - yva)**2)
                scores.append(mse)
            except np.linalg.LinAlgError:
                scores.append(np.inf)
        return np.mean(scores)
    
    # If data is small, skip CV and just use odd model (physically motivated)
    if len(x) < 200 or not _elapsed_ok():
        # Just fit odd model
        try:
            c_odd, _, _, _ = np.linalg.lstsq(V_odd, y, rcond=None)
        except np.linalg.LinAlgError:
            c_odd = np.zeros(3)
        
        result = np.zeros(6)
        result[1] = c_odd[0]
        result[3] = c_odd[1]
        result[5] = c_odd[2]
        return result
    
    # Do CV
    cv_odd = _cv_score(V_odd)
    if not _elapsed_ok():
        # Just use odd model
        try:
            c_odd, _, _, _ = np.linalg.lstsq(V_odd, y, rcond=None)
        except np.linalg.LinAlgError:
            c_odd = np.zeros(3)
        result = np.zeros(6)
        result[1] = c_odd[0]
        result[3] = c_odd[1]
        result[5] = c_odd[2]
        return result
    
    cv_full = _cv_score(V_full)
    
    if not _elapsed_ok():
        # Default to odd model (physically motivated)
        try:
            c_odd, _, _, _ = np.linalg.lstsq(V_odd, y, rcond=None)
        except np.linalg.LinAlgError:
            c_odd = np.zeros(3)
        result = np.zeros(6)
        result[1] = c_odd[0]
        result[3] = c_odd[1]
        result[5] = c_odd[2]
        return result
    
    # Choose the better model
    if cv_odd <= cv_full:
        try:
            c_odd, _, _, _ = np.linalg.lstsq(V_odd, y, rcond=None)
        except np.linalg.LinAlgError:
            c_odd = np.zeros(3)
        result = np.zeros(6)
        result[1] = c_odd[0]
        result[3] = c_odd[1]
        result[5] = c_odd[2]
        return result
    else:
        try:
            c_full, _, _, _ = np.linalg.lstsq(V_full, y, rcond=None)
        except np.linalg.LinAlgError:
            c_full = np.zeros(6)
        return c_full
