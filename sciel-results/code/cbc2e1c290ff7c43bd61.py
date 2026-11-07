"""
Strategy: Multi-Resolution Piecewise Cubic Spline Interpolation.
Instead of fitting a global basis (poly/fourier) or a neural net, we fit a local, high-capacity model
by partitioning the input domain into bins and fitting a cubic spline in each bin.
We search over a grid of bin counts to find the optimal resolution. This approach
can fit arbitrary smooth functions (including discontinuities) with near-zero bias
given sufficient resolution, and is extremely robust to noise (since 200 points is
plenty for cubic splines).

Key improvements over previous spline attempts:
1. Use scipy.interpolate.CubicSpline for stable, high-quality local fits.
2. Search over a range of bin counts (from coarse to fine) to balance bias/variance.
3. For each bin count, evaluate the model on a holdout set to select the best resolution.
4. Since data is noiseless, we can afford fine-grained bins without overfitting to noise.
5. This is structurally different from MLP (iterative optimization) and GP (kernel-based).
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import CubicSpline

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290  # Leave 10s buffer
    
    X = X_train[:, 0].reshape(-1)
    y = y_train
    
    n = len(X)
    
    # Sort by input for consistent binning
    sort_idx = np.argsort(X)
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]
    
    # Determine input range from training data, but extend slightly for test points
    x_min = X_sorted.min()
    x_max = X_sorted.max()
    
    # If range is very small, handle gracefully
    if x_max - x_min < 1e-8:
        x_min -= 1.0
        x_max += 1.0
    
    # Search over different bin counts
    # With 200 points, we can try bin counts from ~5 to ~50
    # Each bin will have ~200/bin_count points on average
    bin_counts = [5, 8, 10, 15, 20, 25, 30, 40, 50]
    
    # Use a holdout set for model selection
    # Since data is noiseless, we can use a small holdout
    val_size = max(20, int(0.1 * n))
    val_idx = np.random.choice(n, val_size, replace=False)
    train_idx = np.setdiff1d(np.arange(n), val_idx)
    
    X_val = X[val_idx]
    y_val = y[val_idx]
    X_tr = X[train_idx]
    y_tr = y[train_idx]
    
    best_val_loss = np.inf
    best_bin_count = 10
    best_splines = None
    
    for n_bins in bin_counts:
        if time.time() > deadline - 30:  # Stop early if running out of time
            break
            
        # Create bin edges
        # Use quantile-based binning to ensure roughly equal points per bin
        # This is more robust than uniform binning if data is non-uniformly distributed
        edges = np.quantile(X_sorted, np.linspace(0, 1, n_bins + 1))
        
        # Ensure unique edges (handle case where quantiles might be identical)
        for i in range(len(edges) - 1):
            if edges[i+1] - edges[i] < 1e-10:
                edges[i+1] = edges[i] + 1e-10
        
        # Fit a cubic spline in each bin
        splines = []
        valid = True
        
        for i in range(n_bins):
            # Get points in this bin
            mask = (X_tr >= edges[i]) & (X_tr < edges[i+1])
            if i == n_bins - 1:  # Last bin includes the right edge
                mask = (X_tr >= edges[i]) & (X_tr <= edges[i+1])
            
            x_bin = X_tr[mask]
            y_bin = y_tr[mask]
            
            # Need at least 4 points for cubic spline
            if len(x_bin) < 4:
                # Fall back to linear interpolation or lower-order polynomial
                if len(x_bin) >= 2:
                    # Linear fit
                    from numpy.polynomial import polynomial as P
                    coeffs = np.polyfit(x_bin, y_bin, 1)
                    def make_linear(c, x_min, x_max):
                        def predict(x):
                            return c[0] * x + c[1]
                        return predict
                    splines.append(('linear', coeffs))
                else:
                    # Constant
                    c = np.mean(y_bin) if len(y_bin) > 0 else 0.0
                    splines.append(('constant', c))
            else:
                # Sort within bin (should already be sorted, but ensure)
                bin_sort_idx = np.argsort(x_bin)
                x_bin_sorted = x_bin[bin_sort_idx]
                y_bin_sorted = y_bin[bin_sort_idx]
                
                # Check for duplicate x values
                if np.any(np.diff(x_bin_sorted) < 1e-10):
                    # Average duplicates
                    unique_x, inverse, counts = np.unique(x_bin_sorted, return_inverse=True, return_counts=True)
                    if len(unique_x) < 4:
                        if len(unique_x) >= 2:
                            y_avg = np.array([np.mean(y_bin_sorted[inverse == i]) for i in range(len(unique_x))])
                            coeffs = np.polyfit(unique_x, y_avg, 1)
                            splines.append(('linear', coeffs))
                        else:
                            c = np.mean(y_bin) if len(y_bin) > 0 else 0.0
                            splines.append(('constant', c))
                    else:
                        y_avg = np.array([np.mean(y_bin_sorted[inverse == i]) for i in range(len(unique_x))])
                        try:
                            cs = CubicSpline(unique_x, y_avg, bc_type='not-a-knot')
                            splines.append(('cubic', cs))
                        except Exception:
                            coeffs = np.polyfit(unique_x, y_avg, 1)
                            splines.append(('linear', coeffs))
                else:
                    try:
                        cs = CubicSpline(x_bin_sorted, y_bin_sorted, bc_type='not-a-knot')
                        splines.append(('cubic', cs))
                    except Exception:
                        coeffs = np.polyfit(x_bin_sorted, y_bin_sorted, 1)
                        splines.append(('linear', coeffs))
        
        # Evaluate on validation set
        pred_val = np.zeros(len(X_val))
        for i in range(len(X_val)):
            x_test = X_val[i]
            
            # Find which bin this point falls into
            if x_test < edges[0]:
                bin_idx = 0
            elif x_test > edges[-1]:
                bin_idx = n_bins - 1
            else:
                bin_idx = np.searchsorted(edges, x_test, side='right') - 1
                bin_idx = max(0, min(bin_idx, n_bins - 1))
            
            spline_type, spline_obj = splines[bin_idx]
            
            if spline_type == 'cubic':
                pred_val[i] = spline_obj(x_test)
            elif spline_type == 'linear':
                pred_val[i] = spline_obj[0] * x_test + spline_obj[1]
            else:  # constant
                pred_val[i] = spline_obj
        
        val_loss = np.mean((pred_val - y_val) ** 2)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_bin_count = n_bins
            best_splines = splines
            best_edges = edges.copy()
    
    # If no valid model was found, fall back to a global cubic spline
    if best_splines is None:
        try:
            cs = CubicSpline(X_sorted, y_sorted, bc_type='not-a-knot')
            def predict(X):
                x = X.flatten()
                return cs(x).reshape(X.shape)
            return predict
        except Exception:
            # Ultimate fallback: nearest neighbor
            def predict(X):
                x = X.flatten()
                preds = np.zeros(len(x))
                for i, xi in enumerate(x):
                    dists = np.abs(X_sorted - xi)
                    nearest = np.argmin(dists)
                    preds[i] = y_sorted[nearest]
                return preds.reshape(X.shape)
            return predict
    
    # Refit on ALL training data with the best bin count
    edges = best_edges
    splines = []
    
    for i in range(len(edges) - 1):
        mask = (X_sorted >= edges[i]) & (X_sorted < edges[i+1])
        if i == len(edges) - 2:  # Last bin includes the right edge
            mask = (X_sorted >= edges[i]) & (X_sorted <= edges[i+1])
        
        x_bin = X_sorted[mask]
        y_bin = y_sorted[mask]
        
        if len(x_bin) < 4:
            if len(x_bin) >= 2:
                coeffs = np.polyfit(x_bin, y_bin, 1)
                splines.append(('linear', coeffs))
            else:
                c = np.mean(y_bin) if len(y_bin) > 0 else 0.0
                splines.append(('constant', c))
        else:
            # Check for duplicate x values
            if np.any(np.diff(x_bin) < 1e-10):
                unique_x, inverse, counts = np.unique(x_bin, return_inverse=True, return_counts=True)
                if len(unique_x) < 4:
                    if len(unique_x) >= 2:
                        y_avg = np.array([np.mean(y_bin[inverse == i]) for i in range(len(unique_x))])
                        coeffs = np.polyfit(unique_x, y_avg, 1)
                        splines.append(('linear', coeffs))
                    else:
                        c = np.mean(y_bin) if len(y_bin) > 0 else 0.0
                        splines.append(('constant', c))
                else:
                    y_avg = np.array([np.mean(y_bin[inverse == i]) for i in range(len(unique_x))])
                    try:
                        cs = CubicSpline(unique_x, y_avg, bc_type='not-a-knot')
                        splines.append(('cubic', cs))
                    except Exception:
                        coeffs = np.polyfit(unique_x, y_avg, 1)
                        splines.append(('linear', coeffs))
            else:
                try:
                    cs = CubicSpline(x_bin, y_bin, bc_type='not-a-knot')
                    splines.append(('cubic', cs))
                except Exception:
                    coeffs = np.polyfit(x_bin, y_bin, 1)
                    splines.append(('linear', coeffs))
    
    def predict(X):
        x = X.flatten()
        preds = np.zeros(len(x))
        
        for i, xi in enumerate(x):
            # Find which bin this point falls into
            if xi < edges[0]:
                bin_idx = 0
            elif xi > edges[-1]:
                bin_idx = len(splines) - 1
            else:
                bin_idx = np.searchsorted(edges, xi, side='right') - 1
                bin_idx = max(0, min(bin_idx, len(splines) - 1))
            
            spline_type, spline_obj = splines[bin_idx]
            
            if spline_type == 'cubic':
                preds[i] = spline_obj(xi)
            elif spline_type == 'linear':
                preds[i] = spline_obj[0] * xi + spline_obj[1]
            else:  # constant
                preds[i] = spline_obj
        
        return preds.reshape(X.shape)
    
    return predict
# EVOLVE-BLOCK-END
