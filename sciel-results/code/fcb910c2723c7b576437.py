        X_simple = np.column_stack([np.ones(n), x_norm])
        final_ridge_w = np.linalg.lstsq(X_simple, y, rcond=None)[0]
        use_ridge = True
        # comments...
        simple_linear_fallback = True
    else:
        simple_linear_fallback = False
