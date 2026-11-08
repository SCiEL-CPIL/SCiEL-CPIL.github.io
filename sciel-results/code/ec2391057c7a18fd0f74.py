<<<<<<< SEARCH
def _polish(step, start_time, deadline):
    """Exact max-based single-flip local search to polish the true C3."""
    x = step.copy()
    best_c3 = _compute_c3(x)
    rng = np.random.default_rng(777)
    no_improve = 0
    patience = 500

    while time.time() < deadline:
        i = rng.integers(N)
        x[i] = -x[i]
        c3 = _compute_c3(x)
        if c3 < best_c3:
            best_c3 = c3
            no_improve = 0
        else:
            x[i] = -x[i]
            no_improve += 1
        if no_improve >= patience:
            break

    return x, best_c3
=======
def _polish(step, start_time, deadline):
    """Exact max-based local search with single and double flips to polish the true C3."""
    x = step.copy()
    best_c3 = _compute_c3(x)
    rng = np.random.default_rng(777)
    no_improve = 0
    patience = 200

    while time.time() < deadline:
        # Alternate between single and double flips
        if rng.random() < 0.6:
            n_flips = 1
        else:
            n_flips = 2
        
        flip_indices = rng.choice(N, size=n_flips, replace=False)
        
        # Apply flips
        x[flip_indices] = -x[flip_indices]
        c3 = _compute_c3(x)
        if c3 < best_c3:
            best_c3 = c3
            no_improve = 0
        else:
            x[flip_indices] = -x[flip_indices]
            no_improve += 1
        if no_improve >= patience:
            break

    return x, best_c3
>>>>>>> REPLACE

