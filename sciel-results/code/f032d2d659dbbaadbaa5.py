# EVOLVE-BLOCK-START
import numpy as np
import time


def _entropy_from_counts(counts):
    """Shannon entropy in bits from a count array (zeros ignored)."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    return -np.sum(probs * np.log2(probs))


def _compute_ratio(P):
    """
    Compute D/M where:
      D = H_{-1} = entropy of distribution of (x - y)
      M = max(H_0, H_1, H_2) = max entropy of x, x+y, x+2y
    P[y, x] is the joint probability.
    Returns the ratio, or 0 if invalid.
    """
    if P.shape != (10, 10):
        return 0.0
    if not np.all(np.isfinite(P)):
        return 0.0
    if np.any(P < -1e-9):
        return 0.0
    total = P.sum()
    if total <= 1e-9:
        return 0.0

    P = P.copy()
    P[P < 0] = 0.0
    P = P / P.sum()

    # x is column index (0..9), y is row index (0..9)
    # P[y, x]
    y_idx, x_idx = np.indices((10, 10))

    # D: distribution of x - y
    val_minus1 = x_idx - y_idx  # range -9..9
    # Count weights for each value
    min_val = val_minus1.min()
    max_val = val_minus1.max()
    counts_minus1 = np.zeros(max_val - min_val + 1)
    for v in range(min_val, max_val + 1):
        mask = (val_minus1 == v)
        counts_minus1[v - min_val] = P[mask].sum()
    H_minus1 = _entropy_from_counts(counts_minus1)

    # H_0: distribution of x
    counts_x = np.zeros(10)
    for x in range(10):
        counts_x[x] = P[:, x].sum()
    H_0 = _entropy_from_counts(counts_x)

    # H_1: distribution of x + y
    val_plus1 = x_idx + y_idx  # range 0..18
    min_val1 = val_plus1.min()
    max_val1 = val_plus1.max()
    counts_plus1 = np.zeros(max_val1 - min_val1 + 1)
    for v in range(min_val1, max_val1 + 1):
        mask = (val_plus1 == v)
        counts_plus1[v - min_val1] = P[mask].sum()
    H_1 = _entropy_from_counts(counts_plus1)

    # H_2: distribution of x + 2y
    val_plus2 = x_idx + 2 * y_idx  # range 0..27
    min_val2 = val_plus2.min()
    max_val2 = val_plus2.max()
    counts_plus2 = np.zeros(max_val2 - min_val2 + 1)
    for v in range(min_val2, max_val2 + 1):
        mask = (val_plus2 == v)
        counts_plus2[v - min_val2] = P[mask].sum()
    H_2 = _entropy_from_counts(counts_plus2)

    M = max(H_0, H_1, H_2)
    if M <= 1e-9:
        return 0.0

    return H_minus1 / M


def _fast_ratio(P):
    """
    Faster computation using vectorized bincount-style approach.
    """
    if P.shape != (10, 10):
        return 0.0
    if not np.all(np.isfinite(P)):
        return 0.0
    if np.any(P < -1e-9):
        return 0.0
    total = P.sum()
    if total <= 1e-9:
        return 0.0

    P = P.copy()
    P[P < 0] = 0.0
    P = P / P.sum()

    y_idx, x_idx = np.indices((10, 10))

    # x - y: range -9..9, offset by 9 to get 0..18
    v_minus1 = (x_idx - y_idx + 9).flatten()
    counts_minus1 = np.bincount(v_minus1, weights=P.flatten(), minlength=19)
    H_minus1 = _entropy_from_counts(counts_minus1)

    # x: range 0..9
    v_x = x_idx.flatten()
    counts_x = np.bincount(v_x, weights=P.flatten(), minlength=10)
    H_0 = _entropy_from_counts(counts_x)

    # x + y: range 0..18
    v_plus1 = (x_idx + y_idx).flatten()
    counts_plus1 = np.bincount(v_plus1, weights=P.flatten(), minlength=19)
    H_1 = _entropy_from_counts(counts_plus1)

    # x + 2y: range 0..27
    v_plus2 = (x_idx + 2 * y_idx).flatten()
    counts_plus2 = np.bincount(v_plus2, weights=P.flatten(), minlength=28)
    H_2 = _entropy_from_counts(counts_plus2)

    M = max(H_0, H_1, H_2)
    if M <= 1e-9:
        return 0.0

    return H_minus1 / M


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Returns a 10x10 array P of finite weights, P[y, x] for integers x, y in 0..9.
    Optimizes the ratio D/M where D = H_{-1} and M = max(H_0, H_1, H_2).
    """
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin before 630s limit

    # Start with uniform distribution
    n = 10
    best_P = np.ones((n, n), dtype=np.float64) / (n * n)
    best_score = _fast_ratio(best_P)

    # Strategy: concentrate mass along lines of constant x - y to maximize H_{-1},
    # but spread across different x values to keep H_0, H_1, H_2 manageable.
    # Actually, to maximize D/M, we want D large and M small.
    # D is entropy of x-y. To make D large, we want x-y to be spread out uniformly.
    # M is max of entropies of x, x+y, x+2y. To make M small, we want these to be concentrated.
    
    # Let's think: if we put all mass on a single diagonal (x-y = c), then D = 0. Bad.
    # If we put mass uniformly on all diagonals, D is max (log2(19) if all 19 values used, but only 10x10 grid).
    # The values of x-y range from -9 to 9, so 19 possible values.
    # But in a 10x10 grid, not all 19 values have the same number of cells.
    
    # Let's try a distribution that is uniform over cells where x-y takes each value with equal probability.
    # i.e., for each d in -9..9, assign equal total probability 1/19 to all cells with x-y=d,
    # and within those cells, distribute uniformly.
    
    # Let's compute this distribution.
    P_candidate = np.zeros((n, n), dtype=np.float64)
    y_idx, x_idx = np.indices((n, n))
    diff = x_idx - y_idx  # range -9..9
    
    for d in range(-9, 10):
        mask = (diff == d)
        num_cells = mask.sum()
        if num_cells > 0:
            P_candidate[mask] = 1.0 / (19.0 * num_cells)
    
    P_candidate = P_candidate / P_candidate.sum()
    score_candidate = _fast_ratio(P_candidate)
    
    if score_candidate > best_score:
        best_P = P_candidate.copy()
        best_score = score_candidate

    # Try another strategy: concentrate on a subset of diagonals
    # Let's try to maximize entropy of x-y while minimizing entropy of x, x+y, x+2y.
    
    # Another idea: use a distribution where x is concentrated (low H_0) but x-y is spread (high H_{-1}).
    # If x is fixed, then x-y = x-y, so as y varies, x-y varies. But if x is fixed to a single value,
    # then x+y = x+y and x+2y = x+2y also vary with y.
    
    # Let's try: put mass on cells where x is in a small set, but y is spread.
    # For example, x in {0, 1, ..., 9} but with specific weights.
    
    # Let's do a local search / optimization.
    # Use a simple hill-climbing approach.
    
    # First, let's try some structured distributions.
    
    # Distribution 1: Uniform (already done)
    # Distribution 2: Equal weight per diagonal (already done)
    
    # Distribution 3: Concentrate on x=0 column
    P_col0 = np.zeros((n, n), dtype=np.float64)
    P_col0[:, 0] = 1.0 / n
    score_col0 = _fast_ratio(P_col0)
    if score_col0 > best_score:
        best_P = P_col0.copy()
        best_score = score_col0

    # Distribution 4: Concentrate on a few columns
    for num_cols in [2, 3, 4, 5]:
        for col_start in range(0, n - num_cols + 1):
            P_test = np.zeros((n, n), dtype=np.float64)
            P_test[:, col_start:col_start+num_cols] = 1.0 / (n * num_cols)
            score_test = _fast_ratio(P_test)
            if score_test > best_score:
                best_P = P_test.copy()
                best_score = score_test

    # Distribution 5: Weighted diagonals - try to find optimal diagonal weights
    # We want to maximize H(p_d) where p_d is the probability of diagonal d,
    # subject to constraints from H_0, H_1, H_2.
    
    # Let's do a numerical optimization using scipy if available, or simple gradient-free search.
    
    # Try a random search with local improvement
    rng = np.random.RandomState(42)
    
    # Parameterize by diagonal weights and within-diagonal distributions
    # For simplicity, let's just do a random search over the 100-cell probability vector.
    
    num_iterations = 50000
    current_P = best_P.copy()
    current_score = best_score
    
    for i in range(num_iterations):
        if time.time() > deadline - 30:
            break
        
        # Random perturbation
        P_test = current_P.copy()
        # Add small random noise
        noise = rng.normal(0, 0.01, size=(n, n))
        P_test += noise
        P_test[P_test < 0] = 0
        P_test_sum = P_test.sum()
        if P_test_sum <= 1e-9:
            continue
        P_test = P_test / P_test_sum
        
        score_test = _fast_ratio(P_test)
        if score_test > current_score:
            current_P = P_test
            current_score = score_test
            if current_score > best_score:
                best_P = current_P.copy()
                best_score = current_score
    
    # Try a more structured approach: optimize diagonal weights
    # Let w_d be the weight for diagonal d (d = x-y, range -9..9)
    # Within each diagonal, distribute uniformly among cells.
    
    # Build the cell-to-diagonal mapping
    diag_of_cell = np.zeros((n, n), dtype=int)
    for y in range(n):
        for x in range(n):
            d = x - y
            diag_of_cell[y, x] = d + 9  # offset to 0..18
    
    # Count cells per diagonal
    cells_per_diag = np.zeros(19, dtype=int)
    for d in range(19):
        cells_per_diag[d] = np.sum(diag_of_cell == d)
    
    # Optimize w_d to maximize H(w) / max(H_0(w), H_1(w), H_2(w))
    # where H(w) is entropy of w, and H_0, H_1, H_2 are entropies of the induced distributions.
    
    # This is a complex optimization. Let's try a simple approach:
    # Start with uniform w_d and do local search.
    
    w = np.ones(19) / 19.0
    best_w = w.copy()
    best_w_score = 0.0
    
    def score_from_weights(w):
        """Compute score from diagonal weights, with uniform distribution within each diagonal."""
        if np.any(w < 0) or w.sum() <= 1e-9:
            return 0.0
        w = w / w.sum()
        
        # Build P
        P_test = np.zeros((n, n), dtype=np.float64)
        for d in range(19):
            if cells_per_diag[d] > 0 and w[d] > 0:
                mask = (diag_of_cell == d)
                P_test[mask] = w[d] / cells_per_diag[d]
        
        return _fast_ratio(P_test)
    
    best_w_score = score_from_weights(w)
    
    # Local search on w
    for i in range(20000):
        if time.time() > deadline - 20:
            break
        
        w_test = w.copy()
        # Perturb two random diagonals
        d1 = rng.randint(0, 19)
        d2 = rng.randint(0, 19)
        if d1 == d2:
            continue
        delta = rng.uniform(0.01, 0.05)
        w_test[d1] += delta
        w_test[d2] -= delta
        if w_test[d1] < 0 or w_test[d2] < 0:
            continue
        
        score_w = score_from_weights(w_test)
        if score_w > best_w_score:
            w = w_test
            best_w_score = score_w
    
    # Reconstruct P from best w
    if best_w_score > best_score:
        P_from_w = np.zeros((n, n), dtype=np.float64)
        w_norm = best_w / best_w.sum()
        for d in range(19):
            if cells_per_diag[d] > 0 and w_norm[d] > 0:
                mask = (diag_of_cell == d)
                P_from_w[mask] = w_norm[d] / cells_per_diag[d]
        P_from_w = P_from_w / P_from_w.sum()
        best_P = P_from_w
        best_score = best_w_score
    
    # Final cleanup: ensure validity
    best_P[best_P < 0] = 0
    total = best_P.sum()
    if total > 1e-9:
        best_P = best_P / total
    
    return best_P


# EVOLVE-BLOCK-END
