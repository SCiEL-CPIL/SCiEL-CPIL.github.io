"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_deadline = time.time() + 135  # Safety margin before 150s kill


def _simulate_bin_packing_fast(items, capacity, param_type, param_value):
    """
    Fast simulation using pre-allocated arrays and direct scoring.
    Returns: number of bins used
    """
    n = len(items)
    max_bins = n + 1
    remaining = np.zeros(max_bins, dtype=np.float64)
    n_bins = 0
    
    for item in items:
        if n_bins == 0:
            remaining[0] = capacity - item
            n_bins = 1
            continue
        
        valid = remaining[:n_bins] >= item
        valid_count = np.count_nonzero(valid)
        
        if valid_count == 0:
            remaining[n_bins] = capacity - item
            n_bins += 1
            continue
        
        # Compute scores for valid bins only
        valid_remaining = remaining[:n_bins][valid]
        gap = valid_remaining - item
        
        if param_type == 'alpha':
            scores = -gap ** param_value
        elif param_type == 'target':
            scores = -((gap) - param_value) ** 2
        elif param_type == 'norm_alpha':
            safe_item = max(item, 1.0)
            ratio = gap / safe_item
            scores = -ratio ** param_value
        elif param_type == 'combined':
            alpha, beta = param_value
            fill_frac = valid_remaining / capacity
            scores = -(gap ** alpha) + beta * fill_frac
        elif param_type == 'content_bf':
            beta = param_value
            content = capacity - valid_remaining
            scores = -gap + beta * content
        elif param_type == 'worst_fit':
            scores = valid_remaining
        elif param_type == 'worst_fit_alpha':
            scores = valid_remaining ** param_value
        elif param_type == 'mixed_wf':
            alpha, beta = param_value
            scores = alpha * valid_remaining - beta * gap
        else:
            scores = -gap
        
        best_local = np.argmax(scores)
        valid_idx = np.where(valid)[0]
        best_global = valid_idx[best_local]
        remaining[best_global] -= item
    
    return n_bins


def _l1_lower_bound(items, capacity):
    total = np.sum(items)
    max_item = np.max(items)
    lb1 = int(np.ceil(total / capacity))
    lb2 = int(np.ceil(max_item))
    return max(lb1, lb2)


def _generate_sequences(n_sequences=5, n_items=1000, capacity=100):
    """Generate Weibull sequences for testing."""
    sequences = []
    for seed in range(n_sequences):
        rng = np.random.RandomState(seed)
        items = rng.weibull(2, size=n_items) * 20
        items = items[items <= capacity]
        while len(items) < n_items:
            new_items = rng.weibull(2, size=n_items) * 20
            new_items = new_items[new_items <= capacity]
            items = np.concatenate([items, new_items])[:n_items]
        sequences.append(items)
    return sequences


def _evaluate_params(sequences, capacity, param_type, param_value):
    """Evaluate a parameter set across all sequences. Returns mean excess %."""
    total_excess = 0.0
    for seq in sequences:
        bins_used = _simulate_bin_packing_fast(seq, capacity, param_type, param_value)
        lb = _l1_lower_bound(seq, capacity)
        excess = (bins_used - lb) / lb * 100
        total_excess += excess
    return total_excess / len(sequences)


def _fit_parameters():
    """
    Search over scoring function families with coarse grid + local refinement.
    """
    sequences = _generate_sequences()
    capacity = 100
    
    best_score = float('inf')
    best_params = ('alpha', 1.0)
    
    # === Phase 1: Coarse grid search ===
    
    # Alpha family: score = -(remaining - item)^alpha
    alphas_coarse = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    for alpha in alphas_coarse:
        if time.time() > _deadline - 20:
            break
        score = _evaluate_params(sequences, capacity, 'alpha', alpha)
        if score < best_score:
            best_score = score
            best_params = ('alpha', alpha)
    
    # Target family: score = -((remaining - item) - t)^2
    targets_coarse = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 75.0]
    for t in targets_coarse:
        if time.time() > _deadline - 20:
            break
        score = _evaluate_params(sequences, capacity, 'target', t)
        if score < best_score:
            best_score = score
            best_params = ('target', t)
    
    # Norm alpha family
    norm_alphas = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    for na in norm_alphas:
        if time.time() > _deadline - 20:
            break
        score = _evaluate_params(sequences, capacity, 'norm_alpha', na)
        if score < best_score:
            best_score = score
            best_params = ('norm_alpha', na)
    
    # Combined family: score = -(gap^alpha) + beta * (remaining/capacity)
    for alpha_c in [0.5, 0.75, 1.0, 1.5, 2.0]:
        if time.time() > _deadline - 20:
            break
        for beta_c in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
            if time.time() > _deadline - 20:
                break
            pv = (alpha_c, beta_c)
            score = _evaluate_params(sequences, capacity, 'combined', pv)
            if score < best_score:
                best_score = score
                best_params = ('combined', pv)
    
    # Content-BF family: score = -gap + beta * content
    betas_cb = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    for beta in betas_cb:
        if time.time() > _deadline - 20:
            break
        score = _evaluate_params(sequences, capacity, 'content_bf', beta)
        if score < best_score:
            best_score = score
            best_params = ('content_bf', beta)
    
    # Worst-fit family: score = remaining (prefer emptiest bin)
    score = _evaluate_params(sequences, capacity, 'worst_fit', 1.0)
    if score < best_score:
        best_score = score
        best_params = ('worst_fit', 1.0)
    
    # Worst-fit alpha family: score = remaining^alpha
    wf_alphas = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    for wa in wf_alphas:
        if time.time() > _deadline - 20:
            break
        score = _evaluate_params(sequences, capacity, 'worst_fit_alpha', wa)
        if score < best_score:
            best_score = score
            best_params = ('worst_fit_alpha', wa)
    
    # Mixed worst-fit family: score = alpha * remaining - beta * gap
    for a_wf in [0.5, 1.0, 1.5, 2.0]:
        if time.time() > _deadline - 20:
            break
        for b_wf in [0.0, 0.1, 0.5, 1.0, 2.0]:
            if time.time() > _deadline - 20:
                break
            pv = (a_wf, b_wf)
            score = _evaluate_params(sequences, capacity, 'mixed_wf', pv)
            if score < best_score:
                best_score = score
                best_params = ('mixed_wf', pv)
    
    # === Phase 2: Local refinement around best ===
    if time.time() < _deadline - 15:
        ptype, pval = best_params
        if ptype == 'alpha':
            lo = max(0.05, pval * 0.5)
            hi = pval * 1.5 + 0.5
            n_steps = 20
            for i in range(n_steps + 1):
                if time.time() > _deadline - 10:
                    break
                a = lo + (hi - lo) * i / n_steps
                score = _evaluate_params(sequences, capacity, 'alpha', a)
                if score < best_score:
                    best_score = score
                    best_params = ('alpha', a)
        elif ptype == 'target':
            lo = max(0.0, pval - 15.0)
            hi = pval + 15.0
            n_steps = 20
            for i in range(n_steps + 1):
                if time.time() > _deadline - 10:
                    break
                t = lo + (hi - lo) * i / n_steps
                score = _evaluate_params(sequences, capacity, 'target', t)
                if score < best_score:
                    best_score = score
                    best_params = ('target', t)
        elif ptype == 'norm_alpha':
            lo = max(0.1, pval * 0.5)
            hi = pval * 1.5 + 0.5
            n_steps = 20
            for i in range(n_steps + 1):
                if time.time() > _deadline - 10:
                    break
                na = lo + (hi - lo) * i / n_steps
                score = _evaluate_params(sequences, capacity, 'norm_alpha', na)
                if score < best_score:
                    best_score = score
                    best_params = ('norm_alpha', na)
        elif ptype == 'combined':
            alpha_c, beta_c = pval
            for da in [-0.3, -0.15, -0.05, 0.0, 0.05, 0.15, 0.3]:
                if time.time() > _deadline - 10:
                    break
                for db in [-0.15, -0.05, 0.0, 0.05, 0.15, 0.3]:
                    if time.time() > _deadline - 10:
                        break
                    a2 = max(0.1, alpha_c + da)
                    b2 = max(0.0, beta_c + db)
                    pv = (a2, b2)
                    score = _evaluate_params(sequences, capacity, 'combined', pv)
                    if score < best_score:
                        best_score = score
                        best_params = ('combined', pv)
        elif ptype == 'content_bf':
            lo = max(0.0, pval * 0.5)
            hi = pval * 1.5 + 1.0
            n_steps = 20
            for i in range(n_steps + 1):
                if time.time() > _deadline - 10:
                    break
                b = lo + (hi - lo) * i / n_steps
                score = _evaluate_params(sequences, capacity, 'content_bf', b)
                if score < best_score:
                    best_score = score
                    best_params = ('content_bf', b)
        elif ptype == 'worst_fit_alpha':
            lo = max(0.1, pval * 0.5)
            hi = pval * 2.0
            n_steps = 20
            for i in range(n_steps + 1):
                if time.time() > _deadline - 10:
                    break
                wa = lo + (hi - lo) * i / n_steps
                score = _evaluate_params(sequences, capacity, 'worst_fit_alpha', wa)
                if score < best_score:
                    best_score = score
                    best_params = ('worst_fit_alpha', wa)
        elif ptype == 'mixed_wf':
            a_wf, b_wf = pval
            for da in [-0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5]:
                if time.time() > _deadline - 10:
                    break
                for db in [-0.3, -0.1, 0.0, 0.1, 0.3, 0.5]:
                    if time.time() > _deadline - 10:
                        break
                    a2 = max(0.1, a_wf + da)
                    b2 = max(0.0, b_wf + db)
                    pv = (a2, b2)
                    score = _evaluate_params(sequences, capacity, 'mixed_wf', pv)
                    if score < best_score:
                        best_score = score
                        best_params = ('mixed_wf', pv)
    
    # === Phase 3: High-precision polish ===
    if time.time() < _deadline - 8:
        ptype, pval = best_params
        if ptype == 'alpha':
            # Golden section search for 1D
            phi = (1 + np.sqrt(5)) / 2
            a = max(0.05, pval - 0.3)
            b = pval + 0.3
            for _ in range(15):
                if time.time() > _deadline - 5:
                    break
                c = b - (b - a) / phi
                d = a + (b - a) / phi
                fc = _evaluate_params(sequences, capacity, 'alpha', c)
                fd = _evaluate_params(sequences, capacity, 'alpha', d)
                if fc < fd:
                    b = d
                else:
                    a = c
                new_val = (a + b) / 2
                score = _evaluate_params(sequences, capacity, 'alpha', new_val)
                if score < best_score:
                    best_score = score
                    best_params = ('alpha', new_val)
        elif ptype == 'content_bf':
            phi = (1 + np.sqrt(5)) / 2
            a = max(0.0, pval - 1.0)
            b = pval + 1.0
            for _ in range(15):
                if time.time() > _deadline - 5:
                    break
                c = b - (b - a) / phi
                d = a + (b - a) / phi
                fc = _evaluate_params(sequences, capacity, 'content_bf', c)
                fd = _evaluate_params(sequences, capacity, 'content_bf', d)
                if fc < fd:
                    b = d
                else:
                    a = c
                new_val = (a + b) / 2
                score = _evaluate_params(sequences, capacity, 'content_bf', new_val)
                if score < best_score:
                    best_score = score
                    best_params = ('content_bf', new_val)
        elif ptype == 'target':
            phi = (1 + np.sqrt(5)) / 2
            a = max(0.0, pval - 5.0)
            b = pval + 5.0
            for _ in range(15):
                if time.time() > _deadline - 5:
                    break
                c = b - (b - a) / phi
                d = a + (b - a) / phi
                fc = _evaluate_params(sequences, capacity, 'target', c)
                fd = _evaluate_params(sequences, capacity, 'target', d)
                if fc < fd:
                    b = d
                else:
                    a = c
                new_val = (a + b) / 2
                score = _evaluate_params(sequences, capacity, 'target', new_val)
                if score < best_score:
                    best_score = score
                    best_params = ('target', new_val)
    
    return best_params, best_score


# Try fitting at module load time, but with a time guard
_start_time = time.time()

try:
    _BEST_PARAMS, _BEST_SCORE = _fit_parameters()
except Exception:
    _BEST_PARAMS = ('alpha', 1.0)
    _BEST_SCORE = float('inf')

# Build the scoring function based on best params
_param_type, param_value = _BEST_PARAMS


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Returns an array of scores; the item goes to the highest-scoring bin.
    """
    bins = np.asarray(bins, dtype=np.float64)
    remaining = bins
    valid_mask = remaining >= item
    
    scores = np.full(len(bins), -np.inf, dtype=np.float64)
    
    if not np.any(valid_mask):
        return scores
    
    gap = remaining[valid_mask] - item
    
    if _param_type == 'alpha':
        scores[valid_mask] = -gap ** param_value
    elif _param_type == 'target':
        scores[valid_mask] = -((gap) - param_value) ** 2
    elif _param_type == 'norm_alpha':
        safe_item = max(item, 1.0)
        ratio = gap / safe_item
        scores[valid_mask] = -ratio ** param_value
    elif _param_type == 'combined':
        alpha, beta = param_value
        fill_frac = remaining[valid_mask] / 100.0
        scores[valid_mask] = -(gap ** alpha) + beta * fill_frac
    elif _param_type == 'content_bf':
        beta = param_value
        content = 100.0 - remaining[valid_mask]
        scores[valid_mask] = -gap + beta * content
    elif _param_type == 'worst_fit':
        scores[valid_mask] = remaining[valid_mask]
    elif _param_type == 'worst_fit_alpha':
        scores[valid_mask] = remaining[valid_mask] ** param_value
    elif _param_type == 'mixed_wf':
        alpha, beta = param_value
        scores[valid_mask] = alpha * remaining[valid_mask] - beta * gap
    else:
        scores[valid_mask] = -gap
    
    return scores
# EVOLVE-BLOCK-END
