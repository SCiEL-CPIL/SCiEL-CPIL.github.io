<<<<<<< SEARCH
def _peak_suppression_obj(h, beta=30.0, gamma=2.0):
    """
    Peak suppression objective: penalize peaks that exceed a threshold t.
    Uses soft-max of (|ac| - t)^+ with additional smoothness.
    This focuses optimization on reducing the largest peaks.
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_full(h)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    # Normalize: work with peaks relative to the current max
    # Penalize all elements above a fraction of the max
    threshold = 0.7 * m
    excess = np.maximum(abs_ac - threshold, 0.0)
    # Soft sum of excesses
    if np.max(excess) < 1e-15:
        return m / (s * s)
    scaled = beta * excess
    scaled = np.clip(scaled, -50, 50)
    log_sum = np.log(np.sum(np.exp(scaled)))
    soft_excess = log_sum / beta
    # Combined objective: base max + penalty for widespread high peaks
    return (m + gamma * soft_excess) / (s * s)
=======
def _peak_suppression_obj(h, beta=30.0, gamma=2.0):
    """
    Peak suppression objective: penalize peaks that exceed a threshold t.
    Uses soft-max of (|ac| - t)^+ with additional smoothness.
    This focuses optimization on reducing the largest peaks.
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_full(h)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    threshold = 0.7 * m
    excess = np.maximum(abs_ac - threshold, 0.0)
    if np.max(excess) < 1e-15:
        return m / (s * s)
    scaled = beta * excess
    scaled = np.clip(scaled, -50, 50)
    log_sum = np.log(np.sum(np.exp(scaled)))
    soft_excess = log_sum / beta
    return (m + gamma * soft_excess) / (s * s)

def _epigraph_obj(params, beta=30.0):
    """
    Epigraph formulation: minimize t subject to soft_max(|ac|) <= t * sum(h)^2.
    params = [h_0, ..., h_{N-1}, t]
    """
    h = params[:N]
    t = params[N]
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_full(h)
    sm = _soft_max_abs(ac, beta)
    constraint_violation = np.maximum(sm - t * s * s, 0.0)
    norm_penalty = 0.01 * (s * s - 1.0) ** 2
    return t + 100.0 * constraint_violation + norm_penalty
>>>>>>> REPLACE
