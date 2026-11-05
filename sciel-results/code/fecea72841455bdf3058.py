def _smooth_max(projs, beta=50.0):
    # projs: array of projections
    # We want max(|p|)
    # log-sum-exp of |p|
    abs_projs = np.abs(projs)
    # Shift for stability
    m = np.max(abs_projs)
    if m == 0:
        return 0.0
    s = np.sum(np.exp(beta * (abs_projs - m)))
    return m + np.log(s) / beta
