def soft_min(values, alpha):
    # values: array of areas
    # Return smooth approximation of min(values)
    # Use log-sum-exp trick for stability
    max_v = np.max(values)
    # If alpha is large, exp(-alpha * (v - max_v)) might under/overflow?
    # Actually, we want min.
    # min(v) = - max(-v)
    # soft_max(v) = 1/alpha * log(sum(exp(alpha * v)))
    # soft_min(v) = - soft_max(-v) = -1/alpha * log(sum(exp(-alpha * v)))
    # To avoid overflow in exp, shift by max(-v) = -min(v).
    # Let m = min(values).
    # exp(-alpha * (v - m)) is <= 1.
    # sum(exp(-alpha * (v - m)))
    # soft_min = m - 1/alpha * log(sum(exp(-alpha * (v - m))))
    m = np.min(values)
    shifted = values - m
    # exp(-alpha * shifted)
    # If alpha is large, and shifted is small, this is fine.
    return m - (1.0 / alpha) * np.log(np.sum(np.exp(-alpha * shifted)))

def objective_smooth(params, alpha):
    u = params[:, 0].copy()
    v = params[:, 1].copy()
    s = u + v
    mask = s > 1
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    
    uv = np.column_stack((u, v))
    points = bary_to_cart(uv)
    n = len(points)
    areas = np.zeros((n*(n-1)*(n-2)//6,)) # 165 for n=11
    idx = 0
    for i in range(n):
        pi = points[i]
        for j in range(i+1, n):
            pj = points[j]
            v1 = pj - pi
            for k in range(j+1, n):
                pk = points[k]
                v2 = pk - pi
                areas[idx] = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                idx += 1
    
    sm = soft_min(areas, alpha)
    return -sm
