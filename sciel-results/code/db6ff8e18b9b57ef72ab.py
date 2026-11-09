import numpy as np
from scipy.optimize import minimize

def _compute_P(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] for discrete measure."""
    w = np.asarray(weights, dtype=np.float64)
    s = np.asarray(support, dtype=np.float64)
    w = w / w.sum()
    n = len(s)
    P = 0.0
    for l in range(n):
        target = 2.0 * s[l]
        # P[X1+X2+X3 < target]
        triple = 0.0
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if s[i] + s[j] + s[k] < target:
                        triple += w[i] * w[j] * w[k]
        P += w[l] * triple
    return P

def _neg_P(params, support_template):
    """Negative P for minimization."""
    # params: [w0, w1, ..., w_{n-1}, M] where M is the last support point (if templated)
    # For simplicity, assume support is fixed except possibly last point is M
    n = len(support_template)
    weights = params[:n]
    if n > 0 and support_template[-1] is None:
        M = params[n]
        support = list(support_template[:-1]) + [M]
    else:
        support = support_template
    if np.any(weights < -1e-12):
        return 1e10
    weights = np.maximum(weights, 0)
    if weights.sum() < 1e-12:
        return 1e10
    P = _compute_P(support, weights)
    return -P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    best_P = -1
    best_support = None
    best_weights = None
    
    # Try uniform on {0, ..., n} for n=1..10
    for n in range(1, 11):
        support = np.arange(n, dtype=np.float64)
        weights = np.ones(n, dtype=np.float64) / n
        P = _compute_P(support, weights)
        if P > best_P:
            best_P = P
            best_support = support.copy()
            best_weights = weights.copy()
    
    # Try 2-point: {0, M} - but M doesn't matter, optimize p
    # P = (1-p) * p^2 * (3-2p)
    # Maximize over p in (0,1)
    from scipy.optimize import minimize_scalar
    def neg_2pt(p):
        if p <= 0 or p >= 1:
            return 1e10
        return -((1-p) * p**2 * (3-2*p))
    res = minimize_scalar(neg_2pt, bounds=(0.01, 0.99), method='bounded')
    p_opt = res.x
    P_2pt = -res.fun
    if P_2pt > best_P:
        best_P = P_2pt
        best_support = np.array([0.0, 1.0])  # M=1 is fine
        best_weights = np.array([p_opt, 1-p_opt])
    
    # Try 3-point: {0, 1, M} with M large (say M=100)
    # Optimize weights (a,b,c)
    def neg_3pt(params):
        a, b, c = params
        if a < -1e-12 or b < -1e-12 or c < -1e-12:
            return 1e10
        a, b, c = max(a,0), max(b,0), max(c,0)
        s = a+b+c
        if s < 1e-12:
            return 1e10
        support = np.array([0.0, 1.0, 100.0])
        weights = np.array([a, b, c])
        return -_compute_P(support, weights)
    
    # Initial guess
    x0 = np.array([0.5, 0.2, 0.3])
    res3 = minimize(neg_3pt, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
    if res3.fun < -1e9:
        P_3pt = -res3.fun
        a,b,c = res3.x
        s = a+b+c
        if s > 1e-12 and min(a,b,c) > -1e-12:
            a,b,c = max(a,0), max(b,0), max(c,0)
            s = a+b+c
            if s > 1e-12:
                support = np.array([0.0, 1.0, 100.0])
                weights = np.array([a, b, c])
                P_3pt = _compute_P(support, weights)
                if P_3pt > best_P:
                    best_P = P_3pt
                    best_support = support
                    best_weights = weights
    
    # Try 4-point: {0,1,2,M}
    def neg_4pt(params):
        a,b,c,d = params
        if min(a,b,c,d) < -1e-12:
            return 1e10
        a,b,c,d = max(a,0),max(b,0),max(c,0),max(d,0)
        s = a+b+c+d
        if s < 1e-12:
            return 1e10
        support = np.array([0.0, 1.0, 2.0, 100.0])
        weights = np.array([a,b,c,d])
        return -_compute_P(support, weights)
    
    x0_4 = np.array([0.4, 0.2, 0.2, 0.2])
    res4 = minimize(neg_4pt, x0_4, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
    if res4.fun < -1e9:
        a,b,c,d = res4.x
        a,b,c,d = max(a,0),max(b,0),max(c,0),max(d,0)
        s = a+b+c+d
        if s > 1e-12:
            support = np.array([0.0, 1.0, 2.0, 100.0])
            weights = np.array([a,b,c,d])
            P_4pt = _compute_P(support, weights)
            if P_4pt > best_P:
                best_P = P_4pt
                best_support = support
                best_weights = weights
    
    # Validate
    support = np.asarray(best_support, dtype=np.float64).flatten()
    weights = np.asarray(best_weights, dtype=np.float64).flatten()
    if len(support) != len(weights) or len(support) < 1:
        # Fallback
        support = np.array([0.0, 1.0])
        weights = np.array([0.5784, 0.4216])
    if np.any(support < -1e-9) or np.any(weights < -1e-9):
        support = np.array([0.0, 1.0])
        weights = np.array([0.5784, 0.4216])
    if weights.sum() <= 1e-9:
        support = np.array([0.0, 1.0])
        weights = np.array([0.5, 0.5])
    
    return support, weights
