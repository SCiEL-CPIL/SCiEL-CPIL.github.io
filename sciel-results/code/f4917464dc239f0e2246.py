import numpy as np

N = 400

def _autoconv_peak(x):
    """Compute max |autoconv| at nonzero lags."""
    conv = np.convolve(x, x, mode='full')
    center = N - 1
    peak = 0.0
    if center > 0:
        peak = max(peak, np.max(np.abs(conv[:center])))
    if center < len(conv) - 1:
        peak = max(peak, np.max(np.abs(conv[center+1:])))
    return peak

def construct():
    np.random.seed(123)
    M = 18
    support = np.sort(np.random.choice(N, M, replace=False))
    signs = np.ones(M)
    
    def eval_x(supp, sgn):
        x = np.zeros(N)
        x[supp] = sgn / np.sqrt(len(supp))
        return _autoconv_peak(x)
    
    best = eval_x(support, signs)
    
    for _ in range(800):
        improved = False
        # Flip signs
        for i in range(M):
            ns = signs.copy(); ns[i] *= -1
            v = eval_x(support, ns)
            if v < best - 1e-12:
                signs = ns; best = v; improved = True
        # Move one support point
        for i in range(M):
            for _ in range(15):
                np_new = np.random.randint(N)
                if np_new in support: continue
                nsupp = support.copy(); nsupp[i] = np_new; nsupp.sort()
                v = eval_x(nsupp, signs)
                if v < best - 1e-12:
                    support = nsupp; best = v; improved = True; break
        if not improved:
            break
    
    x = np.zeros(N)
    x[support] = signs / np.sqrt(M)
    return x
