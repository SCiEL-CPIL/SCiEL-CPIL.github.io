import numpy as np

def equation(b, s, temp, pH, params):
    p = params
    
    # Normalize inputs to [0, 1]
    bn = (b - 1.0) / (4.998 - 1.0)
    sn = (s - 1.0) / (4.999 - 1.0)
    tn = (temp - 1.002) / (39.98 - 1.002)
    pn = (pH - 2.0) / (10.0)
    
    # Clip to [0, 1] for stability
    bn = np.clip(bn, 0.0, 1.0)
    sn = np.clip(sn, 0.0, 1.0)
    tn = np.clip(tn, 0.0, 1.0)
    pn = np.clip(pn, 0.0, 1.0)
    
    # Construct a design matrix with 10 columns
    # Columns: 1, bn, sn, tn, pn, bn*sn, bn*pn, sn*tn, sn*pn, tn*pn
    # This gives a linear model with interactions, convex for BFGS
    
    n = len(bn)
    X = np.empty((n, 10), dtype=np.float64)
    X[:, 0] = 1.0
    X[:, 1] = bn
    X[:, 2] = sn
    X[:, 3] = tn
    X[:, 4] = pn
    X[:, 5] = bn * sn
    X[:, 6] = bn * pn
    X[:, 7] = sn * tn
    X[:, 8] = sn * pn
    X[:, 9] = tn * pn
    
    # Linear prediction
    pred = X @ p
    
    return pred
