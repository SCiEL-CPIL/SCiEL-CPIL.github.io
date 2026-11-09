import numpy as np
import time
from scipy.optimize import minimize

def equation(b, s, temp, pH, params):
    """
    Shallow feedforward network with tanh activations.
    10 parameters: 9 weights + 1 bias, applied to 9 nonlinear basis functions.
    
    Basis functions (all normalized inputs in roughly [-1, 2] range):
      0: tanh(bn)
      1: tanh(sn)
      2: tanh(tn)
      3: tanh(phn)
      4: tanh(bn * tn)
      5: tanh(sn * phn)
      6: tanh(bn * sn)
      7: tanh(tn * phn)
      8: tanh(sn * tn)
      bias: 1.0
    """
    p = params
    
    # Normalize inputs to roughly [-1, 2] for stable tanh inputs
    # b: [1, 5] -> center 3, scale 2 -> [-1, 1]
    bn = (b - 3.0) / 2.0
    # s: [1, 40] -> use log to compress: log(s) in [0, ~3.7], center 1.8, scale 1.5
    sn = (np.log1p(s) - 1.8) / 1.5
    # temp: [1, 40] -> center 20, scale 10 -> [-1.9, 2]
    tn = (temp - 20.0) / 10.0
    # pH: [5, 9] -> center 7, scale 2 -> [-1, 1]
    phn = (pH - 7.0) / 2.0
    
    # Compute 9 basis functions + bias
    h0 = np.tanh(bn)
    h1 = np.tanh(sn)
    h2 = np.tanh(tn)
    h3 = np.tanh(phn)
    h4 = np.tanh(bn * tn)
    h5 = np.tanh(sn * phn)
    h6 = np.tanh(bn * sn)
    h7 = np.tanh(tn * phn)
    h8 = np.tanh(sn * tn)
    
    # Linear combination: p[0..8] are weights, p[9] is bias
    result = (p[0] * h0 + p[1] * h1 + p[2] * h2 + p[3] * h3 +
              p[4] * h4 + p[5] * h5 + p[6] * h6 + p[7] * h7 +
              p[8] * h8 + p[9])
    
    return result

def _objective(params, b, s, temp, pH, y):
    """MSE objective for fitting."""
    pred = equation(b, s, temp, pH, params)
    if not np.all(np.isfinite(pred)):
        return 1e30
    mse = np.mean((pred - y) ** 2)
    # Small L2 regularization to prevent parameter explosion
    reg = 1e-4 * np.sum(params**2)
    return mse + reg

def fit_params(b, s, temp, pH, y, deadline_margin=5.0):
    """Fit parameters with BFGS (as required by contract) + L-BFGS-B polish."""
    start_time = time.time()
    
    # BFGS starting from ones as required
    x0 = np.ones(10)
    res = minimize(_objective, x0, args=(b, s, temp, pH, y), method='BFGS')
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 110 - deadline_margin:
        return res.x
    
    # Polish with L-BFGS-B for higher precision
    bounds = [(-5, 5)] * 10  # All weights in reasonable range
    
    res2 = minimize(_objective, res.x, args=(b, s, temp, pH, y),
                    method='L-BFGS-B', bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
    
    # Check deadline again
    elapsed = time.time() - start_time
    if elapsed > 110 - deadline_margin:
        return res.x
    
    # Use the better result
    if res2.fun < res.fun:
        return res2.x
    return res.x
