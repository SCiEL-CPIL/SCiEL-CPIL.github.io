import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite

_H0_0 = float(eval_hermite(0, 0.0))
_H4_0 = float(eval_hermite(4, 0.0))
_H8_0 = float(eval_hermite(8, 0.0))
_H12_0 = float(eval_hermite(12, 0.0))

def _c12(c0, c4, c8):
    return -(c0*_H0_0 + c4*_H4_0 + c8*_H8_0)/_H12_0

def _bound(c0, c4, c8):
    c12 = _c12(c0, c4, c8)
    x = np.linspace(-9.0, 9.0, 16001)
    H4 = eval_hermite(4, x); H8 = eval_hermite(8, x); H12 = eval_hermite(12, x)
    poly = c0 + c4*H4 + c8*H8 + c12*H12
    g = poly * np.exp(-0.5*x*x)
    g2 = g*g
    I0 = np.trapz(g2, x)
    I2 = np.trapz(x*x*g2, x)
    gp = np.gradient(g, x)
    Ig = np.trapz(gp*gp, x)
    if (I0 <= 0 or I2 <= 0 or Ig <= 0 or
        not np.isfinite(I0) or not np.isfinite(I2) or not np.isfinite(Ig)):
        return 1e10
    return I0*I0/(I2*Ig)

def _optimize():
    seed = np.array([-113/100, 1/25, 1/3240.0])
    best = seed.copy()
    best_val = _bound(*seed)
    t0 = time.time()
    rng = np.random.default_rng(0)
    starts = [seed]
    for _ in range(6):
        starts.append(rng.uniform(-1.5, 1.5, size=3))
    for s in starts:
        if time.time() - t0 > 140:
            break
        try:
            res = minimize(_bound, s, method='Nelder-Mead',
                           options={'maxiter': 6000, 'xatol':1e-10, 'fatol':1e-13,
                                    'adaptive': True})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best = res.x.copy()
        except Exception:
            pass
    if not np.all(np.isfinite(best)):
        return seed
    return best

_OPT = _optimize()

def construct():
    return _OPT
