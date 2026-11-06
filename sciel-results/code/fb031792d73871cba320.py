"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import time

def construct():
    t0 = time.time()
    deadline = t0 + 85.0
    
    cosmologies = [
        {'Om': 0.3, 'Or': 1e-4, 'Ode': 0.7, 'w0': -1.0, 'wa': 0.0},
        {'Om': 0.3, 'Or': 1e-4, 'Ode': 0.7, 'w0': -0.9, 'wa': 0.3},
        {'Om': 0.3, 'Or': 1e-4, 'Ode': 0.7, 'w0': -1.1, 'wa': -0.3},
        {'Om': 0.3, 'Or': 1e-4, 'Ode': 0.7, 'w0': -0.8, 'wa': 0.5},
        {'Om': 0.3, 'Or': 1e-4, 'Ode': 0.7, 'w0': -0.7, 'wa': -0.5},
        {'Om': 0.2, 'Or': 1e-4, 'Ode': 0.8, 'w0': -1.0, 'wa': 0.0},
        {'Om': 0.4, 'Or': 1e-4, 'Ode': 0.6, 'w0': -1.0, 'wa': 0.0},
    ]
    
    z_tests = np.array([0.01, 0.1, 0.5, 1.0, 2.0, 5.0])
    
    def E_of_a(a, cosmo):
        Om, Or, Ode, w0, wa = cosmo['Om'], cosmo['Or'], cosmo['Ode'], cosmo['w0'], cosmo['wa']
        E2 = Om/a**3 + Or/a**4
        if abs(wa) < 1e-12:
            E2 += Ode * a**(-3.0*(1.0+w0))
        else:
            E2 += Ode * a**(-3.0*(1.0+w0+wa*(1.0-a)))
        return np.sqrt(np.maximum(E2, 1e-30))
    
    def true_integral(z, cosmo):
        if z < 1e-10:
            return 0.0
        def integrand(zp):
            a = 1.0/(1.0+zp)
            return 1.0/E_of_a(a, cosmo)
        val, _ = quad(integrand, 0, z, limit=100, epsabs=1e-10, epsrel=1e-10)
        return (1.0+z)*z*val
    
    # Precompute
    true_vals = np.zeros((len(cosmologies), len(z_tests)))
    for i, cosmo in enumerate(cosmologies):
        for j, z in enumerate(z_tests):
            true_vals[i, j] = true_integral(z, cosmo)
    
    def fast_worst_error(nodes, weights):
        max_e = 0.0
        for i, cosmo in enumerate(cosmologies):
            for j, z in enumerate(z_tests):
                a_vals = 1.0/(1.0 + z*nodes)
                E_vals = E_of_a(a_vals, cosmo)
                qv = (1.0+z)*z*np.sum(weights/E_vals)
                tv = true_vals[i, j]
                if abs(tv) < 1e-15:
                    continue
                e = abs(qv - tv)/abs(tv)*100.0
                if e > max_e:
                    max_e = e
                if max_e > 50.0:
                    return max_e
        return max_e
    
    def objective(params):
        if time.time() > deadline:
            return 1e10
        nodes = params[:8]
        log_w = params[8:]
        
        pen = 0.0
        if np.any(nodes <= 0) or np.any(nodes >= 1):
            pen += 100.0 * (np.sum(np.maximum(0, 1e-4-nodes)) + np.sum(np.maximum(0, nodes-1+1e-4)))
        d = np.diff(nodes)
        if np.any(d <= 0):
            pen += 100.0 * np.sum(np.maximum(0, 1e-4-d))
        
        if pen > 10.0:
            return pen
        
        w = np.exp(log_w)
        w = w / np.sum(w)
        
        return fast_worst_error(nodes, w) + pen
    
    x0 = np.concatenate([(np.arange(8)+0.5)/8.0, np.zeros(8)])
    
    best_x = x0.copy()
    best_obj = objective(x0)
    
    if time.time() < deadline - 20:
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x
        except:
            pass
    
    if time.time() < deadline - 5:
        for ss in [0.01, 0.003, 0.001]:
            if time.time() > deadline - 3:
                break
            for _ in range(30):
                if time.time() > deadline - 2:
                    break
                cand = best_x + np.random.randn(len(best_x))*ss
                cand[:8] = np.clip(cand[:8], 1e-3, 1-1e-3)
                for i in range(1, 8):
                    if cand[i] <= cand[i-1]:
                        cand[i] = cand[i-1] + 1e-3
                cand[:8] = np.clip(cand[:8], 1e-3, 1-1e-3)
                ov = objective(cand)
                if ov < best_obj:
                    best_obj = ov
                    best_x = cand
    
    nodes = best_x[:8]
    log_w = best_x[8:]
    w = np.exp(log_w)
    w = w / np.sum(w)
    
    if not (np.all(nodes > 0) and np.all(nodes < 1) and np.all(np.diff(nodes) > 0)):
        nodes = (np.arange(8) + 0.5) / 8.0
        w = np.full(8, 0.125)
    
    return {"nodes": nodes, "weights": w}
# EVOLVE-BLOCK-END
