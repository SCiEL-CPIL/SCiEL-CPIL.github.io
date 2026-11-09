"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 580
    
    seed_roots = np.array([4.0 * (i + 1) for i in range(12)])
    best_roots = seed_roots.copy()
    best_score = np.inf
    
    def is_valid(roots):
        roots = np.sort(roots)
        if len(roots) != 12:
            return False
        if np.any(roots <= 0):
            return False
        if np.any(np.diff(roots) < 1e-6):
            return False
        return True
    
    # We don't know the exact objective, so use multiple proxies
    # and try to find roots that minimize a reasonable measure.
    
    # Proxy 1: minimize largest root
    # Proxy 2: minimize sum of roots
    # Proxy 3: minimize geometric mean
    
    def proxy1(roots):
        return np.sort(roots)[-1]
    
    def proxy2(roots):
        return np.sum(np.sort(roots))
    
    def proxy3(roots):
        r = np.sort(roots)
        return np.exp(np.mean(np.log(r)))
    
    candidates = [seed_roots]
    
    # Try Laguerre roots
    try:
        from scipy.special import roots_laguerre
        r = roots_laguerre(12).real
        if is_valid(r):
            candidates.append(r)
        for scale in [0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]:
            scaled = r * scale
            if is_valid(scaled):
                candidates.append(scaled)
    except Exception:
        pass
    
    # Try linear with different scales
    for a in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 30.0, 36.0, 42.0, 48.0]:
        roots = a * np.arange(1, 13)
        if is_valid(roots):
            candidates.append(roots)
    
    # Try geometric
    for a in [0.1, 0.5, 1.0, 2.0, 4.0, 8.0]:
        for b in [1.01, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5, 1.7, 2.0, 2.5]:
            roots = a * (b ** np.arange(1, 13))
            if is_valid(roots):
                candidates.append(roots)
    
    # Try power law
    for a in [0.5, 1.0, 2.0, 4.0, 8.0]:
        for p in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5]:
            roots = a * (np.arange(1, 13) ** p)
            if is_valid(roots):
                candidates.append(roots)
    
    # Evaluate all candidates with all proxies
    for roots in candidates:
        if not is_valid(roots):
            continue
        for proxy in [proxy1, proxy2, proxy3]:
            s = proxy(roots)
            if s < best_score:
                best_score = s
                best_roots = roots.copy()
    
    # Try scipy optimization with proxy1 (minimize largest root)
    if time.time() < deadline - 100:
        try:
            from scipy.optimize import minimize
            
            def obj(x):
                roots = np.sort(x)
                if np.any(roots <= 0) or np.any(np.diff(roots) < 1e-6):
                    return 1e10
                return roots[-1]
            
            result = minimize(obj, best_roots, method='Nelder-Mead',
                            options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if result.success and is_valid(result.x) and result.fun < best_score:
                best_roots = np.sort(result.x)
                best_score = result.fun
        except Exception:
            pass
    
    # Final check
    best_roots = np.sort(best_roots)
    if not is_valid(best_roots) or time.time() > deadline - 5:
        return [float(r) for r in seed_roots]
    
    return [float(r) for r in best_roots]
# EVOLVE-BLOCK-END
