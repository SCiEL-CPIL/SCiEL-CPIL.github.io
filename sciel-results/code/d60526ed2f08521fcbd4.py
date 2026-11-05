import time
import math
import numpy as np
import random

def _fast_reflectance(n_layers, d_layers, n0, ns, wls):
    """
    Vectorized reflectance calculation.
    n_layers: list of indices
    d_layers: list of thicknesses
    n0, ns: ambient, substrate indices
    wls: array of wavelengths
    Returns: array of reflectance (fraction)
    """
    N = len(n_layers)
    # A and B are complex arrays of shape (len(wls),)
    A = np.ones(len(wls), dtype=complex)
    B = np.zeros(len(wls), dtype=complex)
    
    for i in range(N):
        n = n_layers[i]
        d = d_layers[i]
        # delta = 2 * pi * n * d / wls
        delta = 2.0 * math.pi * n * d / wls
        c = np.cos(delta)
        s = np.sin(delta)
        
        # M = [[c, i*s/n], [i*n*s, c]]
        # A_new = A*c + B*(i*s/n)
        # B_new = A*(i*n*s) + B*c
        A_new = A * c + B * (1j * s / n)
        B_new = A * (1j * n * s) + B * c
        A, B = A_new, B_new
        
    Y = (A + B * ns) / (A * ns + B)
    R = np.abs((Y - n0) / (Y + n0)) ** 2
    return R

def _mean_pct_fast(n_layers, d_layers, n0, ns, wls):
    R = _fast_reflectance(n_layers, d_layers, n0, ns, wls)
    return 100.0 * float(np.mean(R))

def _nelder_mead_vectorized(f, x0, bounds, deadline, max_iter=200, xatol=0.01, fatol=1e-6):
    """
    Nelder-Mead with vectorized function evaluation support (though f is still scalar output).
    Includes deadline guard.
    """
    n = len(x0)
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        xi = np.array(x0, dtype=float)
        step = 0.1 * (bounds[1][i] - bounds[0][i])
        xi[i] += step
        xi = np.clip(xi, bounds[0], bounds[1])
        simplex.append(xi)
    
    vals = [f(x) for x in simplex]
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        idx = np.argsort(vals)
        simplex = [simplex[i] for i in idx]
        vals = [vals[i] for i in idx]
        
        if abs(vals[0] - vals[-1]) < fatol:
            break
        if np.max(np.abs(simplex[0] - simplex[1])) < xatol:
            break
        
        centroid = np.sum(simplex[:-1], axis=0) / n
        xr = centroid + (centroid - simplex[-1])
        xr = np.clip(xr, bounds[0], bounds[1])
        fr = f(xr)
        
        if fr < vals[0]:
            xe = centroid + 2.0 * (centroid - simplex[-1])
            xe = np.clip(xe, bounds[0], bounds[1])
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                vals[-1] = fe
            else:
                simplex[-1] = xr
                vals[-1] = fr
        elif fr < vals[-2]:
            simplex[-1] = xr
            vals[-1] = fr
        else:
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            xc = np.clip(xc, bounds[0], bounds[1])
            fc = f(xc)
            if fc < vals[-1]:
                simplex[-1] = xc
                vals[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], bounds[0], bounds[1])
                    vals[i] = f(simplex[i])
    
    idx = int(np.argmin(vals))
    return simplex[idx], vals[idx]

def _polish_stack(labels, thicknesses, n0, ns, wls, deadline, max_total, max_layer):
    """
    Polish a specific stack structure (labels fixed) by optimizing thicknesses.
    """
    nL, nH = 1.38, 2.35
    nmat = np.array([nL if l == 'L' else nH for l in labels])
    N = len(labels)
    
    # Objective function with constraints
    def obj(x):
        if time.time() > deadline:
            return 1e6
        x = np.clip(x, 0.0, max_layer)
        if x.sum() > max_total:
            # Penalty for exceeding total thickness
            return 1e6 + (x.sum() - max_total) * 10
        return _mean_pct_fast(nmat, x, n0, ns, wls)
    
    x0 = np.array(thicknesses, dtype=float)
    b0 = np.zeros(N)
    b1 = np.full(N, max_layer)
    
    try:
        xb, fb = _nelder_mead_vectorized(obj, x0, (b0, b1), deadline, max_iter=300, xatol=0.1, fatol=1e-5)
    except Exception:
        return labels, thicknesses
    
    # Final constraint check and projection
    xb = np.clip(xb, 0.0, max_layer)
    if xb.sum() > max_total:
        xb = xb * (max_total / xb.sum())
    
    return labels, [float(t) for t in xb]

def design(problem):
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n0 = problem['ambient_index']
    ns = problem['substrate_index']
    nL = problem['materials']['L']
    nH = problem['materials']['H']
    band = problem['band_nm']
    wls = np.linspace(band[0], band[1], 401)
    max_layers = problem['max_layers']
    max_total = problem['max_total_nm']
    max_layer = problem['max_layer_nm']
    
    center_wl = 0.5 * (band[0] + band[1])
    
    best_labels = ['L']
    best_th = [center_wl / (4.0 * nL)]
    best_score = _mean_pct_fast([nL], best_th, n0, ns, wls)
    
    # Generate candidate structures
    # We focus on odd numbers of layers, starting and ending with L, as they provide
    # better index matching to air (1.0) and glass (1.52) than H layers.
    # Pattern: L H L H ... L
    candidates = []
    
    # Single layer
    candidates.append((['L'], [center_wl / (4.0 * nL)]))
    
    # Multi-layer stacks
    for n_layers in [3, 5, 7, 9, 11, 13, 15, 17, 19]:
        if n_layers > max_layers:
            break
        labels = []
        ths = []
        for i in range(n_layers):
            if i % 2 == 0:
                labels.append('L')
                ths.append(center_wl / (4.0 * nL))
            else:
                labels.append('H')
                ths.append(center_wl / (4.0 * nH))
        
        # Check total thickness constraint for initial guess
        # If too thick, scale down
        total_t = sum(ths)
        if total_t > max_total:
            scale = max_total / total_t
            ths = [t * scale for t in ths]
            
        candidates.append((labels, ths))
        
        # Also try starting with H? Less likely to be optimal, but let's include a few
        if n_layers <= 9:
            labels_h = []
            ths_h = []
            for i in range(n_layers):
                if i % 2 == 0:
                    labels_h.append('H')
                    ths_h.append(center_wl / (4.0 * nH))
                else:
                    labels_h.append('L')
                    ths_h.append(center_wl / (4.0 * nL))
            total_t_h = sum(ths_h)
            if total_t_h > max_total:
                scale_h = max_total / total_t_h
                ths_h = [t * scale_h for t in ths_h]
            candidates.append((labels_h, ths_h))
            
    # Shuffle candidates to ensure diverse time allocation
    random.shuffle(candidates)
    
    for labels, ths in candidates:
        if time.time() > deadline - 5.0:
            break
            
        # Polish this candidate
        polished_labels, polished_ths = _polish_stack(labels, ths, n0, ns, wls, deadline, max_total, max_layer)
        
        # Evaluate
        nmat = np.array([nL if l == 'L' else nH for l in polished_labels])
        score = _mean_pct_fast(nmat, np.array(polished_ths), n0, ns, wls)
        
        if score < best_score:
            best_score = score
            best_labels = polished_labels
            best_th = polished_ths
            
    # Final constraint check
    if len(best_labels) > max_layers or len(best_labels) != len(best_th):
        best_labels = ['L']
        best_th = [center_wl / (4.0 * nL)]
        
    for t in best_th:
        if t < 0 or t > max_layer:
            best_labels = ['L']
            best_th = [center_wl / (4.0 * nL)]
            break
            
    if sum(best_th) > max_total:
        scale = max_total / sum(best_th)
        best_th = [t * scale for t in best_th]
        
    return best_labels, best_th
