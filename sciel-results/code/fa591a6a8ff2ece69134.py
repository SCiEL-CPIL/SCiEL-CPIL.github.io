"""Broadband AR coating: genetic algorithm over alternating L/H stacks with thickness optimization."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
import random

def _reflectance_array(n_layers, d_layers, n0, ns, wls):
    """Reflectance (fraction) at each wavelength via characteristic-matrix method.
    n_layers, d_layers are arrays ordered air->substrate."""
    N = len(n_layers)
    out = np.empty(len(wls), dtype=float)
    for wi, lam in enumerate(wls):
        A = 1.0 + 0j
        B = 0.0 + 0j
        for i in range(N):
            n = n_layers[i]
            delta = 2.0 * math.pi * n * d_layers[i] / lam
            c = math.cos(delta)
            s = math.sin(delta)
            # M = [[c, i*s/n],[i*n*s, c]]
            A_new = A * c + 1j * B * s / n
            B_new = A * (1j * n * s) + B * c
            A, B = A_new, B_new
        Y = (A + B * ns) / (A * ns + B)
        out[wi] = abs((Y - n0) / (Y + n0)) ** 2
    return out

def _mean_pct(n_layers, d_layers, n0, ns, wls):
    return 100.0 * float(np.mean(_reflectance_array(n_layers, d_layers, n0, ns, wls)))

def _nelder_mead(f, x0, bounds, deadline, max_iter=200, xatol=0.01, fatol=1e-6):
    """Simple Nelder-Mead optimizer with deadline guard."""
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
                simplex[-1] = xe; vals[-1] = fe
            else:
                simplex[-1] = xr; vals[-1] = fr
        elif fr < vals[-2]:
            simplex[-1] = xr; vals[-1] = fr
        else:
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            xc = np.clip(xc, bounds[0], bounds[1])
            fc = f(xc)
            if fc < vals[-1]:
                simplex[-1] = xc; vals[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], bounds[0], bounds[1])
                    vals[i] = f(simplex[i])
    idx = int(np.argmin(vals))
    return simplex[idx], vals[idx]

def _polish_thicknesses(labels, thicknesses, n0, ns, wls, deadline):
    """Polish the physical thicknesses of a fixed label sequence using Nelder-Mead."""
    nL, nH = 1.38, 2.35
    nmat = np.array([nL if l == 'L' else nH for l in labels])
    N = len(labels)
    def obj(x):
        if time.time() > deadline:
            return 1e6
        x = np.clip(x, 0.0, 400.0)
        if x.sum() > 1000.0:
            return 1e6
        return _mean_pct(nmat, x, n0, ns, wls)
    x0 = np.array(thicknesses, dtype=float)
    b0 = np.zeros(N); b1 = np.full(N, 400.0)
    try:
        xb, fb = _nelder_mead(obj, x0, (b0, b1), deadline, max_iter=150, xatol=0.1, fatol=1e-5)
    except Exception:
        return labels, thicknesses
    xb = np.clip(xb, 0.0, 400.0)
    if xb.sum() > 1000.0:
        xb = xb * (1000.0 / xb.sum())
    return labels, [float(t) for t in xb]

def _ga_optimize(n0, ns, wls, max_layers, max_total, max_layer, deadline, pop_size=40, n_gen=80):
    """Genetic algorithm over alternating L/H stacks with variable layer counts."""
    nL = 1.38
    nH = 2.35
    center = 0.5 * (wls[0] + wls[-1])
    
    # Initialize population: each individual is (n_layers, start_with_H, thicknesses)
    # We encode: n_layers in [1, max_layers], start_with_H in {0,1}, thicknesses array
    population = []
    for _ in range(pop_size):
        n_layers = random.randint(2, max_layers)
        start_with_H = random.randint(0, 1)
        # Initialize thicknesses around quarter-wave
        ths = []
        for i in range(n_layers):
            is_H = (i + start_with_H) % 2 == 1
            base = center / (4.0 * (nH if is_H else nL))
            ths.append(base * random.uniform(0.5, 1.5))
        population.append((n_layers, start_with_H, ths))
    
    def eval_individual(ind):
        n_layers, start_with_H, ths = ind
        if time.time() > deadline:
            return 1e6
        labels = []
        for i in range(n_layers):
            is_H = (i + start_with_H) % 2 == 1
            labels.append('H' if is_H else 'L')
        nmat = np.array([nH if l == 'H' else nL for l in labels])
        th_arr = np.array(ths, dtype=float)
        if th_arr.sum() > max_total:
            return 1e6
        if np.any(th_arr < 0) or np.any(th_arr > max_layer):
            return 1e6
        return _mean_pct(nmat, th_arr, n0, ns, wls)
    
    scores = [eval_individual(ind) for ind in population]
    best_idx = int(np.argmin(scores))
    best_score = scores[best_idx]
    best_ind = population[best_idx]
    
    for gen in range(n_gen):
        if time.time() > deadline:
            break
        
        # Sort by score
        order = np.argsort(scores)
        population = [population[i] for i in order]
        scores = [scores[i] for i in order]
        
        # Elitism: keep top 10%
        n_elite = max(2, pop_size // 10)
        new_pop = population[:n_elite]
        new_scores = scores[:n_elite]
        
        # Fill rest with offspring
        while len(new_pop) < pop_size:
            # Tournament selection
            def select():
                idx1 = random.randint(0, min(10, pop_size-1))
                idx2 = random.randint(0, min(10, pop_size-1))
                return population[idx1] if scores[idx1] < scores[idx2] else population[idx2]
            
            p1 = select()
            p2 = select()
            
            # Crossover: combine n_layers and start patterns
            if random.random() < 0.5:
                # Uniform crossover on thicknesses
                n_layers = random.choice([p1[0], p2[0]])
                start_with_H = random.choice([p1[1], p2[1]])
                ths = []
                for i in range(n_layers):
                    if i < len(p1[2]) and i < len(p2[2]):
                        if random.random() < 0.5:
                            ths.append(p1[2][i])
                        else:
                            ths.append(p2[2][i])
                    elif i < len(p1[2]):
                        ths.append(p1[2][i])
                    elif i < len(p2[2]):
                        ths.append(p2[2][i])
                    else:
                        is_H = (i + start_with_H) % 2 == 1
                        ths.append(center / (4.0 * (nH if is_H else nL)))
            else:
                # Blend crossover
                n_layers = random.choice([p1[0], p2[0]])
                start_with_H = random.choice([p1[1], p2[1]])
                ths = []
                for i in range(n_layers):
                    if i < len(p1[2]) and i < len(p2[2]):
                        alpha = random.uniform(0.3, 0.7)
                        ths.append(alpha * p1[2][i] + (1-alpha) * p2[2][i])
                    elif i < len(p1[2]):
                        ths.append(p1[2][i])
                    elif i < len(p2[2]):
                        ths.append(p2[2][i])
                    else:
                        is_H = (i + start_with_H) % 2 == 1
                        ths.append(center / (4.0 * (nH if is_H else nL)))
            
            # Mutation
            if random.random() < 0.3:
                # Change n_layers
                n_layers = max(1, min(max_layers, n_layers + random.randint(-2, 2)))
                # Rebuild thicknesses if length changed
                if len(ths) != n_layers:
                    new_ths = []
                    for i in range(n_layers):
                        is_H = (i + start_with_H) % 2 == 1
                        if i < len(ths):
                            new_ths.append(ths[i])
                        else:
                            new_ths.append(center / (4.0 * (nH if is_H else nL)))
                    ths = new_ths
            if random.random() < 0.3:
                # Flip start pattern
                start_with_H = 1 - start_with_H
            # Mutate thicknesses
            for i in range(len(ths)):
                if random.random() < 0.2:
                    ths[i] = ths[i] * random.uniform(0.7, 1.3)
                    ths[i] = max(0.0, min(max_layer, ths[i]))
            
            new_pop.append((n_layers, start_with_H, ths))
        
        population = new_pop
        scores = [eval_individual(ind) for ind in population]
        
        gen_best_idx = int(np.argmin(scores))
        if scores[gen_best_idx] < best_score:
            best_score = scores[gen_best_idx]
            best_ind = population[gen_best_idx]
    
    n_layers, start_with_H, ths = best_ind
    labels = []
    for i in range(n_layers):
        is_H = (i + start_with_H) % 2 == 1
        labels.append('H' if is_H else 'L')
    return labels, [float(t) for t in ths], best_score

def design(problem):
    start = time.time()
    deadline = start + 135.0
    n0 = problem['ambient_index']
    ns = problem['substrate_index']
    nL = problem['materials']['L']
    nH = problem['materials']['H']
    band = problem['band_nm']
    wls = np.linspace(band[0], band[1], 401)
    max_layers = problem['max_layers']
    max_total = problem['max_total_nm']
    max_layer = problem['max_layer_nm']
    
    best_labels = ['L']
    best_th = [band[0] / (4.0 * nL)]
    best_score = _mean_pct([nL], best_th, n0, ns, wls)
    
    # Phase 1: Genetic algorithm
    random.seed(42)
    np.random.seed(42)
    ga_labels, ga_th, ga_score = _ga_optimize(n0, ns, wls, max_layers, max_total, max_layer, deadline, 
                                               pop_size=30, n_gen=60)
    if ga_score < best_score:
        best_score = ga_score
        best_labels = ga_labels
        best_th = ga_th
    
    # Phase 2: Polish the best solution
    if time.time() < deadline:
        pl, pt = _polish_thicknesses(best_labels, best_th, n0, ns, wls, deadline)
        nmat = np.array([nL if l == 'L' else nH for l in pl])
        sc = _mean_pct(nmat, np.array(pt), n0, ns, wls)
        if sc < best_score:
            best_score = sc
            best_labels = pl
            best_th = pt
    
    # Phase 3: Try a few fixed patterns and polish
    if time.time() < deadline:
        center = 0.5 * (band[0] + band[1])
        patterns = [
            # Quarter-wave stack, 10 layers
            (['L','H']*5, [center/(4*nL), center/(4*nH)]*5),
            # Quarter-wave stack, 20 layers
            (['L','H']*10, [center/(4*nL), center/(4*nH)]*10),
            # 3-layer quarter-wave
            (['L','H','L'], [center/(4*nL), center/(4*nH), center/(4*nL)]),
        ]
        for labels, ths in patterns:
            if time.time() > deadline:
                break
            if sum(ths) > max_total:
                continue
            pl, pt = _polish_thicknesses(labels, ths, n0, ns, wls, deadline)
            nmat = np.array([nL if l == 'L' else nH for l in pl])
            sc = _mean_pct(nmat, np.array(pt), n0, ns, wls)
            if sc < best_score:
                best_score = sc
                best_labels = pl
                best_th = pt
    
    # Final constraint check
    if len(best_labels) > max_layers or len(best_labels) != len(best_th):
        best_labels = ['L']; best_th = [band[0] / (4.0 * nL)]
    for t in best_th:
        if t < 0 or t > max_layer:
            best_labels = ['L']; best_th = [band[0] / (4.0 * nL)]; break
    if sum(best_th) > max_total:
        scale = max_total / sum(best_th)
        best_th = [t * scale for t in best_th]
    
    return best_labels, best_th
# EVOLVE-BLOCK-END
