"""Broadband antireflection coating design: DE over fixed layer counts + Nelder-Mead polish."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def _reflectance(labels, thicknesses, wls, nL, nH, ns):
    """Mean reflectance (fraction) of a stack at normal incidence over wavelengths wls."""
    n = len(labels)
    if n == 0:
        r0 = (1.0 - ns) / (1.0 + ns)
        return r0 * r0
    nmat = np.array([nL if lb == 'L' else nH for lb in labels], dtype=float)
    t = np.asarray(thicknesses, dtype=float)
    nw = len(wls)
    M = np.empty((2, 2, nw), dtype=complex)
    M[0, 0, :] = 1.0 + 0j
    M[0, 1, :] = 0.0 + 0j
    M[1, 0, :] = 0.0 + 0j
    M[1, 1, :] = 1.0 + 0j
    for j in range(n):
        phase = (2.0 * math.pi / np.asarray(wls, dtype=float)) * nmat[j] * t[j]
        c = np.cos(phase)
        s = np.sin(phase)
        nj = nmat[j]
        m00 = M[0, 0, :]
        m01 = M[0, 1, :]
        m10 = M[1, 0, :]
        m11 = M[1, 1, :]
        L00 = c
        L01 = 1j * s / nj
        L10 = 1j * nj * s
        L11 = c
        n00 = m00 * L00 + m01 * L10
        n01 = m00 * L01 + m01 * L11
        n10 = m10 * L00 + m11 * L10
        n11 = m10 * L01 + m11 * L11
        M[0, 0, :] = n00
        M[0, 1, :] = n01
        M[1, 0, :] = n10
        M[1, 1, :] = n11
    B = M[0, 0, :] + M[0, 1, :] * ns
    C = M[1, 0, :] + M[1, 1, :] * ns
    n0 = 1.0
    num = (B - n0 * C)
    den = (B + n0 * C)
    R = (num * np.conj(num)) / (den * np.conj(den))
    R = np.real(R)
    R = np.clip(R, 0.0, 1.0)
    return float(np.mean(R))

def _seed_stack(n, nL, nH, band0, band1):
    """Alternating L/H starting with L, quarter-wave at band centre."""
    centre = 0.5 * (band0 + band1)
    labels = []
    thicks = []
    for i in range(n):
        lb = 'L' if i % 2 == 0 else 'H'
        nmat = nL if lb == 'L' else nH
        labels.append(lb)
        thicks.append(centre / (4.0 * nmat))
    return labels, thicks

def _validate(labels, thicknesses, max_layers, max_total_nm, max_layer_nm):
    if len(labels) > max_layers:
        return False
    if len(labels) != len(thicknesses):
        return False
    for t in thicknesses:
        if t < 0.0 or t > max_layer_nm + 1e-9:
            return False
    if sum(thicknesses) > max_total_nm + 1e-9:
        return False
    for lb in labels:
        if lb not in ('L', 'H'):
            return False
    return True

def _nelder_mead(f, x0, bounds, maxiter=1500, xatol=1e-6, fatol=1e-9):
    """Simplex-based Nelder-Mead optimizer with bounds enforcement."""
    n = len(x0)
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        xi = x0.copy()
        step = 0.1 * (bounds[i][1] - bounds[i][0]) if bounds[i][1] > bounds[i][0] else 0.05
        xi[i] += step
        simplex.append(xi)
    
    for i in range(len(simplex)):
        for j in range(n):
            simplex[i][j] = np.clip(simplex[i][j], bounds[j][0], bounds[j][1])
    
    for i in range(len(simplex)):
        simplex[i] = np.concatenate([simplex[i], [f(simplex[i][:n])]])
    
    for _ in range(maxiter):
        simplex.sort(key=lambda x: x[-1])
        if abs(simplex[0][-1] - simplex[-1][-1]) < fatol:
            break
        if np.max(np.abs(simplex[0][:n] - simplex[-1][:n])) < xatol:
            break
        
        centroid = np.mean([simplex[i][:n] for i in range(len(simplex) - 1)], axis=0)
        xr = centroid + alpha * (centroid - simplex[-1][:n])
        xr = np.clip(xr, [b[0] for b in bounds], [b[1] for b in bounds])
        fr = f(xr)
        
        if fr < simplex[0][-1]:
            xe = centroid + gamma * (xr - centroid)
            xe = np.clip(xe, [b[0] for b in bounds], [b[1] for b in bounds])
            fe = f(xe)
            if fe < fr:
                simplex[-1] = np.concatenate([xe, [fe]])
            else:
                simplex[-1] = np.concatenate([xr, [fr]])
        elif fr < simplex[-2][-1]:
            simplex[-1] = np.concatenate([xr, [fr]])
        else:
            xc = centroid + rho * (simplex[-1][:n] - centroid)
            xc = np.clip(xc, [b[0] for b in bounds], [b[1] for b in bounds])
            fc = f(xc)
            if fc < simplex[-1][-1]:
                simplex[-1] = np.concatenate([xc, [fc]])
            else:
                for i in range(1, len(simplex)):
                    simplex[i][:n] = simplex[0][:n] + sigma * (simplex[i][:n] - simplex[0][:n])
                    simplex[i][:n] = np.clip(simplex[i][:n], [b[0] for b in bounds], [b[1] for b in bounds])
                    simplex[i][-1] = f(simplex[i][:n])
    
    simplex.sort(key=lambda x: x[-1])
    return simplex[0][:n], simplex[0][-1]

def _differential_evolution(f, bounds, pop_size=40, maxiter=50, F=0.8, CR=0.9, seed=42, time_limit=None, start_time=None):
    """Differential evolution optimizer."""
    np.random.seed(seed)
    n = len(bounds)
    
    pop = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], size=(pop_size, n))
    fitness = np.array([f(ind) for ind in pop])
    
    best_idx = np.argmin(fitness)
    best_x = pop[best_idx].copy()
    best_f = fitness[best_idx]
    
    for gen in range(maxiter):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        for i in range(pop_size):
            a, b, c = np.random.choice(pop_size, size=3, replace=False)
            while a == b or a == c or b == c:
                a, b, c = np.random.choice(pop_size, size=3, replace=False)
            
            v = pop[a] + F * (pop[b] - pop[c])
            v = np.clip(v, [bd[0] for bd in bounds], [bd[1] for bd in bounds])
            
            trial = pop[i].copy()
            j_rand = np.random.randint(n)
            for j in range(n):
                if np.random.rand() < CR or j == j_rand:
                    trial[j] = v[j]
            
            trial = np.clip(trial, [bd[0] for bd in bounds], [bd[1] for bd in bounds])
            
            ft = f(trial)
            if ft <= fitness[i]:
                pop[i] = trial
                fitness[i] = ft
                if ft < best_f:
                    best_f = ft
                    best_x = trial.copy()
    
    return best_x, best_f

def design(problem):
    t_start = time.time()
    DEADLINE = 145.0
    SAFETY = 5.0
    band0, band1 = problem["band_nm"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    ns = problem["substrate_index"]
    n0 = problem.get("ambient_index", 1.0)
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]

    wls = np.linspace(band0, band1, 401)

    best_score = float('inf')
    best_labels = None
    best_thicks = None

    def eval_stack(labels, thicknesses):
        nonlocal best_score, best_labels, best_thicks
        if not _validate(labels, thicknesses, max_layers, max_total_nm, max_layer_nm):
            return float('inf')
        r = _reflectance(labels, thicknesses, wls, nL, nH, ns)
        if r < best_score:
            best_score = r
            best_labels = list(labels)
            best_thicks = list(thicknesses)
        return r

    def make_objective(labels):
        n = len(labels)
        bounds = [(0.0, max_layer_nm)] * n
        def obj(thicks):
            if sum(thicks) > max_total_nm + 1e-6:
                return 1.0 + (sum(thicks) - max_total_nm) * 0.01
            return _reflectance(labels, thicks, wls, nL, nH, ns)
        return obj, bounds

    # Focus on 3, 5, 7 layers which are physically sensible for broadband AR
    layer_counts = [3, 5, 7]
    
    # Time allocation: DE gets 60%, NM gets 40%
    total_budget = DEADLINE - t_start - SAFETY
    de_budget = total_budget * 0.6
    nm_budget = total_budget * 0.4
    
    de_start = time.time()
    
    for c in layer_counts:
        if time.time() + SAFETY > DEADLINE:
            break
        
        # Alternate patterns
        if c % 2 == 1:
            labels = ['L' if i % 2 == 0 else 'H' for i in range(c)]
        else:
            labels = ['L' if i % 2 == 0 else 'H' for i in range(c)]
        
        obj, bounds = make_objective(labels)
        
        # DE search
        time_limit_de = min(de_budget / len(layer_counts), 30.0)
        best_x, best_f = _differential_evolution(
            obj, bounds, pop_size=30, maxiter=40, F=0.8, CR=0.9, seed=42,
            time_limit=time_limit_de, start_time=de_start
        )
        
        # Enforce total thickness constraint by scaling
        if sum(best_x) > max_total_nm:
            s = max_total_nm / sum(best_x)
            best_x = best_x * s
        
        eval_stack(labels, best_x)
        
        # Nelder-Mead polish
        if time.time() + SAFETY < DEADLINE:
            nm_time_limit = min(nm_budget / len(layer_counts), 15.0)
            nm_start = time.time()
            def nm_obj(thicks):
                if time.time() - nm_start > nm_time_limit:
                    return 1.0
                if sum(thicks) > max_total_nm + 1e-6:
                    return 1.0 + (sum(thicks) - max_total_nm) * 0.01
                return _reflectance(labels, thicks, wls, nL, nH, ns)
            
            # Use a simpler NM that respects time
            x_nm, f_nm = _nelder_mead(nm_obj, best_x, bounds, maxiter=500, xatol=1e-5, fatol=1e-8)
            
            if sum(x_nm) > max_total_nm:
                s = max_total_nm / sum(x_nm)
                x_nm = x_nm * s
            
            eval_stack(labels, x_nm)
    
    # Final polish on global best
    if best_labels is not None and time.time() + SAFETY < DEADLINE:
        obj, bounds = make_objective(best_labels)
        nm_time_limit = max(0.5, DEADLINE - SAFETY - time.time())
        nm_start = time.time()
        def nm_obj_final(thicks):
            if time.time() - nm_start > nm_time_limit:
                return 1.0
            if sum(thicks) > max_total_nm + 1e-6:
                return 1.0 + (sum(thicks) - max_total_nm) * 0.01
            return _reflectance(best_labels, thicks, wls, nL, nH, ns)
        
        x_final, f_final = _nelder_mead(nm_obj_final, np.array(best_thicks), bounds, maxiter=1000, xatol=1e-6, fatol=1e-9)
        
        if sum(x_final) > max_total_nm:
            s = max_total_nm / sum(x_final)
            x_final = x_final * s
        
        eval_stack(best_labels, x_final)

    # Fallback
    if best_labels is None:
        centre = 0.5 * (band0 + band1)
        best_labels = ['L']
        best_thicks = [centre / (4.0 * nL)]

    best_thicks = [max(0.0, min(max_layer_nm, float(t))) for t in best_thicks]
    if sum(best_thicks) > max_total_nm:
        s = max_total_nm / sum(best_thicks)
        best_thicks = [t * s for t in best_thicks]

    return best_labels, best_thicks

# EVOLVE-BLOCK-END
