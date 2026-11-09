"""Antireflection coating design: broadband (400-800 nm) low reflectance on glass.

Strategy:
  1. Fast exact characteristic-matrix reflectance model (vectorized over 401 wavelengths).
  2. Focus on symmetric layer patterns (physically optimal for broadband AR).
  3. Differential Evolution for global thickness optimization.
  4. Nelder-Mead local polish on best candidates.
  5. Explicit deadline guard.
"""

import time
import numpy as np
from scipy.optimize import differential_evolution, minimize


def _mean_reflectance(materials, thicknesses, problem):
    """Compute mean reflectance (percent) over the band using characteristic matrices."""
    if not materials:
        return 100.0 * ((problem["substrate_index"] - problem["ambient_index"]) /
                        (problem["substrate_index"] + problem["ambient_index"])) ** 2
    band = problem["band_nm"]
    wavs = np.linspace(band[0], band[1], 401)
    n_mat = {k: v for k, v in problem["materials"].items()}
    ns = np.array([n_mat[m] for m in materials], dtype=float)
    ds = np.asarray(thicknesses, dtype=float)

    # delta_j = 2*pi*n_j*d_j/lambda  shape (N, K)
    delta = 2.0 * np.pi * np.outer(ns * ds, 1.0 / wavs)
    c = np.cos(delta)
    s = np.sin(delta)

    K = len(wavs)
    # Characteristic matrix: start with identity
    M = np.zeros((N, K, 2, 2), dtype=complex) if False else None
    
    # Use 2x2 matrix product per wavelength
    # M_total = prod_j M_j  where M_j = [[cos(delta_j), i*sin(delta_j)/n_j],
    #                                     [i*n_j*sin(delta_j), cos(delta_j)]]
    # Initialize as identity
    A = np.ones(K, dtype=complex)
    B = np.zeros(K, dtype=complex)
    C = np.zeros(K, dtype=complex)
    D = np.ones(K, dtype=complex)

    for j in range(len(materials)):
        n_j = ns[j]
        cj = c[j]
        sj = s[j]
        A_new = A * cj + 1j * B * n_j * sj
        B_new = 1j * A * sj / n_j + B * cj
        C_new = C * cj + 1j * D * n_j * sj
        D_new = 1j * C * sj / n_j + D * cj
        A, B, C, D = A_new, B_new, C_new, D_new

    n_s = problem["substrate_index"]
    n_0 = problem["ambient_index"]
    num = n_0 * (A + B * n_s) - (C + D * n_s)
    den = n_0 * (A + B * n_s) + (C + D * n_s)
    r = num / den
    R = np.abs(r) ** 2
    return 100.0 * np.mean(R)


def design(problem):
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    band = problem["band_nm"]
    wl_center = 0.5 * (band[0] + band[1])
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    materials_dict = problem["materials"]

    # Precompute quarter-wave thicknesses at center wavelength
    qw = {}
    for m, n in materials_dict.items():
        qw[m] = wl_center / (4.0 * n)

    def make_objective(materials_pattern):
        """Create objective function for a fixed material pattern."""
        n = len(materials_pattern)
        bounds = [(0.0, max_layer) for _ in range(n)]
        # Also need total constraint; we'll handle via penalty or scaling
        
        def objective(ths):
            ths = np.asarray(ths, dtype=float)
            # Clip to valid range
            ths = np.clip(ths, 0.0, max_layer)
            total = np.sum(ths)
            if total > max_total:
                # Penalty
                penalty = 10.0 * (total - max_total) / max_total
                return _mean_reflectance(materials_pattern, list(ths), problem) + penalty
            return _mean_reflectance(materials_pattern, list(ths), problem)
        
        return objective, bounds

    def nelder_mead(materials_pattern, init_ths, problem, deadline, max_iters=300):
        """Nelder-Mead simplex local optimization."""
        ths = list(init_ths)
        n = len(ths)
        
        def obj(ths):
            ths = np.clip(ths, 0.0, max_layer)
            total = np.sum(ths)
            if total > max_total:
                return _mean_reflectance(materials_pattern, list(ths), problem) + 10.0 * (total - max_total) / max_total
            return _mean_reflectance(materials_pattern, list(ths), problem)
        
        res = minimize(obj, ths, method='Nelder-Mead', 
                       options={'maxiter': max_iters, 'xatol': 0.1, 'fatol': 1e-5})
        if res.success or res.fun < obj(ths):
            best_ths = np.clip(res.x, 0.0, max_layer)
            best_score = obj(best_ths)
        else:
            best_ths = np.clip(ths, 0.0, max_layer)
            best_score = obj(best_ths)
        
        return materials_pattern, list(best_ths), best_score

    def simulated_annealing(materials_pattern, init_ths, problem, deadline, max_iters=300):
        ths = list(init_ths)
        ths = np.clip(ths, 0.0, max_layer)
        n = len(ths)
        
        def obj(ths):
            ths = np.clip(ths, 0.0, max_layer)
            total = np.sum(ths)
            if total > max_total:
                return _mean_reflectance(materials_pattern, list(ths), problem) + 10.0 * (total - max_total) / max_total
            return _mean_reflectance(materials_pattern, list(ths), problem)
        
        current_score = obj(ths)
        best_score = current_score
        best_ths = list(ths)
        
        T_start = 1.0
        T_end = 0.001
        cooling_rate = (T_end / T_start) ** (1.0 / max_iters)
        
        it = 0
        while it < max_iters and time.time() < deadline:
            it += 1
            T = T_start * (cooling_rate ** it)
            
            idx = np.random.randint(n)
            step = np.random.uniform(-50, 50)
            new_ths = list(ths)
            new_ths[idx] = max(0.0, min(max_layer, ths[idx] + step))
            new_ths = np.clip(new_ths, 0.0, max_layer)
            new_score = obj(new_ths)
            
            delta = new_score - current_score
            if delta < 0 or np.random.random() < np.exp(-delta / (T + 1e-9)):
                ths = new_ths
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_ths = list(ths)
        
        return materials_pattern, best_ths, best_score

    # Generate symmetric candidate patterns
    candidates = []
    
    # 1-layer
    candidates.append(("L",))
    candidates.append(("H",))
    
    # 2-layer
    candidates.append(("L", "H"))
    candidates.append(("H", "L"))
    
    # 3-layer symmetric: A B A
    candidates.append(("L", "H", "L"))
    candidates.append(("H", "L", "H"))
    
    # 4-layer symmetric: A B B A
    candidates.append(("L", "H", "H", "L"))
    candidates.append(("H", "L", "L", "H"))
    
    # 5-layer symmetric: A B C B A
    for a in ("L", "H"):
        for b in ("L", "H"):
            for c in ("L", "H"):
                candidates.append((a, b, c, b, a))
    
    # 6-layer symmetric: A B C C B A
    for a in ("L", "H"):
        for b in ("L", "H"):
            for c in ("L", "H"):
                candidates.append((a, b, c, c, b, a))
    
    # 7-layer symmetric: A B C D C B A
    for a in ("L", "H"):
        for b in ("L", "H"):
            for c in ("L", "H"):
                for d in ("L", "H"):
                    candidates.append((a, b, c, d, c, b, a))
    
    # 8-layer symmetric: A B C D D C B A
    for a in ("L", "H"):
        for b in ("L", "H"):
            for c in ("L", "H"):
                for d in ("L", "H"):
                    candidates.append((a, b, c, d, d, c, b, a))
    
    # 9-layer symmetric: A B C D E D C B A
    for a in ("L", "H"):
        for b in ("L", "H"):
            for c in ("L", "H"):
                for d in ("L", "H"):
                    for e in ("L", "H"):
                        candidates.append((a, b, c, d, e, d, c, b, a))
    
    # 10-layer symmetric: A B C D E E D C B A
    for a in ("L", "H"):
        for b in ("L", "H"):
            for c in ("L", "H"):
                for d in ("L", "H"):
                    for e in ("L", "H"):
                        candidates.append((a, b, c, d, e, e, d, c, b, a))
    
    # Remove duplicates, sort by length
    seen = set()
    unique_candidates = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            unique_candidates.append(p)
    candidates = unique_candidates
    candidates.sort(key=lambda p: len(p))
    
    best_result = None
    best_score = float('inf')
    
    # Phase 1: Differential Evolution on top symmetric patterns
    # Focus on patterns with 3-10 layers (most promising for broadband AR)
    de_patterns = [p for p in candidates if 3 <= len(p) <= 10]
    
    np.random.seed(42)
    
    for pattern in de_patterns:
        if time.time() >= deadline - 60:
            break
        
        n = len(pattern)
        # Initial guess: quarter-wave thicknesses
        init_ths = [qw[m] for m in pattern]
        bounds = [(0.0, max_layer) for _ in range(n)]
        
        def obj(ths):
            ths = np.clip(ths, 0.0, max_layer)
            total = np.sum(ths)
            if total > max_total:
                return _mean_reflectance(pattern, list(ths), problem) + 10.0 * (total - max_total) / max_total
            return _mean_reflectance(pattern, list(ths), problem)
        
        try:
            # Use differential evolution with limited population for speed
            popsize = max(15, 2 * n)
            res = differential_evolution(
                obj, bounds, 
                maxiter=50, popsize=popsize,
                tol=1e-4, mutation=(0.3, 0.9), recombination=0.7,
                seed=42, init='latinhypercube',
                polish=False,  # We'll do NM separately
                workers=1
            )
            if res.success or res.fun < obj(init_ths):
                best_ths = np.clip(res.x, 0.0, max_layer)
                score = obj(best_ths)
                if score < best_score:
                    best_score = score
                    best_result = (list(pattern), list(best_ths))
        except Exception:
            pass
    
    # Phase 2: SA on remaining patterns and to explore more
    remaining_patterns = [p for p in candidates if p not in de_patterns]
    all_patterns = de_patterns + remaining_patterns
    
    for pattern in all_patterns:
        if time.time() >= deadline - 30:
            break
        
        n = len(pattern)
        init_ths = [qw[m] for m in pattern]
        
        mats, ths, score = simulated_annealing(pattern, init_ths, problem, deadline - 30, max_iters=150)
        if score < best_score:
            best_score = score
            best_result = (mats, ths)
    
    # Phase 3: Polish the best result with Nelder-Mead
    if best_result is not None:
        if time.time() < deadline - 5:
            mats, ths, score = nelder_mead(best_result[0], best_result[1], problem, deadline - 5, max_iters=200)
            if score < best_score:
                best_score = score
                best_result = (mats, ths)
    
    # Also try polishing a few other good candidates
    if time.time() < deadline - 10:
        # Re-evaluate top patterns with NM
        for pattern in de_patterns[:5]:
            if time.time() >= deadline - 5:
                break
            n = len(pattern)
            init_ths = [qw[m] for m in pattern]
            mats, ths, score = nelder_mead(pattern, init_ths, problem, deadline - 5, max_iters=150)
            if score < best_score:
                best_score = score
                best_result = (mats, ths)
    
    if best_result is None:
        return ["L"], [qw["L"]]
    
    mats, ths = best_result
    # Final validation: ensure constraints are met
    ths = [max(0.0, min(max_layer, t)) for t in ths]
    total = sum(ths)
    if total > max_total:
        scale = max_total / total
        ths = [t * scale for t in ths]
    
    return list(mats), list(ths)
