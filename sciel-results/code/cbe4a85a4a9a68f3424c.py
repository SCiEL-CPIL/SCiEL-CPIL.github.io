"""Broadband antireflection coating design via differential evolution + Nelder-Mead polish."""
import time
import numpy as np
import random

# EVOLVE-BLOCK-START

def _compute_mean_reflectance(labels, thicknesses, n_amb, n_sub, n_L, n_H, band_nm):
    """Compute mean reflectance in percent over 401 equally spaced wavelengths in band_nm."""
    lam_min, lam_max = band_nm
    wavelengths = np.linspace(lam_min, lam_max, 401)
    
    n_mat = np.array([n_L if l == 'L' else n_H for l in labels], dtype=float)
    d = np.array(thicknesses, dtype=float)
    
    if len(labels) == 0:
        r = (n_amb - n_sub) / (n_amb + n_sub)
        return np.abs(r) ** 2 * 100.0
    
    beta = 2.0 * np.pi * n_mat * d
    
    total_r = 0.0
    n_wl = len(wavelengths)
    chunk = 40
    for start in range(0, n_wl, chunk):
        end = min(start + chunk, n_wl)
        lams = wavelengths[start:end]
        nl = len(lams)
        
        M00 = np.ones(nl, dtype=complex)
        M01 = np.zeros(nl, dtype=complex)
        M10 = np.zeros(nl, dtype=complex)
        M11 = np.ones(nl, dtype=complex)
        
        for i in range(len(labels)):
            delta = beta[i] / lams
            c = np.cos(delta)
            s = np.sin(delta)
            
            a = c
            b = 1j * s / n_mat[i]
            cc = 1j * n_mat[i] * s
            dd = c
            
            new_M00 = M00 * a + M01 * cc
            new_M01 = M00 * b + M01 * dd
            new_M10 = M10 * a + M11 * cc
            new_M11 = M10 * b + M11 * dd
            M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
        
        Y = M10 / M00
        r = (n_amb - Y) / (n_amb + Y)
        R = np.abs(r) ** 2 * 100.0
        total_r += np.sum(R)
    
    return total_r / n_wl


def _differential_evolution(f, bounds, pop_size=50, max_iter=200, F=0.8, CR=0.9,
                            time_deadline=None, seed=42):
    """
    Differential evolution for global search.
    bounds: list of (lo, hi) tuples
    Returns best_x, best_f
    """
    rng = np.random.RandomState(seed)
    n = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    
    # Initialize population
    pop = np.zeros((pop_size, n))
    for i in range(pop_size):
        for j in range(n):
            pop[i, j] = rng.uniform(lo[j], hi[j])
    
    fvals = np.array([f(pop[i]) for i in range(pop_size)])
    best_idx = np.argmin(fvals)
    best_x = pop[best_idx].copy()
    best_f = fvals[best_idx]
    
    for it in range(max_iter):
        if time_deadline is not None and time.time() > time_deadline:
            break
        
        for i in range(pop_size):
            if time_deadline is not None and time.time() > time_deadline:
                break
            
            # Select three distinct random individuals different from i
            r1, r2, r3 = i
            while r1 == i:
                r1 = rng.randint(0, pop_size - 1)
            while r2 == i or r2 == r1:
                r2 = rng.randint(0, pop_size - 1)
            while r3 == i or r3 == r1 or r3 == r2:
                r3 = rng.randint(0, pop_size - 1)
            
            # Create trial vector
            trial = pop[r1] + F * (pop[r2] - pop[r3])
            
            # Crossover
            for j in range(n):
                if rng.random() < CR:
                    pass  # keep trial value
                else:
                    trial[j] = pop[i, j]
            
            # Ensure at least one gene is mutated
            if np.allclose(trial, pop[i]):
                j_rand = rng.randint(0, n)
                trial[j_rand] = pop[r1, j_rand] + F * (pop[r2, j_rand] - pop[r3, j_rand])
            
            # Clip to bounds
            trial = np.clip(trial, lo, hi)
            
            f_trial = f(trial)
            
            if f_trial <= fvals[i]:
                pop[i] = trial
                fvals[i] = f_trial
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = trial.copy()
    
    return best_x, best_f


def _nelder_mead(f, x0, bounds, max_iter=300, time_deadline=None):
    """
    Nelder-Mead simplex with bounds.
    bounds: list of (lo, hi) tuples
    """
    n = len(x0)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    x0 = np.clip(np.asarray(x0, dtype=float), lo, hi)
    
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    
    for i in range(n):
        simplex[i + 1] = x0.copy()
        step = 0.1 * (hi[i] - lo[i]) if hi[i] > lo[i] else 0.1
        simplex[i + 1][i] += step
    
    fvals = np.array([f(simplex[i]) for i in range(n + 1)])
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    for it in range(max_iter):
        if time_deadline is not None and time.time() > time_deadline:
            break
        
        idx = np.argsort(fvals)
        simplex = simplex[idx]
        fvals = fvals[idx]
        
        if np.max(fvals) - np.min(fvals) < 1e-9:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        x_r = centroid + alpha * (centroid - simplex[-1])
        x_r = np.clip(x_r, lo, hi)
        f_r = f(x_r)
        
        if f_r < fvals[0]:
            x_e = centroid + gamma * (x_r - centroid)
            x_e = np.clip(x_e, lo, hi)
            f_e = f(x_e)
            
            if f_e < f_r:
                simplex[-1] = x_e
                fvals[-1] = f_e
            else:
                simplex[-1] = x_r
                fvals[-1] = f_r
        elif f_r < fvals[-2]:
            simplex[-1] = x_r
            fvals[-1] = f_r
        else:
            x_c = centroid + rho * (simplex[-1] - centroid)
            x_c = np.clip(x_c, lo, hi)
            f_c = f(x_c)
            
            if f_c < fvals[-1]:
                simplex[-1] = x_c
                fvals[-1] = f_c
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], lo, hi)
                    fvals[i] = f(simplex[i])
    
    best_idx = np.argmin(fvals)
    return simplex[best_idx], fvals[best_idx]


def design(problem):
    """Design a broadband antireflection coating stack."""
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin before 150s kill
    
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    best_labels = None
    best_thicknesses = None
    best_score = float('inf')
    
    centre = 0.5 * (band_nm[0] + band_nm[1])
    
    def _make_obj(labels):
        """Create objective function for a given label sequence."""
        def obj(x):
            x = np.clip(x, 0.0, max_layer_nm)
            total = np.sum(x)
            if total > max_total_nm:
                penalty = 50.0 * (total - max_total_nm) / max_total_nm
            else:
                penalty = 0.0
            val = _compute_mean_reflectance(labels, list(x), n_amb, n_sub, n_L, n_H, band_nm)
            return val + penalty
        return obj
    
    def _is_feasible(x):
        return all(0.0 <= t <= max_layer_nm for t in x) and sum(x) <= max_total_nm + 1e-9
    
    # Build candidate sequences
    candidate_sequences = []
    seen = set()
    
    def _add_seq(seq):
        key = tuple(seq)
        if key not in seen and len(seq) <= max_layers:
            seen.add(key)
            candidate_sequences.append(list(seq))
    
    # Alternating sequences
    for n_layers in range(1, min(max_layers + 1, 15)):
        seq_L = ["L" if i % 2 == 0 else "H" for i in range(n_layers)]
        seq_H = ["H" if i % 2 == 0 else "L" for i in range(n_layers)]
        _add_seq(seq_L)
        _add_seq(seq_H)
    
    # Some structured non-alternating patterns
    extra_patterns = [
        ["L", "H", "L", "L", "H"],
        ["H", "L", "H", "H", "L"],
        ["L", "L", "H", "H", "L"],
        ["H", "H", "L", "L", "H"],
        ["L", "H", "H", "L"],
        ["H", "L", "L", "H"],
        ["L", "H", "L", "L"],
        ["H", "L", "H", "H"],
        ["L", "H", "H", "L", "H"],
        ["H", "L", "L", "H", "L"],
        ["L", "H", "L", "H", "L", "H"],
        ["H", "L", "H", "L", "H", "L"],
        ["L", "H", "L", "H", "L", "H", "L"],
        ["H", "L", "H", "L", "H", "L", "H"],
    ]
    for p in extra_patterns:
        _add_seq(p)
    
    # Phase 1: Quick evaluation of initial quarter-wave configurations
    for labels in candidate_sequences:
        if time.time() > deadline - 30.0:
            break
        
        if len(labels) > max_layers:
            continue
        
        init_thick = [centre / (4.0 * (n_L if l == 'L' else n_H)) for l in labels]
        if sum(init_thick) > max_total_nm:
            scale = max_total_nm / sum(init_thick)
            init_thick = [t * scale for t in init_thick]
        
        if _is_feasible(init_thick):
            score = _compute_mean_reflectance(labels, init_thick, n_amb, n_sub, n_L, n_H, band_nm)
            if score < best_score:
                best_score = score
                best_labels = labels[:]
                best_thicknesses = init_thick[:]
    
    # Phase 2: Differential evolution + Nelder-Mead for each promising sequence
    # Prioritize shorter sequences for DE (lower dimensionality), then longer ones
    sorted_seqs = sorted(candidate_sequences, key=lambda s: len(s))
    
    de_budget_per_seq = 3.0  # seconds per sequence for DE
    nm_budget_per_seq = 2.0  # seconds per sequence for NM polish
    
    for labels in sorted_seqs:
        if time.time() > deadline - 15.0:
            break
        
        if len(labels) > max_layers:
            continue
        
        n = len(labels)
        bounds = [(0.0, max_layer_nm)] * n
        
        # Create objective
        obj = _make_obj(labels)
        
        # Differential evolution
        de_deadline = time.time() + de_budget_per_seq
        if de_deadline > deadline - 10.0:
            de_deadline = deadline - 10.0
        
        # For small n, use more population
        if n <= 4:
            pop_size = min(60, 20 * n)
            max_iter_de = 150
        elif n <= 8:
            pop_size = min(50, 15 * n)
            max_iter_de = 100
        else:
            pop_size = min(40, 10 * n)
            max_iter_de = 80
        
        de_x, de_f = _differential_evolution(
            obj, bounds, pop_size=pop_size, max_iter=max_iter_de,
            time_deadline=de_deadline, seed=42 + len(labels)
        )
        
        # Nelder-Mead polish on DE result
        nm_deadline = time.time() + nm_budget_per_seq
        if nm_deadline > deadline - 8.0:
            nm_deadline = deadline - 8.0
        
        nm_x, nm_f = _nelder_mead(
            obj, de_x, bounds, max_iter=200, time_deadline=nm_deadline
        )
        
        # Also try NM from quarter-wave initial point
        init_thick = np.array([centre / (4.0 * (n_L if l == 'L' else n_H)) for l in labels])
        init_thick = np.clip(init_thick, 0.0, max_layer_nm)
        if np.sum(init_thick) > max_total_nm:
            init_thick = init_thick * (max_total_nm / np.sum(init_thick))
            init_thick = np.clip(init_thick, 0.0, max_layer_nm)
        
        nm2_deadline = time.time() + nm_budget_per_seq * 0.5
        if nm2_deadline > deadline - 6.0:
            nm2_deadline = deadline - 6.0
        
        nm2_x, nm2_f = _nelder_mead(
            obj, init_thick, bounds, max_iter=150, time_deadline=nm2_deadline
        )
        
        # Check all three candidates
        for x_candidate in [de_x, nm_x, nm2_x]:
            if _is_feasible(x_candidate):
                actual_score = _compute_mean_reflectance(labels, list(x_candidate), n_amb, n_sub, n_L, n_H, band_nm)
                if actual_score < best_score:
                    best_score = actual_score
                    best_labels = labels[:]
                    best_thicknesses = list(x_candidate)
    
    # Phase 3: Random restarts with different sequences if time remains
    if time.time() < deadline - 20.0:
        random.seed(42)
        for trial in range(50):
            if time.time() > deadline - 10.0:
                break
            
            n_layers = random.randint(2, min(12, max_layers))
            labels = [random.choice(['L', 'H']) for _ in range(n_layers)]
            
            key = tuple(labels)
            if key in seen:
                continue
            seen.add(key)
            
            n = len(labels)
            bounds = [(0.0, max_layer_nm)] * n
            obj = _make_obj(labels)
            
            de_deadline = time.time() + 2.0
            if de_deadline > deadline - 8.0:
                de_deadline = deadline - 8.0
            
            pop_size = min(40, 10 * n)
            de_x, de_f = _differential_evolution(
                obj, bounds, pop_size=pop_size, max_iter=80,
                time_deadline=de_deadline, seed=42 + trial
            )
            
            nm_deadline = time.time() + 1.5
            if nm_deadline > deadline - 5.0:
                nm_deadline = deadline - 5.0
            
            nm_x, nm_f = _nelder_mead(
                obj, de_x, bounds, max_iter=150, time_deadline=nm_deadline
            )
            
            for x_candidate in [de_x, nm_x]:
                if _is_feasible(x_candidate):
                    actual_score = _compute_mean_reflectance(labels, list(x_candidate), n_amb, n_sub, n_L, n_H, band_nm)
                    if actual_score < best_score:
                        best_score = actual_score
                        best_labels = labels[:]
                        best_thicknesses = list(x_candidate)
    
    # Phase 4: Final refinement on the best solution
    if time.time() < deadline - 8.0 and best_labels is not None:
        labels = best_labels
        n = len(labels)
        bounds = [(0.0, max_layer_nm)] * n
        obj = _make_obj(labels)
        
        nm_deadline = deadline - 3.0
        nm_x, nm_f = _nelder_mead(
            obj, np.array(best_thicknesses), bounds, max_iter=300, time_deadline=nm_deadline
        )
        
        if _is_feasible(nm_x):
            actual_score = _compute_mean_reflectance(labels, list(nm_x), n_amb, n_sub, n_L, n_H, band_nm)
            if actual_score < best_score:
                best_score = actual_score
                best_thicknesses = list(nm_x)
    
    # Return the best result found
    if best_labels is None:
        centre_val = 0.5 * (band_nm[0] + band_nm[1])
        best_labels = ["L"]
        best_thicknesses = [centre_val / (4.0 * n_L)]
    
    # Final constraint enforcement
    best_thicknesses = [max(0.0, min(t, max_layer_nm)) for t in best_thicknesses]
    if sum(best_thicknesses) > max_total_nm:
        scale = max_total_nm / sum(best_thicknesses)
        best_thicknesses = [t * scale for t in best_thicknesses]
    
    return best_labels, best_thicknesses

# EVOLVE-BLOCK-END
