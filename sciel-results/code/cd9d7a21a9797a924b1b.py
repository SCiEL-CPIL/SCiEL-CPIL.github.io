"""Broadband antireflection coating design via Simulated Annealing on continuous index space."""
import time
import numpy as np
import random

# EVOLVE-BLOCK-START

def _mean_R(labels, thicknesses, n_amb, n_sub, n_L, n_H, band_nm, n_wl=401):
    """Mean reflectance (%) over n_wl equally spaced wavelengths in band_nm."""
    lam_min, lam_max = band_nm
    lams = np.linspace(lam_min, lam_max, n_wl)
    n_mat = np.array([n_L if l == 'L' else n_H for l in labels], dtype=float)
    d = np.array(thicknesses, dtype=float)
    if len(labels) == 0:
        r = (n_amb - n_sub) / (n_amb + n_sub)
        return float(np.abs(r) ** 2 * 100.0)
    beta = 2.0 * np.pi * n_mat * d
    total = 0.0
    chunk = 64
    for s in range(0, n_wl, chunk):
        e = min(s + chunk, n_wl)
        L = lams[s:e]
        m = L.shape[0]
        M00 = np.ones(m, dtype=complex); M01 = np.zeros(m, dtype=complex)
        M10 = np.zeros(m, dtype=complex); M11 = np.ones(m, dtype=complex)
        for i in range(len(labels)):
            delta = beta[i] / L
            c = np.cos(delta); sn = np.sin(delta)
            b = 1j * sn / n_mat[i]
            cc = 1j * n_mat[i] * sn
            n00 = M00 * c + M01 * cc
            n01 = M00 * b + M01 * c
            n10 = M10 * c + M11 * cc
            n11 = M10 * b + M11 * c
            M00, M01, M10, M11 = n00, n01, n10, n11
        Y = M10 / M00
        r = (n_amb - Y) / (n_amb + Y)
        total += np.sum(np.abs(r) ** 2 * 100.0)
    return total / n_wl

def _fast_mean_R_continuous(n_cont, d, n_amb, n_sub, n_L, n_H, band_nm, n_wl=401):
    """
    Fast evaluation for continuous index values.
    n_cont: array of continuous refractive indices.
    d: array of thicknesses.
    We map continuous indices to effective indices for calculation, 
    but for SA we just evaluate the physical stack defined by the current 
    rounded indices and thicknesses to keep the objective landscape consistent 
    with the final discrete problem.
    """
    # Round to nearest valid material index for the objective function
    # This creates a step function, which is hard for SA.
    # Alternative: Use the continuous values directly in the matrix method?
    # The problem specifies discrete materials. The evaluator uses discrete.
    # If we use continuous in the objective, we might find a "good" continuous 
    # solution that maps to a bad discrete one.
    # However, SA on discrete topology is hard.
    # Let's stick to discrete topology but optimize thicknesses with SA?
    # No, the prompt asks for a DIFFERENT strategy.
    # Strategy: SA on discrete topology (L/H) and continuous thicknesses.
    # We represent the state as (labels, thicknesses).
    # Move: 1. Flip a random label. 2. Perturb a random thickness.
    
    labels = ['L' if n < 1.865 else 'H' for n in n_cont] # Threshold 1.865 is midpoint of 1.38 and 2.35
    return _mean_R(labels, d, n_amb, n_sub, n_L, n_H, band_nm, n_wl)

def design(problem):
    start = time.time()
    deadline = start + 140.0  # 10s safety margin for 150s limit

    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]

    centre = 0.5 * (band_nm[0] + band_nm[1])
    
    # We will fix the number of layers to a reasonable high number to allow 
    # the optimizer to "turn off" layers by setting thickness to 0.
    # Using max_layers=20 allows the full search space.
    N = max_layers
    
    best_labels = None
    best_thick = None
    best_score = float('inf')
    
    def upd(labels, thick, score):
        nonlocal best_labels, best_thick, best_score
        if score < best_score:
            # Filter out zero-thickness layers for the final representation?
            # No, the evaluator accepts 0 thickness. 
            # But it's cleaner to keep the list length consistent or trim.
            # The problem says "list of labels" and "list of thicknesses".
            # It doesn't forbid 0 thickness.
            # However, "stack of at most twenty layers" implies we can have fewer.
            # Let's keep the full length in memory but return trimmed if beneficial?
            # Actually, returning 0 thickness layers is valid.
            # Let's just store them.
            best_labels = labels[:]
            best_thick = thick[:]
            best_score = score

    rng = random.Random(42)
    np_rng = np.random.default_rng(42)
    
    # Initial state: Random alternating or random mix
    # Start with a reasonable guess: Alternating L/H with quarter-wave thicknesses
    # This is a strong starting point.
    cur_labels = ['L' if i % 2 == 0 else 'H' for i in range(N)]
    cur_thick = np.array([centre / (4.0 * (n_L if l == 'L' else n_H)) for l in cur_labels])
    cur_thick = np.clip(cur_thick, 0.0, max_layer_nm)
    # Scale to fit total thickness if needed
    if cur_thick.sum() > max_total_nm:
        cur_thick *= (max_total_nm / cur_thick.sum())
        
    cur_score = _mean_R(cur_labels, cur_thick, n_amb, n_sub, n_L, n_H, band_nm, n_wl=401)
    upd(cur_labels, cur_thick, cur_score)
    
    best_score_sa = cur_score
    best_labels_sa = cur_labels[:]
    best_thick_sa = cur_thick.copy()
    
    # Simulated Annealing Parameters
    T_start = 5.0  # Initial temperature (in percent reflectance units)
    T_end = 0.01
    alpha = 0.99  # Cooling rate
    # We need to run until time runs out
    
    while time.time() < deadline - 5.0:
        # Generate a neighbor
        r = rng.random()
        if r < 0.5:
            # Flip a random layer's material
            i = rng.randint(0, N - 1)
            new_labels = cur_labels[:]
            new_labels[i] = 'H' if new_labels[i] == 'L' else 'L'
            new_thick = cur_thick.copy()
        else:
            # Perturb a random thickness
            i = rng.randint(0, N - 1)
            new_labels = cur_labels[:]
            new_thick = cur_thick.copy()
            # Gaussian perturbation
            sigma = max_layer_nm * 0.1
            new_thick[i] += np_rng.normal(0, sigma)
            new_thick[i] = np.clip(new_thick[i], 0.0, max_layer_nm)
            # Handle total thickness constraint: if sum > max, scale down
            if new_thick.sum() > max_total_nm:
                new_thick *= (max_total_nm / new_thick.sum())
        
        new_score = _mean_R(new_labels, new_thick, n_amb, n_sub, n_L, n_H, band_nm, n_wl=401)
        
        # Acceptance criterion
        if new_score < cur_score:
            # Always accept improvement
            cur_labels = new_labels
            cur_thick = new_thick
            cur_score = new_score
            if cur_score < best_score_sa:
                best_score_sa = cur_score
                best_labels_sa = cur_labels[:]
                best_thick_sa = cur_thick.copy()
                upd(cur_labels, cur_thick, cur_score)
        else:
            # Metropolis criterion
            delta_E = new_score - cur_score
            if T_start > 0: # Avoid log(0) or division by zero issues
                accept_prob = np.exp(-delta_E / max(T_start, 1e-9))
                if rng.random() < accept_prob:
                    cur_labels = new_labels
                    cur_thick = new_thick
                    cur_score = new_score
        
        # Cool down
        T_start *= alpha
        
        # If we've cooled too much and are stuck, reheat slightly? 
        # Or just continue. With 150s budget, we can do many iterations.
        
    # Final Polish: Use Nelder-Mead on the best discrete topology found by SA
    if best_labels_sa is not None and time.time() < deadline - 10.0:
        # We have a good topology and thicknesses from SA.
        # Run a local optimization to refine thicknesses precisely.
        # We'll use a simple gradient-free method or just a few more SA steps with lower T.
        # Let's do a final "Polish" phase with very low temperature and only thickness moves.
        T_polish = 0.01
        polish_labels = best_labels_sa[:]
        polish_thick = best_thick_sa.copy()
        polish_score = best_score_sa
        
        while time.time() < deadline - 2.0:
            i = rng.randint(0, N - 1)
            new_thick = polish_thick.copy()
            sigma = max_layer_nm * 0.01
            new_thick[i] += np_rng.normal(0, sigma)
            new_thick[i] = np.clip(new_thick[i], 0.0, max_layer_nm)
            if new_thick.sum() > max_total_nm:
                new_thick *= (max_total_nm / new_thick.sum())
            
            new_score = _mean_R(polish_labels, new_thick, n_amb, n_sub, n_L, n_H, band_nm, n_wl=401)
            
            if new_score < polish_score:
                polish_thick = new_thick
                polish_score = new_score
                if polish_score < best_score:
                    upd(polish_labels, polish_thick, polish_score)
            else:
                delta_E = new_score - polish_score
                accept_prob = np.exp(-delta_E / max(T_polish, 1e-9))
                if rng.random() < accept_prob:
                    polish_thick = new_thick
                    polish_score = new_score
            
            T_polish *= 0.999

    # Fallback
    if best_labels is None:
        best_labels = ["L"]
        best_thick = [centre / (4.0 * n_L)]
        best_score = _mean_R(best_labels, best_thick, n_amb, n_sub, n_L, n_H, band_nm)
        
    # Constraint Enforcement
    best_thick = [max(0.0, min(t, max_layer_nm)) for t in best_thick]
    tot = sum(best_thick)
    if tot > max_total_nm:
        sc = max_total_nm / tot
        best_thick = [t * sc for t in best_thick]
        
    return best_labels, best_thick

# EVOLVE-BLOCK-END
