"""Broadband antireflection coating design via topology-driven global search."""
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
    beta = 2.0 * np.pi * n_mat * d  # beta_i = 2*pi*n_i*d_i ; delta_i = beta_i/lambda
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

def _nm_optimize(labels, x0, n_amb, n_sub, n_L, n_H, band_nm, max_layer_nm,
                 max_total_nm, deadline, iters=100, n_wl=401):
    """Constrained Nelder-Mead optimization."""
    n = len(labels)
    if n == 0:
        return None, float('inf')

    def proj(x):
        x = np.clip(x, 0.0, max_layer_nm)
        tot = x.sum()
        if tot > max_total_nm:
            x = x * (max_total_nm / tot)
        return x

    def obj(x):
        return _mean_R(labels, x, n_amb, n_sub, n_L, n_H, band_nm, n_wl)

    x0 = proj(x0)
    simplex = [x0.copy()]
    for i in range(n):
        p = x0.copy()
        p[i] = np.clip(p[i] + 0.1 * max_layer_nm, 0.0, max_layer_nm)
        simplex.append(proj(p))
    fvals = [obj(v) for v in simplex]

    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    for it in range(iters):
        if time.time() > deadline - 2.0:
            break
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        if abs(fvals[-1] - fvals[0]) < 1e-8:
            break
        centroid = np.mean(simplex[:-1], axis=0)
        xr = proj(centroid + alpha * (centroid - simplex[-1]))
        fr = obj(xr)
        if fr < fvals[0]:
            xe = proj(centroid + gamma * (xr - centroid))
            fe = obj(xe)
            simplex[-1], fvals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < fvals[-2]:
            simplex[-1], fvals[-1] = xr, fr
        else:
            xc = proj(centroid + rho * (xr - centroid)) if fr < fvals[-1] \
                else proj(centroid - rho * (simplex[-1] - centroid))
            fc = obj(xc)
            if fc < fvals[-1]:
                simplex[-1], fvals[-1] = xc, fc
            else:
                for i in range(1, len(simplex)):
                    simplex[i] = proj(simplex[0] + sigma * (simplex[i] - simplex[0]))
                    fvals[i] = obj(simplex[i])
    bi = int(np.argmin(fvals))
    return simplex[bi].tolist(), fvals[bi]

def design(problem):
    start = time.time()
    deadline = start + 135.0  # 15s safety margin

    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]

    centre = 0.5 * (band_nm[0] + band_nm[1])
    best_labels, best_thick, best_score = None, None, float('inf')

    def upd(labels, thick, score):
        nonlocal best_labels, best_thick, best_score
        if score < best_score:
            best_labels, best_thick, best_score = labels[:], thick[:], score

    # --- Phase 1: Rapid Topology Scan ---
    # We generate a diverse set of candidate sequences.
    # Instead of just alternating, we allow random sequences and specific patterns.
    candidates = []
    
    # 1. Alternating sequences (known good starting points)
    for n in range(1, min(max_layers + 1, 15)):
        for start_m in ('L', 'H'):
            seq = [start_m if i % 2 == 0 else ('H' if start_m == 'L' else 'L')
                   for i in range(n)]
            candidates.append(seq)
            
    # 2. Random sequences with varying lengths
    rng = random.Random(42)
    for _ in range(200):
        n = rng.randint(2, min(12, max_layers))
        seq = [rng.choice(['L', 'H']) for _ in range(n)]
        # Avoid immediate duplicates in the list for now
        if seq not in candidates:
            candidates.append(seq)
            
    # 3. Specific patterns like (HL)^n or (LH)^n are covered by alternating.
    # Add some "block" patterns like LLHHLLHH...
    for n in range(2, 8):
        block = ['L', 'L', 'H', 'H']
        seq = []
        for i in range(n):
            seq.extend(block)
        if len(seq) <= max_layers:
            if seq not in candidates:
                candidates.append(seq)
                # Reverse
                seq_rev = seq[::-1]
                if seq_rev not in candidates:
                    candidates.append(seq_rev)

    # Evaluate all candidates with a coarse grid (101 points) to find top performers
    coarse_scores = []
    for seq in candidates:
        # Initial thickness guess: quarter-wave at center wavelength
        x0 = np.array([centre / (4.0 * (n_L if l == 'L' else n_H)) for l in seq])
        if x0.sum() > max_total_nm:
            x0 = x0 * (max_total_nm / x0.sum())
        x0 = np.clip(x0, 0.0, max_layer_nm)
        
        s = _mean_R(seq, x0, n_amb, n_sub, n_L, n_H, band_nm, n_wl=101)
        coarse_scores.append((s, seq, x0))

    # Sort by coarse score
    coarse_scores.sort(key=lambda x: x[0])
    
    # Take the top 10 unique sequences for detailed optimization
    top_candidates = []
    seen = set()
    for score, seq, x0 in coarse_scores:
        if tuple(seq) not in seen:
            top_candidates.append((score, seq, x0))
            seen.add(tuple(seq))
        if len(top_candidates) >= 10:
            break

    # --- Phase 2: High-Precision Optimization of Top Candidates ---
    for score, seq, x0 in top_candidates:
        if time.time() > deadline - 10.0:
            break
        # Use the coarse best as starting point, but maybe perturb it slightly?
        # No, just use it. NM is good at local refinement.
        thick, score_fine = _nm_optimize(seq, x0, n_amb, n_sub, n_L, n_H, band_nm,
                                          max_layer_nm, max_total_nm, deadline, 
                                          iters=150, n_wl=401)
        if thick is not None:
            upd(seq, thick, score_fine)
            
    # --- Phase 3: Local Search / Neighbor Exploration around Best ---
    # If we have time, try swapping layers in the best found sequence.
    if best_labels is not None and time.time() < deadline - 15.0:
        n_layers = len(best_labels)
        # Try swapping any two different layers
        for i in range(n_layers):
            for j in range(i + 1, n_layers):
                if best_labels[i] == best_labels[j]:
                    continue
                cand_labels = best_labels[:]
                cand_labels[i], cand_labels[j] = best_labels[j], best_labels[i]
                # Use best thicknesses as initial guess (reordered)
                cand_thick = best_thick[:]
                cand_thick[i], cand_thick[j] = best_thick[j], best_thick[i]
                
                if time.time() > deadline - 5.0:
                    break
                
                thick, score = _nm_optimize(cand_labels, cand_thick, n_amb, n_sub, n_L, n_H, 
                                            band_nm, max_layer_nm, max_total_nm, deadline,
                                            iters=100, n_wl=401)
                if thick is not None:
                    upd(cand_labels, thick, score)
    
    # --- Phase 4: Final Polish ---
    if best_labels is not None and time.time() < deadline - 5.0:
        thick, score = _nm_optimize(best_labels, best_thick, n_amb, n_sub, n_L, n_H,
                                     band_nm, max_layer_nm, max_total_nm, deadline,
                                     iters=200, n_wl=401)
        if thick is not None:
            upd(best_labels, thick, score)

    # --- Final Constraint Enforcement ---
    if best_labels is None:
        # Fallback: single layer
        best_labels = ["L"]
        best_thick = [centre / (4.0 * n_L)]
        best_score = _mean_R(best_labels, best_thick, n_amb, n_sub, n_L, n_H, band_nm)
        
    best_thick = [max(0.0, min(t, max_layer_nm)) for t in best_thick]
    tot = sum(best_thick)
    if tot > max_total_nm:
        sc = max_total_nm / tot
        best_thick = [t * sc for t in best_thick]
        
    return best_labels, best_thick

# EVOLVE-BLOCK-END
