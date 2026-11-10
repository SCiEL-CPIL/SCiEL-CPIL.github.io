import numpy as np
import time
from scipy.optimize import minimize_scalar

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    n = 24

    def grs_sequence(length=24):
        p, q = [1], [1]
        while len(p) < length:
            p, q = p + q, p + [-c for c in q]
        return p[:length]

    def eval_poly(coeffs, z):
        """Evaluate polynomial at complex z using Horner's method."""
        result = 0.0
        for c in reversed(coeffs):
            result = result * z + c
        return result

    def sup_norm(coeffs, num_samples=4096, refine=True):
        """Compute sup norm of P(z) over |z|=1 via dense sampling + golden-section refinement."""
        angles = 2 * np.pi * np.arange(num_samples) / num_samples
        zs = np.exp(1j * angles)

        # Vectorized Horner evaluation
        vals = np.zeros(num_samples, dtype=complex)
        for c in reversed(coeffs):
            vals = vals * zs + c

        mag = np.abs(vals)

        if not refine:
            return float(np.max(mag))

        # Find top peaks for refinement
        top_k = 8
        peak_indices = np.argsort(mag)[-top_k:][::-1]
        best_val = float(np.max(mag))
        window = 2 * np.pi / num_samples

        for idx in peak_indices:
            center = angles[idx]
            def obj(theta):
                z = np.exp(1j * theta)
                return -abs(eval_poly(coeffs, z))
            res = minimize_scalar(obj, bounds=(center - window, center + window),
                                  method='bounded', options={'xatol': 1e-11})
            if res.success:
                peak_mag = -res.fun
                if peak_mag > best_val:
                    best_val = peak_mag

        return best_val

    def score(coeffs, num_samples=4096):
        """Score: max |P(z)| / sqrt(n)"""
        return sup_norm(coeffs, num_samples) / np.sqrt(n)

    def score_fast(coeffs, num_samples=2048):
        """Faster score for screening: dense sampling without refinement."""
        angles = 2 * np.pi * np.arange(num_samples) / num_samples
        zs = np.exp(1j * angles)
        vals = np.zeros(num_samples, dtype=complex)
        for c in reversed(coeffs):
            vals = vals * zs + c
        return float(np.max(np.abs(vals))) / np.sqrt(n)

    # Generate candidate sequences
    candidates = set()
    base = grs_sequence(n)

    # Add base and its variants
    def add_variants(seq):
        s = list(seq)
        candidates.add(tuple(s))
        candidates.add(tuple(-c for c in s))
        candidates.add(tuple(reversed(s)))
        candidates.add(tuple(-c for c in reversed(s)))

    add_variants(base)

    # Add all rotations and their sign/reversal variants
    for shift in range(n):
        rotated = base[shift:] + base[:shift]
        add_variants(rotated)

    # Add near-Rudin-Shapiro sequences (Hamming distance <= 4 from base)
    from itertools import combinations
    for d in range(1, 5):
        for indices in combinations(range(n), d):
            seq = list(base)
            for i in indices:
                seq[i] = -seq[i]
            add_variants(seq)

    # Convert to list of lists
    candidate_list = [list(c) for c in candidates]

    if time.time() > deadline - 30:
        # Fallback: just return base
        return base

    # Phase 1: Fast screening
    scored = []
    for c in candidate_list:
        if time.time() > deadline - 25:
            break
        s = score_fast(c)
        scored.append((s, c))

    scored.sort(key=lambda x: x[0])

    # Phase 2: Accurate scoring of top candidates
    best_coeffs = scored[0][1] if scored else base
    best_score = score(best_coeffs) if scored else score(base)

    for s, c in scored[:30]:
        if time.time() > deadline - 10:
            break
        accurate_s = score(c)
        if accurate_s < best_score:
            best_score = accurate_s
            best_coeffs = c

    # Phase 3: Local refinement of best with single/pair flips
    def local_refine(coeffs, time_limit=5.0):
        current = list(coeffs)
        current_score = score_fast(current)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Single flips
            for i in range(n):
                if time.time() > time_limit:
                    break
                old = current[i]
                current[i] = -old
                new_score = score_fast(current)
                if new_score < current_score - 1e-12:
                    current_score = new_score
                    improved = True
                else:
                    current[i] = old
            # Pair flips
            for i in range(n):
                if time.time() > time_limit:
                    break
                for j in range(i + 1, n):
                    if time.time() > time_limit:
                        break
                    vi, vj = current[i], current[j]
                    current[i] = -vi
                    current[j] = -vj
                    new_score = score_fast(current)
                    if new_score < current_score - 1e-12:
                        current_score = new_score
                        improved = True
                    else:
                        current[i] = vi
                        current[j] = vj
        return current, current_score

    if time.time() < deadline - 15:
        refined, _ = local_refine(best_coeffs, time_limit=deadline - 15 - time.time())
        refined_score = score(refined)
        if refined_score < best_score:
            best_score = refined_score
            best_coeffs = refined

    return best_coeffs
