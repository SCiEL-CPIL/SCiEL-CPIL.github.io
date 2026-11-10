"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize_scalar

# EVOLVE-BLOCK-START
import random

def _eval_poly(coeffs, z):
    """Evaluate polynomial P(z) = sum c_k z^k at complex z using Horner's method."""
    result = 0.0
    for c in reversed(coeffs):
        result = result * z + c
    return result

def _sup_norm(coeffs, num_samples=2048):
    """Compute sup norm of P(z) over |z|=1 via dense sampling + golden-section refinement."""
    n = len(coeffs)
    angles = 2 * np.pi * np.arange(num_samples) / num_samples
    zs = np.exp(1j * angles)
    powers = np.ones((num_samples, n), dtype=complex)
    for k in range(1, n):
        powers[:, k] = powers[:, k-1] * zs
    vals = powers @ coeffs
    mag = np.abs(vals)
    
    top_k = 8
    peak_indices = np.argsort(mag)[-top_k:][::-1]
    
    best_val = 0.0
    for idx in peak_indices:
        center = angles[idx]
        window = 2 * np.pi / num_samples
        def obj(theta):
            z = np.exp(1j * theta)
            return -abs(_eval_poly(coeffs, z))
        res = minimize_scalar(obj, bounds=(center - window, center + window), 
                              method='bounded', options={'xatol': 1e-10})
        if not res.success:
            continue
        peak_mag = -res.fun
        if peak_mag > best_val:
            best_val = peak_mag
    
    return best_val

def _score(coeffs, num_samples=2048):
    """Score: max |P(z)| / sqrt(n)"""
    return _sup_norm(coeffs, num_samples) / np.sqrt(len(coeffs))

def _score_fast(coeffs, num_samples=512):
    """Faster score using fewer samples for local search steps."""
    return _sup_norm(coeffs, num_samples) / np.sqrt(len(coeffs))

def _grs_sequence(length=24):
    """Generate Golay-Rudin-Shapiro sequence truncated to given length."""
    p, q = [1], [1]
    while len(p) < length:
        p, q = p + q, p + [-c for c in q]
    return p[:length]

def _perturb(coeffs, num_flips):
    """Create a perturbed version of coeffs by flipping random coefficients."""
    result = list(coeffs)
    for _ in range(num_flips):
        idx = random.randint(0, len(result) - 1)
        result[idx] = -result[idx]
    return result

def _vectorized_single_flip_scores(coeffs, num_samples=512):
    """
    For each coefficient position i, compute the sup-norm score if we flip c_i.
    Uses the fact that flipping c_i changes P(z) by -2*c_i*z^i.
    We precompute z^k for all k and use vectorized operations.
    Returns array of scores for flipping each position.
    """
    n = len(coeffs)
    angles = 2 * np.pi * np.arange(num_samples) / num_samples
    zs = np.exp(1j * angles)
    
    # Compute all powers z^k
    powers = np.ones((num_samples, n), dtype=complex)
    for k in range(1, n):
        powers[:, k] = powers[:, k-1] * zs
    
    # Base polynomial values
    base_vals = powers @ np.array(coeffs, dtype=complex)
    base_mag = np.abs(base_vals)
    
    # For each flip position i, new_vals = base_vals - 2*c_i*z^i
    # We want max over z of |base_vals - 2*c_i*z^i| for each i
    # This is a (num_samples x n) operation
    # delta[i, j] = -2 * c_i * z_j^i
    # We can compute: for each i, mag_new = |base_vals - 2*c_i*powers[:,i]|
    
    # Vectorized: create a (n, num_samples) array of magnitudes after flip
    # flip_mag[i, j] = |base_vals[j] - 2*coeffs[i]*powers[j, i]|
    # = |base_vals[j] - 2*coeffs[i]*zs[j]^i|
    
    # Use broadcasting: coeffs[:, None] * powers.T -> (n, num_samples)
    delta = 2.0 * np.array(coeffs, dtype=complex)[:, None] * powers.T  # (n, num_samples)
    new_vals = base_vals[None, :] - delta  # (n, num_samples)
    new_mag = np.abs(new_vals)  # (n, num_samples)
    
    # Sup norm for each flip
    flip_scores = np.max(new_mag, axis=1) / np.sqrt(n)
    
    return flip_scores

def _vectorized_pair_flip_scores(coeffs, num_samples=256):
    """
    For each pair (i,j), compute the sup-norm score if we flip both c_i and c_j.
    Returns a 2D array of scores.
    """
    n = len(coeffs)
    angles = 2 * np.pi * np.arange(num_samples) / num_samples
    zs = np.exp(1j * angles)
    
    powers = np.ones((num_samples, n), dtype=complex)
    for k in range(1, n):
        powers[:, k] = powers[:, k-1] * zs
    
    base_vals = powers @ np.array(coeffs, dtype=complex)
    
    # delta for flipping i and j: -2*c_i*z^i - 2*c_j*z^j
    # For all pairs, we need: new_vals[j] = base_vals[j] - 2*c_i*powers[j,i] - 2*c_j*powers[j,j]
    
    # This is expensive: n*(n-1)/2 pairs, each requiring num_samples evaluations
    # Let's be smarter: precompute the effect of each single flip
    # single_delta[i, j] = 2*coeffs[i]*powers[j, i]  (n, num_samples)
    single_delta = 2.0 * np.array(coeffs, dtype=complex)[:, None] * powers.T  # (n, num_samples)
    
    # For pair (i,j): new_mag = |base_vals - single_delta[i] - single_delta[j]|
    # We can iterate over pairs but use vectorized inner computation
    
    pair_scores = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i+1, n):
            new_vals = base_vals - single_delta[i] - single_delta[j]
            pair_scores[i, j] = np.max(np.abs(new_vals)) / np.sqrt(n)
            pair_scores[j, i] = pair_scores[i, j]
    
    return pair_scores

def _greedy_local_search(coeffs, time_limit=30.0, num_samples=512):
    """Local search: try flipping single coefficients and pairs using vectorized scoring."""
    start_time = time.time()
    current = list(coeffs)
    current_score = _score_fast(current, num_samples)
    n = len(current)
    
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > time_limit:
            break
        
        # Single flips - vectorized
        flip_scores = _vectorized_single_flip_scores(current, num_samples)
        best_flip_idx = np.argmin(flip_scores)
        if flip_scores[best_flip_idx] < current_score - 1e-10:
            current[best_flip_idx] = -current[best_flip_idx]
            current_score = flip_scores[best_flip_idx]
            improved = True
        
        # Pair flips - vectorized
        if time.time() - start_time > time_limit:
            break
        pair_scores = _vectorized_pair_flip_scores(current, num_samples)
        # Find best pair
        np.fill_diagonal(pair_scores, np.inf)
        best_pair_idx = np.unravel_index(np.argmin(pair_scores), pair_scores.shape)
        best_pair_score = pair_scores[best_pair_idx]
        if best_pair_score < current_score - 1e-10:
            i, j = best_pair_idx
            current[i] = -current[i]
            current[j] = -current[j]
            current_score = best_pair_score
            improved = True
    
    return current, current_score

def _random_restarts(num_restarts, time_per_restart, num_samples=256):
    """Try multiple random initializations with local search."""
    best_coeffs = None
    best_score = float('inf')
    start_time = time.time()
    
    for r in range(num_restarts):
        if time.time() - start_time > time_per_restart:
            break
        # Random initialization
        coeffs = [random.choice([-1, 1]) for _ in range(24)]
        coeffs, score = _greedy_local_search(coeffs, time_limit=0.5, num_samples=num_samples)
        if score < best_score:
            best_score = score
            best_coeffs = coeffs
    
    return best_coeffs, best_score

def _known_good_sequences():
    """Return a list of known good starting sequences."""
    seqs = []
    # Golay-Rudin-Shapiro
    seqs.append(_grs_sequence(24))
    # Some manually tuned sequences
    seqs.append([1,1,1,-1,-1,-1,1,-1,1,1,-1,-1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,1])
    seqs.append([1,-1,1,1,-1,1,1,-1,-1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1])
    return seqs

def construct():
    deadline = time.time() + 118  # Safety margin before 150s kill
    
    # Start with known good sequences
    best_coeffs = None
    best_score = float('inf')
    
    for seq in _known_good_sequences():
        if time.time() > deadline - 30:
            break
        coeffs, score = _greedy_local_search(seq, time_limit=5.0, num_samples=512)
        if score < best_score:
            best_score = score
            best_coeffs = coeffs
    
    print(f"After initial LS: {best_score:.9f}")
    
    # Random restarts with local search
    if time.time() < deadline - 30:
        num_restarts = 200
        time_per_restart = min(60.0, deadline - time.time() - 20)
        coeffs, score = _random_restarts(num_restarts, time_per_restart, num_samples=256)
        if coeffs is not None and score < best_score:
            best_score = score
            best_coeffs = coeffs
            print(f"After random restarts: {best_score:.9f}")
    
    # Perturbed local search from best
    if time.time() < deadline - 15:
        for _ in range(20):
            if time.time() >= deadline - 10:
                break
            perturbed = _perturb(best_coeffs, random.randint(1, 4))
            coeffs, score = _greedy_local_search(perturbed, time_limit=1.0, num_samples=256)
            if score < best_score:
                best_score = score
                best_coeffs = coeffs
    
    # Final local search from best with higher precision
    if time.time() < deadline - 5:
        coeffs, score = _greedy_local_search(best_coeffs, time_limit=min(8.0, deadline - time.time() - 3), num_samples=512)
        if score < best_score:
            best_score = score
            best_coeffs = coeffs
    
    # Final score with high precision
    final_score = _score(best_coeffs)
    print(f"Final score: {final_score:.9f}")
    
    return best_coeffs
# EVOLVE-BLOCK-END
