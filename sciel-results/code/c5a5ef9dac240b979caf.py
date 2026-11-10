"""Rewritten program: precomputed powers + delta-based local search for flat ±1 polynomials."""
import time
import numpy as np
from scipy.optimize import minimize_scalar

# EVOLVE-BLOCK-START

def _eval_poly(coeffs, z):
    """Evaluate polynomial P(z) = sum c_k z^k at complex z using Horner's method."""
    result = 0.0
    for c in reversed(coeffs):
        result = result * z + c
    return result

def _sup_norm_precise(coeffs, num_samples=8192, top_k=12):
    """Compute sup norm of P(z) over |z|=1 via dense sampling + golden-section refinement."""
    n = len(coeffs)
    angles = 2 * np.pi * np.arange(num_samples) / num_samples
    zs = np.exp(1j * angles)
    # Vectorized Horner
    vals = np.zeros(num_samples, dtype=complex)
    for c in reversed(coeffs):
        vals = vals * zs + c
    mag = np.abs(vals)
    
    peak_indices = np.argsort(mag)[-top_k:][::-1]
    
    best_val = 0.0
    for idx in peak_indices:
        center = angles[idx]
        window = 2 * np.pi / num_samples
        def obj(theta):
            z = np.exp(1j * theta)
            return -abs(_eval_poly(coeffs, z))
        res = minimize_scalar(obj, bounds=(center - window, center + window),
                              method='bounded', options={'xatol': 1e-11})
        if not res.success:
            continue
        peak_mag = -res.fun
        if peak_mag > best_val:
            best_val = peak_mag
    
    return best_val

def _score_precise(coeffs):
    """Score: max |P(z)| / sqrt(n) with high precision."""
    return _sup_norm_precise(coeffs) / np.sqrt(len(coeffs))

def _precompute_powers(num_samples=1024):
    """Precompute z^k for k=0..23 on the unit circle."""
    angles = 2 * np.pi * np.arange(num_samples) / num_samples
    z0 = np.exp(1j * angles)
    powers = np.zeros((24, num_samples), dtype=complex)
    powers[0] = 1.0
    for k in range(1, 24):
        powers[k] = powers[k-1] * z0
    return powers, angles, z0

def _score_fast(coeffs, powers, num_samples=1024, top_k=6):
    """Fast score using precomputed powers."""
    # P(z_j) = sum_k c_k * z_j^k
    # Using matrix: vals = powers.T @ coeffs (but coeffs is real)
    vals = np.zeros(num_samples, dtype=complex)
    for k in range(24):
        vals += coeffs[k] * powers[k]
    mag = np.abs(vals)
    
    peak_indices = np.argsort(mag)[-top_k:][::-1]
    best_val = 0.0
    for idx in peak_indices:
        center = mag[idx]
        # Quick golden section
        window = 2 * np.pi / num_samples
        def obj(theta):
            z = np.exp(1j * theta)
            s = 0.0
            for k in range(24):
                s += coeffs[k] * (z ** k)
            return -abs(s)
        res = minimize_scalar(obj, bounds=(2*np.pi*idx/num_samples - window,
                                           2*np.pi*idx/num_samples + window),
                              method='bounded', options={'xatol': 1e-10})
        if res.success:
            pm = -res.fun
            if pm > best_val:
                best_val = pm
    return best_val / np.sqrt(24)

def _score_ultra_fast(coeffs, powers, num_samples=512, top_k=4):
    """Very fast score for SA steps."""
    vals = np.zeros(num_samples, dtype=complex)
    for k in range(24):
        vals += coeffs[k] * powers[k]
    mag = np.abs(vals)
    
    peak_indices = np.argsort(mag)[-top_k:][::-1]
    best_val = 0.0
    for idx in peak_indices:
        center = mag[idx]
        window = 2 * np.pi / num_samples
        def obj(theta):
            z = np.exp(1j * theta)
            s = 0.0
            for k in range(24):
                s += coeffs[k] * (z ** k)
            return -abs(s)
        res = minimize_scalar(obj, bounds=(2*np.pi*idx/num_samples - window,
                                           2*np.pi*idx/num_samples + window),
                              method='bounded', options={'xatol': 1e-10})
        if res.success:
            pm = -res.fun
            if pm > best_val:
                best_val = pm
    return best_val / np.sqrt(24)

def _delta_single_flip(coeffs, i, powers, num_samples=512):
    """Compute change in max |P| when flipping coefficient i. Returns new max magnitude."""
    # P_new = P - 2*c_i * z^i
    c_i = coeffs[i]
    delta = -2 * c_i * powers[i]
    
    # Current P values
    vals = np.zeros(num_samples, dtype=complex)
    for k in range(24):
        vals += coeffs[k] * powers[k]
    
    new_vals = vals + delta
    new_mag = np.abs(new_vals)
    return np.max(new_mag)

def _delta_pair_flip(coeffs, i, j, powers, num_samples=512):
    """Compute change in max |P| when flipping coefficients i and j."""
    c_i, c_j = coeffs[i], coeffs[j]
    delta = -2 * c_i * powers[i] - 2 * c_j * powers[j]
    
    vals = np.zeros(num_samples, dtype=complex)
    for k in range(24):
        vals += coeffs[k] * powers[k]
    
    new_vals = vals + delta
    new_mag = np.abs(new_vals)
    return np.max(new_mag)

def _grs_sequence(length=24):
    """Generate Golay-Rudin-Shapiro sequence truncated to given length."""
    p, q = [1], [1]
    while len(p) < length:
        p, q = p + q, p + [-c for c in q]
    return p[:length]

def _known_good_sequences():
    """Return known good sequences."""
    seqs = []
    seqs.append(_grs_sequence(24))
    seqs.append([1]*24)
    seqs.append([1 if i%2==0 else -1 for i in range(24)])
    
    import random
    random.seed(42)
    for _ in range(30):
        seqs.append([random.choice([-1, 1]) for _ in range(24)])
    return seqs

def _local_search_delta(coeffs, powers, time_limit, num_samples=512, max_rounds=200):
    """Local search using delta computation for single and pair flips."""
    start_time = time.time()
    current = list(coeffs)
    n = len(current)
    
    # Compute current max magnitude
    vals = np.zeros(num_samples, dtype=complex)
    for k in range(24):
        vals += current[k] * powers[k]
    current_max = np.max(np.abs(vals))
    
    best = list(current)
    best_max = current_max
    
    improved = True
    rounds = 0
    while improved and rounds < max_rounds:
        if time.time() - start_time > time_limit:
            break
        improved = False
        rounds += 1
        
        # Single flips
        for i in range(n):
            if time.time() - start_time > time_limit:
                return best, best_max
            new_max = _delta_single_flip(current, i, powers, num_samples)
            if new_max < current_max - 1e-12:
                current[i] = -current[i]
                current_max = new_max
                improved = True
                if current_max < best_max - 1e-12:
                    best_max = current_max
                    best = list(current)
        
        # Pair flips
        if time.time() - start_time > time_limit:
            return best, best_max
        for i in range(n):
            if time.time() - start_time > time_limit:
                return best, best_max
            for j in range(i+1, n):
                if time.time() - start_time > time_limit:
                    return best, best_max
                new_max = _delta_pair_flip(current, i, j, powers, num_samples)
                if new_max < current_max - 1e-12:
                    current[i] = -current[i]
                    current[j] = -current[j]
                    current_max = new_max
                    improved = True
                    if current_max < best_max - 1e-12:
                        best_max = current_max
                        best = list(current)
    
    return best, best_max

def _simulated_annealing_delta(coeffs, powers, time_limit, num_samples=512):
    """Simulated annealing using delta-based evaluation."""
    start_time = time.time()
    current = list(coeffs)
    n = len(current)
    
    vals = np.zeros(num_samples, dtype=complex)
    for k in range(24):
        vals += current[k] * powers[k]
    current_max = np.max(np.abs(vals))
    
    best = list(current)
    best_max = current_max
    temp = 0.3
    
    while time.time() - start_time < time_limit:
        elapsed = time.time() - start_time
        progress = min(1.0, elapsed / time_limit)
        temp = 0.3 * (0.01 / 0.3) ** progress
        
        idx = np.random.randint(0, n)
        current[idx] = -current[idx]
        new_max = _delta_single_flip(current, idx, powers, num_samples)
        # Note: _delta_single_flip assumes the flip is NOT yet applied, so we need to be careful
        # Actually let me re-think: we already flipped current[idx], so the delta function
        # would compute the effect of flipping it again (undoing). Let me fix this.
        
        # Actually, let me recompute properly:
        # After flipping, the new polynomial is P_new = P_old - 2*c_i*z^i where c_i was the OLD value
        # But we already changed current[idx], so let's just recompute the max directly
        vals2 = np.zeros(num_samples, dtype=complex)
        for k in range(24):
            vals2 += current[k] * powers[k]
        new_max = np.max(np.abs(vals2))
        
        delta = new_max - current_max
        if delta < 0 or np.random.random() < np.exp(-delta / max(temp, 1e-10)):
            current_max = new_max
            if new_max < best_max - 1e-12:
                best_max = new_max
                best = list(current)
        else:
            current[idx] = -current[idx]
    
    return best, best_max

def _polish_precise(coeffs, time_limit=10.0):
    """High-precision polish with full evaluation."""
    start_time = time.time()
    current = list(coeffs)
    current_score = _score_precise(current)
    best = list(current)
    best_score = current_score
    
    while time.time() - start_time < time_limit:
        # Try single flips with precise scoring
        improved = False
        for i in range(24):
            if time.time() - start_time > time_limit:
                return best, best_score
            old_val = current[i]
            current[i] = -old_val
            new_score = _score_precise(current)
            if new_score < current_score - 1e-13:
                current_score = new_score
                improved = True
                if new_score < best_score - 1e-13:
                    best_score = new_score
                    best = list(current)
            else:
                current[i] = old_val
        
        if not improved:
            break
    
    return best, best_score

def construct():
    deadline = time.time() + 145  # Safety margin before 150s kill
    
    # Precompute powers
    powers, angles, z0 = _precompute_powers(512)
    
    best_coeffs = None
    best_score = float('inf')
    
    # Try known good sequences
    for seq in _known_good_sequences():
        if time.time() > deadline - 60:
            break
        score = _score_fast(seq, powers, num_samples=512, top_k=6)
        if score < best_score:
            best_score = score
            best_coeffs = list(seq)
    
    # Local search with delta
    if time.time() < deadline - 40:
        coeffs, max_mag = _local_search_delta(best_coeffs, powers, time_limit=15.0, num_samples=512)
        score = max_mag / np.sqrt(24)
        if score < best_score:
            best_score = score
            best_coeffs = coeffs
    
    # SA
    if time.time() < deadline - 30:
        sa_coeffs, sa_max = _simulated_annealing_delta(best_coeffs, powers, time_limit=15.0, num_samples=512)
        score = sa_max / np.sqrt(24)
        if score < best_score:
            best_score = score
            best_coeffs = sa_coeffs
    
    # More local search
    if time.time() < deadline - 20:
        coeffs, max_mag = _local_search_delta(best_coeffs, powers, time_limit=10.0, num_samples=512)
        score = max_mag / np.sqrt(24)
        if score < best_score:
            best_score = score
            best_coeffs = coeffs
    
    # Perturbed restarts
    if time.time() < deadline - 10:
        for _ in range(20):
            if time.time() >= deadline - 8:
                break
            perturbed = list(best_coeffs)
            for _ in range(np.random.randint(1, 6)):
                idx = np.random.randint(0, 24)
                perturbed[idx] = -perturbed[idx]
            coeffs, max_mag = _local_search_delta(perturbed, powers, time_limit=0.5, num_samples=512)
            score = max_mag / np.sqrt(24)
            if score < best_score:
                best_score = score
                best_coeffs = coeffs
    
    # High-precision polish
    if time.time() < deadline - 3:
        polished, polished_score = _polish_precise(best_coeffs, time_limit=min(5.0, deadline - time.time() - 2))
        if polished_score < best_score:
            best_score = polished_score
            best_coeffs = polished
    
    # Final precise score
    final_score = _score_precise(best_coeffs)
    
    return best_coeffs

# EVOLVE-BLOCK-END
