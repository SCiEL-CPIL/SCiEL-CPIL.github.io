"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac))


def _autoconv_smooth_max(h, tau=0.5):
    """Smooth surrogate for max|autoconv| using log-sum-exp."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    abs_ac = np.abs(ac)
    # Log-sum-exp for smooth max
    m = np.max(abs_ac)
    if m < 1e-10:
        return m
    shifted = abs_ac / m
    log_sum = np.log(np.sum(np.exp(shifted - 1.0))) + 1.0
    return m * np.exp(log_sum) / tau * (1 - np.exp(-tau * m / m))  # This is getting complex
    # Simpler: just use a softened version
    # Actually, let's use a simpler smooth max: sqrt(sum(ac^2 * exp(-|ac|/tau)))
    # Or even simpler: use the L2 norm as a proxy when max is hard
    return np.sqrt(np.sum(ac * ac))


def _legendre_sequence(n):
    """Generate a Legendre symbol sequence of length n.
    
    Uses the Legendre symbol (x/p) for a prime p close to n.
    For x = 0, 1, ..., n-1, the sequence is (x/p) where p is the smallest prime >= n.
    This gives a sequence with good autocorrelation properties (Burgess bound).
    """
    # Find smallest prime >= n
    def is_prime(m):
        if m < 2:
            return False
        if m == 2:
            return True
        if m % 2 == 0:
            return False
        for i in range(3, int(np.sqrt(m)) + 1, 2):
            if m % i == 0:
                return False
        return True
    
    p = n
    while not is_prime(p):
        p += 1
    
    # Compute Legendre symbols for x = 0, 1, ..., n-1 mod p
    # (x/p) = x^((p-1)/2) mod p, which is 0 if x=0, 1 if quadratic residue, -1 otherwise
    half = (p - 1) // 2
    h = np.zeros(n)
    for x in range(n):
        if x == 0:
            h[x] = 1.0  # Convention: use +1 for x=0
        else:
            # Compute x^((p-1)/2) mod p
            val = pow(x, half, p)
            if val == 1:
                h[x] = 1.0
            elif val == p - 1:  # -1 mod p
                h[x] = -1.0
            else:
                h[x] = 1.0  # Should not happen for prime p and 0 < x < p
    
    return h


def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)


def _score_smooth(h, tau=0.5):
    """Smooth objective using L2 norm of autoconv as surrogate for max."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    # Use a blend of max and L2 to smooth the objective
    max_val = np.max(np.abs(ac))
    l2_val = np.sqrt(np.sum(ac * ac))
    # Smooth max approximation: for large arrays, max ~ l2 / sqrt(peak_concentration)
    # Use a parameterized blend
    return (max_val + 0.1 * l2_val) / (s * s)


def _local_sign_flip(h, max_passes=5, deadline=None):
    """Local optimization: try flipping each element's sign, keep if improves."""
    h = h.copy()
    best_val = _score(h)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            h[i] = -h[i]
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = -h[i]  # revert
        if not improved:
            break
    return h, best_val


def _pair_flip(h, max_passes=3, deadline=None):
    """Try flipping pairs of elements to escape local minima of single-flip."""
    h = h.copy()
    best_val = _score(h)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N - 1):
            if deadline is not None and time.time() > deadline:
                break
            # Try flipping h[i] and h[i+1]
            h[i] = -h[i]
            h[i + 1] = -h[i + 1]
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = -h[i]
                h[i + 1] = -h[i + 1]
            # Try flipping h[i] and h[(i+10) % N]
            j = (i + 10) % N
            h[i] = -h[i]
            h[j] = -h[j]
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = -h[i]
                h[j] = -h[j]
        if not improved:
            break
    return h, best_val


def _continuous_objective(x):
    """Objective for continuous optimization: max|autoconv| / (sum h)^2."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)


def _continuous_objective_smooth(x):
    """Smooth surrogate objective using L2 norm of autoconv."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    l2_val = np.sqrt(np.sum(ac * ac))
    max_val = np.max(np.abs(ac))
    # Blend to smooth the max
    return (0.8 * max_val + 0.2 * l2_val) / (s * s)


def _continuous_optimize(h_init, deadline=None):
    """
    Optimize in continuous space using scipy.optimize.minimize (L-BFGS-B),
    then quantize to ±1 and refine with local sign flips.
    This allows exploring the landscape more smoothly than pure sign flips.
    """
    h0 = h_init.astype(float).copy()
    
    # Ensure sum is not too small for stability
    s = np.sum(h0)
    if abs(s) < N * 0.1:
        if s < 0:
            h0 = -h0
        # Add a small bias to push away from zero sum, then normalize
        # We'll let the optimizer handle it, but avoid singularities
        if abs(np.sum(h0)) < 1e-6:
            h0[0] = 1.0

    best_h = h0.copy()
    best_val = _continuous_objective(h0)

    try:
        # Use L-BFGS-B which handles bounds well
        res = minimize(
            _continuous_objective,
            h0,
            method='L-BFGS-B',
            options={'maxiter': 50, 'ftol': 1e-6, 'maxfun': 200}
        )
        if res.x is not None and not np.any(np.isnan(res.x)):
            h_cont = res.x
            val_cont = _continuous_objective(h_cont)
            if val_cont < best_val:
                best_val = val_cont
                best_h = h_cont
    except Exception:
        pass

    # Also try smooth objective optimization
    if time.time() < deadline - 5:
        try:
            res2 = minimize(
                _continuous_objective_smooth,
                h0,
                method='L-BFGS-B',
                options={'maxiter': 40, 'ftol': 1e-6, 'maxfun': 150}
            )
            if res2.x is not None and not np.any(np.isnan(res2.x)):
                h_cont2 = res2.x
                val_cont2 = _continuous_objective(h_cont2)
                if val_cont2 < best_val:
                    best_val = val_cont2
                    best_h = h_cont2
        except Exception:
            pass

    # Quantize the best continuous solution to ±1
    # We can try different quantization thresholds or just sign
    h_quant = np.sign(best_h)
    # Handle zeros
    h_quant[h_quant == 0] = 1.0
    
    # Also try a biased quantization to maintain non-zero sum if needed
    s_q = np.sum(h_quant)
    if abs(s_q) < N * 0.1:
        # Flip the whole thing if sum is negative
        if s_q < 0:
            h_quant = -h_quant
        # If still small, force some elements to +1
        # Simple heuristic: set the elements with largest positive value to +1
        # But since it's ±1, we just pick a random subset to flip to +1
        # Or better, just use the continuous values to guide a soft sign
        # For now, just ensure it's not zero
        if abs(np.sum(h_quant)) < 1e-6:
            h_quant[0] = 1.0

    # Refine with local sign flips
    h_opt, val_opt = _local_sign_flip(h_quant, max_passes=3, deadline=deadline)
    
    return h_opt, val_opt


def _chirp_sequence(n, k=1):
    """Generate a chirp-like sequence with parameter k."""
    x = np.arange(n)
    h = np.sign(np.cos(2 * np.pi * k * x * (x + 1) / (2 * n)))
    h[h == 0] = 1.0
    return h


def _barker_like(n):
    """Generate a Barker-like sequence of length n using known Barker codes or random ±1."""
    barkers = {
        2: [1, 1],
        3: [1, 1, -1],
        4: [1, 1, -1, 1],
        5: [1, 1, -1, -1, 1],
        7: [1, -1, -1, 1, -1, 1, 1],
        11: [1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1],
        13: [1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1],
    }
    if n in barkers:
        return np.array(barkers[n], dtype=float)
    return np.random.choice([-1.0, 1.0], size=n)


def _difference_set_sequence(n, m=None):
    """Generate a sequence from a difference set construction.
    
    A (v, k, lambda) difference set in a group of order v has the property
    that every non-identity element is represented exactly lambda times as a
    difference of two elements of the set. This gives balanced autoconvolution.
    
    We use the Singer difference set from finite geometry:
    For prime power q, there exists a (q^2+q+1, q+1, 1) difference set in Z_{q^2+q+1}.
    We map this to a ±1 sequence of length n.
    """
    if m is None:
        # Find largest q such that q^2+q+1 <= n
        m = int(np.sqrt(n))
        while (m + 1)**2 + (m + 1) + 1 <= n:
            m += 1
        while m**2 + m + 1 > n and m > 1:
            m -= 1
    
    v = m**2 + m + 1  # order of the group
    
    # Singer difference set: elements {x : x is a power of g mod p where p = m+1}
    # Actually, for the standard Singer construction in Z_v:
    # The difference set is {g^i : i = 0, ..., m} where g is a primitive element
    # of the multiplicative group of GF(q^3) mapped to Z_v
    
    # Simpler approach: use a quadratic residue difference set if available
    # For p = 4k+1 prime, the quadratic residues form a (p, (p-1)/2, (p-3)/4) difference set
    # But we need v = q^2+q+1 which is not always prime
    
    # Fallback: use a structured sequence based on finite field multiplication
    # Create a ±1 sequence where +1 positions form a low-additive-energy set
    
    # Use a greedy construction: pick positions that minimize pairwise collisions
    h = np.zeros(n)
    positions = []
    
    # Target size: approximately sqrt(n) for difference-set-like density
    target_k = min(n, max(10, int(np.sqrt(n)) * 2))
    
    # Greedy: pick positions that maximize minimum distance from existing
    used = np.zeros(n, dtype=bool)
    for _ in range(target_k):
        if time.time() > 150:
            break
        # Score each candidate position by minimum distance to existing positions
        if len(positions) == 0:
            pos = np.random.randint(n)
        else:
            # Compute distances to all existing positions
            dists = np.zeros(n)
            for p in positions:
                # Circular distance
                d = np.abs(np.arange(n) - p)
                d = np.minimum(d, n - d)
                dists = np.minimum(dists, d)
            # Pick position with maximum minimum distance, with some randomness
            max_d = np.max(dists[~used])
            # Pick from top 10% of candidates for diversity
            threshold = max_d * 0.7
            candidates = np.where((dists >= threshold) & (~used))[0]
            if len(candidates) == 0:
                candidates = np.where(~used)[0]
            pos = candidates[np.random.randint(len(candidates))]
        
        positions.append(pos)
        used[pos] = True
    
    # Create ±1 sequence: +1 at selected positions, -1 elsewhere
    # But we want balanced: roughly equal +1 and -1
    n_plus = len(positions)
    n_minus = n - n_plus
    
    # If too unbalanced, adjust
    if n_plus < n // 3:
        # Add more +1 positions randomly
        remaining = np.where(~used)[0]
        extra = np.random.choice(remaining, size=min(n // 3 - n_plus, len(remaining)), replace=False)
        for e in extra:
            positions.append(e)
            used[e] = True
    
    h[used] = 1.0
    h[~used] = -1.0
    
    return h


def _sidon_like_sequence(n):
    """Generate a Sidon-like sequence: positions where all pairwise sums are distinct.
    
    A Sidon set (B_2 sequence) has the property that all pairwise sums a_i + a_j 
    (with i <= j) are distinct. This minimizes additive energy.
    We use a greedy construction.
    """
    h = np.zeros(n)
    positions = []
    used_sums = set()
    
    # Target size: approximately sqrt(n) for Sidon sets
    target_k = min(n, max(10, int(np.sqrt(n)) * 3))
    
    for _ in range(target_k):
        if time.time() > 150:
            break
        # Try to find a position that doesn't create duplicate sums
        found = False
        for pos in np.random.permutation(n):
            # Check if adding pos creates duplicate sums
            valid = True
            for p in positions:
                s = pos + p
                if s in used_sums:
                    valid = False
                    break
            if valid:
                # Add all new sums
                for p in positions:
                    used_sums.add(pos + p)
                used_sums.add(2 * pos)
                positions.append(pos)
                found = True
                break
        
        if not found:
            # Relax: allow some collisions
            pos = np.random.randint(n)
            for p in positions:
                used_sums.add(pos + p)
            used_sums.add(2 * pos)
            positions.append(pos)
    
    # Create ±1 sequence
    mask = np.zeros(n, dtype=bool)
    for p in positions:
        mask[p] = True
    
    h[mask] = 1.0
    h[~mask] = -1.0
    
    return h


def _balanced_low_energy_sequence(n, n_iter=50):
    """Iteratively construct a ±1 sequence with low additive energy.
    
    Additive energy E(A) = |{(a,b,c,d) in A^4 : a+b=c+d}|.
    We minimize this by greedily choosing signs that reduce energy.
    """
    h = np.random.choice([-1.0, 1.0], size=n)
    best_h = h.copy()
    best_val = _score(h)
    
    for it in range(n_iter):
        if time.time() > 150:
            break
        # Try flipping each element
        improved = False
        for i in range(n):
            h[i] = -h[i]
            val = _score(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()
                improved = True
            else:
                h[i] = -h[i]
        if not improved:
            break
    
    return best_h


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    best_h = np.ones(N)
    best_val = _score(best_h)

    np.random.seed(42)
    
    # Phase 1: Diverse initializations with continuous optimization
    # We use a smaller number of restarts for continuous opt as it's more expensive
    n_restarts_cont = 20
    
    # Start with Legendre sequence as it has provably good autocorrelation
    h_legendre = _legendre_sequence(N)
    h_opt, val_opt = _local_sign_flip(h_legendre, max_passes=10, deadline=deadline)
    if val_opt < best_val:
        best_val = val_opt
        best_h = h_opt
    
    # Try difference set construction
    if time.time() < deadline - 20:
        h_diff = _difference_set_sequence(N)
        h_opt, val_opt = _local_sign_flip(h_diff, max_passes=8, deadline=deadline)
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt
    
    # Try Sidon-like construction
    if time.time() < deadline - 20:
        h_sidon = _sidon_like_sequence(N)
        h_opt, val_opt = _local_sign_flip(h_sidon, max_passes=8, deadline=deadline)
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt
    
    for r in range(n_restarts_cont):
        if time.time() > deadline:
            break

        # Different initialization strategies
        if r % 7 == 0:
            # Random ±1
            h = np.random.choice([-1.0, 1.0], size=N)
        elif r % 7 == 1:
            # Low frequency cosine
            k = (r // 7) % 5 + 1
            x = np.arange(N)
            h = np.sign(np.cos(2 * np.pi * k * x / N))
            h[h == 0] = 1.0
        elif r % 7 == 2:
            # Chirp
            h = _chirp_sequence(N, k=1 + (r // 7) * 0.5)
        elif r % 7 == 3:
            # Random continuous values in [-1, 1]
            h = np.random.uniform(-1, 1, size=N)
        elif r % 7 == 4:
            # Block pattern
            h = np.ones(N)
            pos = 0
            while pos < N:
                block_len = np.random.randint(5, 20)
                sign = np.random.choice([-1.0, 1.0])
                h[pos:pos + block_len] = sign
                pos += block_len
        elif r % 7 == 5:
            # Difference set variant with different parameter
            h = _difference_set_sequence(N, m=max(2, int(np.sqrt(N)) - 1))
        else:
            # Balanced low-energy greedy
            h = _balanced_low_energy_sequence(N, n_iter=20)

        # Continuous optimization
        h_opt, val_opt = _continuous_optimize(h, deadline=deadline)

        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt

    # Phase 2: Refine the best solution found so far with more local sign flips
    # and try a few more discrete-only restarts if time permits
    if time.time() < deadline - 10:
        # Refine best_h further
        h_refined, val_refined = _local_sign_flip(best_h, max_passes=10, deadline=deadline)
        if val_refined < best_val:
            best_val = val_refined
            best_h = h_refined
        
        # Try pair flips to escape local minima
        h_pair, val_pair = _pair_flip(best_h, max_passes=3, deadline=deadline)
        if val_pair < best_val:
            best_val = val_pair
            best_h = h_pair
        h_refined2, val_refined2 = _local_sign_flip(best_h, max_passes=5, deadline=deadline)
        if val_refined2 < best_val:
            best_val = val_refined2
            best_h = h_refined2

        # Try a few more discrete restarts
        n_restarts_disc = 50
        for r in range(n_restarts_disc):
            if time.time() > deadline:
                break
            
            h = np.random.choice([-1.0, 1.0], size=N)
            # Small perturbation: flip a few random bits
            n_flips = np.random.randint(1, 10)
            flip_idx = np.random.choice(N, n_flips, replace=False)
            h[flip_idx] = -h[flip_idx]
            
            h_opt, val_opt = _local_sign_flip(h, max_passes=3, deadline=deadline)
            
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt

    # Phase 3: Final high-precision polish pass with more iterations
    if time.time() < deadline - 5:
        h_final, val_final = _local_sign_flip(best_h, max_passes=20, deadline=deadline)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final
        
        # Final pair flip polish
        h_pf, val_pf = _pair_flip(best_h, max_passes=2, deadline=deadline)
        if val_pf < best_val:
            best_val = val_pf
            best_h = h_pf
        h_ff, val_ff = _local_sign_flip(best_h, max_passes=10, deadline=deadline)
        if val_ff < best_val:
            best_val = val_ff
            best_h = h_ff

    return best_h

# EVOLVE-BLOCK-END
