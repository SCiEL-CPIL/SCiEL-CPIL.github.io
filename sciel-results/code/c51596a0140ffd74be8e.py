import time
import random
import numpy as np

_START = time.time()
_DEADLINE = 55.0  # safety margin before 60s kill

def _sq_nonzero_count(coefs):
    """Count nonzero terms in the exact square using numpy convolution."""
    arr = np.array(coefs, dtype=np.int64)
    sq = np.convolve(arr, arr)
    return int(np.count_nonzero(sq))

def _valid(coefs):
    if len(coefs) > 41:
        return False
    nz = 0
    for c in coefs:
        if not isinstance(c, int):
            return False
        if c < -64 or c > 64:
            return False
        if c != 0:
            nz += 1
    if nz != 12:
        return False
    if coefs[-1] == 0:
        return False
    return True

def _make_palindromic_poly(rng, n_terms=12, max_deg=40):
    """Create a palindromic polynomial: P(x) = x^d * P(1/x).
    For even degree d, coefficients satisfy c[i] = c[d-i].
    This structure can force cancellations in P²."""
    # Choose an even degree for palindromic structure
    d = rng.choice([10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40])
    
    # For a palindromic poly of degree d with n nonzero terms:
    # Terms come in pairs (i, d-i) plus possibly the middle term (d/2)
    half = d // 2
    support = set()
    
    # Add symmetric pairs
    available_pairs = [(i, d - i) for i in range(0, half + 1) if i != d - i or d % 2 == 0]
    # Filter: only include i where i <= d-i to avoid duplicates
    available_pairs = [(i, d - i) for i in range(0, half + 1)]
    rng.shuffle(available_pairs)
    
    for i, j in available_pairs:
        if len(support) >= n_terms:
            break
        support.add(i)
        if i != j:
            support.add(j)
    
    # Ensure we have exactly n_terms
    while len(support) < n_terms:
        # Add more terms
        for i in range(d + 1):
            if i not in support and (d - i) not in support:
                support.add(i)
                if i != d - i:
                    support.add(d - i)
                if len(support) >= n_terms:
                    break
        if len(support) < n_terms:
            # Just add random terms
            for i in range(d + 1):
                if i not in support:
                    support.add(i)
                    if len(support) >= n_terms:
                        break
    
    support = sorted(support)[:n_terms]
    
    # Ensure highest degree is included
    if d not in support:
        support = sorted(support[:-1] + [d])[:n_terms]
    
    # Build coefficients with palindromic structure
    coefs = [0] * (d + 1)
    for pos in support:
        if pos <= half:
            val = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4])
            coefs[pos] = val
            if pos != d - pos:
                coefs[d - pos] = val  # palindromic
        else:
            # Already set by symmetric pair
            pass
    
    # Ensure nonzero final coefficient
    if coefs[d] == 0:
        coefs[d] = 1
        if d > 0 and d - d == 0:
            coefs[0] = 1
    
    # Trim trailing zeros but keep length <= 41
    while len(coefs) > 1 and coefs[-1] == 0:
        coefs.pop()
    
    if len(coefs) > 41:
        coefs = coefs[:41]
    
    return coefs

def _try_factorization(best, rng):
    """Try to construct P = A * B where A, B have few terms, 
    then check if P has exactly 12 nonzero terms and minimize P² nonzero count."""
    val_pool = [-4, -3, -2, -1, 1, 2, 3, 4]
    
    for _ in range(500):
        if time.time() > _DEADLINE:
            break
        
        # Generate sparse A and B
        deg_a = rng.randint(3, 15)
        deg_b = rng.randint(3, 15)
        if deg_a + deg_b > 40:
            continue
        
        n_a = rng.randint(3, 6)
        n_b = rng.randint(3, 6)
        
        # Random supports for A and B
        supp_a = sorted(rng.sample(range(deg_a + 1), min(n_a, deg_a + 1)))
        supp_b = sorted(rng.sample(range(deg_b + 1), min(n_b, deg_b + 1)))
        
        if deg_a not in supp_a:
            supp_a = sorted(supp_a[:-1] + [deg_a])
        if deg_b not in supp_b:
            supp_b = sorted(supp_b[:-1] + [deg_b])
        
        # Random values
        coefs_a = [0] * (deg_a + 1)
        for d in supp_a:
            coefs_a[d] = rng.choice(val_pool)
        
        coefs_b = [0] * (deg_b + 1)
        for d in supp_b:
            coefs_b[d] = rng.choice(val_pool)
        
        # Multiply A * B
        coefs_p = np.convolve(np.array(coefs_a, dtype=np.int64), 
                               np.array(coefs_b, dtype=np.int64))
        coefs_p = coefs_p.tolist()
        
        # Check constraints
        if len(coefs_p) > 41:
            continue
        if coefs_p[-1] == 0:
            continue
        
        nz_p = sum(1 for c in coefs_p if c != 0)
        if nz_p != 12:
            continue
        
        # Check coefficient bounds
        if any(c < -64 or c > 64 for c in coefs_p):
            continue
        
        # Count nonzero in P²
        sc = _sq_nonzero_count(coefs_p)
        
        if sc < best[0]:
            best[0] = sc
            best[1] = coefs_p[:]
    
    return best

def _palindromic_search(best, rng):
    """Search for palindromic polynomials with good square properties."""
    val_pool = [-4, -3, -2, -1, 1, 2, 3, 4]
    
    for _ in range(1000):
        if time.time() > _DEADLINE:
            break
        
        # Choose degree (even for clean palindromic structure)
        d = rng.choice([8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40])
        
        # For palindromic: c[i] = c[d-i]
        # We need exactly 12 nonzero terms
        # Terms come in pairs (i, d-i) or singleton if i = d-i
        
        half = d // 2
        is_even = (d % 2 == 0)
        
        # Number of independent positions: 0 to half (if even) or 0 to half-1 (if odd)
        # Each position i < d/2 gives 2 terms, position d/2 (if exists) gives 1 term
        
        # We need 12 terms total
        # Let k = number of pair positions, m = 1 if middle term used else 0
        # 2k + m = 12
        # If m=0: k=6, need 6 positions from [0, half-1] (if even) or [0, half-1] (if odd)
        # If m=1: k=5.5, not integer, so m must be 0 for even d
        # For odd d: no middle term, so 2k=12, k=6
        
        if is_even:
            # Can use middle term or not
            if rng.random() < 0.5:
                # No middle term: 6 pairs from [0, half-1]
                if half - 1 >= 6:
                    pair_positions = sorted(rng.sample(range(0, half), 6))
                    support = set()
                    for i in pair_positions:
                        support.add(i)
                        support.add(d - i)
                else:
                    continue
            else:
                # With middle term: 5 pairs + 1 middle
                if half - 1 >= 5:
                    pair_positions = sorted(rng.sample(range(0, half), 5))
                    support = set()
                    for i in pair_positions:
                        support.add(i)
                        support.add(d - i)
                    support.add(half)
                else:
                    continue
        else:
            # Odd degree: no middle term, 6 pairs from [0, half]
            if half >= 6:
                pair_positions = sorted(rng.sample(range(0, half + 1), 6))
                support = set()
                for i in pair_positions:
                    support.add(i)
                    support.add(d - i)
            else:
                continue
        
        if len(support) != 12:
            continue
        
        # Ensure degree d is in support
        if d not in support:
            # Replace a random position with d
            positions = list(support)
            idx = rng.randrange(len(positions))
            old_pos = positions[idx]
            # If we remove old_pos, we need to also remove d-old_pos if it exists
            support.remove(old_pos)
            if d - old_pos in support:
                support.remove(d - old_pos)
            support.add(d)
            support.add(0)  # palindromic partner of d
            if len(support) != 12:
                continue
        
        # Assign palindromic coefficients
        coefs = [0] * (d + 1)
        for pos in support:
            if pos <= half:
                if coefs[pos] == 0:
                    val = rng.choice(val_pool)
                    coefs[pos] = val
                    if pos != d - pos:
                        coefs[d - pos] = val
        
        # Ensure nonzero final coefficient
        if coefs[d] == 0:
            coefs[d] = 1
            if d > 0:
                coefs[0] = 1
        
        # Check constraints
        nz = sum(1 for c in coefs if c != 0)
        if nz != 12:
            continue
        if any(c < -64 or c > 64 for c in coefs):
            continue
        
        sc = _sq_nonzero_count(coefs)
        if sc < best[0]:
            best[0] = sc
            best[1] = coefs[:]
    
    return best

def _polish(best, support, rng):
    """Local hill-climbing polish using numpy-accelerated scoring."""
    hi = max(support)
    best_local = best[1][:] if best[1] else None
    if best_local is None:
        return best
    best_score = best[0]
    
    # Value candidates to try
    val_candidates = [-64, -32, -16, -8, -4, -3, -2, -1, 1, 2, 3, 4, 8, 16, 32, 64]
    
    for _pass in range(5):
        if time.time() > _DEADLINE:
            break
        improved = True
        while improved and time.time() < _DEADLINE:
            improved = False
            for d in support:
                if time.time() > _DEADLINE:
                    break
                old_val = best_local[d]
                for new_val in val_candidates:
                    if new_val == old_val:
                        continue
                    best_local[d] = new_val
                    if best_local[hi] == 0:
                        best_local[d] = old_val
                        continue
                    sc = _sq_nonzero_count(best_local)
                    if sc < best_score:
                        best_score = sc
                        best[0] = sc
                        best[1] = best_local[:]
                        improved = True
                    else:
                        best_local[d] = old_val
    return best

def _try_place(best, support, rng):
    """Assign coefficients to support and optimize."""
    n = len(support)
    hi = max(support)
    best_local = None
    best_score = 10**9
    val_pool = [-4, -3, -2, -1, 1, 2, 3, 4]
    
    for _ in range(200):
        if time.time() > _DEADLINE:
            break
        coefs = [0] * 41
        for d in support:
            c = rng.choice(val_pool)
            coefs[d] = c
        if coefs[hi] == 0:
            coefs[hi] = 1
        sc = _sq_nonzero_count(coefs)
        if sc < best_score:
            best_score = sc
            best_local = coefs[:]
        
        # Local perturbation
        if best_local is not None:
            cand = best_local[:]
            d = support[rng.randrange(n)]
            old = cand[d]
            new = old + rng.choice([-4, -3, -2, -1, 1, 2, 3, 4])
            if -64 <= new <= 64 and new != 0:
                cand[d] = new
            if cand[hi] == 0:
                cand[hi] = 1
            sc2 = _sq_nonzero_count(cand)
            if sc2 < best_score:
                best_score = sc2
                best_local = cand[:]
    
    if best_local is not None and _valid(best_local):
        sc = _sq_nonzero_count(best_local)
        if sc < best[0]:
            best[0] = sc
            best[1] = best_local[:]
            _polish(best, support, rng)
    return best

def construct():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 55.0
    rng = random.Random(12345)
    best = [10**9, None]
    
    # 1) Factorization-based construction: P = A * B
    best = _try_factorization(best, rng)
    
    # 2) Palindromic polynomial search
    best = _palindromic_search(best, rng)
    
    # 3) If we found something, polish it
    if best[1] is not None:
        support = [i for i, c in enumerate(best[1]) if c != 0]
        best = _polish(best, support, rng)
    
    # 4) Fallback: random supports if needed
    if best[1] is None:
        for t in range(50):
            if time.time() > _DEADLINE:
                break
            pool = list(range(0, 40))
            rng.shuffle(pool)
            chosen = sorted(pool[:11]) + [40]
            _try_place(best, chosen, rng)
    
    if best[1] is None:
        coefs = [0] * 41
        for i in range(12):
            coefs[i] = 1
        return coefs
    
    return best[1]
