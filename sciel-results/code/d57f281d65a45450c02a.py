import time
import random
import numpy as np

def _max_abs_ap_sum(signs, threshold=None):
    """
    Compute the maximum absolute sum over any arithmetic progression
    contained in [1, n].
    signs: list or numpy array of length n, values -1 or 1.
    threshold: if provided, early-exit and return threshold+1 if exceeded.
    Returns the max absolute sum.
    """
    n = len(signs)
    signs_arr = np.asarray(signs, dtype=np.int8)
    max_sum = 0
    
    for d in range(1, n + 1):
        for s in range(d):
            if s >= n:
                continue
            # Vectorized: get all indices in this AP
            idx = np.arange(s, n, d)
            # Cumulative sum along the AP
            cumsum = np.cumsum(signs_arr[idx])
            current_max = int(np.max(cumsum) - np.min(cumsum))
            if current_max > max_sum:
                max_sum = current_max
                if threshold is not None and max_sum > threshold:
                    return max_sum
    
    return max_sum

def _get_tight_ap_info(signs, top_k=40):
    """
    Find indices that appear in APs with sums close to the maximum.
    Returns a list of candidate indices to flip, weighted by tightness.
    """
    n = len(signs)
    signs_arr = np.asarray(signs, dtype=np.int8)
    
    ap_sums = []
    max_sum = 0
    
    for d in range(1, n + 1):
        for s in range(d):
            if s >= n:
                continue
            idx = np.arange(s, n, d)
            cumsum = np.cumsum(signs_arr[idx])
            current_max = int(np.max(cumsum) - np.min(cumsum))
            if current_max > 0:
                ap_sums.append((current_max, s, d))
            if current_max > max_sum:
                max_sum = current_max
    
    if max_sum == 0:
        return list(range(n))
    
    tight_threshold = max_sum - 1
    tight_aps = [(s, d) for (val, s, d) in ap_sums if val >= tight_threshold]
    
    idx_count = np.zeros(n, dtype=np.int32)
    for s, d in tight_aps:
        idx = s
        while idx < n:
            idx_count[idx] += 1
            idx += d
    
    candidates = np.argsort(-idx_count)[:top_k]
    return candidates.tolist()

def _legendre_init(n):
    """
    Generate Legendre symbol sequence modulo a prime near n.
    For prime p, Legendre symbol (a/p) = 1 if a is QR mod p, -1 if QNR, 0 if a≡0.
    We use the first n values of the character modulo p (skipping 0).
    """
    # Find a prime near or slightly above n
    def is_prime(x):
        if x < 2:
            return False
        if x == 2:
            return True
        if x % 2 == 0:
            return False
        for i in range(3, int(x**0.5) + 1, 2):
            if x % i == 0:
                return False
        return True
    
    # Try primes near n
    p = n
    while not is_prime(p):
        p += 1
    while not is_prime(p):
        p -= 1
    
    # Use the larger prime
    if p < n:
        p = n
        while not is_prime(p):
            p += 1
    
    # Compute Legendre symbols for 1..n using quadratic reciprocity or simple check
    signs = []
    for a in range(1, n + 1):
        # Check if a is a quadratic residue mod p
        # a^((p-1)/2) mod p = 1 if QR, -1 if QNR
        legendre = pow(a % p, (p - 1) // 2, p)
        if legendre == 1:
            signs.append(1)
        elif legendre == p - 1:  # -1 mod p
            signs.append(-1)
        else:
            # Should not happen for 1<=a<p
            signs.append(1)
    
    return signs

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 256
    rng = random.Random(42)
    
    # Multiple initializations
    initializations = []
    
    # 1. Alternating
    initializations.append([1 if i % 2 == 0 else -1 for i in range(n)])
    
    # 2. Thue-Morse
    tm = []
    for i in range(n):
        tm.append(1 if bin(i).count('1') % 2 == 0 else -1)
    initializations.append(tm)
    
    # 3-6. Random starts
    for _ in range(4):
        initializations.append([rng.choice((-1, 1)) for _ in range(n)])
    
    # 7. Block pattern +1+1-1-1
    initializations.append([1 if (i // 2) % 2 == 0 else -1 for i in range(n)])
    
    # 8. Block pattern +1+1+1-1-1-1
    initializations.append([1 if (i // 3) % 2 == 0 else -1 for i in range(n)])
    
    # 9. Greedy construction: at each position, choose sign that minimizes
    #    the max absolute AP sum for APs ending at that position
    greedy_signs = [0] * n
    for i in range(n):
        # Try +1
        greedy_signs[i] = 1
        # Compute max abs sum for APs ending at position i
        max_plus = 0
        for d in range(1, i + 1):
            # AP: i-d, i-2d, ..., i
            s = i
            total = 0
            while s >= 0:
                total += greedy_signs[s]
                max_plus = max(max_plus, abs(total))
                s -= d
        # Try -1
        greedy_signs[i] = -1
        max_minus = 0
        for d in range(1, i + 1):
            s = i
            total = 0
            while s >= 0:
                total += greedy_signs[s]
                max_minus = max(max_minus, abs(total))
                s -= d
        # Choose the better one
        if max_plus <= max_minus:
            greedy_signs[i] = 1
        else:
            greedy_signs[i] = -1
    initializations.append(greedy_signs)
    
    # 10. Another greedy variant with different tie-breaking
    greedy2_signs = [0] * n
    for i in range(n):
        greedy2_signs[i] = 1
        max_plus = 0
        for d in range(1, i + 1):
            s = i
            total = 0
            while s >= 0:
                total += greedy2_signs[s]
                max_plus = max(max_plus, abs(total))
                s -= d
        greedy2_signs[i] = -1
        max_minus = 0
        for d in range(1, i + 1):
            s = i
            total = 0
            while s >= 0:
                total += greedy2_signs[s]
                max_minus = max(max_minus, abs(total))
                s -= d
        if max_plus < max_minus:
            greedy2_signs[i] = 1
        else:
            greedy2_signs[i] = -1
    initializations.append(greedy2_signs)
    
    # 11. Legendre symbol initialization
    legendre = _legendre_init(n)
    initializations.append(legendre)
    
    # 12. Negated Legendre
    initializations.append([-x for x in legendre])
    
    best_signs = None
    best_score = float('inf')
    
    for init in initializations:
        score = _max_abs_ap_sum(init)
        if score < best_score:
            best_score = score
            best_signs = init[:]
    
    current_signs = best_signs[:]
    current_score = best_score
    
    no_improve_count = 0
    max_no_improve = 60
    iteration = 0
    restart_interval = 1500
    
    # Pre-compute candidate list, update periodically
    candidates = _get_tight_ap_info(current_signs, top_k=40)
    
    while time.time() < deadline:
        iteration += 1
        
        if no_improve_count > max_no_improve or iteration % restart_interval == 0:
            if rng.random() < 0.4:
                current_signs = best_signs[:]
                for _ in range(rng.randint(3, 15)):
                    idx = rng.randint(0, n - 1)
                    current_signs[idx] = -current_signs[idx]
            else:
                current_signs = [rng.choice((-1, 1)) for _ in range(n)]
            
            current_score = _max_abs_ap_sum(current_signs)
            if current_score < best_score:
                best_signs = current_signs[:]
                best_score = current_score
            no_improve_count = 0
            candidates = _get_tight_ap_info(current_signs, top_k=40)
            continue
        
        # Update candidates periodically
        if iteration % 30 == 0:
            candidates = _get_tight_ap_info(current_signs, top_k=40)
            if not candidates:
                candidates = list(range(n))
        
        # Choose flip strategy: 80% single flip from tight indices, 20% random or 2-flip
        if rng.random() < 0.8:
            # Single flip from tight AP indices
            if candidates:
                idx = candidates[rng.randint(0, len(candidates) - 1)]
            else:
                idx = rng.randint(0, n - 1)
            
            old_val = current_signs[idx]
            current_signs[idx] = -old_val
            new_score = _max_abs_ap_sum(current_signs, threshold=current_score)
            
            if new_score < current_score:
                current_score = new_score
                no_improve_count = 0
                if new_score < best_score:
                    best_score = new_score
                    best_signs = current_signs[:]
            else:
                current_signs[idx] = old_val
                no_improve_count += 1
        else:
            # Try 2-flip move: flip two indices simultaneously
            if candidates and len(candidates) >= 2:
                i1 = candidates[rng.randint(0, len(candidates) - 1)]
                i2 = candidates[rng.randint(0, len(candidates) - 1)]
                if i1 == i2:
                    i2 = (i2 + 1) % n
            else:
                i1 = rng.randint(0, n - 1)
                i2 = rng.randint(0, n - 1)
                if i1 == i2:
                    i2 = (i2 + 1) % n
            
            v1 = current_signs[i1]
            v2 = current_signs[i2]
            current_signs[i1] = -v1
            current_signs[i2] = -v2
            new_score = _max_abs_ap_sum(current_signs, threshold=current_score)
            
            if new_score < current_score:
                current_score = new_score
                no_improve_count = 0
                if new_score < best_score:
                    best_score = new_score
                    best_signs = current_signs[:]
            else:
                current_signs[i1] = v1
                current_signs[i2] = v2
                no_improve_count += 1
    
    return best_signs
