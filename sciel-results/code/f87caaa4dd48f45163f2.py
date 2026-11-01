import time
import random
import math

_START = time.time()
_DEADLINE = _START + 115.0

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _next_prime(n):
    if n <= 2:
        return 2
    if n % 2 == 0:
        n += 1
    else:
        n += 2
    while not _is_prime(n):
        n += 2
    return n

def _score(u):
    if not u or len(u) < 2:
        return 0.0
    m = max(u)
    if m < 1:
        return 0.0
    
    # Compute unique sums and differences
    # For small n, O(n^2) is fast
    u_set = set(u)
    sums = set()
    diffs = set()
    for a in u:
        for b in u:
            sums.add(a + b)
            diffs.add(a - b)
            
    n_sums = len(sums)
    n_diffs = len(diffs)
    
    if n_sums == 0:
        return 0.0
        
    ratio = n_diffs / float(n_sums)
    if ratio <= 0:
        return 0.0
        
    return 1.0 + math.log(ratio) / math.log(2 * m + 1)

def _quadratic_residues_prime(p, max_elements=1000):
    """
    Construct U from quadratic residues modulo a prime p.
    QR mod p has size (p-1)/2 + 1 (including 0).
    The difference set size is approximately p, sumset approximately p/2.
    Ratio ~ 2, which is high.
    
    We take QR values in [0, p-1] as nonnegative integers.
    """
    if p < 3 or p % 2 == 0:
        return [0, 1]
    
    # Compute quadratic residues mod p
    qr = set()
    for x in range(p):
        qr.add((x * x) % p)
    
    # QR includes 0, so we have (p+1)/2 elements
    u = sorted(qr)
    
    # If too large, we can't use it directly - but for our purposes
    # the score depends on log ratio / log(2M+1)
    # With QR, ratio ≈ 2, M = p-1
    # Score ≈ 1 + log(2)/log(2p) 
    # This is a good baseline
    
    return u

def _construct_golomb(n):
    """
    Construct a Golomb ruler of length n using a simple greedy algorithm.
    This ensures |U-U| is maximized (all differences distinct).
    """
    if n == 1:
        return [0]
    if n == 2:
        return [0, 1]
    
    # Greedy construction
    u = [0]
    diff_set = set()
    for i in range(1, n):
        best_pos = -1
        max_pos = (i+1)**2 + 10
        for pos in range(1, max_pos + 1):
            valid = True
            for a in u:
                d = abs(pos - a)
                if d in diff_set:
                    valid = False
                    break
            if valid:
                best_pos = pos
                break
        
        if best_pos == -1:
            # Fallback: find smallest valid position with larger search
            found = False
            for pos in range(1, 10000):
                valid = True
                for a in u:
                    d = abs(pos - a)
                    if d in diff_set:
                        valid = False
                        break
                if valid:
                    best_pos = pos
                    found = True
                    break
            if not found:
                # Just append with large gap
                best_pos = max(u) + i * i + i
        
        # Update diff set
        for a in u:
            diff_set.add(abs(best_pos - a))
        u.append(best_pos)
        
    return sorted(u)

def _local_search(u, timeout=2.0):
    u = sorted(set(u))
    if len(u) < 2:
        return u, _score(u)
        
    current_score = _score(u)
    best_u = u[:]
    best_score = current_score
    start_time = time.time()
    n = len(u)
    max_val = max(u)
    
    # We will try to improve by changing one element at a time.
    # Search space for a new value: [0, max_val * 1.5]
    
    while time.time() < start_time + timeout and time.time() < _DEADLINE - 2.0:
        improved = False
        # Try each element
        for i in range(n):
            if time.time() > start_time + timeout or time.time() > _DEADLINE - 2.0:
                break
            old_val = u[i]
            # Try a range of new values
            search_range = max(5, max_val // 2)
            # Sample candidates
            candidates = set()
            # Local neighborhood
            step = max(1, search_range // 20)
            for delta in range(-search_range, search_range + 1, step):
                new_val = old_val + delta
                if new_val >= 0:
                    candidates.add(new_val)
            # Random jumps
            for _ in range(5):
                candidates.add(random.randint(0, max_val + search_range))
                
            for new_val in candidates:
                if new_val in u and new_val != old_val:
                    continue
                new_u = u[:]
                new_u[i] = new_val
                new_u = sorted(set(new_u))
                if len(new_u) != n:
                    continue
                s = _score(new_u)
                if s > current_score:
                    u = new_u
                    current_score = s
                    if current_score > best_score:
                        best_score = current_score
                        best_u = u[:]
                    max_val = max(u)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            # Random restart or mutation if stuck
            if random.random() < 0.1:
                idx = random.randint(0, n-1)
                u[idx] = random.randint(0, max_val + 10)
                u = sorted(set(u))
                if len(u) < n:
                    u = best_u[:]
                else:
                    current_score = _score(u)

    return best_u, best_score

def _polish(u, timeout=3.0):
    """
    High-precision polish: try small integer shifts of individual elements
    to fine-tune the score. Uses a tighter local neighborhood.
    """
    u = sorted(set(u))
    if len(u) < 2:
        return u, _score(u)
    
    current_score = _score(u)
    best_u = u[:]
    best_score = current_score
    start_time = time.time()
    n = len(u)
    max_val = max(u)
    
    while time.time() < start_time + timeout and time.time() < _DEADLINE - 2.0:
        improved = False
        for i in range(n):
            if time.time() > start_time + timeout or time.time() > _DEADLINE - 2.0:
                break
            old_val = u[i]
            # Very small perturbations: +/- 1, 2, 3, 5, 8, 13
            for delta in [-13, -8, -5, -3, -2, -1, 1, 2, 3, 5, 8, 13]:
                new_val = old_val + delta
                if new_val < 0:
                    continue
                if new_val in u and new_val != old_val:
                    continue
                new_u = u[:]
                new_u[i] = new_val
                new_u = sorted(set(new_u))
                if len(new_u) != n:
                    continue
                s = _score(new_u)
                if s > current_score + 1e-12:
                    u = new_u
                    current_score = s
                    if current_score > best_score:
                        best_score = current_score
                        best_u = u[:]
                    max_val = max(u)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return best_u, best_score

def construct():
    global _DEADLINE
    random.seed(42)
    
    best_u = [0, 1]
    best_s = _score(best_u)
    
    # Strategy 1: Try quadratic residues from primes
    # For prime p, QR mod p has (p+1)/2 elements in [0, p-1]
    # |U-U| ≈ p, |U+U| ≈ p/2, ratio ≈ 2
    # Score ≈ 1 + log(2)/log(2p)
    # Try various primes to find the best
    primes_to_try = []
    for p in range(7, 200, 2):
        if _is_prime(p):
            primes_to_try.append(p)
    
    for p in primes_to_try:
        if time.time() > _DEADLINE - 15.0:
            break
        u_qr = _quadratic_residues_prime(p)
        if len(u_qr) < 2:
            continue
        s_qr = _score(u_qr)
        if s_qr > best_s:
            best_s = s_qr
            best_u = u_qr[:]
    
    # Strategy 2: Local search from Golomb rulers and random for small n
    for n in range(2, 16):
        if time.time() > _DEADLINE - 10.0:
            break
        
        time_budget = 2.0
        restarts = 4
        
        for r in range(restarts):
            if time.time() > _DEADLINE - 5.0:
                break
                
            if r == 0:
                u_init = _construct_golomb(n)
            elif r == 1:
                # Random with moderate span
                max_span = n * n
                u_init = [0]
                for _ in range(n - 1):
                    u_init.append(random.randint(1, max_span))
                u_init = sorted(set(u_init))
                while len(u_init) < n:
                    u_init.append(random.randint(1, max_span))
                    u_init = sorted(set(u_init))
            elif r == 2:
                # Random with larger span
                max_span = n * n * 2
                u_init = [0]
                for _ in range(n - 1):
                    u_init.append(random.randint(1, max_span))
                u_init = sorted(set(u_init))
                while len(u_init) < n:
                    u_init.append(random.randint(1, max_span))
                    u_init = sorted(set(u_init))
            else:
                # Start from best_u and perturb
                u_init = best_u[:]
                if len(u_init) != n:
                    if len(u_init) > n:
                        u_init = u_init[:n]
                    else:
                        while len(u_init) < n:
                            u_init.append(random.randint(1, max(best_u) + 10 if best_u else 10))
                            u_init = sorted(set(u_init))
                else:
                    # Perturb one element
                    idx = random.randint(0, n-1)
                    u_init[idx] = random.randint(0, max(u_init) + 5)
                    u_init = sorted(set(u_init))
                    while len(u_init) < n:
                        u_init.append(random.randint(1, max(u_init) + 10 if u_init else 10))
                        u_init = sorted(set(u_init))
            
            u_opt, s_opt = _local_search(u_init, timeout=time_budget)
            
            if s_opt > best_s:
                best_s = s_opt
                best_u = u_opt[:]
                
    # Strategy 3: Polish the best result found
    if time.time() < _DEADLINE - 3.0:
        polished_u, polished_s = _polish(best_u, timeout=min(5.0, _DEADLINE - time.time() - 2.0))
        if polished_s > best_s:
            best_s = polished_s
            best_u = polished_u[:]
    
    return best_u
